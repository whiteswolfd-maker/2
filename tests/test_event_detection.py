"""Physical failure modes and extraction/replay contract; no real DYNA run."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from data.event_detection import (
    EventConfig, PROBE_FIELDS, analyze_events, detect_gap_frame,
    select_cj_probe, write_event_outputs,
)
from data.detect_events import detect_directory


REF = dict(rho=2000.0, u=1500.0, P=1e7)


def profile(shock=0.085, contact=0.05):
    r = np.linspace(0, 0.12, 121)
    vf = (r <= contact).astype(float)
    air = (r > contact) & (r <= shock)
    rho = np.where(vf > 0, 2000.0, np.where(air, 5.0, 1.0))
    u = np.where(vf > 0, 1500.0, np.where(air, 100.0, 0.0))
    P = np.where(vf > 0, 1e7, np.where(air, 1e6, 1e5))
    return r, rho, u, P, vf


def gap(p, **kwargs):
    return detect_gap_frame(*p, P_atm=1e5, rho_air=1.0, **kwargs)


def series(shocks=(0.051, 0.052, 0.085, 0.088, 0.09)):
    frames = [profile(s) for s in shocks]
    r = frames[0][0]
    fields = [np.array([frame[j] for frame in frames]) for j in range(1, 5)]
    return np.arange(len(frames))*1e-6, r, *fields


def test_resolved_gap_and_material_interface():
    p = profile()
    result = gap(p, sample_count=np.ones(len(p[0])), cell_width=np.full(len(p[0]), 0.001))
    assert result["resolved"]
    assert result["R_s_lower"] > result["R_c_upper"]
    assert 0.05 < result["R_c"] < 0.051


def test_internal_detonation_is_not_air_shock():
    r, rho, u, P, vf = profile(shock=0)
    P[r <= 0.025] = 1e10
    assert not gap((r, rho, u, P, vf))["resolved"]


@pytest.mark.parametrize("mode", ["small_gap", "clipped_front", "density_only", "disconnected_pulse"])
def test_false_separation_rejected(mode):
    p = list(profile(shock=0.053 if mode == "small_gap" else 0.085))
    if mode == "clipped_front":
        p = [a[:80] for a in p]
    elif mode == "density_only":
        p[-1] = None
    elif mode == "disconnected_pulse":
        r = p[0]
        mask = (r > 0.05) & (r < 0.065)
        p[1][mask], p[2][mask], p[3][mask] = 1, 0, 1e5
    assert not gap(p)["resolved"]


def test_grid_interpolation_cannot_create_resolved_air_cells():
    p = profile()
    count = np.zeros(len(p[0]))
    count[::10] = 1
    assert not gap(p, sample_count=count, cell_width=np.full(len(count), 0.02))["resolved"]


def test_smeared_front_needs_full_upstream_state():
    p = list(profile())
    p[2][86:89] = 3.0  # pressure already near ambient but velocity still disturbed
    result = gap(p)
    assert result["resolved"]
    assert result["R_s_upper"] >= p[0][89]


def test_persistence_and_time_origin_not_zeroed():
    t, r, rho, u, P, vf = series()
    t += 0.007
    report, _, _ = analyze_events(t, r, rho, u, P, vf=vf, R_0=0.05,
                                 P_atm=1e5, rho_air=1, reference=REF,
                                 sample_count=np.ones_like(P), cell_width=np.full_like(P, 0.001))
    assert report["separation"]["status"] == "detected"
    assert report["t_sep_detected_s"] == t[2]
    assert report["separation"]["time_interval_s"] == [t[1], t[2]]


def test_single_frame_gap_noise_and_left_censoring():
    for shocks, expected in [((0.051, 0.085, 0.051, 0.051), None),
                             ((0.085, 0.088, 0.09), 0.0)]:
        t, r, rho, u, P, vf = series(shocks)
        report, _, _ = analyze_events(t, r, rho, u, P, vf=vf, R_0=0.05,
                                     P_atm=1e5, rho_air=1)
        assert report["t_sep_detected_s"] == expected
        assert report["separation"]["time_interval_s"][0] is None


def probe(**changes):
    values = dict(r=np.array([0.046, 0.049]), rho=np.array([2000.0, 2000.0]),
                  u=np.array([1500.0, 1500.0]), P=np.array([1e7, 1e7]),
                  vf=np.ones(2), R_0=0.05, reference=REF)
    values.update(changes)
    return select_cj_probe(**values)


def test_cj_joint_match_uses_one_actual_sample():
    # Every component matches somewhere, but no single cell matches all three.
    result = probe(rho=np.array([2000.0, 1000.0]), u=np.array([700.0, 1500.0]))
    assert result["relative_error"] > EventConfig().cj_rel_tol
    # A short original-cell peak may disappear under shell averaging.
    assert probe(P=np.array([1e7, 0.0]))["relative_error"] == 0
    assert probe(P=np.array([5e6, 5e6]))["relative_error"] == 0.5


def test_cj_requires_material_and_rejects_incomplete_burn():
    assert probe(vf=None)["eligible_samples"] == 0
    assert probe(vf=np.zeros(2))["eligible_samples"] == 0
    assert probe(burn=np.array([0.4, 0.8]))["eligible_samples"] == 0
    assert probe(burn=np.ones(2))["eligible_samples"] == 2
    assert probe(r=np.array([0.02, 0.03]))["r"] is None


def test_cj_transient_single_frame_and_no_nearest_time_fallback():
    t, r, rho, u, P, vf = series()
    probes = [probe(P=np.array([4e6, 4e6])) for _ in t]
    probes[1] = probe(burn=np.ones(2))
    kwargs = dict(vf=vf, R_0=0.05, P_atm=1e5, rho_air=1, reference=REF, cj_probes=probes)
    report, _, _ = analyze_events(t, r, rho, u, P, **kwargs)
    assert report["t_cj_candidate_s"] == t[1]
    assert report["cj"]["status"] == "state_and_burn_candidate"
    probes[1] = probe(P=np.array([4e6, 4e6]))
    report, _, _ = analyze_events(t, r, rho, u, P, **kwargs)
    assert report["t_cj_candidate_s"] is None
    assert report["cj"]["closest_sample"] is not None


def test_invalid_slot_values_and_nonmonotone_times():
    with pytest.raises(ValueError, match="outside"):
        probe(vf=np.full(2, 2000.0))
    t, r, rho, u, P, vf = series()
    t[1] = t[0]
    with pytest.raises(ValueError, match="increasing"):
        analyze_events(t, r, rho, u, P, R_0=0.05)
    with pytest.raises(ValueError):
        EventConfig(cj_rel_tol=2.0)


def test_old_npz_reports_missing_evidence_without_mutating_anchors(tmp_path):
    t, r, rho, u, P, vf = series()
    np.savez(tmp_path/"raw_line.npz", t=t, x=r, rho=rho, u=u, P=P)
    meta = dict(R_0=0.05, P_CJ=1e7, rho_CJ=2000, u_CJ=1500, t_0=0, t_sep=7e-5)
    text = json.dumps(meta)
    (tmp_path/"metadata.json").write_text(text)
    report = detect_directory(tmp_path)
    assert report["t_sep_detected_s"] is None and report["t_cj_candidate_s"] is None
    assert (tmp_path/"metadata.json").read_text() == text
    saved = (tmp_path/"event_check"/"events.json").read_text()
    assert "NaN" not in saved and "Infinity" not in saved


def test_compact_probe_replay_preserves_original_cell_peak(tmp_path):
    t, r, rho, u, P, vf = series()
    cfg = EventConfig()
    probes = [probe() for _ in t]
    probes[0] = probe(P=np.array([1e6, 1e6]))
    raw = np.array([[p[k] if p[k] is not None else np.nan for k in PROBE_FIELDS] for p in probes])
    np.savez(tmp_path/"raw_line.npz", t=t, x=r, rho=rho, u=u, P=P*0.5, vf_tnt=vf,
             cj_probe=raw)
    meta = dict(R_0=0.05, event_cj_reference=REF, event_criteria=asdict(cfg),
                cj_probe_fields=list(PROBE_FIELDS))
    (tmp_path/"metadata.json").write_text(json.dumps(meta))
    report = detect_directory(tmp_path)
    assert report["t_cj_candidate_s"] == t[1]
    changed = tmp_path/"changed.yaml"
    changed.write_text("events:\n  cj_band_inner_fraction: 0.8\n")
    with pytest.raises(ValueError, match="rerun"):
        detect_directory(tmp_path, config_path=changed)


def test_batch_coordinates_do_not_reclassify_late_displacements():
    from data.extract_d3plot import _node_positions, _absolute_coordinates_at_frame_zero
    at = SimpleNamespace(node_coordinates="nodes", node_displacement="disp")
    nodes = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    zero = SimpleNamespace(arrays={"nodes": nodes, "disp": np.zeros((1, 2, 3))})
    mode = _absolute_coordinates_at_frame_zero(zero, at)
    late = SimpleNamespace(arrays={"nodes": nodes, "disp": np.full((1, 2, 3), 0.3)})
    assert not mode
    assert np.allclose(_node_positions(late, at, 0, mode), nodes+0.3)


def test_outside_cells_do_not_contaminate_last_export_shell():
    from data.extract_d3plot import _interpolate_to_grid
    pos = np.array([[0.09, 0, 0], [0.1, 0, 0], [0.8, 0, 0]])
    r = np.linspace(0, 0.1, 11)
    _, _, P, _ = _interpolate_to_grid(pos, np.ones(3), np.zeros(3), np.array([1e5, 1e5, 1e10]), r)
    assert P[-1] == 1e5


def test_extract_driver_emits_events_and_replay_matches(tmp_path, monkeypatch):
    """Exercise CLI->batched extraction->original probes->disk->replay.

    A synthetic D3plot interface replaces lasso binary IO only. This is not
    validation against the user's original binary output or material slots.
    """
    import data.extract_d3plot as ex
    from physics.cj_state import TNTParams, compute_cj_state
    cj = compute_cj_state(TNTParams())
    times = np.arange(5)*1e-6
    radii = np.arange(0.0005, 0.12, 0.001)
    corners = np.array([[x, y, z] for x in [-0.0004, 0.0004]
                        for y in [-0.0001, 0.0001] for z in [-0.0001, 0.0001]])
    nodes = (np.c_[radii, np.zeros((len(radii), 2))][:, None, :]+corners).reshape(-1, 3)
    conn = np.arange(len(nodes)).reshape(-1, 8)
    at = SimpleNamespace(**{k: k for k in ["node_coordinates", "node_displacement", "node_velocity",
                                          "element_solid_node_indexes", "element_solid_stress",
                                          "element_solid_history_variables", "global_timesteps"]})
    def load(directory, state_subset=None, **kwargs):
        indices = state_subset
        states_rho, states_u, states_P, states_vf = [], [], [], []
        for i in indices:
            vf = (radii < 0.05).astype(float)
            air = (radii >= 0.05) & (radii < (0.051 if i < 2 else 0.085))
            rho = np.where(vf > 0, cj.rho_CJ, np.where(air, 5.0, 1.225))
            u = np.where(vf > 0, cj.u_CJ if i == 1 else 0.0, np.where(air, 100.0, 0.0))
            P = np.where(vf > 0, cj.P_CJ if i == 1 else 1e6, np.where(air, 1e6, 101325.0))
            states_rho.append(rho); states_u.append(u); states_P.append(P); states_vf.append(vf)
        stress = np.zeros((len(indices), len(radii), 6))
        stress[:, :, :3] = -np.asarray(states_P)[:, :, None]
        vel = np.zeros((len(indices), len(nodes), 3))
        vel[:, :, 0] = np.repeat(states_u, 8, axis=1)
        return SimpleNamespace(arrays=dict(node_coordinates=nodes, node_displacement=np.zeros_like(vel),
            node_velocity=vel, element_solid_node_indexes=conn, element_solid_stress=stress,
            element_solid_history_variables=np.stack([states_rho, states_vf], axis=2),
            global_timesteps=times[indices])), at
    monkeypatch.setattr(ex, "_load_d3plot", load)
    monkeypatch.setattr(ex, "_read_state_times", lambda d: times)
    dest = tmp_path/"out"
    assert ex.main(["--d3plot", str(tmp_path), "--out", str(dest), "--R-0", "0.05",
                    "--x-end", "0.12", "--n-x", "121", "--vf-tnt-slot", "1",
                    "--unit-system", "si", "--batch-size", "2"]) == 0
    before = json.loads((dest/"events.json").read_text())
    assert before["t_cj_candidate_s"] == times[1]
    assert before["t_sep_detected_s"] == times[2]
    after = detect_directory(dest)
    assert before["cj"] == after["cj"]
    assert before["separation"] == after["separation"]
    with np.load(dest/"raw_line.npz") as f:
        assert {"vf_tnt", "sample_count", "cell_width", "state_index", "cj_probe"} <= set(f.files)
