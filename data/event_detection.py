"""Observable events in SI-valued DYNA fields; no theoretical time fallback.

The gap event is a resolution-dependent observation, not the thesis outflow
definition. A CJ state match is a candidate, not proof of CJ formation.
This module requires only numpy and never changes training anchors.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EventConfig:
    # Numerical diagnostics, not universal physical constants.
    vf_low: float = 0.1
    vf_high: float = 0.9
    air_pressure_ratio: float = 1.05
    air_density_ratio: float = 1.02
    air_velocity_min: float = 1.0
    ambient_rel_tol: float = 0.05
    min_air_samples: int = 3
    min_gap_cells: float = 2.0
    persistence_frames: int = 3
    cj_band_inner_fraction: float = 0.9
    cj_rel_tol: float = 0.15
    cj_vf_min: float = 0.95
    burn_complete_min: float = 0.99

    def __post_init__(self):
        vals = asdict(self)
        if not all(np.isfinite(v) for v in vals.values()):
            raise ValueError("Event thresholds must be finite")
        if not 0 < self.vf_low < 0.5 < self.vf_high < 1:
            raise ValueError("Require 0 < vf_low < 0.5 < vf_high < 1")
        if self.air_pressure_ratio <= 1 or self.air_density_ratio <= 1:
            raise ValueError("Air compression ratios must exceed 1")
        if self.air_velocity_min < 0 or not 0 < self.ambient_rel_tol < 1:
            raise ValueError("Invalid air velocity/ambient tolerance")
        for name in ("min_air_samples", "persistence_frames"):
            if vals[name] < 1 or int(vals[name]) != vals[name]:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_gap_cells <= 0 or not 0 < self.cj_rel_tol < 1:
            raise ValueError("Invalid gap width/CJ tolerance")
        if not 0 < self.cj_band_inner_fraction < 1:
            raise ValueError("CJ band must lie inside the original charge")
        if not self.vf_high <= self.cj_vf_min <= 1:
            raise ValueError("cj_vf_min must be between vf_high and 1")
        if not 0 < self.burn_complete_min <= 1:
            raise ValueError("burn_complete_min must lie in (0, 1]")


def validate_fraction(values, name):
    if values is None:
        return
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size and (finite.min() < -1e-5 or finite.max() > 1 + 1e-5):
        raise ValueError(f"{name} is outside [0, 1]; verify history-slot meaning, "
                         "slot indices and units (volume fraction is not burn progress)")


def _widths(r, cell_width):
    # Spacing of populated samples, not spacing of a densely filled export.
    spacing = np.diff(r)
    local = np.maximum(np.r_[spacing[0], spacing], np.r_[spacing, spacing[-1]])
    if cell_width is not None:
        local = np.maximum(local, np.nan_to_num(cell_width, nan=0.0))
    return local


def detect_gap_frame(r, rho, u, P, vf, *, P_atm, rho_air,
                     config=EventConfig(), sample_count=None, cell_width=None):
    """Locate the central material interface and its adjacent compressed air.

    Empty shells are excluded. Unknown material identity or a wave clipped by
    the outer export boundary cannot produce a detected event.
    """
    result = dict(resolved=False, reason="missing_volume_fraction", R_c=None,
                  R_c_lower=None, R_c_upper=None, R_s=None, R_s_lower=None,
                  R_s_upper=None, gap_lower=None, resolution=None,
                  compressed_air_samples=0)
    if vf is None:
        return result
    validate_fraction(vf, "vf_tnt")
    r, rho, u, P, vf = [np.asarray(v, dtype=float) for v in (r, rho, u, P, vf)]
    valid = np.isfinite(r + rho + u + P + vf) & (rho > 0)
    if sample_count is not None:
        valid &= np.asarray(sample_count) > 0
    width = None if cell_width is None else np.asarray(cell_width)[valid]
    r, rho, u, P, vf = [v[valid] for v in (r, rho, u, P, vf)]
    if len(r) < 4:
        result["reason"] = "insufficient_populated_shells"
        return result
    h = _widths(r, width)
    core = np.flatnonzero(vf >= config.vf_high)
    if not core.size:
        result["reason"] = "no_product_core"
        return result
    first = int(core[0])
    crossings = {}
    for threshold in (config.vf_high, 0.5, config.vf_low):
        below = np.flatnonzero((np.arange(len(r)) > first) & (vf < threshold))
        if not below.size:
            result["reason"] = "contact_outside_export_or_unresolved"
            return result
        j = int(below[0])
        frac = (vf[j-1] - threshold) / (vf[j-1] - vf[j])
        crossings[threshold] = (float(r[j-1] + frac * (r[j] - r[j-1])), j)
    rc, _ = crossings[0.5]
    ri, ji = crossings[config.vf_high]
    ro, jo = crossings[config.vf_low]
    contact_h = float(np.max(h[ji-1:jo+1]))
    rc_lo, rc_hi = max(0.0, ri-contact_h/2), ro+contact_h/2
    result.update(R_c=rc, R_c_lower=rc_lo, R_c_upper=rc_hi)

    # Follow the first air layer adjoining this contact, not a disconnected
    # distant pressure spike or the pressure front inside unburned TNT.
    air_start = jo
    if not (vf[air_start] < config.vf_low and
            P[air_start] > config.air_pressure_ratio * P_atm):
        result["reason"] = "no_adjacent_compressed_air"
        return result
    end = air_start
    while end + 1 < len(r) and vf[end+1] < config.vf_low and \
            P[end+1] > config.air_pressure_ratio * P_atm:
        end += 1
    ahead = end + 1
    # Pressure, density and velocity may have different smeared transition
    # widths. Include the entire transition up to two ambient samples.
    while ahead + 1 < len(r):
        up = slice(ahead, ahead+2)
        ambient = ((vf[up] < config.vf_low) &
                   (np.abs(P[up]/P_atm-1) <= config.ambient_rel_tol) &
                   (np.abs(rho[up]/rho_air-1) <= config.ambient_rel_tol) &
                   (np.abs(u[up]) <= config.air_velocity_min))
        if np.all(ambient):
            break
        if vf[ahead] >= config.vf_low:
            result["reason"] = "upstream_air_not_resolved"
            return result
        ahead += 1
    if ahead + 1 >= len(r) or P[end] <= P[end+1]:
        result["reason"] = "shock_outside_export_or_no_upstream_samples"
        return result
    fraction = (P[end] - config.air_pressure_ratio*P_atm) / (P[end]-P[end+1])
    rs = float(r[end] + fraction*(r[end+1]-r[end]))
    front_h = float(max(np.max(h[end:ahead+1]), r[ahead]-r[end]))
    rs_lo, rs_hi = float(r[end]-front_h/2), float(r[ahead]+front_h/2)
    layer = (r > rc_hi) & (r < rs_lo)
    compressed = (layer & (vf < config.vf_low) &
                  (P > config.air_pressure_ratio*P_atm) &
                  (rho > config.air_density_ratio*rho_air) &
                  (u > config.air_velocity_min))
    n = int(compressed.sum())
    resolution = float(max(contact_h, front_h, np.max(h[layer]) if layer.any() else 0))
    # A pocket of ambient air in between breaks adjacency.
    resolved = (n >= config.min_air_samples and np.all(compressed[layer]) and
                rs_lo-rc_hi >= config.min_gap_cells*resolution)
    result.update(R_s=rs, R_s_lower=rs_lo, R_s_upper=rs_hi,
                  gap_lower=float(rs_lo-rc_hi), resolution=resolution,
                  compressed_air_samples=n, resolved=bool(resolved),
                  reason="resolved" if resolved else "gap_below_resolution")
    return result


PROBE_FIELDS = ("r", "rho", "u", "P", "vf_tnt", "burn_fraction", "sample_id",
                "cell_width", "relative_error", "eligible_samples")


def select_cj_probe(r, rho, u, P, *, R_0, reference, config=EventConfig(),
                    vf=None, burn=None, sample_ids=None, cell_width=None,
                    sample_count=None):
    """Return the best simultaneous state match in a fixed near-surface band.

    Called on ORIGINAL cells during d3plot extraction. Only a compact best
    sample per frame is retained; no whole-domain argmax or arbitrary time
    fallback. Without VF there can be a closest diagnostic, but no CJ event.
    """
    out = {k: None for k in PROBE_FIELDS}
    out["eligible_samples"] = 0
    if not reference or any(not np.isfinite(reference.get(k, np.nan)) or
                            reference.get(k, 0) <= 0 for k in ("rho", "u", "P")):
        return out
    r, rho, u, P = [np.asarray(v, dtype=float) for v in (r, rho, u, P)]
    valid = (np.isfinite(r+rho+u+P) & (rho > 0) & (P >= 0) &
             (r >= config.cj_band_inner_fraction*R_0) & (r <= R_0))
    if sample_count is not None:
        valid &= np.asarray(sample_count) > 0
    if not valid.any():
        return out
    validate_fraction(vf, "vf_tnt")
    validate_fraction(burn, "burn_fraction")
    eligible = valid.copy()
    if vf is None:
        eligible[:] = False
    else:
        eligible &= np.isfinite(vf) & (np.asarray(vf) >= config.cj_vf_min)
    if burn is not None:
        eligible &= np.isfinite(burn) & (np.asarray(burn) >= config.burn_complete_min)
    error = np.maximum.reduce([np.abs(rho/reference["rho"]-1),
                               np.abs(u/reference["u"]-1),
                               np.abs(P/reference["P"]-1)])
    candidates = np.flatnonzero(eligible if eligible.any() else valid)
    j = int(candidates[np.argmin(error[candidates])])
    out.update(r=float(r[j]), rho=float(rho[j]), u=float(u[j]), P=float(P[j]),
               vf_tnt=None if vf is None else float(vf[j]),
               burn_fraction=None if burn is None else float(burn[j]),
               sample_id=int(j if sample_ids is None else sample_ids[j]),
               cell_width=None if cell_width is None else float(cell_width[j]),
               relative_error=float(error[j]), eligible_samples=int(eligible.sum()))
    return out


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def analyze_events(t, r, rho, u, P, *, R_0, P_atm=101325.0, rho_air=1.225,
                   reference=None, vf=None, burn=None, sample_count=None,
                   cell_width=None, state_indices=None, cj_probes=None,
                   config=EventConfig(), source="radial_profiles"):
    """Return an event report plus frame records, all in the original clock."""
    t, r = np.asarray(t, dtype=float), np.asarray(r, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Times must be finite, nonempty and strictly increasing")
    if r.ndim != 1 or len(r) < 4 or not np.isfinite(r).all() or r[0] < 0 or np.any(np.diff(r) <= 0):
        raise ValueError("Radii must be finite, nonnegative and strictly increasing")
    if not np.isfinite([R_0, P_atm, rho_air]).all() or min(R_0, P_atm, rho_air) <= 0:
        raise ValueError("R_0, P_atm and rho_air must be finite and positive")
    if R_0 > r[-1]:
        raise ValueError("The radial export must include the original charge surface")
    shape = (len(t), len(r))
    for name, a in (("rho", rho), ("u", u), ("P", P), ("vf_tnt", vf),
                    ("burn", burn), ("sample_count", sample_count), ("cell_width", cell_width)):
        if a is not None and np.shape(a) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {np.shape(a)}")
    if sample_count is not None and (not np.isfinite(sample_count).all() or
                                    np.any(sample_count < 0) or np.any(sample_count != np.floor(sample_count))):
        raise ValueError("sample_count must contain finite nonnegative integers")
    if cell_width is not None and (not np.isfinite(cell_width).all() or np.any(cell_width < 0)):
        raise ValueError("cell_width must contain finite nonnegative radial spans")
    if state_indices is None:
        state_indices = np.arange(len(t))
    if len(state_indices) != len(t) or (cj_probes is not None and len(cj_probes) != len(t)):
        raise ValueError("Frame IDs/probes must match the time axis")
    rows, probes = [], []
    pick = lambda a, i: None if a is None else a[i]
    for i, ti in enumerate(t):
        row = detect_gap_frame(r, rho[i], u[i], P[i], pick(vf, i), P_atm=P_atm,
                               rho_air=rho_air, config=config,
                               sample_count=pick(sample_count, i), cell_width=pick(cell_width, i))
        row.update(t=float(ti), state_index=int(state_indices[i]))
        rows.append(row)
        probe = (cj_probes[i] if cj_probes is not None else
                 select_cj_probe(r, rho[i], u[i], P[i], R_0=R_0, reference=reference,
                                 config=config, vf=pick(vf, i), burn=pick(burn, i),
                                 cell_width=pick(cell_width, i), sample_count=pick(sample_count, i)))
        probes.append(dict(probe, t=float(ti), state_index=int(state_indices[i])))

    sep = dict(status="not_identified", time_s=None, time_interval_s=[None, None],
               R_c_m=None, R_s_m=None,
               definition="first_persistent_resolvable_compressed_air_layer",
               reason="no_persistent_resolved_gap")
    if vf is None:
        sep.update(status="insufficient_data", reason="missing_volume_fraction")
    n = int(config.persistence_frames)
    for i in range(len(t)-n+1):
        segment = rows[i:i+n]
        if not all(row["resolved"] for row in segment):
            continue
        # A moving outward shock may fluctuate by one positional uncertainty.
        if any(b["R_s"] < a["R_s"]-max(a["resolution"], b["resolution"])
               for a, b in zip(segment, segment[1:])):
            continue
        row = rows[i]
        sep.update(status="detected" if cell_width is not None and sample_count is not None else "candidate",
                   time_s=float(t[i]), time_interval_s=[float(t[i-1]) if i else None, float(t[i])],
                   R_c_m=row["R_c"], R_s_m=row["R_s"], state_index=int(state_indices[i]),
                   contact_interval_m=[row["R_c_lower"], row["R_c_upper"]],
                   shock_interval_m=[row["R_s_lower"], row["R_s_upper"]],
                   confirmed_through_time_s=float(t[i+n-1]),
                   reason="criterion_already_met_at_first_frame" if i == 0 else "persistent_gap")
        break
    cj = dict(status="not_identified", time_s=None, position_m=None,
              time_interval_s=[None, None], reason="no_joint_CJ_state_match",
              definition="near_CJ_state_in_original_surface_inner_band",
              band_m=[config.cj_band_inner_fraction*R_0, R_0], closest_sample=None)
    finite = [p for p in probes if p.get("relative_error") is not None and np.isfinite(p["relative_error"])]
    if finite:
        cj["closest_sample"] = min(finite, key=lambda p: p["relative_error"])
    else:
        cj["reason"] = "missing_reference_or_no_valid_samples_in_surface_band"
    for i, probe in enumerate(probes):
        error = probe.get("relative_error")
        if probe.get("eligible_samples", 0) <= 0 or error is None or not np.isfinite(error) or error > config.cj_rel_tol:
            continue
        cj.update(status="state_and_burn_candidate" if probe.get("burn_fraction") is not None else "state_candidate",
                  time_s=float(t[i]), position_m=probe["r"], sample=probe,
                  time_interval_s=[float(t[i-1]) if i else None,
                                   float(t[i+1]) if i+1 < len(t) else None],
                  reason="joint_state_match_not_proof_of_CJ_formation")
        break
    report = dict(schema_version=1, time_origin="original_DYNA_time",
                  units=dict(time="s", radius="m", pressure="Pa", density="kg/m^3", velocity="m/s"),
                  R_0_m=R_0, source=source, criteria=asdict(config), cj_reference=reference,
                  t_sep_detected_s=sep["time_s"], t_cj_candidate_s=cj["time_s"],
                  separation=sep, cj=cj,
                  notes=["Separation interval brackets observability, not physical creation or thesis outflow.",
                         "CJ interval lists neighboring output times; it does not bracket CJ formation.",
                         "CJ candidate is local to the reported radius, not an initial profile for the entire charge.",
                         "No missing event is replaced with t=0, R_0/D, or theoretical t_sep.",
                         "Numerical thresholds require mesh/output sensitivity checks before hard constraints."])
    if cell_width is None or sample_count is None:
        report["notes"].append("Actual cell widths or populated-shell counts missing; gap has export-grid uncertainty only.")
    if vf is None:
        report["notes"].append("TNT material fraction missing: density alone does not identify the material boundary.")
    if all(p.get("burn_fraction") is None for p in probes):
        report["notes"].append("Burn progress unavailable; CJ state matches do not establish reaction completion.")
    if np.any(np.diff(state_indices) != 1):
        report["notes"].append("Frames were skipped; temporal intervals are limited by the selected output frames.")
    if sep["time_s"] is not None and cj["time_s"] is not None and cj["time_s"] >= sep["time_s"]:
        report["notes"].append("CJ candidate is not earlier than gap event; review data/model before defining an A interval.")
    return _clean(report), _clean(rows), _clean(probes)


def write_event_outputs(out_dir, report, rows, probes):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"events.json").write_text(json.dumps(_clean(report), indent=2, ensure_ascii=False,
                                             allow_nan=False)+"\n", encoding="utf-8")
    for name, records in (("event_trajectories.csv", rows), ("cj_candidates.csv", probes)):
        with (out/name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else [])
            writer.writeheader()
            writer.writerows(_clean(records))
    def time_text(value):
        return "未识别" if value is None else f"{value:.9e} s ({value*1e6:.6g} us)"
    statuses = {"detected": "满足分离检测判据", "candidate": "候选（网格信息不完整）",
                "insufficient_data": "数据缺项", "not_identified": "未识别",
                "state_candidate": "局部状态匹配候选",
                "state_and_burn_candidate": "局部状态及燃烧程度匹配候选"}
    reasons = {"missing_volume_fraction": "缺少材料体积分数，需要从原始结果重新提取",
               "no_persistent_resolved_gap": "没有找到持续满足分辨率要求的相邻压缩空气层",
               "criterion_already_met_at_first_frame": "第一帧已经满足判据，无法给出更早的时间下界",
               "persistent_gap": "首次满足且后续帧持续支持",
               "no_joint_CJ_state_match": "没有找到同时通过材料、P、密度、速度和可用燃烧判据的样本",
               "missing_reference_or_no_valid_samples_in_surface_band": "缺少有效 CJ 参考值或近表面有效样本",
               "joint_state_match_not_proof_of_CJ_formation": "联合状态匹配通过，尚不能证明真实 CJ 形成"}
    sep, cj = report["separation"], report["cj"]
    lines = ["DYNA 数据事件识别（原始仿真时间）", "",
             f"可分辨分离：{time_text(sep['time_s'])}；状态：{statuses[sep['status']]}",
             f"分离判据跨越区间/s：{sep['time_interval_s']}",
             f"接触面半径/m：{sep['R_c_m']}；激波半径/m：{sep['R_s_m']}",
             f"分离诊断：{reasons.get(sep['reason'], sep['reason'])}", "",
             f"CJ 候选：{time_text(cj['time_s'])}；状态：{statuses[cj['status']]}",
             f"CJ 候选位置/m：{cj['position_m']}；相邻输出时间/s：{cj['time_interval_s']}",
             f"CJ 诊断：{reasons.get(cj['reason'], cj['reason'])}", "",
             "分离是网格可分辨的空气层事件，尚未等同于论文出流。",
             "CJ 是局部状态候选；仅有 P/rho/u 匹配不能证明 CJ 形成。",
             "未识别时，events.json 中时间为 null，不用理论时刻替代。",
             "本报告不自动修改训练初始条件或 A/B 硬约束。"]
    closest = cj.get("closest_sample")
    if closest is not None:
        lines.extend(["", f"最接近的诊断样本：t={closest['t']:.9e} s，r={closest['r']:.9e} m，"
                      f"三项中最大相对误差={closest['relative_error']*100:.3g}%。",
                      "该样本仅供排查，不能代替上面未识别的 CJ 时刻。"])
    (out/"events_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"[events] separation={time_text(sep['time_s'])} ({sep['status']}); "
          f"CJ={time_text(cj['time_s'])} ({cj['status']})")
    print(f"[events] details: {out/'events_summary.txt'}")
