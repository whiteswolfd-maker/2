"""Trainer smoke test: builds the full sequential pipeline on synthetic data
and runs 2 steps per substage to verify wiring.

Does NOT verify convergence (no real d3plot data here).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from pinn.networks import build_networks
from pinn.trainer import (
    _build_air_shock_losses,
    _build_detonation_losses,
    _run_substage,
)


# ============================================================ synthetic data


def _make_synthetic(out: Path, n_t: int = 32, n_x: int = 32,
                    R_0: float = 0.05, x_end: float = 0.5):
    """Synthetic spherical TNT blast (Taylor-Sadovsky-driven r_c(t))."""
    t = np.linspace(0.0, 5e-3, n_t)
    x = np.linspace(0.0, x_end, n_x)

    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    rho_TNT = tnt.rho_TNT
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)

    R_c_t = np.interp(t, sep.ode.t, sep.ode.r_c)
    R_s_t = np.where(t < sep.t_sep, R_c_t, sep.R_c + 1500.0 * (t - sep.t_sep))
    Pbar_p_t = sep.cj_bundle.P_CJ * (R_0 / np.maximum(R_c_t, 1e-9)) ** 3

    X, _ = np.meshgrid(x, t, indexing="xy")
    R_c_grid = R_c_t[:, None]; R_s_grid = R_s_t[:, None]
    rho = np.where(X < R_c_grid, sep.cj_bundle.rho_CJ, 1.225).astype(np.float64)
    u   = np.where((X >= R_c_grid) & (X < R_s_grid), 200.0, 0.0).astype(np.float64)
    P   = np.where(X < R_c_grid, Pbar_p_t[:, None],
                   np.where(X < R_s_grid, 5e5, 1.01325e5)).astype(np.float64)

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "raw_line.npz", t=t, x=x, rho=rho, u=u, P=P)
    np.savetxt(out / "shock.csv",
               np.column_stack([t, R_s_t]), delimiter=",", header="t,R_s", comments="")
    np.savetxt(out / "contact.csv",
               np.column_stack([t, R_c_t]), delimiter=",", header="t,R_c", comments="")
    np.savetxt(out / "product_pressure.csv",
               np.column_stack([t, Pbar_p_t]), delimiter=",", header="t,P_p_bar", comments="")
    np.savetxt(out / "r_c_taylor_sadovsky.csv",
               np.column_stack([sep.ode.t, sep.ode.r_c]),
               delimiter=",", header="t,r_c", comments="")
    metadata = {
        "x_end": x_end, "t_0": float(t[0]), "t_end": float(t[-1]),
        "rho_TNT": rho_TNT, "rho_CJ": sep.cj_bundle.rho_CJ,
        "P_CJ": sep.cj_bundle.P_CJ, "u_CJ": sep.cj_bundle.u_CJ,
        "n_x": n_x, "n_t": n_t,
        "R_0": R_0, "R_c": sep.R_c, "t_sep": sep.t_sep,
        "P_x": sep.P_x, "u_x": sep.u_x, "rho_x": sep.rho_x, "V_x": sep.V_x,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return sep


def _smoke_cfg(extracted: Path, ckpt: Path) -> dict:
    return {
        "data": {"extracted_dir": str(extracted)},
        "air":  {"gamma": 1.4, "rho_a": 1.225, "P_a": 101325.0},
        "domain": {
            "u_ref_A": 2000.0, "t_ref_A": 1e-5,
            "r_anchor_A": 1e-4,
            "u_ref_B": 1000.0, "P_ref_B": 1e6, "t_ref_B": 1e-3,
            "rh_eps":  1e-3, "outflow_scale": 1.0,
        },
        "networks": {
            "detonation": {"depth": 3, "width": 8, "omega0": 30.0},
            "air_shock":  {"depth": 3, "width": 8, "omega0": 30.0},
        },
        "sampling": {
            "A": {"N_data": 64, "N_f": 32, "N_slope": 16},
            "B": {"N_data": 64, "N_f": 32, "N_RH": 16, "N_outflow": 8,
                  "N_IC_amb": 16, "contact_repeat": 4},
        },
        "loss_weights": {
            "A1": {"data_A": 1.0, "IC_A": 1.0, "BC_A_slope": 1.0},
            "A2": {"data_A": 1.0, "IC_A": 1.0, "BC_A_slope": 1.0, "PDE_A": 1.0},
            "B1": {"data_B": 1.0, "IC_B": 1.0},
            "B2a": {"data_B": 1.0, "RH_B": 1.0, "BC_B_outflow": 1.0},
            "B2b": {"data_B": 1.0, "PDE_B": 1.0, "RH_B": 1.0, "BC_B_outflow": 1.0},
        },
        "training": {
            "device": "cpu", "seed": 0, "log_interval": 1,
            "checkpoint_dir": str(ckpt),
            "gate_rel_tol": 0.5,
            "A1": {"steps": 2, "lr": 1e-3},
            "A2": {"steps": 2, "lr": 1e-4},
            "B1": {"steps": 2, "lr": 1e-3},
            "B2a": {"steps": 2, "lr": 1e-4},
            "B2b": {"steps": 2, "lr": 1e-4},
        },
        "validation": {
            "Rs_error_tol": 0.01, "Rc_error_tol": 0.02, "Pshock_error_tol": 0.05,
            "cj_pressure_reference": 21e9, "cj_pressure_tol": 0.1,
            "separation_gate_tol": 0.05,
        },
    }


@pytest.fixture
def smoke(tmp_path):
    extracted = tmp_path / "extracted"
    ckpt = tmp_path / "ckpt"
    sep = _make_synthetic(extracted)
    cfg = _smoke_cfg(extracted, ckpt)
    ds = D3plotLineDataset(extracted)
    return cfg, ds, sep


def test_phase_A_smoke(smoke):
    cfg, ds, sep = smoke
    cj = sep.cj_bundle
    det, _ = build_networks(
        cfg["networks"],
        rho_ref_A=cj.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj.P_CJ,
    )
    losses = _build_detonation_losses(cfg, ds, det, sep, torch.device("cpu"))
    final = _run_substage(
        "A1-smoke", det, losses,
        keys=["data_A", "IC_A", "BC_A_slope"],
        weights=cfg["loss_weights"]["A1"],
        steps=2, lr=1e-3, log_n=10**9,
    )
    assert math.isfinite(final[0])
    final = _run_substage(
        "A2-smoke", det, losses,
        keys=["data_A", "IC_A", "BC_A_slope", "PDE_A"],
        weights=cfg["loss_weights"]["A2"],
        steps=2, lr=1e-4, log_n=10**9,
    )
    assert math.isfinite(final[0])


def test_phase_B_smoke(smoke):
    cfg, ds, sep = smoke
    cj = sep.cj_bundle
    det, asn = build_networks(
        cfg["networks"],
        rho_ref_A=cj.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj.P_CJ,
        rho_ref_B=cfg["air"]["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"],
        P_ref_B=cfg["domain"]["P_ref_B"],
    )
    # Pretend Phase A has been frozen.
    losses = _build_air_shock_losses(cfg, ds, asn, det, sep, torch.device("cpu"))
    final = _run_substage(
        "B1-smoke", asn, losses,
        keys=["data_B", "IC_B"],
        weights=cfg["loss_weights"]["B1"],
        steps=2, lr=1e-3, log_n=10**9,
    )
    assert math.isfinite(final[0])
    final = _run_substage(
        "B2a-smoke", asn, losses,
        keys=["data_B", "RH_B", "BC_B_outflow"],
        weights=cfg["loss_weights"]["B2a"],
        steps=2, lr=1e-4, log_n=10**9,
    )
    assert math.isfinite(final[0])
    final = _run_substage(
        "B2b-smoke", asn, losses,
        keys=["data_B", "PDE_B", "RH_B", "BC_B_outflow"],
        weights=cfg["loss_weights"]["B2b"],
        steps=2, lr=1e-4, log_n=10**9,
    )
    assert math.isfinite(final[0])


def test_spherical_yaml_loads():
    """The shipped spherical yaml must be valid and contain all required keys."""
    with open(ROOT / "configs/tnt_spherical_50mm.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for top in ("data", "air", "domain", "networks", "sampling",
                "loss_weights", "training", "validation"):
        assert top in cfg, f"missing top-level section: {top}"
    for sub in ("A1", "A2", "B1", "B2a", "B2b"):
        assert sub in cfg["loss_weights"]
        assert sub in cfg["training"]
