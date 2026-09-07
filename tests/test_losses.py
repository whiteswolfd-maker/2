"""Smoke tests for the new loss modules (Phase A + Phase B).

Each test wires up a synthetic d3plot dataset (with the Taylor-Sadovsky
slope CSV included) and verifies each loss returns a finite, non-negative
scalar.  Numerical accuracy is checked separately in the operator and
physics tests; this file exercises the wiring.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from pinn.losses import (
    AirShockDataLoss,
    AirShockICLoss,
    AirShockOutflowLoss,
    AirShockPDELoss,
    AirShockRHLoss,
    DetonationDataLoss,
    DetonationICLoss,
    DetonationPDELoss,
    DetonationSlopeBCLoss,
)
from pinn.networks import AirShockNet, DetonationNet


# ============================================================ synthetic data + fixtures


def _make_synthetic(out: Path, n_t: int = 80, n_x: int = 64,
                    R_0: float = 0.05, x_end: float = 0.5):
    """Synthetic spherical TNT blast for the loss-module wiring tests.

    Uses the project's actual physics to drive the slope curve r_c(t) and
    the post-separation R_s(t) trajectory, so the loss objects see
    realistic spherical inputs.
    """
    t = np.linspace(0.0, 5e-3, n_t)
    x = np.linspace(0.0, x_end, n_x)

    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    rho_TNT = tnt.rho_TNT
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)

    # Trajectories: R_s(t) = R_c(t_sep) + 1500 (t - t_sep) for t > t_sep
    R_c_t = np.interp(t, sep.ode.t, sep.ode.r_c)
    R_s_t = np.where(t < sep.t_sep, R_c_t,
                     sep.R_c + 1500.0 * (t - sep.t_sep))
    Pbar_p_t = sep.cj_bundle.P_CJ * (R_0 / np.maximum(R_c_t, 1e-9)) ** 3

    X, T = np.meshgrid(x, t, indexing="xy")
    R_c_grid = R_c_t[:, None]
    R_s_grid = R_s_t[:, None]

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


@pytest.fixture
def setup(tmp_path):
    sep = _make_synthetic(tmp_path)
    ds = D3plotLineDataset(tmp_path)
    return ds, sep


# ============================================================ Phase A losses


def test_detonation_data_loss_runs(setup):
    ds, sep = setup
    net = DetonationNet(depth=3, width=16,
                        rho_ref=sep.cj_bundle.rho_CJ, u_ref=2000.0,
                        P_ref=sep.cj_bundle.P_CJ)
    t_tl = torch.as_tensor(sep.t_traj_tail, dtype=torch.float32)
    r_tl = torch.as_tensor(sep.r_traj_tail, dtype=torch.float32)
    L = DetonationDataLoss(
        net, ds, tau_sep=sep.t_sep,
        tau_traj_rc=ds.t_traj_rc, Z_traj_rc=ds.r_traj_rc,
        tau_traj_tail=t_tl, Z_traj_tail=r_tl,
        N=128, u_ref=2000.0,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_detonation_pde_loss_runs(setup):
    ds, sep = setup
    tnt = sep.tnt; cj = sep.cj_bundle
    net = DetonationNet(depth=3, width=16,
                        rho_ref=cj.rho_CJ, u_ref=2000.0, P_ref=cj.P_CJ)
    t_tl = torch.as_tensor(sep.t_traj_tail, dtype=torch.float32)
    r_tl = torch.as_tensor(sep.r_traj_tail, dtype=torch.float32)
    L = DetonationPDELoss(
        net, tau_sep=sep.t_sep,
        tau_traj_rc=ds.t_traj_rc, Z_traj_rc=ds.r_traj_rc,
        tau_traj_tail=t_tl, Z_traj_tail=r_tl,
        A=tnt.A, B=tnt.B, R1=tnt.R1, R2=tnt.R2, omega=tnt.omega,
        C=tnt.omega * tnt.E0, rho0=tnt.rho_TNT, v_cj=cj.v_CJ, e_cj=0.0,
        N_f=64,
        rho_ref=cj.rho_CJ, u_ref=2000.0, P_ref=cj.P_CJ, t_ref=1e-5,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_detonation_ic_loss_runs(setup):
    _, sep = setup
    cj = sep.cj_bundle
    net = DetonationNet(depth=3, width=16,
                        rho_ref=cj.rho_CJ, u_ref=2000.0, P_ref=cj.P_CJ)
    L = DetonationICLoss(net, rho_CJ=cj.rho_CJ, P_CJ=cj.P_CJ,
                         Z_anchor=1e-3, u_ref=2000.0)
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_detonation_slope_bc_loss_runs(setup):
    ds, sep = setup
    cj = sep.cj_bundle
    net = DetonationNet(depth=3, width=16,
                        rho_ref=cj.rho_CJ, u_ref=2000.0, P_ref=cj.P_CJ)
    L = DetonationSlopeBCLoss(
        net, ds, tau_sep=sep.t_sep,
        tau_traj_rc=ds.t_traj_rc, Z_traj_rc=ds.r_traj_rc,
        N=64, u_ref=2000.0,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


# ============================================================ Phase B losses


def test_air_shock_data_loss_runs(setup):
    ds, sep = setup
    net = AirShockNet(depth=3, width=16, rho_ref=1.225, u_ref=1000.0, P_ref=1e6)
    L = AirShockDataLoss(net, ds, t_sep=sep.t_sep, R_c=sep.R_c, N=128, u_ref=1000.0)
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_air_shock_ic_loss_runs(setup):
    ds, sep = setup
    cj = sep.cj_bundle
    det = DetonationNet(depth=3, width=16,
                        rho_ref=cj.rho_CJ, u_ref=2000.0, P_ref=cj.P_CJ)
    asn = AirShockNet(depth=3, width=16, rho_ref=1.225, u_ref=1000.0, P_ref=1e6)
    L = AirShockICLoss(
        asn, det,
        t_sep=sep.t_sep, R_c=sep.R_c, x_end=ds.x_end,
        rho_a=1.225, P_a=101325.0,
        N_amb=64, contact_repeat=8, u_ref=1000.0,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_air_shock_pde_loss_runs(setup):
    ds, sep = setup
    asn = AirShockNet(depth=3, width=16, rho_ref=1.225, u_ref=1000.0, P_ref=1e6)
    L = AirShockPDELoss(
        asn, t_sep=sep.t_sep, t_end=ds.t_end, R_c=sep.R_c, x_end=ds.x_end,
        gamma=1.4, N_f=64,
        rho_ref=1.225, u_ref=1000.0, P_ref=1e6, t_ref=1e-3,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_air_shock_rh_loss_runs(setup):
    ds, sep = setup
    asn = AirShockNet(depth=3, width=16, rho_ref=1.225, u_ref=1000.0, P_ref=1e6)
    L = AirShockRHLoss(
        asn, ds, t_sep=sep.t_sep, t_end=ds.t_end,
        gamma=1.4, rho_a=1.225, P_a=101325.0,
        eps=1e-3, N_s=64, u_ref=1000.0,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_air_shock_outflow_loss_runs(setup):
    ds, sep = setup
    asn = AirShockNet(depth=3, width=16, rho_ref=1.225, u_ref=1000.0, P_ref=1e6)
    L = AirShockOutflowLoss(
        asn, t_sep=sep.t_sep, t_end=ds.t_end, x_end=ds.x_end,
        gamma=1.4, N=32, scale=1.0,
    )
    out = L()
    assert torch.isfinite(out) and out.item() >= 0.0


def test_air_shock_outflow_ambient_zero():
    """A network that always returns ambient (rho_a, 0, P_a) should yield
    zero outflow residual since R+ = u + 2 c / (gamma - 1) is then constant
    in r."""
    rho_a, P_a, gamma = 1.225, 101325.0, 1.4

    class Ambient(torch.nn.Module):
        def forward(self, t, r):
            # Multiply by 0*r so the output stays in the autograd graph
            zero = 0.0 * r
            return (
                zero + rho_a,
                zero,
                zero + P_a,
            )

    L = AirShockOutflowLoss(
        Ambient(), t_sep=0.0, t_end=1e-3, x_end=0.5,
        gamma=gamma, N=64, scale=1.0,
    )
    out = L()
    assert out.item() < 1e-10
