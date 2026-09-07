"""Dataset tests: 4th-order diff, linear interp, bilinear state_at, sampling.

Builds a small synthetic extracted/ dir in a tmp_path and checks:
* _central_diff_4th(dy/dt of t^2) == 2t on the interior.
* _interp1d_torch linear + out-of-range clamp.
* state_at reproduces an analytic bilinear field at interior points.
* sample_grid_points returns the requested number of points.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from data.d3plot_dataset import (
    D3plotLineDataset,
    _central_diff_4th,
    _interp1d_torch,
)


def _make_synthetic(extracted: pytest.TempPathFactory):
    t = np.linspace(0.0, 2.0e-4, 21)              # s
    x = np.linspace(0.0, 1.0, 41)                 # m
    # bilinear-friendly fields:  rho = 1.0 + 200*t + 2*x   (linear)
    T, X = np.meshgrid(t, x, indexing="ij")
    rho = 1.0 + 200.0 * T + 2.0 * X
    u = 50.0 * X - 10.0 * T
    P = 1.0e5 + 2.0e8 * T * X

    # trajectories (monotone shock/contact for detection sanity)
    t_traj = np.linspace(0.0, 2.0e-4, 21)
    R_s = 0.05 + 3000.0 * t_traj
    R_c = 0.05 + 1500.0 * t_traj
    Pbar = 1.0e6 * (R_c / 0.05) ** (-3.0)

    np.savez(extracted / "raw_line.npz", t=t, x=x, rho=rho, u=u, P=P)
    np.savetxt(extracted / "shock.csv", np.column_stack([t_traj, R_s]),
               delimiter=",", header="t,R_s", comments="")
    np.savetxt(extracted / "contact.csv", np.column_stack([t_traj, R_c]),
               delimiter=",", header="t,R_c", comments="")
    np.savetxt(extracted / "product_pressure.csv",
               np.column_stack([t_traj, Pbar]),
               delimiter=",", header="t,P_p_bar", comments="")
    metadata = {
        "R_0": 0.05, "x_end": 1.0, "t_0": 0.0, "t_end": 2.0e-4,
        "rho_TNT": 1630.0, "rho_CJ": 2206.0, "P_CJ": 21.8e9, "u_CJ": 1811.0,
        "R_c": 0.11, "t_sep": 1.05e-5,
    }
    (extracted / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return t, x


@pytest.fixture()
def dataset(tmp_path):
    t, x = _make_synthetic(tmp_path)
    ds = D3plotLineDataset(tmp_path)
    return ds, t, x


def test_central_diff_4th_quadratic():
    t = np.linspace(0.0, 1.0, 30)
    y = t ** 2
    dy = _central_diff_4th(y, t)
    assert np.allclose(dy[3:-3], 2.0 * t[3:-3], atol=1e-6)


def test_interp1d_linear_and_clamp():
    xp = torch.linspace(0.0, 1.0, 11)
    fp = 2.0 * xp
    xq = torch.tensor([0.25, 0.5, 0.75, 2.0, -1.0])
    yq = _interp1d_torch(xq, xp, fp)
    assert torch.allclose(yq[:3], 2.0 * xq[:3], atol=1e-5)
    assert yq[3].item() == pytest.approx(2.0)   # clamped to fp[-1]
    assert yq[4].item() == pytest.approx(0.0)   # clamped to fp[0]


def test_state_at_bilinear_interior(dataset):
    ds, t, x = dataset
    t_q = torch.tensor([[5.0e-5], [1.0e-4]])
    x_q = torch.tensor([[0.25], [0.75]])
    rho_q, u_q, P_q = ds.state_at(t_q, x_q)
    # rho = 1.0 + 200*t + 2*x is linear -> bilinear interp is exact.
    expected = 1.0 + 200.0 * t_q + 2.0 * x_q
    assert torch.allclose(rho_q, expected, atol=1e-4)
    assert rho_q.shape == (2, 1)


def test_required_files_loaded(dataset):
    ds, t, x = dataset
    assert ds.rho_grid.shape == (len(t), len(x))
    assert ds.R_0 == pytest.approx(0.05)
    assert ds.Z_R0 == pytest.approx(0.05)
    assert ds.t_traj_rc is None or ds.t_traj_rc.numel() > 0


def test_sample_grid_points_count(dataset):
    ds, t, x = dataset
    t_s, x_s, rho, u, P = ds.sample_grid_points(n=128, seed=0)
    assert t_s.shape == (128, 1)
    assert rho.shape == (128, 1)


def test_trajectory_accessors(dataset):
    ds, t, x = dataset
    tq = torch.tensor([[5.0e-5]])
    R_s = ds.R_s(tq)
    R_c = ds.R_c(tq)
    assert R_s.shape == (1, 1) and R_c.shape == (1, 1)
    assert R_c.item() > 0.05
