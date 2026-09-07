"""Tests for data/extract_d3plot.py — pure-numpy detection / interpolation
helpers that do NOT require an actual d3plot file.

The lasso-python integration (loading, axis filtering, history-variable
slot mapping) must be validated against real LS-DYNA output by running
``python -m data.extract_d3plot --d3plot <path> --debug`` locally.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.extract_d3plot import (
    _detect_R_s,
    _detect_R_c,
    _avg_product_pressure,
    _find_t0,
    _interpolate_to_grid,
)


# ============================================================ R_s detection


def test_detect_R_s_basic():
    """Step-up pressure profile: R_s = last x with P > 1.05 P_atm."""
    x = np.linspace(0, 1, 101)
    P_atm = 1e5
    P = np.full_like(x, P_atm)
    P[(x >= 0.0) & (x <= 0.4)] = 5e5  # shocked region
    R_s = _detect_R_s(x, P, P_atm, 1.05)
    assert abs(R_s - 0.4) < 1e-2, f"R_s = {R_s}, expected ~0.4"


def test_detect_R_s_no_shock():
    """Uniform ambient: R_s = 0."""
    x = np.linspace(0, 1, 50)
    P = np.full_like(x, 1e5)
    assert _detect_R_s(x, P, 1e5, 1.05) == 0.0


# ============================================================ R_c detection


def test_detect_R_c_basic():
    """Density profile with high product behind contact face at x=0.05."""
    x = np.linspace(0, 0.5, 201)
    rho = np.where(x < 0.05, 200.0, 1.225)
    R_c = _detect_R_c(x, rho, rho_threshold=5.0)
    assert abs(R_c - 0.05) < 1e-2, f"R_c = {R_c}, expected ~0.05"


def test_detect_R_c_no_product():
    rho = np.full(50, 1.225)
    x = np.linspace(0, 0.5, 50)
    assert _detect_R_c(x, rho, rho_threshold=5.0) == 0.0


# ============================================================ avg product pressure


def test_avg_product_pressure_constant():
    """Constant P over [0, R_c] should give P̄_p = P."""
    x = np.linspace(0, 0.5, 101)
    P = np.full_like(x, 1e9)
    Pbar = _avg_product_pressure(x, P, R_c=0.05)
    assert abs(Pbar - 1e9) / 1e9 < 1e-3, f"P̄_p = {Pbar:.3e}, expected ~1e9"


def test_avg_product_pressure_linear():
    """Linear P from 2e9 (x=0) down to 1e9 (x=0.05) → mean = 1.5e9."""
    x = np.linspace(0, 0.5, 1001)
    R_c = 0.05
    P = np.where(x <= R_c, 2e9 - (1e9 / R_c) * x, 1e9)
    Pbar = _avg_product_pressure(x, P, R_c)
    assert abs(Pbar - 1.5e9) / 1.5e9 < 5e-3, f"P̄_p = {Pbar:.3e}, expected 1.5e9"


def test_avg_product_pressure_zero_Rc():
    x = np.linspace(0, 0.5, 51)
    P = np.full_like(x, 1e9)
    assert _avg_product_pressure(x, P, R_c=0.0) == 0.0


# ============================================================ t_0 root find


def test_find_t0_linear_R_c():
    """R_c grows linearly from 0.04 to 0.06 over [0, 1e-5];
    R_c = 0.05 reached at t = 5e-6."""
    times = np.linspace(0, 1e-5, 11)
    R_c = 0.04 + (0.06 - 0.04) * times / 1e-5
    t_0 = _find_t0(times, R_c, L_0=0.05)
    assert abs(t_0 - 5e-6) < 1e-7, f"t_0 = {t_0}, expected 5e-6"


def test_find_t0_already_above():
    """If R_c >= L_0 from start, t_0 = times[0]."""
    times = np.linspace(0, 1e-5, 11)
    R_c = np.full_like(times, 0.06)
    t_0 = _find_t0(times, R_c, L_0=0.05)
    assert t_0 == times[0]


def test_find_t0_never_reaches():
    """If R_c never reaches L_0, t_0 = times[-1] (best fallback)."""
    times = np.linspace(0, 1e-5, 11)
    R_c = np.full_like(times, 0.04)
    t_0 = _find_t0(times, R_c, L_0=0.05)
    assert t_0 == times[-1]


# ============================================================ slab interpolation


def test_interpolate_to_grid_axisymmetric():
    """Cells on the same spherical shell average to the input value."""
    n_per_shell = 8
    r_vals = np.linspace(0.05, 0.4, 20)
    centroids, rho_in, u_in, P_in = [], [], [], []
    for ri in r_vals:
        for _ in range(n_per_shell):
            v = np.random.randn(3)
            v /= np.linalg.norm(v)
            centroids.append(v * ri)
            rho_in.append(2.0)   # constant fields
            u_in.append(100.0)
            P_in.append(5e6)
    centroids = np.array(centroids)
    rho_in = np.array(rho_in); u_in = np.array(u_in); P_in = np.array(P_in)

    r_grid = np.linspace(0.0, 0.5, 51)   # dr = 0.01
    rho_g, u_g, P_g, _ = _interpolate_to_grid(
        centroids, rho_in, u_in, P_in, r_grid,
    )
    # in-range portion should be ~ constant
    in_range = (r_grid >= r_vals[0]) & (r_grid <= r_vals[-1])
    assert np.allclose(rho_g[in_range], 2.0, atol=1e-6)
    assert np.allclose(P_g[in_range],   5e6, atol=1e-3)


def test_interpolate_to_grid_shell_separation():
    """Cells at very different true radii land in their own shells; a cell
    with the same x but large |y| (hence larger true radius) does NOT
    contaminate the near shell."""
    centroids = np.array([
        [0.10, 0.0,  0.0],    # r = 0.100
        [0.10, 0.40, 0.0],    # r = 0.412 (same x, far y)
        [0.50, 0.0,  0.0],    # r = 0.500
    ])
    rho = np.array([1.0, 999.0, 3.0])
    u   = np.zeros(3)
    P   = np.array([1e5, 9e9, 3e5])

    r_grid = np.linspace(0.0, 0.5, 51)   # dr = 0.01
    rho_g, _, P_g, _ = _interpolate_to_grid(centroids, rho, u, P, r_grid)
    # r=0.100 -> shell idx 10 ; r=0.412 -> idx 41 ; r=0.500 -> idx 50
    assert abs(rho_g[10] - 1.0) < 1e-9
    assert abs(rho_g[41] - 999.0) < 1e-6
    assert abs(rho_g[50] - 3.0) < 1e-9
    assert abs(P_g[10] - 1e5) < 1.0
