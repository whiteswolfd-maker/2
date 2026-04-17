"""Unit tests for the physics layer (no PyTorch required).

Anchors:
- CJ pressure within ±15% of 21 GPa (tabulated TNT)
- JWL isentrope strictly decreasing
- RH float64 residuals < 1e-6
- Contact matching relative error < 1%
- R_s > R_c throughout quasi-steady series
- u_p(v_CJ) == u_CJ  (boundary of Riemann table)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physics import solve_cj, match_contact, evolve_quasisteady, find_separation
from physics.jwl_isentrope import JWLParams
from physics.rh_relations import rh_residuals, shock_speed, shock_velocity


def _params():
    with open(ROOT / "configs/tnt_spherical.yaml") as f:
        cfg = yaml.safe_load(f)
    exp = cfg["explosive"]
    air = cfg["air"]
    prep = cfg["preprocessing"]
    params = JWLParams(
        A=exp["jwl"]["A"], B=exp["jwl"]["B"],
        R1=exp["jwl"]["R1"], R2=exp["jwl"]["R2"],
        omega=exp["jwl"]["omega"], E0=exp["jwl"]["E0"],
        rho0=exp["rho0"],
    )
    return cfg, params, air, prep


@pytest.fixture(scope="module")
def physics_objects():
    cfg, params, air, prep = _params()
    cj = solve_cj(params, cfg["explosive"]["D_CJ"], bracket=tuple(prep["v_cj_bracket"]))
    contact = match_contact(cj, air["gamma"], air["rho_a"], air["P_a"])
    W   = cfg["explosive"]["W"]
    rho0 = cfg["explosive"]["rho0"]
    R0  = (3.0 * W / (4.0 * math.pi * rho0)) ** (1.0 / 3.0)
    series = evolve_quasisteady(
        R0, cj, contact, air["gamma"], air["rho_a"], air["P_a"],
        dt=prep["quasisteady_dt"], t_max=prep["quasisteady_t_max"],
    )
    sep = find_separation(series, air["P_a"])
    return cfg, cj, contact, series, sep


def test_cj_pressure_range(physics_objects):
    _, cj, *_ = physics_objects
    ref = 21.0e9  # Pa
    err = abs(cj.P_CJ - ref) / ref
    assert err < 0.15, f"P_CJ={cj.P_CJ/1e9:.2f} GPa, err={100*err:.1f}% > 15%"


def test_isentrope_monotone(physics_objects):
    _, cj, *_ = physics_objects
    P = cj.isentrope.P_grid
    assert np.all(np.diff(P) < 0), "JWL isentrope is not strictly decreasing"


def test_u_p_boundary(physics_objects):
    _, cj, *_ = physics_objects
    # At v = v_CJ the integral is zero; u_p should equal u_CJ
    u_p_at_cj = cj.isentrope.u_p_of_v(cj.v_CJ)
    assert abs(u_p_at_cj - cj.u_CJ) < 1.0, (
        f"u_p(v_CJ)={u_p_at_cj:.1f} != u_CJ={cj.u_CJ:.1f}"
    )


def test_rh_float64():
    """Exact Mach-5 shock: RH residuals must be < 1e-6."""
    g, P_a, rho_a = 1.4, 101325.0, 1.225
    c_a = math.sqrt(g * P_a / rho_a)
    M = 5.0
    D_s = M * c_a
    P_s  = P_a * (2.0 * g * M**2 - (g - 1.0)) / (g + 1.0)
    rho_s = rho_a * ((g + 1.0) * M**2) / ((g - 1.0) * M**2 + 2.0)
    u_s  = D_s * (1.0 - rho_a / rho_s)
    r1, r2, r3 = rh_residuals(rho_s, u_s, P_s, D_s, g, P_a, rho_a)
    for name, val in [("r1", r1), ("r2", r2), ("r3", r3)]:
        assert abs(val) < 1e-6, f"RH residual {name}={val:.2e} >= 1e-6"


def test_contact_matching(physics_objects):
    cfg, cj, contact, *_ = physics_objects
    air = cfg["air"]
    u_p = float(cj.isentrope.u_p_of_P(contact.P_c))
    u_s = float(shock_velocity(contact.P_c, air["gamma"], air["P_a"], air["rho_a"]))
    rel = abs(u_p - u_s) / (abs(u_s) + 1e-10)
    assert rel < 0.01, f"Contact mismatch: |u_p - u_s| / u_s = {100*rel:.3f}% > 1%"


def test_shock_ahead_of_contact(physics_objects):
    _, _, _, series, _ = physics_objects
    assert np.all(series.R_s >= series.R_c), "R_s < R_c detected in quasi-steady series"


def test_separation_found(physics_objects):
    _, _, _, _, sep = physics_objects
    assert sep.t_sep > 0, "Separation time should be positive"
    assert sep.R_s > sep.R_c, "At separation R_s must exceed R_c"
