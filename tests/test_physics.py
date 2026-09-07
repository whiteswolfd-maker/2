"""Physics-layer unit tests (CLAUDE.md anchors).

* CJ pressure: 21 GPa +/- 15% (LS-DYNA card nominal).
* JWL isentrope strictly decreasing; e_iso(v) torch <-> numpy machine-precision.
* RH float64 closure at M=5.
* Sec.4.2.1 contact match |u_p - u_s| / u_s < 1%.
* Taylor-Sadovsky: r_c(0)=R_0, monotone, endpoint == analytic R_c.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from physics.jwl_isentrope import JWLParams
from physics.rh_relations import rh_residuals, shock_velocity

R_0 = 0.05  # 50 mm spherical charge


@pytest.fixture(scope="module")
def bundles():
    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)
    return tnt, cj_b, sep


def test_cj_pressure_near_card(bundles):
    _, cj_b, _ = bundles
    assert abs(cj_b.P_CJ - 21.0e9) / 21.0e9 < 0.15


def test_isentrope_strictly_decreasing(bundles):
    _, cj_b, _ = bundles
    P = cj_b.cj.isentrope.P_grid
    assert np.all(np.diff(P) < 0)


def test_jwl_e_iso_torch_matches_numpy(bundles):
    """torch (pinn.operators.jwl_isentropic_e) vs numpy (JWLIsentrope.e_iso_of_v)."""
    tnt, cj_b, _ = bundles
    iso = cj_b.cj.isentrope
    vs = np.geomspace(iso.v_cj, 8.0, 64)
    e_np = iso.e_iso_of_v(vs, e_cj=0.0)

    A, B, R1, R2, om = tnt.A, tnt.B, tnt.R1, tnt.R2, tnt.omega
    C = tnt.omega * tnt.E0
    from pinn.operators import jwl_isentropic_e
    rho = torch.as_tensor(tnt.rho_TNT / vs, dtype=torch.float64)
    e_t = jwl_isentropic_e(
        rho, A=A, B=B, R1=R1, R2=R2, omega=om,
        C=C, rho0=tnt.rho_TNT, v_cj=iso.v_cj, e_cj=0.0,
    )
    assert torch.allclose(e_t, torch.as_tensor(e_np, dtype=torch.float64),
                          rtol=1e-9, atol=1e-9)


def test_rh_float64_closure():
    g, P_a, rho_a = 1.4, 101325.0, 1.225
    c_a = math.sqrt(g * P_a / rho_a)
    M = 5.0
    D_s = M * c_a
    P_s = P_a * (2.0 * g * M ** 2 - (g - 1.0)) / (g + 1.0)
    rho_s = rho_a * ((g + 1.0) * M ** 2) / ((g - 1.0) * M ** 2 + 2.0)
    u_s = D_s * (1.0 - rho_a / rho_s)
    r1, r2, r3 = rh_residuals(rho_s, u_s, P_s, D_s, g, P_a, rho_a)
    assert max(abs(r1), abs(r2), abs(r3)) < 1e-6


def test_contact_match_consistency(bundles):
    _, cj_b, sep = bundles
    u_p = float(cj_b.cj.isentrope.u_p_of_P(sep.P_x))
    u_s = float(shock_velocity(sep.P_x, cj_b.tnt.gamma_a, cj_b.tnt.P_atm,
                               cj_b.tnt.rho_a))
    assert abs(u_p - u_s) / (abs(u_s) + 1e-10) < 1e-2


def test_taylor_sadovsky_endpoint(bundles):
    _, cj_b, sep = bundles
    res = sep.ode
    assert math.isclose(res.r_c[0], R_0, rel_tol=1e-9)
    assert math.isclose(res.r_c[-1], sep.R_c, rel_tol=1e-9)
    assert math.isclose(res.t_sep, sep.t_sep, rel_tol=1e-9)
    assert np.all(np.diff(res.t) > 0)
    assert np.all(np.diff(res.u_c) > 0)   # u_c strictly increasing
    assert math.isclose(sep.R_c, R_0 * sep.V_x ** (1.0 / 3.0), rel_tol=1e-9)


def test_jwl_params_c_is_omega_times_e0():
    p = JWLParams(A=371.2e9, B=3.7471e9, R1=4.15, R2=0.95, omega=0.30,
                  E0=6.0e9, rho0=1630.0)
    assert math.isclose(p.C, p.omega * p.E0, rel_tol=1e-15)
