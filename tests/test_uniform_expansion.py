"""Taylor-Sadovsky uniform-expansion ODE tests (CLAUDE.md anchors).

* r_c(0) = R_0, r_c ends at the analytic R_c, t strictly increasing,
  u_c strictly increasing.
* Raises when the requested R_c does not exceed R_0.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from physics.cj_state import TNTParams, compute_cj_state
from physics.initial_coupling import match_contact
from physics.uniform_expansion import solve_taylor_sadovsky

R_0 = 0.05


@pytest.fixture(scope="module")
def solved():
    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    cj = cj_b.cj
    contact = match_contact(cj, gamma=tnt.gamma_a, rho_a=tnt.rho_a, P_a=tnt.P_atm)
    R_c = R_0 * contact.v_c ** (1.0 / 3.0)
    res = solve_taylor_sadovsky(
        cj.isentrope, R_0=R_0, R_c=R_c,
        rho_a=tnt.rho_a, gamma_a=tnt.gamma_a,
    )
    return tnt, cj, contact, R_c, res


def test_starts_at_charge_radius(solved):
    _, _, _, _, res = solved
    assert math.isclose(res.r_c[0], R_0, rel_tol=1e-9)


def test_ends_at_requested_Rc(solved):
    _, _, _, R_c, res = solved
    assert math.isclose(res.r_c[-1], R_c, rel_tol=1e-9)


def test_t_strictly_increasing(solved):
    _, _, _, _, res = solved
    assert np.all(np.diff(res.t) > 0)


def test_u_c_strictly_increasing(solved):
    _, _, _, _, res = solved
    assert np.all(np.diff(res.u_c) > 0)


def test_t_sep_positive_and_tabulated(solved):
    tnt, cj, contact, R_c, res = solved
    assert res.t_sep > 0
    assert math.isclose(res.V[-1], (res.r_c[-1] / R_0) ** 3, rel_tol=1e-9)


def test_raises_when_Rc_not_outward():
    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    with pytest.raises(ValueError):
        solve_taylor_sadovsky(
            cj_b.cj.isentrope, R_0=R_0, R_c=R_0 * 0.9,
            rho_a=tnt.rho_a, gamma_a=tnt.gamma_a,
        )
