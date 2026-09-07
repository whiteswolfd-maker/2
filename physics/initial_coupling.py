"""Sec.4.2.1 contact-state Riemann match (BIT thesis, TNT-only path).

At the moment the detonation wave reaches the TNT-air interface the
contact face emits a left-going rarefaction (into the products) and a
right-going shock (into the air).  The contact pressure ``P_x``,
contact velocity ``u_x``, and the products-side specific volume ``v_x``
are determined by matching the two Riemann invariants:

    u_x = u_CJ + Delta_u                                    (4.1)
    Delta_u = integral_{P_x}^{P_CJ} dp/(rho c) along JWL    (4.2)
    u_x = sqrt(2/(gamma_a + 1) * P_x / rho_a)               (4.6)

Both legs are evaluated on the closed-form JWL isentrope (helpers in
:mod:`physics.jwl_isentrope`); the air-shock leg uses the strong-shock
Rankine-Hugoniot expression in :mod:`physics.rh_relations`.

This module exposes only the Riemann-match routine ``match_contact``.
The Taylor-Sadovsky uniform-expansion ODE for the slope curve r_c(t)
lives in :mod:`physics.uniform_expansion`; both are bundled by
:func:`physics.cj_state.compute_separation_state` for the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cj_solver import CJState
from .rh_relations import shock_speed, shock_velocity


@dataclass
class ContactState:
    """Contact-face Riemann match at the moment of charge release."""

    P_c: float   # Pa   -- contact-face pressure (= P_x in BIT notation)
    u_c: float   # m/s  -- contact-face velocity (= u_x; air post-shock side)
    D_s: float   # m/s  -- initial air-shock front speed
    v_c: float   # dimensionless V/V_0 on the JWL isentrope (= V_x / V_0)


def match_contact(
    cj: CJState,
    gamma: float,
    rho_a: float,
    P_a: float,
) -> ContactState:
    """Find the contact-face state by Brent's method on
    ``u_p_products(P_c) - u_s_air(P_c) == 0`` over P_c in (P_a, P_CJ).

    At P_c = P_a the air shock is infinitesimally weak (u_s -> 0) and the
    products are nearly fully expanded, so f > 0; at P_c = P_CJ the
    products have barely expanded (u_p = u_CJ) but the air shock is
    extremely strong (u_s >> u_CJ), so f < 0.  The root is well bracketed.
    """
    from scipy.optimize import brentq  # type: ignore[import]

    iso = cj.isentrope

    def f(P: float) -> float:
        return float(iso.u_p_of_P(P)) - float(shock_velocity(P, gamma, P_a, rho_a))

    fa = f(P_a * 1.01)
    fb = f(cj.P_CJ * 0.999)
    if fa * fb >= 0:
        raise RuntimeError(
            f"Contact-face Riemann root not bracketed: "
            f"f(P_a)={fa:.2e}, f(P_CJ)={fb:.2e}"
        )

    P_c = brentq(f, P_a * 1.01, cj.P_CJ * 0.999,
                 xtol=P_a * 1e-6, rtol=1e-8, maxiter=200)

    return ContactState(
        P_c=float(P_c),
        u_c=float(iso.u_p_of_P(P_c)),
        D_s=float(shock_speed(P_c, gamma, P_a, rho_a)),
        v_c=float(iso.v_of_P(P_c)),
    )
