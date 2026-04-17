"""Chapman-Jouguet detonation state solver.

Finds the dimensionless specific volume v_CJ = V_CJ / V0 where the Rayleigh
line is tangent to the JWL principal isentrope.  The tangency condition is

    -dP_s/dv |_{v_CJ} = D_CJ^2 * rho0

which is a single transcendental equation in v solved with Brent's method
(scipy.optimize.brentq).  No additional unknowns are introduced because
C = omega * E0 is determined by the JWL parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq  # type: ignore[import]

from .jwl_isentrope import JWLIsentrope, JWLParams, build_isentrope_from_cj


@dataclass(frozen=True)
class CJState:
    """Fully-determined Chapman-Jouguet thermodynamic state."""

    v_CJ: float    # dimensionless V_CJ / V0
    P_CJ: float    # Pa
    rho_CJ: float  # kg/m^3
    u_CJ: float    # m/s  (detonation-product particle velocity in lab frame)
    c_CJ: float    # m/s  (isentropic sound speed at CJ)
    isentrope: JWLIsentrope  # pre-built isentrope object for downstream use


def solve_cj(
    params: JWLParams,
    D_CJ: float,
    bracket: tuple[float, float] = (0.50, 0.95),
    v_max: float = 15.0,
    n_points: int = 800,
) -> CJState:
    """Return the CJ state for TNT with Lee-Tarver JWL parameters.

    Parameters
    ----------
    params : JWLParams
        JWL equation-of-state parameters.
    D_CJ : float
        Detonation velocity (m/s).
    bracket : (float, float)
        Search interval for Brent's method on v = V/V0.
    v_max, n_points :
        Forwarded to :class:`JWLIsentrope` for the u_p look-up table.

    Returns
    -------
    CJState
        Complete CJ state together with a fully-initialised isentrope object.
    """
    rho0 = params.rho0
    target = D_CJ ** 2 * rho0  # Pa: RHS of tangency condition

    def residual(v: float) -> float:
        """-dP_s/dv - D_CJ^2 * rho0  (should be zero at v_CJ)."""
        p = params
        return (
            p.R1 * p.A * np.exp(-p.R1 * v)
            + p.R2 * p.B * np.exp(-p.R2 * v)
            + (1.0 + p.omega) * p.C * v ** (-(2.0 + p.omega))
            - target
        )

    # Validate that the bracket brackets the root
    fa, fb = residual(bracket[0]), residual(bracket[1])
    if fa * fb > 0:
        raise RuntimeError(
            f"CJ tangency root not bracketed: f({bracket[0]:.3f})={fa:.3e}, "
            f"f({bracket[1]:.3f})={fb:.3e}.  Adjust bracket in YAML."
        )

    v_cj = brentq(residual, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12)

    P_cj = (
        params.A * np.exp(-params.R1 * v_cj)
        + params.B * np.exp(-params.R2 * v_cj)
        + params.C * v_cj ** (-(1.0 + params.omega))
    )
    rho_cj = rho0 / v_cj
    # In the lab frame the CJ products move at D_CJ * (1 - v_CJ)
    u_cj = D_CJ * (1.0 - v_cj)
    # Isentropic sound speed squared = V_CJ^2 * (-dP_s/dV)|_{CJ}
    #   = (v_cj/rho0)^2 * rho0 * D_CJ^2 * rho0 = (v_cj * D_CJ)^2
    c_cj = v_cj * D_CJ

    isentrope = build_isentrope_from_cj(
        params=params, v_cj=v_cj, u_cj=u_cj, v_max=v_max, n_points=n_points
    )

    return CJState(
        v_CJ=v_cj,
        P_CJ=P_cj,
        rho_CJ=rho_cj,
        u_CJ=u_cj,
        c_CJ=c_cj,
        isentrope=isentrope,
    )
