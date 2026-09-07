"""Uniform-expansion model for the Phase-A contact-face trajectory r_c(t).

Closed-form ODE for the products-air contact face under the assumptions

    1. Products are spatially uniform inside r in [0, r_c(t)] (homogeneous V, P)
    2. Pressure follows the JWL principal isentrope, P = P_JWL(V)
    3. Spherical mass conservation: V(t) = (r_c(t)/R_0)**3
    4. The contact-face velocity is the JWL Riemann invariant u_p(V):
           u_c(t) = u_p(V) = u_CJ + (1/sqrt(rho_0)) * integral_{v_CJ}^{V}
                    sqrt(-dP_s/dv') dv'

The ODE

    dr_c/dt = u_p( (r_c/R_0)**3 )

is integrated from r_c = R_0 (V = 1, products freshly released at t = 0)
up to r_c = R_c, where R_c is supplied by the BIT Sec.4.2.1 contact-state
Riemann match.  The terminal time is t_sep = r_c^{-1}(R_c).

Note: the earlier implementation used the strong-shock formula
u_c = sqrt(2 P_JWL(V) / ((gamma+1) rho_a)) which over-estimates the
contact speed by 2-10x at early expansion (V < 5) and systematically
under-predicts t_sep.  The correct contact-face velocity is the Riemann
invariant u_p(V), which equals the air post-shock velocity ONLY at the
Riemann-match point V = V_c; elsewhere u_p(V) < sqrt(2P/((gamma+1)rho_a)).

References
----------
- Taylor, G.I. (1950) "The formation of a blast wave by a very intense
  explosion. II. The atmospheric explosion of 1945."
- BIT thesis Sec.4.2.1 -- Riemann-match and integration of dt/dV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import]

from .jwl_isentrope import JWLIsentrope


@dataclass(frozen=True)
class UniformExpansionResult:
    """Tabulated solution of the contact-face expansion ODE."""

    t:   np.ndarray   # s        (length n_grid; t[0] = 0, t[-1] = t_sep)
    r_c: np.ndarray   # m        (r_c[0] = R_0, r_c[-1] = R_c)
    V:   np.ndarray   # V/V0     (= (r_c/R_0)**3, dimensionless)
    P:   np.ndarray   # Pa       (P_JWL(V))
    u_c: np.ndarray   # m/s      (= dr_c/dt = u_p(V))
    R_0: float        # m        (charge radius, integration start)
    R_c: float        # m        (terminal contact radius; = r_c[-1])
    t_sep: float      # s        (= t[-1])


def solve_taylor_sadovsky(
    isentrope: JWLIsentrope,
    R_0: float,
    R_c: float,
    rho_a: float,
    gamma_a: float = 1.4,
    n_grid: int = 200,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> UniformExpansionResult:
    """Integrate dt/dr_c = 1/u_p from r_c = R_0 to r_c = R_c.

    The contact-face velocity u_p(V) = u_p( (r_c/R_0)**3 ) is the JWL
    Riemann invariant (products-side expansion speed).  This replaces the
    old strong-shock approximation that under-predicted t_sep by ~2.5x.

    Parameters
    ----------
    isentrope : JWLIsentrope
        Built from physics.cj_solver.solve_cj(...).isentrope.
    R_0 : float
        Charge radius (m).
    R_c : float
        Terminal contact-face radius (m), supplied externally by
        physics.initial_coupling.match_contact via R_c = R_0 * v_x**(1/3).
    rho_a, gamma_a : float
        Ambient air density (kg/m^3) and ratio of specific heats.
        (Retained for API compatibility; no longer used in the ODE.)
    n_grid : int, default 200
        Number of output samples (linearly spaced in r_c on [R_0, R_c]).
    rtol, atol : float
        scipy.integrate.solve_ivp tolerances.
    """
    if R_c <= R_0:
        raise ValueError(
            f"R_c={R_c} must exceed R_0={R_0} (contact must expand outward)."
        )

    def dt_dr(r: float, _t: np.ndarray) -> list[float]:
        v = (r / R_0) ** 3
        u_c = float(isentrope.u_p_of_v(v))
        return [1.0 / max(u_c, 1e-30)]

    r_eval = np.linspace(R_0, R_c, n_grid)
    sol = solve_ivp(
        dt_dr, t_span=(R_0, R_c), y0=[0.0],
        t_eval=r_eval, rtol=rtol, atol=atol, method="LSODA",
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t_arr = sol.y[0]
    r_arr = sol.t
    V_arr = (r_arr / R_0) ** 3
    P_arr = isentrope.P_s(V_arr)
    u_c_arr = isentrope.u_p_of_v(V_arr)

    return UniformExpansionResult(
        t=t_arr, r_c=r_arr,
        V=V_arr, P=P_arr, u_c=u_c_arr,
        R_0=R_0, R_c=float(r_arr[-1]),
        t_sep=float(t_arr[-1]),
    )


def write_csv(result: UniformExpansionResult, path) -> None:
    """Persist the (t, r_c) table to CSV (`extracted/r_c_taylor_sadovsky.csv`).

    Two-column format with header `t,r_c`.  Other columns (V, P, u_c) are
    derived locally and not persisted.
    """
    import os
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    np.savetxt(
        str(path),
        np.column_stack([result.t, result.r_c]),
        delimiter=",", header="t,r_c", comments="",
    )


if __name__ == "__main__":
    # Quick stand-alone sanity print using TNT defaults and a 50 mm sphere.
    from .cj_solver import solve_cj
    from .initial_coupling import match_contact
    from .jwl_isentrope import JWLParams

    params = JWLParams(A=371.2e9, B=3.7471e9, R1=4.15, R2=0.95,
                       omega=0.30, E0=6.0e9, rho0=1630.0)
    cj = solve_cj(params, D_CJ=6930.0)
    R_0 = 0.05
    contact = match_contact(cj, gamma=1.4, rho_a=1.225, P_a=101325.0)
    R_c = R_0 * contact.v_c ** (1.0 / 3.0)

    res = solve_taylor_sadovsky(
        cj.isentrope, R_0=R_0, R_c=R_c,
        rho_a=1.225, gamma_a=1.4,
    )
    print(f"R_0     = {R_0*1e3:6.2f} mm   (P(V_0) = {float(cj.isentrope.P_s(1.0))/1e9:.2f} GPa)")
    print(f"R_c     = {R_c*1e3:6.2f} mm   (V/V_0  = {contact.v_c:.2f})")
    print(f"t_sep   = {res.t_sep*1e6:6.2f} us")
    print(f"u_c[0]  = {res.u_c[0]:8.1f} m/s   (P[0]={res.P[0]/1e9:.2f} GPa, V[0]={res.V[0]:.3f})")
    print(f"u_c[-1] = {res.u_c[-1]:8.1f} m/s   (P[-1]={res.P[-1]/1e9:.3f} GPa, V[-1]={res.V[-1]:.2f})")
