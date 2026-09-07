"""High-level facade unifying CJ + contact-matching + Taylor-Sadovsky ODE.

This module is a thin convenience layer over

* :mod:`physics.cj_solver`         (CJ tangency solve)
* :mod:`physics.initial_coupling`  (BIT Sec.4.2.1 contact-state Riemann match)
* :mod:`physics.uniform_expansion` (Taylor-Sadovsky ODE for r_c(t), t_sep)

It bundles the JWL parameters together with ambient air properties under a
single ``TNTParams`` dataclass and provides two top-level helpers used by
both the d3plot extractor and the PINN trainer:

* ``compute_cj_state(tnt) -> CJStateBundle``
* ``compute_separation_state(tnt, R_0, *, n_grid=200) -> SeparationStateBundle``

The bundles wrap the lower-level result objects so callers don't have to
juggle three modules' worth of types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cj_solver import CJState, solve_cj
from .initial_coupling import ContactState, match_contact
from .jwl_isentrope import JWLIsentrope, JWLParams
from .uniform_expansion import (
    UniformExpansionResult,
    solve_taylor_sadovsky,
)


# --------------------------------------------------------------------------- params

@dataclass(frozen=True)
class TNTParams:
    """JWL + ambient air parameters bundled together.

    JWL coefficients copied verbatim from the LS-DYNA *MAT_HIGH_EXPLOSIVE_BURN
    + *EOS_JWL card used in this project's simulation
    (B = 3.7471 GPa, E0 = 6.0 GJ/m^3, P_CJ = 21.0 GPa).
    """
    # JWL (Pa, m, s units)
    A:       float = 371.2e9       # 3.712e11 Pa
    B:       float = 3.7471e9      # 3.7471e9 Pa  (LS-DYNA card)
    R1:      float = 4.15
    R2:      float = 0.95
    omega:   float = 0.30
    E0:      float = 6.0e9         # 6e9 J/m^3   (LS-DYNA card)
    rho_TNT: float = 1630.0
    D_CJ:    float = 6930.0
    P_CJ:    float = 21.0e9        # 2.1e10 Pa   (LS-DYNA card; reference target)

    # Ambient air
    P_atm:   float = 101325.0
    rho_a:   float = 1.225
    gamma_a: float = 1.4

    def jwl(self) -> JWLParams:
        return JWLParams(
            A=self.A, B=self.B, R1=self.R1, R2=self.R2,
            omega=self.omega, E0=self.E0, rho0=self.rho_TNT,
        )


# --------------------------------------------------------------------------- bundles

@dataclass
class CJStateBundle:
    """Wraps :class:`CJState` plus the originating :class:`TNTParams`."""

    tnt: TNTParams
    cj:  CJState

    @property
    def isentrope(self) -> JWLIsentrope:
        return self.cj.isentrope

    @property
    def P_CJ(self)   -> float: return float(self.cj.P_CJ)
    @property
    def rho_CJ(self) -> float: return float(self.cj.rho_CJ)
    @property
    def u_CJ(self)   -> float: return float(self.cj.u_CJ)
    @property
    def c_CJ(self)   -> float: return float(self.cj.c_CJ)
    @property
    def v_CJ(self)   -> float: return float(self.cj.v_CJ)


@dataclass
class SeparationStateBundle:
    """Connection-point state at (t_sep, R_c) consumed by the PINN trainer.

    Naming convention (matches the BIT thesis Sec.4.2.1):

    * ``P_x, u_x, rho_x, V_x`` : analytical Riemann-match result at the
      products-air contact face (used as Phase-A convergence gate / sanity
      check, NOT injected into BC or loss directly).
    * ``R_c, t_sep`` : geometry/timing for the AirShockNet inputs.
    * ``t_traj_tail, r_traj_tail`` : rarefaction-tail C- characteristic
      from (t=0, r=R_0) propagating inward at dr/dt = u_CJ - c_CJ.
      Phase-A domain is the fan r in [r_tail(t), r_c(t)].

    ``contact`` and ``ode`` retain the underlying solver outputs for
    diagnostic / plotting purposes.
    """
    tnt: TNTParams
    cj_bundle: CJStateBundle
    R_0: float

    P_x:   float
    u_x:   float
    rho_x: float
    V_x:   float        # = rho_TNT / rho_x   (== contact.v_c)
    R_c:   float
    t_sep: float

    t_traj_tail: np.ndarray = field(repr=False)
    r_traj_tail: np.ndarray = field(repr=False)

    contact: ContactState        = field(repr=False)
    ode:     UniformExpansionResult = field(repr=False)


# --------------------------------------------------------------------------- helpers

def compute_cj_state(tnt: TNTParams) -> CJStateBundle:
    """Solve the JWL CJ tangency for the given TNT parameters."""
    cj = solve_cj(tnt.jwl(), D_CJ=tnt.D_CJ)
    return CJStateBundle(tnt=tnt, cj=cj)


def compute_separation_state(
    tnt: TNTParams,
    R_0: float,
    cj_bundle: Optional[CJStateBundle] = None,
    n_grid: int = 200,
) -> SeparationStateBundle:
    """End-to-end: CJ -> Sec.4.2.1 contact-match -> Taylor-Sadovsky ODE.

    Returns the connection-point state ``(P_x, u_x, rho_x, V_x, R_c, t_sep)``
    plus the underlying solver outputs (for diagnostics / writing CSVs).
    """
    if cj_bundle is None:
        cj_bundle = compute_cj_state(tnt)
    cj = cj_bundle.cj

    contact = match_contact(
        cj, gamma=tnt.gamma_a, rho_a=tnt.rho_a, P_a=tnt.P_atm,
    )

    V_x = float(contact.v_c)
    R_c = R_0 * V_x ** (1.0 / 3.0)
    rho_x = tnt.rho_TNT / V_x

    ode = solve_taylor_sadovsky(
        cj.isentrope, R_0=R_0, R_c=R_c,
        rho_a=tnt.rho_a, gamma_a=tnt.gamma_a,
        n_grid=n_grid,
    )

    # Rarefaction-tail C- characteristic: from (t=0, r=R_0) inward at
    # dr/dt = u_CJ - c_CJ.  Regions inside this line remain at undisturbed
    # CJ state and are excluded from the Phase-A PINN domain.
    t_sep_val = ode.t_sep
    c_CJ = cj.c_CJ
    u_CJ = cj.u_CJ
    dr_dt_tail = u_CJ - c_CJ  # < 0 (inward propagation)
    t_tail = np.linspace(0.0, t_sep_val, n_grid, dtype=np.float64)
    r_tail = R_0 + dr_dt_tail * t_tail

    return SeparationStateBundle(
        tnt=tnt, cj_bundle=cj_bundle, R_0=R_0,
        P_x=float(contact.P_c), u_x=float(contact.u_c), rho_x=rho_x, V_x=V_x,
        R_c=R_c, t_sep=t_sep_val,
        t_traj_tail=t_tail, r_traj_tail=r_tail,
        contact=contact, ode=ode,
    )


if __name__ == "__main__":
    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    print(f"CJ: P_CJ={cj_b.P_CJ/1e9:.3f} GPa  rho_CJ={cj_b.rho_CJ:.1f} kg/m^3  "
          f"u_CJ={cj_b.u_CJ:.1f} m/s  c_CJ={cj_b.c_CJ:.1f} m/s")
    R_0 = 0.050  # spherical charge radius (m)
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)
    print(f"Sec.4.2.1 separation:")
    print(f"  P_x   = {sep.P_x/1e9:.3f} GPa")
    print(f"  u_x   = {sep.u_x:.1f} m/s")
    print(f"  rho_x = {sep.rho_x:.1f} kg/m^3")
    print(f"  V_x   = {sep.V_x:.2f}  (V/V_0)")
    print(f"  R_0   = {R_0*1e3:.2f} mm")
    print(f"  R_c   = {sep.R_c*1e3:.2f} mm")
    print(f"  t_sep = {sep.t_sep*1e6:.2f} us")
