"""JWL principal isentrope and Riemann invariant u_p(V).

For a Jones-Wilkins-Lee equation of state, the principal isentrope passing
through the Chapman-Jouguet point is conventionally written

    P_s(v) = A * exp(-R1 * v) + B * exp(-R2 * v) + C * v ** (-(1 + omega))

with the constant ``C = omega * E0`` (Wescott-Stewart-Davis form), where
``v = V / V0 = rho0 / rho`` is the dimensionless specific volume relative to
the unreacted explosive.

Along this isentrope the Riemann invariant for an outgoing characteristic is

    u_p(V) = u_CJ + integral_{V_CJ}^{V} sqrt(-dP_s/dV') dV'
           = u_CJ + (1/sqrt(rho0)) * integral_{v_CJ}^{v} sqrt(-dP_s/dv') dv'

This module provides the closed-form pressure / slope evaluations, and a
look-up table for ``u_p(v)`` accessible by either ``v`` or ``P``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class JWLParams:
    """Lee-Tarver style JWL equation-of-state parameters."""

    A: float        # Pa
    B: float        # Pa
    R1: float       # dimensionless
    R2: float       # dimensionless
    omega: float    # dimensionless
    E0: float       # J/m^3 (detonation energy per unit initial volume)
    rho0: float     # kg/m^3 (undetonated density)

    @property
    def C(self) -> float:
        """C = omega * E0 (Pa).  Standard WSD-form isentrope constant."""
        return self.omega * self.E0


class JWLIsentrope:
    """Closed-form principal isentrope plus tabulated Riemann invariant u_p(v).

    Parameters
    ----------
    params : JWLParams
        EOS parameters.
    v_cj : float
        Dimensionless specific volume at the CJ point; root of the tangency
        condition (provided externally by ``physics.cj_solver``).
    u_cj : float
        Detonation product velocity at the CJ point (m/s).
    v_max : float, default 15.0
        Upper end of the look-up table in v-space.
    n_points : int, default 800
        Log-spaced grid size.
    """

    def __init__(
        self,
        params: JWLParams,
        v_cj: float,
        u_cj: float,
        v_max: float = 15.0,
        n_points: int = 800,
    ) -> None:
        if v_max <= v_cj:
            raise ValueError("v_max must exceed v_cj")
        self.params = params
        self.v_cj = float(v_cj)
        self.u_cj = float(u_cj)

        # Log-spaced grid on [v_cj, v_max] for u_p(v) integration
        self._v_grid = np.geomspace(self.v_cj, v_max, n_points)
        self._integrand = np.sqrt(self._minus_dPdv(self._v_grid))
        # u_p(v) = u_cj + (1/sqrt(rho0)) * integral
        cumulative = np.concatenate(
            ([0.0], np.cumsum(0.5 * (self._integrand[:-1] + self._integrand[1:])
                              * np.diff(self._v_grid)))
        )
        self._up_grid = self.u_cj + cumulative / np.sqrt(self.params.rho0)
        self._P_grid = self.P_s(self._v_grid)

        # Pre-sort decreasing pressure for inverse interpolation
        order = np.argsort(self._P_grid)  # ascending
        self._P_sorted = self._P_grid[order]
        self._v_sorted = self._v_grid[order]
        self._up_sorted = self._up_grid[order]

    # ---------------------------------------------------------------- closed-form
    def P_s(self, v: np.ndarray | float) -> np.ndarray | float:
        """Principal-isentrope pressure (Pa)."""
        p = self.params
        v_arr = np.asarray(v, dtype=np.float64)
        return (
            p.A * np.exp(-p.R1 * v_arr)
            + p.B * np.exp(-p.R2 * v_arr)
            + p.C * v_arr ** (-(1.0 + p.omega))
        )

    def _minus_dPdv(self, v: np.ndarray | float) -> np.ndarray | float:
        """Return -dP_s/dv (Pa, strictly positive on the principal isentrope)."""
        p = self.params
        v_arr = np.asarray(v, dtype=np.float64)
        return (
            p.R1 * p.A * np.exp(-p.R1 * v_arr)
            + p.R2 * p.B * np.exp(-p.R2 * v_arr)
            + (1.0 + p.omega) * p.C * v_arr ** (-(2.0 + p.omega))
        )

    def minus_dPdV(self, v: np.ndarray | float) -> np.ndarray | float:
        """Return -dP_s/dV (Pa·kg/m^3); useful for sound-speed checks."""
        return self.params.rho0 * self._minus_dPdv(v)

    def sound_speed(self, v: np.ndarray | float) -> np.ndarray | float:
        """Isentropic sound speed c = sqrt(-V^2 dP/dV)  (m/s)."""
        v_arr = np.asarray(v, dtype=np.float64)
        V = v_arr / self.params.rho0
        return np.sqrt(V * V * self.minus_dPdV(v_arr))

    # ---------------------------------------------------------------- table look-ups
    def u_p_of_v(self, v: np.ndarray | float) -> np.ndarray | float:
        """Riemann invariant u_p as a function of v (m/s)."""
        v_arr = np.asarray(v, dtype=np.float64)
        return np.interp(v_arr, self._v_grid, self._up_grid)

    def v_of_P(self, P: np.ndarray | float) -> np.ndarray | float:
        """Inverse isentrope: given P, return v.  Valid for P <= P_s(v_cj)."""
        P_arr = np.asarray(P, dtype=np.float64)
        return np.interp(P_arr, self._P_sorted, self._v_sorted)

    def u_p_of_P(self, P: np.ndarray | float) -> np.ndarray | float:
        """u_p as a function of pressure on the principal isentrope (m/s)."""
        P_arr = np.asarray(P, dtype=np.float64)
        return np.interp(P_arr, self._P_sorted, self._up_sorted)

    # ---------------------------------------------------------------- diagnostics
    @property
    def v_grid(self) -> np.ndarray:
        return self._v_grid

    @property
    def P_grid(self) -> np.ndarray:
        return self._P_grid

    @property
    def up_grid(self) -> np.ndarray:
        return self._up_grid


def build_isentrope_from_cj(
    params: JWLParams,
    v_cj: float,
    u_cj: float,
    v_max: Optional[float] = None,
    n_points: Optional[int] = None,
) -> JWLIsentrope:
    """Convenience constructor with sensible defaults (v_max=15, n=800)."""
    return JWLIsentrope(
        params=params,
        v_cj=v_cj,
        u_cj=u_cj,
        v_max=15.0 if v_max is None else v_max,
        n_points=800 if n_points is None else n_points,
    )
