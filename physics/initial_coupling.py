"""Quasi-steady detonation-products/air coupling from t=0+ to t_sep.

Implements §1.5-1.6 of the specification:

1.5  Contact-face Riemann matching at t=0+:
     Find P_c* such that u_p(P_c*) [products] == u_s(P_c*) [air shock].

1.6  Quasi-steady explicit time integration of {R_c(t), R_s(t), v_c(t)}:
     Each step the contact-face state is read from the JWL isentrope via the
     homogeneous-expansion approximation v_c = (R_c / R0)^3, then the air-shock
     Hugoniot supplies D_s, and both fronts are advanced with a forward-Euler
     step.  Separation is declared when P_c < 20*P_a OR v_c > 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cj_solver import CJState
from .rh_relations import shock_speed, shock_velocity


# --------------------------------------------------------------------------- data types

@dataclass
class ContactState:
    """Contact-face match at t=0+."""

    P_c: float   # Pa   — contact-face pressure
    u_c: float   # m/s  — contact-face velocity (= air post-shock velocity)
    D_s: float   # m/s  — initial shock-front speed
    v_c: float   # dimensionless V/V0 on JWL isentrope


@dataclass
class QuasiSteadySeries:
    """Time-series produced by :func:`evolve_quasisteady`."""

    t:   np.ndarray   # s
    R_c: np.ndarray   # m
    R_s: np.ndarray   # m
    u_c: np.ndarray   # m/s
    D_s: np.ndarray   # m/s
    P_c: np.ndarray   # Pa
    v_c: np.ndarray   # dimensionless


@dataclass
class SeparationState:
    """State at the products-release (separation) moment."""

    t_sep:  float
    R_c:    float
    R_s:    float
    u_c:    float
    D_s:    float
    P_c:    float
    v_c:    float
    series: QuasiSteadySeries = field(repr=False)  # full history


# --------------------------------------------------------------------------- §1.5

def match_contact(
    cj: CJState,
    gamma: float,
    rho_a: float,
    P_a: float,
    P_init_frac: float = 0.5,   # kept for API compatibility; not used (Brent)
    tol: float = 1e-3,
    max_iter: int = 100,
) -> ContactState:
    """Find P_c* such that u_p^{products}(P_c) = u_s^{air}(P_c).

    At P = P_a the air shock is infinitesimally weak (u_s → 0) while the
    Riemann invariant u_p is large (fully expanded products), so f > 0.
    At P = P_CJ the products have barely expanded (u_p = u_CJ ≈ 1800 m/s)
    while the air shock is extremely strong (u_s ≫ u_CJ), so f < 0.
    The root is bracketed in [P_a, P_CJ] and found with Brent's method.
    """
    from scipy.optimize import brentq

    iso = cj.isentrope

    def f(P: float) -> float:
        """u_p_products - u_s_air (positive for low P, negative for high P)."""
        return float(iso.u_p_of_P(P)) - float(shock_velocity(P, gamma, P_a, rho_a))

    # Validate bracket
    fa = f(P_a * 1.01)      # just above ambient (u_p ≫ u_s ≈ 0)
    fb = f(cj.P_CJ * 0.999) # just below CJ (u_s ≫ u_p)
    if fa * fb >= 0:
        raise RuntimeError(
            f"Contact-face Riemann root not bracketed: "
            f"f(P_a)={fa:.2e}, f(P_CJ)={fb:.2e}"
        )

    P_c = brentq(f, P_a * 1.01, cj.P_CJ * 0.999,
                 xtol=P_a * 1e-6, rtol=1e-8, maxiter=200)

    u_c = float(iso.u_p_of_P(P_c))
    D_s = float(shock_speed(P_c, gamma, P_a, rho_a))
    v_c = float(iso.v_of_P(P_c))
    return ContactState(P_c=P_c, u_c=u_c, D_s=D_s, v_c=v_c)


# --------------------------------------------------------------------------- §1.6

def evolve_quasisteady(
    R0: float,
    cj: CJState,
    contact: ContactState,
    gamma: float,
    rho_a: float,
    P_a: float,
    dt: float = 1e-7,
    t_max: float = 5e-3,
) -> QuasiSteadySeries:
    """Explicit forward-Euler integration of the quasi-steady blast model.

    v_c = (R_c / R0)^3 is the AVERAGE dimensionless specific volume of the
    detonation products (homogeneous-expansion approximation).  At t=0 this
    equals 1 (products fill the charge volume at explosive density × v_CJ
    averaged over the sphere, which rounds to ~1 for the bulk).  The Riemann-
    matched contact state (§1.5) characterises the outer contact face, not the
    bulk; it is used by ICBuilder / BCLoss for PINN boundary conditions.

    The initial shock front is set from the Riemann matching D_s* so the
    early air-shock speed is physically consistent.

    Returns the full time-series up to separation or t_max.
    """
    iso = cj.isentrope

    # v_c=1 at R_c=R0: average products state at the moment the detonation
    # front exits the charge surface.
    v_c = 1.0
    P_c = float(iso.P_s(v_c))
    u_c = float(iso.u_p_of_v(v_c))
    # Initial shock speed from Riemann matching (more physical than bulk value)
    D_s = contact.D_s

    t_vals  = [0.0]
    Rc_vals = [R0]
    Rs_vals = [R0]
    uc_vals = [u_c]
    Ds_vals = [D_s]
    Pc_vals = [P_c]
    vc_vals = [v_c]

    R_c = R0
    R_s = R0
    t = 0.0

    while t < t_max:
        R_c += u_c * dt
        R_s += D_s * dt
        t += dt

        v_c = (R_c / R0) ** 3
        P_c = float(iso.P_s(v_c))
        if P_c <= 0:
            break
        u_c = float(iso.u_p_of_v(v_c))
        D_s = float(shock_speed(P_c, gamma, P_a, rho_a))

        t_vals.append(t)
        Rc_vals.append(R_c)
        Rs_vals.append(R_s)
        uc_vals.append(u_c)
        Ds_vals.append(D_s)
        Pc_vals.append(P_c)
        vc_vals.append(v_c)

        # Physical stopping: shock front must always lead the contact face.
        # Once D_s <= u_c the quasi-steady model has broken down.
        if D_s <= u_c:
            break

    return QuasiSteadySeries(
        t=np.asarray(t_vals),
        R_c=np.asarray(Rc_vals),
        R_s=np.asarray(Rs_vals),
        u_c=np.asarray(uc_vals),
        D_s=np.asarray(Ds_vals),
        P_c=np.asarray(Pc_vals),
        v_c=np.asarray(vc_vals),
    )


def find_separation(
    series: QuasiSteadySeries,
    P_a: float,
    P_threshold_ratio: float = 20.0,
    v_threshold: float = 7.0,
) -> SeparationState:
    """Locate the earliest index where P_c < thresh * P_a or v_c > v_thresh.

    If no separation is found the last point in the series is returned.
    """
    P_thresh = P_threshold_ratio * P_a
    sep_idx = None
    for i in range(len(series.t)):
        if series.P_c[i] < P_thresh or series.v_c[i] > v_threshold:
            sep_idx = i
            break

    if sep_idx is None:
        sep_idx = len(series.t) - 1

    return SeparationState(
        t_sep=float(series.t[sep_idx]),
        R_c=float(series.R_c[sep_idx]),
        R_s=float(series.R_s[sep_idx]),
        u_c=float(series.u_c[sep_idx]),
        D_s=float(series.D_s[sep_idx]),
        P_c=float(series.P_c[sep_idx]),
        v_c=float(series.v_c[sep_idx]),
        series=series,
    )
