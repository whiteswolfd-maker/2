"""Rankine-Hugoniot (RH) jump relations for a perfect ideal gas.

All formulae correspond to a normal shock propagating at speed D_s into
undisturbed air with state (rho_a, 0, P_a).

The post-shock (air-side) state satisfies:
    mass   :  rho_a * D_s = rho_s * (D_s - u_s)
    momentum:  P_a + rho_a * D_s^2 = P_s + rho_s * (D_s - u_s)^2
    energy  :  h_a + D_s^2 / 2 = h_s + (D_s - u_s)^2 / 2

which, for a gamma-law gas, give the closed-form expressions below.

Reference: any gasdynamics textbook, e.g. Anderson (2003) §3.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- public API

def shock_speed(P_s: np.ndarray | float, gamma: float, P_a: float, rho_a: float) -> np.ndarray | float:
    """Shock-front speed D_s (m/s) into undisturbed air at pressure P_s.

    Derived from momentum + mass conservation:
        D_s = sqrt([(gamma+1)*P_s + (gamma-1)*P_a] / (2*rho_a))

    Parameters
    ----------
    P_s : array_like
        Post-shock pressure (Pa).
    gamma : float
        Specific-heat ratio of air (1.4).
    P_a : float
        Ambient pressure (Pa).
    rho_a : float
        Ambient density (kg/m^3).
    """
    P_s_arr = np.asarray(P_s, dtype=np.float64)
    return np.sqrt(((gamma + 1.0) * P_s_arr + (gamma - 1.0) * P_a) / (2.0 * rho_a))


def shock_velocity(
    P_s: np.ndarray | float, gamma: float, P_a: float, rho_a: float
) -> np.ndarray | float:
    """Post-shock particle velocity u_s (m/s) in the lab frame.

        u_s = (P_s - P_a) / (rho_a * D_s)
            = (P_s - P_a) * sqrt(2 / (rho_a * [(gamma+1)*P_s + (gamma-1)*P_a]))
    """
    P_s_arr = np.asarray(P_s, dtype=np.float64)
    D_s = shock_speed(P_s_arr, gamma, P_a, rho_a)
    return (P_s_arr - P_a) / (rho_a * D_s)


def rho_ratio(
    P_s: np.ndarray | float, gamma: float, P_a: float
) -> np.ndarray | float:
    """Density ratio rho_s / rho_a across the shock.

        rho_s / rho_a = [(gamma+1)*P_s + (gamma-1)*P_a]
                        / [(gamma-1)*P_s + (gamma+1)*P_a]
    """
    P_s_arr = np.asarray(P_s, dtype=np.float64)
    num = (gamma + 1.0) * P_s_arr + (gamma - 1.0) * P_a
    den = (gamma - 1.0) * P_s_arr + (gamma + 1.0) * P_a
    return num / den


def rh_residuals(
    rho_m: float,
    u_m: float,
    P_m: float,
    D_s: float,
    gamma: float,
    P_a: float,
    rho_a: float,
) -> tuple[float, float, float]:
    """Evaluate the three RH jump-condition residuals.

    Quantities with subscript ``m`` are post-shock (minus side, just behind
    the front); the undisturbed state ahead is (rho_a, 0, P_a).

    Returns
    -------
    r1 : mass-flux residual
    r2 : momentum-flux residual
    r3 : energy-flux (enthalpy) residual
    """
    h_m = gamma * P_m / ((gamma - 1.0) * rho_m)
    h_a = gamma * P_a / ((gamma - 1.0) * rho_a)
    r1 = rho_m * (D_s - u_m) - rho_a * D_s
    r2 = rho_m * (D_s - u_m) ** 2 + P_m - rho_a * D_s ** 2 - P_a
    r3 = h_m + 0.5 * (D_s - u_m) ** 2 - h_a - 0.5 * D_s ** 2
    return r1, r2, r3
