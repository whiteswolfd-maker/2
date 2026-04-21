"""LS-DYNA data loader protocol and back-ends.

The :class:`LSDYNALoader` protocol specifies the three data products
required by the PINN training pipeline:

1. ``separation_snapshot()``     – field (rho, u, P)(r) at t = t_sep over
                                   the air region [R_c^sep, R_s^sep].
2. ``contact_pressure_history()`` – P_c(t) time series for L_BC(a).
3. ``shock_radius_history()``    – R_s(t) time series for ShockNet prior.

``AnalyticalLoader`` satisfies the protocol by synthesising these from the
Phase-1 quasi-steady model (§1.6) and a Sedov-Taylor self-similar profile
(§1.7).  ``FileLoader`` is a stub that accepts a CSV directory path and will
raise ``NotImplementedError`` until implemented.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import]


@runtime_checkable
class LSDYNALoader(Protocol):
    """Minimal data-access interface consumed by ICBuilder and trainer."""

    def separation_snapshot(self) -> dict:
        """Return the flow-field snapshot at t = t_sep.

        Returns
        -------
        dict with keys:
            't_sep'  : float   – separation time (s)
            'R_c'    : float   – contact-face radius at t_sep (m)
            'R_s'    : float   – shock-front radius at t_sep (m)
            'r'      : ndarray – radial sample points in [R_c, R_s] (m)
            'rho'    : ndarray – density (kg/m^3)
            'u'      : ndarray – radial velocity (m/s)
            'P'      : ndarray – pressure (Pa)
        """
        ...

    def contact_pressure_history(self) -> dict:
        """Return the contact-face pressure history P_c(t).

        Returns
        -------
        dict with keys:
            't'    : ndarray  – time array (s), starts at t_sep
            'P_c'  : ndarray  – contact-face pressure (Pa)
            'u_c'  : ndarray  – contact-face velocity (m/s)
            'R_c'  : ndarray  – contact-face radius (m)
        """
        ...

    def shock_radius_history(self) -> dict:
        """Return the shock-front radius R_s(t) for ShockNet warm-up.

        Returns
        -------
        dict with keys:
            't'    : ndarray  – time array (s)
            'R_s'  : ndarray  – shock-front radius (m)
            'D_s'  : ndarray  – shock speed (m/s)
        """
        ...


# =====================================================================  AnalyticalLoader


def _sedov_profile(xi: np.ndarray, gamma: float = 1.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate Sedov-Taylor self-similar profiles f_rho, f_u, f_P (xi in [0,1]).

    Uses the analytic approximations of Sedov (1959) / Landau & Lifshitz §106.
    The peak values (at xi=1, behind the shock) are unity; profiles are valid
    for gamma=1.4.

    xi = r / R_s (self-similar variable).

    Returns
    -------
    f_rho, f_u, f_P : ndarray
        Normalised profiles, each in [0, 1].
    """
    # For gamma=1.4 Sedov strong-shock limit the classical approximation:
    # f_u(xi)   ~ xi
    # f_rho(xi) ~ rho_ratio * xi^k  (k ~ 6 for gamma=1.4 hemisphere)
    # f_P(xi)   ~ P_behind * xi^m   (m ~ 12 for gamma=1.4)
    # These are rough fits; adequate for IC initialisation.
    xi = np.asarray(xi, dtype=np.float64)
    f_u = xi
    f_P = xi ** 12
    # density profile: approaches max-rho just behind shock, falls off rapidly
    f_rho = xi ** 6
    # Clamp to [0, 1]
    f_u   = np.clip(f_u,   0.0, 1.0)
    f_P   = np.clip(f_P,   0.0, 1.0)
    f_rho = np.clip(f_rho, 0.0, 1.0)
    return f_rho, f_u, f_P


class AnalyticalLoader:
    """Synthesise LS-DYNA outputs from §1.6 quasi-steady model + Sedov-Taylor.

    Parameters
    ----------
    sep : SeparationState
        Output of :func:`physics.initial_coupling.find_separation`.
    series : QuasiSteadySeries
        Full quasi-steady time series (sep.series).
    gamma, rho_a, P_a :
        Ambient air properties.
    n_r : int
        Number of radial sample points in the IC snapshot.
    rh_funcs : dict, optional
        Pre-computed post-shock state at t_sep for density normalisation.
        If ``None``, computed internally from rh_relations.
    """

    def __init__(
        self,
        sep,          # SeparationState (import avoided to prevent circularity)
        gamma: float = 1.4,
        rho_a: float = 1.225,
        P_a:   float = 101325.0,
        n_r:   int = 500,
    ) -> None:
        self._sep = sep
        self._series = sep.series
        self._gamma = gamma
        self._rho_a = rho_a
        self._P_a = P_a
        self._n_r = n_r

    # ---------------------------------------------------------------- public API

    def separation_snapshot(self) -> dict:
        sep = self._sep
        g = self._gamma
        rho_a = self._rho_a
        P_a = self._P_a

        R_c = sep.R_c
        R_s = sep.R_s
        D_s = sep.D_s
        P_s = sep.P_c  # post-shock pressure at separation moment

        # Post-shock state (Rankine-Hugoniot)
        from physics.rh_relations import rho_ratio as rh_rho_ratio, shock_velocity
        rho_ratio_val = float(rh_rho_ratio(P_s, g, P_a))
        rho_s = rho_a * rho_ratio_val
        u_s   = float(shock_velocity(P_s, g, P_a, rho_a))

        # Sedov self-similar profile
        r = np.linspace(R_c, R_s, self._n_r)
        xi_sedov = r / R_s
        f_rho, f_u, f_P = _sedov_profile(xi_sedov, gamma=g)

        rho_arr = rho_a + (rho_s - rho_a) * f_rho
        u_arr   = f_u * u_s
        P_arr   = P_a + (P_s - P_a) * f_P

        # Hard-enforce exact RH values at r = R_s
        rho_arr[-1] = rho_s
        u_arr[-1]   = u_s
        P_arr[-1]   = P_s
        # Contact face: match products pressure
        P_arr[0]    = P_s   # approximate; exact coupling handled by L_BC
        u_arr[0]    = sep.u_c

        return {
            "t_sep": sep.t_sep,
            "R_c":   R_c,
            "R_s":   R_s,
            "r":     r,
            "rho":   rho_arr,
            "u":     u_arr,
            "P":     P_arr,
        }

    def contact_pressure_history(self) -> dict:
        s = self._series
        # Trim to t >= t_sep (there may be only a few points beyond t_sep
        # in the quasi-steady series; extrapolate gently if needed)
        mask = s.t >= (self._sep.t_sep - 1e-12)
        return {
            "t":   s.t[mask],
            "P_c": s.P_c[mask],
            "u_c": s.u_c[mask],
            "R_c": s.R_c[mask],
        }

    def shock_radius_history(self) -> dict:
        s = self._series
        return {
            "t":   s.t,
            "R_s": s.R_s,
            "D_s": s.D_s,
        }


# =====================================================================  FileLoader


class FileLoader:
    """Load LS-DYNA outputs from CSV files in a results directory.

    CSV format (one file per quantity, header row + numeric rows):
        snapshot.csv   columns: r, rho, u, P
        contact.csv    columns: t, P_c, u_c, R_c
        shock.csv      columns: t, R_s, D_s
        metadata.json  keys:    t_sep, R_c_sep, R_s_sep

    Raises
    ------
    NotImplementedError
        Until the user supplies data files and this method is implemented.
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def separation_snapshot(self) -> dict:
        raise NotImplementedError(
            "FileLoader: provide LS-DYNA CSV results in "
            f"'{self._data_dir}' and implement parsing here."
        )

    def contact_pressure_history(self) -> dict:
        raise NotImplementedError(
            "FileLoader: contact pressure CSV not yet implemented."
        )

    def shock_radius_history(self) -> dict:
        raise NotImplementedError(
            "FileLoader: shock radius CSV not yet implemented."
        )
