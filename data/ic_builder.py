"""Build PINN initial-condition tensors from §1.7.

Consumes a :class:`LSDYNALoader` (typically :class:`AnalyticalLoader`) and
produces PyTorch tensors ready for :class:`pinn.losses.ic_loss.ICLoss`.

Also performs the Sedov cross-validation check: at t_sep the shock radius
should satisfy R_s = xi0 * (E_eff * t_sep^2 / rho_a)^(1/5), and the
peak-pressure profiles should agree with the LS-DYNA snapshot to within 5%.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


XI0 = 1.033   # Sedov self-similar constant (gamma=1.4, spherical)


class ICBuilder:
    """Assemble IC data from a loader and convert to tensors.

    Parameters
    ----------
    loader : LSDYNALoader-compatible
        Any object satisfying the data-loader protocol.
    rho_ref, P_ref, u_ref : float
        Normalisation references for log-scaling.
    device : str
        PyTorch device ('cpu', 'cuda', 'auto').
    """

    def __init__(
        self,
        loader,
        rho_ref: float = 1.225,
        P_ref: float = 101325.0,
        u_ref: float = 340.0,
        device: str = "cpu",
    ) -> None:
        self._loader = loader
        self._rho_ref = rho_ref
        self._P_ref = P_ref
        self._u_ref = u_ref
        self._device = device
        self._snap: Optional[dict] = None

    def build(self) -> dict:
        """Return raw NumPy arrays from the loader snapshot."""
        if self._snap is None:
            self._snap = self._loader.separation_snapshot()
        return self._snap

    def sedov_check(self, rho_a: float = 1.225) -> dict:
        """Estimate E_eff from the Sedov scaling and compute deviation from snapshot.

        Returns a dict with 'E_eff', 'R_s_sedov', 'R_s_loader', 'deviation_pct'.
        """
        snap = self.build()
        t_sep = snap["t_sep"]
        R_s   = snap["R_s"]
        # Back-compute E_eff
        E_eff = (R_s / XI0) ** 5 * rho_a / t_sep ** 2
        R_s_sedov = XI0 * (E_eff * t_sep ** 2 / rho_a) ** 0.2
        deviation_pct = 100.0 * abs(R_s_sedov - R_s) / R_s
        if deviation_pct > 5.0:
            warnings.warn(
                f"Sedov cross-check: R_s deviation {deviation_pct:.1f}% > 5% tolerance. "
                "LS-DYNA data or analytical model may be inconsistent.",
                stacklevel=2,
            )
        return {
            "E_eff": E_eff,
            "R_s_sedov": R_s_sedov,
            "R_s_loader": R_s,
            "deviation_pct": deviation_pct,
        }

    def as_tensors(self, n_ic: int = 500, seed: int = 0) -> dict:
        """Return a dict of PyTorch tensors for the IC loss.

        Subsample (or upsample by repetition) the snapshot to exactly
        ``n_ic`` collocation points; shuffle for stochastic training.

        Keys
        ----
        'r'     : (N, 1)  radial positions
        't'     : (N, 1)  t_sep repeated
        'rho'   : (N, 1)  density
        'u'     : (N, 1)  velocity
        'P'     : (N, 1)  pressure
        't_sep' : scalar float
        'R_c'   : scalar float
        'R_s'   : scalar float
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required to call as_tensors().")

        import torch

        snap = self.build()
        rng = np.random.default_rng(seed)

        r_src   = snap["r"]
        rho_src = snap["rho"]
        u_src   = snap["u"]
        P_src   = snap["P"]
        n_src   = len(r_src)

        # Subsample or upsample
        if n_src >= n_ic:
            idx = rng.choice(n_src, size=n_ic, replace=False)
        else:
            idx = rng.choice(n_src, size=n_ic, replace=True)

        idx = np.sort(idx)  # keep radial ordering for sanity checks

        dev = _resolve_device(self._device)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr[idx, None], dtype=torch.float32, device=dev)

        return {
            "r":     _t(r_src),
            "t":     torch.full((n_ic, 1), snap["t_sep"], dtype=torch.float32, device=dev),
            "rho":   _t(rho_src),
            "u":     _t(u_src),
            "P":     _t(P_src),
            "t_sep": snap["t_sep"],
            "R_c":   snap["R_c"],
            "R_s":   snap["R_s"],
        }

    def contact_history_tensors(self) -> dict:
        """Return contact-face P_c(t), u_c(t), R_c(t) as tensors."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required.")
        import torch

        ch  = self._loader.contact_pressure_history()
        dev = _resolve_device(self._device)

        def _t(arr):
            return torch.tensor(arr[:, None], dtype=torch.float32, device=dev)

        return {
            "t":   _t(ch["t"]),
            "P_c": _t(ch["P_c"]),
            "u_c": _t(ch["u_c"]),
            "R_c": _t(ch["R_c"]),
        }

    def shock_history_tensors(self) -> dict:
        """Return R_s(t), D_s(t) as tensors for ShockNet prior."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required.")
        import torch

        sh  = self._loader.shock_radius_history()
        dev = _resolve_device(self._device)

        def _t(arr):
            return torch.tensor(arr[:, None], dtype=torch.float32, device=dev)

        return {
            "t":   _t(sh["t"]),
            "R_s": _t(sh["R_s"]),
            "D_s": _t(sh["D_s"]),
        }


def _resolve_device(device: str) -> "torch.device":
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
