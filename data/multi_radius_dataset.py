"""Multi-radius dataset with Hopkinson-Cranz coordinate scaling.

Wraps multiple ``D3plotLineDataset`` instances (one per charge radius)
and exposes a single unified interface where spatial/temporal coordinates
are scaled to the dimensionless Hopkinson-Cranz variables::

    tau = t / M^(1/3)       [s / kg^(1/3)]
    Z   = r / M^(1/3)       [m / kg^(1/3)]

where ``M^(1/3) = (4/3 pi rho_TNT)^(1/3) * R_0``.

Under this scaling the Phase-A rarefaction fan and the Phase-B contact
point are **identical** for all radii, so a single PINN can be trained
on merged data.

Usage
-----
    spec = [
        {"extracted_dir": "extracted/",       "R_0": 0.05},
        {"extracted_dir": "extracted_100mm/", "R_0": 0.10},
        {"extracted_dir": "extracted_100mm/", "R_0": 0.10},
    ]
    ds = MultiRadiusDataset(spec, device="cuda")
    tau, Z, rho, u, P = ds.sample_grid_points(4096)
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from data.d3plot_dataset import D3plotLineDataset
from physics.cj_state import TNTParams, compute_cj_state, compute_separation_state

# ---------------------------------------------------------------------------
# Global constants (verified constant across all R_0)
# ---------------------------------------------------------------------------
_ALPHA = (4.0 / 3.0 * math.pi * 1630.0) ** (1.0 / 3.0)  # ~18.97  kg^(1/3)/m
TAU_SEP = 1.112e-5    # s / kg^(1/3)   — identical for all R_0
Z_C     = 0.117576     # m / kg^(1/3)   — contact radius at separation
Z_R0    = 0.052712     # m / kg^(1/3)   — charge surface
Z_MAX   = 15.0         # m / kg^(1/3)   — physical domain boundary (ΔP→0)


def _M_third(R_0: float) -> float:
    """Return M^(1/3) = alpha * R_0."""
    return _ALPHA * float(R_0)


def _interp1d_torch(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation (same impl as d3plot_dataset._interp1d_torch)."""
    flat = x.flatten()
    n = xp.numel()
    idx = torch.searchsorted(xp, flat).clamp(1, n - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    w = ((flat - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    return ((1.0 - w) * y0 + w * y1).view(x.shape)


class MultiRadiusDataset:
    """Unified multi-radius dataset with Hopkinson-Cranz scaled coordinates.

    Wraps one ``D3plotLineDataset`` per charge radius and presents them as a
    single dataset whose coordinates ``(tau, Z)`` are identical across all
    radii.  On each call to ``sample_grid_points`` a radius is chosen at
    random and the raw ``(t, r)`` data is scaled on-the-fly.

    The 50-mm dataset provides the reference shock trajectory (the shock
    front in scaled coordinates is the same for all radii).
    """

    def __init__(
        self,
        radius_specs: list[dict],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        t_data_min_phys: float = 8.0e-6,
    ) -> None:
        """Construct the unified dataset.

        Parameters
        ----------
        radius_specs:
            List of dicts with keys ``"extracted_dir"`` (str) and
            ``"R_0"`` (float, metres).
        device:
            Torch device string.
        dtype:
            Torch dtype for stored tensors.
        t_data_min_phys:
            Physical-time cutoff (seconds).  LS-DYNA d3plot frames before
            this time contain P=0 artefacts from *MAT_HIGH_EXPLOSIVE_BURN.
            Per-radius ``tau_data_min`` is computed as this value divided
            by each radius's ``M^(1/3)``.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.t_data_min_phys = float(t_data_min_phys)

        # ---- load per-radius datasets ----------------------------------------
        self._datasets: list[D3plotLineDataset] = []
        self._M_third: list[float] = []
        self._R_0: list[float] = []
        self._tau_data_min: list[float] = []
        self._tau_end: list[float] = []
        self._Z_max_data: list[float] = []  # x_end / M^(1/3) per radius

        for spec in radius_specs:
            R_0 = float(spec["R_0"])
            ds = D3plotLineDataset(str(spec["extracted_dir"]), device=str(device), dtype=dtype)
            mt = _M_third(R_0)
            self._datasets.append(ds)
            self._M_third.append(mt)
            self._R_0.append(R_0)
            self._tau_data_min.append(t_data_min_phys / mt)
            self._tau_end.append(float(ds.t_end) / mt)
            self._Z_max_data.append(float(ds.x_end) / mt)

        self._num_radii = len(self._datasets)

        # ---- derived constants (identical across radii) ---------------------
        # Use the first dataset's physics (they're all the same in scaled coords)
        ds0 = self._datasets[0]
        self.tau_sep = TAU_SEP
        self.Z_c = Z_C
        self.Z_R0 = Z_R0
        self.Z_max = Z_MAX
        self.Z_max_data = min(self._Z_max_data)  # conservative: all cover this

        # ---- scaled Taylor-Sadovsky trajectory (same for all radii) ----------
        # Use the first dataset's TS table and scale it
        ts_t = ds0.t_traj_rc.to(dtype=dtype)
        ts_r = ds0.r_traj_rc.to(dtype=dtype)
        mt0 = self._M_third[0]
        self.tau_traj_rc: torch.Tensor = ts_t / mt0
        self.Z_traj_rc: torch.Tensor = ts_r / mt0

        # ---- build sep bundle for the 50-mm (reference) radius --------------
        tnt = TNTParams()
        self._sep_bundle = compute_separation_state(tnt, R_0=self._R_0[0])

        # ---- scaled rarefaction tail (from sep_bundle) -----------------------
        self.tau_traj_tail = torch.as_tensor(
            self._sep_bundle.t_traj_tail, dtype=dtype
        ) / mt0
        self.Z_traj_tail = torch.as_tensor(
            self._sep_bundle.r_traj_tail, dtype=dtype
        ) / mt0

        # ---- scaled shock trajectory (from 50-mm, used for all radii) --------
        # R_s(tau) is unique in scaled coordinates
        t_traj = ds0.t_traj.to(dtype=dtype) / mt0
        r_traj = ds0.R_s_traj.to(dtype=dtype) / mt0
        valid = (r_traj > 0) & (t_traj > 0)
        self._tau_shock = t_traj[valid]
        self._Z_shock = r_traj[valid]

        # Shock speed dZ_s/dtau via central differences
        dz = torch.diff(self._Z_shock)
        dt = torch.diff(self._tau_shock).clamp(min=1e-30)
        self._D_zs_traj = torch.cat([dz / dt, (dz[-1] / dt[-1]).unsqueeze(0)])
        self._dRs_dt_traj = self._D_zs_traj  # compat alias for AirShockRHLoss

        # ---- CJ state -------------------------------------------------------
        cj = compute_cj_state(tnt)
        self.rho_CJ = cj.rho_CJ
        self.u_CJ = cj.u_CJ
        self.P_CJ = cj.P_CJ
        self.rho_TNT = tnt.rho_TNT

        # ---- separation state (from sep_bundle, same for all radii) ----------
        self.P_x = self._sep_bundle.P_x
        self.u_x = self._sep_bundle.u_x
        self.rho_x = self._sep_bundle.rho_x
        self.V_x = self._sep_bundle.V_x
        self.R_c_sep = self._sep_bundle.R_c
        self.t_sep_phys = self._sep_bundle.t_sep

        # ---- grid dimensions (for downstream code that expects these) --------
        self.N_t = max(ds.t_arr.numel() for ds in self._datasets)
        self.N_x = max(ds.x_arr.numel() for ds in self._datasets)
        self.t_end = max(self._tau_end)  # max tau across all radii
        self.x_end = max(self._Z_max_data)  # full data span; state_at clamps per-radius

        # ---- expose raw grid (scaled, from first dataset for reference) ------
        self.t_arr = ds0.t_arr.to(dtype=dtype) / mt0
        self.x_arr = ds0.x_arr.to(dtype=dtype) / mt0

    # -------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------

    def sample_grid_points(
        self, n: int, seed: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``n`` grid points proportional to each radius's Z coverage.

        Three naturally overlapping domains::

        """
        # Uniform pick across radii (all cover same scaled domain)
        i = torch.randint(0, self._num_radii, (1,)).item()
        ds = self._datasets[i]
        mt = self._M_third[i]
        tau_min = self._tau_data_min[i]

        # Rejection-loop to filter early-time artifacts
        tau_vals: list[torch.Tensor] = []
        Z_vals: list[torch.Tensor] = []
        rho_vals: list[torch.Tensor] = []
        u_vals: list[torch.Tensor] = []
        P_vals: list[torch.Tensor] = []
        target = n
        attempts = 0
        while (sum(t.numel() for t in tau_vals) < target) and attempts < 8:
            t, x, rho, u, P = ds.sample_grid_points(target)
            tau = t / mt
            Z = x / mt
            mask = tau.squeeze(-1) >= tau_min
            if mask.any():
                tau_vals.append(tau[mask])
                Z_vals.append(Z[mask])
                rho_vals.append(rho[mask])
                u_vals.append(u[mask])
                P_vals.append(P[mask])
            attempts += 1
        if not tau_vals:
            return self.sample_grid_points(n, seed)  # retry with different radius
        return (
            torch.cat(tau_vals)[:target].view(-1, 1).to(self.device),
            torch.cat(Z_vals)[:target].view(-1, 1).to(self.device),
            torch.cat(rho_vals)[:target].view(-1, 1).to(self.device),
            torch.cat(u_vals)[:target].view(-1, 1).to(self.device),
            torch.cat(P_vals)[:target].view(-1, 1).to(self.device),
        )

    # -------------------------------------------------------------------
    # Interpolation (for slope-BC and data losses)
    # -------------------------------------------------------------------

    def state_at(
        self, tau: torch.Tensor, Z: torch.Tensor, radius_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bilinear-interpolate (rho, u, P) at scaled coordinates."""
        mt = self._M_third[radius_idx]
        t = tau * mt
        r = Z * mt
        return self._datasets[radius_idx].state_at(t, r)

    # -------------------------------------------------------------------
    # Trajectory accessors (scaled)
    # -------------------------------------------------------------------

    def r_c_taylor_sadovsky(self, tau: torch.Tensor) -> torch.Tensor:
        """Interpolate Z_c(tau) from the scaled Taylor-Sadovsky table."""
        return _interp1d_torch(tau, self._to_device(self.tau_traj_rc), self._to_device(self.Z_traj_rc))

    def _to_device(self, t):
        """Move tensor to input device if needed."""
        if t.device != self.device:
            return t.to(self.device)
        return t

    def R_s(self, tau: torch.Tensor) -> torch.Tensor:
        """Shock radius Z_s(tau) from 50-mm reference trajectory."""
        return _interp1d_torch(tau, self._to_device(self._tau_shock), self._to_device(self._Z_shock))

    def R_c(self, tau: torch.Tensor) -> torch.Tensor:
        """Contact-surface radius Z_c(tau)."""
        return self.r_c_taylor_sadovsky(tau)

    def D_s(self, tau: torch.Tensor) -> torch.Tensor:
        """Shock speed dZ_s/dtau."""
        return _interp1d_torch(tau, self._to_device(self._tau_shock), self._to_device(self._D_zs_traj))

    # -------------------------------------------------------------------
    # Properties for backward compatibility
    # -------------------------------------------------------------------

    @property
    def t_traj_rc(self) -> torch.Tensor:
        return self.tau_traj_rc

    @property
    def r_traj_rc(self) -> torch.Tensor:
        return self.Z_traj_rc

    @property
    def R_s_traj(self) -> torch.Tensor:
        return self._Z_shock

    @property
    def t_traj(self) -> torch.Tensor:
        return self._tau_shock

    @property
    def R_c_traj(self) -> torch.Tensor:
        return self.Z_traj_rc  # Taylor-Sadovsky IS the contact trajectory

    @property
    def R_0(self) -> float:
        return self._R_0[0]

    @property
    def sep_bundle(self):
        return self._sep_bundle
