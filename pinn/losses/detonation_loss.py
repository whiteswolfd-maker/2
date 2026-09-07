"""Loss modules for the Phase-A DetonationNet.

Four terms (see plan §"Loss functions"):

* ``DetonationDataLoss``  L_data,A  — MSE vs. d3plot inside the FAN
* ``DetonationPDELoss``   L_PDE,A   — spherical Euler residuals with JWL EOS
* ``DetonationICLoss``    L_IC,A    — single-point CJ anchor at (t=0, r=0)
* ``DetonationSlopeBC``   L_BC,A,slope — MSE on the slope r = r_c(t) vs. d3plot


All sampling is rejected against the fan ``r_tail(t) <= r <= r_c(t), t <= t_sep``
so the DetonationNet only ever sees data / collocation inside the rarefaction
fan.  The CJ core (r < r_tail(t)) is handled analytically and excluded from
the PINN learning domain.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pinn.networks import DetonationNet
from pinn.operators import (
    jwl_isentropic_e,
    spherical_euler_residuals,
)


# ============================================================ helpers


def _interp_rc(x: torch.Tensor, x_tab: torch.Tensor, y_tab: torch.Tensor) -> torch.Tensor:
    """Linear interpolation y(x) clamped at [x_tab[0], x_tab[-1]]."""
    flat = x.flatten()
    n = x_tab.numel()
    idx = torch.searchsorted(x_tab, flat).clamp(1, n - 1)
    x0 = x_tab[idx - 1]; x1 = x_tab[idx]
    y0 = y_tab[idx - 1]; y1 = y_tab[idx]
    w = ((flat - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    return ((1 - w) * y0 + w * y1).view(x.shape)


# ============================================================ L_data,A


class DetonationDataLoss(nn.Module):
    """L_data,A — d3plot supervision inside the Phase-A fan domain.

    Operates in scaled (tau, Z) or physical (t, r) coordinates — the
    logic is identical either way.  Rejects points outside the fan and
    ``rho < rho_min`` (air-domain contamination guard).
    """

    def __init__(
        self,
        net: DetonationNet,
        dataset,
        tau_sep: float,
        tau_traj_rc: torch.Tensor,
        Z_traj_rc: torch.Tensor,
        tau_traj_tail: torch.Tensor,
        Z_traj_tail: torch.Tensor,
        N: int = 4096,
        tau_data_min: float = 0.0,
        rho_min: float = 0.0,
        u_ref: float = 2000.0,
        w_rho: float = 1.0,
        w_u:   float = 0.5,
        w_P:   float = 1.0,
        max_resample: int = 8,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.tau_sep = float(tau_sep)
        self.tau_data_min = float(tau_data_min)
        self.rho_min = float(rho_min)
        self.register_buffer("tau_traj_rc", tau_traj_rc.to(device).float(), persistent=False)
        self.register_buffer("Z_traj_rc", Z_traj_rc.to(device).float(), persistent=False)
        self.register_buffer("tau_traj_tail", tau_traj_tail.to(device).float(), persistent=False)
        self.register_buffer("Z_traj_tail", Z_traj_tail.to(device).float(), persistent=False)
        self.N = N
        self.u_ref = u_ref
        self.w_rho = w_rho
        self.w_u   = w_u
        self.w_P   = w_P
        self.max_resample = max_resample
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        ds = self.dataset
        target = self.N
        # Earliest data frame inside the Phase-A window.  Sampling starts there
        # (not at tau_data_min) so the interpolated targets between t=0
        # (unreacted TNT) and the first product frame — a non-physical blend —
        # never enter the loss.
        t_arr = ds.t_arr
        in_win = t_arr >= self.tau_data_min
        first_tau = float(t_arr[in_win].min()) if in_win.any() else float(self.tau_data_min)
        tau_lo = max(first_tau, self.tau_data_min)
        if tau_lo >= self.tau_sep:
            return torch.tensor(0.0, device=self.device)

        # Fan-targeted sampling: tau uniform in [tau_lo, tau_sep], Z uniform
        # inside [Z_tail(tau), Z_c(tau)].  All points are in the fan by
        # construction (the old whole-grid rejection kept ~6/32768 points).
        tau_vals, Z_vals, rho_t, u_t, P_t = [], [], [], [], []
        attempts = 0
        while sum(t_.numel() for t_ in tau_vals) < target and attempts < self.max_resample:
            tau = tau_lo + torch.rand(target, 1, device=self.device) * (self.tau_sep - tau_lo)
            Z_c = _interp_rc(tau, self.tau_traj_rc.to(tau.device), self.Z_traj_rc.to(tau.device))
            Z_tl = _interp_rc(tau, self.tau_traj_tail.to(tau.device), self.Z_traj_tail.to(tau.device))
            Z = Z_tl + torch.rand(target, 1, device=self.device) * (Z_c - Z_tl).clamp(min=1e-12)
            rho_d, u_d, P_d = ds.state_at(tau, Z)
            mask = rho_d.squeeze(-1) >= self.rho_min
            if mask.any():
                tau_vals.append(tau[mask]); Z_vals.append(Z[mask])
                rho_t.append(rho_d[mask].view(-1, 1))
                u_t.append(u_d[mask].view(-1, 1))
                P_t.append(P_d[mask].view(-1, 1))
            attempts += 1
        if not tau_vals:
            return torch.tensor(0.0, device=self.device)

        tau = torch.cat(tau_vals)[:target].view(-1, 1).to(self.device)
        Z   = torch.cat(Z_vals)[:target].view(-1, 1).to(self.device)
        rho_t = torch.cat(rho_t)[:target].view(-1, 1).to(self.device)
        u_t   = torch.cat(u_t)[:target].view(-1, 1).to(self.device)
        P_t   = torch.cat(P_t)[:target].view(-1, 1).to(self.device)

        rho_p, u_p, P_p = self.net(tau, Z)
        eps = 1e-10
        log_rho_err = torch.log(rho_p + eps) - torch.log(rho_t + eps)
        log_P_err   = torch.log(P_p   + eps) - torch.log(P_t   + eps)
        u_err       = (u_p - u_t) / self.u_ref
        return (
            self.w_rho * (log_rho_err ** 2).mean()
            + self.w_u  * (u_err       ** 2).mean()
            + self.w_P  * (log_P_err   ** 2).mean()
        )


# ============================================================ L_PDE,A


class DetonationPDELoss(nn.Module):
    """L_PDE,A — spherical Euler residual at uniformly sampled collocation points.

    Sampling:  uniform (tau, Z) in ``[0, tau_sep] x [Z_tail_min, Z_c_max]``
    with rejection to confine to the fan.

    All coordinates can be scaled (tau, Z) or physical (t, r) — the PDE
    form is identical under linear scaling.
    """

    def __init__(
        self,
        net: DetonationNet,
        tau_sep: float,
        tau_traj_rc: torch.Tensor,
        Z_traj_rc: torch.Tensor,
        tau_traj_tail: torch.Tensor,
        Z_traj_tail: torch.Tensor,
        *,
        A: float, B: float, R1: float, R2: float, omega: float,
        C: float, rho0: float, v_cj: float, e_cj: float = 0.0,
        N_f: int = 8192,
        rho_ref: float = 1630.0,
        u_ref:   float = 2000.0,
        P_ref:   float = 21.0e9,
        t_ref:   float = 1.0e-5,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.tau_sep = float(tau_sep)
        self.register_buffer("tau_traj_rc", tau_traj_rc.to(device).float(), persistent=False)
        self.register_buffer("Z_traj_rc", Z_traj_rc.to(device).float(), persistent=False)
        self.register_buffer("tau_traj_tail", tau_traj_tail.to(device).float(), persistent=False)
        self.register_buffer("Z_traj_tail", Z_traj_tail.to(device).float(), persistent=False)
        self.N_f   = int(N_f)
        self.A = float(A); self.B = float(B); self.R1 = float(R1); self.R2 = float(R2)
        self.omega = float(omega); self.C = float(C)
        self.rho0 = float(rho0); self.v_cj = float(v_cj); self.e_cj = float(e_cj)
        self.rho_ref = float(rho_ref); self.u_ref = float(u_ref)
        self.P_ref   = float(P_ref);   self.t_ref = float(t_ref)
        self.device  = torch.device(device)

    def _sample_collocation(self) -> tuple[torch.Tensor, torch.Tensor]:
        kept_tau, kept_Z = [], []
        target = self.N_f
        attempts = 0
        max_att = 8
        Z_tail_min = float(self.Z_traj_tail.min().item())
        Z_c_max = float(self.Z_traj_rc[-1].item())
        while sum(t_.numel() for t_ in kept_tau) < target and attempts < max_att:
            tau = torch.rand(target, 1, device=self.device) * self.tau_sep
            Z = Z_tail_min + torch.rand(target, 1, device=self.device) * (Z_c_max - Z_tail_min)
            Z_c = _interp_rc(tau, self.tau_traj_rc.to(tau.device), self.Z_traj_rc.to(tau.device))
            Z_tl = _interp_rc(tau, self.tau_traj_tail.to(tau.device), self.Z_traj_tail.to(tau.device))
            mask = ((Z.squeeze(-1) <= Z_c.squeeze(-1))
                    & (Z.squeeze(-1) >= Z_tl.squeeze(-1)))
            if mask.any():
                kept_tau.append(tau[mask])
                kept_Z.append(Z[mask])
            attempts += 1
        if not kept_tau:
            tau = torch.rand(target, 1, device=self.device) * self.tau_sep
            Z_c = _interp_rc(tau, self.tau_traj_rc.to(tau.device), self.Z_traj_rc.to(tau.device))
            Z_tl = _interp_rc(tau, self.tau_traj_tail.to(tau.device), self.Z_traj_tail.to(tau.device))
            Z_mid = (Z_tl + Z_c) * 0.5
            return tau, Z_mid.view(-1, 1)
        return (torch.cat(kept_tau)[:target].view(-1, 1),
                torch.cat(kept_Z)[:target].view(-1, 1))

    def forward(self) -> torch.Tensor:
        tau, Z = self._sample_collocation()
        tau = tau.requires_grad_(True)
        Z = Z.requires_grad_(True)
        rho, u, P = self.net(tau, Z)
        e = jwl_isentropic_e(
            rho,
            A=self.A, B=self.B, R1=self.R1, R2=self.R2, omega=self.omega,
            C=self.C, rho0=self.rho0, v_cj=self.v_cj, e_cj=self.e_cj,
        )
        R_m, R_p, R_e = spherical_euler_residuals(rho, u, P, e, Z, tau)

        s_m = self.rho_ref * self.u_ref / self.t_ref
        s_p = self.rho_ref * self.u_ref ** 2 / self.t_ref
        s_e = self.P_ref   * self.u_ref / self.t_ref
        return ((R_m / s_m) ** 2).mean() + ((R_p / s_p) ** 2).mean() + ((R_e / s_e) ** 2).mean()


# ============================================================ L_IC,A


class DetonationICLoss(nn.Module):
    """L_IC,A — single-point CJ anchor at tau = 0 (t = 0, charge at rest).

    At t=0 the unreacted charge inside the surface is at the CJ state with
    zero particle velocity (u = 0 everywhere at the instant of release).
    ``Z_anchor`` locates the anchor point inside the charge region: physical
    configs place it near the origin (r_anchor_A ~ 1e-4 m); the scaled
    unified config places it at the charge surface (Z_R0).  Both are inside
    the t=0 CJ core, so the pinned state is the same.
    """

    def __init__(
        self,
        net: DetonationNet,
        rho_CJ: float, P_CJ: float,
        Z_anchor: float = 0.052712,
        u_ref: float = 2000.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.rho_CJ = float(rho_CJ)
        self.P_CJ   = float(P_CJ)
        self.Z_anchor = float(Z_anchor)
        self.u_ref = float(u_ref)
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        tau = torch.zeros(1, 1, device=self.device)
        Z = torch.full((1, 1), self.Z_anchor, device=self.device)
        rho_p, u_p, P_p = self.net(tau, Z)
        eps = 1e-10
        return (
            (torch.log(rho_p + eps) - torch.log(torch.tensor(self.rho_CJ, device=self.device) + eps)) ** 2
            + ((u_p - 0.0) / self.u_ref) ** 2
            + (torch.log(P_p + eps) - torch.log(torch.tensor(self.P_CJ, device=self.device) + eps)) ** 2
        ).mean()


# ============================================================ L_BC,A,slope


class DetonationSlopeBCLoss(nn.Module):
    """L_BC,A,slope — MSE along Z = Z_c(tau) vs. d3plot bilinear interp.

    Samples ``N`` tau in [0, tau_sep], evaluates Z_c(tau), queries the
    dataset at (tau, Z_c(tau)) for targets.  Rejects points where d3plot
    density < rho_min (air-domain guard).
    """

    def __init__(
        self,
        net: DetonationNet,
        dataset,
        tau_sep: float,
        tau_traj_rc: torch.Tensor,
        Z_traj_rc: torch.Tensor,
        N: int = 256,
        rho_min: float = 0.0,
        u_ref: float = 2000.0,
        w_rho: float = 1.0,
        w_u:   float = 0.5,
        w_P:   float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.tau_sep = float(tau_sep)
        self.register_buffer("tau_traj_rc", tau_traj_rc.to(device).float(), persistent=False)
        self.register_buffer("Z_traj_rc", Z_traj_rc.to(device).float(), persistent=False)
        self.N = int(N)
        self.rho_min = float(rho_min)
        self.u_ref = float(u_ref)
        self.w_rho = w_rho; self.w_u = w_u; self.w_P = w_P
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        ds = self.dataset
        target = self.N
        kept_tau, kept_Z, kept_rho, kept_u, kept_P = [], [], [], [], []
        attempts = 0
        max_attempts = 8
        while sum(t_.numel() for t_ in kept_tau) < target and attempts < max_attempts:
            tau = torch.rand(target, 1, device=self.device) * self.tau_sep
            Z = _interp_rc(tau, self.tau_traj_rc.to(tau.device), self.Z_traj_rc.to(tau.device))
            rho_t, u_t, P_t = ds.state_at(tau, Z)
            rho_t = rho_t.view(-1, 1)
            mask = (rho_t.squeeze(-1) >= self.rho_min)
            if mask.any():
                kept_tau.append(tau[mask]); kept_Z.append(Z[mask])
                kept_rho.append(rho_t[mask].view(-1, 1))
                kept_u.append(u_t[mask].view(-1, 1))
                kept_P.append(P_t[mask].view(-1, 1))
            attempts += 1
        if not kept_tau:
            return torch.tensor(0.0, device=self.device)

        tau   = torch.cat(kept_tau)[:target].view(-1, 1).to(self.device)
        Z     = torch.cat(kept_Z)[:target].view(-1, 1).to(self.device)
        rho_t = torch.cat(kept_rho)[:target].view(-1, 1).to(self.device)
        u_t   = torch.cat(kept_u)[:target].view(-1, 1).to(self.device)
        P_t   = torch.cat(kept_P)[:target].view(-1, 1).to(self.device)

        rho_p, u_p, P_p = self.net(tau, Z)
        eps = 1e-10
        log_rho_err = torch.log(rho_p + eps) - torch.log(rho_t + eps)
        log_P_err   = torch.log(P_p   + eps) - torch.log(P_t   + eps)
        u_err       = (u_p - u_t) / self.u_ref
        return (
            self.w_rho * (log_rho_err ** 2).mean()
            + self.w_u  * (u_err       ** 2).mean()
            + self.w_P  * (log_P_err   ** 2).mean()
        )
