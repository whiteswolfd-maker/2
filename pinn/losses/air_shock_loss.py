"""Loss modules for the Phase-B AirShockNet.

Five terms (see plan §"Loss functions"):

* ``AirShockDataLoss``  L_data,B  — MSE vs. d3plot on the rectangle
* ``AirShockICLoss``    L_IC,B    — two-segment IC at t = t_sep
* ``AirShockPDELoss``   L_PDE,B   — spherical Euler residuals (ideal-gas EOS)
* ``AirShockRHLoss``    L_RH,B    — Rankine-Hugoniot at r = R_s(t) +/- eps
* ``AirShockOutflowLoss`` L_BC,B,outflow — Riemann invariant ∂R+/∂r ≈ 0
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data.d3plot_dataset import D3plotLineDataset
from pinn.networks import AirShockNet, DetonationNet
from pinn.operators import (
    ideal_gas_internal_energy,
    space_derivative,
    spherical_euler_residuals,
)


# ============================================================ L_data,B


class AirShockDataLoss(nn.Module):
    """L_data,B — d3plot supervision on the (t >= t_sep, r >= R_c) rectangle.

    Samples a fraction ``shock_frac`` of points from a band around R_s(t)
    so the network sees enough shock-front signal to avoid collapsing to
    the constant-field PDE attractor.  The remainder is sampled uniformly
    over the full Phase-B rectangle.
    """

    def __init__(
        self,
        net: AirShockNet,
        dataset: D3plotLineDataset,
        t_sep: float, R_c: float,
        N: int = 4096,
        t_data_min: float = 0.0,
        u_ref: float = 1000.0,
        w_rho: float = 1.0,
        w_u:   float = 1.0,
        w_P:   float = 1.0,
        max_resample: int = 8,
        shock_frac: float = 0.5,
        shock_band: float = 0.05,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.t_sep = float(t_sep); self.R_c = float(R_c)
        self.t_data_min = float(t_data_min)
        self.t_end = float(dataset.t_end)
        self.x_end = float(dataset.x_end)
        self.N = N; self.u_ref = u_ref
        self.w_rho = w_rho; self.w_u = w_u; self.w_P = w_P
        self.max_resample = max_resample
        self.shock_frac = float(shock_frac)
        self.shock_band = float(shock_band)
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        ds = self.dataset
        t_start = max(self.t_sep, self.t_data_min)
        n_total = self.N
        n_shock = int(n_total * self.shock_frac)
        n_unif  = n_total - n_shock

        # 1) Shock-band sampling — query data via state_at
        t_s = t_start + torch.rand(n_shock, 1, device=self.device) * (self.t_end - t_start)
        R_s = ds.R_s(t_s).view(-1, 1)
        lo = (R_s - self.shock_band).clamp(min=self.R_c, max=self.x_end)
        hi = (R_s + self.shock_band).clamp(min=self.R_c, max=self.x_end)
        r_s = lo + torch.rand(n_shock, 1, device=self.device) * (hi - lo).clamp(min=1e-9)
        rho_s, u_s, P_s = ds.state_at(t_s, r_s)

        # 2) Uniform sampling — all radii cover the same scaled domain
        t_u = t_start + torch.rand(n_unif, 1, device=self.device) * (self.t_end - t_start)
        r_u = self.R_c + torch.rand(n_unif, 1, device=self.device) * (self.x_end - self.R_c)
        rho_u, u_u, P_u = ds.state_at(t_u, r_u)

        t = torch.cat([t_s, t_u], dim=0)
        r = torch.cat([r_s, r_u], dim=0)
        rho_t = torch.cat([rho_s, rho_u], dim=0)
        u_t   = torch.cat([u_s,   u_u],   dim=0)
        P_t   = torch.cat([P_s,   P_u],   dim=0)

        rho_p, u_p, P_p = self.net(t, r)
        eps = 1e-10
        log_rho_err = torch.log(rho_p + eps) - torch.log(rho_t + eps)
        log_P_err   = torch.log(P_p   + eps) - torch.log(P_t   + eps)
        u_err = (u_p - u_t) / self.u_ref
        return (
            self.w_rho * (log_rho_err ** 2).mean()
            + self.w_u  * (u_err       ** 2).mean()
            + self.w_P  * (log_P_err   ** 2).mean()
        )


# ============================================================ L_IC,B


class AirShockICLoss(nn.Module):
    """L_IC,B — two-segment initial condition at t = t_sep.

    At t = t_sep the products-air contact is at r = R_c and the leading air
    shock is co-located there (R_s(t_sep) = R_c, no intermediate post-shock
    region).  The IC therefore has just two pieces:

    * r = R_c (single point):  match the frozen DetonationNet output
      (rho_x_pred, u_x_pred, P_x_pred).
    * r in (R_c, x_end]:        ambient (rho_a, 0, P_a).

    The contact-point sample is repeated ``contact_repeat`` times so the
    point-mass term receives an effective ``contact_repeat`` x weight (or
    use ``w_contact`` in the trainer's loss-weight dict, redundant either
    way).
    """

    def __init__(
        self,
        net: AirShockNet,
        frozen_detonation: DetonationNet,
        t_sep: float, R_c: float, x_end: float,
        rho_a: float, P_a: float,
        N_amb: int = 256,
        contact_repeat: int = 16,
        u_ref: float = 1000.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.frozen = frozen_detonation
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        self.t_sep = float(t_sep); self.R_c = float(R_c); self.x_end = float(x_end)
        self.rho_a = float(rho_a); self.P_a = float(P_a)
        self.N_amb = int(N_amb)
        self.contact_repeat = int(contact_repeat)
        self.u_ref = float(u_ref)
        self.device = torch.device(device)

        # Cache the connection-point target (frozen network is constant)
        with torch.no_grad():
            t_p = torch.tensor([[self.t_sep]], device=self.device, dtype=torch.float32)
            r_p = torch.tensor([[self.R_c]],  device=self.device, dtype=torch.float32)
            rho_x, u_x, P_x = self.frozen(t_p, r_p)
            self.register_buffer("rho_x_pred", rho_x.detach().clone(), persistent=False)
            self.register_buffer("u_x_pred",   u_x.detach().clone(),   persistent=False)
            self.register_buffer("P_x_pred",   P_x.detach().clone(),   persistent=False)

    def forward(self) -> torch.Tensor:
        # ---- contact-point single-point IC, replicated for emphasis
        t_p = torch.full((self.contact_repeat, 1), self.t_sep, device=self.device)
        r_p = torch.full((self.contact_repeat, 1), self.R_c,   device=self.device)
        rho_p, u_p, P_p = self.net(t_p, r_p)
        eps = 1e-10
        loss_contact = (
            (torch.log(rho_p + eps) - torch.log(self.rho_x_pred + eps)) ** 2
            + ((u_p - self.u_x_pred) / self.u_ref) ** 2
            + (torch.log(P_p + eps) - torch.log(self.P_x_pred + eps)) ** 2
        ).sum()

        # ---- ambient IC for r > R_c (avoid duplicating the contact point)
        gap = max(self.x_end - self.R_c, 1e-9)
        r_amb = self.R_c + 1e-6 * gap + torch.rand(self.N_amb, 1, device=self.device) * gap * (1.0 - 1e-6)
        t_amb = torch.full_like(r_amb, self.t_sep)
        rho_a, u_a, P_a = self.net(t_amb, r_amb)
        rho_t = torch.full_like(rho_a, self.rho_a)
        P_t   = torch.full_like(P_a,   self.P_a)
        loss_amb = (
            (torch.log(rho_a + eps) - torch.log(rho_t + eps)) ** 2
            + (u_a / self.u_ref) ** 2
            + (torch.log(P_a + eps) - torch.log(P_t + eps)) ** 2
        ).sum()

        n_total = self.contact_repeat + self.N_amb
        return (loss_contact + loss_amb) / n_total


# ============================================================ L_PDE,B


class AirShockPDELoss(nn.Module):
    """L_PDE,B — spherical Euler residuals (ideal-gas EOS) on the rectangle.

    Collocation points within ``shock_mask`` of R_s(t) are discarded —
    the strong-form PDE is ill-defined at a discontinuity.  The shock
    jump is enforced through data and Rankine-Hugoniot losses instead.
    """

    def __init__(
        self,
        net: AirShockNet,
        t_sep: float, t_end: float,
        R_c: float, x_end: float,
        dataset: D3plotLineDataset | None = None,
        gamma: float = 1.4,
        N_f: int = 8192,
        rho_ref: float = 1.225,
        u_ref:   float = 1000.0,
        P_ref:   float = 1.0e6,
        t_ref:   float = 1.0e-3,
        t_margin: float = 0.0,
        shock_mask: float = 0.02,
        device:  str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.t_sep = float(t_sep); self.t_end = float(t_end)
        self.R_c = float(R_c); self.x_end = float(x_end)
        self.gamma = float(gamma)
        self.N_f = int(N_f)
        self.t_margin = float(t_margin)
        self.shock_mask = float(shock_mask)
        self.rho_ref = float(rho_ref); self.u_ref = float(u_ref)
        self.P_ref   = float(P_ref);   self.t_ref = float(t_ref)
        self.dataset = dataset
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        t_start = self.t_sep + self.t_margin
        if t_start >= self.t_end:
            return torch.tensor(0.0, device=self.device)
        t = t_start + torch.rand(self.N_f, 1, device=self.device) * (self.t_end - t_start)
        r = self.R_c  + torch.rand(self.N_f, 1, device=self.device) * (self.x_end - self.R_c)

        # Mask points inside the shock band (strong-form PDE invalid at shock)
        if self.dataset is not None and self.shock_mask > 0:
            R_s = self.dataset.R_s(t).view(-1, 1)
            keep = ((r - R_s).abs() > self.shock_mask).squeeze(-1)
            if keep.any():
                t = t[keep]; r = r[keep]
            else:
                return torch.tensor(0.0, device=self.device)

        t = t.requires_grad_(True); r = r.requires_grad_(True)

        rho, u, P = self.net(t, r)
        e = ideal_gas_internal_energy(rho, P, self.gamma)
        R_m, R_p, R_e = spherical_euler_residuals(rho, u, P, e, r, t)
        s_m = self.rho_ref * self.u_ref / self.t_ref
        s_p = self.rho_ref * self.u_ref ** 2 / self.t_ref
        s_e = self.P_ref   * self.u_ref / self.t_ref
        return ((R_m / s_m) ** 2).mean() + ((R_p / s_p) ** 2).mean() + ((R_e / s_e) ** 2).mean()


# ============================================================ L_RH,B


class AirShockRHLoss(nn.Module):
    """L_RH,B — interior Rankine-Hugoniot enforcement at the shock front.

    At each sampled time ``t in [t_sep + t_margin, t_end]`` we read R_s(t)
    and D_s(t) = dR_s/dt (from the dataset's 4th-order central-diff buffer
    if available, else linear gradient on the trajectory).  The post-shock
    state is the analytical RH jump from ambient with that D_s.  The
    AirShockNet output at ``r = R_s(t) - eps`` is matched to the post-shock
    state.

    The temporal margin excludes the IC corner singularity at (t_sep, R_c)
    where the shock is still co-located with the contact surface.
    """

    def __init__(
        self,
        net: AirShockNet,
        dataset: D3plotLineDataset,
        t_sep: float, t_end: float,
        gamma: float, rho_a: float, P_a: float,
        eps: float = 1e-3,
        N_s: int = 256,
        u_ref: float = 1000.0,
        t_margin: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.t_sep = float(t_sep); self.t_end = float(t_end)
        self.gamma = float(gamma); self.rho_a = float(rho_a); self.P_a = float(P_a)
        self.eps = float(eps); self.N_s = int(N_s)
        self.u_ref = float(u_ref)
        self.t_margin = float(t_margin)
        self.device = torch.device(device)

        # Use precomputed D_s buffer from the dataset (np.gradient on
        # SG-smoothed R_s trajectory).
        self.register_buffer(
            "_dRs_dt_traj",
            dataset._dRs_dt_traj.clone().detach().to(device),
            persistent=False,
        )

    def _D_s(self, t: torch.Tensor) -> torch.Tensor:
        """Linear-interp dR_s/dt at query times t (returns shape of t)."""
        from data.d3plot_dataset import _interp1d_torch
        return _interp1d_torch(t, self.dataset.t_traj.to(t.device),
                               self._dRs_dt_traj.to(t.device))

    def forward(self) -> torch.Tensor:
        t_start = self.t_sep + self.t_margin
        if t_start >= self.t_end:
            return torch.tensor(0.0, device=self.device)
        t = t_start + torch.rand(self.N_s, 1, device=self.device) * (self.t_end - t_start)
        R_s = self.dataset.R_s(t).view(-1, 1)
        D_s = self._D_s(t).view(-1, 1)

        # Filter to genuinely supersonic frames (M > 2.0).  Below M ≈ 2 the
        # shock is weak and the RH relation is increasingly sensitive to small
        # D_s errors; leaving those frames to data + PDE avoids polluting the
        # RH loss with unreliable targets.
        gamma = self.gamma; rho_a = self.rho_a; P_a = self.P_a
        c_a = (gamma * P_a / rho_a) ** 0.5
        M_all = D_s.abs() / c_a
        supersonic = (M_all.squeeze(-1) > 2.0)
        if not supersonic.any():
            return torch.tensor(0.0, device=self.device)
        t = t[supersonic]; R_s = R_s[supersonic]; D_s = D_s[supersonic]

        # Analytical RH post-shock state for ideal gas (lab frame, ahead state ambient)
        # rho_post / rho_a = (gamma+1) M^2 / [(gamma-1) M^2 + 2]
        # with M = D_s / c_a
        M = D_s.abs() / c_a
        rho_post = rho_a * ((gamma + 1.0) * M ** 2) / ((gamma - 1.0) * M ** 2 + 2.0)
        P_post   = P_a   * (2.0 * gamma * M ** 2 - (gamma - 1.0)) / (gamma + 1.0)
        u_post   = D_s - D_s * rho_a / rho_post   # mass-flux relation

        r_eval = R_s - self.eps   # post-shock side (inside the shock)
        rho_p, u_p, P_p = self.net(t, r_eval)
        e_floor = 1e-10
        log_rho_err = torch.log(rho_p + e_floor) - torch.log(rho_post + e_floor)
        log_P_err   = torch.log(P_p   + e_floor) - torch.log(P_post   + e_floor)
        u_err = (u_p - u_post) / self.u_ref
        return (log_rho_err ** 2).mean() + (u_err ** 2).mean() + (log_P_err ** 2).mean()


# ============================================================ L_BC,B,outflow


class AirShockOutflowLoss(nn.Module):
    """L_BC,B,outflow — Riemann invariant outflow at r = x_end.

    Imposes ``∂R+/∂r ≈ 0`` at the right boundary, where
    ``R+ = u + 2 c / (gamma - 1)`` is the outgoing Riemann invariant of the
    1D radial Euler system.  Sound speed ``c = sqrt(gamma P / rho)``.
    """

    def __init__(
        self,
        net: AirShockNet,
        t_sep: float, t_end: float, x_end: float,
        gamma: float = 1.4,
        N: int = 128,
        scale: float = 1.0,
        rho_a: float = 1.225,
        P_a:   float = 101325.0,
        u_ref: float = 1000.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.net = net
        self.t_sep = float(t_sep); self.t_end = float(t_end)
        self.x_end = float(x_end)
        self.gamma = float(gamma)
        self.N = int(N)
        # Dimensional Riemann-invariant scale R+_ref = u_ref + 2 c_a / (gamma-1).
        # The natural normalization for d(R+)/dr is R+_ref / x_end (1/s).
        # ``scale`` from yaml multiplies that natural scale (1.0 is the
        # physics default; bump it to soft-pedal the loss).
        c_a = (gamma * P_a / rho_a) ** 0.5
        R_plus_ref = float(u_ref + 2.0 * c_a / (gamma - 1.0))
        self.dRp_dr_scale = max(scale, 1e-30) * (R_plus_ref / max(self.x_end, 1e-30))
        self.device = torch.device(device)

    def forward(self) -> torch.Tensor:
        t = self.t_sep + torch.rand(self.N, 1, device=self.device) * (self.t_end - self.t_sep)
        r = torch.full((self.N, 1), self.x_end, device=self.device, requires_grad=True)
        rho, u, P = self.net(t, r)
        # Floor rho to avoid 1/sqrt(near-zero) blowing up the sound speed
        # before the network has converged near ambient.  1e-3 kg/m^3 is
        # well below ambient air (1.225) but large enough to keep grads sane.
        c = torch.sqrt(self.gamma * P / rho.clamp(min=1e-3))
        R_plus = u + 2.0 * c / (self.gamma - 1.0)
        dRp_dr = space_derivative(R_plus, r)
        return ((dRp_dr / self.dRp_dr_scale) ** 2).mean()
