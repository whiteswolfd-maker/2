"""L_PDE — Spherical-Euler PDE residual loss with AV and RAR.

Computes residuals for the three conservation equations (§2.2) at
interior collocation points.  Includes:

- Shock-sensor weight σ(r, t) = 1 / (1 + α (∂P/P)^2) to downweight
  residuals in the sharp-gradient region.
- Artificial-viscosity (AV) coefficient that decays linearly from
  Stage 1 → Stage 3.
- Residual-Adaptive Refinement (RAR): every ``rar_interval`` optimiser
  steps, the highest-residual ``rar_fraction`` of points are replaced by
  new samples drawn from a pool concentrated near high-residual regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pinn.operators import euler_residuals, artificial_viscosity


class PDELoss(nn.Module):
    """L_PDE with shock sensor, AV annealing and RAR.

    Parameters
    ----------
    main_net : MainNet
    gamma : float
        Specific-heat ratio.
    t_sep, t_end : float
        Training time domain boundaries (s).
    R_c_fn : callable (t_tensor) -> r_tensor
        Returns contact-face radius at given times; used to exclude
        r < R_c from interior collocation.
    R_far : float
        Outer boundary radius (m).
    N_f : int
        Interior collocation count.
    alpha_sensor : float
        Shock-sensor sharpness (§2.3.②, default 1e-4).
    ell : float
        AV length scale (m).
    c1_start, c1_end : float
    c2_start, c2_end : float
        AV coefficient linear-annealing bounds.
    rar_interval : int
        Steps between RAR resamples.
    rar_fraction : float
        Fraction of points replaced in each RAR step.
    rar_pool_factor : int
        Candidate-pool size = factor × resample count.
    device : str
    """

    def __init__(
        self,
        main_net: nn.Module,
        gamma: float = 1.4,
        t_sep: float = 0.0,
        t_end: float = 15e-3,
        R_c_fn=None,
        R_c_min: float = 0.0,
        R_far: float = 1.0,
        N_f: int = 20_000,
        alpha_sensor: float = 1e-4,
        ell: float = 1e-3,
        c1_start: float = 2.0,
        c1_end: float = 0.1,
        c2_start: float = 0.5,
        c2_end: float = 0.05,
        rar_interval: int = 500,
        rar_fraction: float = 0.20,
        rar_pool_factor: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.main_net       = main_net
        self.gamma          = gamma
        self.t_sep          = t_sep
        self.t_end          = t_end
        self.R_c_fn         = R_c_fn
        self.R_c_min        = R_c_min
        self.R_far          = R_far
        self.N_f            = N_f
        self.alpha_sensor   = alpha_sensor
        self.ell            = ell
        self.c1_start       = c1_start
        self.c1_end         = c1_end
        self.c2_start       = c2_start
        self.c2_end         = c2_end
        self.rar_interval   = rar_interval
        self.rar_fraction   = rar_fraction
        self.rar_pool_factor = rar_pool_factor
        self.device         = torch.device(device)

        self._step  = 0
        self._c1    = c1_start
        self._c2    = c2_start
        self._points: torch.Tensor = self._sample_points(N_f)

    # ---------------------------------------------------------------- sampling

    def _sample_points(self, n: int) -> torch.Tensor:
        """Sample (r, t) uniformly in [R_c_min, R_far] × [t_sep, t_end]."""
        r = torch.empty(n, 1, device=self.device).uniform_(self.R_c_min, self.R_far)
        t = torch.empty(n, 1, device=self.device).uniform_(self.t_sep, self.t_end)
        return torch.cat([r, t], dim=1)   # (N, 2)

    def _rar_resample(self, residuals_sq: torch.Tensor) -> None:
        """Replace the highest-residual fraction with new high-error samples."""
        n_replace = int(self.N_f * self.rar_fraction)
        pool_n    = n_replace * self.rar_pool_factor
        pool      = self._sample_points(pool_n)

        # Evaluate residuals on pool (need grad for autograd inside euler_residuals)
        rp = pool[:, 0:1].requires_grad_(True)
        tp = pool[:, 1:2].requires_grad_(True)
        rho_p, u_p, P_p = self.main_net(rp, tp)
        rm, rmom, ren = euler_residuals(rho_p, u_p, P_p, rp, tp,
                                        gamma=self.gamma, mu_av=0.0)
        pool_res = (rm ** 2 + rmom ** 2 + ren ** 2).detach()

        # Pick top n_replace
        _, top_idx = torch.topk(pool_res.squeeze(), n_replace)
        new_pts = pool[top_idx]

        # Replace lowest-residual points in current set
        _, bot_idx = torch.topk(residuals_sq.squeeze(), n_replace, largest=False)
        self._points[bot_idx] = new_pts.detach()

    # ---------------------------------------------------------------- AV annealing

    def anneal(self, fraction: float) -> None:
        """Update AV coefficients.  ``fraction`` in [0, 1] is training progress."""
        fraction = max(0.0, min(1.0, fraction))
        self._c1 = self.c1_start + fraction * (self.c1_end - self.c1_start)
        self._c2 = self.c2_start + fraction * (self.c2_end - self.c2_start)

    # ---------------------------------------------------------------- forward

    def forward(self) -> torch.Tensor:
        pts = self._points
        r = pts[:, 0:1].clone().requires_grad_(True)
        t = pts[:, 1:2].clone().requires_grad_(True)

        rho, u, P = self.main_net(r, t)

        # Artificial viscosity
        mu_av = artificial_viscosity(
            rho, u, P, r,
            gamma=self.gamma,
            c1=self._c1,
            c2=self._c2,
            ell=self.ell,
        )

        R_mass, R_mom, R_energy = euler_residuals(rho, u, P, r, t,
                                                   gamma=self.gamma, mu_av=mu_av)

        # Shock sensor: σ = 1 / (1 + α (∂_r P / P)^2)
        dP_dr = torch.autograd.grad(
            P.sum(), r, create_graph=False, retain_graph=True
        )[0].detach()
        sensor = 1.0 / (1.0 + self.alpha_sensor * (dP_dr / (P.detach() + 1e-10)) ** 2)

        res_sq = sensor * (R_mass ** 2 + R_mom ** 2 + R_energy ** 2)

        self._step += 1
        if self._step % self.rar_interval == 0:
            self._rar_resample(res_sq.detach())

        return res_sq.mean()
