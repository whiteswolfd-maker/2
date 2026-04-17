"""L_BC — Three boundary-condition losses (§2.3.③).

Sub-terms:
(a) Contact-face pressure: P_net(R_c(t), t) == P_prod(t)
(b) Contact-face kinematics: u_net(R_c(t), t) == dR_c/dt
(c) Farfield non-reflecting: ∂_t J+ + (u+c) ∂_r J+ + 2uc/r = 0
    at r = R_far, where J+ = u + 2c/(γ-1).

P_prod(t) in (a) is supplied either:
- By LSDYNALoader.contact_pressure_history()  (Stage-2 warmup window)
- By JWL online calculation: P_s(v_c(t))      (once ContactNet is trusted)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from pinn.operators import space_derivative, time_derivative


class BCLoss(nn.Module):
    """L_BC = β_a L_BC^(a) + β_b L_BC^(b) + β_c L_BC^(c).

    Parameters
    ----------
    main_net : MainNet
    contact_net : ContactNet
    gamma : float
    R_far : float
    N_c : int
        Number of contact-face time samples.
    N_inf : int
        Number of farfield samples.
    t_sep, t_end : float
    beta_a, beta_b, beta_c : float
        Sub-term weights (10 : 1 : 1 default).
    P_prod_fn : callable (t_tensor) -> P_tensor, optional
        If provided, used for (a) instead of the online JWL calculation.
    jwl_isentrope : JWLIsentrope-like, optional
        Used for online JWL calculation if P_prod_fn is None.
    R0 : float, optional
        Charge radius; needed for v_c = (R_c / R0)^3.
    device : str
    """

    def __init__(
        self,
        main_net: nn.Module,
        contact_net: nn.Module,
        gamma: float = 1.4,
        R_far: float = 1.0,
        N_c:   int = 500,
        N_inf: int = 200,
        t_sep: float = 0.0,
        t_end: float = 15e-3,
        beta_a: float = 10.0,
        beta_b: float = 1.0,
        beta_c: float = 1.0,
        P_prod_fn: Optional[Callable] = None,
        jwl_isentrope=None,
        R0: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.main_net    = main_net
        self.contact_net = contact_net
        self.gamma       = gamma
        self.R_far       = R_far
        self.N_c         = N_c
        self.N_inf       = N_inf
        self.t_sep       = t_sep
        self.t_end       = t_end
        self.beta_a      = beta_a
        self.beta_b      = beta_b
        self.beta_c      = beta_c
        self.P_prod_fn   = P_prod_fn   # None → use JWL online
        self.jwl_iso     = jwl_isentrope
        self.R0          = R0
        self.device      = torch.device(device)

    def _sample_contact_times(self) -> torch.Tensor:
        return torch.empty(self.N_c, 1, device=self.device).uniform_(
            self.t_sep, self.t_end
        ).requires_grad_(True)

    def _P_prod(self, R_c: torch.Tensor) -> torch.Tensor:
        """Evaluate detonation-product pressure at contact face.

        Uses P_prod_fn if set (LS-DYNA warmup mode), otherwise JWL isentrope.
        """
        if self.P_prod_fn is not None:
            # Use the pre-built interpolator from LS-DYNA time series
            return self.P_prod_fn(R_c)

        if self.jwl_iso is None:
            raise RuntimeError("BCLoss: provide either P_prod_fn or jwl_isentrope.")

        # v_c = (R_c / R0)^3 — homogeneous-expansion approximation
        v_c = (R_c / self.R0) ** 3
        # Query JWL via NumPy interpolation (detach to avoid graph through NumPy)
        v_c_np = v_c.detach().cpu().numpy().ravel()
        P_np   = self.jwl_iso.P_s(v_c_np)
        P_tensor = torch.tensor(
            P_np[:, None], dtype=R_c.dtype, device=R_c.device
        )
        return P_tensor

    # ---------------------------------------------------------------- (a)

    def _loss_a(self, t_c: torch.Tensor) -> torch.Tensor:
        """Contact-face pressure continuity."""
        R_c = self.contact_net(t_c)
        P_net_rc, _, _ = self.main_net(R_c, t_c)
        P_prod = self._P_prod(R_c)
        return ((P_net_rc - P_prod) ** 2).mean()

    # ---------------------------------------------------------------- (b)

    def _loss_b(self, t_c: torch.Tensor) -> torch.Tensor:
        """Contact-face kinematic condition: u_net(R_c, t) == dR_c/dt."""
        t_c_g = t_c.requires_grad_(True)
        R_c   = self.contact_net(t_c_g)
        dRc_dt = torch.autograd.grad(
            R_c.sum(), t_c_g,
            create_graph=True, retain_graph=True
        )[0]
        _, u_net_rc, _ = self.main_net(R_c.detach(), t_c_g)
        return ((u_net_rc - dRc_dt) ** 2).mean()

    # ---------------------------------------------------------------- (c)

    def _loss_c(self) -> torch.Tensor:
        """Farfield non-reflecting C+ characteristic condition.

        ∂_t J+ + (u+c) ∂_r J+ + 2uc/r = 0 at r = R_far.
        J+ = u + 2c/(γ-1).
        """
        t_inf = torch.empty(self.N_inf, 1, device=self.device).uniform_(
            self.t_sep, self.t_end
        )
        r_inf = torch.full_like(t_inf, self.R_far, requires_grad=True)
        t_inf = t_inf.requires_grad_(True)

        rho, u, P = self.main_net(r_inf, t_inf)
        c = torch.sqrt(self.gamma * P / rho)
        Jplus = u + 2.0 * c / (self.gamma - 1.0)

        dJdt = time_derivative(Jplus, t_inf)
        dJdr = space_derivative(Jplus, r_inf)

        residual = dJdt + (u + c) * dJdr + 2.0 * u * c / r_inf
        return (residual ** 2).mean()

    # ---------------------------------------------------------------- forward

    def forward(self) -> torch.Tensor:
        t_c = self._sample_contact_times()
        L_a = self._loss_a(t_c)
        L_b = self._loss_b(t_c)
        L_c = self._loss_c()
        return self.beta_a * L_a + self.beta_b * L_b + self.beta_c * L_c
