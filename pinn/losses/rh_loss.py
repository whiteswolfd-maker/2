"""L_RH — Rankine-Hugoniot jump-condition loss (§2.3.④).

Enforces all three conservation-law jumps across the shock surface r = R_s(t):

    r1 = ρ⁻(D_s − u⁻) − ρ_a D_s                              (mass)
    r2 = ρ⁻(D_s − u⁻)² + P⁻ − ρ_a D_s² − P_a               (momentum)
    r3 = h⁻ + ½(D_s−u⁻)² − h_a − ½ D_s²                     (energy)

where (ρ⁻, u⁻, P⁻) is the post-shock state from MainNet evaluated just
behind R_s(t), and (ρ_a, 0, P_a) is the known undisturbed air.
D_s = dR_s/dt is obtained via autograd through ShockNet.

Each residual is normalised by a characteristic pressure scale so that all
three contribute comparably.
"""

import math

from __future__ import annotations

import torch
import torch.nn as nn


class RHLoss(nn.Module):
    """L_RH = (1/N_s) Σ_k [r1² + r2² + r3²]  (dimension-normalised).

    Parameters
    ----------
    main_net : MainNet
    shock_net : ShockNet
    gamma : float
    rho_a, P_a : float
        Undisturbed ambient air state.
    N_s : int
        Number of shock-surface sample times.
    t_sep, t_end : float
    rho_scale, P_scale : float, optional
        Normalisation denominators; defaults to rho_a and P_a.
    device : str
    """

    def __init__(
        self,
        main_net:  nn.Module,
        shock_net: nn.Module,
        gamma: float = 1.4,
        rho_a: float = 1.225,
        P_a:   float = 101325.0,
        N_s:   int = 500,
        t_sep: float = 0.0,
        t_end: float = 15e-3,
        rho_scale: float | None = None,
        P_scale:   float | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.main_net  = main_net
        self.shock_net = shock_net
        self.gamma     = gamma
        self.rho_a     = rho_a
        self.P_a       = P_a
        self.N_s       = N_s
        self.t_sep     = t_sep
        self.t_end     = t_end
        self.rho_scale = rho_scale if rho_scale is not None else rho_a
        self.P_scale   = P_scale   if P_scale   is not None else P_a
        self.device    = torch.device(device)

    def forward(self) -> torch.Tensor:
        # Sample N_s times in [t_sep, t_end]
        t_s = torch.empty(self.N_s, 1, device=self.device).uniform_(
            self.t_sep, self.t_end
        ).requires_grad_(True)

        # Shock-front radius and speed via autograd through ShockNet
        R_s = self.shock_net(t_s)
        D_s = torch.autograd.grad(
            R_s.sum(), t_s,
            create_graph=True, retain_graph=True
        )[0]

        # Evaluate MainNet just behind the shock (r = R_s, t = t_s)
        rho_m, u_m, P_m = self.main_net(R_s, t_s)

        # Ambient state
        rho_a = self.rho_a
        P_a   = self.P_a
        g     = self.gamma

        # Specific enthalpy
        h_m = g * P_m / ((g - 1.0) * rho_m)
        h_a = g * P_a / ((g - 1.0) * rho_a)

        # Jump residuals
        r1 = rho_m * (D_s - u_m) - rho_a * D_s
        r2 = rho_m * (D_s - u_m) ** 2 + P_m - rho_a * D_s ** 2 - P_a
        r3 = h_m + 0.5 * (D_s - u_m) ** 2 - h_a - 0.5 * D_s ** 2

        # Normalise to make residuals dimensionless
        # r1: [kg/(m²·s)]  scale ~ rho_a * D_s ~ rho_a * sqrt(γ P_a / rho_a) ~ sqrt(rho_a * P_a)
        rho_D_scale = self.rho_scale * math.sqrt(g * P_a / rho_a)
        # r2: [Pa]
        # r3: [J/kg] = [m²/s²]
        r1_n = r1 / (rho_D_scale + 1e-10)
        r2_n = r2 / (self.P_scale + 1e-10)
        r3_n = r3 / (g * self.P_scale / (self.rho_scale + 1e-10) + 1e-10)

        return (r1_n ** 2 + r2_n ** 2 + r3_n ** 2).mean()
