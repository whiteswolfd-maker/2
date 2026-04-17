"""Neural network architectures for the blast-wave PINN.

Three sub-networks:

MainNet
    SIREN (Sinusoidal Representation Network) mapping (r, t) -> (rho~, u~, P~).
    Activations: sine, depth 6, width 128, ω₀=30.
    Outputs are post-processed with softplus to guarantee ρ > 0, P > 0.

ContactNet
    Monotone MLP giving the contact-face radius R_c(t).
    R_c(t) = R_c_sep + ∫_{t_sep}^{t} softplus(f_c(τ)) dτ
    implemented via a learned velocity sub-net f_c and cumulative trapezoidal
    integration on a fine time grid.

ShockNet
    MLP giving R_s(t) = R_c(t) + softplus(g_s(t)), ensuring R_s > R_c > 0.

References
----------
Sitzmann et al. "Implicit Neural Representations with Periodic Activation
Functions." NeurIPS 2020.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ SIREN helpers

def _siren_init(layer: nn.Linear, is_first: bool, omega0: float) -> None:
    """SIREN weight initialisation (Sitzmann et al., Supplement S1)."""
    with torch.no_grad():
        fan_in = layer.weight.size(1)
        if is_first:
            bound = 1.0 / fan_in
        else:
            bound = math.sqrt(6.0 / fan_in) / omega0
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)


class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega0: float, is_first: bool) -> None:
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_dim, out_dim)
        _siren_init(self.linear, is_first=is_first, omega0=omega0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(x))


# ============================================================ MainNet

class MainNet(nn.Module):
    """SIREN mapping (r, t) → (rho, u, P) with positivity constraints.

    Parameters
    ----------
    depth : int, default 6
        Number of hidden SIREN layers.
    width : int, default 128
        Width of each hidden layer.
    omega0 : float, default 30.0
        SIREN frequency factor.
    rho_ref, u_ref, P_ref : float
        Reference scales for output normalisation.
    """

    def __init__(
        self,
        depth: int = 6,
        width: int = 128,
        omega0: float = 30.0,
        rho_ref: float = 1.225,
        u_ref: float = 340.0,
        P_ref: float = 101325.0,
    ) -> None:
        super().__init__()
        self.rho_ref = rho_ref
        self.u_ref   = u_ref
        self.P_ref   = P_ref

        layers: list[nn.Module] = [SirenLayer(2, width, omega0=omega0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SirenLayer(width, width, omega0=omega0, is_first=False))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, 3)   # raw: (rho_tilde, u_tilde, P_tilde)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho, u, P) tensors with shapes matching r and t.

        ``r`` and ``t`` must be broadcastable and should have
        ``requires_grad=True`` for PDE residual computation.
        """
        x = torch.cat([r, t], dim=-1)   # (..., 2)
        h = self.hidden(x)
        raw = self.out(h)                # (..., 3)

        rho = self.rho_ref * F.softplus(raw[..., 0:1])
        u   = self.u_ref   * raw[..., 1:2]          # velocity can be negative
        P   = self.P_ref   * F.softplus(raw[..., 2:3])
        return rho, u, P


# ============================================================ ContactNet

class ContactNet(nn.Module):
    """Monotone contact-face radius R_c(t).

    The underlying network f_c(t) outputs an instantaneous velocity.
    R_c is obtained by exact (initial-value) integration:

        R_c(t) = R_c_sep + ∫_{t_sep}^t softplus(f_c(τ)) dτ

    implemented with trapezoidal quadrature on ``n_quad`` uniform points.
    """

    def __init__(
        self,
        depth: int = 4,
        width: int = 64,
        R_c_sep: float = 0.1,
        t_sep: float = 0.0,
        n_quad: int = 64,
    ) -> None:
        super().__init__()
        self.R_c_sep = R_c_sep
        self.t_sep   = t_sep
        self.n_quad  = n_quad

        # Velocity sub-net f_c(t) -> scalar
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def velocity(self, t: torch.Tensor) -> torch.Tensor:
        """Contact-face velocity dR_c/dt = softplus(f_c(t))."""
        return F.softplus(self.net(t))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return R_c(t) for each time in t (shape: same as t)."""
        # For each t_query, integrate from t_sep to t_query
        # t has shape (B, 1)
        t_sep = torch.tensor([[self.t_sep]], dtype=t.dtype, device=t.device)
        # Quadrature grid: (B, n_quad) spanning [t_sep, t_query]
        alpha = torch.linspace(0.0, 1.0, self.n_quad, device=t.device)  # (n_quad,)
        # tau: (B, n_quad)
        tau = t_sep + (t - t_sep) * alpha.unsqueeze(0)  # broadcast
        # Evaluate velocity at each quadrature point
        B = t.shape[0]
        tau_flat = tau.reshape(-1, 1)    # (B*n_quad, 1)
        v_flat   = self.velocity(tau_flat)
        v_quad   = v_flat.reshape(B, self.n_quad)   # (B, n_quad)
        # Trapezoidal integration: dt = (t - t_sep) / (n_quad - 1) per interval
        dt = (t - t_sep) / (self.n_quad - 1)  # (B, 1)
        integral = dt * (0.5 * v_quad[:, 0:1] + v_quad[:, 1:-1].sum(dim=1, keepdim=True) + 0.5 * v_quad[:, -1:])
        return self.R_c_sep + integral


# ============================================================ ShockNet

class ShockNet(nn.Module):
    """Shock-front radius R_s(t) = R_c(t) + softplus(g_s(t)).

    Guarantees R_s > R_c at all times.

    Parameters
    ----------
    contact_net : ContactNet
        The ContactNet instance (shared parameters; NOT frozen inside ShockNet).
    gap_min : float
        Minimum gap enforced via softplus offset (m); default 1e-3 m = 1 mm.
    """

    def __init__(
        self,
        contact_net: ContactNet,
        depth: int = 4,
        width: int = 64,
        gap_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.contact_net = contact_net
        self.gap_min     = gap_min

        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def gap(self, t: torch.Tensor) -> torch.Tensor:
        """R_s - R_c = gap_min + softplus(g_s(t))."""
        return self.gap_min + F.softplus(self.net(t))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return R_s(t) (same shape as t)."""
        R_c = self.contact_net(t)
        return R_c + self.gap(t)

    def shock_speed(self, t: torch.Tensor) -> torch.Tensor:
        """D_s = dR_s/dt computed by autograd."""
        t = t.requires_grad_(True)
        R_s = self.forward(t)
        (D_s,) = torch.autograd.grad(
            R_s, t,
            grad_outputs=torch.ones_like(R_s),
            create_graph=True,
        )
        return D_s


# ============================================================ factory

def build_networks(cfg: dict, sep) -> tuple[MainNet, ContactNet, ShockNet]:
    """Construct the three sub-networks from a config dict and SeparationState.

    Parameters
    ----------
    cfg : dict
        Loaded from ``configs/tnt_spherical.yaml`` (``networks`` sub-dict).
    sep : SeparationState
        Provides ``R_c_sep``, ``t_sep`` for ContactNet initialisation.
    """
    mc = cfg["main"]
    cc = cfg["contact"]
    sc = cfg["shock"]

    main_net = MainNet(
        depth=mc["depth"],
        width=mc["width"],
        omega0=mc["omega0"],
    )
    contact_net = ContactNet(
        depth=cc["depth"],
        width=cc["width"],
        R_c_sep=sep.R_c,
        t_sep=sep.t_sep,
    )
    shock_net = ShockNet(
        contact_net=contact_net,
        depth=sc["depth"],
        width=sc["width"],
    )
    return main_net, contact_net, shock_net
