"""Autograd differential operators for spherical-coordinate Euler equations.

All operators assume inputs are PyTorch tensors with ``requires_grad=True``
where differentiation is needed.  Use ``create_graph=True`` consistently so
that second-order derivatives (needed by the artificial-viscosity energy term)
propagate correctly through the computational graph.
"""

from __future__ import annotations

import torch


def grad(
    f: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """∂f/∂x via autograd.  f and x must be graph-connected.

    Parameters
    ----------
    f : Tensor of shape (...,) or (..., 1)
    x : Tensor same shape, with requires_grad=True
    create_graph : bool
        Must be True when second derivatives are needed later.
    """
    if f.shape != x.shape:
        if f.shape == x.shape[:-1] + (1,):
            f = f.squeeze(-1)
        elif x.shape == f.shape[:-1] + (1,):
            x = x.squeeze(-1)
    (df_dx,) = torch.autograd.grad(
        f, x,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=True,
    )
    return df_dx


def grad_batched(
    f: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Batched version: f has shape (N, 1) or (N,), x has shape (N, 1)."""
    (df,) = torch.autograd.grad(
        f.sum(), x,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )
    if df is None:
        df = torch.zeros_like(x)
    return df


def spherical_divergence(
    F: torch.Tensor, r: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """∂_r F + 2·F/r  (spherical-coordinate radial divergence).

    Adds the 2/r geometric source term to the standard ∂_r F.
    ``F`` and ``r`` must be connected through the computational graph.
    """
    dF_dr = grad_batched(F, r, create_graph=create_graph)
    return dF_dr + 2.0 * F / r


def time_derivative(f: torch.Tensor, t: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """∂f/∂t."""
    return grad_batched(f, t, create_graph=create_graph)


def space_derivative(f: torch.Tensor, r: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """∂f/∂r."""
    return grad_batched(f, r, create_graph=create_graph)


def second_space_derivative(
    f: torch.Tensor, r: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """∂²f/∂r²  via two autograd passes.

    Uses a cached first derivative to avoid recomputing the graph.
    The caller must NOT detach ``df_dr`` between the two calls.
    """
    df_dr = grad_batched(f, r, create_graph=True)
    d2f_dr2 = grad_batched(df_dr, r, create_graph=create_graph)
    return d2f_dr2


# ---------------------------------------------------------------- Euler residuals

def euler_residuals(
    rho: torch.Tensor,
    u:   torch.Tensor,
    P:   torch.Tensor,
    r:   torch.Tensor,
    t:   torch.Tensor,
    gamma: float = 1.4,
    mu_av: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spherical-coordinate Euler PDE residuals (primitive form).

    All inputs have shape (N, 1) with r.requires_grad=True, t.requires_grad=True.

    The governing equations are (§2.2):

        ∂_t ρ + ∂_r(ρu) + 2ρu/r = 0                         (mass)
        ∂_t(ρu) + ∂_r(ρu²+P) + 2ρu²/r = ∂_r(μ_AV ∂_r u)   (momentum)
        ∂_t E + ∂_r[(E+P)u] + 2(E+P)u/r = ∂_r(μ_AV u ∂_r u) (energy)

    Returns
    -------
    R_mass, R_mom, R_energy : Tensor (N, 1)
        PDE residuals; zero for an exact solution.
    """
    E = P / (gamma - 1.0) + 0.5 * rho * u ** 2

    # ---- mass
    rho_u = rho * u
    drho_dt  = time_derivative(rho, t)
    drhou_dr = space_derivative(rho_u, r)
    R_mass = drho_dt + drhou_dr + 2.0 * rho_u / r

    # ---- momentum
    rho_u2_P = rho * u ** 2 + P
    d_rhout_dt  = time_derivative(rho * u, t)
    d_rhou2P_dr = space_derivative(rho_u2_P, r)

    if isinstance(mu_av, torch.Tensor) and mu_av.abs().max() > 0:
        du_dr = space_derivative(u, r)
        av_mom = space_derivative(mu_av * du_dr, r)
    else:
        av_mom = torch.zeros_like(P)

    R_mom = d_rhout_dt + d_rhou2P_dr + 2.0 * rho * u ** 2 / r - av_mom

    # ---- energy
    Ep_u = (E + P) * u
    dE_dt    = time_derivative(E, t)
    dEpu_dr  = space_derivative(Ep_u, r)

    if isinstance(mu_av, torch.Tensor) and mu_av.abs().max() > 0:
        av_energy = space_derivative(mu_av * u * du_dr, r)
    else:
        av_energy = torch.zeros_like(P)

    R_energy = dE_dt + dEpu_dr + 2.0 * Ep_u / r - av_energy

    return R_mass, R_mom, R_energy


def artificial_viscosity(
    rho: torch.Tensor,
    u:   torch.Tensor,
    P:   torch.Tensor,
    r:   torch.Tensor,
    gamma: float = 1.4,
    c1: float = 2.0,
    c2: float = 0.5,
    ell: float = 1e-3,
) -> torch.Tensor:
    """Compute μ_AV = ℓ² ρ [c1 |∂_r u| 𝟙(∂_r u < 0) + c2·c].

    c = sqrt(γ P / ρ) is the local sound speed.
    The compression indicator 𝟙(∂_r u < 0) is approximated by a smooth
    sigmoid to keep the graph differentiable.
    """
    du_dr = space_derivative(u, r, create_graph=True)
    c_sound = torch.sqrt(gamma * P / rho)

    # Smooth compression indicator: sigmoid(-100 * du_dr)
    compress = torch.sigmoid(-100.0 * du_dr)
    mu = ell ** 2 * rho * (c1 * du_dr.abs() * compress + c2 * c_sound)
    return mu
