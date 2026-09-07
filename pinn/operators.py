"""Autograd differential operators for the 1D spherical Euler PINN.

The production training loop uses :func:`spherical_euler_residuals` with
the ``+ 2 q / r`` geometric source on each conservation law and an EOS
supplied via ``e`` (JWL-isentropic for DetonationNet, ideal gas for
AirShockNet).

:func:`euler_residuals` is the planar (Cartesian) analogue; it is kept
**only as a method-of-manufactured-solutions reference** in the operator
unit tests (``tests/test_operators.py``).  No production loss imports it.
"""

from __future__ import annotations

import torch


def grad(
    f: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """∂f/∂x via autograd.  f and x must be graph-connected."""
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
    """Batched gradient: f shape (N, 1) or (N,), x shape (N, 1)."""
    (df,) = torch.autograd.grad(
        f.sum(), x,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )
    if df is None:
        df = torch.zeros_like(x)
    return df


def time_derivative(f: torch.Tensor, t: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """∂f/∂t."""
    return grad_batched(f, t, create_graph=create_graph)


def space_derivative(f: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """∂f/∂x."""
    return grad_batched(f, x, create_graph=create_graph)


def second_space_derivative(
    f: torch.Tensor, x: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """∂²f/∂x²  via two autograd passes."""
    df_dx = grad_batched(f, x, create_graph=True)
    return grad_batched(df_dx, x, create_graph=create_graph)


# ---------------------------------------------------------------- Euler residuals

def euler_residuals(
    rho: torch.Tensor,
    u:   torch.Tensor,
    P:   torch.Tensor,
    x:   torch.Tensor,
    t:   torch.Tensor,
    gamma: float = 1.4,
    mu_av: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1D planar Cartesian Euler PDE residuals (primitive form).

    All inputs have shape (N, 1) with x.requires_grad=True, t.requires_grad=True.

    Governing equations (no geometric source):

        ∂_t ρ      + ∂_x(ρu)         = 0                         (mass)
        ∂_t(ρu)    + ∂_x(ρu²+P)      = ∂_x(μ_AV ∂_x u)           (momentum)
        ∂_t E      + ∂_x[(E+P)u]     = ∂_x(μ_AV u ∂_x u)         (energy)

    Returns
    -------
    R_mass, R_mom, R_energy : Tensor (N, 1)
        PDE residuals; zero for an exact planar inviscid solution.
    """
    E = P / (gamma - 1.0) + 0.5 * rho * u ** 2

    # ---- mass
    rho_u = rho * u
    drho_dt  = time_derivative(rho, t)
    drhou_dx = space_derivative(rho_u, x)
    R_mass = drho_dt + drhou_dx

    # ---- momentum
    rho_u2_P = rho * u ** 2 + P
    d_rhou_dt   = time_derivative(rho_u, t)
    d_rhou2P_dx = space_derivative(rho_u2_P, x)

    if isinstance(mu_av, torch.Tensor) and mu_av.abs().max() > 0:
        du_dx = space_derivative(u, x)
        av_mom = space_derivative(mu_av * du_dx, x)
    else:
        du_dx = None
        av_mom = torch.zeros_like(P)

    R_mom = d_rhou_dt + d_rhou2P_dx - av_mom

    # ---- energy
    Ep_u = (E + P) * u
    dE_dt   = time_derivative(E, t)
    dEpu_dx = space_derivative(Ep_u, x)

    if isinstance(mu_av, torch.Tensor) and mu_av.abs().max() > 0:
        if du_dx is None:
            du_dx = space_derivative(u, x)
        av_energy = space_derivative(mu_av * u * du_dx, x)
    else:
        av_energy = torch.zeros_like(P)

    R_energy = dE_dt + dEpu_dx - av_energy

    return R_mass, R_mom, R_energy


def spherical_euler_residuals(
    rho: torch.Tensor,
    u:   torch.Tensor,
    P:   torch.Tensor,
    e:   torch.Tensor,
    r:   torch.Tensor,
    t:   torch.Tensor,
    mu_av: torch.Tensor | None = None,
    dmu_dr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1D spherical Euler PDE residuals (primitive form, EOS-agnostic).

    With ``E = rho * (e + 0.5 u^2)`` the conservation laws read

        ∂_t ρ        + ∂_r(ρu)            + 2 ρ u / r            = 0
        ∂_t(ρu)      + ∂_r(ρu² + P)       + 2 ρ u² / r           = 0
        ∂_t E        + ∂_r[(E+P) u]       + 2 (E+P) u / r        = 0

    When ``mu_av`` (scalar or per-point tensor) is provided and > 0, the
    momentum and energy equations acquire a local artificial-viscosity
    diffusion term that regularises the shock into a steep but smooth
    transition::

        R_mom    +=  − ∂_r(μ_av · ∂_r u)
        R_energy +=  − ∂_r(μ_av · u · ∂_r u)

    The mass equation is unchanged (viscosity does not affect mass).
    The extra terms are compatible with the standard von Neumann–Richtmyer
    form used in most shock-capturing hydrocodes.
    """
    # Total energy density per unit volume
    E = rho * (e + 0.5 * u ** 2)
    rho_u = rho * u
    rho_u2_P = rho_u * u + P
    Epu = (E + P) * u

    # ---- mass
    drho_dt  = time_derivative(rho, t)
    drhou_dr = space_derivative(rho_u, r)
    R_mass   = drho_dt + drhou_dr + 2.0 * rho_u / r

    # ---- momentum
    d_rhou_dt   = time_derivative(rho_u, t)
    d_rhou2P_dr = space_derivative(rho_u2_P, r)
    R_mom       = d_rhou_dt + d_rhou2P_dr + 2.0 * rho * u ** 2 / r

    # ---- energy
    dE_dt   = time_derivative(E, t)
    dEpu_dr = space_derivative(Epu, r)
    R_energy = dE_dt + dEpu_dr + 2.0 * Epu / r

    # ---- optional local artificial viscosity
    # Viscous terms are expanded analytically (product rule) so that
    # ∂u/∂r and ∂²u/∂r² are each computed only once, avoiding the deep
    # nested-autograd cost of space_derivative(mu · ∂u/∂r, r).
    if mu_av is not None:
        du_dr = space_derivative(u, r)
        d2u_dr2 = second_space_derivative(u, r)
        if dmu_dr is None:
            raise ValueError("dmu_dr must be provided when mu_av is not None")
        av_mom = dmu_dr * du_dr + mu_av * d2u_dr2
        av_energy = dmu_dr * u * du_dr + mu_av * (du_dr ** 2 + u * d2u_dr2)
        R_mom = R_mom - av_mom
        R_energy = R_energy - av_energy

    return R_mass, R_mom, R_energy


def ideal_gas_internal_energy(
    rho: torch.Tensor, P: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Specific internal energy for a calorically perfect gas.

        e = P / [(gamma - 1) * rho]
    """
    return P / ((gamma - 1.0) * rho)


def jwl_isentropic_e(
    rho: torch.Tensor,
    *, A: float, B: float, R1: float, R2: float, omega: float,
    C: float, rho0: float, v_cj: float, e_cj: float,
) -> torch.Tensor:
    """Closed-form specific internal energy along the JWL principal isentrope.

    e(v) = e_cj - (1/rho0) * (F(v) - F(v_cj))
    F(v) = -A/R1 exp(-R1 v) - B/R2 exp(-R2 v) - C/omega * v^{-omega}

    All scalar parameters mirror the dataclass fields of
    :class:`physics.jwl_isentrope.JWLParams` plus ``v_cj`` (set when the CJ
    state is solved) and ``e_cj`` (the integration constant).  ``C`` equals
    ``omega * E0`` for the Wescott-Stewart-Davis form; it is passed
    explicitly here to keep the helper EOS-agnostic.
    """
    v = rho0 / rho
    F_v   = -A / R1 * torch.exp(-R1 * v)   - B / R2 * torch.exp(-R2 * v)   - C / omega * v   ** (-omega)
    F_cj  = -A / R1 * torch.exp(-R1 * torch.as_tensor(v_cj, dtype=v.dtype, device=v.device)) \
            - B / R2 * torch.exp(-R2 * torch.as_tensor(v_cj, dtype=v.dtype, device=v.device)) \
            - C / omega * (torch.as_tensor(v_cj, dtype=v.dtype, device=v.device)) ** (-omega)
    return e_cj - (F_v - F_cj) / rho0
