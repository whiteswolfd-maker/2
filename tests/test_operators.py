"""Operator unit tests: autograd derivatives + Euler residuals (MMS anchors).

Covers the anchors in CLAUDE.md:
* spherical Euler MMS: a quiescent state has zero residual;
  the 2/r geometric source is present (differs from the planar operator).
* first / second spatial and time derivatives of a known field.
"""
from __future__ import annotations

import torch

from pinn.operators import (
    euler_residuals,
    grad_batched,
    ideal_gas_internal_energy,
    second_space_derivative,
    space_derivative,
    spherical_euler_residuals,
    time_derivative,
)


def _field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x ** 2 * t


def test_grad_batched_sin():
    x = torch.linspace(0.0, 1.0, 8, requires_grad=True)
    df = grad_batched(torch.sin(x), x)
    assert torch.allclose(df, torch.cos(x), atol=1e-5)


def test_time_derivative_field():
    t = torch.linspace(0.0, 1.0, 8, requires_grad=True)
    x = torch.linspace(1.0, 2.0, 8, requires_grad=True)
    df_dt = time_derivative(_field(x, t), t)
    assert torch.allclose(df_dt, x ** 2, atol=1e-4)


def test_space_derivative_field():
    t = torch.linspace(0.0, 1.0, 8, requires_grad=True)
    x = torch.linspace(1.0, 2.0, 8, requires_grad=True)
    df_dx = space_derivative(_field(x, t), x)
    assert torch.allclose(df_dx, 2.0 * x * t, atol=1e-4)


def test_second_space_derivative_field():
    t = torch.linspace(0.0, 1.0, 8, requires_grad=True)
    x = torch.linspace(1.0, 2.0, 8, requires_grad=True)
    d2f = second_space_derivative(_field(x, t), x)
    assert torch.allclose(d2f, 2.0 * t * torch.ones_like(x), atol=1e-3)


def test_planar_euler_quiescent_zero():
    """Planar Euler residual is zero for a constant rest state."""
    x = torch.linspace(0.1, 1.0, 16, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 16, requires_grad=True)
    gamma = 1.4
    # keep an autograd graph by multiplying through 0*x (constants alone have no
    # grad_fn, which autograd.grad rejects).
    rho = 1.225 + 0.0 * x
    u = 0.0 * x
    P = 101325.0 + 0.0 * x
    R_m, R_p, R_e = euler_residuals(rho, u, P, x, t, gamma=gamma)
    assert torch.abs(R_m).max().item() < 1e-8
    assert torch.abs(R_p).max().item() < 1e-8
    assert torch.abs(R_e).max().item() < 1e-8


def test_spherical_euler_quiescent_zero():
    """Spherical Euler residual is zero for a constant rest state (no source)."""
    r = torch.linspace(0.05, 1.0, 16, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 16, requires_grad=True)
    gamma = 1.4
    rho = 1.225 + 0.0 * r
    u = 0.0 * r
    P = 101325.0 + 0.0 * r
    e = ideal_gas_internal_energy(rho, P, gamma)
    R_m, R_p, R_e = spherical_euler_residuals(rho, u, P, e, r, t)
    assert torch.abs(R_m).max().item() < 1e-8
    assert torch.abs(R_p).max().item() < 1e-8
    assert torch.abs(R_e).max().item() < 1e-8


def test_spherical_source_terms_present():
    """For an outward-moving uniform flow the spherical residual must pick up
    the 2/r geometric source terms that the planar operator does not have."""
    x = torch.linspace(0.2, 1.0, 16, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 16, requires_grad=True)
    gamma = 1.4
    rho = 1.225 + 0.0 * x
    u = 100.0 + 0.0 * x            # uniform outward flow
    P = 101325.0 + 0.0 * x
    e = ideal_gas_internal_energy(rho, P, gamma)
    R_planar = euler_residuals(rho, u, P, x, t, gamma=gamma)
    R_sph = spherical_euler_residuals(rho, u, P, e, x, t)
    # planar residuals are ~0 (uniform flow); spherical ones are not, because of
    # the +2 rho u / r and +2 rho u^2 / r sources.
    assert torch.abs(R_planar[0]).max().item() < 1e-6
    assert torch.abs(R_sph[0]).max().item() > 1e-3
    assert torch.abs(R_sph[1]).max().item() > 1e-3
