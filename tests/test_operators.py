"""Unit tests for pinn/operators.py — autograd derivative correctness.

Verifies that the autograd wrappers reproduce analytical derivatives for
simple polynomials, then checks the spherical-divergence formula.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pinn.operators import (
    grad_batched,
    space_derivative,
    time_derivative,
    second_space_derivative,
    spherical_divergence,
)


# ---------------------------------------------------------------- helpers

def _rt(n: int = 64, r_range=(0.1, 1.0), t_range=(0.0, 1.0)):
    """Return (r, t) tensors with requires_grad=True."""
    r = torch.empty(n, 1).uniform_(*r_range).requires_grad_(True)
    t = torch.empty(n, 1).uniform_(*t_range).requires_grad_(True)
    return r, t


# ---------------------------------------------------------------- tests

def test_df_dr_polynomial():
    """f(r, t) = r^2 * t  →  ∂f/∂r = 2*r*t."""
    r, t = _rt()
    f = r ** 2 * t
    df_dr = space_derivative(f, r)
    expected = 2.0 * r * t
    assert torch.allclose(df_dr, expected, atol=1e-5), (
        f"max |∂f/∂r - 2rt| = {(df_dr - expected).abs().max():.2e}"
    )


def test_df_dt_polynomial():
    """f(r, t) = r^2 * t  →  ∂f/∂t = r^2."""
    r, t = _rt()
    f = r ** 2 * t
    df_dt = time_derivative(f, t)
    expected = r ** 2
    assert torch.allclose(df_dt, expected, atol=1e-5), (
        f"max |∂f/∂t - r^2| = {(df_dt - expected).abs().max():.2e}"
    )


def test_d2f_dr2_polynomial():
    """f(r, t) = r^2 * t  →  ∂²f/∂r² = 2*t."""
    r, t = _rt()
    f = r ** 2 * t
    d2f = second_space_derivative(f, r)
    expected = 2.0 * t * torch.ones_like(r)
    assert torch.allclose(d2f, expected, atol=1e-4), (
        f"max |∂²f/∂r² - 2t| = {(d2f - expected).abs().max():.2e}"
    )


def test_spherical_divergence_constant():
    """div_sph(F=c) at radius r = ∂_r c + 2c/r = 2c/r."""
    r = torch.empty(64, 1).uniform_(0.1, 1.0).requires_grad_(True)
    c = 3.0
    F = torch.full_like(r, c, requires_grad=True)
    # For a constant F, ∂_r F = 0, so div = 2c/r
    div = spherical_divergence(F, r)
    expected = 2.0 * c / r
    # The autograd of a constant w.r.t. r is 0, so we just check the 2F/r term
    assert torch.allclose(div, expected, atol=1e-5), (
        f"max |div_sph(c) - 2c/r| = {(div - expected).abs().max():.2e}"
    )


def test_spherical_divergence_linear():
    """div_sph(F=a*r) = a + 2a = 3a  (for F linear in r)."""
    r = torch.empty(64, 1).uniform_(0.1, 1.0).requires_grad_(True)
    a = 2.5
    F = a * r
    div = spherical_divergence(F, r)
    expected = 3.0 * a * torch.ones_like(r)
    assert torch.allclose(div, expected, atol=1e-4), (
        f"max |div_sph(a*r) - 3a| = {(div - expected).abs().max():.2e}"
    )


def test_grad_batched_sin():
    """∂/∂r sin(r) = cos(r)."""
    r = torch.empty(64, 1).uniform_(0.1, 2.0).requires_grad_(True)
    f = torch.sin(r)
    df = grad_batched(f, r)
    expected = torch.cos(r)
    assert torch.allclose(df, expected, atol=1e-5), (
        f"max |d/dr sin(r) - cos(r)| = {(df - expected).abs().max():.2e}"
    )
