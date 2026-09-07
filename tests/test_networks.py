"""Network unit tests (CLAUDE.md anchors).

* DetonationNet: u(tau, 0) = 0 structurally enforced; rho, P > 0.
* AirShockNet variants initialise near ambient air.
* build_networks returns (DetonationNet, net_type-selected Phase-B net).
"""
from __future__ import annotations

import torch

from pinn.networks import (
    AirShockNet,
    DetonationNet,
    FourierFeatureAirShockNet,
    GeluAirShockNet,
    build_networks,
)

_DET = {"depth": 3, "width": 16, "omega0": 30.0}
_ASN = {"depth": 3, "width": 16, "omega0": 30.0}


def test_detonation_net_u_zero_at_origin():
    net = DetonationNet(depth=3, width=16, omega0=30.0)
    tau = torch.zeros(16, 1)
    Z = torch.zeros(16, 1)
    rho, u, P = net(tau, Z)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)


def test_detonation_net_positivity():
    net = DetonationNet(depth=3, width=16, omega0=30.0)
    tau = torch.linspace(0.0, 1.0, 32).view(-1, 1) * 1e-5
    Z = torch.linspace(1e-4, 0.12, 32).view(-1, 1)
    rho, u, P = net(tau, Z)
    assert (rho > 0).all() and (P > 0).all()
    assert rho.shape == (32, 1) and u.shape == (32, 1) and P.shape == (32, 1)


def test_air_shock_net_starts_near_ambient():
    rho_ref, P_a, P_ref = 1.225, 101325.0, 1.0e6
    net = AirShockNet(depth=3, width=16, rho_ref=rho_ref, P_ref=P_ref)
    t = torch.zeros(8, 1)
    r = torch.full((8, 1), 0.3)
    rho, u, P = net(t, r)
    assert (rho > 0).all() and (P > 0).all()
    # Bias init pins rho ~ rho_ref and P ~ P_a to ~1e-2 relative accuracy.
    assert torch.abs(rho - rho_ref).max().item() < 0.05 * rho_ref
    assert torch.abs(P - P_a).max().item() < 0.05 * P_a


def test_fourier_feature_net_forward_shape():
    net = FourierFeatureAirShockNet(depth=3, width=32, num_octaves=6)
    t = torch.linspace(1.0e-5, 1.0e-3, 16).view(-1, 1)
    r = torch.linspace(0.1, 1.0, 16).view(-1, 1)
    rho, u, P = net(t, r)
    assert (rho > 0).all() and (P > 0).all()
    assert rho.shape == (16, 1)


def test_gelu_net_forward_shape():
    net = GeluAirShockNet(depth=3, width=32)
    t = torch.linspace(1.0e-5, 1.0e-3, 16).view(-1, 1)
    r = torch.linspace(0.1, 1.0, 16).view(-1, 1)
    rho, u, P = net(t, r)
    assert rho.shape == (16, 1)


def test_build_networks_fourier():
    cfg = {
        "detonation": dict(_DET),
        "air_shock": dict(_ASN, net_type="fourier_mlp", num_octaves=8),
    }
    det, asn = build_networks(cfg)
    assert isinstance(det, DetonationNet)
    assert isinstance(asn, FourierFeatureAirShockNet)
