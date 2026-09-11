"""Endpoint, material-jump and predictor round-trip regression checks."""
import pytest
import torch

from physics.rh_relations import rh_residuals, shock_speed, shock_velocity
from pinn.checkpoints import load_checkpoint, save_checkpoint
from pinn.coupling import air_contact_density
from pinn.networks import AirShockNet, DetonationNet, HardContactConstrainedASN, HardDetNetConstraint
from pinn.losses.air_shock_loss import AirShockICLoss


def product_net():
    raw = DetonationNet(depth=2, width=8, tau_scale=2e-5, Z_scale=0.2)
    return HardDetNetConstraint(
        raw, tau_sep=1e-5, Z_c=0.11, Z_R0=0.05,
        rho_cj=2200., u_cj=1800., P_cj=21e9,
        rho_x=147., u_x=7000., P_x=70e6,
        # Deliberately overlapping envelopes expose anchor contamination.
        tau_t=2e-5, tau_r=0.15, tau_t_cj=2e-5, tau_r_cj=0.15,
        use_contact=True,
    )


def test_A_both_anchors_exact_with_overlapping_envelopes_and_center_symmetry():
    net = product_net()
    t = torch.tensor([[0.], [net.tau_sep], [5e-6]], requires_grad=True)
    r = torch.tensor([[net.Z_R0], [net.Z_c], [0.]], requires_grad=True)
    outputs = net(t, r)
    for pred, first, last in zip(outputs, (2200., 1800., 21e9), (147., 7000., 70e6)):
        torch.testing.assert_close(pred[:2, 0], torch.tensor([first, last]), rtol=1e-6, atol=0.)
    assert outputs[1][2].item() == 0.
    for pred in outputs:
        for derivative in torch.autograd.grad(pred.sum(), (t, r), retain_graph=True):
            assert torch.isfinite(derivative).all()
    # Positive thermodynamic fields throughout the interpolation region.
    rho, _, P = net(torch.rand(64, 1)*1e-5, torch.rand(64, 1)*0.11)
    assert (rho > 0).all() and (P > 0).all()


@pytest.mark.parametrize("P", [101325., 2e5, 70e6])
def test_air_density_satisfies_mass_momentum_energy_RH(P):
    air = dict(gamma=1.4, rho_a=1.225, P_a=101325.)
    rho = float(air_contact_density(torch.tensor(P, dtype=torch.float64), **air))
    D = float(shock_speed(P, **air))
    u = float(shock_velocity(P, **air))
    residuals = rh_residuals(rho, u, P, D, **air)
    scales = (air["rho_a"]*D, P, D**2)
    assert max(abs(v)/scale for v, scale in zip(residuals, scales)) < 1e-12
    assert air["rho_a"] <= rho <= 6*air["rho_a"]


def test_air_density_rejects_nonshock_pressure():
    with pytest.raises(ValueError, match="ambient pressure"):
        air_contact_density(torch.tensor(1.), gamma=1.4, rho_a=1.225, P_a=101325.)


def test_B_wrapper_and_IC_use_air_density_not_product_density():
    det = product_net()
    t = torch.tensor([[det.tau_sep]])
    r = torch.tensor([[det.Z_c]])
    rho_prod, u, P = det(t, r)
    rho_air = air_contact_density(P, gamma=1.4, rho_a=1.225, P_a=101325.)
    net = HardContactConstrainedASN(
        AirShockNet(depth=2, width=8), tau_sep=det.tau_sep, Z_c=det.Z_c,
        target_rho=rho_air, target_u=u, target_P=P, density_source="air_rh",
    )
    rho_B, u_B, P_B = net(t, r)
    assert not torch.allclose(rho_B, rho_prod)
    for actual, expected in ((rho_B, rho_air), (u_B, u), (P_B, P)):
        torch.testing.assert_close(actual, expected)
    loss = AirShockICLoss(net, det, det.tau_sep, det.Z_c, 1.,
                         rho_a=1.225, P_a=101325., N_amb=0, gamma=1.4)
    assert float(loss().detach()) == 0.


@pytest.mark.parametrize("phase", ["detonation", "air_shock"])
@pytest.mark.parametrize("suffix", ["", "_best"])
def test_full_predictor_roundtrip_restores_scales_and_constraints(tmp_path, phase, suffix):
    net = product_net()
    if phase == "air_shock":
        net = HardContactConstrainedASN(
            AirShockNet(depth=2, width=8, t_scale=0.002, r_scale=1.5, P_ref=2e6),
            tau_sep=1e-5, Z_c=0.11, target_rho=torch.tensor(7.3),
            target_u=torch.tensor(7000.), target_P=torch.tensor(70e6),
            density_source="air_rh",
        )
    path = tmp_path / (phase + suffix + ".pt")
    save_checkpoint(path, net, 10, 0.01)
    # A stale legacy sidecar must NEVER replace metadata in a new checkpoint.
    torch.save({"deliberately": "invalid"}, tmp_path / "air_shock_hc_meta.pt")
    raw = (DetonationNet(depth=2, width=8) if phase == "detonation" else
           AirShockNet(depth=2, width=8))
    restored, _ = load_checkpoint(path, raw, require_constraints=True)
    t = torch.tensor([[0.], [1e-5], [7e-6], [1e-4]])
    r = torch.tensor([[0.05], [0.11], [0.073], [0.5]])
    for before, after in zip(net(t, r), restored(t, r)):
        torch.testing.assert_close(before, after, rtol=0., atol=0.)


def test_old_A_cannot_be_used_as_corrected_B_input(tmp_path):
    raw = DetonationNet(depth=2, width=8)
    path = tmp_path / "detonation.pt"
    torch.save({"step": 4, "loss": 0., "state_dict": raw.state_dict()}, path)
    with pytest.raises(ValueError, match="Re-train Phase A"):
        load_checkpoint(path, raw, require_constraints=True)
