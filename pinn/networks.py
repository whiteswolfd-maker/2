"""Two PINN networks for the spherical R_0 = 50 mm TNT blast model.

Forward-coupled, sequentially trained:

DetonationNet
    SIREN (t, r) -> (rho, u, P) on the triangular Phase-A domain
        {t in [0, t_sep], r in [0, r_c(t)]}
    EOS: JWL principal isentrope (products evolve isentropically until t_sep).
    Spherical symmetry r >= 0 enforced by feeding |r| as the radial input.

AirShockNet / FourierFeatureAirShockNet
    (t, r) -> (rho, u, P) on the rectangular Phase-B domain
        {t in [t_sep, t_end], r in [R_c, x_end]}
    EOS: ideal-gas air, gamma_a = 1.4.
    Use ``FourierFeatureAirShockNet`` for blast-wave problems (overcomes
    SIREN spectral bias); set ``air_shock.net_type: fourier_mlp`` in config.

The two networks are coupled in one direction at the single point
(t_sep, R_c): once DetonationNet is trained and frozen, its prediction
(u_x_pred, P_x_pred) at that point supplies the air-side matching state.
Air density is computed separately from the initial air-shock RH relation.

References
----------
Sitzmann et al. "Implicit Neural Representations with Periodic Activation
Functions." NeurIPS 2020.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ SIREN helpers


def _siren_init(layer: nn.Linear, is_first: bool, omega0: float) -> None:
    """SIREN weight initialisation (Sitzmann et al., Supplement S1)."""
    with torch.no_grad():
        fan_in = layer.weight.size(1)
        bound = (1.0 / fan_in) if is_first else (math.sqrt(6.0 / fan_in) / omega0)
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


# ============================================================ Phase-A network


class DetonationNet(nn.Module):
    """SIREN (tau, Z) -> (rho, u, P) for the JWL-products triangular domain.

    Inputs are normalised to O(1) so the SIREN first layer sees comparable
    contributions from tau and Z.  Without this raw tau (~1e-5) is 2500x
    weaker than Z (~0.1) and the network is effectively blind to time.

    Spherical symmetry is hard-wired by feeding ``|Z|`` to the network so
    ``u(tau, 0) = 0`` is automatically satisfied.

    Outputs are post-processed to guarantee positivity (rho > 0, P > 0).
    Velocity u is signed.
    """

    def __init__(
        self,
        depth: int = 6,
        width: int = 128,
        omega0: float = 30.0,
        rho_ref: float = 1630.0,
        u_ref:   float = 2000.0,
        P_ref:   float = 21.0e9,
        tau_scale: float = 1.0e-5,     # normalise tau -> O(1)
        Z_scale:   float = 0.12,        # normalise Z   -> O(1)
        eps_origin: float = 1.0e-4,     # length scale for u(tau,0)=0 (in Z units)
    ) -> None:
        super().__init__()
        self.rho_ref = rho_ref
        self.u_ref   = u_ref
        self.P_ref   = P_ref
        self.tau_scale = float(tau_scale)
        self.Z_scale   = float(Z_scale)
        self.eps_origin = float(eps_origin)

        layers: list[nn.Module] = [SirenLayer(2, width, omega0=omega0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SirenLayer(width, width, omega0=omega0, is_first=False))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, tau: torch.Tensor, Z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho, u, P).  ``tau`` and ``Z`` can be physical or scaled."""
        tau_norm = tau / self.tau_scale
        Z_norm   = Z   / self.Z_scale
        Z_sym = Z_norm.abs()
        h = self.hidden(torch.cat([tau_norm, Z_sym], dim=-1))
        raw = self.out(h)
        rho = self.rho_ref * F.softplus(raw[..., 0:1])
        u   = self.u_ref   * raw[..., 1:2]
        P   = self.P_ref   * F.softplus(raw[..., 2:3])
        u = u * torch.tanh(Z / self.eps_origin)
        return rho, u, P


# ============================================================ Phase-B networks


class AirShockNet(nn.Module):
    """SIREN (t, r) -> (rho, u, P) for the ideal-gas Phase-B rectangular domain.

    No spherical-symmetry trick: r >= R_c > 0 throughout the domain.

    Inputs are normalised to O(1) so the SIREN first layer sees comparable
    contributions from t and r.  Without this the raw t signal (~1e-5 to
    1e-3 s) is 2500× weaker than r (~0.1 to 2.5 m), making the network
    effectively blind to time.

    Outputs are post-processed for positivity; u is signed (positive outward
    away from the products).
    """

    def __init__(
        self,
        depth: int = 6,
        width: int = 128,
        omega0: float = 30.0,
        rho_ref: float = 1.225,         # rho_a scale
        u_ref:   float = 1000.0,        # post-shock air velocity scale
        P_ref:   float = 1.0e6,         # post-shock pressure scale
        t_scale: float = 1.0e-3,        # normalise t/tau → O(1)
        r_scale: float = 2.64,          # normalise r/Z  → O(1) (data Z range)
    ) -> None:
        super().__init__()
        self.rho_ref = rho_ref
        self.u_ref   = u_ref
        self.P_ref   = P_ref
        self.t_scale = float(t_scale)
        self.r_scale = float(r_scale)

        layers: list[nn.Module] = [SirenLayer(2, width, omega0=omega0, is_first=True)]
        for _ in range(depth - 1):
            layers.append(SirenLayer(width, width, omega0=omega0, is_first=False))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, 3)
        # Small output weights + bias so initial prediction ≈ ambient air
        # softplus⁻¹(1.0) ≈ 0.5413  → rho = rho_ref
        # softplus⁻¹(P_a/P_ref) = softplus⁻¹(0.1013) ≈ -2.239  → P = P_a
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.5413, 0.0, -2.239]))

    def forward(
        self, t: torch.Tensor, r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho, u, P) tensors with shapes matching ``t`` and ``r``."""
        t_norm = t / self.t_scale
        r_norm = r / self.r_scale
        h = self.hidden(torch.cat([t_norm, r_norm], dim=-1))
        raw = self.out(h)
        rho = self.rho_ref * F.softplus(raw[..., 0:1])
        u   = self.u_ref   * raw[..., 1:2]
        P   = self.P_ref   * F.softplus(raw[..., 2:3])
        return rho, u, P


class GeluAirShockNet(nn.Module):
    """Plain GELU MLP (t, r) -> (rho, u, P) for the Phase-B rectangular domain.

    No Fourier features, no SIREN sin activations — just a standard deep MLP
    with GELU nonlinearities.  Inputs are normalised to O(1) before the first
    layer so temporal and spatial coordinates contribute comparably.

    Output bias is initialised so the network starts near ambient air, giving
    low initial PDE residuals and stable early training.
    """

    def __init__(
        self,
        depth: int = 6,
        width: int = 384,
        rho_ref: float = 1.225,
        u_ref:   float = 1000.0,
        P_ref:   float = 1.0e6,
        t_scale: float = 1.0e-3,
        r_scale: float = 2.64,
    ) -> None:
        super().__init__()
        self.rho_ref = rho_ref
        self.u_ref   = u_ref
        self.P_ref   = P_ref
        self.t_scale = float(t_scale)
        self.r_scale = float(r_scale)

        layers: list[nn.Module] = [nn.Linear(2, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 3)
        # Initialise near ambient (same as FourierFeatureAirShockNet)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-4)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.5413, 0.0, -2.239]))

    def forward(
        self, t: torch.Tensor, r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_norm = t / self.t_scale
        r_norm = r / self.r_scale
        h = self.mlp(torch.cat([t_norm, r_norm], dim=-1))
        raw = self.out(h)
        rho = self.rho_ref * F.softplus(raw[..., 0:1])
        u   = self.u_ref   * raw[..., 1:2]
        P   = self.P_ref   * F.softplus(raw[..., 2:3])
        return rho, u, P


class FourierFeatureAirShockNet(nn.Module):
    """Fourier-feature MLP for Phase B — overcomes SIREN spectral bias.

    Instead of learning high frequencies from scratch (which SIREN failed at),
    the input (t, r) is first projected onto sin/cos features at multiple
    octaves, then processed by a standard GELU MLP.  This gives the network
    explicit access to frequencies from 1× up to 2^{L-1}× the domain scale,
    so the shock front — a near-discontinuity at the ~5 mm / 5 μs scale — is
    within the representable bandwidth.

    Architecture
    ------------
    γ(x) = [cos(π·2^l·x), sin(π·2^l·x)]  for l = 0 … L-1, x ∈ {t_norm, r_norm}
    MLP:  encode(4·L) → width (× depth) → 3 outputs

    References
    ----------
    Tancik et al. "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains." NeurIPS 2020.
    """

    def __init__(
        self,
        depth: int = 6,
        width: int = 256,
        num_octaves: int = 10,
        rho_ref: float = 1.225,
        u_ref:   float = 1000.0,
        P_ref:   float = 1.0e6,
        t_scale: float = 1.0e-3,
        r_scale: float = 2.64,          # data Z range, not domain Z_max
    ) -> None:
        super().__init__()
        self.rho_ref = rho_ref
        self.u_ref   = u_ref
        self.P_ref   = P_ref
        self.t_scale = float(t_scale)
        self.r_scale = float(r_scale)
        self.num_octaves = int(num_octaves)

        # Fourier-feature encoding: 2 inputs × 2 (sin,cos) × L octaves
        encode_dim = 4 * self.num_octaves
        layers: list[nn.Module] = [nn.Linear(encode_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 3)
        # Small output weights; bias set so initial prediction = ambient air
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-4)
        # softplus_inv(1.0) ≈ 0.5413  → rho = rho_ref
        # softplus_inv(P_a/P_ref) = softplus_inv(0.1013) ≈ -2.239  → P = P_a
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.5413, 0.0, -2.239]))

        # Cache frequency multipliers (not parameters, just constants)
        freqs = 2.0 ** torch.arange(self.num_octaves, dtype=torch.float32)
        self.register_buffer("_freqs", freqs, persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """γ(x) = [cos(π·f·x), sin(π·f·x)] for f ∈ freqs, for each input dim."""
        proj = math.pi * x.unsqueeze(-1) * self._freqs  # (N, 1, L)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1).flatten(1)

    def forward(
        self, t: torch.Tensor, r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_norm = t / self.t_scale
        r_norm = r / self.r_scale
        feat_t = self._encode(t_norm)
        feat_r = self._encode(r_norm)
        h = self.mlp(torch.cat([feat_t, feat_r], dim=-1))
        raw = self.out(h)
        rho = self.rho_ref * F.softplus(raw[..., 0:1])
        u   = self.u_ref   * raw[..., 1:2]
        P   = self.P_ref   * F.softplus(raw[..., 2:3])
        return rho, u, P


# ============================================================ hard-constraint wrappers


class HardDetNetConstraint(nn.Module):
    """Smooth interpolation fixing CJ and the products-side outflow point.

    Gaussian envelopes are multiplied by cardinal weights, so the influence
    of the OTHER anchor is exactly zero at each anchor, even for overlapping
    envelopes. Nonnegative weights sum to <= 1, preserving rho/P positivity.
    Coordinates and envelope widths use the same units as the base network.
    """

    def __init__(
        self,
        net: nn.Module,
        tau_sep: float, Z_c: float,
        rho_cj: float, u_cj: float, P_cj: float,
        rho_x: float,  u_x: float,  P_x: float,
        Z_R0: float = 0.052712,
        tau_t:   float = 1.0e-6,   tau_r:   float = 1.0e-3,
        tau_t_cj: float = 1.0e-7,  tau_r_cj: float = 5.0e-4,
        use_contact: bool = True,
    ) -> None:
        super().__init__()
        self.net = net
        self.tau_sep = float(tau_sep); self.Z_c = float(Z_c)
        self.Z_R0 = float(Z_R0)
        self.tau_t = float(tau_t); self.tau_r = float(tau_r)
        self.tau_t_cj = float(tau_t_cj); self.tau_r_cj = float(tau_r_cj)
        self.use_contact = bool(use_contact)
        if not all(math.isfinite(v) and v > 0 for v in (
            self.tau_sep, self.Z_c, self.Z_R0, self.tau_t, self.tau_r,
            self.tau_t_cj, self.tau_r_cj,
        )):
            raise ValueError("Anchor time, radii and envelope widths must be positive and finite.")
        self.register_buffer("rho_cj", torch.tensor(rho_cj))
        self.register_buffer("u_cj",   torch.tensor(u_cj))
        self.register_buffer("P_cj",   torch.tensor(P_cj))
        self.register_buffer("rho_x",  torch.tensor(rho_x))
        self.register_buffer("u_x",    torch.tensor(u_x))
        self.register_buffer("P_x",    torch.tensor(P_x))

    def forward(self, tau: torch.Tensor, Z: torch.Tensor):
        rho_raw, u_raw, P_raw = self.net(tau, Z)

        d_cj = (tau / self.tau_t_cj).square() + ((Z - self.Z_R0) / self.tau_r_cj).square()
        d_cont = ((tau - self.tau_sep) / self.tau_t).square() + \
                 ((Z - self.Z_c) / self.tau_r).square()
        w_cj = torch.exp(-d_cj)
        w_cont = torch.zeros_like(w_cj)
        if self.use_contact:
            denom = (d_cj + d_cont).clamp_min(torch.finfo(tau.dtype).tiny)
            w_cj = w_cj * d_cont / denom
            w_cont = torch.exp(-d_cont) * d_cj / denom
        w_raw = (1.0 - w_cj - w_cont).clamp_min(0.0)
        rho = rho_raw * w_raw + self.rho_cj * w_cj + self.rho_x * w_cont
        P = P_raw * w_raw + self.P_cj * w_cj + self.P_x * w_cont
        # Preserve the base network's u(t,0)=0 without shifting either anchor.
        eps = float(getattr(self.net, "eps_origin", 1e-4))
        radial = torch.tanh(Z / eps)
        u = u_raw * w_raw + radial * (
            self.u_cj * w_cj / math.tanh(self.Z_R0 / eps)
            + self.u_x * w_cont / math.tanh(self.Z_c / eps)
        )
        return rho, u, P


class HardContactConstrainedASN(nn.Module):
    """Wraps AirShockNet with a hard constraint at the coupling point (tau_sep, Z_c).

    The raw AirShockNet output is blended with the frozen DetonationNet
    prediction via a narrow exponential envelope::

        w(tau,Z) = exp(-|tau - tau_sep|/tau_t  -  |Z - Z_c|/tau_r)
        output = raw * (1 - w)  +  target * w

    At (tau_sep, Z_c), w = 1, so the output is EXACTLY the DetNet value.
    All coordinates in scaled (tau, Z) units.
    """

    def __init__(
        self,
        net: nn.Module,
        tau_sep: float,
        Z_c: float,
        target_rho: torch.Tensor,
        target_u:   torch.Tensor,
        target_P:   torch.Tensor,
        tau_t: float = 1.0e-6,
        tau_r: float = 1.0e-3,
        density_source: str = "explicit",
    ) -> None:
        super().__init__()
        self.net = net
        self.tau_sep = float(tau_sep)
        self.Z_c = float(Z_c)
        self.tau_t = float(tau_t)
        self.tau_r = float(tau_r)
        self.density_source = density_source
        self.register_buffer("target_rho", target_rho.detach().clone())
        self.register_buffer("target_u",   target_u.detach().clone())
        self.register_buffer("target_P",   target_P.detach().clone())

    def forward(
        self, tau: torch.Tensor, Z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho_raw, u_raw, P_raw = self.net(tau, Z)
        w = torch.exp(
            -torch.abs(tau - self.tau_sep) / self.tau_t
            - torch.abs(Z   - self.Z_c)    / self.tau_r
        )
        rho = rho_raw * (1.0 - w) + self.target_rho * w
        u   = u_raw   * (1.0 - w) + self.target_u   * w
        P   = P_raw   * (1.0 - w) + self.target_P   * w
        return rho, u, P


# ============================================================ factory


def build_networks(
    cfg: dict,
    rho_ref_A: float | None = None,
    u_ref_A:   float | None = None,
    P_ref_A:   float | None = None,
    eps_origin_A: float | None = None,
    rho_ref_B: float | None = None,
    u_ref_B:   float | None = None,
    P_ref_B:   float | None = None,
    t_scale_B: float | None = None,
    r_scale_B: float | None = None,
) -> tuple[DetonationNet, nn.Module]:
    """Construct (DetonationNet, AirShockNet) from the ``networks`` config.

    Expected sub-dicts: ``cfg["detonation"]`` and ``cfg["air_shock"]`` each
    with keys ``depth``, ``width``, ``omega0`` (SIREN) or ``num_octaves``
    (Fourier-feature MLP).  Set ``air_shock.net_type: "fourier_mlp"`` to
    use the Fourier-feature architecture (default ``"siren"``).
    """
    dc = cfg["detonation"]
    ac = cfg["air_shock"]

    A_kwargs = {"depth": dc["depth"], "width": dc["width"], "omega0": dc["omega0"]}
    if rho_ref_A is not None: A_kwargs["rho_ref"] = rho_ref_A
    if u_ref_A   is not None: A_kwargs["u_ref"]   = u_ref_A
    if P_ref_A   is not None: A_kwargs["P_ref"]   = P_ref_A
    if eps_origin_A is not None: A_kwargs["eps_origin"] = eps_origin_A

    net_type = ac.get("net_type", "siren")
    B_kwargs: dict = {"depth": ac["depth"], "width": ac["width"]}
    if rho_ref_B is not None: B_kwargs["rho_ref"] = rho_ref_B
    if u_ref_B   is not None: B_kwargs["u_ref"]   = u_ref_B
    if P_ref_B   is not None: B_kwargs["P_ref"]   = P_ref_B
    if t_scale_B is not None: B_kwargs["t_scale"] = t_scale_B
    if r_scale_B is not None: B_kwargs["r_scale"] = r_scale_B

    if net_type == "fourier_mlp":
        B_kwargs["num_octaves"] = ac.get("num_octaves", 10)
        B_net: nn.Module = FourierFeatureAirShockNet(**B_kwargs)
    elif net_type == "gelu_mlp":
        B_net = GeluAirShockNet(**B_kwargs)
    else:
        B_kwargs["omega0"] = ac["omega0"]
        B_net = AirShockNet(**B_kwargs)

    return DetonationNet(**A_kwargs), B_net
