"""Air-side state at the model's initial shock-outflow matching point."""
from __future__ import annotations

import math

import torch


def air_contact_density(P: torch.Tensor, *, gamma: float, rho_a: float,
                        P_a: float) -> torch.Tensor:
    """Return post-shock AIR density from the ideal-gas RH pressure ratio.

    This is the initial matching-state approximation used by match_contact:
    the contact-side air state is identified with the initial post-shock
    state. It is not a density-continuity condition, nor a formula for the
    contact density throughout later spherical propagation. P is absolute.
    """
    if not all(math.isfinite(v) for v in (gamma, rho_a, P_a)) or \
            gamma <= 1 or rho_a <= 0 or P_a <= 0:
        raise ValueError("Air RH parameters require gamma > 1, rho_a > 0, P_a > 0.")
    if not torch.isfinite(P).all() or torch.any(P < P_a):
        raise ValueError("Initial air-shock pressure must be finite and >= ambient pressure.")
    return rho_a * ((gamma + 1) * P + (gamma - 1) * P_a) / \
        ((gamma - 1) * P + (gamma + 1) * P_a)
