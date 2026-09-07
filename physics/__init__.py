"""Pure-NumPy analytical preprocessing for the spherical TNT PINN.

Modules:
    - jwl_isentrope    : JWL principal isentrope + u_p(V) Riemann invariant
    - cj_solver        : Chapman-Jouguet tangency via Brent
    - rh_relations     : Rankine-Hugoniot jump relations (ideal gas)
    - initial_coupling : BIT Sec.4.2.1 contact-state Riemann match
    - uniform_expansion: Taylor-Sadovsky uniform-expansion ODE for r_c(t)
    - cj_state         : Facade bundling the four above + TNTParams

This package has **no PyTorch dependency**: all outputs are NumPy arrays
and plain-data containers, consumable by both ``data/`` and ``validation/``.
"""

from .cj_solver import CJState, solve_cj
from .cj_state import (
    CJStateBundle,
    SeparationStateBundle,
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from .initial_coupling import ContactState, match_contact
from .jwl_isentrope import JWLIsentrope, JWLParams
from .rh_relations import rho_ratio, shock_speed, shock_velocity
from .uniform_expansion import (
    UniformExpansionResult,
    solve_taylor_sadovsky,
)

__all__ = [
    "CJState",
    "CJStateBundle",
    "ContactState",
    "JWLIsentrope",
    "JWLParams",
    "SeparationStateBundle",
    "TNTParams",
    "UniformExpansionResult",
    "compute_cj_state",
    "compute_separation_state",
    "match_contact",
    "rho_ratio",
    "shock_speed",
    "shock_velocity",
    "solve_cj",
    "solve_taylor_sadovsky",
]
