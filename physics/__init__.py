"""Pure-NumPy analytical preprocessing for the TNT blast-wave PINN.

Sections map to the user's specification:
    - jwl_isentrope  : §1.2-1.3  JWL principal isentrope + u_p(V) Riemann invariant
    - cj_solver      : §1.2      Chapman-Jouguet tangency solve via Brent
    - rh_relations   : §1.4      Rankine-Hugoniot jump relations for an ideal gas
    - initial_coupling : §1.5-1.6  Contact matching and quasi-steady evolution to t_sep

This package has **no PyTorch dependency**.  All outputs are NumPy arrays and
plain-data containers, consumable by both ``data/`` and ``validation/``.
"""

from .cj_solver import CJState, solve_cj
from .initial_coupling import (
    ContactState,
    QuasiSteadySeries,
    SeparationState,
    evolve_quasisteady,
    find_separation,
    match_contact,
)
from .jwl_isentrope import JWLIsentrope, JWLParams
from .rh_relations import rho_ratio, shock_speed, shock_velocity

__all__ = [
    "CJState",
    "ContactState",
    "JWLIsentrope",
    "JWLParams",
    "QuasiSteadySeries",
    "SeparationState",
    "evolve_quasisteady",
    "find_separation",
    "match_contact",
    "rho_ratio",
    "shock_speed",
    "shock_velocity",
    "solve_cj",
]
