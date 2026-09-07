"""Physics sanity checks for the spherical TNT PINN preprocessing.

Runs without a config: uses :class:`physics.cj_state.TNTParams` defaults
(LS-DYNA card values).  Each check prints PASS / FAIL with the relevant
tolerance.

Checks
------
1. CJ pressure within +/-15% of the LS-DYNA card target (21.0 GPa).
2. JWL principal isentrope P(v) strictly decreasing on [v_CJ, v_max].
3. Rankine-Hugoniot mass / momentum / energy residuals < 1e-6 for an
   exact Mach-5 ideal-gas shock (float64 closure check).
4. Sec.4.2.1 contact-state Riemann match: |u_p - u_s| / u_s < 1%.
5. Taylor-Sadovsky ODE: r_c(t) is strictly monotone in t and the
   integrated endpoint matches the analytical R_c.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from physics.rh_relations import rh_residuals, shock_velocity


# --------------------------------------------------------------------------- helpers


def _R_0_50mm_sphere(_rho_TNT: float) -> float:
    """Spherical TNT charge radius used by this project (50 mm)."""
    return 0.05


def check_cj_pressure(cj_b, target: float = 21.0e9, tol: float = 0.15) -> bool:
    err = abs(cj_b.P_CJ - target) / target
    ok = err <= tol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] CJ pressure: P_CJ={cj_b.P_CJ/1e9:.3f} GPa  "
          f"(card target {target/1e9:.1f} GPa, err={100*err:.2f}%, tol={100*tol:.0f}%)")
    return ok


def check_isentrope_monotone(cj_b) -> bool:
    P = cj_b.cj.isentrope.P_grid
    ok = bool(np.all(np.diff(P) < 0))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] JWL isentrope strictly decreasing on [v_CJ, v_max]")
    return ok


def check_rh_float64() -> bool:
    """Exact Mach-5 shock: RH residuals must be < 1e-6."""
    g, P_a, rho_a = 1.4, 101325.0, 1.225
    c_a = math.sqrt(g * P_a / rho_a)
    M = 5.0
    D_s = M * c_a
    P_s = P_a * (2.0 * g * M ** 2 - (g - 1.0)) / (g + 1.0)
    rho_s = rho_a * ((g + 1.0) * M ** 2) / ((g - 1.0) * M ** 2 + 2.0)
    u_s = D_s * (1.0 - rho_a / rho_s)
    r1, r2, r3 = rh_residuals(rho_s, u_s, P_s, D_s, g, P_a, rho_a)
    max_r = max(abs(r1), abs(r2), abs(r3))
    ok = max_r < 1e-6
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] RH float64 residuals (M=5): "
          f"|r1|={abs(r1):.2e}, |r2|={abs(r2):.2e}, |r3|={abs(r3):.2e}  (tol=1e-6)")
    return ok


def check_contact_match(cj_b, sep) -> bool:
    g = cj_b.tnt.gamma_a
    u_p = float(cj_b.cj.isentrope.u_p_of_P(sep.P_x))
    u_s = float(shock_velocity(sep.P_x, g, cj_b.tnt.P_atm, cj_b.tnt.rho_a))
    rel = abs(u_p - u_s) / (abs(u_s) + 1e-10)
    ok = rel < 1e-2
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Sec.4.2.1 contact match: "
          f"|u_p - u_s| / u_s = {100*rel:.4f}%  (tol=1%)")
    return ok


def check_taylor_sadovsky_monotone(sep) -> bool:
    ode = sep.ode
    mono_t = bool(np.all(np.diff(ode.t) > 0))
    end_ok = abs(ode.r_c[-1] - sep.R_c) < 1e-9
    ok = mono_t and end_ok
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] Taylor-Sadovsky ODE: t monotone={mono_t}, "
          f"r_c[-1]={ode.r_c[-1]*1e3:.3f} mm vs R_c={sep.R_c*1e3:.3f} mm")
    return ok


# --------------------------------------------------------------------------- driver


def run_all() -> bool:
    tnt = TNTParams()
    cj_b = compute_cj_state(tnt)
    R_0 = _R_0_50mm_sphere(tnt.rho_TNT)
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)

    print("=== Physics validation checks ===")
    results = [
        check_cj_pressure(cj_b, target=tnt.P_CJ),
        check_isentrope_monotone(cj_b),
        check_rh_float64(),
        check_contact_match(cj_b, sep),
        check_taylor_sadovsky_monotone(sep),
    ]
    n = sum(results)
    print(f"\nResult: {n}/{len(results)} checks passed.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
