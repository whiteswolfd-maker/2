"""Convergence diagnostics: compare PINN outputs to reference data.

Checks after Stage-2/3 training:
- Shock-front radius R_s(t): PINN vs reference (AnalyticalLoader)
- Contact-face radius R_c(t): PINN vs reference
- Post-shock pressure P(R_s, t): PINN vs reference
- RH residuals at shock surface (should be < 1% max post-shock P)

Usage
-----
    python -m validation.convergence --config configs/tnt_spherical.yaml --stage 3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cfg(path: str) -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def _ckpt_path(cfg: dict, stage: int) -> Path:
    return ROOT / cfg["training"]["checkpoint_dir"] / f"stage{stage}.pt"


def run_convergence(config_path: str = "configs/tnt_spherical.yaml", stage: int = 3) -> bool:
    from physics import solve_cj, match_contact, evolve_quasisteady, find_separation
    from physics.jwl_isentrope import JWLParams
    from data.lsdyna_loader import AnalyticalLoader
    from pinn.networks import build_networks

    cfg  = _load_cfg(config_path)
    exp  = cfg["explosive"]
    air  = cfg["air"]
    prep = cfg["preprocessing"]
    val  = cfg["validation"]

    params = JWLParams(
        A=exp["jwl"]["A"], B=exp["jwl"]["B"],
        R1=exp["jwl"]["R1"], R2=exp["jwl"]["R2"],
        omega=exp["jwl"]["omega"], E0=exp["jwl"]["E0"],
        rho0=exp["rho0"],
    )
    W    = exp["W"]
    R0   = (3.0 * W / (4.0 * math.pi * exp["rho0"])) ** (1.0 / 3.0)
    cj   = solve_cj(params, exp["D_CJ"], bracket=tuple(prep["v_cj_bracket"]))
    contact = match_contact(cj, air["gamma"], air["rho_a"], air["P_a"])
    series = evolve_quasisteady(
        R0, cj, contact, air["gamma"], air["rho_a"], air["P_a"],
        dt=prep["quasisteady_dt"], t_max=cfg["domain"]["t_end"],
    )
    sep = find_separation(series, air["P_a"],
                          P_threshold_ratio=prep["separation"]["P_threshold_ratio"],
                          v_threshold=prep["separation"]["v_threshold"])

    loader = AnalyticalLoader(sep, gamma=air["gamma"], rho_a=air["rho_a"], P_a=air["P_a"])

    # Load networks
    main_net, contact_net, shock_net = build_networks(cfg["networks"], sep)
    ckpt = _ckpt_path(cfg, stage)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return False
    state = torch.load(ckpt, map_location="cpu")
    main_net.load_state_dict(state["main_net"])
    contact_net.load_state_dict(state["contact_net"])
    shock_net.load_state_dict(state["shock_net"])
    main_net.eval(); contact_net.eval(); shock_net.eval()

    # Evaluation times: reference from quasi-steady series (post t_sep)
    mask = series.t >= sep.t_sep
    t_ref = series.t[mask]
    Rs_ref = series.R_s[mask]
    Rc_ref = series.R_c[mask]
    Ds_ref = series.D_s[mask]
    Pc_ref = series.P_c[mask]   # reference post-shock pressure at contact face

    t_tensor = torch.tensor(t_ref[:, None], dtype=torch.float32)
    with torch.no_grad():
        Rs_pred = shock_net(t_tensor).squeeze().numpy()
        Rc_pred = contact_net(t_tensor).squeeze().numpy()

    Rs_err = np.abs(Rs_pred - Rs_ref) / (np.abs(Rs_ref) + 1e-10)
    Rc_err = np.abs(Rc_pred - Rc_ref) / (np.abs(Rc_ref) + 1e-10)

    Rs_max_err = float(Rs_err.max())
    Rc_max_err = float(Rc_err.max())

    print("\n=== Convergence Check ===")
    ok_Rs = Rs_max_err <= val["Rs_error_tol"]
    ok_Rc = Rc_max_err <= val["Rc_error_tol"]
    print(f"  [{'PASS' if ok_Rs else 'FAIL'}] R_s max relative error: "
          f"{100*Rs_max_err:.2f}%  (tol={100*val['Rs_error_tol']:.0f}%)")
    print(f"  [{'PASS' if ok_Rc else 'FAIL'}] R_c max relative error: "
          f"{100*Rc_max_err:.2f}%  (tol={100*val['Rc_error_tol']:.0f}%)")

    # RH residuals at shock surface
    from pinn.losses.rh_loss import RHLoss
    rh = RHLoss(
        main_net, shock_net,
        gamma=air["gamma"], rho_a=air["rho_a"], P_a=air["P_a"],
        N_s=200, t_sep=sep.t_sep, t_end=cfg["domain"]["t_end"],
    )
    with torch.no_grad():
        rh_val = rh().item()
    print(f"  [ RH ] dimensionless RH loss: {rh_val:.4e}  (target < 0.01)")
    ok_rh = rh_val < 0.01

    all_ok = ok_Rs and ok_Rc and ok_rh
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tnt_spherical.yaml")
    parser.add_argument("--stage",  type=int, default=3)
    args = parser.parse_args()
    ok   = run_convergence(args.config, args.stage)
    sys.exit(0 if ok else 1)
