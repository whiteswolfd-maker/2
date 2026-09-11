"""Quick unified model evaluation across all three radii."""
from __future__ import annotations
import sys, math, yaml, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from data.multi_radius_dataset import _M_third
from pinn.networks import build_networks
from pinn.checkpoints import load_checkpoint
from physics.cj_state import TNTParams, compute_cj_state, compute_separation_state


def main():
    with open(ROOT / "configs/tnt_spherical_unified.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    air = cfg["air"]
    tnt = TNTParams()
    cj = compute_cj_state(tnt)

    det, asn = build_networks(
        cfg["networks"],
        rho_ref_A=cj.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj.P_CJ,
        rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
    )

    ckpt_dir = ROOT / cfg["training"]["checkpoint_dir"]
    det_eval, _ = load_checkpoint(ckpt_dir / "detonation.pt", det, require_constraints=True)
    air_path = ckpt_dir / "air_shock_best.pt"
    if not air_path.exists():
        air_path = ckpt_dir / "air_shock.pt"
    asn_eval, _ = load_checkpoint(air_path, asn, require_constraints=True)
    if (asn_eval.tau_sep, asn_eval.Z_c) != (det_eval.tau_sep, det_eval.Z_c):
        raise RuntimeError("A/B checkpoints use different connection coordinates; re-train B.")
    TAU_SEP = det_eval.tau_sep
    Z_C = det_eval.Z_c
    evaluations: list[dict] = []

    for spec in cfg["data"]["radii"]:
        R0 = float(spec["R_0"])
        edir = str(spec["extracted_dir"])
        mt = _M_third(R0)
        Z_max_data = float(D3plotLineDataset(edir).x_end) / mt
        ds = D3plotLineDataset(edir)
        print(f"\n=== R0={R0*1000:.0f}mm  M^(1/3)={mt:.4f}  Z_max={Z_max_data:.3f} ===")

        rho_errs, P_errs, u_errs = [], [], []

        for _ in range(30):
            t, r, rho_t, u_t, P_t = ds.sample_grid_points(300)
            tau = t / mt
            Z = r / mt
            # Phase B: tau >= tau_sep, Z between contact and data boundary
            mask = ((tau.squeeze(-1) >= TAU_SEP * 1.01)
                    & (Z.squeeze(-1) >= Z_C)
                    & (Z.squeeze(-1) <= Z_max_data))
            if mask.sum() < 20:
                continue
            tau = tau[mask][:100].view(-1, 1)
            Z = Z[mask][:100].view(-1, 1)
            rho_t = rho_t[mask][:100].view(-1, 1)
            P_t = P_t[mask][:100].view(-1, 1)
            u_t = u_t[mask][:100].view(-1, 1)

            rho_p, u_p, P_p = asn_eval(tau, Z)
            rho_errs.extend(((rho_p - rho_t).abs() / rho_t.clamp(min=1)).flatten().tolist())
            P_errs.extend(((P_p - P_t).abs() / P_t.clamp(min=1)).flatten().tolist())
            u_errs.extend(((u_p - u_t).abs() / (u_t.abs() + 100).clamp(min=100)).flatten().tolist())

        n = len(rho_errs)
        rho_med = float(np.median(rho_errs))
        P_med = float(np.median(P_errs))
        u_med = float(np.median(u_errs))
        print(f"  Phase B: n={n}  median err: rho={rho_med:.3f}  P={P_med:.3f}  u={u_med:.3f}")

        evaluations.append({"R0_mm": R0*1000, "n": n, "rho_med": rho_med, "P_med": P_med, "u_med": u_med})

    print("\n=== Summary ===")
    for e in evaluations:
        print(f"  R0={e['R0_mm']:.0f}mm: rho_err={e['rho_med']:.3f}  P_err={e['P_med']:.3f}  u_err={e['u_med']:.3f}")


if __name__ == "__main__":
    main()
