"""Quick unified model evaluation across all three radii."""
from __future__ import annotations
import sys, math, yaml, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from data.multi_radius_dataset import _M_third
from pinn.networks import build_networks, HardDetNetConstraint, HardContactConstrainedASN
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

    det.load_state_dict(torch.load(
        ROOT / "checkpoints_unified/detonation.pt", map_location="cpu", weights_only=False,
    )["state_dict"])
    # Use best checkpoint for AirShockNet
    asn.load_state_dict(torch.load(
        ROOT / "checkpoints_unified/air_shock_best.pt", map_location="cpu", weights_only=False,
    )["state_dict"])
    det.eval(); asn.eval()

    sep0 = compute_separation_state(tnt, R_0=0.05, cj_bundle=cj)
    sm = cfg["sampling"]

    det_eval = HardDetNetConstraint(
        det,
        tau_sep=1.112e-5, Z_c=0.117576, Z_R0=0.052712,
        rho_cj=cj.rho_CJ, u_cj=0.0, P_cj=cj.P_CJ,
        rho_x=sep0.rho_x, u_x=sep0.u_x, P_x=sep0.P_x,
        tau_t=float(sm.get("tau_t", 1e-6)),
        tau_r=float(sm.get("tau_r", 1e-3)),
        use_contact=False,   # matches new training (analytic contact anchor disabled)
    )

    with torch.no_grad():
        tc = torch.tensor([[1.112e-5]], dtype=torch.float32)
        rc = torch.tensor([[0.117576]], dtype=torch.float32)
        rho_c, u_c, P_c = det_eval(tc, rc)

    asn_eval = HardContactConstrainedASN(
        asn, tau_sep=1.112e-5, Z_c=0.117576,
        target_rho=rho_c, target_u=u_c, target_P=P_c,
        tau_t=float(sm.get("tau_t", 1e-6)),
        tau_r=float(sm.get("tau_r", 1e-3)),
    )

    # Gate check
    print(f"Gate: rho={float(rho_c):.1f}  u={float(u_c):.0f}  P={float(P_c)/1e6:.2f} MPa")
    print(f"      vs analytical rho={sep0.rho_x:.1f}  u={sep0.u_x:.0f}  P={sep0.P_x/1e6:.2f} MPa")

    evaluations: list[dict] = []

    TAU_SEP = 1.112e-5     # scaled separation time (constant across radii)
    Z_C = 0.117576          # scaled contact radius

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
