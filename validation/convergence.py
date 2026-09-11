"""Convergence diagnostics for the spherical two-network PINN.

Reports after Phase A and Phase B training:

* Phase A (DetonationNet)
    - (rho, u, P) error vs d3plot inside the triangle
    - Sec.4.2.1 gate: DetonationNet(t_sep, R_c) vs analytical (P_x, u_x, rho_x)
* Phase B (AirShockNet)
    - (rho, u, P) error vs d3plot on the rectangle
    - Connection-point coupling error: AirShockNet(t_sep, R_c) vs frozen
      DetonationNet(t_sep, R_c)
    - L_RH residual at the shock front (already non-dimensional)

Usage
-----
    python -m validation.convergence --config configs/tnt_spherical_50mm.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from pinn.losses.air_shock_loss import AirShockRHLoss
from pinn.losses.detonation_loss import _interp_rc
from pinn.networks import build_networks
from pinn.checkpoints import load_checkpoint
from pinn.coupling import air_contact_density


def _load_cfg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ckpt(cfg: dict, name: str) -> Path:
    return ROOT / cfg["training"]["checkpoint_dir"] / f"{name}.pt"


def _load_into(net, path: Path):
    return load_checkpoint(path, net)[0]


def _make_sep_bundle(dataset: D3plotLineDataset):
    if dataset.R_0 is None:
        raise RuntimeError(
            f"Dataset {dataset.dir} is missing the R_0 metadata field; "
            f"re-run the extractor."
        )
    tnt = TNTParams(rho_TNT=dataset.rho_TNT)
    cj_b = compute_cj_state(tnt)
    return compute_separation_state(tnt, R_0=float(dataset.R_0), cj_bundle=cj_b)


def run_convergence(config_path: str = "configs/tnt_spherical_50mm.yaml") -> bool:
    cfg = _load_cfg(config_path)
    air = cfg["air"]
    val = cfg["validation"]

    extracted_dir = cfg["data"]["extracted_dir"]
    if not Path(extracted_dir).is_absolute():
        extracted_dir = ROOT / extracted_dir
    print(f"[load] dataset {extracted_dir}")
    dataset = D3plotLineDataset(extracted_dir)
    sep = _make_sep_bundle(dataset)
    cj  = sep.cj_bundle
    print(f"[sep ] R_0={sep.R_0*1e3:.2f} mm  R_c={sep.R_c*1e3:.2f} mm  "
          f"t_sep={sep.t_sep*1e6:.2f} us")

    # Build networks with the same scales used in training
    det, asn = build_networks(
        cfg["networks"],
        rho_ref_A=cj.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj.P_CJ,
        rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
    )

    det_ckpt = _ckpt(cfg, "detonation")
    asn_ckpt = _ckpt(cfg, "air_shock")
    if not det_ckpt.exists():
        print(f"DetonationNet checkpoint not found: {det_ckpt}")
        return False
    det_eval = _load_into(det, det_ckpt)
    results: list[tuple[str, bool, str]] = []

    # This is a model/data diagnostic at a fixed analytical anchor.
    # It must not be reported as learned endpoint convergence.
    with torch.no_grad():
        t_p = torch.tensor([[sep.t_sep]], dtype=torch.float32)
        r_p = torch.tensor([[sep.R_c]],   dtype=torch.float32)
        rho_p, u_p, P_p = det_eval(t_p, r_p)
        rho_d, u_d, P_d = dataset.state_at(t_p, r_p)
    rho_err = abs(float(rho_p) - float(rho_d)) / max(abs(float(rho_d)), 1e-10)
    u_err   = abs(float(u_p)   - float(u_d))   / max(abs(float(u_d)), 1.0)
    P_err   = abs(float(P_p)   - float(P_d))   / max(abs(float(P_d)), 1e-10)
    gate_max = max(u_err, P_err)
    gate_tol = val.get("separation_gate_tol", 0.08)
    results.append(("Phase A endpoint/data agreement (fixed anchor; u/P)",
                    gate_max <= gate_tol,
                    f"max err {100*gate_max:.2f}% (tol {100*gate_tol:.0f}%); "
                    f"rho {100*rho_err:.1f}% diag"))

    # ---- Phase A interior (sample inside fan)
    if dataset.t_traj_rc is not None and hasattr(sep, 't_traj_tail'):
        n_samples = 1024
        kept = []
        attempts = 0
        t_tl = torch.as_tensor(sep.t_traj_tail, dtype=torch.float32)
        r_tl = torch.as_tensor(sep.r_traj_tail, dtype=torch.float32)
        while sum(t.numel() for t, *_ in kept) < n_samples and attempts < 8:
            t, x, rho, u, P = dataset.sample_grid_points(n_samples)
            r_c = dataset.r_c_taylor_sadovsky(t)
            r_tail = _interp_rc(t, t_tl, r_tl)
            mask = ((t.squeeze(-1) <= sep.t_sep)
                    & (x.squeeze(-1) <= r_c.squeeze(-1))
                    & (x.squeeze(-1) >= r_tail.squeeze(-1)))
            if mask.any():
                kept.append((t[mask], x[mask], rho[mask], u[mask], P[mask]))
            attempts += 1
        if kept:
            t   = torch.cat([k[0] for k in kept])[:n_samples].view(-1, 1)
            r   = torch.cat([k[1] for k in kept])[:n_samples].view(-1, 1)
            rho_t = torch.cat([k[2] for k in kept])[:n_samples].view(-1, 1)
            P_t   = torch.cat([k[4] for k in kept])[:n_samples].view(-1, 1)
            with torch.no_grad():
                rho_pA, _, P_pA = det_eval(t, r)
            rho_med = float(((rho_pA - rho_t).abs() / rho_t.clamp(min=1e-10)).median())
            P_med   = float(((P_pA   - P_t  ).abs() / P_t  .clamp(min=1e-10)).median())
            results.append(("Phase A median rel err (rho, P)", rho_med < 0.1 and P_med < 0.1,
                            f"rho {100*rho_med:.2f}%, P {100*P_med:.2f}%"))

    # ---- Phase B (if AirShockNet checkpoint exists)
    if asn_ckpt.exists():
        asn_eval = _load_into(asn, asn_ckpt)
        # Coupling consistency at (t_sep, R_c)
        with torch.no_grad():
            rho_a_pred, u_a_pred, P_a_pred = asn_eval(t_p, r_p)
        _r = float(rho_p); _u = float(u_p); _P = float(P_p)
        _ra = float(rho_a_pred); _ua = float(u_a_pred); _Pa = float(P_a_pred)
        couple_err = max(
            abs(_ua - _u) / max(abs(_u), 1.0),
            abs(_Pa - _P) / max(_P, 1e-10),
        )
        results.append(("Connection-point u/P continuity",
                        couple_err < 0.01,
                        f"max err {100*couple_err:.2f}% (target < 1%)"))

        rho_air_target = float(air_contact_density(P_p, gamma=air["gamma"],
                                                   rho_a=air["rho_a"], P_a=air["P_a"]))
        rho_air_err = abs(_ra - rho_air_target) / max(rho_air_target, 1e-10)
        results.append(("Initial air-side density vs RH (density may jump)",
                        rho_air_err < 0.01,
                        f"air={_ra:.4g}, RH={rho_air_target:.4g}, products={_r:.4g} kg/m^3"))

        # Phase B interior d3plot error
        n_samples = 1024
        kept = []
        attempts = 0
        while sum(t.numel() for t, *_ in kept) < n_samples and attempts < 8:
            t, x, rho, u, P = dataset.sample_grid_points(n_samples)
            mask = (t.squeeze(-1) >= sep.t_sep) & (x.squeeze(-1) >= sep.R_c)
            if mask.any():
                kept.append((t[mask], x[mask], rho[mask], u[mask], P[mask]))
            attempts += 1
        if kept:
            t   = torch.cat([k[0] for k in kept])[:n_samples].view(-1, 1)
            r   = torch.cat([k[1] for k in kept])[:n_samples].view(-1, 1)
            rho_t = torch.cat([k[2] for k in kept])[:n_samples].view(-1, 1)
            P_t   = torch.cat([k[4] for k in kept])[:n_samples].view(-1, 1)
            with torch.no_grad():
                rho_pB, _, P_pB = asn_eval(t, r)
            rho_med = float(((rho_pB - rho_t).abs() / rho_t.clamp(min=1e-10)).median())
            P_med   = float(((P_pB   - P_t  ).abs() / P_t  .clamp(min=1e-10)).median())
            results.append(("Phase B median rel err (rho, P)", rho_med < 0.1 and P_med < 0.1,
                            f"rho {100*rho_med:.2f}%, P {100*P_med:.2f}%"))

        # L_RH residual
        rh_loss = AirShockRHLoss(
            asn_eval, dataset, t_sep=sep.t_sep, t_end=dataset.t_end,
            gamma=air["gamma"], rho_a=air["rho_a"], P_a=air["P_a"],
            eps=cfg["domain"]["rh_eps"], N_s=256,
            u_ref=cfg["domain"]["u_ref_B"],
        )
        rh_val = rh_loss().detach().cpu().item()
        results.append(("L_RH,B residual", rh_val < 1e-2,
                        f"{rh_val:.4e} (target < 1e-2)"))
    else:
        print(f"[note] AirShockNet checkpoint not found ({asn_ckpt}); skip Phase B checks.")

    # ---- Print summary
    print("\n=== Convergence Summary ===")
    n_pass = 0
    for name, ok, info in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: {info}")
        n_pass += int(ok)
    print(f"\nResult: {n_pass}/{len(results)} checks passed.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tnt_spherical_50mm.yaml")
    args = parser.parse_args()
    ok = run_convergence(args.config)
    sys.exit(0 if ok else 1)
