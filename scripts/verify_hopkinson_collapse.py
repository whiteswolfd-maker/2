#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hopkinson-Cranz collapse check across charge radii.

The whole multi-radius dataset design rests on one premise: under the scaled
coordinates

    tau = t / M^(1/3)          Z = r / M^(1/3)          M^(1/3) = alpha * R_0

the continuum solution (rho, u, P) is IDENTICAL for every charge radius.  This
script verifies that premise against the extracted LS-DYNA data, using the
SAME bilinear interpolation the PINN dataset uses (state_at), so "perfect"
here means "the network will see one self-consistent answer at every (tau, Z)".

Checks
------
1. Field collapse  : on a common (tau, Z) grid, the relative deviation of
                     rho / u / P between each radius and the reference radius.
2. Shock trajectory: scaled R_s(tau) vs the reference (used by the unified
                     model for the RH loss).
3. Peak overpressure: Delta-P(Z) vs the reference (the standard blast metric).

A dataset with strictly scaled mesh & domain (the Hopkinson twin produced by
``scripts/gen_tnt_spherical_deck.py``) passes all three within a few percent.

Usage
-----
    python scripts/verify_hopkinson_collapse.py \
        extracted/ 0.05 extracted_100mm_twin/ 0.10

Exit code 0 if collapse within tolerance, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.d3plot_dataset import D3plotLineDataset
from physics.cj_state import TNTParams

ALPHA = (4.0 / 3.0 * np.pi * TNTParams().rho_TNT) ** (1.0 / 3.0)  # ~18.97


def _load(spec: tuple[str, float]) -> tuple[D3plotLineDataset, float]:
    extracted, R_0 = spec
    ds = D3plotLineDataset(extracted)
    m3 = ALPHA * float(R_0)
    return ds, m3


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a-b|/|b| with a floor so ambient-tiny values don't dominate."""
    denom = np.abs(b)
    denom = np.maximum(denom, 1e-6 * np.max(denom) if denom.size else 1.0)
    return np.abs(a - b) / denom


def field_collapse(specs: list[tuple[str, float]], n_tau: int = 400,
                   n_z: int = 300, tol_pct: float = 5.0,
                   tau_min: float = 6e-6, tau_max: float = 1.04e-3,
                   z_lo: float = 0.13) -> dict:
    """Compare rho/u/P at a common scaled grid (via dataset.state_at)."""
    ds, m3 = _load(specs[0])
    ref_tau_end = ds.t_end / m3
    ref_z_max = ds.x_end / m3

    tau_max = min(tau_max, ref_tau_end)
    # Common Z window: above the contact surface of every radius, below the
    # smallest radius's domain edge.
    z_hi = min(ds.x_end / m3 for ds, m3 in (_load(s) for s in specs))
    z_lo = max(z_lo, max(float(ds.R_c_sep) / m3 for ds, _ in (_load(s) for s in specs)))
    z_hi = min(z_hi, float(ref_z_max))
    if z_lo >= z_hi:
        print(f"[collapse] WARNING empty common Z window [{z_lo:.3f}, {z_hi:.3f}]")
        z_hi = z_lo + 0.01

    rng = np.random.default_rng(0)
    tau_g = np.sort(rng.uniform(tau_min, tau_max, n_tau))
    z_g = np.linspace(z_lo, z_hi, n_z)

    TT, ZZ = np.meshgrid(tau_g, z_g, indexing="ij")  # (n_tau, n_z)
    fields = {}
    for label, spec in enumerate(specs):
        ds, m3 = _load(spec)
        t = torch.as_tensor(TT.reshape(-1, 1) * m3, dtype=torch.float32)
        r = torch.as_tensor(ZZ.reshape(-1, 1) * m3, dtype=torch.float32)
        rho, u, P = ds.state_at(t, r)
        fields[label] = {
            "rho": rho.numpy().reshape(TT.shape),
            "u": u.numpy().reshape(TT.shape),
            "P": P.numpy().reshape(TT.shape),
        }
        print(f"[collapse] {label}: {spec[0]}  tau in "
              f"[{tau_min*1e6:.1f}, {ref_tau_end*1e6:.1f}] us/kg^(1/3), "
              f"Z in [{z_lo:.3f}, {z_hi:.3f}]  (grid {n_tau}x{n_z})")

    results = {}
    for k in range(1, len(specs)):
        per = {}
        for f in ("rho", "u", "P"):
            rel = _rel(fields[k][f], fields[0][f])
            # skip the near-ambient tail (both radii at ambient -> ratio noise)
            strong = fields[0][f] > 0.02 * fields[0][f].max()
            s = np.sort(rel[strong])
            per[f] = {
                "median_pct": float(np.median(s)) * 100.0 if s.size else 0.0,
                "p90_pct": float(np.percentile(s, 90)) * 100.0 if s.size else 0.0,
                "max_pct": float(s[-1]) * 100.0 if s.size else 0.0,
            }
        results[k] = per
        med = max(per[f]["median_pct"] for f in per)
        ok = all(per[f]["p90_pct"] <= tol_pct for f in per)
        print(f"\n[collapse] radius #{k} vs #0 (scaled field medians, "
              f"cells with P>2% of peak):")
        for f in per:
            print(f"    {f:>4}: median {per[f]['median_pct']:.2f}%   "
                  f"p90 {per[f]['p90_pct']:.2f}%   max {per[f]['max_pct']:.2f}%")
        print(f"    {'PASS' if ok else 'FAIL'}  (p90 must be <= {tol_pct}%)")
    return results


def shock_collapse(specs: list[tuple[str, float]], tol_pct: float = 5.0) -> None:
    """Compare scaled shock trajectory R_s(tau) to the reference radius."""
    ref_ds, ref_m3 = _load(specs[0])
    tau_ref = ref_ds.t_traj.numpy() / ref_m3
    z_ref = ref_ds.R_s_traj.numpy() / ref_m3
    valid = (z_ref > 0) & (tau_ref > 0) & np.isfinite(z_ref)
    tau_ref, z_ref = tau_ref[valid], z_ref[valid]
    if tau_ref.size < 3:
        print("[collapse] reference shock trajectory too short; skipped")
        return

    print(f"\n[shock] scaled R_s(tau) — reference #{0} ({specs[0][0]})")
    for k in range(1, len(specs)):
        ds, m3 = _load(specs[k])
        tau_k = ds.t_traj.numpy() / m3
        z_k = ds.R_s_traj.numpy() / m3
        ok_ = (z_k > 0) & (tau_k > 0) & np.isfinite(z_k)
        zk = np.interp(tau_ref, tau_k[ok_], z_k[ok_], left=np.nan, right=np.nan)
        m = np.isfinite(zk) & (z_ref > 0)
        if m.sum() > 3:
            rel = np.abs(zk[m] - z_ref[m]) / z_ref[m] * 100.0
            print(f"    #{k}: median {np.median(rel):.2f}%  "
                  f"max {rel.max():.2f}%  ({m.sum()} pts)  "
                  f"{'PASS' if np.median(rel) <= tol_pct else 'FAIL'}")
        else:
            print(f"    #{k}: no overlapping trajectory points")


def peak_overpressure_collapse(specs: list[tuple[str, float]],
                               tol_pct: float = 5.0) -> None:
    """Compare peak incident overpressure Delta-P(Z) to the reference radius."""
    P_a = 101325.0
    ds0, m3_0 = _load(specs[0])
    t0 = ds0.t_arr.numpy()
    x0 = ds0.x_arr.numpy()
    t_sep0 = ds0.t_sep
    Rc0 = float(ds0.R_c_sep)
    # common scaled-distance window
    z_hi = min(ds.x_end / m3 for ds, m3 in (_load(s) for s in specs))
    z_lo = max(float(ds.R_c_sep) / m3 for ds, m3 in (_load(s) for s in specs))
    if z_lo >= z_hi:
        print("[overpressure] empty Z window; skipped")
        return

    Z = np.linspace(z_lo, z_hi, 50)
    peak = {}
    for k, spec in enumerate(specs):
        ds, m3 = _load(spec)
        t = ds.t_arr.numpy()
        x = ds.x_arr.numpy()
        t_mask = t >= ds.t_sep
        vals = np.zeros_like(Z)
        for i, z in enumerate(Z):
            r = z * m3
            r_idx = int(np.argmin(np.abs(x - r)))
            Pcol = ds.P_grid.numpy()[:, r_idx]
            vals[i] = Pcol[t_mask].max() if t_mask.any() else np.nan
        peak[k] = vals
        if k == 0:
            dP0 = (peak[0] - P_a)
        print(f"[overpressure] #{k}: Z in [{z_lo:.3f}, {z_hi:.3f}] "
              f"peak dP {((vals - P_a) * 1e-6).max():.3f} MPa @ "
              f"Z={Z[np.nanargmax(vals - P_a)]:.3f}")

    print(f"\n[overpressure] Delta-P(Z) vs reference (#0):")
    for k in range(1, len(specs)):
        rel = np.abs(peak[k] - peak[0]) / np.maximum(np.abs(peak[0] - P_a), 1e-6)
        rel = rel[np.isfinite(rel)]
        print(f"    #{k}: median {np.median(rel)*100:.2f}%  "
              f"max {rel.max()*100:.2f}%  "
              f"{'PASS' if np.median(rel)*100 <= tol_pct else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("radii", nargs="+",
                    help="pairs: <extracted_dir> <R_0_m> ...  (first is reference)")
    ap.add_argument("--tol-pct", type=float, default=5.0,
                    help="median-field / peak p90 tolerance (%%), default 5")
    args = ap.parse_args()

    if len(args.radii) < 4 or len(args.radii) % 2:
        print("usage: verify_hopkinson_collapse.py dir1 R01 dir2 R02 ...")
        return 2
    specs = [(args.radii[i], float(args.radii[i + 1]))
             for i in range(0, len(args.radii), 2)]

    results = field_collapse(specs, tol_pct=args.tol_pct)
    shock_collapse(specs, tol_pct=args.tol_pct)
    peak_overpressure_collapse(specs, tol_pct=args.tol_pct)

    ok = all(all(r["p90_pct"] <= args.tol_pct for r in per.values())
             for per in results.values())
    print(f"\n=== Hopkinson-Cranz collapse: "
          f"{'PASS (data self-consistent for merged NN)' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
