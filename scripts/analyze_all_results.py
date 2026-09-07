#!/usr/bin/env python
"""Comprehensive analysis: PINN training results vs d3plot data for all 3 radii.

Extracts training log metrics, runs convergence checks, computes per-field error
distributions, and prints a summary table.

Usage:
    python scripts/analyze_all_results.py
    python scripts/analyze_all_results.py --skip-convergence  # log parsing only
"""
from __future__ import annotations
import re, sys, os, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from pinn.networks import HardContactConstrainedASN, build_networks
from physics.cj_state import compute_separation_state, TNTParams, compute_cj_state

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------
RADII = {
    50: {
        "config": "configs/tnt_spherical_50mm.yaml",
        "extracted": "extracted",
        "checkpoint": "checkpoints",
        "R_0": 0.050,
        "log": "training_final.log",
        "log_phaseA": "training_50mm_phaseA.log",
        "log_phaseB": "training_50mm_phaseB.log",
    },
    80: {
        "config": "configs/tnt_spherical_80mm.yaml",
        "extracted": "extracted_80mm",
        "checkpoint": "checkpoints_80mm",
        "R_0": 0.080,
        "log": "training_80mm.log",
        "log_phaseA": "training_80mm_phaseA.log",
        "log_phaseB": "training_80mm_phaseB.log",
    },
    100: {
        "config": "configs/tnt_spherical_100mm.yaml",
        "extracted": "extracted_100mm",
        "checkpoint": "checkpoints_100mm",
        "R_0": 0.100,
        "log": "training_100mm.log",
        "log_phaseA": "training_100mm_phaseA.log",
        "log_phaseB": "training_100mm_phaseB.log",
    },
}

# ----------------------------------------------------------------------
#  Log parsing
# ----------------------------------------------------------------------

def parse_training_log(log_path: Path) -> dict:
    """Extract loss curves and final metrics from a training log."""
    result = {
        "steps": [], "data": [], "PDE": [], "RH": [], "BC": [], "tot": [],
        "best_step": None, "best_PDE": None,
        "final_step": None, "final_data": None, "final_PDE": None,
        "final_RH": None, "final_BC": None, "final_tot": None,
        "phase": None, "det_loss": None,
    }

    if not log_path.exists():
        return result

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Detect phase markers
    for line in lines:
        if "Phase A" in line or "Phase B" in line or "[A1]" in line or "[A2]" in line or "[B1]" in line or "[B2]" in line:
            result["phase"] = line.strip()

    # Parse per-step metrics
    step_pat = re.compile(
        r"step=\s*(\d+).*?"
        r"data(?:_[AB])?=\s*([\d\.e\+\-]+).*?"
        r"PDE(?:_[AB])?=\s*([\d\.e\+\-]+)"
    )
    rh_pat = re.compile(r"RH(?:_[AB])?=\s*([\d\.e\+\-]+)")
    bc_pat = re.compile(r"BC(?:_[AB])?.*?=\s*([\d\.e\+\-]+)")
    tot_pat = re.compile(r"tot=\s*([\d\.e\+\-]+)")

    for line in lines:
        m = step_pat.search(line)
        if m:
            step = int(m.group(1))
            data_val = float(m.group(2))
            pde_val = float(m.group(3))
            result["steps"].append(step)
            result["data"].append(data_val)
            result["PDE"].append(pde_val)

            rh_m = rh_pat.search(line)
            result["RH"].append(float(rh_m.group(1)) if rh_m else None)

            bc_m = bc_pat.search(line)
            result["BC"].append(float(bc_m.group(1)) if bc_m else None)

            tot_m = tot_pat.search(line)
            result["tot"].append(float(tot_m.group(1)) if tot_m else None)

    # Best PDE
    best_pat = re.compile(r"\[best\].*step\s+(\d+).*PDE(?:_[AB])?=\s*([\d\.e\+\-]+)")
    for line in lines:
        m = best_pat.search(line)
        if m:
            result["best_step"] = int(m.group(1))
            result["best_PDE"] = float(m.group(2))

    # Final values (last parsed step)
    if result["steps"]:
        result["final_step"] = result["steps"][-1]
        result["final_data"] = result["data"][-1]
        result["final_PDE"] = result["PDE"][-1]
        result["final_RH"] = result["RH"][-1] if any(v is not None for v in result["RH"]) else None
        result["final_BC"] = result["BC"][-1] if any(v is not None for v in result["BC"]) else None
        result["final_tot"] = result["tot"][-1] if any(v is not None for v in result["tot"]) else None

    return result


def parse_phase_log(log_path: Path) -> dict:
    """Extract final loss from phase-specific training log."""
    result = {"final_data": None, "final_PDE": None, "final_tot": None,
              "final_RH": None, "final_BC": None, "steps": 0}
    if not log_path.exists():
        return result

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    pat = re.compile(
        r"step=\s*(\d+).*?"
        r"data(?:_[AB])?=\s*([\d\.e\+\-]+).*?"
        r"PDE(?:_[AB])?=\s*([\d\.e\+\-]+)"
    )
    rh_pat = re.compile(r"RH(?:_[AB])?=\s*([\d\.e\+\-]+)")
    bc_pat = re.compile(r"BC(?:_[AB])?.*?=\s*([\d\.e\+\-]+)")
    tot_pat = re.compile(r"tot=\s*([\d\.e\+\-]+)")

    for line in lines:
        m = pat.search(line)
        if m:
            result["steps"] = int(m.group(1))
            result["final_data"] = float(m.group(2))
            result["final_PDE"] = float(m.group(3))
            rh_m = rh_pat.search(line)
            if rh_m:
                result["final_RH"] = float(rh_m.group(1))
            bc_m = bc_pat.search(line)
            if bc_m:
                result["final_BC"] = float(bc_m.group(1))
            tot_m = tot_pat.search(line)
            if tot_m:
                result["final_tot"] = float(tot_m.group(1))

    return result


# ----------------------------------------------------------------------
#  PINN vs d3plot error analysis
# ----------------------------------------------------------------------

def compute_field_errors(dataset, asn_eval, t_min, t_max, r_min, r_max,
                         n_samples=2000) -> dict:
    """Compute per-field errors (rho, u, P) of PINN vs d3plot in a domain window."""
    import random
    random.seed(42)
    torch.manual_seed(42)

    t_arr = dataset.t_arr.numpy()
    x_arr = dataset.x_arr.numpy()
    t_mask = (t_arr >= t_min) & (t_arr <= t_max)
    x_mask = (x_arr >= r_min) & (x_arr <= r_max)
    t_idx = np.where(t_mask)[0]
    x_idx = np.where(x_mask)[0]

    if len(t_idx) < 2 or len(x_idx) < 2:
        return {"rho_mae": None, "u_mae": None, "P_mae": None,
                "rho_med": None, "u_med": None, "P_med": None,
                "n_points": 0}

    # Sample grid points
    n_actual = min(n_samples, len(t_idx) * len(x_idx))
    pairs = []
    for _ in range(n_actual * 3):  # oversample
        ti = np.random.choice(t_idx)
        xi = np.random.choice(x_idx)
        pairs.append((float(t_arr[ti]), float(x_arr[xi])))
        if len(pairs) >= n_actual:
            break

    t_vals = np.array([p[0] for p in pairs])
    r_vals = np.array([p[1] for p in pairs])

    t_t = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
    r_t = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)

    # d3plot ground truth
    rho_d3, u_d3, P_d3 = dataset.state_at(t_t, r_t)

    # PINN prediction
    with torch.no_grad():
        rho_p, u_p, P_p = asn_eval(t_t, r_t)

    rho_d3 = rho_d3.numpy().flatten()
    rho_p  = rho_p.numpy().flatten()
    u_d3   = u_d3.numpy().flatten()
    u_p    = u_p.numpy().flatten()
    P_d3   = P_d3.numpy().flatten()
    P_p    = P_p.numpy().flatten()

    # Clip extreme outliers (> 1e12) -- likely data artifacts
    valid = (np.abs(P_d3) < 1e12) & (np.abs(P_p) < 1e12) & (np.abs(rho_d3) < 1e6)
    rho_d3 = rho_d3[valid]; rho_p = rho_p[valid]
    u_d3 = u_d3[valid]; u_p = u_p[valid]
    P_d3 = P_d3[valid]; P_p = P_p[valid]

    def _metrics(true, pred):
        mask = np.isfinite(true) & np.isfinite(pred) & (np.abs(true) > 1e-12)
        t = true[mask]; p = pred[mask]
        if len(t) < 10:
            return {"mae": None, "med_rel": None, "r2": None, "n": len(t)}
        abs_err = np.abs(p - t)
        rel_err = abs_err / np.abs(t)
        mae = float(np.mean(abs_err))
        med_rel = float(np.median(rel_err))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
        return {"mae": mae, "med_rel": med_rel, "r2": r2, "n": len(t)}

    return {
        "rho": _metrics(rho_d3, rho_p),
        "u": _metrics(u_d3, u_p),
        "P": _metrics(P_d3, P_p),
        "n_points": len(valid),
    }


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-convergence", action="store_true",
                        help="Skip PINN-vs-d3plot field error computation (slow)")
    args = parser.parse_args()

    print("=" * 90)
    print("  PINN Blast-Wave Solver -- Comprehensive Training Analysis")
    print("  50mm / 80mm / 100mm TNT spherical charge")
    print("=" * 90)

    # -- Part 1: Training log metrics ----------------------------
    print("\n" + "-" * 90)
    print("  PART 1: Training Log Metrics")
    print("-" * 90)

    for r_mm, info in RADII.items():
        print(f"\n  * R_0 = {r_mm} mm  (M_TNT ~ {(4/3)*np.pi*info['R_0']**3*1630:.3f} kg)")
        print(f"    Config: {info['config']}")

        # Phase A log
        pa = parse_phase_log(ROOT / info["log_phaseA"])
        print(f"    Phase A (DetonationNet):  {pa['steps']} steps")
        if pa["final_data"] is not None:
            print(f"      L_data = {pa['final_data']:.4e}  "
                  f"L_PDE = {pa['final_PDE']:.4e}  "
                  f"L_tot = {pa['final_tot']:.4e}")

        # Phase B log
        pb = parse_phase_log(ROOT / info["log_phaseB"])
        print(f"    Phase B (AirShockNet):    {pb['steps']} steps")
        if pb["final_data"] is not None:
            print(f"      L_data = {pb['final_data']:.4e}  "
                  f"L_PDE = {pb['final_PDE']:.4e}  "
                  f"L_RH  = {pb['final_RH']:.4e}  "
                  f"L_BC  = {pb['final_BC']:.4e}  "
                  f"L_tot = {pb['final_tot']:.4e}")

        # Full log (best PDE step)
        full = parse_training_log(ROOT / info["log"])
        if full["best_step"] is not None:
            print(f"    Best PDE at step {full['best_step']}: L_PDE = {full['best_PDE']:.4e}")

    # -- Part 2: Separation state & gate -------------------------
    print("\n" + "-" * 90)
    print("  PART 2: Separation State (Sec.4.2.1 Gate)")
    print("-" * 90)

    from physics.cj_state import compute_separation_state, TNTParams
    tp = TNTParams()
    cj_b = compute_cj_state(tp)

    print(f"\n  {'R_0':>6s}  {'R_c':>8s}  {'t_sep':>8s}  {'P_x':>9s}  {'u_x':>8s}  {'rho_x':>8s}  {'V_x/V_0':>8s}")
    print(f"  {'(mm)':>6s}  {'(mm)':>8s}  {'(us)':>8s}  {'(GPa)':>9s}  {'(m/s)':>8s}  {'(kg/m3)':>8s}  {'':>8s}")
    print("  " + "-" * 72)

    for r_mm, info in RADII.items():
        sep = compute_separation_state(tp, R_0=info["R_0"])
        print(f"  {r_mm:>4d}   {sep.R_c*1e3:>7.2f}  {sep.t_sep*1e6:>7.2f}  "
              f"{sep.P_x*1e-9:>8.4f}  {sep.u_x:>7.0f}  {sep.rho_x:>7.1f}  {sep.V_x:>7.2f}")

    # Check VF (d3plot) contact radius at t_sep
    print(f"\n  Contact radius comparison (analytical vs d3plot VF):")
    print(f"  {'R_0':>6s}  {'R_c_ana':>9s}  {'R_c_VF':>9s}  {'Delta':>8s}  {'Delta%':>8s}")
    print(f"  {'(mm)':>6s}  {'(mm)':>9s}  {'(mm)':>9s}  {'(mm)':>8s}  {'':>8s}")
    for r_mm, info in RADII.items():
        sep = compute_separation_state(tp, R_0=info["R_0"])
        contact_path = ROOT / info["extracted"] / "contact.csv"
        if contact_path.exists():
            contact = np.loadtxt(str(contact_path), delimiter=",", skiprows=1)
            t_vf = contact[:, 0]; r_vf = contact[:, 1]
            idx = np.argmin(np.abs(t_vf - sep.t_sep))
            R_c_vf = r_vf[idx] * 1000  # m -> mm
            delta = R_c_vf - sep.R_c * 1000
            delta_pct = 100 * delta / (sep.R_c * 1000)
            print(f"  {r_mm:>4d}   {sep.R_c*1e3:>7.2f}   {R_c_vf:>7.2f}   {delta:>+6.2f}  {delta_pct:>+6.1f}%")

    # -- Part 3: PINN vs d3plot per-field errors -----------------
    if not args.skip_convergence:
        print("\n" + "-" * 90)
        print("  PART 3: PINN vs d3plot -- Per-Field Error Analysis")
        print("-" * 90)

        for r_mm, info in RADII.items():
            print(f"\n  * R_0 = {r_mm} mm")
            cfg_path = ROOT / info["config"]
            with open(cfg_path, encoding="utf-8", errors="replace") as f:
                cfg = yaml.safe_load(f)
            air = cfg["air"]

            try:
                dataset = D3plotLineDataset(str(ROOT / info["extracted"]))
            except Exception as e:
                print(f"    [SKIP] Cannot load dataset: {e}")
                continue

            # Load model
            det, asn = build_networks(
                cfg["networks"],
                rho_ref_A=cj_b.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj_b.P_CJ,
                rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
            )

            ckpt_dir = ROOT / info["checkpoint"]
            asn_pt = ckpt_dir / "air_shock.pt"
            hc_pt = ckpt_dir / "air_shock_hc_meta.pt"
            asn_eval = asn

            if not asn_pt.exists():
                print(f"    [SKIP] No checkpoint at {asn_pt}")
                continue

            asn.load_state_dict(torch.load(str(asn_pt), map_location="cpu", weights_only=False)["state_dict"])
            asn.eval()

            if hc_pt.exists():
                hc = torch.load(str(hc_pt), map_location="cpu", weights_only=False)
                asn_eval = HardContactConstrainedASN(
                    asn, tau_sep=float(hc["t_sep"]), Z_c=float(hc["R_c"]),
                    target_rho=hc["target_rho"], target_u=hc["target_u"],
                    target_P=hc["target_P"],
                    tau_t=float(hc["tau_t"]), tau_r=float(hc["tau_r"]),
                )

            t_sep = dataset.t_sep
            t_end = dataset.t_end
            R_c = dataset.R_c_sep
            x_end = dataset.x_end

            # Phase A domain: [0, t_sep]  x [0, R_c] — use DetonationNet
            det_pt = ckpt_dir / "detonation.pt"
            det_loaded = False
            if det_pt.exists():
                det.load_state_dict(torch.load(str(det_pt), map_location="cpu", weights_only=False)["state_dict"])
                det.eval()
                det_loaded = True
                errors_A = compute_field_errors(dataset, det,
                                                t_min=0, t_max=t_sep,
                                                r_min=0, r_max=R_c,
                                                n_samples=1500)
            else:
                print(f"    [SKIP] No detonation checkpoint for Phase A")
                errors_A = {"n_points": 0}

            # Phase B domain: [t_sep, t_end]  x [R_c, x_end] — use AirShockNet
            errors_B = compute_field_errors(dataset, asn_eval,
                                            t_min=t_sep, t_max=t_end,
                                            r_min=R_c, r_max=x_end,
                                            n_samples=1500)

            # Print tables
            for phase, errs in [("Phase A (detonation domain, DetNet raw)", errors_A),
                                ("Phase B (air shock domain, ASN+HC)", errors_B)]:
                print(f"\n    {phase} -- {errs.get('n_points', 0)} points")
                if errs.get("rho", {}).get("mae") is not None:
                    hdr = "      {:>6s}  {:>12s}  {:>12s}  {:>8s}  {:>6s}"
                    sep_fmt = "      {:>6s}  {:>12s}  {:>12s}  {:>8s}  {:>6s}"
                    print(hdr.format("Field", "MAE", "Med Rel Err", "R2", "N"))
                    print(sep_fmt.format("-"*6, "-"*12, "-"*12, "-"*8, "-"*6))
                    for fld in ["rho", "u", "P"]:
                        m = errs.get(fld, {})
                        if m.get("mae") is not None:
                            print(f"      {fld:>6s}  {m['mae']:>12.4e}  {100*m['med_rel']:>11.2f}%  "
                                  f"{m['r2']:>8.4f}  {m['n']:>6d}")
                else:
                    print(f"      (insufficient data)")

            # Analytical gate check (DetNet vs Sec.4.2.1 analytical solution)
            if det_loaded:
                sep_ref = compute_separation_state(tp, R_0=info["R_0"])
                print(f"\n    Sec.4.2.1 Gate: DetNet at (t_sep, R_c) vs analytical:")
                with torch.no_grad():
                    t_p = torch.tensor([[float(t_sep)]], dtype=torch.float32)
                    r_p = torch.tensor([[float(R_c)]], dtype=torch.float32)
                    rho_det, u_det, P_det = det(t_p, r_p)
                rho_err = abs(float(rho_det) - sep_ref.rho_x) / sep_ref.rho_x
                u_err   = abs(float(u_det) - sep_ref.u_x) / max(abs(sep_ref.u_x), 1.0)
                P_err   = abs(float(P_det) - sep_ref.P_x) / sep_ref.P_x
                gate_max = max(rho_err, u_err, P_err)
                gate_status = "[PASS]" if gate_max < 0.05 else "[FAIL]"
                print(f"      rho: {100*rho_err:.1f}%  u: {100*u_err:.1f}%  P: {100*P_err:.1f}%")
                print(f"      Max error = {100*gate_max:.1f}%  {gate_status} (target < 5%)")

            # Coupling consistency (DetNet raw vs ASN at (t_sep, R_c))
            if det_loaded:
                print(f"\n    Coupling consistency at (t_sep, R_c) [DetNet raw vs ASN+HC]:")
                with torch.no_grad():
                    t_p = torch.tensor([[float(t_sep)]], dtype=torch.float32)
                    r_p = torch.tensor([[float(R_c)]], dtype=torch.float32)
                    rho_det, u_det, P_det = det(t_p, r_p)
                    rho_asn, u_asn, P_asn = asn_eval(t_p, r_p)
                errors = {
                    "rho": abs(float(rho_asn) - float(rho_det)) / max(float(rho_det), 1e-10),
                    "u":   abs(float(u_asn) - float(u_det)) / max(abs(float(u_det)), 1.0),
                    "P":   abs(float(P_asn) - float(P_det)) / max(float(P_det), 1e-10),
                }
                print(f"      Deltarho/rho = {100*errors['rho']:.3f}%  "
                      f"Deltau/u = {100*errors['u']:.3f}%  "
                      f"DeltaP/P = {100*errors['P']:.3f}%")
                max_err = max(errors.values())
                cs_status = "[PASS]" if max_err < 0.01 else "[INFO]"
                print(f"      Max error = {100*max_err:.3f}%  {cs_status} (target < 1%; note: raw DetNet vs ASN+HC, expect mismatch)")
            else:
                print(f"      [SKIP] No detonation checkpoint")

    # -- Part 4: Data quality overview ---------------------------
    print("\n" + "-" * 90)
    print("  PART 4: Dataset Overview")
    print("-" * 90)

    print(f"\n  {'R_0':>6s}  {'N_t':>6s}  {'N_x':>6s}  {'t_end':>10s}  {'x_end':>10s}  {'Grid size':>12s}")
    print(f"  {'(mm)':>6s}  {'':>6s}  {'':>6s}  {'(us)':>10s}  {'(m)':>10s}  {'':>12s}")
    print("  " + "-" * 60)
    for r_mm, info in RADII.items():
        try:
            ds = D3plotLineDataset(str(ROOT / info["extracted"]))
            print(f"  {r_mm:>4d}   {ds.N_t:>5d}  {ds.N_x:>5d}  "
                  f"{ds.t_end*1e6:>9.2f}  {ds.x_end:>9.3f}  "
                  f"{ds.N_t} x{ds.N_x} = {ds.N_t*ds.N_x}")
        except Exception as e:
            print(f"  {r_mm:>4d}   {'ERROR':>5s}  {e}")

    # Data range sanity
    print(f"\n  Data range sanity (rho ∈ [rho_a, rho_max] kg/m3, P ∈ [P_a, P_max] MPa):")
    for r_mm, info in RADII.items():
        try:
            ds = D3plotLineDataset(str(ROOT / info["extracted"]))
            rho_arr = ds.rho_grid.numpy()
            P_arr = ds.P_grid.numpy()
            print(f"    R_0={r_mm}mm: rho ∈ [{rho_arr.min():.1f}, {rho_arr.max():.1f}] kg/m3  "
                  f"P ∈ [{P_arr.min()*1e-6:.3f}, {P_arr.max()*1e-6:.1f}] MPa  "
                  f"u ∈ [{ds.u_grid.numpy().min():.0f}, {ds.u_grid.numpy().max():.0f}] m/s")
        except Exception as e:
            print(f"    R_0={r_mm}mm: ERROR -- {e}")

    # -- Summary ------------------------------------------------
    print("\n" + "=" * 90)
    print("  Analysis complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
