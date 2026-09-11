#!/usr/bin/env python
"""Multi-radius TNT blast-wave PINN: train 50/80/100 mm & generate comparison plot.

Trains AirShockNet for all three charge radii, then produces publication-quality
comparison figures:

1. Peak overpressure Delta-P vs scaled distance Z = R / M^(1/3) (log-log)
   — PINN predictions, d3plot data, and Henrych (1979) empirical formula.
2. Shock trajectory R_s(t) — dimensional + scaled by R_0.
3. Training loss curves per radius.
4. PINN vs d3plot parity scatter (rho, u, P).
5. Error distribution histograms.

Usage:
    python scripts/multi_radius_train.py          # train + plot all
    python scripts/multi_radius_train.py --plot-only  # use existing checkpoints
"""
from __future__ import annotations
import sys, os, json, subprocess, argparse
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Radii to process ───────────────────────────────────────────
RADII = {
    50:  {"config": "configs/tnt_spherical_50mm.yaml",  "extracted": "extracted/",
          "checkpoint": "checkpoints", "R_0": 0.050, "M_TNT": 0.8535, "M_third": 0.9486},
    80:  {"config": "configs/tnt_spherical_80mm.yaml",  "extracted": "extracted_80mm/",
          "checkpoint": "checkpoints_80mm", "R_0": 0.080, "M_TNT": 3.496, "M_third": 1.518},
    100: {"config": "configs/tnt_spherical_100mm.yaml", "extracted": "extracted_100mm/",
          "checkpoint": "checkpoints_100mm", "R_0": 0.100, "M_TNT": 6.828, "M_third": 1.897},
}

# ── Henrych (1979) peak overpressure (Pa) ──────────────────────
def henrych_dP(Z):
    Z = np.asarray(Z, dtype=float)
    dP_bar = np.zeros_like(Z)
    m1 = Z < 0.3
    if m1.any():
        z = Z[m1]
        dP_bar[m1] = 1.4072/z + 0.554/z**2 - 0.0357/z**3 + 0.000625/z**4
    m2 = (Z >= 0.3) & (Z <= 1.0)
    if m2.any():
        z = Z[m2]
        dP_bar[m2] = 0.61938/z - 0.03262/z**2 + 0.21324/z**3
    m3 = Z > 1.0
    if m3.any():
        z = Z[m3]
        dP_bar[m3] = 0.0662/z + 0.405/z**2 + 0.3288/z**3
    return dP_bar * 1e5  # bar -> Pa

# ── Plot settings ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 10, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
COLORS = {50: "#2166AC", 80: "#D6604D", 100: "#4DAF4A"}
MARKERS = {50: "o", 80: "s", 100: "^"}

OUTDIR = ROOT / "reports" / "multi_radius"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════
def train_one(config_path: str, label: str):
    """Run `python -m pinn.trainer --all --config <config_path>`."""
    print(f"\n{'='*70}\nTraining {label}\n{'='*70}")
    result = subprocess.run(
        [sys.executable, "-m", "pinn.trainer", "--all", "--config", str(config_path)],
        cwd=str(ROOT), capture_output=False,
    )
    if result.returncode != 0:
        print(f"[WARN] Training {label} exited with code {result.returncode}")
    return result.returncode


def train_all():
    for r, info in RADII.items():
        cfg = ROOT / info["config"]
        train_one(str(cfg), f"R0={r}mm")


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════
def load_model_and_data(r_mm: int):
    """Load trained AirShockNet + dataset for one radius."""
    from data.d3plot_dataset import D3plotLineDataset
    from pinn.networks import build_networks
    from pinn.checkpoints import load_checkpoint

    info = RADII[r_mm]
    cfg_path = ROOT / info["config"]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    air = cfg["air"]
    dataset = D3plotLineDataset(str(ROOT / info["extracted"]))

    det, asn = build_networks(
        cfg["networks"],
        rho_ref_A=1630.0, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=21.0e9,
        rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
    )

    ckpt_dir = ROOT / info["checkpoint"]
    det_pt = ckpt_dir / "detonation.pt"
    asn_pt = ckpt_dir / "air_shock.pt"
    if det_pt.exists():
        det, _ = load_checkpoint(det_pt, det)
    if not asn_pt.exists():
        raise FileNotFoundError(asn_pt)
    asn_eval, _ = load_checkpoint(asn_pt, asn)

    return dataset, asn_eval, cfg, air


# ═══════════════════════════════════════════════════════════════
#  Figure 1 — Peak overpressure vs scaled distance (log-log)
# ═══════════════════════════════════════════════════════════════
def fig_peak_overpressure(models: dict):
    """Henrych-style Delta-P(Z) plot with PINN & d3plot data points."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Henrych reference curve
    Z_hen = np.logspace(-0.6, 1.2, 300)
    dP_hen = henrych_dP(Z_hen) * 1e-6  # MPa
    ax.loglog(Z_hen, dP_hen, 'k-', linewidth=1.8, label="Henrych (1979)", zorder=2)

    # TM5-1300 reference (approximate)
    Z_tm5 = np.logspace(-0.5, 1.2, 200)
    dP_tm5 = 0.096/Z_tm5 + 0.144/Z_tm5**2 + 0.574/Z_tm5**3  # MPa, spherical free-air
    ax.loglog(Z_tm5, dP_tm5, 'k--', linewidth=1.2, alpha=0.5, label="TM5-1300 (approx)", zorder=1)

    for r_mm, (dataset, asn_eval, cfg, air) in models.items():
        info = RADII[r_mm]
        color = COLORS[r_mm]
        marker = MARKERS[r_mm]
        P_a = air["P_a"]

        t_arr = dataset.t_arr.numpy()
        x_arr = dataset.x_arr.numpy()
        t_sep = dataset.t_sep
        R_c_sep = float(dataset.R_c_sep)
        M_third = info["M_third"]

        # Sample radii > R_c + 10mm (exclude product-contaminated near-field)
        r_sample = np.linspace(R_c_sep + 0.01, x_arr[-1], 40)
        Z_sample = r_sample / M_third

        # d3plot peak P at each radius
        P_peak_d3 = np.zeros(len(r_sample))
        P_peak_pi = np.zeros(len(r_sample))
        t_mask = t_arr >= t_sep

        for i, r_val in enumerate(r_sample):
            r_idx = np.argmin(np.abs(x_arr - r_val))
            # d3plot peak
            P_col = dataset.P_grid[:, r_idx].numpy()
            P_peak_d3[i] = P_col[t_mask].max() if t_mask.any() else np.nan

            # PINN peak
            with torch.no_grad():
                t_batch = torch.tensor(t_arr[t_mask].reshape(-1, 1), dtype=torch.float32)
                r_batch = torch.full((len(t_batch), 1), r_val, dtype=torch.float32)
                _, _, P_p = asn_eval(t_batch, r_batch)
            P_peak_pi[i] = float(P_p.max())

        dP_d3 = (P_peak_d3 - P_a) * 1e-6  # MPa
        dP_pi = (P_peak_pi - P_a) * 1e-6

        ax.loglog(Z_sample, dP_d3, marker, color=color, markersize=5, alpha=0.6,
                  label=f"d3plot R$_0$={r_mm}mm", markerfacecolor="none", zorder=3)
        ax.loglog(Z_sample, dP_pi, marker, color=color, markersize=5,
                  label=f"PINN R$_0$={r_mm}mm", zorder=4)

    ax.set_xlabel("Scaled distance  $Z = R / M^{1/3}$  (m/kg$^{1/3}$)")
    ax.set_ylabel("Peak overpressure  $\\Delta P$  (MPa)")
    ax.set_title("Peak incident overpressure vs scaled distance")
    ax.legend(loc="upper right", ncol=2, framealpha=0.85)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.15, 20)
    ax.set_ylim(5e-3, 50)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_peak_overpressure.png", dpi=200)
    fig.savefig(OUTDIR / "fig1_peak_overpressure.pdf")
    plt.close(fig)
    print("[plot] fig1_peak_overpressure saved")


# ═══════════════════════════════════════════════════════════════
#  Figure 2 — Shock trajectory R_s(t)
# ═══════════════════════════════════════════════════════════════
def fig_shock_trajectory(models: dict):
    """Shock front position vs time — dimensional + scaled."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for r_mm, (dataset, asn_eval, cfg, air) in models.items():
        info = RADII[r_mm]
        color = COLORS[r_mm]
        R_0 = info["R_0"] * 1000  # mm

        R_s_traj = dataset.R_s_traj.numpy()
        t_traj = dataset.t_traj.numpy()
        valid = R_s_traj > 0

        # Dimensional
        ax1.plot(t_traj[valid] * 1e6, R_s_traj[valid] * 1e3, color=color, linewidth=1.2,
                 label=f"R$_0$={r_mm}mm")
        # Scaled
        ax2.plot(t_traj[valid] * 1e6, R_s_traj[valid] / (info["R_0"]), color=color, linewidth=1.2,
                 label=f"R$_0$={r_mm}mm")

    ax1.set_xlabel("Time  $t$  ($\\mu$s)")
    ax1.set_ylabel("Shock radius  $R_s$  (mm)")
    ax1.set_title("Shock front trajectory")
    ax1.legend(); ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Time  $t$  ($\\mu$s)")
    ax2.set_ylabel("$R_s / R_0$")
    ax2.set_title("Scaled shock trajectory")
    ax2.legend(); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_shock_trajectory.png", dpi=200)
    fig.savefig(OUTDIR / "fig2_shock_trajectory.pdf")
    plt.close(fig)
    print("[plot] fig2_shock_trajectory saved")


# ═══════════════════════════════════════════════════════════════
#  Figure 3 — PINN vs d3plot parity scatter (Phase B only)
# ═══════════════════════════════════════════════════════════════
def fig_parity(models: dict):
    """Parity scatter: PINN prediction vs d3plot for rho, u, P in Phase B."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    quantities = ["rho", "u", "P"]
    units = ["kg/m$^3$", "m/s", "MPa"]
    scales = [1.0, 1.0, 1e-6]

    for col, (r_mm, (dataset, asn_eval, cfg, air)) in enumerate(models.items()):
        info = RADII[r_mm]
        color = COLORS[r_mm]
        t_arr = dataset.t_arr.numpy()
        x_arr = dataset.x_arr.numpy()
        t_sep = dataset.t_sep

        # Sample Phase B grid (subsample for speed)
        t_mask = t_arr >= t_sep
        t_idx = np.where(t_mask)[0]
        # subsample ~500 points
        stride = max(1, len(t_idx) // 500)
        t_sample_idx = t_idx[::stride]
        x_sample_idx = np.linspace(0, len(x_arr)-1, min(50, len(x_arr))).astype(int)

        d3p_vals = {q: [] for q in quantities}
        pinn_vals = {q: [] for q in quantities}

        for ti in t_sample_idx:
            t_t = torch.tensor([[t_arr[ti]]], dtype=torch.float32)
            for xi in x_sample_idx:
                r_t = torch.tensor([[x_arr[xi]]], dtype=torch.float32)
                rho_d, u_d, P_d = dataset.state_at(t_t, r_t)
                with torch.no_grad():
                    _, _, _ = asn_eval(t_t, r_t)  # placeholder
                    rho_p, u_p, P_p = asn_eval(t_t, r_t)

                d3p_vals["rho"].append(float(rho_d))
                d3p_vals["u"].append(float(u_d))
                d3p_vals["P"].append(float(P_d))
                pinn_vals["rho"].append(float(rho_p))
                pinn_vals["u"].append(float(u_p))
                pinn_vals["P"].append(float(P_p))

        for row, q in enumerate(quantities):
            ax = axes[row, col]
            d3 = np.array(d3p_vals[q]) * scales[row]
            pi = np.array(pinn_vals[q]) * scales[row]
            ax.scatter(d3, pi, s=0.8, alpha=0.3, color=color, rasterized=True)
            vmin = min(d3.min(), pi.min())
            vmax = max(d3.max(), pi.max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k-', linewidth=0.5)
            ax.set_xlabel(f"d3plot {quantities[row]} ({units[row]})", fontsize=9)
            ax.set_ylabel(f"PINN {quantities[row]} ({units[row]})", fontsize=9)
            ax.set_title(f"R$_0$={r_mm}mm — {quantities[row]}", fontsize=10)
            # R²
            ss_res = np.sum((d3 - pi)**2)
            ss_tot = np.sum((d3 - d3.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            ax.text(0.95, 0.05, f"$R^2={r2:.4f}$", transform=ax.transAxes,
                    fontsize=8, ha="right")
            ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_parity.png", dpi=200)
    fig.savefig(OUTDIR / "fig3_parity.pdf")
    plt.close(fig)
    print("[plot] fig3_parity saved")


# ═══════════════════════════════════════════════════════════════
#  Figure 4 — Loss curves (from training logs)
# ═══════════════════════════════════════════════════════════════
def fig_loss_curves():
    """Parse training_final.log-style files for loss histories."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    log_files = {
        50: ROOT / "training_final.log",
        80: ROOT / "training_80mm.log",
        100: ROOT / "training_100mm.log",
    }

    for idx, (r_mm, log_path) in enumerate(log_files.items()):
        ax = axes[idx]
        color = COLORS[r_mm]
        if not log_path.exists():
            ax.text(0.5, 0.5, f"no log for R$_0$={r_mm}mm", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"R$_0$={r_mm}mm")
            continue

        # Parse log: look for lines with "Step" and "Loss"
        stages = defaultdict(lambda: {"steps": [], "losses": []})
        current_stage = "unknown"
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Detect stage transitions
                for stage_kw in ["A1", "A2", "B1", "B2a", "B2b", "B2c", "Gate"]:
                    if f"[{stage_kw}]" in line or f"Phase {stage_kw}" in line or f"--- {stage_kw}" in line:
                        current_stage = stage_kw
                # Parse loss entries
                if "Step" in line and ("Loss" in line or "loss" in line):
                    parts = line.split()
                    try:
                        step_idx = parts.index("Step") + 1
                        loss_idx = parts.index("Loss") + 1 if "Loss" in parts else None
                        if loss_idx is None:
                            loss_idx = [i for i, p in enumerate(parts) if "loss" in p.lower()][0] + 1
                        step = int(parts[step_idx].strip(","))
                        loss = float(parts[loss_idx])
                        stages[current_stage]["steps"].append(step)
                        stages[current_stage]["losses"].append(loss)
                    except (ValueError, IndexError):
                        pass

        for stage, data in stages.items():
            if data["steps"]:
                ax.semilogy(data["steps"], data["losses"], linewidth=0.8,
                            label=stage, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"R$_0$={r_mm}mm")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Training loss curves", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_loss_curves.png", dpi=200)
    fig.savefig(OUTDIR / "fig4_loss_curves.pdf")
    plt.close(fig)
    print("[plot] fig4_loss_curves saved")


# ═══════════════════════════════════════════════════════════════
#  Figure 5 — Error histograms (Phase B, per radius)
# ═══════════════════════════════════════════════════════════════
def fig_error_histograms(models: dict):
    """Relative error distributions for rho, P in Phase B."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    quantities = ["rho", "P"]
    units = ["kg/m$^3$", "Pa"]
    titles = ["Density  $\\rho$", "Pressure  $P$"]

    for col, (r_mm, (dataset, asn_eval, cfg, air)) in enumerate(models.items()):
        info = RADII[r_mm]
        color = COLORS[r_mm]
        t_arr = dataset.t_arr.numpy()
        x_arr = dataset.x_arr.numpy()
        t_mask = t_arr >= dataset.t_sep
        t_idx = np.where(t_mask)[0]
        stride = max(1, len(t_idx) // 300)
        t_sample_idx = t_idx[::stride]
        x_sample_idx = np.linspace(0, len(x_arr)-1, min(40, len(x_arr))).astype(int)

        errors = {"rho": [], "P": []}
        for ti in t_sample_idx:
            t_t = torch.tensor([[t_arr[ti]]], dtype=torch.float32)
            for xi in x_sample_idx:
                r_t = torch.tensor([[x_arr[xi]]], dtype=torch.float32)
                rho_d, u_d, P_d = dataset.state_at(t_t, r_t)
                with torch.no_grad():
                    rho_p, u_p, P_p = asn_eval(t_t, r_t)
                if rho_d > 0.01:
                    errors["rho"].append((float(rho_p) - float(rho_d)) / float(rho_d) * 100)
                if P_d > 100:
                    errors["P"].append((float(P_p) - float(P_d)) / float(P_d) * 100)

        for row, q in enumerate(quantities):
            ax = axes[row, col]
            err = np.array(errors[q])
            if len(err) > 0:
                ax.hist(err, bins=60, color=color, alpha=0.7, density=True,
                        range=(-50, 50) if q == "rho" else (-30, 30))
                ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--')
                ax.axvline(x=np.median(err), color='k', linewidth=1.0,
                           label=f"median={np.median(err):+.1f}%")
                ax.legend(fontsize=8)
            ax.set_xlabel(f"Relative error (%)")
            ax.set_ylabel("Density" if not col else "")
            ax.set_title(f"R$_0$={r_mm}mm — {titles[row]}", fontsize=10)
            ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig5_error_histograms.png", dpi=200)
    fig.savefig(OUTDIR / "fig5_error_histograms.pdf")
    plt.close(fig)
    print("[plot] fig5_error_histograms saved")


# ═══════════════════════════════════════════════════════════════
#  Figure 6 — Summary: similarity-law collapse
# ═══════════════════════════════════════════════════════════════
def fig_similarity_collapse(models: dict):
    """Demonstrate similarity collapse: scaled quantities vs scaled distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    Z_hen = np.logspace(-0.6, 1.2, 300)
    dP_hen = henrych_dP(Z_hen) * 1e-6
    ax1.loglog(Z_hen, dP_hen, 'k-', linewidth=1.8, label="Henrych (1979)", zorder=1)

    for r_mm, (dataset, asn_eval, cfg, air) in models.items():
        info = RADII[r_mm]
        color = COLORS[r_mm]
        marker = MARKERS[r_mm]
        P_a = air["P_a"]
        M_third = info["M_third"]
        R_c_sep = float(dataset.R_c_sep)
        t_arr = dataset.t_arr.numpy()
        x_arr = dataset.x_arr.numpy()
        t_mask = t_arr >= dataset.t_sep

        r_sample = np.linspace(R_c_sep + 0.01, x_arr[-1], 30)
        Z_sample = r_sample / M_third
        P_peak_d3 = np.zeros(len(r_sample))
        P_peak_pi = np.zeros(len(r_sample))

        for i, r_val in enumerate(r_sample):
            r_idx = np.argmin(np.abs(x_arr - r_val))
            P_col = dataset.P_grid[:, r_idx].numpy()
            P_peak_d3[i] = P_col[t_mask].max() if t_mask.any() else np.nan
            with torch.no_grad():
                t_b = torch.tensor(t_arr[t_mask].reshape(-1, 1), dtype=torch.float32)
                r_b = torch.full((len(t_b), 1), r_val, dtype=torch.float32)
                _, _, P_p = asn_eval(t_b, r_b)
            P_peak_pi[i] = float(P_p.max())

        dP_pi = (P_peak_pi - P_a) * 1e-6
        ax1.loglog(Z_sample, dP_pi, marker, color=color, markersize=5,
                   label=f"PINN R$_0$={r_mm}mm", zorder=3)

        # Shock trajectory scaled
        R_s = dataset.R_s_traj.numpy()
        t_t = dataset.t_traj.numpy()
        valid = (R_s > 0) & (t_t > 0)
        t_scaled = t_t[valid] * 1e6 / info["M_third"]
        R_scaled = R_s[valid] / info["M_third"]
        ax2.plot(t_scaled, R_scaled, color=color, linewidth=1.0,
                 label=f"R$_0$={r_mm}mm")

    ax1.set_xlabel("$Z = R / M^{1/3}$  (m/kg$^{1/3}$)")
    ax1.set_ylabel("$\\Delta P$  (MPa)")
    ax1.set_title("Peak overpressure — similarity collapse")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Scaled time  $t / M^{1/3}$  ($\\mu$s/kg$^{1/3}$)")
    ax2.set_ylabel("$R_s / M^{1/3}$  (m/kg$^{1/3}$)")
    ax2.set_title("Shock trajectory — similarity collapse")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig6_similarity_collapse.png", dpi=200)
    fig.savefig(OUTDIR / "fig6_similarity_collapse.pdf")
    plt.close(fig)
    print("[plot] fig6_similarity_collapse saved")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Multi-radius PINN training & plotting")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, use existing checkpoints")
    parser.add_argument("--train-only", action="store_true",
                        help="Train only, skip plotting")
    parser.add_argument("--radii", type=str, default="50,80,100",
                        help="Comma-separated radii to process (default: 50,80,100)")
    args = parser.parse_args()

    radii_list = [int(r) for r in args.radii.split(",")]
    selected = {r: RADII[r] for r in radii_list if r in RADII}
    if not selected:
        print("No valid radii selected."); return 1

    if not args.plot_only:
        for r_mm in radii_list:
            cfg = ROOT / RADII[r_mm]["config"]
            if not cfg.exists():
                print(f"[SKIP] Config not found: {cfg}")
                continue
            train_one(str(cfg), f"R0={r_mm}mm")

    if args.train_only:
        return 0

    # Load models
    print("\nLoading trained models...")
    models = {}
    for r_mm in radii_list:
        try:
            ds, asn_eval, cfg, air = load_model_and_data(r_mm)
            models[r_mm] = (ds, asn_eval, cfg, air)
            print(f"  R0={r_mm}mm: loaded ({cfg['data']['extracted_dir']})")
        except Exception as e:
            print(f"  R0={r_mm}mm: FAILED — {e}")

    if not models:
        print("No models loaded."); return 1

    print("\nGenerating figures...")
    fig_peak_overpressure(models)
    fig_shock_trajectory(models)
    fig_parity(models)
    fig_loss_curves()
    fig_error_histograms(models)
    fig_similarity_collapse(models)

    print(f"\nDone. Figures saved to: {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
