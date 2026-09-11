#!/usr/bin/env python
"""Multi-radius PINN comparison plots (paper-quality).

Generates:
  fig1: Peak overpressure vs scaled distance (Hopkinson-Cranz collapse)
  fig2: Separation analysis — analytical vs VF R_c
  fig3: PINN vs d3plot parity scatter (Phase B)
  fig4: Pressure profiles at selected times
  fig5: Error comparison table
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import torch, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from pinn.networks import build_networks
from pinn.checkpoints import load_checkpoint

# ── Config ─────────────────────────────────────────────────────
RADII = {
    50:  {"config": "configs/tnt_spherical_50mm.yaml",  "R_0": 0.050, "M_third": 0.9486, "label": "50 mm"},
    80:  {"config": "configs/tnt_spherical_80mm.yaml",  "R_0": 0.080, "M_third": 1.5177, "label": "80 mm"},
    100: {"config": "configs/tnt_spherical_100mm.yaml", "R_0": 0.100, "M_third": 1.8971, "label": "100 mm"},
}
COLORS = {50: "#2166AC", 80: "#D6604D", 100: "#4DAF4A"}
MARKERS = {50: "o", 80: "s", 100: "^"}
OUTDIR = ROOT / "reports" / "multi_radius"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

# ── Henrych ─────────────────────────────────────────────────────
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
    return dP_bar * 1e5

# ── Load helpers ────────────────────────────────────────────────
def load_model(r_mm):
    info = RADII[r_mm]
    cfg_path = ROOT / info["config"]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    air = cfg["air"]
    dataset = D3plotLineDataset(str(ROOT / cfg["data"]["extracted_dir"]))

    det, asn = build_networks(
        cfg["networks"],
        rho_ref_A=1630.0, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=21.0e9,
        rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
    )

    ckpt_dir = ROOT / cfg["training"]["checkpoint_dir"]
    det_pt = ckpt_dir / "detonation.pt"
    asn_pt = ckpt_dir / "air_shock.pt"
    if det_pt.exists():
        det, _ = load_checkpoint(det_pt, det)
    if not asn_pt.exists():
        raise FileNotFoundError(asn_pt)
    asn_eval, _ = load_checkpoint(asn_pt, asn)
    return dataset, asn_eval, cfg, air


# ═══════════════════════════════════════════════════════════════
#  FIG 1 — Peak overpressure vs scaled distance (log-log)
# ═══════════════════════════════════════════════════════════════
def fig1_overpressure(models):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    Z_hen = np.logspace(-0.6, 1.3, 300)
    dP_hen = henrych_dP(Z_hen) * 1e-6
    ax.loglog(Z_hen, dP_hen, 'k-', linewidth=1.8, label="Henrych (1979)", zorder=2)
    ax.loglog(Z_hen, dP_hen * 1.15, 'k--', linewidth=1.0, alpha=0.35, label="Henrych +15%", zorder=1)
    ax.loglog(Z_hen, dP_hen * 0.85, 'k--', linewidth=1.0, alpha=0.35, label="Henrych -15%", zorder=1)

    for r_mm, (dataset, asn_eval, cfg, air) in models.items():
        info = RADII[r_mm]
        color = COLORS[r_mm]; marker = MARKERS[r_mm]
        P_a = air["P_a"]; M_third = info["M_third"]
        t_arr = dataset.t_arr.numpy(); x_arr = dataset.x_arr.numpy()
        t_mask = t_arr >= dataset.t_sep
        R_c_sep = float(dataset.R_c_sep)

        r_sample = np.linspace(R_c_sep + 0.01, x_arr[-1], 30)
        Z_sample = r_sample / M_third
        P_peak_d3 = np.zeros(len(r_sample)); P_peak_pi = np.zeros(len(r_sample))

        for i, r_val in enumerate(r_sample):
            r_idx = np.argmin(np.abs(x_arr - r_val))
            P_col = dataset.P_grid[:, r_idx].numpy()
            P_peak_d3[i] = P_col[t_mask].max() if t_mask.any() else np.nan
            with torch.no_grad():
                t_b = torch.tensor(t_arr[t_mask].reshape(-1, 1), dtype=torch.float32)
                r_b = torch.full((len(t_b), 1), r_val, dtype=torch.float32)
                _, _, P_p = asn_eval(t_b, r_b)
            P_peak_pi[i] = float(P_p.max())

        dP_d3 = (P_peak_d3 - P_a) * 1e-6; dP_pi = (P_peak_pi - P_a) * 1e-6

        ax.loglog(Z_sample, dP_d3, marker, color=color, markersize=5, alpha=0.55,
                  markerfacecolor="none", label=f"d3plot {info['label']}", zorder=3)
        ax.loglog(Z_sample, dP_pi, marker, color=color, markersize=5,
                  label=f"PINN {info['label']}", zorder=4)

    ax.set_xlabel("Scaled distance  $Z = R / M^{1/3}$  (m / kg$^{1/3}$)")
    ax.set_ylabel("Peak overpressure  $\\Delta P$  (MPa)")
    ax.set_title("Peak incident overpressure — Hopkinson-Cranz scaling")
    ax.legend(ncol=2, framealpha=0.85, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(0.12, 25); ax.set_ylim(4e-3, 60)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_peak_overpressure.png", dpi=200)
    fig.savefig(OUTDIR / "fig1_peak_overpressure.pdf")
    plt.close(fig)
    print("[plot] fig1 saved")


# ═══════════════════════════════════════════════════════════════
#  FIG 2 — Separation analysis
# ═══════════════════════════════════════════════════════════════
def fig2_separation():
    from physics.cj_state import compute_separation_state, TNTParams
    tp = TNTParams()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    R0_arr = np.array([50, 80, 100])
    R_c_phy = np.zeros(3); t_sep_phy = np.zeros(3)
    R_c_vf = np.zeros(3); M_third_arr = np.zeros(3)

    for i, R0_mm in enumerate([50, 80, 100]):
        R0 = R0_mm / 1000.0
        result = compute_separation_state(tp, R0)
        R_c_phy[i] = result.R_c * 1000
        t_sep_phy[i] = result.t_sep * 1e6
        M = (4./3.) * np.pi * R0**3 * tp.rho_TNT
        M_third_arr[i] = M ** (1./3.)

        path = "extracted/" if R0_mm == 50 else f"extracted_{R0_mm}mm/"
        contact = np.loadtxt(f"{path}contact.csv", delimiter=",", skiprows=1)
        t_vf = contact[:, 0]; r_vf = contact[:, 1] * 1000
        idx = np.argmin(np.abs(t_vf - result.t_sep))
        R_c_vf[i] = r_vf[idx]

    # Left: R_c comparison
    ax1.plot(R0_arr, R_c_phy, 'ko-', linewidth=2, markersize=8, label="Analytical R_c")
    ax1.plot(R0_arr, R_c_vf, 'rs--', linewidth=1.5, markersize=8, label="d3plot VF R_c")
    ax1.plot(R0_arr, R0_arr, 'k:', alpha=0.3, label="R_0 (charge surface)")
    for i in range(3):
        ax1.annotate(f"{R_c_phy[i]:.0f}", (R0_arr[i], R_c_phy[i]), textcoords="offset points", xytext=(0,10), fontsize=8, ha='center')
        ax1.annotate(f"{R_c_vf[i]:.0f}", (R0_arr[i], R_c_vf[i]), textcoords="offset points", xytext=(0,-15), fontsize=8, ha='center', color='red')
    ax1.set_xlabel("Charge radius R_0 (mm)"); ax1.set_ylabel("R_c (mm)")
    ax1.set_title("Contact radius at separation"); ax1.legend(); ax1.grid(True, alpha=0.25)

    # Right: scaled separation Z_sep
    Z_sep_phy = R_c_phy / M_third_arr
    Z_sep_vf = R_c_vf / M_third_arr
    ax2.plot(R0_arr, Z_sep_phy, 'ko-', linewidth=2, markersize=8, label="Analytical Z_sep")
    ax2.plot(R0_arr, Z_sep_vf, 'rs--', linewidth=1.5, markersize=8, label="d3plot VF Z_sep")
    ax2.axhline(y=Z_sep_phy[0], color='k', linestyle=':', alpha=0.3, label=f"Z_sep = {Z_sep_phy[0]:.4f}")
    ax2.set_xlabel("Charge radius R_0 (mm)"); ax2.set_ylabel("Z_sep = R_c / M^{1/3} (m/kg^{1/3})")
    ax2.set_title("Scaled separation distance (similarity check)"); ax2.legend(); ax2.grid(True, alpha=0.25)

    fig.suptitle("Separation position: analytical vs d3plot VF detection", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig2_separation.png", dpi=200)
    fig.savefig(OUTDIR / "fig2_separation.pdf")
    plt.close(fig)
    print("[plot] fig2 saved")


# ═══════════════════════════════════════════════════════════════
#  FIG 3 — PINN vs d3plot parity (Phase B only)
# ═══════════════════════════════════════════════════════════════
def fig3_parity(models):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    quantities = ["P", "rho"]
    units = ["MPa", "kg/m$^3$"]
    scales = [1e-6, 1.0]

    for col, (r_mm, (dataset, asn_eval, cfg, air)) in enumerate(models.items()):
        info = RADII[r_mm]; color = COLORS[r_mm]
        t_arr = dataset.t_arr.numpy(); x_arr = dataset.x_arr.numpy()
        t_idx = np.where(t_arr >= dataset.t_sep)[0]
        stride = max(1, len(t_idx) // 400)
        t_sample = t_idx[::stride]
        x_sample = np.linspace(0, len(x_arr)-1, min(40, len(x_arr))).astype(int)

        d3p_vals = {"rho": [], "P": []}; pinn_vals = {"rho": [], "P": []}
        for ti in t_sample:
            t_t = torch.tensor([[t_arr[ti]]], dtype=torch.float32)
            for xi in x_sample:
                r_t = torch.tensor([[x_arr[xi]]], dtype=torch.float32)
                rho_d, u_d, P_d = dataset.state_at(t_t, r_t)
                with torch.no_grad():
                    rho_p, u_p, P_p = asn_eval(t_t, r_t)
                d3p_vals["rho"].append(float(rho_d)); d3p_vals["P"].append(float(P_d))
                pinn_vals["rho"].append(float(rho_p)); pinn_vals["P"].append(float(P_p))

        for row, q in enumerate(quantities):
            ax = axes[row, col]
            d3 = np.array(d3p_vals[q]) * scales[row]; pi = np.array(pinn_vals[q]) * scales[row]
            ax.scatter(d3, pi, s=0.8, alpha=0.25, color=color, rasterized=True)
            vmin = min(d3.min(), pi.min()); vmax = max(d3.max(), pi.max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k-', linewidth=0.5)
            ss_res = np.sum((d3 - pi)**2); ss_tot = np.sum((d3 - d3.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            ax.text(0.95, 0.05, f"$R^2={r2:.4f}$", transform=ax.transAxes, fontsize=8, ha="right")
            ax.set_xlabel(f"d3plot {q} ({units[row]})", fontsize=9)
            ax.set_ylabel(f"PINN {q} ({units[row]})", fontsize=9)
            ax.set_title(f"R$_0$={r_mm}mm — {q}", fontsize=10)
            ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig3_parity.png", dpi=200)
    fig.savefig(OUTDIR / "fig3_parity.pdf")
    plt.close(fig)
    print("[plot] fig3 saved")


# ═══════════════════════════════════════════════════════════════
#  FIG 4 — Pressure profiles at selected times
# ═══════════════════════════════════════════════════════════════
def fig4_pressure_profiles(models):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for col, (r_mm, (dataset, asn_eval, cfg, air)) in enumerate(models.items()):
        ax = axes[col]; color = COLORS[r_mm]
        t_arr = dataset.t_arr.numpy(); x_arr = dataset.x_arr.numpy()
        t_sep = dataset.t_sep; P_a = air["P_a"]

        # Select 4 representative times after t_sep
        t_mask = t_arr >= t_sep
        t_valid = t_arr[t_mask]
        if len(t_valid) > 4:
            idx = np.linspace(0, len(t_valid)-1, 4).astype(int)
            t_sample = t_valid[idx]
        else:
            t_sample = t_valid

        for ti, t_val in enumerate(t_sample):
            t_t = torch.tensor([[t_val]], dtype=torch.float32)
            # d3plot
            t_idx = np.argmin(np.abs(t_arr - t_val))
            P_d3 = dataset.P_grid[t_idx, :].numpy() * 1e-6
            # PINN
            r_batch = torch.tensor(x_arr.reshape(-1, 1), dtype=torch.float32)
            with torch.no_grad():
                _, _, P_p = asn_eval(t_t.expand(len(x_arr), 1), r_batch)
            P_pi = P_p.numpy().flatten() * 1e-6

            alpha = 0.4 + 0.6 * ti / len(t_sample)
            ax.semilogy(x_arr*1e3, P_d3, color=color, linewidth=1.0, alpha=alpha, linestyle='-')
            ax.semilogy(x_arr*1e3, P_pi, color=color, linewidth=1.0, alpha=alpha, linestyle='--')

        ax.axhline(y=P_a*1e-6, color='k', linewidth=0.5, linestyle=':', label=f"P_atm = {P_a*1e-6:.3f} MPa")
        ax.set_xlabel("r (mm)"); ax.set_ylabel("P (MPa)")
        ax.set_title(f"R$_0$={r_mm}mm — P(r) profiles"); ax.grid(True, alpha=0.2)
        # legend for line styles
        ax.plot([], [], 'k-', linewidth=1.0, label="d3plot"); ax.plot([], [], 'k--', linewidth=1.0, label="PINN")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig4_pressure_profiles.png", dpi=200)
    fig.savefig(OUTDIR / "fig4_pressure_profiles.pdf")
    plt.close(fig)
    print("[plot] fig4 saved")


# ═══════════════════════════════════════════════════════════════
#  FIG 5 — Shock trajectory R_s(t)
# ═══════════════════════════════════════════════════════════════
def fig5_shock_trajectory(models):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for r_mm, (dataset, asn_eval, cfg, air) in models.items():
        info = RADII[r_mm]; color = COLORS[r_mm]
        R_s = dataset.R_s_traj.numpy(); t_t = dataset.t_traj.numpy()
        valid = (R_s > 0) & (t_t > 0)
        # Dimensional
        ax1.plot(t_t[valid]*1e6, R_s[valid]*1e3, color=color, linewidth=1.2, label=info['label'])
        # Scaled by Hopkinson-Cranz
        ax2.plot(t_t[valid]*1e6 / info["M_third"], R_s[valid] / info["M_third"],
                 color=color, linewidth=1.2, label=info['label'])

    ax1.set_xlabel("t (us)"); ax1.set_ylabel("R_s (mm)")
    ax1.set_title("Shock front trajectory"); ax1.legend(); ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("t / M^{1/3}  (us/kg^{1/3})"); ax2.set_ylabel("R_s / M^{1/3}  (m/kg^{1/3})")
    ax2.set_title("Shock trajectory — similarity collapse"); ax2.legend(); ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig5_shock_trajectory.png", dpi=200)
    fig.savefig(OUTDIR / "fig5_shock_trajectory.pdf")
    plt.close(fig)
    print("[plot] fig5 saved")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("Loading models...")
    models = {}
    for r_mm in [50, 80, 100]:
        try:
            ds, asn_e, cfg, air = load_model(r_mm)
            models[r_mm] = (ds, asn_e, cfg, air)
            print(f"  R0={r_mm}mm: OK")
        except Exception as e:
            print(f"  R0={r_mm}mm: FAILED — {e}")

    if len(models) < 2:
        print("Need at least 2 models loaded."); return 1

    print("\nGenerating figures...")
    fig1_overpressure(models)
    fig2_separation()
    fig3_parity(models)
    fig4_pressure_profiles(models)
    fig5_shock_trajectory(models)
    print(f"\nAll figures saved to: {OUTDIR}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
