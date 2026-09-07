"""Compare PINN & d3plot vs Henrych empirical blast-wave formula.

Henrych, J. "The Dynamics of Explosion and Its Use." Elsevier, 1979.
Peak overpressure ΔP (MPa) as function of scaled distance Z = R / M^(1/3).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from data.d3plot_dataset import D3plotLineDataset
from pinn.networks import HardContactConstrainedASN, build_networks

# ── Henrych peak overpressure (returns Pa) ───────────────────
def henrych_dP(Z):
    """Peak incident overpressure in Pa. Henrych (1979), spherical TNT free air."""
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
    return dP_bar * 1e5  # bar → Pa


# ── Load ─────────────────────────────────────────────────────
cfg_path = ROOT / "configs" / "tnt_spherical_50mm.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
air = cfg["air"]; P_a = air["P_a"]
dataset = D3plotLineDataset(str(ROOT / cfg["data"]["extracted_dir"]))
M_TNT = float(dataset.M_TNT)
M_third = M_TNT ** (1.0 / 3.0)
print(f"M_TNT = {M_TNT:.3f} kg  M^(1/3) = {M_third:.4f} kg^(1/3)")

det, asn = build_networks(
    cfg["networks"],
    rho_ref_A=1630.0, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=21.0e9,
    rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
)
det.load_state_dict(torch.load(ROOT / cfg["training"]["checkpoint_dir"] / "detonation.pt",
                     map_location="cpu", weights_only=False)["state_dict"]); det.eval()
asn.load_state_dict(torch.load(ROOT / cfg["training"]["checkpoint_dir"] / "air_shock.pt",
                     map_location="cpu", weights_only=False)["state_dict"]); asn.eval()

hc_meta = ROOT / cfg["training"]["checkpoint_dir"] / "air_shock_hc_meta.pt"
asn_eval = asn
if hc_meta.exists():
    hc = torch.load(hc_meta, map_location="cpu", weights_only=False)
    asn_eval = HardContactConstrainedASN(
        asn, tau_sep=float(hc["t_sep"]), Z_c=float(hc["R_c"]),
        target_rho=hc["target_rho"], target_u=hc["target_u"], target_P=hc["target_P"],
        tau_t=float(hc["tau_t"]), tau_r=float(hc["tau_r"]),
    )

R_c = float(dataset.R_c_sep); x_end = dataset.x_end; t_sep = dataset.t_sep; t_end = dataset.t_end
t_arr = dataset.t_arr.numpy()
x_arr = dataset.x_arr.numpy()
R_s_traj = dataset.R_s_traj.numpy()
t_traj = dataset.t_traj.numpy()

# ── 1) Shock-front pressure P(R_s + 1mm) vs scaled distance ──
# This is the direct Henrych comparison: overpressure at the shock front.
print(f"\n{'='*80}")
print(f"Shock-front overpressure: d3plot vs PINN vs Henrych")
print(f"{'t (us)':>8s} {'R_s (mm)':>10s} {'Z':>8s} {'d3plot ΔP':>12s} {'PINN ΔP':>12s} {'Henrych ΔP':>12s} {'PINN err':>10s} {'d3plot err':>10s}")
print("-" * 90)

t_samples = np.linspace(max(t_sep + 2e-6, 1.5e-5), t_end * 0.95, 12)
for ti in t_samples:
    t_t = torch.tensor([[ti]], dtype=torch.float32)
    R_s_i = float(dataset.R_s(t_t))
    eps_r = min(2e-3, (x_end - R_s_i) * 0.9)  # 2mm behind shock
    if eps_r <= 0:
        continue
    r_eval = torch.tensor([[R_s_i + eps_r]], dtype=torch.float32)
    t_eval = torch.tensor([[ti]], dtype=torch.float32)
    _, _, P_shock_d3p = dataset.state_at(t_eval, r_eval)
    with torch.no_grad():
        _, _, P_shock_pinn = asn_eval(t_eval, r_eval)
    Z_i = R_s_i / M_third
    dP_hen = henrych_dP(np.array([Z_i]))[0]  # Pa, overpressure only

    P_d3 = float(P_shock_d3p); P_pi = float(P_shock_pinn)
    dP_d3 = P_d3 - P_a  # overpressure
    dP_pi = P_pi - P_a
    err_pi = abs(dP_pi - P_d3) / max(P_d3, 1e-10) * 100
    err_d3 = abs(dP_d3 - dP_hen) / max(dP_hen, 1e-10) * 100
    print(f"{ti*1e6:8.1f} {R_s_i*1e3:10.2f} {Z_i:8.3f} {dP_d3*1e-6:11.4f}  {dP_pi*1e-6:11.4f}  {dP_hen*1e-6:11.4f}  {err_pi:9.1f}% {err_d3:9.1f}%")

# ── 2) Peak overpressure vs radius (post-contact only) ───────
# For each radius > R_c, find max P over t where R_s(t) > r (i.e., shock has passed)
print(f"\n{'='*80}")
print(f"Peak pressure at each radius (shock has passed): d3plot vs PINN vs Henrych")
print(f"{'r (mm)':>8s} {'Z':>8s} {'d3plot P_peak':>15s} {'PINN P_peak':>15s} {'Henrych P_abs':>15s} {'PINN/d3plot':>12s} {'d3plot/Hen':>12s}")
print("-" * 95)

r_milestones = np.linspace(R_c + 5e-3, x_end, 20)
for r_val in r_milestones:
    Z_i = r_val / M_third
    # d3plot peak: max pressure at this r across all t > t_sep where R_s(t) > r
    r_idx = np.argmin(np.abs(x_arr - r_val))
    r_actual = x_arr[r_idx]

    # Find times where shock has passed this radius
    shock_passed = R_s_traj > r_actual
    if not shock_passed.any():
        continue
    t_valid = t_traj[shock_passed]
    t_mask = t_arr >= t_sep
    t_valid_grid = np.intersect1d(t_arr[t_mask], t_valid)  # this won't work, let me just sample

    # Simplified: scan all t >= t_sep, find max P at this r
    P_col = dataset.P_grid[:, r_idx].numpy()
    P_col = P_col[t_mask]
    P_peak_d3 = P_col.max()

    # PINN peak
    with torch.no_grad():
        t_batch = torch.tensor(t_arr[t_mask].reshape(-1, 1), dtype=torch.float32)
        r_batch = torch.full((len(t_batch), 1), r_actual, dtype=torch.float32)
        _, _, P_p = asn_eval(t_batch, r_batch)
    P_peak_pi = float(P_p.max())

    dP_hen = henrych_dP(np.array([Z_i]))[0]
    P_hen_abs = dP_hen + P_a

    err_pi = abs(P_peak_pi - P_peak_d3) / max(P_peak_d3, 1e-10) * 100
    err_d3 = abs(P_peak_d3 - P_hen_abs) / max(P_hen_abs, 1e-10) * 100
    print(f"{r_actual*1e3:8.1f} {Z_i:8.3f} {P_peak_d3*1e-6:14.4f}  {P_peak_pi*1e-6:14.4f}  {P_hen_abs*1e-6:14.4f}  {err_pi:11.1f}% {err_d3:11.1f}%")

# ── 3) Summary statistics ────────────────────────────────────
r_phaseB = x_arr[x_arr >= R_c]
t_phaseB = t_arr[t_arr >= t_sep]
n_r = len(r_phaseB)

# d3plot peak overpressure (excluding region R < R_c + 10mm where products contaminate)
far_start = R_c + 0.01
far_mask = r_phaseB >= far_start
r_far = r_phaseB[far_mask]

P_peak_d3_arr = np.zeros(len(r_phaseB))
P_peak_pi_arr = np.zeros(len(r_phaseB))
for i in range(len(r_phaseB)):
    r_idx = np.argmin(np.abs(x_arr - r_phaseB[i]))
    P_col = dataset.P_grid[:, r_idx].numpy()
    P_col = P_col[t_arr >= t_sep]
    P_peak_d3_arr[i] = P_col.max()

# PINN peak at far radii (cheaper: sample 100 radii evenly)
for i in np.linspace(0, len(r_phaseB)-1, 100).astype(int):
    r_val = r_phaseB[i]
    r_idx = np.argmin(np.abs(x_arr - r_val))
    with torch.no_grad():
        t_batch = torch.tensor(t_phaseB.reshape(-1, 1), dtype=torch.float32)
        r_batch = torch.full((len(t_batch), 1), r_val, dtype=torch.float32)
        _, _, P_p = asn_eval(t_batch, r_batch)
    P_peak_pi_arr[i] = float(P_p.max())

Z_far = r_far / M_third
P_hen_far = henrych_dP(Z_far) + P_a

rel_err_pi = np.abs(P_peak_pi_arr[far_mask][P_peak_pi_arr[far_mask] > 0] - P_peak_d3_arr[far_mask][P_peak_pi_arr[far_mask] > 0]) / np.maximum(P_peak_d3_arr[far_mask][P_peak_pi_arr[far_mask] > 0], 1e-10)
rel_err_d3 = np.abs(P_peak_d3_arr[far_mask] - P_hen_far) / np.maximum(P_hen_far, 1e-10)
rel_err_pi_hen = np.abs(P_peak_pi_arr[far_mask][P_peak_pi_arr[far_mask] > 0] - P_hen_far[P_peak_pi_arr[far_mask] > 0]) / np.maximum(P_hen_far[P_peak_pi_arr[far_mask] > 0], 1e-10)

print(f"\n{'='*80}")
print(f"Summary (r > R_c + 10mm, excluding product-contaminated near-field)")
print(f"  PINN vs d3plot peak P:")
print(f"    Median: {np.median(rel_err_pi)*100:.1f}%")
print(f"    Mean:   {np.mean(rel_err_pi)*100:.1f}%")
print(f"  d3plot vs Henrych peak P:")
print(f"    Median: {np.median(rel_err_d3)*100:.1f}%")
print(f"    Mean:   {np.mean(rel_err_d3)*100:.1f}%")
print(f"  PINN vs Henrych peak P:")
print(f"    Median: {np.median(rel_err_pi_hen)*100:.1f}%")
print(f"    Mean:   {np.mean(rel_err_pi_hen)*100:.1f}%")
