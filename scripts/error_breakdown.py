"""Break down Phase B PINN error by spatial region: post-shock, shock-front, ambient."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import yaml

from data.d3plot_dataset import D3plotLineDataset
from pinn.networks import HardContactConstrainedASN, build_networks
from physics.cj_state import TNTParams, compute_cj_state, compute_separation_state

cfg_path = ROOT / "configs" / "tnt_spherical_50mm.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

air = cfg["air"]
extracted_dir = ROOT / cfg["data"]["extracted_dir"]
dataset = D3plotLineDataset(str(extracted_dir))

tnt = TNTParams(rho_TNT=dataset.rho_TNT)
cj_b = compute_cj_state(tnt)
sep = compute_separation_state(tnt, R_0=float(dataset.R_0), cj_bundle=cj_b)

det, asn = build_networks(
    cfg["networks"],
    rho_ref_A=cj_b.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj_b.P_CJ,
    rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"], P_ref_B=cfg["domain"]["P_ref_B"],
)

# Load checkpoints
det_ckpt = ROOT / cfg["training"]["checkpoint_dir"] / "detonation.pt"
asn_ckpt = ROOT / cfg["training"]["checkpoint_dir"] / "air_shock.pt"
state = torch.load(det_ckpt, map_location="cpu", weights_only=False)
det.load_state_dict(state["state_dict"]); det.eval()
state = torch.load(asn_ckpt, map_location="cpu", weights_only=False)
asn.load_state_dict(state["state_dict"]); asn.eval()

# Reconstruct hard-constraint wrapper
hc_meta = ROOT / cfg["training"]["checkpoint_dir"] / "air_shock_hc_meta.pt"
asn_eval = asn
if hc_meta.exists():
    hc = torch.load(hc_meta, map_location="cpu", weights_only=False)
    asn_eval = HardContactConstrainedASN(
        asn, tau_sep=float(hc["t_sep"]), Z_c=float(hc["R_c"]),
        target_rho=hc["target_rho"], target_u=hc["target_u"],
        target_P=hc["target_P"],
        tau_t=float(hc["tau_t"]), tau_r=float(hc["tau_r"]),
    )

t_sep = sep.t_sep; R_c = sep.R_c; x_end = dataset.x_end; t_end = dataset.t_end

# Pick diagnostic times spread across the simulation
t_diag = np.linspace(max(t_sep, 2e-5), t_end * 0.95, 5)
print(f"\nDiagnostic times: {[f'{t*1e6:.1f}us' for t in t_diag]}")
print(f"R_c = {R_c*1e3:.2f} mm, x_end = {x_end:.3f} m, t_sep = {t_sep*1e6:.2f} us")
print(f"t_end = {t_end*1e6:.0f} us")

print("\n" + "="*100)
print(f"{'Time':>8s} {'Region':>20s} {'PINN ρ':>10s} {'d3plot ρ':>10s} {'ρ err%':>8s} {'PINN P':>10s} {'d3plot P':>10s} {'P err%':>8s} {'PINN u':>10s} {'d3plot u':>10s} {'u err%':>8s}")
print("="*100)

for ti in t_diag:
    t_t = torch.tensor([[ti]], dtype=torch.float32)
    R_s = float(dataset.R_s(t_t))

    # --- Region 1: Post-shock compressed air (R_c + eps < r < R_s - eps) ---
    eps_m = 2e-3  # 2mm margin
    if R_s - R_c > 4 * eps_m:
        r_vals = np.linspace(R_c + eps_m, R_s - eps_m, 20)
        t_vals = np.full_like(r_vals, ti)
        t_in = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
        r_in = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)

        rho_t, u_t, P_t = dataset.state_at(t_in, r_in)
        with torch.no_grad():
            rho_p, u_p, P_p = asn_eval(t_in, r_in)

        err_rho = float(((rho_p - rho_t).abs() / rho_t.clamp(min=1e-10)).mean())
        err_P = float(((P_p - P_t).abs() / P_t.clamp(min=1e-10)).mean())
        err_u = float(((u_p - u_t).abs() / (u_t.abs() + 1e-10)).mean())

        print(f"{ti*1e6:8.1f}us {'post-shock':>20s} "
              f"{float(rho_p.mean()):10.3f} {float(rho_t.mean()):10.3f} {err_rho*100:7.1f}% "
              f"{float(P_p.mean())/1e6:10.4f} {float(P_t.mean())/1e6:10.4f} {err_P*100:7.1f}% "
              f"{float(u_p.mean()):10.1f} {float(u_t.mean()):10.1f} {err_u*100:7.1f}%")

    # --- Region 2: Shock front (R_s ± 5mm band) ---
    band = 0.005
    r_vals = np.linspace(max(R_c, R_s - band), min(x_end, R_s + band), 20)
    t_vals = np.full_like(r_vals, ti)
    t_in = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
    r_in = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)

    rho_t, u_t, P_t = dataset.state_at(t_in, r_in)
    with torch.no_grad():
        rho_p, u_p, P_p = asn_eval(t_in, r_in)

    err_rho = float(((rho_p - rho_t).abs() / rho_t.clamp(min=1e-10)).mean())
    err_P = float(((P_p - P_t).abs() / P_t.clamp(min=1e-10)).mean())
    err_u = float(((u_p - u_t).abs() / (u_t.abs() + 1e-10)).mean())

    print(f"{ti*1e6:8.1f}us {'shock-front (±5mm)':>20s} "
          f"{float(rho_p.mean()):10.3f} {float(rho_t.mean()):10.3f} {err_rho*100:7.1f}% "
          f"{float(P_p.mean())/1e6:10.4f} {float(P_t.mean())/1e6:10.4f} {err_P*100:7.1f}% "
          f"{float(u_p.mean()):10.1f} {float(u_t.mean()):10.1f} {err_u*100:7.1f}%")

    # --- Region 3: Ahead of shock / ambient (r > R_s + band) ---
    r_start = R_s + band + 1e-3
    if r_start < x_end:
        r_vals = np.linspace(r_start, x_end, 10)
        t_vals = np.full_like(r_vals, ti)
        t_in = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
        r_in = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)

        rho_t, u_t, P_t = dataset.state_at(t_in, r_in)
        with torch.no_grad():
            rho_p, u_p, P_p = asn_eval(t_in, r_in)

        err_rho = float(((rho_p - rho_t).abs() / rho_t.clamp(min=1e-10)).mean())
        err_P = float(((P_p - P_t).abs() / P_t.clamp(min=1e-10)).mean())
        err_u = float(((u_p - u_t).abs() / (u_t.abs() + 1e-10)).mean())

        print(f"{ti*1e6:8.1f}us {'ambient (ahead)':>20s} "
              f"{float(rho_p.mean()):10.3f} {float(rho_t.mean()):10.3f} {err_rho*100:7.1f}% "
              f"{float(P_p.mean())/1e6:10.4f} {float(P_t.mean())/1e6:10.4f} {err_P*100:7.1f}% "
              f"{float(u_p.mean()):10.1f} {float(u_t.mean()):10.1f} {err_u*100:7.1f}%")

    print("-"*100)

# ---- Overall summary stats ----
print("\n=== Global error summary (full Phase B rectangle) ===")
n_samples = 4096
kept = []
attempts = 0
while sum(t.numel() for t, *_ in kept) < n_samples and attempts < 8:
    t, x, rho, u, P = dataset.sample_grid_points(n_samples)
    mask = (t.squeeze(-1) >= t_sep) & (x.squeeze(-1) >= R_c)
    if mask.any():
        kept.append((t[mask], x[mask], rho[mask], u[mask], P[mask]))
    attempts += 1
t_all = torch.cat([k[0] for k in kept])[:n_samples].view(-1, 1)
r_all = torch.cat([k[1] for k in kept])[:n_samples].view(-1, 1)
rho_t = torch.cat([k[2] for k in kept])[:n_samples].view(-1, 1)
u_t = torch.cat([k[3] for k in kept])[:n_samples].view(-1, 1)
P_t = torch.cat([k[4] for k in kept])[:n_samples].view(-1, 1)

with torch.no_grad():
    rho_p, u_p, P_p = asn_eval(t_all, r_all)

err_rho = ((rho_p - rho_t).abs() / rho_t.clamp(min=1e-10)).squeeze()
err_P = ((P_p - P_t).abs() / P_t.clamp(min=1e-10)).squeeze()
err_u = ((u_p - u_t).abs() / (u_t.abs() + 1.0)).squeeze()

print(f"Median ρ error: {float(err_rho.median())*100:.1f}%")
print(f"Median P error: {float(err_P.median())*100:.1f}%")
print(f"Median u error: {float(err_u.median())*100:.1f}%")
print(f"Mean ρ error:   {float(err_rho.mean())*100:.1f}%")
print(f"Mean P error:   {float(err_P.mean())*100:.1f}%")
print(f"Mean u error:   {float(err_u.mean())*100:.1f}%")

# Percentile breakdown
for pct in [25, 50, 75, 90, 95]:
    q = pct / 100.0
    print(f"  P{pct}: ρ {float(err_rho.quantile(q))*100:.1f}%  P {float(err_P.quantile(q))*100:.1f}%  u {float(err_u.quantile(q))*100:.1f}%")

# Check: does PINN output vary spatially at all?
print(f"\nPINN output range: ρ [{float(rho_p.min()):.3f}, {float(rho_p.max()):.3f}]")
print(f"                   P [{float(P_p.min())/1e6:.4f}, {float(P_p.max())/1e6:.4f}] MPa")
print(f"                   u [{float(u_p.min()):.1f}, {float(u_p.max()):.1f}] m/s")
print(f"d3plot range:      ρ [{float(rho_t.min()):.3f}, {float(rho_t.max()):.3f}]")
print(f"                   P [{float(P_t.min())/1e6:.4f}, {float(P_t.max())/1e6:.4f}] MPa")
print(f"                   u [{float(u_t.min()):.1f}, {float(u_t.max()):.1f}] m/s")
print(f"\nPINN std:  ρ={float(rho_p.std()):.3f}  P={float(P_p.std())/1e6:.4f} MPa  u={float(u_p.std()):.1f} m/s")
print(f"d3plot std: ρ={float(rho_t.std()):.3f}  P={float(P_t.std())/1e6:.4f} MPa  u={float(u_t.std()):.1f} m/s")
