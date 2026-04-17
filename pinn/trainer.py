"""Three-stage PINN training pipeline (§2.5).

Stage 1  (5 000 steps, Adam, lr=1e-3)
    Losses:   L_IC only.
    ContactNet and ShockNet are frozen; MainNet learns the IC profile.

Stage 2  (50 000 steps, Adam, lr=1e-4, cosine schedule)
    Losses:   L_IC + L_PDE + L_BC + L_RH.
    RAR + AV annealing + GradNorm dynamic weight balancing.
    ShockNet is frozen for the first `freeze_shocknet_steps` steps and driven
    by an R_s prior from the AnalyticalLoader.
    P_prod in L_BC(a) uses the LS-DYNA time series for the first
    `pprod_from_lsdyna_steps` steps, then switches to JWL online.

Stage 3  (≤8 000 steps, L-BFGS)
    All losses active, AV nearly zero, RH dominates.

Usage
-----
    python -m pinn.trainer --stage 1 --config configs/tnt_spherical.yaml
    python -m pinn.trainer --stage 2 --config configs/tnt_spherical.yaml
    python -m pinn.trainer --stage 3 --config configs/tnt_spherical.yaml
    python -m pinn.trainer --all    --config configs/tnt_spherical.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import yaml

# Add project root to sys.path when run as __main__
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics import solve_cj, match_contact, evolve_quasisteady, find_separation
from physics.jwl_isentrope import JWLParams
from data.lsdyna_loader import AnalyticalLoader
from data.ic_builder import ICBuilder
from pinn.networks import build_networks
from pinn.losses import ICLoss, PDELoss, BCLoss, RHLoss


# ============================================================ utilities

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(cfg: dict) -> torch.device:
    d = cfg["training"]["device"]
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _ckpt_path(cfg: dict, stage: int) -> Path:
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir / f"stage{stage}.pt"


def _save_checkpoint(path: Path, main_net, contact_net, shock_net, step: int, loss: float) -> None:
    torch.save({
        "step": step,
        "loss": loss,
        "main_net":    main_net.state_dict(),
        "contact_net": contact_net.state_dict(),
        "shock_net":   shock_net.state_dict(),
    }, path)
    print(f"  [checkpoint] saved → {path}  (step={step}, loss={loss:.4e})")


def _load_checkpoint(path: Path, main_net, contact_net, shock_net) -> int:
    ckpt = torch.load(path, map_location="cpu")
    main_net.load_state_dict(ckpt["main_net"])
    contact_net.load_state_dict(ckpt["contact_net"])
    shock_net.load_state_dict(ckpt["shock_net"])
    print(f"  [checkpoint] loaded ← {path}  (step={ckpt['step']}, loss={ckpt['loss']:.4e})")
    return ckpt["step"]


# ============================================================ physics setup

def build_physics(cfg: dict):
    """Run Phase-1 preprocessing and return (sep, loader, ic_builder)."""
    exp   = cfg["explosive"]
    air   = cfg["air"]
    prep  = cfg["preprocessing"]

    params = JWLParams(
        A=exp["jwl"]["A"],
        B=exp["jwl"]["B"],
        R1=exp["jwl"]["R1"],
        R2=exp["jwl"]["R2"],
        omega=exp["jwl"]["omega"],
        E0=exp["jwl"]["E0"],
        rho0=exp["rho0"],
    )
    D_CJ = exp["D_CJ"]
    W    = exp["W"]
    rho0 = exp["rho0"]
    R0   = (3.0 * W / (4.0 * math.pi * rho0)) ** (1.0 / 3.0)

    bracket = tuple(prep["v_cj_bracket"])
    cj = solve_cj(
        params, D_CJ,
        bracket=bracket,
        v_max=prep["isentrope_v_max"],
        n_points=prep["isentrope_n_points"],
    )
    print(f"  CJ state: v_CJ={cj.v_CJ:.4f}, P_CJ={cj.P_CJ/1e9:.2f} GPa, "
          f"u_CJ={cj.u_CJ:.1f} m/s")

    contact = match_contact(
        cj,
        gamma=air["gamma"],
        rho_a=air["rho_a"],
        P_a=air["P_a"],
        P_init_frac=prep["secant_P_init_frac"],
        tol=prep["secant_tol"],
        max_iter=prep["secant_max_iter"],
    )
    print(f"  Contact match: P_c*={contact.P_c/1e6:.2f} MPa, "
          f"u_c*={contact.u_c:.1f} m/s, D_s*={contact.D_s:.1f} m/s")

    series = evolve_quasisteady(
        R0=R0,
        cj=cj,
        contact=contact,
        gamma=air["gamma"],
        rho_a=air["rho_a"],
        P_a=air["P_a"],
        dt=prep["quasisteady_dt"],
        t_max=prep["quasisteady_t_max"],
    )

    sep = find_separation(
        series,
        P_a=air["P_a"],
        P_threshold_ratio=prep["separation"]["P_threshold_ratio"],
        v_threshold=prep["separation"]["v_threshold"],
    )
    print(f"  Separation: t_sep={sep.t_sep*1e3:.3f} ms, "
          f"R_c={sep.R_c*100:.2f} cm, R_s={sep.R_s*100:.2f} cm")

    loader = AnalyticalLoader(
        sep=sep,
        gamma=air["gamma"],
        rho_a=air["rho_a"],
        P_a=air["P_a"],
    )
    ic_builder = ICBuilder(
        loader=loader,
        rho_ref=cfg["domain"]["rho_ref"],
        P_ref=cfg["domain"]["P_ref"],
        u_ref=cfg["domain"]["u_ref"],
    )
    return cj, sep, R0, loader, ic_builder


# ============================================================ GradNorm weight balancing

class GradNormBalancer:
    """Periodically rebalance loss weights so each loss's gradient has equal norm.

    Simplified from Chen et al. "GradNorm" (ICML 2018).
    """

    def __init__(self, weights: dict[str, float], alpha: float = 1.5) -> None:
        self.weights = dict(weights)
        self.alpha   = alpha
        self._loss0: Optional[dict[str, float]] = None

    def update(self, losses: dict[str, float], grad_norms: dict[str, float]) -> None:
        if self._loss0 is None:
            self._loss0 = dict(losses)
            return

        # Relative training rates
        r = {k: losses[k] / (self._loss0[k] + 1e-10) for k in losses}
        r_mean = np.mean(list(r.values()))
        # Target norm for each loss
        target = {k: grad_norms[k] * (r[k] / r_mean) ** self.alpha for k in grad_norms}
        target_mean = np.mean(list(target.values()))

        if target_mean < 1e-20:
            return

        for k in self.weights:
            if k in target and target[k] > 0:
                self.weights[k] *= float(grad_norms.get(k, 1.0)) / (target[k] + 1e-10)

        # Re-normalise so that the sum = initial sum
        total = sum(self.weights.values())
        init_sum = sum(self.weights[k] for k in self.weights)
        if total > 0:
            scale = init_sum / total if init_sum > 0 else 1.0
            for k in self.weights:
                self.weights[k] = max(self.weights[k] * scale, 0.01)


# ============================================================ stage implementations

def _log(step: int, losses: dict, elapsed: float) -> None:
    parts = [f"step={step:6d}", f"t={elapsed:.1f}s"]
    for k, v in losses.items():
        parts.append(f"{k}={v:.3e}")
    print("  " + "  ".join(parts))


def run_stage1(cfg: dict, main_net, contact_net, shock_net, ic_loss: ICLoss, device) -> None:
    scfg  = cfg["training"]["stage1"]
    steps = scfg["steps"]
    log_n = cfg["training"]["log_interval"]

    optimizer = optim.Adam(main_net.parameters(), lr=scfg["lr"])

    # Freeze contact / shock nets
    for p in contact_net.parameters():
        p.requires_grad_(False)
    for p in shock_net.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = ic_loss()
        loss.backward()
        optimizer.step()

        if step % log_n == 0:
            _log(step, {"L_IC": loss.item()}, time.time() - t0)

    # Unfreeze
    for p in contact_net.parameters():
        p.requires_grad_(True)
    for p in shock_net.parameters():
        p.requires_grad_(True)

    path = _ckpt_path(cfg, 1)
    _save_checkpoint(path, main_net, contact_net, shock_net, steps, loss.item())


def run_stage2(
    cfg: dict,
    main_net, contact_net, shock_net,
    ic_loss: ICLoss,
    pde_loss: PDELoss,
    bc_loss: BCLoss,
    rh_loss: RHLoss,
    device,
    shock_prior_fn=None,
) -> None:
    scfg  = cfg["training"]["stage2"]
    steps = scfg["steps"]
    log_n = cfg["training"]["log_interval"]

    wc = cfg["loss_weights"]["stage2"]
    weights = {"IC": wc["IC"], "PDE": wc["PDE"], "BC": wc["BC"], "RH": wc["RH"]}
    balancer_cfg = cfg["loss_weights"]["dynamic_balancing"]
    balancer = GradNormBalancer(weights) if balancer_cfg["enabled"] else None

    all_params = (list(main_net.parameters())
                  + list(contact_net.parameters())
                  + list(shock_net.parameters()))
    optimizer = optim.Adam(all_params, lr=scfg["lr"])

    # Cosine LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    freeze_shock   = scfg.get("freeze_shocknet_steps", 5000)
    pprod_lsdyna_n = scfg.get("pprod_from_lsdyna_steps", 5000)

    # Temporarily freeze ShockNet
    for p in shock_net.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    balancer_interval = balancer_cfg["interval"]

    for step in range(1, steps + 1):
        # Unfreeze ShockNet after warmup
        if step == freeze_shock + 1:
            for p in shock_net.parameters():
                p.requires_grad_(True)
            print(f"  [step {step}] ShockNet unfrozen.")

        # Switch P_prod source from LS-DYNA to JWL online
        if step == pprod_lsdyna_n + 1 and bc_loss.P_prod_fn is not None:
            bc_loss.P_prod_fn = None
            print(f"  [step {step}] BCLoss: switched P_prod from LS-DYNA to JWL online.")

        # AV annealing
        pde_loss.anneal(step / steps)

        optimizer.zero_grad()
        L_ic  = ic_loss()
        L_pde = pde_loss()
        L_bc  = bc_loss()
        L_rh  = rh_loss()

        w = weights
        total = w["IC"] * L_ic + w["PDE"] * L_pde + w["BC"] * L_bc + w["RH"] * L_rh
        total.backward()

        # GradNorm balancing
        if balancer is not None and step % balancer_interval == 0:
            loss_vals = {
                "IC": L_ic.item(), "PDE": L_pde.item(),
                "BC": L_bc.item(), "RH": L_rh.item(),
            }
            # Approximate gradient norms via main_net first-layer grads
            grad_norms = {}
            for k, L in [("IC", L_ic), ("PDE", L_pde), ("BC", L_bc), ("RH", L_rh)]:
                gn = sum(
                    p.grad.norm().item() ** 2
                    for p in main_net.parameters()
                    if p.grad is not None
                ) ** 0.5
                grad_norms[k] = gn + 1e-10
            balancer.update(loss_vals, grad_norms)
            weights.update(balancer.weights)

        optimizer.step()
        scheduler.step()

        if step % log_n == 0:
            _log(step, {
                "IC": L_ic.item(), "PDE": L_pde.item(),
                "BC": L_bc.item(), "RH": L_rh.item(),
                "tot": total.item(),
            }, time.time() - t0)

    path = _ckpt_path(cfg, 2)
    _save_checkpoint(path, main_net, contact_net, shock_net, steps, total.item())


def run_stage3(
    cfg: dict,
    main_net, contact_net, shock_net,
    ic_loss: ICLoss,
    pde_loss: PDELoss,
    bc_loss: BCLoss,
    rh_loss: RHLoss,
    device,
) -> None:
    scfg  = cfg["training"]["stage3"]
    steps = scfg["steps"]
    log_n = cfg["training"]["log_interval"]

    wc = cfg["loss_weights"]["stage3"]
    weights = {"IC": wc["IC"], "PDE": wc["PDE"], "BC": wc["BC"], "RH": wc["RH"]}

    # Near-zero AV
    pde_loss.anneal(1.0)

    all_params = (list(main_net.parameters())
                  + list(contact_net.parameters())
                  + list(shock_net.parameters()))
    optimizer = optim.LBFGS(
        all_params,
        lr=scfg["lr"],
        max_iter=scfg["max_iter_per_step"],
        history_size=scfg["history_size"],
        line_search_fn="strong_wolfe",
    )

    t0 = time.time()
    step_count = [0]
    last_total = [torch.tensor(0.0)]

    for step in range(1, steps + 1):
        def closure():
            optimizer.zero_grad()
            L_ic  = ic_loss()
            L_pde = pde_loss()
            L_bc  = bc_loss()
            L_rh  = rh_loss()
            total = (weights["IC"] * L_ic + weights["PDE"] * L_pde
                     + weights["BC"] * L_bc + weights["RH"] * L_rh)
            total.backward()
            last_total[0] = total
            return total

        optimizer.step(closure)
        step_count[0] += 1

        if step % log_n == 0:
            _log(step, {"total": last_total[0].item()}, time.time() - t0)

    path = _ckpt_path(cfg, 3)
    _save_checkpoint(path, main_net, contact_net, shock_net, steps, last_total[0].item())


# ============================================================ main entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="PINN blast-wave trainer")
    parser.add_argument("--config", default="configs/tnt_spherical.yaml")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    device = _resolve_device(cfg)
    print(f"Device: {device}")

    torch.manual_seed(cfg["training"]["seed"])

    # ---- Phase-1 preprocessing
    print("\n[Phase 1] Analytical preprocessing...")
    cj, sep, R0, _loader, ic_builder = build_physics(cfg)

    # ---- IC data
    samp = cfg["sampling"]
    ic_data = ic_builder.as_tensors(n_ic=samp["N_IC"], seed=cfg["training"]["seed"])
    for k, v in ic_data.items():
        if hasattr(v, "to"):
            ic_data[k] = v.to(device)

    # ---- Networks
    print("\n[Init] Building networks...")
    main_net, contact_net, shock_net = build_networks(cfg["networks"], sep)
    main_net    = main_net.to(device)
    contact_net = contact_net.to(device)
    shock_net   = shock_net.to(device)

    # ---- Domain info
    air  = cfg["air"]
    dom  = cfg["domain"]
    R_far = sep.R_s * cfg["domain"]["R_far_factor"]
    av_cfg = cfg["artificial_viscosity"]
    ell  = av_cfg["length_scale_factor"] * (R_far - sep.R_c) / samp["N_f"] ** 0.5

    # ---- Loss objects
    ic_loss  = ICLoss(main_net, ic_data)

    pde_loss = PDELoss(
        main_net=main_net,
        gamma=air["gamma"],
        t_sep=sep.t_sep,
        t_end=cfg["domain"]["t_end"],
        R_c_fn=lambda t: contact_net(t),
        R_far=R_far,
        N_f=samp["N_f"],
        alpha_sensor=cfg["pde_sensor"]["alpha"],
        ell=ell,
        c1_start=av_cfg["c1_start"], c1_end=av_cfg["c1_end"],
        c2_start=av_cfg["c2_start"], c2_end=av_cfg["c2_end"],
        rar_interval=samp["rar"]["interval"],
        rar_fraction=samp["rar"]["fraction"],
        rar_pool_factor=samp["rar"]["pool_factor"],
        device=str(device),
    )

    bcsub = cfg["loss_weights"]["bc_sub_weights"]
    bc_loss = BCLoss(
        main_net=main_net,
        contact_net=contact_net,
        gamma=air["gamma"],
        R_far=R_far,
        N_c=samp["N_c"],
        N_inf=samp["N_inf"],
        t_sep=sep.t_sep,
        t_end=cfg["domain"]["t_end"],
        beta_a=bcsub["contact_pressure"],
        beta_b=bcsub["contact_velocity"],
        beta_c=bcsub["farfield_C_plus"],
        P_prod_fn=None,             # use JWL online from the start via AnalyticalLoader
        jwl_isentrope=cj.isentrope,
        R0=R0,
        device=str(device),
    )

    rh_loss = RHLoss(
        main_net=main_net,
        shock_net=shock_net,
        gamma=air["gamma"],
        rho_a=air["rho_a"],
        P_a=air["P_a"],
        N_s=samp["N_s"],
        t_sep=sep.t_sep,
        t_end=cfg["domain"]["t_end"],
        device=str(device),
    )

    # ---- Optionally load Stage-1 checkpoint before Stage 2/3
    ckpt1 = _ckpt_path(cfg, 1)
    ckpt2 = _ckpt_path(cfg, 2)
    if ckpt1.exists() and args.stage in (2, 3, None):
        _load_checkpoint(ckpt1, main_net, contact_net, shock_net)
    if ckpt2.exists() and args.stage == 3:
        _load_checkpoint(ckpt2, main_net, contact_net, shock_net)

    run_stages = []
    if args.all:
        run_stages = [1, 2, 3]
    elif args.stage is not None:
        run_stages = [args.stage]
    else:
        run_stages = [1]

    for s in run_stages:
        print(f"\n{'='*60}\n[Stage {s}] Starting...\n{'='*60}")
        if s == 1:
            run_stage1(cfg, main_net, contact_net, shock_net, ic_loss, device)
        elif s == 2:
            run_stage2(cfg, main_net, contact_net, shock_net,
                       ic_loss, pde_loss, bc_loss, rh_loss, device)
        elif s == 3:
            run_stage3(cfg, main_net, contact_net, shock_net,
                       ic_loss, pde_loss, bc_loss, rh_loss, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
