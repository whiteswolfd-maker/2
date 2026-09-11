"""Sequential two-network PINN trainer for the spherical TNT blast model.

Training proceeds in two strictly sequential phases:

Phase A (DetonationNet)
    A1: Adam, L_data,A + L_IC,A + L_BC,A,slope          (data + slope BC + apex)
    A2: Adam, + L_PDE,A                                  (add JWL spherical Euler)
    Hard endpoint: analytical products-side outflow state.
    Endpoint vs d3plot: model/data diagnostic, not a convergence test of a
          value that has already been fixed by construction.
    Freeze: DetonationNet weights frozen; cached at (t_sep, R_c).

Phase B (AirShockNet)
    B1: Adam, L_data,B + L_IC,B                          (data + frozen-A IC)
    B2: Adam, + L_PDE,B + L_RH,B + L_BC,B,outflow         (add physics + outflow)

Usage
-----
    python -m pinn.trainer --net detonation --config configs/tnt_spherical_50mm.yaml
    python -m pinn.trainer --net airshock   --config configs/tnt_spherical_50mm.yaml
    python -m pinn.trainer --all            --config configs/tnt_spherical_50mm.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.d3plot_dataset import D3plotLineDataset
from data.multi_radius_dataset import MultiRadiusDataset
from physics.cj_state import (
    TNTParams,
    compute_cj_state,
    compute_separation_state,
)
from pinn.networks import (
    AirShockNet,
    DetonationNet,
    HardContactConstrainedASN,
    HardDetNetConstraint,
    build_networks,
)
from pinn.losses.detonation_loss import (
    DetonationDataLoss,
    DetonationICLoss,
    DetonationPDELoss,
    DetonationSlopeBCLoss,
)
from pinn.losses.air_shock_loss import (
    AirShockDataLoss,
    AirShockICLoss,
    AirShockOutflowLoss,
    AirShockPDELoss,
    AirShockRHLoss,
)
from pinn.checkpoints import constraint_metadata, load_checkpoint, save_checkpoint
from pinn.coupling import air_contact_density


# ============================================================ utilities


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(cfg: dict) -> torch.device:
    d = cfg["training"].get("device", "cpu")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _ckpt_path(cfg: dict, name: str) -> Path:
    p = Path(cfg["training"]["checkpoint_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.pt"


def _save_checkpoint(path: Path, net, step: int, loss: float, *, training_complete=False) -> None:
    save_checkpoint(path, net, step, loss, training_complete=training_complete)
    print(f"  [checkpoint] saved -> {path}  (step={step}, loss={loss:.4e})")


def _load_checkpoint(path: Path, net):
    net, ckpt = load_checkpoint(path, net, require_constraints=True)
    print(f"  [checkpoint] loaded <- {path}  (step={ckpt['step']}, "
          f"loss={ckpt['loss']:.4e})")
    return net, ckpt


def _constrain_detonation(cfg, dataset, sep, det, device):
    """One definition of the A anchors for training and B-only loading."""
    sm = cfg["sampling"]
    cj = sep.cj_bundle
    return HardDetNetConstraint(
        det,
        tau_sep=float(getattr(dataset, "tau_sep", sep.t_sep)),
        Z_c=float(getattr(dataset, "Z_c", sep.R_c)),
        Z_R0=float(getattr(dataset, "Z_R0", sep.R_0)),
        rho_cj=cj.rho_CJ, u_cj=0.0, P_cj=cj.P_CJ,
        rho_x=sep.rho_x, u_x=sep.u_x, P_x=sep.P_x,
        tau_t=float(sm.get("tau_t", 1e-6)), tau_r=float(sm.get("tau_r", 1e-3)),
        tau_t_cj=float(sm.get("tau_t_cj", 1e-7)), tau_r_cj=float(sm.get("tau_r_cj", 5e-4)),
        use_contact=True,
    ).to(device)


def _log(step: int, losses: dict, elapsed: float) -> None:
    parts = [f"step={step:6d}", f"t={elapsed:.1f}s"]
    for k, v in losses.items():
        parts.append(f"{k}={v:.3e}")
    print("  " + "  ".join(parts))


def _adam(net, lr: float) -> optim.Optimizer:
    return optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)


# ============================================================ Phase A loss factory


def _build_detonation_losses(
    cfg: dict, dataset: D3plotLineDataset, net: DetonationNet,
    sep_bundle, device: torch.device,
) -> dict:
    sm = cfg["sampling"]
    tnt = sep_bundle.tnt
    cj  = sep_bundle.cj_bundle
    # Support both scaled (MultiRadiusDataset) and physical (D3plotLineDataset) coords
    tau_sep = float(getattr(dataset, "tau_sep", sep_bundle.t_sep))
    tau_traj = dataset.t_traj_rc
    Z_traj = dataset.r_traj_rc
    if tau_traj is None:
        raise RuntimeError("Dataset is missing the Taylor-Sadovsky CSV.")

    import torch as _torch
    # For scaled datasets, sep_bundle trajectories may also need scaling
    if hasattr(dataset, "tau_traj_tail"):
        tau_traj_tail = dataset.tau_traj_tail
        Z_traj_tail = dataset.Z_traj_tail
    else:
        tau_traj_tail = _torch.as_tensor(sep_bundle.t_traj_tail, dtype=_torch.float32)
        Z_traj_tail = _torch.as_tensor(sep_bundle.r_traj_tail, dtype=_torch.float32)
    tau_data_min = float(sm.get("t_data_min", 0.0))
    # Scale t_data_min if using scaled dataset (physical 8e-6 → per-radius)
    if hasattr(dataset, "_tau_data_min"):
        tau_data_min = dataset._tau_data_min[0]  # use smallest as safe default

    rho_min_A = float(sm.get("rho_min_A", 0.0))
    Z_anchor = float(cfg["domain"].get("Z_anchor_A", cfg["domain"].get("r_anchor_A", 0.052712)))
    Z_anchor = float(getattr(net, "Z_R0", Z_anchor))

    return {
        "data_A": DetonationDataLoss(
            net, dataset, tau_sep=tau_sep,
            tau_traj_rc=tau_traj, Z_traj_rc=Z_traj,
            tau_traj_tail=tau_traj_tail, Z_traj_tail=Z_traj_tail,
            N=sm["A"]["N_data"], tau_data_min=tau_data_min,
            rho_min=rho_min_A,
            u_ref=cfg["domain"]["u_ref_A"],
            device=str(device),
        ),
        "PDE_A": DetonationPDELoss(
            net, tau_sep=tau_sep,
            tau_traj_rc=tau_traj, Z_traj_rc=Z_traj,
            tau_traj_tail=tau_traj_tail, Z_traj_tail=Z_traj_tail,
            A=tnt.A, B=tnt.B, R1=tnt.R1, R2=tnt.R2, omega=tnt.omega,
            C=tnt.omega * tnt.E0,
            rho0=tnt.rho_TNT, v_cj=cj.v_CJ, e_cj=0.0,
            N_f=sm["A"]["N_f"],
            rho_ref=cj.rho_CJ, u_ref=cfg["domain"]["u_ref_A"],
            P_ref=cj.P_CJ, t_ref=cfg["domain"]["t_ref_A"],
            device=str(device),
        ),
        "IC_A": DetonationICLoss(
            net,
            rho_CJ=cj.rho_CJ, P_CJ=cj.P_CJ,
            Z_anchor=Z_anchor,
            u_ref=cfg["domain"]["u_ref_A"],
            device=str(device),
        ),
        "BC_A_slope": DetonationSlopeBCLoss(
            net, dataset, tau_sep=tau_sep,
            tau_traj_rc=tau_traj, Z_traj_rc=Z_traj,
            N=sm["A"]["N_slope"], rho_min=rho_min_A,
            u_ref=cfg["domain"]["u_ref_A"],
            device=str(device),
        ),
    }


# ============================================================ Phase B loss factory


def _build_air_shock_losses(
    cfg: dict, dataset: D3plotLineDataset, net: AirShockNet,
    frozen_det: DetonationNet, sep_bundle, device: torch.device,
) -> dict:
    sm = cfg["sampling"]
    air = cfg["air"]
    # Phase-B anchor in the SAME coordinate system the losses/network live in:
    # scaled (tau, Z) when ``dataset`` is a MultiRadiusDataset, physical (s, m)
    # for a single-radius D3plotLineDataset.  ``sep_bundle`` is always physical
    # and (for a unified dataset) refers to the first radius only — reading it
    # directly would shift the Phase-B window ~M^(1/3) off the scaled contact
    # point whenever the first radius is not the 50 mm reference.
    t_sep = float(getattr(dataset, "tau_sep", sep_bundle.t_sep))
    R_c = float(getattr(dataset, "Z_c", sep_bundle.R_c))
    t_end = dataset.t_end; x_end = dataset.x_end
    t_data_min = float(sm.get("t_data_min", 0.0))
    # IC-corner exclusion margin.  Physical single-radius configs name it
    # ``t_margin_B`` (seconds); the scaled unified config names it
    # ``tau_margin_B`` (s/kg^(1/3)) because Phase-B coordinates there are tau.
    t_margin = float(sm.get("tau_margin_B", sm.get("t_margin_B", 0.0)))

    sm_B = sm["B"]
    return {
        "data_B": AirShockDataLoss(
            net, dataset, t_sep=t_sep, R_c=R_c,
            N=sm_B["N_data"], t_data_min=t_data_min,
            u_ref=cfg["domain"]["u_ref_B"],
            shock_frac=float(sm_B.get("shock_frac_B", 0.5)),
            shock_band=float(sm_B.get("shock_band_B", 0.05)),
            device=str(device),
        ),
        "IC_B": AirShockICLoss(
            net, frozen_det,
            t_sep=t_sep, R_c=R_c, x_end=x_end,
            rho_a=air["rho_a"], P_a=air["P_a"],
            gamma=air["gamma"],
            N_amb=sm_B["N_IC_amb"],
            contact_repeat=sm_B["contact_repeat"],
            u_ref=cfg["domain"]["u_ref_B"],
            device=str(device),
        ),
        "PDE_B": AirShockPDELoss(
            net, t_sep=t_sep, t_end=t_end, R_c=R_c, x_end=x_end,
            dataset=dataset,
            gamma=air["gamma"],
            N_f=sm_B["N_f"],
            rho_ref=air["rho_a"], u_ref=cfg["domain"]["u_ref_B"],
            P_ref=cfg["domain"]["P_ref_B"],
            t_ref=cfg["domain"]["t_ref_B"],
            t_margin=t_margin,
            shock_mask=float(sm.get("shock_mask_B", 0.02)),
            device=str(device),
        ),
        "RH_B": AirShockRHLoss(
            net, dataset,
            t_sep=t_sep, t_end=t_end,
            gamma=air["gamma"], rho_a=air["rho_a"], P_a=air["P_a"],
            eps=cfg["domain"]["rh_eps"],
            N_s=sm_B["N_RH"], u_ref=cfg["domain"]["u_ref_B"],
            t_margin=t_margin,
            device=str(device),
        ),
        "BC_B_outflow": AirShockOutflowLoss(
            net, t_sep=t_sep, t_end=t_end, x_end=x_end,
            gamma=air["gamma"],
            N=sm_B["N_outflow"], scale=cfg["domain"]["outflow_scale"],
            rho_a=air["rho_a"], P_a=air["P_a"],
            u_ref=cfg["domain"]["u_ref_B"],
            device=str(device),
        ),
    }


# ============================================================ adaptive loss weighting


class _RunningWeight:
    """EMA-tracked adaptive weight for a single loss term."""
    __slots__ = ("ema", "weight")
    def __init__(self, base: float):
        self.ema   = 1.0        # running mean of raw loss value
        self.weight = float(base)


def _adaptive_weighted_sum(losses: dict, keys: list[str], base_weights: dict,
                          running: dict[str, _RunningWeight],
                          alpha: float = 0.01, beta: float = 0.1,
                          ) -> tuple[torch.Tensor, dict]:
    """Compute weighted sum with per-term adaptive weights.

    Each loss term tracks an EMA of its raw value.  The adaptive weight is
    **proportional** to the EMA: losses that are currently large (poorly
    satisfied) get *more* weight, losses that are already small (well
    satisfied, like PDE at a constant solution) get *less* weight.  The
    weights are normalised so the mean equals 1.0, then multiplied by the
    user-supplied base weights.

    ``alpha`` controls how fast the EMA tracks recent values (higher =
    faster).  ``beta`` controls how fast the weight itself is updated
    (higher = more responsive, but noisier).
    """
    total = None
    log = {}
    eps = 1e-10
    # --- compute raw losses ---
    raw_vals: dict[str, float] = {}
    for k in keys:
        L = losses[k]()
        raw_vals[k] = L.item()
    # --- update EMA and adaptive weights ---
    for k in keys:
        rw = running[k]
        rw.ema = (1.0 - alpha) * rw.ema + alpha * raw_vals[k]
        target_w = rw.ema
        # adaptive component: proportional to EMA
        # (larger loss → needs more gradient attention)
        adapt = target_w
        rw.weight = (1.0 - beta) * rw.weight + beta * adapt
    # --- normalise adaptive component so mean = 1 ---
    adapt_mean = sum(running[k].weight for k in keys) / len(keys)
    for k in keys:
        running[k].weight = running[k].weight / max(adapt_mean, eps)
    # --- weighted sum ---
    for k in keys:
        L = losses[k]()  # fresh forward (losses may be stateful sampling)
        w = base_weights.get(k, 1.0) * running[k].weight
        log[k] = L.item()
        term = w * L
        total = term if total is None else total + term
    return total, log


# ============================================================ stage runners


def _weighted_sum(losses: dict, keys: list[str], weights: dict) -> tuple[torch.Tensor, dict]:
    """Static weighted sum (used when adaptive weighting is off)."""
    total = None
    log = {}
    for k in keys:
        L = losses[k]()
        log[k] = L.item()
        term = weights.get(k, 1.0) * L
        total = term if total is None else total + term
    return total, log


def _run_substage(
    name: str, net, losses: dict, keys: list[str], weights: dict,
    steps: int, lr: float, log_n: int = 100, grad_clip: float = 1.0,
    best_ckpt: str | None = None, best_net=None,
    best_ckpt_key: str = "PDE_B",
    adaptive: bool = False,
) -> tuple[float, int, dict]:
    print(f"\n--- {name}: {steps} steps  lr={lr}  losses={keys}"
          f"{'  [adaptive]' if adaptive else ''}")
    opt = _adam(net, lr)
    t_start = time.time()
    total_val = 0.0
    nan_streak = 0
    best_loss = float("inf")
    best_step = 0
    running_weights: dict[str, _RunningWeight] = {
        k: _RunningWeight(weights.get(k, 1.0)) for k in keys
    } if adaptive else {}
    for step in range(1, steps + 1):
        opt.zero_grad()
        if adaptive:
            total, log = _adaptive_weighted_sum(losses, keys, weights, running_weights)
        else:
            total, log = _weighted_sum(losses, keys, weights)
        if not torch.isfinite(total):
            nan_streak += 1
            if nan_streak == 1:
                print(f"  [warn] step {step}: non-finite loss "
                      f"({ {k: v for k, v in log.items()} }); skipping update")
            if nan_streak > 50:
                raise RuntimeError(
                    f"[{name}] {nan_streak} consecutive non-finite losses -- "
                    f"likely a singular PDE/RH sample.  Check r_min, rh_eps, "
                    f"or reduce learning rate."
                )
            continue
        nan_streak = 0
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        opt.step()
        total_val = total.item()
        log["tot"] = total_val
        if step % log_n == 0 or step == steps:
            _log(step, log, time.time() - t_start)
        # Track best PDE_B (lowest value after first 10% of training)
        if best_ckpt and steps > 1000:
            pde_val = log.get(best_ckpt_key, total_val)
            if pde_val < best_loss and step > steps // 10:
                best_loss = pde_val
                best_step = step
                save_net = best_net if best_net is not None else net
                _save_checkpoint(Path(best_ckpt), save_net, step, total_val)
    if best_ckpt and best_step > 0:
        print(f"  [best] saved at step {best_step} (PDE_B={best_loss:.3e})")
    return total_val, step, opt.state_dict()


def _gate_vs_data(det, dataset, tau_sep, Z_c, device, tol: float) -> float:
    """Report fixed endpoint vs data; this does not measure learned convergence.

    An error here may indicate timing, interface interpolation or model
    mismatch. More optimizer steps cannot change a hard endpoint value.
    """
    with torch.no_grad():
        tau_p = torch.tensor([[tau_sep]], device=device, dtype=torch.float32)
        Z_p   = torch.tensor([[Z_c]],     device=device, dtype=torch.float32)
        rho_p, u_p, P_p = det(tau_p, Z_p)
        rho_d, u_d, P_d = dataset.state_at(tau_p, Z_p)
    u_err   = abs(float(u_p) - float(u_d)) / max(abs(float(u_d)), 1.0)
    P_err   = abs(float(P_p) - float(P_d)) / max(abs(float(P_d)), 1.0)
    rho_err = abs(float(rho_p) - float(rho_d)) / max(abs(float(rho_d)), 1.0)
    print(f"  [endpoint-data diagnostic] A @ (t_sep, R_c) vs d3plot data "
          f"(tol={100*tol:.0f}% on u,P):")
    print(f"    rho: pred={float(rho_p):8.1f}  data={float(rho_d):8.1f}  "
          f"err={100*rho_err:6.1f}%   (diagnostic only)")
    print(f"    u:   pred={float(u_p):8.0f}  data={float(u_d):8.0f}  "
          f"err={100*u_err:6.1f}%")
    print(f"    P:   pred={float(P_p)/1e6:8.1f}MPa  data={float(P_d)/1e6:8.1f}MPa  "
          f"err={100*P_err:6.1f}%")
    return max(u_err, P_err)


# ============================================================ Phase A driver


def train_detonation(cfg: dict, dataset: D3plotLineDataset, sep_bundle,
                     device: torch.device) -> HardDetNetConstraint:
    cj = sep_bundle.cj_bundle
    tau_sep_gate = float(getattr(dataset, "tau_sep", sep_bundle.t_sep))
    Z_c_gate = float(getattr(dataset, "Z_c", sep_bundle.R_c))
    print(f"\n=== Phase A — DetonationNet "
          f"(tau_sep={tau_sep_gate*1e6:.2f}, Z_c={Z_c_gate:.4f}) ===")

    det, _ = build_networks(
        cfg["networks"],
        rho_ref_A=cj.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"], P_ref_A=cj.P_CJ,
    )
    det = det.to(device)
    det_train = _constrain_detonation(cfg, dataset, sep_bundle, det, device)
    expected_constraint = constraint_metadata(det_train)
    tau_sep, Z_c = det_train.tau_sep, det_train.Z_c
    ckpt_path = _ckpt_path(cfg, "detonation")
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # A fixed endpoint cannot be used to decide whether a raw/old model
        # has converged. Reuse only completed snapshots of this constraint.
        if saved.get("training_complete") and saved.get("constraint") == expected_constraint:
            restored, _ = _load_checkpoint(ckpt_path, det)
            _gate_vs_data(restored, dataset, tau_sep, Z_c, device,
                          cfg["training"]["gate_rel_tol"])
            print("  [checkpoint] reusing completed A snapshot with matching constraints")
            return restored
        print("  [checkpoint] legacy/incomplete/different A constraints; training a new A model")
    print(f"  [hard-A] CJ and analytical products-side outflow anchors active; "
          f"end=(tau={tau_sep:.6g}, Z={Z_c:.6g})")

    losses = _build_detonation_losses(cfg, dataset, det_train, sep_bundle, device)
    weights_A1 = cfg["loss_weights"]["A1"]
    A1 = cfg["training"]["A1"]

    _run_substage("A1", det_train, losses,
                  keys=["data_A", "IC_A", "BC_A_slope"],
                  weights=weights_A1, steps=A1["steps"], lr=A1["lr"],
                  log_n=cfg["training"].get("log_interval", 100))

    # A2: supports split A2a/A2b (unified config) or single A2 (legacy config)
    total_A2_steps = 0
    default_A2_weights = cfg["loss_weights"].get("A2",
                          cfg["loss_weights"].get("A2a", {"data_A": 2.0, "IC_A": 0.1,
                                      "BC_A_slope": 0.001, "PDE_A": 10.0}))

    if "A2a" in cfg["training"]:
        A2a = cfg["training"]["A2a"]
        w2a = cfg["loss_weights"].get("A2a", default_A2_weights)
        _run_substage("A2a", det_train, losses,
                      keys=["data_A", "IC_A", "BC_A_slope", "PDE_A"],
                      weights=w2a, steps=A2a["steps"], lr=A2a["lr"],
                      log_n=cfg["training"].get("log_interval", 100))
        total_A2_steps += A2a["steps"]

    if "A2b" in cfg["training"]:
        A2b = cfg["training"]["A2b"]
        w2b = cfg["loss_weights"].get("A2b", default_A2_weights)
        _run_substage("A2b", det_train, losses,
                      keys=["data_A", "IC_A", "BC_A_slope", "PDE_A"],
                      weights=w2b, steps=A2b["steps"], lr=A2b["lr"],
                      log_n=cfg["training"].get("log_interval", 100),
                      best_ckpt=str(_ckpt_path(cfg, "detonation_best")),
                      best_net=det_train,
                      best_ckpt_key="PDE_A")
        total_A2_steps += A2b["steps"]

    if total_A2_steps == 0:
        # Fallback: single A2 stage (legacy config)
        A2 = cfg["training"]["A2"]
        weights_A2 = cfg["loss_weights"]["A2"]
        _run_substage("A2", det_train, losses,
                      keys=["data_A", "IC_A", "BC_A_slope", "PDE_A"],
                      weights=weights_A2, steps=A2["steps"], lr=A2["lr"],
                      log_n=cfg["training"].get("log_interval", 100))
        total_A2_steps = A2["steps"]

    # Compare the ACTUAL predictor with data. This fixed-value mismatch is
    # reported independently of hard-constraint satisfaction and convergence.
    gate_tol = cfg["training"]["gate_rel_tol"]
    max_err = _gate_vs_data(det_train, dataset, tau_sep, Z_c, device, gate_tol)
    if max_err > gate_tol:
        print("  [WARN] analytical endpoint differs from data beyond tolerance. "
              "Check event coordinates/material parameters; extra A2 steps cannot fix this anchor.")
    _save_checkpoint(_ckpt_path(cfg, "detonation"), det_train,
                     A1["steps"] + total_A2_steps, loss=0.0, training_complete=True)
    return det_train


# ============================================================ Phase B driver


def train_airshock(cfg: dict, dataset, sep_bundle,
                   frozen_det: HardDetNetConstraint, device: torch.device) -> HardContactConstrainedASN:
    tau_sep = float(getattr(dataset, "tau_sep", sep_bundle.t_sep))
    Z_c = float(getattr(dataset, "Z_c", sep_bundle.R_c))
    print(f"\n=== Phase B — AirShockNet "
          f"(tau_sep={tau_sep*1e6:.2f}, Z_c={Z_c:.4f}) ===")
    _, asn = build_networks(
        cfg["networks"],
        rho_ref_B=cfg["air"]["rho_a"],
        u_ref_B=cfg["domain"]["u_ref_B"],
        P_ref_B=cfg["domain"]["P_ref_B"],
        t_scale_B=float(cfg["networks"].get("air_shock", {}).get("t_scale", cfg["domain"].get("t_ref_B", 1.0e-3))),
        r_scale_B=float(cfg["networks"].get("air_shock", {}).get("r_scale", cfg["domain"].get("r_scale_B", 2.64))),
    )
    asn = asn.to(device)
    frozen_det = frozen_det.to(device)
    if not isinstance(frozen_det, HardDetNetConstraint):
        raise ValueError("Phase B requires the constrained A predictor; re-train/load Phase A first.")
    expected = constraint_metadata(_constrain_detonation(cfg, dataset, sep_bundle,
                                    frozen_det.net, device))
    if constraint_metadata(frozen_det) != expected:
        raise ValueError("Phase B requires A with the current analytical endpoint constraint.")

    # A supplies products-side u/P at the initial outflow matching point.
    # The AIR density is obtained from RH, never copied from products.
    with torch.no_grad():
        t_p = torch.tensor([[tau_sep]], device=device, dtype=torch.float32)
        r_p = torch.tensor([[Z_c]],     device=device, dtype=torch.float32)
        _, u_c, P_c = frozen_det(t_p, r_p)
        rho_c = air_contact_density(P_c, gamma=cfg["air"]["gamma"],
                                    rho_a=cfg["air"]["rho_a"], P_a=cfg["air"]["P_a"])
    sm = cfg["sampling"]
    asn_wrapped = HardContactConstrainedASN(
        asn,
        tau_sep=tau_sep, Z_c=Z_c,
        target_rho=rho_c, target_u=u_c, target_P=P_c,
        density_source="air_rh",
        tau_t=float(sm.get("tau_t", 1e-6)),
        tau_r=float(sm.get("tau_r", 1e-3)),
    )
    print(f"  [hard-IC] tau_t={sm.get('tau_t', 1e-6):.1e}  "
          f"tau_r={sm.get('tau_r', 1e-3):.1e}  "
          f"target=(rho={float(rho_c):.1f}, u={float(u_c):.0f}, P={float(P_c)/1e6:.2f} MPa)")

    losses = _build_air_shock_losses(cfg, dataset, asn_wrapped, frozen_det, sep_bundle, device)
    weights_B1 = cfg["loss_weights"]["B1"]
    B1 = cfg["training"]["B1"]

    _, _, _ = _run_substage("B1", asn_wrapped, losses,
                  keys=["data_B", "IC_B"],
                  weights=weights_B1, steps=B1["steps"], lr=B1["lr"],
                  log_n=cfg["training"].get("log_interval", 100))

    total_steps = B1["steps"]

    # B2 (simple): all five losses simultaneously — the original design.
    if "B2" in cfg["training"]:
        B2 = cfg["training"]["B2"]
        w2  = cfg["loss_weights"]["B2"]
        _run_substage("B2", asn_wrapped, losses,
                      keys=["data_B", "IC_B", "PDE_B", "RH_B", "BC_B_outflow"],
                      weights=w2, steps=B2["steps"], lr=B2["lr"],
                      log_n=cfg["training"].get("log_interval", 100),
                      grad_clip=float(cfg["training"].get("grad_clip_B", 1.0)),
                      adaptive=cfg["training"].get("adaptive", False),
                      best_ckpt=str(_ckpt_path(cfg, "air_shock_best")),
                      best_net=asn_wrapped)
        total_steps += B2["steps"]

    else:
        # ---- curriculum-learning fallback (B2a → B2b → B2c) ----
        # B2a: data + RH + BC  (no PDE)
        if "B2a" in cfg["training"]:
            B2a = cfg["training"]["B2a"]
            w2a  = cfg["loss_weights"]["B2a"]
            _run_substage("B2a", asn_wrapped, losses,
                          keys=["data_B", "RH_B", "BC_B_outflow"],
                          weights=w2a, steps=B2a["steps"], lr=B2a["lr"],
                          log_n=cfg["training"].get("log_interval", 100),
                          grad_clip=float(cfg["training"].get("grad_clip_B", 1.0)))
            total_steps += B2a["steps"]

        # B2b: add PDE with low initial weight + adaptive balancing
        if "B2b" in cfg["training"]:
            B2b = cfg["training"]["B2b"]
            w2b  = cfg["loss_weights"]["B2b"]
            _run_substage("B2b", asn_wrapped, losses,
                          keys=["data_B", "PDE_B", "RH_B", "BC_B_outflow"],
                          weights=w2b, steps=B2b["steps"], lr=B2b["lr"],
                          log_n=cfg["training"].get("log_interval", 100),
                          grad_clip=float(cfg["training"].get("grad_clip_B", 1.0)),
                          adaptive=cfg["training"].get("adaptive", True),
                          best_ckpt=str(_ckpt_path(cfg, "air_shock_best")),
                          best_net=asn_wrapped)
            total_steps += B2b["steps"]

        # B2c: shock sharpening with moderate PDE weight + lower LR
        if "B2c" in cfg["training"]:
            B2c = cfg["training"]["B2c"]
            w2c  = cfg["loss_weights"]["B2c"]
            _run_substage("B2c", asn_wrapped, losses,
                          keys=["data_B", "PDE_B", "RH_B", "BC_B_outflow"],
                          weights=w2c, steps=B2c["steps"], lr=B2c["lr"],
                          log_n=cfg["training"].get("log_interval", 100),
                          grad_clip=float(cfg["training"].get("grad_clip_B", 1.0)),
                          best_ckpt=str(_ckpt_path(cfg, "air_shock_best")),
                          best_net=asn_wrapped)
            total_steps += B2c["steps"]

    _save_checkpoint(_ckpt_path(cfg, "air_shock"), asn_wrapped, total_steps,
                     0.0, training_complete=True)
    return asn_wrapped


# ============================================================ main


def _make_sep_bundle(cfg: dict, dataset: D3plotLineDataset):
    """Construct SeparationStateBundle from dataset metadata.

    The dataset must carry the spherical-charge radius ``R_0`` (written by
    the extractor).  No cube-to-sphere conversion is performed.
    """
    if dataset.R_0 is None:
        raise RuntimeError(
            f"Dataset {dataset.dir} is missing the R_0 metadata field; "
            f"re-run `python -m data.extract_d3plot --config <yaml>` with the "
            f"updated extractor that writes the spherical charge radius."
        )
    tnt = TNTParams(rho_TNT=dataset.rho_TNT)
    cj_b = compute_cj_state(tnt)
    return compute_separation_state(tnt, R_0=float(dataset.R_0), cj_bundle=cj_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spherical TNT PINN trainer")
    parser.add_argument("--config", default="configs/tnt_spherical_50mm.yaml")
    parser.add_argument("--net", choices=["detonation", "airshock"], default=None,
                        help="Train just one phase (default: both, in order).")
    parser.add_argument("--all", action="store_true", help="Train both phases sequentially.")
    parser.add_argument("--extracted", default=None,
                        help="Override extracted/ directory (default cfg.data.extracted_dir).")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    device = _resolve_device(cfg)
    print(f"Device: {device}")
    torch.manual_seed(cfg["training"]["seed"])

    # Support both unified (multi-radius) and single-radius configs
    if "radii" in cfg["data"]:
        # Unified multi-radius config
        t_data_min = float(cfg["sampling"].get("t_data_min", 8.0e-6))
        dataset = MultiRadiusDataset(cfg["data"]["radii"], device=device,
                                     t_data_min_phys=t_data_min)
        print(f"\n[data] unified dataset: {len(cfg['data']['radii'])} radii")
        sep = dataset.sep_bundle
        tau_sep = dataset.tau_sep
        Z_c = dataset.Z_c
        print(f"[sep ] tau_sep={tau_sep*1e6:.2f}  Z_c={Z_c:.4f}  "
              f"P_x={sep.P_x/1e9:.3f} GPa")
    else:
        # Single-radius config (backward compatible)
        extracted_dir = args.extracted or cfg["data"]["extracted_dir"]
        print(f"\n[data] loading {extracted_dir}")
        dataset = D3plotLineDataset(extracted_dir, device=device)
        sep = _make_sep_bundle(cfg, dataset)
        print(f"[sep ] R_0={sep.R_0*1e3:.2f} mm  R_c={sep.R_c*1e3:.2f} mm  "
              f"t_sep={sep.t_sep*1e6:.2f} us  P_x={sep.P_x/1e9:.3f} GPa")

    do_A = args.all or args.net in (None, "detonation")
    do_B = args.all or args.net == "airshock"

    det = None
    if do_A:
        det = train_detonation(cfg, dataset, sep, device)
    if do_B:
        if det is None:
            # Need to load from checkpoint
            det, _ = build_networks(cfg["networks"],
                                    rho_ref_A=sep.cj_bundle.rho_CJ,
                                    u_ref_A=cfg["domain"]["u_ref_A"],
                                    P_ref_A=sep.cj_bundle.P_CJ)
            ckpt = _ckpt_path(cfg, "detonation")
            if not ckpt.exists():
                print(f"[error] {ckpt} not found; run Phase A first.")
                sys.exit(2)
            expected = constraint_metadata(_constrain_detonation(cfg, dataset, sep, det, device))
            det, saved = _load_checkpoint(ckpt, det)
            if not saved.get("training_complete") or saved.get("constraint") != expected:
                raise RuntimeError("Saved A constraints do not match this dataset/config; train A first.")
            det = det.to(device)
        train_airshock(cfg, dataset, sep, det, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
