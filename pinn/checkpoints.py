"""Save and restore the predictor actually used by the losses, including anchors.

Weights, output/input scales and hard-constraint metadata travel together in
both best and final checkpoints. These files are model snapshots, not full
optimizer/RNG resume checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import torch

from pinn.networks import HardContactConstrainedASN, HardDetNetConstraint


_SCALES = ("rho_ref", "u_ref", "P_ref", "tau_scale", "Z_scale",
           "eps_origin", "t_scale", "r_scale")


def constraint_metadata(net) -> dict | None:
    if isinstance(net, HardDetNetConstraint):
        names = ("tau_sep", "Z_c", "Z_R0", "rho_cj", "u_cj", "P_cj",
                 "rho_x", "u_x", "P_x", "tau_t", "tau_r", "tau_t_cj", "tau_r_cj")
        params = {name: float(getattr(net, name)) for name in names}
        params["use_contact"] = net.use_contact
        return {"kind": "detonation", "projection": "cardinal_gaussian_v1", "params": params}
    if isinstance(net, HardContactConstrainedASN):
        names = ("tau_sep", "Z_c", "target_rho", "target_u", "target_P", "tau_t", "tau_r")
        params = {name: float(getattr(net, name)) for name in names}
        params["density_source"] = net.density_source
        return {"kind": "air_shock", "params": params}
    return None


def save_checkpoint(path, net, step: int, loss: float, *, training_complete=False) -> None:
    hc = constraint_metadata(net)
    raw = net.net if hc else net
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2, "step": step, "loss": loss,
        "training_complete": bool(training_complete),
        "state_dict": raw.state_dict(), "constraint": hc,
        "network_type": type(raw).__name__,
        "scales": {name: float(getattr(raw, name)) for name in _SCALES if hasattr(raw, name)},
        "siren_omega0": {name: float(layer.omega0) for name, layer in raw.named_modules()
                         if hasattr(layer, "omega0")},
    }, path)


def load_checkpoint(path, raw, *, require_constraints=False):
    """Return (restored predictor, metadata); callers must use the returned net.

    Old snapshots remain readable for comparison with a warning. They are
    rejected for the new A->B training path, because the missing A anchor or
    old shared-product density cannot be retroactively called new training.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    hc = ckpt.get("constraint")
    valid = ckpt.get("format_version") == 2 and hc is not None
    if valid and hc["kind"] == "detonation":
        valid = hc.get("projection") == "cardinal_gaussian_v1" and hc["params"]["use_contact"]
    elif valid and hc["kind"] == "air_shock":
        valid = hc["params"].get("density_source") == "air_rh"
    elif valid:
        raise ValueError(f"Unknown constraint kind in {path}: {hc['kind']}")
    if require_constraints and not valid:
        raise ValueError(f"{path} lacks the current hard-constraint model. Re-train Phase A, then B.")
    if ckpt.get("network_type", type(raw).__name__) != type(raw).__name__:
        raise ValueError(f"Network architecture in {path} does not match the supplied config.")
    raw.load_state_dict(ckpt["state_dict"])
    for name, value in ckpt.get("scales", {}).items():
        if name not in _SCALES or not hasattr(raw, name):
            raise ValueError(f"Unsupported saved scale: {name}")
        setattr(raw, name, value)
    layers = dict(raw.named_modules())
    for name, value in ckpt.get("siren_omega0", {}).items():
        layers[name].omega0 = value
    if ckpt.get("format_version") != 2:
        warnings.warn(f"Legacy checkpoint {path}: missing full constraints/scales; "
                      "re-train A and B for the corrected model.", stacklevel=2)
        # Preserve the old B predictor for historical comparisons only.
        old_meta = path.parent / "air_shock_hc_meta.pt"
        if path.stem.startswith("air_shock") and old_meta.exists():
            old = torch.load(old_meta, map_location="cpu", weights_only=True)
            hc = {"kind": "air_shock", "params": {
                "tau_sep": old["t_sep"], "Z_c": old["R_c"],
                **{k: old[k] for k in ("target_rho", "target_u", "target_P", "tau_t", "tau_r")},
                "density_source": "legacy_shared_product",
            }}
    net = raw
    if hc:
        params = dict(hc["params"])
        parameter = next(raw.parameters())
        if hc["kind"] == "detonation":
            if hc.get("projection") != "cardinal_gaussian_v1":
                raise ValueError(f"Unsupported A projection in {path}")
            net = HardDetNetConstraint(raw, **params)
        elif hc["kind"] == "air_shock":
            for key in ("target_rho", "target_u", "target_P"):
                params[key] = torch.as_tensor(params[key], dtype=parameter.dtype, device=parameter.device)
            net = HardContactConstrainedASN(raw, **params)
        else:
            raise ValueError(f"Unknown constraint kind: {hc['kind']}")
        net = net.to(device=parameter.device, dtype=parameter.dtype)
    net.eval()
    return net, ckpt
