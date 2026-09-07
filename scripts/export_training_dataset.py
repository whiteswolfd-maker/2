#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the extracted LS-DYNA/PINN data as a framework-agnostic training set.

Reads the standard extraction outputs of ``data.extract_d3plot``
(raw_line.npz + shock.csv + contact.csv + metadata.json) and writes a clean,
self-describing dataset that ANY neural-network framework can load (PyTorch,
TensorFlow, JAX, or plain NumPy) -- no dependency on this project's code.

Output layout
-------------
    <out>/
      dataset.npz          all arrays in SI units (m, kg, s, Pa)
      fields.json          array names, shapes, units, descriptions
      metadata.json        R_0, t_sep, CJ state, extraction settings, ...
      README.md            how to load the data in each framework
      slices/              1D radial slices P(r) at selected times (CSV)

dataset.npz arrays
------------------
    t        (N_t,)            simulation time (s)
    r        (N_x,)            radial coordinate (m), 0 .. x_end
    rho      (N_t, N_x)        density (kg/m^3)
    u        (N_t, N_x)        radial velocity (m/s, outward +)
    P        (N_t, N_x)        pressure (Pa)
    R_s      (N_t,)            shock-front radius (m)
    R_c      (N_t,)            product-air contact radius (m)

The field ordering rho,u,P with a shared (t,r) grid is the canonical layout
for physics-informed / operator-learning datasets.  Each sample at (t_i, r_j)
is simply ``(t[i], r[j]) -> (rho, u, P)[i, j]``.

Usage
-----
    python scripts/export_training_dataset.py --input extracted/ --out dataset/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


FIELDS = [
    ("t",   "N_t",      "simulation time (s)"),
    ("r",   "N_x",      "radial coordinate (m)"),
    ("rho", "(N_t,N_x)", "density (kg/m^3)"),
    ("u",   "(N_t,N_x)", "radial velocity (m/s, outward positive)"),
    ("P",   "(N_t,N_x)", "pressure (Pa)"),
    ("R_s", "N_t",      "shock-front radius (m)"),
    ("R_c", "N_t",      "product-air contact radius (m)"),
]


def _load_float_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def export(in_dir: Path, out_dir: Path, times: list[float] | None = None,
           max_frames: int = 0):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "slices").mkdir(exist_ok=True)

    data = np.load(in_dir / "raw_line.npz")
    t = np.asarray(data["t"])
    x = np.asarray(data["x"])
    rho = np.asarray(data["rho"])
    u = np.asarray(data["u"])
    P = np.asarray(data["P"])

    # optional temporal sub-sampling (every max_frames-th frame)
    if max_frames and max_frames > 1:
        idx = np.arange(0, len(t), max_frames)
        t, rho, u, P = t[idx], rho[idx], u[idx], P[idx]

    shock = _load_float_csv(in_dir / "shock.csv")
    contact = _load_float_csv(in_dir / "contact.csv")
    R_s = np.interp(t, shock[:, 0], shock[:, 1])
    R_c = np.interp(t, contact[:, 0], contact[:, 1])

    meta = {}
    mpath = in_dir / "metadata.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))

    np.savez_compressed(
        out_dir / "dataset.npz",
        t=t, r=x, rho=rho, u=u, P=P, R_s=R_s, R_c=R_c,
    )

    fields = [
        {"name": n, "shape": s, "description": d}
        for (n, s, d) in FIELDS
    ]
    (out_dir / "fields.json").write_text(
        json.dumps({"n_t": int(len(t)), "n_x": int(len(x)), "fields": fields},
                   indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1D radial pressure slices at requested times (for quick inspection / MLP input)
    if times is not None:
        for tt_us in times:
            tt = tt_us * 1e-6           # argument is microseconds, t is seconds
            i = int(np.argmin(np.abs(t - tt)))
            with open(out_dir / "slices" / f"P_r_t{t[i]*1e6:.1f}us.csv", "w",
                      newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["r_m", "P_Pa", "rho_kgm3", "u_ms"])
                for j in range(len(x)):
                    wr.writerow([f"{x[j]:.6e}", f"{P[i, j]:.6e}",
                                 f"{rho[i, j]:.6e}", f"{u[i, j]:.6e}"])

    readme = f"""# Training dataset (R_0 = {meta.get('R_0', '?')} m)

Extracted from an LS-DYNA multi-material ALE spherical TNT blast simulation.

## Load in any framework

    import numpy as np
    d = np.load("dataset.npz")
    t, r = d["t"], d["r"]            # (N_t,), (N_x,)
    rho, u, P = d["rho"], d["u"], d["P"]   # each (N_t, N_x)
    R_s, R_c = d["R_s"], d["R_c"]    # (N_t,) shock / contact radii

### PyTorch

    import torch
    X = torch.stack(torch.meshgrid(torch.from_numpy(t), torch.from_numpy(r), indexing="ij"), dim=-1)  # (N_t, N_x, 2)
    Y = torch.stack([torch.from_numpy(rho), torch.from_numpy(u), torch.from_numpy(P)], dim=-1)         # (N_t, N_x, 3)

### TensorFlow / JAX

    X = tf.stack(tf.meshgrid(t, r, indexing="ij"), axis=-1)   # (N_t, N_x, 2)
    X = jnp.stack(jnp.meshgrid(t, r, indexing="ij"), axis=-1) # (N_t, N_x, 2)

## Physical anchors (from metadata.json)

- R_0 (charge radius), t_sep (separation time), R_c / P_x / u_x / rho_x at the
  product-air contact from the Sec.4.2.1 Riemann match.
- Shock front R_s(t) and contact R_c(t) are included as supervised targets.

## Units

SI: m, kg, s, Pa.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[export] wrote {out_dir}/dataset.npz  ({len(t)} x {len(x)}), "
          f"fields.json, metadata.json, README.md")
    print(f"[export] arrays: t {t.shape}, rho/u/P {rho.shape}, R_s/R_c {R_s.shape}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("extracted/"),
                    help="extraction output dir (raw_line.npz etc.)")
    ap.add_argument("--out", type=Path, default=Path("dataset/"),
                    help="output dataset dir")
    ap.add_argument("--times-us", type=str, default="",
                    help="comma-separated times (us) to write P(r) CSV slices")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="keep every Nth frame (temporal sub-sampling)")
    args = ap.parse_args()

    times = [float(x) for x in args.times_us.split(",") if x.strip()]
    return export(args.input, args.out, times=times or None,
                  max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
