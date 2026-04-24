"""Post-process LS-DYNA d3plot output into PINN-ready CSV/JSON files.

Reads a d3plot binary produced by the LS-DYNA keyword generators,
extracts (rho, u, P) on the mesh at every saved timestep, identifies
the shock front R_s(t) and the TNT/air contact surface R_c(t), detects
the separation time t_sep, and writes:

    <out_dir>/snapshot.csv   columns: r, rho, u, P           (at t = t_sep)
    <out_dir>/contact.csv    columns: t, P_c, u_c, R_c
    <out_dir>/shock.csv      columns: t, R_s, D_s
    <out_dir>/metadata.json  keys:    t_sep, R_c_sep, R_s_sep

Supports both 2D axisymmetric (shell, ELFORM=15) and 3D wedge (solid,
ELFORM=11) d3plot files.  All values are converted from g-mm-ms back to
SI (kg, m, s, Pa) so they plug directly into :class:`FileLoader`.

Usage
-----
    python -m data.extract_lsdyna_output --d3plot /path/to/d3plot --out-dir lsdyna_data/

Requires ``lasso-python`` (pip install lasso-python).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from lasso.dyna import D3plot, ArrayType
except ImportError as exc:
    raise ImportError(
        "lasso-python is required. Install with `pip install lasso-python`."
    ) from exc


# ===================== Unit conversion: g-mm-ms -> SI =====================
MM_TO_M    = 1.0e-3
MS_TO_S    = 1.0e-3
DENS_TO_SI = 1.0e6       # g/mm^3 -> kg/m^3
VEL_TO_SI  = 1.0         # mm/ms  -> m/s
PRES_TO_SI = 1.0e6       # MPa    -> Pa


# ===================== Auto-detect element type =====================


def _is_solid(d3: D3plot) -> bool:
    """Return True if the d3plot contains solid (hex) elements, else shells."""
    return ArrayType.element_solid_node_indexes in d3.arrays


def _conn(d3: D3plot) -> np.ndarray:
    if _is_solid(d3):
        return d3.arrays[ArrayType.element_solid_node_indexes]
    return d3.arrays[ArrayType.element_shell_node_indexes]


def _part_ids(d3: D3plot) -> np.ndarray:
    if _is_solid(d3):
        return d3.arrays[ArrayType.element_solid_part_indexes]
    return d3.arrays[ArrayType.element_shell_part_indexes]


def _stress(d3: D3plot, state: int) -> np.ndarray:
    if _is_solid(d3):
        return d3.arrays[ArrayType.element_solid_stress][state]
    return d3.arrays[ArrayType.element_shell_stress][state]


def _history_vars(d3: D3plot, state: int):
    try:
        if _is_solid(d3):
            return d3.arrays[ArrayType.element_solid_history_variables][state]
        return d3.arrays[ArrayType.element_shell_history_vars][state]
    except KeyError:
        return None


# ===================== LS-DYNA d3plot extraction =====================


def load_d3plot(d3plot_path: str) -> D3plot:
    print(f"[extract] loading d3plot: {d3plot_path}")
    # Load WITHOUT state_array_filter so global_timesteps is included.
    d3 = D3plot(d3plot_path)

    times = d3.arrays.get(ArrayType.global_timesteps)
    if times is None:
        raise RuntimeError(
            "d3plot has no timestep data.  Available keys: "
            + ", ".join(sorted(str(k) for k in d3.arrays.keys()))
        )
    n_states = times.shape[0]

    is_solid = _is_solid(d3)
    conn = _conn(d3)
    n_elem = conn.shape[0]
    elem_type = "solid (hex8)" if is_solid else "shell"
    print(f"[extract]   states: {n_states}, {elem_type} elements: {n_elem}")
    return d3


def element_centroids_r(d3: D3plot, state: int) -> np.ndarray:
    """Radial coordinate (mm) of each element's centroid at ``state``.

    For both 2D (shell, r = x) and 3D (solid, r = sqrt(x²+y²+z²)).
    """
    X0 = d3.arrays[ArrayType.node_coordinates]             # (N_node, 3)

    disp_arr = d3.arrays.get(ArrayType.node_displacement)
    if disp_arr is not None:
        X = X0 + disp_arr[state]
    else:
        X = X0

    conn = _conn(d3)  # (N_ele, 4 or 8)

    if _is_solid(d3):
        # 3D wedge: r = distance from origin
        xc = X[conn, 0].mean(axis=1)
        yc = X[conn, 1].mean(axis=1)
        zc = X[conn, 2].mean(axis=1)
        return np.sqrt(xc**2 + yc**2 + zc**2)
    else:
        # 2D axisymmetric: r = x coordinate
        return X[conn, 0].mean(axis=1)


def element_centroid_velocity_r(d3: D3plot, state: int) -> np.ndarray:
    """Radial velocity (mm/ms) of each element centroid at ``state``."""
    vel = d3.arrays[ArrayType.node_velocity][state]        # (N_node, 3)
    conn = _conn(d3)

    if _is_solid(d3):
        # 3D: project velocity onto radial direction
        X0 = d3.arrays[ArrayType.node_coordinates]
        disp_arr = d3.arrays.get(ArrayType.node_displacement)
        X = X0 + disp_arr[state] if disp_arr is not None else X0

        xc = X[conn, 0].mean(axis=1)
        yc = X[conn, 1].mean(axis=1)
        zc = X[conn, 2].mean(axis=1)
        rc = np.sqrt(xc**2 + yc**2 + zc**2)

        vx = vel[conn, 0].mean(axis=1)
        vy = vel[conn, 1].mean(axis=1)
        vz = vel[conn, 2].mean(axis=1)
        # v_r = v . r_hat
        return (vx * xc + vy * yc + vz * zc) / (rc + 1e-30)
    else:
        return vel[conn, 0].mean(axis=1)


def element_pressure(d3: D3plot, state: int) -> np.ndarray:
    """Scalar pressure (MPa) per element at ``state``."""
    stress = _stress(d3, state)     # (N_ele, N_ip, 6)
    sxx = stress[..., 0].mean(axis=-1)
    syy = stress[..., 1].mean(axis=-1)
    szz = stress[..., 2].mean(axis=-1)
    return -(sxx + syy + szz) / 3.0


def element_density(d3: D3plot, state: int) -> np.ndarray:
    """Density (g/mm^3) per element at ``state``."""
    hv = _history_vars(d3, state)
    if hv is not None:
        if hv.ndim == 3 and hv.shape[-1] >= 1:
            return hv[..., 0].mean(axis=-1)
        if hv.ndim == 2 and hv.shape[-1] >= 1:
            return hv[..., 0]

    part_ids = _part_ids(d3)
    return np.where(part_ids == 0, 1.63e-3, 1.225e-6)


# ===================== Shock & contact detection =====================


def detect_shock_front(r: np.ndarray, P: np.ndarray,
                       P_a: float = 0.101325) -> float:
    """Return R_s (mm): outermost element with P > 1.5 * P_a."""
    order = np.argsort(r)
    r_s, P_s = r[order], P[order]
    thr = 1.5 * P_a
    above = P_s > thr
    if not above.any():
        return float(r_s[0])
    idx = int(np.where(above)[0].max())
    if idx + 1 >= len(r_s):
        return float(r_s[idx])
    P0, P1 = P_s[idx], P_s[idx + 1]
    r0, r1 = r_s[idx], r_s[idx + 1]
    frac = (thr - P0) / (P1 - P0) if P1 != P0 else 0.0
    return float(r0 + frac * (r1 - r0))


def detect_contact_radius(r: np.ndarray, part_ids: np.ndarray,
                           tnt_part_idx: int = 0) -> float:
    """Return R_c (mm): outermost radius of TNT-products elements."""
    mask = part_ids == tnt_part_idx
    if not mask.any():
        return 0.0
    return float(r[mask].max())


def detect_separation_state(times: np.ndarray, P_c: np.ndarray,
                             P_a: float = 0.101325,
                             P_ratio: float = 20.0) -> int:
    """Return the state index where P_c drops below P_ratio * P_a."""
    thr = P_ratio * P_a
    below = P_c < thr
    if not below.any():
        return len(times) - 1
    return int(np.where(below)[0].min())


# ===================== Main extraction pipeline =====================


def extract(d3plot_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d3 = load_d3plot(d3plot_path)

    times_ms = d3.arrays[ArrayType.global_timesteps]
    part_ids = _part_ids(d3)
    n_states = len(times_ms)

    print(f"[extract] scanning {n_states} states to build histories...")
    R_c_mm   = np.zeros(n_states)
    R_s_mm   = np.zeros(n_states)
    P_c_MPa  = np.zeros(n_states)
    u_c_mmms = np.zeros(n_states)
    tnt_part = 0

    for k in range(n_states):
        r_k = element_centroids_r(d3, k)
        u_k = element_centroid_velocity_r(d3, k)
        P_k = element_pressure(d3, k)

        R_s_mm[k] = detect_shock_front(r_k, P_k, P_a=0.101325)
        R_c_mm[k] = detect_contact_radius(r_k, part_ids, tnt_part_idx=tnt_part)

        tnt_mask = part_ids == tnt_part
        if tnt_mask.any():
            idx_c = np.argmin(np.abs(r_k[tnt_mask] - R_c_mm[k]))
            tnt_idx_global = np.where(tnt_mask)[0][idx_c]
            P_c_MPa[k]  = P_k[tnt_idx_global]
            u_c_mmms[k] = u_k[tnt_idx_global]

    # Separation detection
    sep_idx = detect_separation_state(times_ms, P_c_MPa)
    t_sep_ms = float(times_ms[sep_idx])
    print(f"[extract] separation at state {sep_idx}, t_sep = {t_sep_ms:.4f} ms")
    print(f"[extract]   R_c_sep = {R_c_mm[sep_idx]:.2f} mm, "
          f"R_s_sep = {R_s_mm[sep_idx]:.2f} mm")

    # ---------------------- Snapshot at t_sep ----------------------
    r_snap_mm     = element_centroids_r(d3, sep_idx)
    u_snap_mmms   = element_centroid_velocity_r(d3, sep_idx)
    P_snap_MPa    = element_pressure(d3, sep_idx)
    rho_snap_gmm3 = element_density(d3, sep_idx)

    air_mask = part_ids != tnt_part
    in_domain = (
        air_mask
        & (r_snap_mm >= R_c_mm[sep_idx])
        & (r_snap_mm <= R_s_mm[sep_idx])
    )
    order = np.argsort(r_snap_mm[in_domain])
    r_SI   = r_snap_mm[in_domain][order]       * MM_TO_M
    rho_SI = rho_snap_gmm3[in_domain][order]   * DENS_TO_SI
    u_SI   = u_snap_mmms[in_domain][order]      * VEL_TO_SI
    P_SI   = P_snap_MPa[in_domain][order]       * PRES_TO_SI

    np.savetxt(
        out_dir / "snapshot.csv",
        np.column_stack([r_SI, rho_SI, u_SI, P_SI]),
        delimiter=",", header="r,rho,u,P", comments="", fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir / 'snapshot.csv'}  ({len(r_SI)} points)")

    # ---------------------- Contact history ----------------------
    t_mask = times_ms >= t_sep_ms - 1e-12
    t_s    = times_ms[t_mask]   * MS_TO_S
    P_c_SI = P_c_MPa[t_mask]    * PRES_TO_SI
    u_c_SI = u_c_mmms[t_mask]   * VEL_TO_SI
    R_c_SI = R_c_mm[t_mask]     * MM_TO_M

    np.savetxt(
        out_dir / "contact.csv",
        np.column_stack([t_s, P_c_SI, u_c_SI, R_c_SI]),
        delimiter=",", header="t,P_c,u_c,R_c", comments="", fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir / 'contact.csv'}   ({len(t_s)} points)")

    # ---------------------- Shock history ----------------------
    R_s_SI = R_s_mm * MM_TO_M
    t_SI   = times_ms * MS_TO_S
    D_s_SI = np.gradient(R_s_SI, t_SI)

    np.savetxt(
        out_dir / "shock.csv",
        np.column_stack([t_SI, R_s_SI, D_s_SI]),
        delimiter=",", header="t,R_s,D_s", comments="", fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir / 'shock.csv'}     ({len(t_SI)} points)")

    # ---------------------- Metadata ----------------------
    meta = {
        "t_sep":   t_sep_ms * MS_TO_S,
        "R_c_sep": float(R_c_mm[sep_idx]) * MM_TO_M,
        "R_s_sep": float(R_s_mm[sep_idx]) * MM_TO_M,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] wrote {out_dir / 'metadata.json'}: {meta}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract PINN training data from LS-DYNA d3plot",
    )
    p.add_argument("--d3plot", required=True, help="path to d3plot file")
    p.add_argument("--out-dir", default="lsdyna_data", help="output directory")
    args = p.parse_args()
    extract(args.d3plot, Path(args.out_dir))


if __name__ == "__main__":
    main()
