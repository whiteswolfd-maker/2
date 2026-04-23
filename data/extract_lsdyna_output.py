"""Post-process LS-DYNA d3plot output into PINN-ready CSV/JSON files.

Reads a d3plot binary produced by ``generate_lsdyna_input.py`` + LS-DYNA,
extracts (rho, u, P) on the 2D axisymmetric mesh at every saved timestep,
identifies the shock front R_s(t) and the TNT-products / air contact
surface R_c(t), detects the separation time t_sep, and writes:

    <out_dir>/snapshot.csv   columns: r, rho, u, P           (at t = t_sep)
    <out_dir>/contact.csv    columns: t, P_c, u_c, R_c
    <out_dir>/shock.csv      columns: t, R_s, D_s
    <out_dir>/metadata.json  keys:    t_sep, R_c_sep, R_s_sep

All values are converted from g-mm-ms (LS-DYNA internal) back to SI
(kg, m, s, Pa) so they plug directly into :class:`FileLoader`.

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
# length: mm  -> m    : 1e-3
# time:   ms  -> s    : 1e-3
# mass:   g   -> kg   : 1e-3
# density: g/mm^3 -> kg/m^3 : 1e-3 / (1e-3)^3 = 1e6
# velocity: mm/ms -> m/s    : 1e-3 / 1e-3 = 1.0  (numerically identical)
# pressure: MPa  -> Pa      : 1e6

MM_TO_M   = 1.0e-3
MS_TO_S   = 1.0e-3
DENS_TO_SI = 1.0e6       # g/mm^3 -> kg/m^3
VEL_TO_SI  = 1.0         # mm/ms  -> m/s
PRES_TO_SI = 1.0e6       # MPa    -> Pa


# ===================== LS-DYNA d3plot extraction =====================


def load_d3plot(d3plot_path: str) -> D3plot:
    print(f"[extract] loading d3plot: {d3plot_path}")
    d3 = D3plot(
        d3plot_path,
        state_array_filter=[
            ArrayType.element_shell_stress,
            ArrayType.element_shell_history_vars,
            ArrayType.node_displacement,
            ArrayType.node_velocity,
        ],
    )
    n_states = d3.arrays[ArrayType.global_timesteps].shape[0]
    n_shells = d3.arrays[ArrayType.element_shell_node_indexes].shape[0]
    print(f"[extract]   states: {n_states}, shell elements: {n_shells}")
    return d3


def element_centroids_r(d3: D3plot, state: int) -> np.ndarray:
    """Radial coordinate (mm) of each shell element's centroid at ``state``."""
    X0 = d3.arrays[ArrayType.node_coordinates]            # (N_node, 3)
    disp = d3.arrays[ArrayType.node_displacement]         # (N_state, N_node, 3)
    X = X0 + disp[state]                                  # current nodal coords
    conn = d3.arrays[ArrayType.element_shell_node_indexes]  # (N_ele, 4)
    # Centroid x-coordinate (x = r in axisymmetric convention)
    xc = X[conn, 0].mean(axis=1)
    return xc


def element_centroid_velocity_r(d3: D3plot, state: int) -> np.ndarray:
    """Radial velocity (mm/ms) of each shell centroid at ``state``."""
    vel = d3.arrays[ArrayType.node_velocity][state]       # (N_node, 3)
    conn = d3.arrays[ArrayType.element_shell_node_indexes]
    vx = vel[conn, 0].mean(axis=1)
    return vx


def element_pressure(d3: D3plot, state: int) -> np.ndarray:
    """Scalar pressure (MPa) per shell element at ``state``.

    In LS-DYNA stress convention, pressure P = -(sxx + syy + szz)/3.
    For ALE shells the integration-point stress is stored in
    ``element_shell_stress`` with shape (N_state, N_ele, N_ip, 6).
    """
    stress = d3.arrays[ArrayType.element_shell_stress][state]  # (N_ele, N_ip, 6)
    # average over integration points
    sxx = stress[..., 0].mean(axis=-1)
    syy = stress[..., 1].mean(axis=-1)
    szz = stress[..., 2].mean(axis=-1)
    P = -(sxx + syy + szz) / 3.0
    return P


def element_density(d3: D3plot, state: int, n_ele: int) -> np.ndarray:
    """Density (g/mm^3) per shell element at ``state``.

    LS-DYNA stores density in the history variables (history var 1 for ALE
    with JWL/ideal-gas EOS).  If history vars are not available, fall back
    to initial density per part.
    """
    try:
        hv = d3.arrays[ArrayType.element_shell_history_vars][state]  # (N_ele, N_ip, N_hv)
        if hv.ndim == 3 and hv.shape[-1] >= 1:
            return hv[..., 0].mean(axis=-1)
        if hv.ndim == 2 and hv.shape[-1] >= 1:
            return hv[..., 0]
    except KeyError:
        pass

    # Fallback: use initial density per part (TNT=1.63e-3 g/mm^3, air=1.225e-6)
    part_ids = d3.arrays[ArrayType.element_shell_part_indexes]  # (N_ele,)
    dens = np.where(part_ids == 0, 1.63e-3, 1.225e-6)
    return dens


# ===================== Shock & contact detection =====================


def detect_shock_front(r: np.ndarray, P: np.ndarray, P_a: float = 0.101325) -> float:
    """Return R_s (mm): outermost element with P > threshold_factor * P_a.

    Sorted ascending in r.  Picks the last index where pressure is still
    elevated above ambient, then interpolates between that element and its
    outward neighbour for sub-element accuracy.
    """
    order = np.argsort(r)
    r_s = r[order]
    P_s = P[order]
    thr = 1.5 * P_a   # 1.5 * ambient = clearly shocked
    above = P_s > thr
    if not above.any():
        return float(r_s[0])
    # last index where pressure is above threshold
    idx = int(np.where(above)[0].max())
    if idx + 1 >= len(r_s):
        return float(r_s[idx])
    # Linear interp between r_s[idx] (above) and r_s[idx+1] (below)
    P0, P1 = P_s[idx], P_s[idx + 1]
    r0, r1 = r_s[idx], r_s[idx + 1]
    frac = (thr - P0) / (P1 - P0) if P1 != P0 else 0.0
    return float(r0 + frac * (r1 - r0))


def detect_contact_radius(
    r: np.ndarray,
    part_ids: np.ndarray,
    tnt_part_idx: int = 0,
) -> float:
    """Return R_c (mm): outermost radius of a TNT-products element.

    In the LS-DYNA mesh, elements with ``part_ids == tnt_part_idx`` are TNT,
    the rest are air.  The contact surface is the boundary between them.
    Because the mesh itself is Eulerian (fixed in space), we identify the
    contact via the history-variable material fraction would be exact — but
    for the simple pure-Eulerian layout here, the TNT elements' outer edge
    is a reasonable proxy that moves outward as products expand.
    """
    mask = part_ids == tnt_part_idx
    if not mask.any():
        return 0.0
    return float(r[mask].max())


def detect_separation_state(
    times: np.ndarray,
    P_c:   np.ndarray,
    R_c:   np.ndarray,
    P_a:   float = 0.101325,
    P_ratio: float = 20.0,
) -> int:
    """Return the state index where P_c drops below P_ratio * P_a."""
    thr = P_ratio * P_a
    below = P_c < thr
    if not below.any():
        # never separated: use last state
        return len(times) - 1
    return int(np.where(below)[0].min())


# ===================== Main extraction pipeline =====================


def extract(d3plot_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d3 = load_d3plot(d3plot_path)

    times_ms = d3.arrays[ArrayType.global_timesteps]                # ms
    part_ids = d3.arrays[ArrayType.element_shell_part_indexes]      # (N_ele,)
    n_ele = part_ids.shape[0]
    n_states = len(times_ms)

    print(f"[extract] scanning {n_states} states to build histories...")
    R_c_mm = np.zeros(n_states)
    R_s_mm = np.zeros(n_states)
    P_c_MPa = np.zeros(n_states)
    u_c_mmms = np.zeros(n_states)

    # Assume part index 0 = TNT, >0 = air; adjust if your layout differs
    tnt_part = 0

    for k in range(n_states):
        r_k = element_centroids_r(d3, k)                    # mm
        u_k = element_centroid_velocity_r(d3, k)            # mm/ms
        P_k = element_pressure(d3, k)                       # MPa

        # Shock radius: last element with elevated pressure
        R_s_mm[k] = detect_shock_front(r_k, P_k, P_a=0.101325)

        # Contact radius: outer edge of TNT-products
        R_c_mm[k] = detect_contact_radius(r_k, part_ids, tnt_part_idx=tnt_part)

        # Contact-face pressure & velocity: element whose r is closest to R_c
        # but on the TNT side (inner neighbour).
        tnt_mask = part_ids == tnt_part
        if tnt_mask.any():
            idx_c = np.argmin(np.abs(r_k[tnt_mask] - R_c_mm[k]))
            tnt_idx_global = np.where(tnt_mask)[0][idx_c]
            P_c_MPa[k] = P_k[tnt_idx_global]
            u_c_mmms[k] = u_k[tnt_idx_global]

    # Separation detection
    sep_idx = detect_separation_state(times_ms, P_c_MPa, R_c_mm)
    t_sep_ms = float(times_ms[sep_idx])
    print(f"[extract] separation at state {sep_idx}, t_sep = {t_sep_ms:.4f} ms")
    print(f"[extract]   R_c_sep = {R_c_mm[sep_idx]:.2f} mm, R_s_sep = {R_s_mm[sep_idx]:.2f} mm")

    # ---------------------- Snapshot at t_sep ----------------------
    r_snap_mm = element_centroids_r(d3, sep_idx)
    u_snap_mmms = element_centroid_velocity_r(d3, sep_idx)
    P_snap_MPa = element_pressure(d3, sep_idx)
    rho_snap_gmm3 = element_density(d3, sep_idx, n_ele)

    # Keep only air-side elements in the physical PINN domain [R_c, R_s]
    air_mask = part_ids != tnt_part
    in_domain = (
        air_mask
        & (r_snap_mm >= R_c_mm[sep_idx])
        & (r_snap_mm <= R_s_mm[sep_idx])
    )
    order = np.argsort(r_snap_mm[in_domain])
    r_mm_sorted   = r_snap_mm[in_domain][order]
    rho_sorted    = rho_snap_gmm3[in_domain][order]
    u_sorted      = u_snap_mmms[in_domain][order]
    P_sorted      = P_snap_MPa[in_domain][order]

    # Convert to SI
    r_SI   = r_mm_sorted   * MM_TO_M
    rho_SI = rho_sorted    * DENS_TO_SI
    u_SI   = u_sorted      * VEL_TO_SI
    P_SI   = P_sorted      * PRES_TO_SI

    np.savetxt(
        out_dir / "snapshot.csv",
        np.column_stack([r_SI, rho_SI, u_SI, P_SI]),
        delimiter=",",
        header="r,rho,u,P",
        comments="",
        fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir/'snapshot.csv'}  ({len(r_SI)} points)")

    # ---------------------- Contact history ----------------------
    # Only retain t >= t_sep for the PINN (pre-separation is not our regime)
    t_mask = times_ms >= t_sep_ms - 1e-12
    t_s    = times_ms[t_mask]    * MS_TO_S
    P_c_SI = P_c_MPa[t_mask]     * PRES_TO_SI
    u_c_SI = u_c_mmms[t_mask]    * VEL_TO_SI
    R_c_SI = R_c_mm[t_mask]      * MM_TO_M

    np.savetxt(
        out_dir / "contact.csv",
        np.column_stack([t_s, P_c_SI, u_c_SI, R_c_SI]),
        delimiter=",",
        header="t,P_c,u_c,R_c",
        comments="",
        fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir/'contact.csv'}   ({len(t_s)} points)")

    # ---------------------- Shock history ----------------------
    # D_s via central differences (2nd-order)
    R_s_SI_full = R_s_mm * MM_TO_M
    t_SI_full   = times_ms * MS_TO_S
    D_s_SI_full = np.gradient(R_s_SI_full, t_SI_full)

    np.savetxt(
        out_dir / "shock.csv",
        np.column_stack([t_SI_full, R_s_SI_full, D_s_SI_full]),
        delimiter=",",
        header="t,R_s,D_s",
        comments="",
        fmt="%.6e",
    )
    print(f"[extract] wrote {out_dir/'shock.csv'}     ({len(t_SI_full)} points)")

    # ---------------------- Metadata ----------------------
    meta = {
        "t_sep":    t_sep_ms        * MS_TO_S,
        "R_c_sep":  float(R_c_mm[sep_idx]) * MM_TO_M,
        "R_s_sep":  float(R_s_mm[sep_idx]) * MM_TO_M,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] wrote {out_dir/'metadata.json'}: {meta}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d3plot", required=True, help="path to d3plot file")
    p.add_argument("--out-dir", default="lsdyna_data", help="output directory")
    args = p.parse_args()
    extract(args.d3plot, Path(args.out_dir))


if __name__ == "__main__":
    main()
