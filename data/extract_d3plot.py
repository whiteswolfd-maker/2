"""Extract a 1D state along the +x axis from LS-DYNA d3plot binary output.

Reads every d3plot state with ``lasso-python``, filters solid elements whose
centroids lie within a (y, z) tolerance of the +x ray, sorts by x, and
linearly interpolates (rho, u_x, P) onto a uniform 1D grid x ∈ [0, x_end].

Detects R_s(t) (shock front, P > 1.05 P_atm) and R_c(t) (product–air contact
face, density jump above ``rho-contact-threshold``).  Computes
P_bar_p(t) = (1/R_c) ∫_0^{R_c} P(x, t) dx (trapezoidal).  Computes t_0 from
R_c=L_0 root, then writes:

    extracted/raw_line.npz          (t, x, rho, u, P)  shape (N_t, N_x)
    extracted/shock.csv             columns: t, R_s
    extracted/contact.csv           columns: t, R_c
    extracted/product_pressure.csv  columns: t, P_p_bar
    extracted/metadata.json         {L_0, t_0, t_end, x_end, rho_TNT, ...}

Background: LS-DYNA d3plot does NOT store per-cell density/pressure as named
arrays.  Pressure must be computed from the stress tensor:
    P = -(σ_xx + σ_yy + σ_zz) / 3
Density must be extracted from a history variable slot whose meaning depends
on the *DATABASE_EXTENT_BINARY card (NEIPH) and the EOS / multi-material
configuration.  Run with ``--debug`` first to inspect available arrays, then
re-run specifying ``--rho-hv-slot N`` for the right history-variable slot.

Usage
-----
    # Inspect d3plot structure first (no extraction):
    python -m data.extract_d3plot --d3plot sim_data/3dTNT1/ --debug

    # Full extraction (after identifying the density slot):
    python -m data.extract_d3plot --d3plot sim_data/3dTNT1/ \\
        --out extracted/ --rho-hv-slot 0 --L0 0.025 --x-end 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# np.trapezoid renamed from np.trapz in numpy 2.0; both names exist on 2.x but
# only the legacy spelling exists on 1.x.  Pick whichever the local install has.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ============================================================ helpers


def _count_states(d3plot_dir: Path) -> int:
    """True number of output states in the d3plot family.

    The old code counted ``d3plotNN`` files, which is only correct when each
    file holds exactly one state (true for huge models where each state fills
    a 150 MB file).  For small models LS-DYNA packs many states per file, so
    file-counting silently drops frames.  Reading just the global timesteps
    array is cheap and returns the exact state count.
    """
    from lasso.dyna import D3plot, ArrayType
    master = d3plot_dir / "d3plot"
    d3 = D3plot(str(master), buffered_reading=True,
                state_array_filter=[ArrayType.global_timesteps])
    return int(len(d3.arrays[ArrayType.global_timesteps]))


def _load_d3plot(d3plot_dir: Path, state_stride: int = 1,
                 state_subset: Optional[list[int]] = None):
    """Open the d3plot family, reading only the requested states.

    If ``state_subset`` (list of state indices) is given, load exactly those
    states — used for batched extraction so peak memory stays bounded no
    matter how many output frames the model wrote.  Otherwise subsample all
    states by ``state_stride`` (legacy path).

    Returns ``(d3, ArrayType)``; the loaded arrays are indexed 0..len(subset)-1
    in the order of ``state_subset``.
    """
    from lasso.dyna import D3plot, ArrayType
    master = d3plot_dir / "d3plot"
    if not master.exists():
        raise FileNotFoundError(
            f"Master d3plot file not found at {master}.\n"
            f"Expected directory layout: {d3plot_dir}/d3plot, d3plot01, d3plot02, ...\n"
            f"Listed: {sorted(p.name for p in d3plot_dir.iterdir())[:10]}"
        )
    if state_subset is not None:
        needed = set(int(s) for s in state_subset)
        n_total = len(needed)
    else:
        # True state count from the d3plot timesteps array, not file count.
        n_total = _count_states(d3plot_dir)
        n_total = max(n_total, 1)
        needed = set(range(0, n_total, state_stride))
    print(f"[load] {len(needed)} states "
          f"({'subset' if state_subset is not None else f'stride={state_stride}, ~{len(needed)*150/1024:.1f} GB estimated'})")
    d3 = D3plot(str(master), state_filter=needed, buffered_reading=True,
                state_array_filter=[
                    ArrayType.global_timesteps,
                    ArrayType.node_displacement, ArrayType.node_velocity,
                    ArrayType.element_solid_stress,
                    ArrayType.element_solid_history_variables,
                ])
    return d3, ArrayType


def _solid_pick_ipt(arr: np.ndarray, ipt: int = 0) -> np.ndarray:
    """LS-DYNA stores solid stress / strain / history-variables with shape
    (n_states, n_solids, n_intpts, n_components).  When NEIPS=0 / fully
    integrated 1pt, n_intpts == 1 and we want to drop that axis to recover
    the 3D (states, cells, comps) layout the rest of the pipeline expects.
    """
    if arr is None:
        return None
    if arr.ndim == 4:
        return arr[:, :, ipt, :]
    return arr


# Per-quantity scale factors converting common LS-DYNA unit systems to SI
# (m, kg, s, Pa, m/s).  Workbench Explicit Dynamics defaults to mm-Mg-s; the
# CGS unit set used by some explosive cards is g-cm-us.
_UNIT_SYSTEMS = {
    # Workbench Explicit Dynamics native (mm, Mg=tonne, s)
    "mm-Mg-s": dict(length=1e-3, vel=1e-3, rho=1e12, P=1e6, label="mm-Mg-s (Workbench Explicit)"),
    # SI -- already canonical
    "si":      dict(length=1.0,  vel=1.0,  rho=1.0,  P=1.0, label="SI (m-kg-s)"),
}


def _detect_unit_and_axis(d3, ArrayType, override_unit: Optional[str] = None,
                          override_axis: Optional[str] = None):
    """Heuristic auto-detection of the d3plot's unit system + radial axis sign.

    Returns ``(scales: dict, axis_sign: int, unit_key: str)``.  ``axis_sign``
    is +1 when the charge expands toward +x, -1 when toward -x.
    """
    nodes = d3.arrays[ArrayType.node_coordinates]
    span = float(max(
        nodes[:, 0].ptp(), nodes[:, 1].ptp(), nodes[:, 2].ptp(),
    ))

    if override_unit is not None:
        unit_key = override_unit
    else:
        # > 10 m of geometry strongly suggests mm-based units
        unit_key = "mm-Mg-s" if span > 10.0 else "si"
    scales = _UNIT_SYSTEMS[unit_key]

    if override_axis is not None:
        axis_sign = +1 if override_axis.lstrip("+") == "x" else -1
    else:
        x_min = float(nodes[:, 0].min())
        x_max = float(nodes[:, 0].max())
        if x_max <= 1e-9 and x_min < 0:
            axis_sign = -1
        elif x_min >= -1e-9 and x_max > 0:
            axis_sign = +1
        else:
            axis_sign = +1   # bbox crosses zero - default to +x

    return scales, axis_sign, unit_key


def _print_debug(d3, ArrayType, scales, axis_sign, unit_key,
                 R_0_for_vf_hint: float = 0.05) -> None:
    """Print array inventory + history-variables sample so user can identify
    which slot corresponds to density and which to TNT volume fraction.
    """
    print("\n=== ArrayType inventory ===")
    for k, v in sorted(d3.arrays.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k:40s}  shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:40s}  type={type(v).__name__}")
    print("===========================\n")

    print(f"[detect] unit system  : {scales['label']}  "
          f"(length x{scales['length']:g}, vel x{scales['vel']:g}, "
          f"rho x{scales['rho']:g}, P x{scales['P']:g})")
    print(f"[detect] radial axis  : {'+x' if axis_sign > 0 else '-x'}  "
          f"(axis_sign={axis_sign:+d})")

    ale_ids = d3.arrays.get(ArrayType.ale_material_ids)
    if ale_ids is not None:
        print(f"[detect] ale_material_ids: {list(map(int, ale_ids))}")
    part_titles = d3.arrays.get(ArrayType.part_titles)
    if part_titles is not None:
        names = [t.decode("ascii", errors="replace").strip() for t in part_titles]
        print(f"[detect] part_titles    : {names}")

    times = d3.arrays.get(ArrayType.global_timesteps)
    if times is not None:
        print(f"\ntimesteps: n={len(times)}, t_min={times.min():.3e} s, "
              f"t_max={times.max():.3e} s")

    nodes = d3.arrays.get(ArrayType.node_coordinates)
    if nodes is not None:
        L = scales["length"]
        print("node_coordinates bbox (native -> SI metres):")
        for axis_label, col in zip("xyz", range(3)):
            lo, hi = nodes[:, col].min(), nodes[:, col].max()
            print(f"  {axis_label}: [{lo:+.3e}, {hi:+.3e}]   "
                  f"->  [{lo*L:+.3e}, {hi*L:+.3e}] m")

    hv = _solid_pick_ipt(d3.arrays.get(ArrayType.element_solid_history_variables))
    if hv is not None:
        n_states, _, n_hv = hv.shape
        last_state = n_states - 1
        print(f"\nelement_solid_history_variables: {n_hv} slots available "
              f"(last state index = {last_state})")
        last = hv[last_state]
        rho_scale = scales["rho"]
        print(f"\nPer-slot stats at last state (raw value, then * rho_scale "
              f"= {rho_scale:g} -> kg/m^3):")
        print(f"  {'slot':>4} {'min':>13} {'med':>13} {'max':>13} | "
              f"{'med kg/m^3':>14}  hint")
        for s in range(n_hv):
            col = last[:, s]
            finite = col[np.isfinite(col)]
            if not finite.size:
                continue
            md = float(np.median(finite))
            md_kg = md * rho_scale
            hint = "<-- density candidate" if 0.1 <= md_kg <= 3000.0 else ""
            print(f"  {s:>4d} {finite.min():>+13.3e} {md:>+13.3e} "
                  f"{finite.max():>+13.3e} | {md_kg:>+14.3e}  {hint}")

        # --- TNT volume-fraction slot detection ---
        # Build cell radii at t=0 (undeformed mesh) and look at slots whose
        # full range is in [0, 1]: those are ALE volume fractions.  Map each
        # such slot onto its part_title and tag any title NOT containing
        # "air" as a TNT-like component.  Recommend the union (the contact
        # face is then "where the sum of TNT-side fractions exceeds the
        # threshold"), which is robust to per-block split materials.
        nodes0 = d3.arrays.get(ArrayType.node_coordinates)
        elem_idx = d3.arrays.get(ArrayType.element_solid_node_indexes)
        title_arr = d3.arrays.get(ArrayType.part_titles)
        part_titles_str = (
            [t.decode("ascii", errors="replace").strip() for t in title_arr]
            if title_arr is not None else []
        )
        if nodes0 is not None and elem_idx is not None:
            n_per_elem = min(8, elem_idx.shape[1])
            cell_x = nodes0[elem_idx[:, :n_per_elem]].mean(axis=1)[:, 0]
            r_native = np.abs(cell_x * scales["length"])    # metres, axis-flipped
            inside = r_native < R_0_for_vf_hint              # in TNT region at t=0
            n_inside = int(inside.sum())
            print(f"\nVolume-fraction slot scan (cells with r < {R_0_for_vf_hint*1e3:.0f} "
                  f"mm at t=0; {n_inside} cells selected):")
            print(f"  {'slot':>4} {'min':>9} {'max':>9} {'mean_inside':>14}  "
                  f"{'part_title':<14}  hint")
            hv_t0 = hv[0]
            vf_slots: list[int] = []   # any slot whose values stay in [0, 1]
            for s in range(n_hv):
                col_all = hv_t0[:, s]
                fin_all = col_all[np.isfinite(col_all)]
                if fin_all.size == 0:
                    continue
                if fin_all.min() < -1e-6 or fin_all.max() > 1.0 + 1e-6:
                    continue
                vf_slots.append(s)

            # Heuristic: assume vf slots map 1-to-1 onto parts in order.
            # First vf slot -> part 0; second -> part 1; ... .
            tnt_slots: list[int] = []
            for slot_idx, s in enumerate(vf_slots):
                col_all = hv_t0[:, s]
                mean_inside = float(col_all[inside].mean()) if n_inside > 0 else 0.0
                title = (part_titles_str[slot_idx] if slot_idx < len(part_titles_str)
                         else "?")
                is_air = "air" in title.lower()
                hint = ""
                if not is_air and mean_inside > 0.01:
                    hint = "<-- TNT-side"
                    tnt_slots.append(s)
                elif is_air:
                    hint = "(air)"
                print(f"  {s:>4d} {col_all.min():>+9.3e} {col_all.max():>+9.3e} "
                      f"{mean_inside:>+14.3e}  {title:<14}  {hint}")

            if tnt_slots:
                if len(tnt_slots) == 1:
                    rec = str(tnt_slots[0])
                else:
                    rec = "[" + ", ".join(str(s) for s in tnt_slots) + "]"
                total = float(sum(
                    hv_t0[:, s][inside].mean() if n_inside > 0 else 0.0
                    for s in tnt_slots
                ))
                print(f"\n[detect] TNT-side volume-fraction slots = {rec}")
                print(f"         sum mean vf inside r<R_0 = {total:.3f} "
                      f"(should be close to 1.0 if R_0 covers the charge cleanly)")
                print(f"         set in yaml ->  data.vf_tnt_slot: {rec}")
            else:
                print("\n[detect] no non-air volume-fraction slot found.  "
                      "Stick with density-based R_c detection.")
    else:
        print("\nNo element_solid_history_variables -- need NEIPH > 0 in "
              "*DATABASE_EXTENT_BINARY for MMALE density extraction.")

    stress = _solid_pick_ipt(d3.arrays.get(ArrayType.element_solid_stress))
    if stress is not None:
        last_state = stress.shape[0] - 1
        sigma = stress[last_state]
        P_native = -(sigma[:, 0] + sigma[:, 1] + sigma[:, 2]) / 3.0
        P_si = P_native * scales["P"]
        print(f"\nelement_solid_stress: pressure P=-tr(sigma)/3 at last state")
        print(f"  native: min={P_native.min():+.3e}  med={np.median(P_native):+.3e}  "
              f"max={P_native.max():+.3e}")
        print(f"  Pa    : min={P_si.min():+.3e}  med={np.median(P_si):+.3e}  "
              f"max={P_si.max():+.3e}")
    else:
        print("\nWarning: no element_solid_stress -- pressure extraction not possible.")


# ============================================================ extraction core


def _compute_centroids_and_velocities(d3, ArrayType, state_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids (n_solids, 3), vel_cell (n_solids, 3)) at one state.

    Centroid = mean of 8 hex node positions (uses node_coordinates +
    node_displacement at this state).  Velocity = mean of the FULL node
    velocity vectors (all 3 components), per cell — needed to project onto the
    radial direction for the spherical-shell average.
    """
    nodes0 = d3.arrays[ArrayType.node_coordinates]
    elem_idx = d3.arrays[ArrayType.element_solid_node_indexes]  # (n_solids, n_per_elem)

    # Use displaced coordinates if available (otherwise undeformed mesh).
    # CAUTION: some d3plot writers (ANSYS Workbench Explicit Dynamics) store
    # ABSOLUTE node positions in `node_displacement` instead of a true
    # displacement.  For a true displacement, state 0 is ~0; for positions it
    # equals node_coordinates (span-scale magnitude).  Adding the latter to
    # node_coordinates DOUBLES every radius (50 mm charge -> reported ~99 mm,
    # mass check 8.6 kg).  Detect and use the array as positions when needed.
    disp = d3.arrays.get(ArrayType.node_displacement)
    if disp is not None and disp.shape[0] > state_idx:
        span = float(max(nodes0[:, 0].ptp(), nodes0[:, 1].ptp(), nodes0[:, 2].ptp()))
        d0_max = float(np.abs(disp[0]).max()) if disp[0].size else 0.0
        if d0_max > 0.01 * span:
            # node_displacement holds positions (not deltas) -> use directly
            nodes_t = disp[state_idx]
        else:
            nodes_t = nodes0 + disp[state_idx]
    else:
        nodes_t = nodes0

    # Select first 8 columns (hex8); for tet/wedge/shell take available
    n_per_elem = min(8, elem_idx.shape[1])
    elem_nodes = elem_idx[:, :n_per_elem]  # (n_solids, ≤8)
    centroids = nodes_t[elem_nodes].mean(axis=1)  # (n_solids, 3)

    vel = d3.arrays.get(ArrayType.node_velocity)
    if vel is not None and vel.shape[0] > state_idx:
        vel_cell = vel[state_idx][elem_nodes].mean(axis=1)  # (n_solids, 3)
    else:
        vel_cell = np.zeros_like(centroids)

    return centroids, vel_cell


def _extract_state(
    d3, ArrayType, state_idx: int,
    rho_hv_slot: int,
    rho_sign: float,
    scales: dict,
    vf_tnt_slot: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (centroids, rho, u_r, P, vf_tnt) for one state, all in SI units.

    ``centroids`` is the full 3D cell-centroid array (scaled to metres);
    ``u_r`` is the outward-positive RADIAL velocity (projection of the cell
    velocity onto the centroid unit vector — orientation independent).
    ``vf_tnt`` is None when no volume-fraction slot is configured; otherwise
    it is the sum of volume fractions over every listed slot (so a TNT charge
    split across N parts can be reduced to a single product-side indicator).
    """
    centroids, vel_cell = _compute_centroids_and_velocities(d3, ArrayType, state_idx)
    centroids = centroids.copy()
    centroids *= scales["length"]
    # Radial velocity: project the cell velocity onto the outward unit vector.
    # This is the correct spherical reduction (the old code used the x-velocity
    # alone, which is only valid on the +x axis).
    r = np.linalg.norm(centroids, axis=1)
    unit = np.zeros_like(centroids)
    nz = r > 1e-12
    unit[nz] = centroids[nz] / r[nz, None]
    u_r = (vel_cell * scales["vel"] * unit).sum(axis=1)

    stress = _solid_pick_ipt(d3.arrays[ArrayType.element_solid_stress])
    sigma = stress[state_idx]  # (n_solids, 6) -- Voigt xx, yy, zz, xy, yz, zx
    P = -(sigma[:, 0] + sigma[:, 1] + sigma[:, 2]) / 3.0 * scales["P"]

    hv = _solid_pick_ipt(d3.arrays.get(ArrayType.element_solid_history_variables))
    if hv is None:
        raise RuntimeError(
            "element_solid_history_variables not in d3plot -- need NEIPH > 0 "
            "in *DATABASE_EXTENT_BINARY card to write history variables."
        )
    if rho_hv_slot >= hv.shape[-1]:
        raise IndexError(
            f"--rho-hv-slot {rho_hv_slot} out of range; "
            f"history_variables has {hv.shape[-1]} slots. Run --debug first."
        )
    rho = rho_sign * hv[state_idx, :, rho_hv_slot] * scales["rho"]

    vf_tnt = None
    if vf_tnt_slot:
        vf_tnt = np.zeros(hv.shape[1], dtype=hv.dtype)
        for slot in vf_tnt_slot:
            if slot >= hv.shape[-1] or slot < 0:
                raise IndexError(
                    f"vf_tnt_slot {slot} out of range "
                    f"({hv.shape[-1]} slots available)."
                )
            vf_tnt = vf_tnt + hv[state_idx, :, slot]

    return centroids, rho, u_r, P, vf_tnt


def _interpolate_to_grid(
    centroids: np.ndarray,
    rho: np.ndarray, u: np.ndarray, P: np.ndarray,
    r_grid: np.ndarray,
    vf_tnt: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Spherical-shell average of the 3D field onto the radial grid ``r_grid``.

    Each cell is assigned to the nearest radial grid node by its TRUE centroid
    radius ``r = sqrt(x^2 + y^2 + z^2)``; (rho, u_r, P, vf) are averaged per
    shell.  Empty far shells are filled from the last populated shell (so the
    ambient tail is preserved when the mesh does not reach ``r_grid[-1]``).

    This is the correct 1D projection of a spherically-symmetric 3D field.  It
    replaces the old |y|<y_tol ∧ |z|<z_tol slab binned by x, which mixed cells
    of different true radii (a cell at (x=30,y=40) has r=50) and smeared the
    near field.

    Returns ``(rho_grid, u_grid, P_grid, vf_tnt_grid)``; the last entry is
    ``None`` when ``vf_tnt`` was not supplied.
    """
    r = np.sqrt((centroids ** 2).sum(axis=1))  # (n_solids,) true radius
    n_shells = len(r_grid)
    if n_shells == 0:
        raise ValueError("r_grid must be non-empty")
    dr = float(r_grid[1] - r_grid[0]) if n_shells > 1 else 1.0
    idx = np.clip(np.round(r / dr).astype(int), 0, n_shells - 1)

    def _shell_avg(values: np.ndarray) -> np.ndarray:
        s = np.bincount(idx, weights=values, minlength=n_shells)
        c = np.bincount(idx, minlength=n_shells)
        avg = s / np.maximum(c, 1)
        valid = c > 0
        if not valid.any():
            return avg
        # Fill empty shells from the NEAREST populated shell in both
        # directions: forward-fill interior/tail from the last populated shell
        # (so an ambient tail is preserved when the mesh ends before r_grid[-1]),
        # and map leading shells before the first populated one onto that first
        # shell's value (uniform-core assumption near the origin) instead of a
        # spurious 0.
        n = n_shells
        first = int(np.argmax(valid))
        last_idx = np.maximum.accumulate(np.where(valid, np.arange(n), -1))
        filled = np.where(last_idx >= 0, avg[np.clip(last_idx, first, None)], avg[first])
        return filled

    rho_g = _shell_avg(rho)
    u_g   = _shell_avg(u)
    P_g   = _shell_avg(P)
    vf_g  = _shell_avg(vf_tnt) if vf_tnt is not None else None
    return rho_g, u_g, P_g, vf_g


# ============================================================ trajectory detection


def _detect_R_s(x: np.ndarray, P: np.ndarray, P_atm: float, P_threshold_factor: float) -> float:
    """R_s = location of steepest pressure drop (max |dP/dr|), i.e. the true
    shock front.  Falls back to the outermost ``P > threshold`` cell when the
    gradient is too flat to identify a shock unambiguously.
    """
    above = P > P_threshold_factor * P_atm
    if not above.any():
        return 0.0
    above_idx = np.where(above)[0]
    r_outer = float(x[above_idx[-1]])

    # dP/dr on the uniform extraction grid via 2nd-order central difference
    dP_dr = np.gradient(P, x, edge_order=2)
    # Search in the outer 40 % of the above-threshold region so the rarefaction
    # tail behind the contact surface is never mistaken for the shock.
    n_above = len(above_idx)
    if n_above < 5:
        return r_outer
    search_start = above_idx[int(n_above * 0.6)]
    search_slice = slice(search_start, above_idx[-1] + 1)
    dP_sr = dP_dr[search_slice]
    x_sr  = x[search_slice]

    # Steepest NEGATIVE gradient = shock.  Require |dP/dr| > 1 % of max |dP/dr|
    # in the search window to reject near-flat noise.
    if np.abs(dP_sr).max() < 1e-9:
        return r_outer
    threshold = 0.01 * np.abs(dP_sr).max()
    steep = dP_sr < -threshold
    if not steep.any():
        return r_outer
    return float(x_sr[np.where(steep)[0][-1]])

def _sg_smooth(y: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    """Savitzky-Golay filter (preserves peak sharpness better than Gaussian)."""
    from scipy.signal import savgol_filter
    w = min(window, len(y) - (len(y) + 1) % 2 - 1)  # must be odd, < len
    if w < order + 2:
        return y
    return savgol_filter(y, w, order)


def _detect_R_c(x: np.ndarray, rho: np.ndarray, rho_threshold: float) -> float:
    """R_c via density threshold (legacy fallback).  Returns the outermost
    radius where rho > rho_threshold.  Use ``_detect_R_c_vf`` whenever an
    ALE TNT-product volume-fraction array is available -- it is far more
    robust because shocked air can reach densities (~7 kg/m^3) that fall
    in the same band as expanded TNT products.
    """
    above = rho > rho_threshold
    if not above.any():
        return 0.0
    return float(x[np.where(above)[0][-1]])


def _detect_R_c_vf(x: np.ndarray, vf_tnt: np.ndarray, vf_threshold: float) -> float:
    """R_c = outermost x where the TNT-product volume fraction exceeds
    ``vf_threshold``.  ALE multi-material always preserves a clean
    [0, 1] indicator field for each material, so this is immune to the
    Hugoniot-air ambiguity that contaminates rho-based detection.
    """
    above = vf_tnt > vf_threshold
    if not above.any():
        return 0.0
    return float(x[np.where(above)[0][-1]])


def _avg_product_pressure(x: np.ndarray, P: np.ndarray, R_c: float) -> float:
    """P_bar_p = (1/R_c) ∫_0^{R_c} P dx via trapezoid.  Returns 0 if R_c<=0."""
    if R_c <= x[0] or R_c <= 0:
        return 0.0
    mask = x <= R_c
    if mask.sum() < 2:
        return 0.0
    x_sub = x[mask]
    P_sub = P[mask]
    # Append exact P(R_c) by interpolation for a clean upper bound
    P_at_Rc = float(np.interp(R_c, x, P))
    x_sub = np.concatenate([x_sub, [R_c]])
    P_sub = np.concatenate([P_sub, [P_at_Rc]])
    return float(_trapezoid(P_sub, x_sub) / R_c)


def _find_t0(times: np.ndarray, R_c: np.ndarray, L_0: float) -> float:
    """t_0 = first t at which R_c crosses L_0 from below.  Linear interp."""
    if (R_c >= L_0).all():
        return float(times[0])
    if (R_c <= L_0).all():
        return float(times[-1])
    # First crossing
    idx = np.where(R_c >= L_0)[0]
    if idx.size == 0:
        return float(times[-1])
    i = idx[0]
    if i == 0:
        return float(times[0])
    t1, t2 = times[i - 1], times[i]
    R1, R2 = R_c[i - 1], R_c[i]
    if R2 == R1:
        return float(t1)
    frac = (L_0 - R1) / (R2 - R1)
    return float(t1 + frac * (t2 - t1))


# ============================================================ main


def main():
    p = argparse.ArgumentParser(
        description="Extract 1D +x line from LS-DYNA d3plot for PINN training."
    )
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (e.g. configs/tnt_spherical_50mm.yaml). "
                        "If given, --d3plot / --R-0 / --x-end / --n-x / --rho-hv-slot / "
                        "--y-tol / --z-tol fall back to data.* keys in the file.")
    p.add_argument("--d3plot", type=Path, default=None,
                   help="Directory containing d3plot, d3plot01, ... "
                        "(overrides data.d3plot_dir from --config).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory for npz/csv/json (default: extracted/ "
                        "or data.extracted_dir from --config).")
    p.add_argument("--debug", action="store_true",
                   help="Print array inventory + history-variable slots, then exit.")
    p.add_argument("--rho-hv-slot", type=int, default=None,
                   help="History-variable slot index for mixture density "
                        "(use --debug to find it).")
    p.add_argument("--rho-sign", type=float, default=1.0,
                   help="Multiplier for density values from HV (default 1.0).")
    p.add_argument("--R-0", "--R0", dest="R_0", type=float, default=None,
                   help="Spherical TNT charge radius (m).  For this project "
                        "R_0 = 0.05 m (diameter 100 mm, mass ~= 0.853 kg).")
    p.add_argument("--x-end", type=float, default=None,
                   help="Outer +x extent of grid (m).")
    p.add_argument("--n-x", type=int, default=None,
                   help="Number of grid points in x.")
    p.add_argument("--P-atm", type=float, default=101325.0,
                   help="Ambient pressure (Pa).")
    p.add_argument("--P-threshold-factor", type=float, default=None,
                   help="R_s detection: P > factor x P_atm "
                        "(default 1.05; or set data.P_threshold_factor in yaml).")
    p.add_argument("--rho-contact-threshold", type=float, default=None,
                   help="R_c detection: rho > threshold kg/m^3 (default 100; "
                        "must exceed Hugoniot-compressed air ~7.4 kg/m^3 to "
                        "avoid mistaking shocked air for product.  Or set "
                        "data.rho_contact_threshold in yaml).")
    p.add_argument("--rho-tnt", type=float, default=1630.0,
                   help="Initial TNT density (kg/m^3, default 1630).")
    p.add_argument("--state-stride", type=int, default=1,
                   help="Subsample every Nth state to reduce memory (default 1). "
                        "WARNING: the old default was 4, silently dropping 3/4 of "
                        "the output frames (this is why extracted_100mm had only 30).")
    p.add_argument("--unit-system", choices=("auto", "si", "mm-Mg-s"), default="auto",
                   help="d3plot unit system.  'auto' -> infer from bbox span "
                        "(>10 m -> mm-Mg-s, else SI).")
    p.add_argument("--axis-sign", choices=("auto", "+x", "-x"), default="auto",
                   help="Direction the charge expands along.  'auto' -> infer "
                        "from node bbox sign.")
    p.add_argument("--vf-tnt-slot", type=str, default=None,
                   help="ALE TNT-product volume-fraction history slot, or a "
                        "comma-separated list of slots that get summed (e.g. "
                        "'31' or '32,33,34,37').  When set, R_c is detected "
                        "as the outermost radius where vf_tnt > vf_threshold "
                        "(immune to shocked-air contamination of rho-based "
                        "detection).  Or set data.vf_tnt_slot in yaml -- yaml "
                        "accepts a list directly: ``vf_tnt_slot: [32, 33]``.")
    p.add_argument("--vf-threshold", type=float, default=None,
                   help="Volume-fraction threshold for R_c detection "
                        "(default 0.01; lower -> picks up the diffuse outer "
                        "edge of the ALE interface).")
    args = p.parse_args()

    # Merge config defaults (CLI flags win when explicitly given).
    cfg_data: dict = {}
    if args.config is not None:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg_data = (yaml.safe_load(f) or {}).get("data", {}) or {}

    def _pick(name: str, cli_value, default):
        if cli_value is not None:
            return cli_value
        if name in cfg_data and cfg_data[name] is not None:
            return cfg_data[name]
        return default

    d3plot_dir = _pick("d3plot_dir", args.d3plot, None)
    if d3plot_dir is None:
        print("ERROR: provide --d3plot DIR or set data.d3plot_dir in --config",
              file=sys.stderr)
        return 2
    d3plot_dir = Path(d3plot_dir).resolve()

    out_dir = _pick("extracted_dir", args.out, "extracted/")
    out_dir = Path(out_dir).resolve()

    # Wrap the resolved values back onto args so the rest of the function can
    # keep using its existing namespace.
    args.d3plot       = d3plot_dir
    args.out          = out_dir
    args.rho_hv_slot  = _pick("rho_hv_slot", args.rho_hv_slot, 0)
    args.R_0          = _pick("R_0",         args.R_0,         0.05)
    args.x_end        = _pick("x_end",       args.x_end,       0.5)
    args.n_x          = _pick("n_x",         args.n_x,         500)
    args.rho_contact_threshold = _pick("rho_contact_threshold",
                                       args.rho_contact_threshold, 100.0)
    args.P_threshold_factor    = _pick("P_threshold_factor",
                                       args.P_threshold_factor, 1.05)
    raw_vf = _pick("vf_tnt_slot", args.vf_tnt_slot, None)
    if raw_vf is None:
        args.vf_tnt_slot = None
    elif isinstance(raw_vf, (list, tuple)):
        args.vf_tnt_slot = [int(s) for s in raw_vf]
    elif isinstance(raw_vf, str):
        args.vf_tnt_slot = [int(s.strip()) for s in raw_vf.split(",") if s.strip()]
    else:
        args.vf_tnt_slot = [int(raw_vf)]
    args.vf_threshold          = _pick("vf_threshold", args.vf_threshold, 0.01)

    if not d3plot_dir.is_dir():
        print(f"ERROR: {d3plot_dir} is not a directory", file=sys.stderr)
        return 2

    # Probe load: just state 0 (static arrays + one state) for unit/axis
    # detection and --debug slot identification.  The full extraction below
    # re-loads states in bounded chunks.
    d3, ArrayType = _load_d3plot(d3plot_dir, state_subset=[0])

    override_unit = None if args.unit_system == "auto" else args.unit_system
    override_axis = None if args.axis_sign == "auto" else args.axis_sign
    scales, axis_sign, unit_key = _detect_unit_and_axis(
        d3, ArrayType,
        override_unit=override_unit,
        override_axis=override_axis,
    )

    if args.debug:
        _print_debug(d3, ArrayType, scales, axis_sign, unit_key,
                     R_0_for_vf_hint=float(args.R_0))
        return 0

    print(f"[extract] unit system : {scales['label']}")
    print(f"[extract] radial axis : {'+x' if axis_sign > 0 else '-x'}")
    if args.vf_tnt_slot:
        slots_str = (str(args.vf_tnt_slot[0]) if len(args.vf_tnt_slot) == 1
                     else "+".join(str(s) for s in args.vf_tnt_slot))
        print(f"[extract] R_c source  : ALE volume fraction slot(s) "
              f"{slots_str}, sum > {args.vf_threshold}")
    else:
        print(f"[extract] R_c source  : density threshold "
              f"rho > {args.rho_contact_threshold:.1f} kg/m^3")

    # ----------- core extraction (batched to bound peak memory) -----------
    n_total = _count_states(d3plot_dir)
    n_total = max(n_total, 1)
    state_indices = list(range(0, n_total, args.state_stride))
    n_states = len(state_indices)
    print(f"[extract] {n_states}/{n_total} states (stride={args.state_stride})")

    x_grid = np.linspace(0.0, args.x_end, args.n_x)
    rho_grid = np.zeros((n_states, args.n_x))
    u_grid   = np.zeros((n_states, args.n_x))
    P_grid   = np.zeros((n_states, args.n_x))
    R_s_arr  = np.zeros(n_states)
    R_c_arr  = np.zeros(n_states)
    Pbar_p_arr = np.zeros(n_states)
    times_out = np.zeros(n_states)

    CHUNK = 25          # states loaded at once (~3.4 GB for 1.2M-element models)
    log_every = max(1, n_states // 20)
    k = 0
    for c0 in range(0, n_states, CHUNK):
        chunk = state_indices[c0:c0 + CHUNK]
        d3c, _ = _load_d3plot(d3plot_dir, state_subset=chunk)
        times_c = d3c.arrays[ArrayType.global_timesteps]
        for rel, s in enumerate(chunk):
            centroids, rho, u, P, vf = _extract_state(
                d3c, ArrayType, rel, args.rho_hv_slot, args.rho_sign,
                scales,
                vf_tnt_slot=args.vf_tnt_slot,
            )
            rho_g, u_g, P_g, vf_g = _interpolate_to_grid(
                centroids, rho, u, P,
                x_grid,          # radial grid (spherical-shell average)
                vf_tnt=vf,
            )
            rho_grid[k] = rho_g
            u_grid[k]   = u_g
            P_grid[k]   = P_g
            R_s_arr[k] = _detect_R_s(x_grid, P_g, args.P_atm, args.P_threshold_factor)
            if vf_g is not None:
                R_c_arr[k] = _detect_R_c_vf(x_grid, vf_g, args.vf_threshold)
            else:
                R_c_arr[k] = _detect_R_c(x_grid, rho_g, args.rho_contact_threshold)
            Pbar_p_arr[k] = _avg_product_pressure(x_grid, P_g, R_c_arr[k])
            times_out[k] = float(times_c[rel])
            if k % log_every == 0:
                print(f"  state {s:5d}  t={times_out[k]:.3e}  "
                      f"R_s={R_s_arr[k]*1e3:6.2f} mm  R_c={R_c_arr[k]*1e3:6.2f} mm  "
                      f"Pbar_p={Pbar_p_arr[k]:.3e} Pa")
            k += 1
        del d3c   # free the chunk's arrays before loading the next

    # ----------- t_0 -----------
    # The contact face starts at the spherical charge surface r = R_0.
    t_0 = _find_t0(times_out, R_c_arr, args.R_0)
    print(f"[extract] t_0 (R_c={args.R_0*1e3:.1f} mm) = {t_0:.3e} s")

    # ----------- CJ + Sec.4.2.1 separation + Taylor-Sadovsky ODE -----------
    print("[extract] computing CJ + separation state via physics/cj_state.py...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from physics.cj_state import (
        TNTParams,
        compute_cj_state,
        compute_separation_state,
    )
    from physics.uniform_expansion import write_csv as write_uniform_csv

    tnt = TNTParams(rho_TNT=args.rho_tnt)
    cj_b = compute_cj_state(tnt)
    P_CJ = cj_b.P_CJ; rho_CJ = cj_b.rho_CJ; u_CJ = cj_b.u_CJ
    print(f"[extract] CJ: P_CJ={P_CJ/1e9:.3f} GPa, rho_CJ={rho_CJ:.1f} kg/m^3, "
          f"u_CJ={u_CJ:.1f} m/s")

    # The LS-DYNA model is a SPHERICAL TNT charge of radius R_0 with mass
    # M = (4/3) pi R_0^3 rho_TNT.  No cube-to-sphere conversion is needed.
    R_0 = float(args.R_0)
    M_TNT = (4.0 / 3.0) * np.pi * args.rho_tnt * R_0 ** 3
    sep = compute_separation_state(tnt, R_0=R_0, cj_bundle=cj_b)
    print(f"[extract] Spherical charge R_0 = {R_0*1e3:.2f} mm  "
          f"(diameter {R_0*2e3:.1f} mm, mass {M_TNT*1e3:.1f} g)")
    print(f"[extract] Sec.4.2.1: P_x={sep.P_x/1e9:.3f} GPa  u_x={sep.u_x:.1f} m/s  "
          f"rho_x={sep.rho_x:.1f} kg/m^3  R_c={sep.R_c*1e3:.2f} mm  t_sep={sep.t_sep*1e6:.2f} us")

    # ----------- density-slot mass-conservation self-check -----------
    # A correct density history-variable slot must satisfy
    #     M_prod = ∫ 4π r^2 ρ dr  ≈  M_TNT
    # over the product region at a time near t_sep (products still compact,
    # not yet diluted into the air).  An inflated slot (wrong history
    # variable) gives 5-7x the charge mass — this is the fastest way to
    # catch a bad rho_hv_slot before the PINN trains on garbage density.
    i_check = int(np.argmin(np.abs(times_out - sep.t_sep)))
    rho_row = rho_grid[i_check]
    mask_p = rho_row > args.rho_contact_threshold
    if mask_p.sum() >= 2:
        M_prod = 4.0 * np.pi * _trapezoid(
            (x_grid[mask_p] ** 2) * rho_row[mask_p], x_grid[mask_p])
        ratio = M_prod / M_TNT
        if 0.5 <= ratio <= 1.5:
            print(f"[mass ] OK: 4*pi*r^2*rho dr = {M_prod:.3f} kg ~ M_TNT "
                  f"({ratio:.2f}x) - density slot looks right.")
        else:
            print(f"[mass ] WARN: 4*pi*r^2*rho dr = {M_prod:.2f} kg vs M_TNT = "
                  f"{M_TNT:.3f} kg  ({ratio:.1f}x).  rho_hv_slot="
                  f"{args.rho_hv_slot} is probably NOT the true density - "
                  f"re-run with --debug and pick the slot passing this check.")
    else:
        print("[mass ] (skipped: product region too thin at t_sep)")

    # ----------- output -----------
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "raw_line.npz",
        t=times_out, x=x_grid, rho=rho_grid, u=u_grid, P=P_grid,
    )
    # SG-smooth R_s to suppress 1mm-grid quantization noise before computing D_s
    R_s_smooth = _sg_smooth(R_s_arr, window=31, order=3)
    np.savetxt(args.out / "shock.csv",
               np.column_stack([times_out, R_s_smooth]),
               delimiter=",", header="t,R_s", comments="")
    np.savetxt(args.out / "contact.csv",
               np.column_stack([times_out, R_c_arr]),
               delimiter=",", header="t,R_c", comments="")
    np.savetxt(args.out / "product_pressure.csv",
               np.column_stack([times_out, Pbar_p_arr]),
               delimiter=",", header="t,P_p_bar", comments="")
    write_uniform_csv(sep.ode, args.out / "r_c_taylor_sadovsky.csv")

    metadata = {
        # Spherical TNT charge geometry
        "R_0":     R_0,
        "M_TNT":   float(M_TNT),
        "x_end":   args.x_end,
        "t_0":     t_0,
        "t_end":   float(times_out[-1]),
        "rho_TNT": args.rho_tnt,
        "rho_CJ":  rho_CJ,
        "P_CJ":    P_CJ,
        "u_CJ":    u_CJ,
        "n_x":     args.n_x,
        "n_t":     len(times_out),
        "rho_hv_slot": args.rho_hv_slot,
        # Sec.4.2.1 separation state at the connection point (t_sep, R_c)
        "R_c":     sep.R_c,
        "t_sep":   sep.t_sep,
        "P_x":     sep.P_x,
        "u_x":     sep.u_x,
        "rho_x":   sep.rho_x,
        "V_x":     sep.V_x,
    }
    with open(args.out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[done] wrote {args.out}/raw_line.npz "
          f"({rho_grid.shape[0]} x {rho_grid.shape[1]}), shock.csv, contact.csv, "
          f"product_pressure.csv, r_c_taylor_sadovsky.csv, metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
