#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a complete LS-DYNA keyword deck for a spherical TNT air-blast.

This replaces the Workbench-GUI-generated decks that produced unusable data
(the 100 mm run had coarse output + no detonation; see data_quality_audit.py).

Design
------
* 1D spherical problem resolved with a thin spherical wedge mesh: a 2x2 hex
  cross-section, ~500 graded radial layers whose cell size is PROPORTIONAL to
  R_0 (1 mm -> 8 mm at R_0 = 50 mm; every length/time scales with R_0 / R_REF).
  This makes each deck a Hopkinson-Cranz twin of the 50-mm reference, so the
  dimensionless solution (tau = t/M^(1/3), Z = r/M^(1/3)) is identical for all
  charge radii.  The extraction pipeline (data/extract_d3plot.py) does
  spherical-shell averaging over true radius, so any 3D solid mesh works; the
  wedge is minimal in cell count.
* Multi-material Eulerian ALE (ELFORM=11, DCT=2, fixed mesh):
  AMMG 1 = TNT (MAT_HIGH_EXPLOSIVE_BURN + EOS_JWL),
  AMMG 2 = air  (MAT_NULL + EOS_LINEAR_POLYNOMIAL, ideal gas gamma=1.4).
* Charge placed by *INITIAL_VOLUME_FRACTION_GEOMETRY (sphere of radius R_0),
  ignited from the centre with *INITIAL_DETONATION.
* d3plot every dt_out s; NEIPH=10, STRFLG=0 so ALE history variable 1
  (slot 0) is density and slot 1 is the TNT volume fraction -- matching the
  project config (data.rho_hv_slot: 0, data.vf_tnt_slot: [1]).
* Non-reflecting outer boundary; the 4 wedge side faces are impermeable
  walls, which for a 1D-spherical solution is exactly the reflective
  symmetry condition (no tangential flow).

Units: SI (m, kg, s, Pa).  R_0 = 0.1 m (100 mm) by default.

Usage
-----
    python scripts/gen_tnt_spherical_deck.py --R0 0.1 --out decks/tnt_spherical_100mm.k
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# physics parameters (SI) -- from configs/tnt_spherical_*.yaml / CLAUDE.md
# ----------------------------------------------------------------------------
RHO_TNT = 1630.0        # kg/m^3
D_CJ = 6930.0           # m/s detonation velocity
P_CJ = 21.0e9           # Pa  (LS-DYNA card nominal; JWL self-consistent is 21.83 GPa)

JWL_A = 371.2e9         # Pa
JWL_B = 3.7471e9        # Pa
JWL_R1 = 4.15
JWL_R2 = 0.95
JWL_OMEG = 0.30
JWL_E0 = 6.0e9          # J/m^3

AIR_RHO = 1.225         # kg/m^3
AIR_GAMMA = 1.4
AIR_P0 = 101325.0       # Pa
AIR_E0 = AIR_P0 / (AIR_GAMMA - 1.0)   # 253312.5 J/m^3

# domain -- Hopkinson-Cranz scaled from the 50-mm reference deck: every length
# and every time scales with R_0 / R_REF, so the dimensionless solution
# (tau = t/M^(1/3), Z = r/M^(1/3)) is identical for all charge radii and a
# multi-radius neural network can train on merged data.
R_REF = 0.05            # m reference charge radius ("50 mm" deck)
X_END_FACTOR = 50.0     # outer radius = X_END_FACTOR * R_0   (2.5 m @ 50 mm)
R_MIN_FACTOR = 0.04     # centre hole  = R_MIN_FACTOR * R_0   (2 mm @ 50 mm)
ENDTIME_REF = 1.0e-3    # s termination time @ 50 mm (scales with R_0 / R_REF)
DT_PLOT_REF = 1.0e-6    # s d3plot interval   @ 50 mm (scales with R_0 / R_REF)
DT_GLSTAT_REF = 5.0e-5  # s glstat/matsum     @ 50 mm
DTMIN_REF = 1.0e-9      # s timestep floor    @ 50 mm

# wedge angular extent (radians); 1 element across each direction
THETA_LO = math.radians(10.0)
THETA_HI = math.radians(20.0)
PHI_LO = math.radians(0.0)
PHI_HI = math.radians(10.0)


def radial_mesh(r_min: float, r_end: float, R_0: float,
                n_cells_charge: int = 50) -> list[float]:
    """Graded radial node distribution, scaled with R_0 (Hopkinson-Cranz).

    Region boundaries and cell sizes are proportional to R_0, so the number of
    cells across the charge (and across R_c) is identical for every radius --
    the scaled numerical solution collapses onto one curve:

      [r_min,   2 R_0]   R_0 / N       -- charge + near products (N cells/R_0)
      [2 R_0,   6 R_0]   2 R_0 / N
      [6 R_0,  12 R_0]   4 R_0 / N
      [12 R_0, 24 R_0]   6 R_0 / N
      [24 R_0,  r_end]   8 R_0 / N

    At R_0 = 50 mm, N = 50 reproduces the original 1/2/4/6/8 mm grading.
    The user's Workbench 50 mm deck resolves the charge with ~0.7 mm cells
    (~71 cells/R_0) -- pass ``n_cells_charge=71`` to match that fidelity.
    """
    regions = [
        (r_min,            2.0 * R_0, R_0 / n_cells_charge),
        (2.0 * R_0,        6.0 * R_0, 2.0 * R_0 / n_cells_charge),
        (6.0 * R_0,       12.0 * R_0, 4.0 * R_0 / n_cells_charge),
        (12.0 * R_0,      24.0 * R_0, 6.0 * R_0 / n_cells_charge),
        (24.0 * R_0,      r_end,       8.0 * R_0 / n_cells_charge),
    ]
    r = [r_min]
    for a, b, dr in regions:
        # step carefully so consecutive region boundaries coincide
        rk = r[-1]
        while rk + 1e-12 < b - 1e-12:
            rk = min(rk + dr, b)
            r.append(round(rk, 12))
    return r


def corner(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """Cartesian point on the wedge (theta = polar angle from +x, phi azimuth)."""
    x = r * math.cos(theta)
    y = r * math.sin(theta) * math.cos(phi)
    z = r * math.sin(theta) * math.sin(phi)
    return x, y, z


def hex_signed_volume(p: list[tuple[float, float, float]]) -> float:
    """Signed volume of a hex (trilinear Jacobian at the centre).

    Node order 1..8: bottom face 1-4, top face 5-8, edges 1-5..4-8.
    Positive => valid (positive Jacobian) element.
    """
    x1, y1, z1 = p[0]; x2, y2, z2 = p[1]
    x3, y3, z3 = p[2]; x4, y4, z4 = p[3]
    x5, y5, z5 = p[4]; x6, y6, z6 = p[5]
    x7, y7, z7 = p[6]; x8, y8, z8 = p[7]
    # dP/dxi, dP/deta, dP/dzeta at centre (0,0,0)
    dxi = ((x2 + x3 + x6 + x7) - (x1 + x4 + x5 + x8)) / 8.0
    deta = ((x3 + x4 + x7 + x8) - (x1 + x2 + x5 + x6)) / 8.0
    dzeta = ((x5 + x6 + x7 + x8) - (x1 + x2 + x3 + x4)) / 8.0
    # same for y and z
    dyi = ((y2 + y3 + y6 + y7) - (y1 + y4 + y5 + y8)) / 8.0
    ddet = ((y3 + y4 + y7 + y8) - (y1 + y2 + y5 + y6)) / 8.0
    dze = ((y5 + y6 + y7 + y8) - (y1 + y2 + y3 + y4)) / 8.0
    dzi = ((z2 + z3 + z6 + z7) - (z1 + z4 + z5 + z8)) / 8.0
    ddetz = ((z3 + z4 + z7 + z8) - (z1 + z2 + z5 + z6)) / 8.0
    dzez = ((z5 + z6 + z7 + z8) - (z1 + z2 + z3 + z4)) / 8.0
    jac = [[dxi, dyi, dzi], [deta, ddet, ddetz], [dzeta, dze, dzez]]
    return (jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]))


def np_centroid_r(pts: list[tuple[float, float, float]]) -> float:
    """Radius of the element centroid (mean of the 8 corners)."""
    cx = sum(p[0] for p in pts) / 8.0
    cy = sum(p[1] for p in pts) / 8.0
    cz = sum(p[2] for p in pts) / 8.0
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def build_mesh(R_0: float, r_end: float, r_min: float,
               n_cells_charge: int = 71, cross_n: int = 16):
    """Return (nodes, elements) lists for a 1/8-sphere octant wedge.

    Replicates the user's proven Workbench 50 mm mesh: theta in [90,175] deg
    (the -x half), phi in [0,90] deg.  The polar axis (theta in [175,180]) is
    left as a tiny unmapped cap gap: LS-DYNA rejects collapsed pole elements
    (``the first 4 nodes must be unique``), so every element keeps 8 unique
    nodes and the cells near the gap stay reasonably shaped (sin theta >= 0.09).

    ``n_cells_charge`` = radial cells across R_0 (Workbench uses ~71).
    ``cross_n``        = angular cells per direction (theta AND phi each split
                         into ``cross_n`` intervals; ~16 is comparable to the
                         Workbench's angular resolution at the charge surface).
    """
    radii = radial_mesh(r_min, r_end, R_0, n_cells_charge)
    n_rings = len(radii)

    # Octant wedge: theta (polar from +x) in [90, TH_HI], phi in [0,90].  The
    # -x polar axis is NOT meshed: TH_HI < 180 leaves a tiny polar-cap gap, so
    # every element keeps 8 unique nodes (LS-DYNA rejects collapsed poles:
    # "the first 4 nodes must be unique") while the cells near the gap stay
    # reasonably shaped (sin(theta) is bounded away from 0).
    TH_LO, TH_HI = math.radians(90.0), math.radians(175.0)
    PH_LO, PH_HI = math.radians(0.0), math.radians(90.0)
    NTH, NPH = cross_n, cross_n
    thetas = [TH_LO + (TH_HI - TH_LO) * i / NTH for i in range(NTH + 1)]
    phis = [PH_LO + (PH_HI - PH_LO) * j / NPH for j in range(NPH + 1)]

    nodes: list[tuple[int, float, float, float]] = []
    ring_node_ids: list[list[int]] = []   # ring i -> (NTH+1)*(NPH+1) unique ids
    interior_ids: set[int] = set()        # free nodes NOT on a symmetry face
    nid = 1
    for rr in radii:
        ids: list[int] = []
        for it in range(NTH + 1):
            for ip in range(NPH + 1):
                x, y, z = corner(rr, thetas[it], phis[ip])
                nodes.append((nid, x, y, z))
                ids.append(nid)
                if 0 < it < NTH and 0 < ip < NPH:
                    interior_ids.add(nid)
                nid += 1
        ring_node_ids.append(ids)

    elements: list[tuple[int, int, list[int]]] = []
    outer_quads: list[list[int]] = []   # top faces of the outermost ring pair
    eid = 1
    min_vol, max_vol = 1e300, -1e300
    n_negative = 0
    n_tnt = 0
    for i in range(n_rings - 1):
        b = ring_node_ids[i]
        t = ring_node_ids[i + 1]
        for it in range(NTH):
            for ip in range(NPH):
                b0 = b[it * (NPH + 1) + ip]
                b1 = b[it * (NPH + 1) + ip + 1]
                b2 = b[(it + 1) * (NPH + 1) + ip + 1]
                b3 = b[(it + 1) * (NPH + 1) + ip]
                t0 = t[it * (NPH + 1) + ip]
                t1 = t[it * (NPH + 1) + ip + 1]
                t2 = t[(it + 1) * (NPH + 1) + ip + 1]
                t3 = t[(it + 1) * (NPH + 1) + ip]
                conn = [b0, b1, b2, b3, t0, t1, t2, t3]
                pts = [nodes[j - 1][1:] for j in conn]
                vol = hex_signed_volume(pts)
                min_vol = min(min_vol, vol)
                max_vol = max(max_vol, vol)
                if vol <= 0:
                    n_negative += 1
                    conn = [b0, b3, b2, b1, t0, t3, t2, t1]
                    pts = [nodes[j - 1][1:] for j in conn]
                    vol2 = hex_signed_volume(pts)
                    if vol2 <= 0:
                        print(f"[mesh] WARNING element {eid} non-positive "
                              f"volume ({vol2:.3e}) even after flip",
                              file=sys.stderr)
                    else:
                        n_negative = 0
                rc = float(np_centroid_r(pts))
                pid = 1 if rc <= R_0 else 2
                if pid == 1:
                    n_tnt += 1
                elements.append((eid, pid, conn))
                if i == n_rings - 2:
                    outer_quads.append(list(conn[4:8]))
                eid += 1
    print(f"[mesh] elements in TNT part (r<R0): {n_tnt}, air part: {len(elements)-n_tnt}")

    print(f"[mesh] rings={n_rings} nodes={len(nodes)} elements={len(elements)} "
          f"(octant {cross_n}x{cross_n} angular, {n_cells_charge} cells/R0)")
    print(f"[mesh] element volume range: {min_vol:.4e} .. {max_vol:.4e} m^3 "
          f"(R0={R_0} m, charge volume={(4/3)*math.pi*R_0**3:.4e})")
    print(f"[mesh] negative-volume elements: {n_negative}")
    return nodes, elements, interior_ids, outer_quads


# ----------------------------------------------------------------------------
# keyword file
# ----------------------------------------------------------------------------
def fmt(values) -> str:
    """10-character-column fixed format (LS-DYNA data-card native).

    Data cards (*CONTROL_*, *MAT_*, *EOS_*, *PART, *ALE_*, *DATABASE_*,
    *INITIAL_*) are read as 10-char fields -- this matches the Workbench
    generated input.k exactly (e.g. `         2    371200    3747.1`).
    Integers are written as integers (scientific notation breaks integer
    fields); floats use uppercase ``E`` like the ANSYS-generated decks.
    *ELEMENT_SOLID uses 8-char fields; *NODE uses 8-char id + 16-char coords.
    """
    out = []
    for v in values:
        if isinstance(v, int):
            out.append(f"{v:>10d}")
        else:
            out.append(f"{v:>10.6G}")
    return "".join(out)


def write_deck(out: Path, R_0: float, endtime: float | None = None,
               n_cells_charge: int = 71, cross_n: int = 16):
    lam = R_0 / R_REF
    r_end = X_END_FACTOR * R_0
    r_min = R_MIN_FACTOR * R_0
    nodes, elements, interior_ids, outer_quads = build_mesh(
        R_0, r_end, r_min, n_cells_charge, cross_n)
    if endtime is None:
        endtime = ENDTIME_REF * lam
    dt_plot = DT_PLOT_REF * lam
    dt_glstat = DT_GLSTAT_REF * lam
    dtmin = DTMIN_REF * lam

    lines: list[str] = []
    w = lines.append

    w("*KEYWORD")
    w(f"*TITLE")
    w(f"Spherical TNT R0={R_0*1e3:.0f} mm air blast  |  SI units (m, kg, s, Pa)")
    w("$ Generated by scripts/gen_tnt_spherical_deck.py")

    w("*CONTROL_ALE")
    w("$     dct     nadv     meth     afac     bfac     cfac     dfac     efac")
    w(fmt([-1, 1, 1, -1.0, 0.0, 0.0, 0.0, 0.0]))
    w("$    start       end     aafac     vfact      prit       ebc      pref   nsidebc")
    w(fmt([0.0, 1.0e20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    w("$ DCT=2 fixed Eulerian mesh, METH=1 donor cell advection; the wedge")
    w("$ side faces are sealed with BOUNDARY_SPC_SET (symmetry walls).")

    w("*CONTROL_TERMINATION")
    w("$    endtim    endcyc      dtmin    endeng    endmas")
    w(fmt([endtime, 10000000, dtmin, 0, 100000]))

    w("*CONTROL_TIMESTEP")
    w("$    dtinit    tssfac       isdo     tslmit      dt2ms      lctm     erode     ms1st")
    w(fmt([0.0, 0.9, 0, 0, 0.0, 0, 1, 1]))

    # High-fidelity ALE settings replicated from the user's proven Workbench
    # 50 mm deck (F:/tnt/tnt_files/dp0/SYS/MECH/input.k).  These sharpen the
    # detonation front / shock: without them the donor-cell ALE solution is
    # over-diffused (peak P ~9 GPa vs ~14 GPa, slower shock).
    w("*CONTROL_BULK_VISCOSITY")
    w("$        q1        q2      type")
    w(fmt([1.5, 0.06, -2]))

    w("*CONTROL_ACCURACY")
    w("$      osu       inn    pidosu")
    w(fmt([0, 1, 0]))

    w("*CONTROL_SOLID")
    w("$    esort    fmatrx   niptets    swlocl")
    w(fmt([1, 0, 0, 0]))

    w("*CONTROL_HOURGLASS")
    w("$      ihq        qh")
    w(fmt([5, 0.1]))

    w("*CONTROL_ENERGY")
    w("$      hgen      rwen     slnten      rylen")
    w(fmt([2, 2, 2, 2]))

    w("*SECTION_SOLID")
    w("$     secid    elform")
    w(fmt([1, 11]))   # ELFORM=11: 1-pt multi-material ALE

    w("*MAT_HIGH_EXPLOSIVE_BURN")
    w("$      mid        ro         d       pcj      beta         k         g      sigy")
    w(fmt([1, RHO_TNT, D_CJ, P_CJ, 0.0]))

    w("*EOS_JWL")
    w("$    eosid         a         b        r1        r2      omeg        e0        v0")
    w(fmt([1, JWL_A, JWL_B, JWL_R1, JWL_R2, JWL_OMEG, JWL_E0, 1.0]))

    w("*MAT_NULL")
    w("$      mid        ro        pc        mu     terod     cerod        ym        pr")
    w(fmt([2, AIR_RHO, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    w("*EOS_LINEAR_POLYNOMIAL")
    w("$    eosid        c0        c1        c2        c3        c4        c5        c6")
    w(fmt([2, 0.0, 0.0, 0.0, 0.0, AIR_GAMMA - 1.0, AIR_GAMMA - 1.0, 0.0]))
    w("$         e0        v0")
    w(fmt([AIR_E0, 1.0]))

    # no *HOURGLASS card: LS-DYNA auto-manages hourglass for ALE parts
    # (matches the Workbench-generated deck).

    # parts: 1 = TNT (owns charge elements), 2 = air (owns ambient elements).
    # LS-DYNA *PART reads a PART-NAME line before each data line; without it
    # the first data line is consumed as the name and the part is lost.
    w("*PART")
    w("air_mesh")
    w(fmt([2, 1, 2, 2, 0]))   # pid 2: air, MAT_NULL (mid 2) + LP EOS (eosid 2)
    w("*PART")
    w("tnt_charge")
    w(fmt([1, 1, 1, 1, 0]))   # pid 1: TNT, HEB (mid 1) + JWL (eosid 1)

    # elements (all in part 2 = air part) -- *ELEMENT_SOLID uses 8-char fields
    w("*ELEMENT_SOLID")
    w("$      eid      pid      n1      n2      n3      n4      n5      n6      n7      n8")
    for (eid, pid, conn) in elements:
        w("".join(f"{v:>8d}" for v in [eid, pid] + list(conn)))

    # nodes
    w("*NODE")
    w("$      nid        x        y        z")
    for (nid, x, y, z) in nodes:
        w(f"{nid:>8d}{x:>16.8e}{y:>16.8e}{z:>16.8e}")

    w("*ALE_MULTI-MATERIAL_GROUP")
    w("$      sid   idtype")
    w(fmt([1, 1]))   # AMMG 1 = part 1 (TNT, owns charge elements)
    w(fmt([2, 1]))   # AMMG 2 = part 2 (air, owns ambient elements)

    w("*INITIAL_DETONATION")
    w("$      pid        x        y        z       lt")
    w(fmt([0, 0.0, 0.0, 0.0, 0.0]))

    # Symmetry: the wedge's 4 side faces (theta_lo/theta_hi cones and phi=0/
    # phi=DF faces) must be impermeable walls -- otherwise Eulerian advection
    # leaks mass out the sides.  Constrain the side-face nodes (all non-interior
    # nodes) EXCEPT the outermost ring (the non-reflecting open boundary), with
    # SPCs fixing all 3 translational DOFs -- mirroring the proven Workbench
    # deck's BOUNDARY_SPC_SET.  Interior cross-section nodes stay free so the
    # ALE solve is not frozen.
    # Symmetry walls on the octant's 3 cut faces (x=0, y=0, z=0) + the polar
    # axis line; the outermost ring stays free (non-reflecting open boundary).
    outer_ids = {nid for (nid, x, y, z) in nodes
                 if math.sqrt(x * x + y * y + z * z) > r_end - 1e-9}
    spc_ids = [nid for (nid, x, y, z) in nodes
               if nid not in outer_ids and nid not in interior_ids]
    w("*SET_NODE_LIST")
    w("$      sid")
    w(fmt([2]))
    w("$      nid       nid       nid       nid       nid       nid       nid       nid")
    for i in range(0, len(spc_ids), 8):
        w(fmt(spc_ids[i:i + 8]))
    w("*BOUNDARY_SPC_SET")
    w("$     nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz")
    w(fmt([2, 0, 1, 1, 1, 0, 0, 0]))

    # outer boundary: non-reflecting on the r = r_end face (all outer quads).
    w("*SET_SEGMENT")
    w("$      sid")
    w(fmt([1]))
    w("$      n1      n2      n3      n4")
    for q in outer_quads:
        w(fmt(q))
    w("*BOUNDARY_NON_REFLECTING")
    w("$     ssid")
    w(fmt([1]))

    w("*DATABASE_EXTENT_BINARY")
    w("$     neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg")
    w(fmt([0, 0, 0, 0, 1, 3, 0, 0]))
    w("$   cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat")
    w(fmt([0, 0, 0, 0, 0, 0, 0, 0]))
    w("$  nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout")
    w(fmt([0, 0, 0, 0, 2, 0, 0, 0]))

    w("*DATABASE_BINARY_D3PLOT")
    w("$       dt      lcdt      beam     npltc    psetid")
    w(fmt([dt_plot, 0, 0, 0, 0]))
    w("*DATABASE_GLSTAT")
    w(f"{dt_glstat:>10.4G}")
    w("*DATABASE_MATSUM")
    w(f"{dt_glstat:>10.4G}")
    w("*END")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[deck] wrote {out}  ({len(lines)} lines, "
          f"{len(elements)} elements, {len(nodes)} nodes)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R0", type=float, default=0.1, help="charge radius (m)")
    ap.add_argument("--endtime", type=float, default=None,
                    help="termination time (s); default scales with R0 (1 ms @ 50 mm)")
    ap.add_argument("--cells-across-r0", type=int, default=71,
                    help="radial cells across the charge radius (Workbench 50mm "
                         "uses ~71; larger = sharper detonation front)")
    ap.add_argument("--cross-n", type=int, default=16,
                    help="angular cells per direction in the 1/8-sphere octant "
                         "wedge (theta and phi each split into cross_n)")
    ap.add_argument("--out", type=Path, default=Path("decks/tnt_spherical_100mm.k"))
    args = ap.parse_args()
    write_deck(args.out, args.R0, endtime=args.endtime,
               n_cells_charge=args.cells_across_r0, cross_n=args.cross_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
