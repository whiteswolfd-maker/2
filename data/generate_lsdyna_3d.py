"""Generate a 3D wedge LS-DYNA keyword file for TRUE spherical blast.

Unlike the 2D axisymmetric model (ELFORM=15, cylindrical symmetry), this
creates a thin 3D conical wedge of hex8 solid elements.  The wedge subtends
a small solid angle (2° × 2°) and uses multi-material ALE (ELFORM=12), so
the geometric divergence is 1/r² — exactly matching the spherical Euler
equations that the PINN solves.

Unit system: g-mm-ms  (length=mm, mass=g, time=ms, pressure=MPa).

Usage
-----
    python -m data.generate_lsdyna_3d
    # -> tnt_blast_3d.k
    # Copy to Windows, then: ls-dyna i=tnt_blast_3d.k memory=500m ncpu=4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


# ======================= Formatting helpers =======================
# LS-DYNA standard format: 8 fields x 10 characters = 80 chars per line


def _f10(v: float) -> str:
    """Format a float into exactly 10 characters."""
    if v == 0.0:
        return "       0.0"
    if abs(v) >= 1e6 or abs(v) < 0.01:
        s = f"{v:10.3E}"
        if len(s) > 10:
            s = f"{v:10.2E}"
        return s
    return f"{v:10.4f}"[:10] if len(f"{v:10.4f}") <= 10 else f"{v:10.3E}"


def _i10(v: int) -> str:
    return f"{v:10d}"


def _i8(v: int) -> str:
    return f"{v:8d}"


def _f16(v: float) -> str:
    return f"{v:16.8f}"


# ======================= Unit conversion: SI -> g-mm-ms =======================


def si_to_gmmms(cfg: dict) -> dict:
    exp = cfg["explosive"]
    air = cfg["air"]
    W = exp["W"]
    rho0_SI = exp["rho0"]
    R0_m = (3.0 * W / (4.0 * math.pi * rho0_SI)) ** (1.0 / 3.0)

    return {
        "R0_mm": R0_m * 1e3,
        "R_far_mm": 5000.0,
        "rho0": rho0_SI * 1e-6,
        "D_CJ": exp["D_CJ"],           # m/s = mm/ms
        "P_CJ": 21.0e9 * 1e-6,         # 21 GPa -> 21000 MPa
        "A": exp["jwl"]["A"] * 1e-6,
        "B": exp["jwl"]["B"] * 1e-6,
        "R1": exp["jwl"]["R1"],
        "R2": exp["jwl"]["R2"],
        "omega": exp["jwl"]["omega"],
        "E0": exp["jwl"]["E0"] * 1e-6,
        "gamma": air["gamma"],
        "rho_a": air["rho_a"] * 1e-6,
        "P_a": air["P_a"] * 1e-6,
        "E_a": air["P_a"] / (air["gamma"] - 1.0) * 1e-6,
        "t_end": cfg["domain"]["t_end"] * 1e3,
    }


# ======================= 3D wedge mesh =======================


def generate_3d_wedge_mesh(
    R0: float,
    R_far: float,
    n_tnt: int,
    n_air: int,
    r_min: float = 0.5,
    wedge_deg: float = 2.0,
):
    """Build a thin 3D conical wedge of hex8 elements.

    The wedge is placed near the equator (theta ~ 90 deg from z-axis)
    in the first quadrant (y >= 0, z >= 0).  Two faces lie exactly on
    coordinate planes (z=0 and y=0), so symmetry BCs are simple SPC
    constraints.

    Parameters
    ----------
    R0      : float  – TNT charge outer radius (mm).
    R_far   : float  – far-field boundary radius (mm).
    n_tnt   : int    – number of radial elements in the TNT region.
    n_air   : int    – number of radial elements in the air region.
    r_min   : float  – inner radius (avoids r=0 singularity).
    wedge_deg : float – opening angle of the wedge in degrees.

    Returns
    -------
    nodes : list of (nid, x, y, z)
    elems : list of (eid, pid, n1..n8)
    n_r   : total number of radial node layers
    """
    # Radial node positions
    r_tnt = [r_min + (R0 - r_min) * i / n_tnt for i in range(n_tnt + 1)]
    r_air = [R0 + (R_far - R0) * i / n_air for i in range(1, n_air + 1)]
    radii = r_tnt + r_air
    n_r = len(radii)       # number of radial node layers
    n_elem = n_r - 1       # number of radial elements

    # Angular positions: wedge from (theta=90-wedge, phi=0) to (theta=90, phi=wedge)
    # Face at theta=90 deg (z=0 plane) -> SPC DOF z
    # Face at phi=0 deg   (y=0 plane) -> SPC DOF y
    alpha = math.radians(wedge_deg)        # opening angle
    theta_min = math.radians(90.0 - wedge_deg)  # e.g. 88 deg
    theta_max = math.radians(90.0)               # 90 deg
    phi_min = 0.0
    phi_max = alpha

    # 4 angular corners (theta, phi) per radial layer
    # Ordering chosen so hex8 faces have consistent normals.
    #   Corner 0: theta_max, phi_min  -> (r, 0, 0)       on both sym planes
    #   Corner 1: theta_max, phi_max  -> (r*c, r*s, 0)   on z=0 only
    #   Corner 2: theta_min, phi_max  -> (r*cc, r*cs, r*s2) interior
    #   Corner 3: theta_min, phi_min  -> (r*c2, 0, r*s2) on y=0 only
    corners_ang = [
        (theta_max, phi_min),
        (theta_max, phi_max),
        (theta_min, phi_max),
        (theta_min, phi_min),
    ]

    nodes = []
    for k, r in enumerate(radii):
        for j, (th, ph) in enumerate(corners_ang):
            nid = k * 4 + j + 1
            x = r * math.sin(th) * math.cos(ph)
            y = r * math.sin(th) * math.sin(ph)
            z = r * math.cos(th)
            nodes.append((nid, x, y, z))

    # Hex8 elements: inner face 1-2-3-4, outer face 5-6-7-8
    elems = []
    for k in range(n_elem):
        eid = k + 1
        pid = 1 if k < n_tnt else 2
        base_in = k * 4 + 1
        base_out = (k + 1) * 4 + 1
        elems.append((
            eid, pid,
            base_in, base_in + 1, base_in + 2, base_in + 3,
            base_out, base_out + 1, base_out + 2, base_out + 3,
        ))

    return nodes, elems, n_r


# ======================= Keyword writer =======================


def write_keyword(path: Path, u: dict, n_tnt: int, n_air: int,
                  wedge_deg: float = 2.0) -> None:
    nodes, elems, n_r = generate_3d_wedge_mesh(
        R0=u["R0_mm"], R_far=u["R_far_mm"],
        n_tnt=n_tnt, n_air=n_air, wedge_deg=wedge_deg,
    )
    n_nodes = len(nodes)

    L: list[str] = []

    # ---- Header
    L.append("*KEYWORD")
    L.append("*TITLE")
    L.append(f"TNT Spherical Blast 1kg – 3D Wedge {wedge_deg:.0f}deg (g-mm-ms)")
    L.append("$")

    # ---- Control: ALE (3D, Eulerian mesh)
    # DCT=2 (Eulerian): mesh fixed, material flows through elements.
    # ELFORM=12 (MMALE) allows multi-material tracking per element.
    L.append("*CONTROL_ALE")
    L.append("$      DCT      NADV      METH      AFAC      BFAC"
             "      CFAC      DFAC      EFAC")
    L.append(f"{_i10(2)}{_i10(1)}{_i10(2)}{_f10(0.0)}{_f10(0.0)}"
             f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    L.append("$    START       END     AAFAC     VFACT      PRIT"
             "       EBC      PREF   NSIDEBC")
    L.append(f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(1.0e-6)}"
             f"{_i10(0)}{_i10(0)}{_f10(0.0)}{_i10(0)}")

    # ---- Control: Termination
    L.append("*CONTROL_TERMINATION")
    L.append("$   ENDTIM    ENDCYC     DTMIN    ENDENG    ENDMAS")
    L.append(f"{_f10(u['t_end'])}{_i10(0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")

    # ---- Control: Timestep
    L.append("*CONTROL_TIMESTEP")
    L.append("$   DTINIT    TSSFAC      ISDO    TSLIMT"
             "     DT2MS      LCTM     ERODE     MS1ST")
    L.append(f"{_f10(0.0)}{_f10(0.3)}{_i10(0)}{_f10(0.0)}"
             f"{_f10(0.0)}{_i10(0)}{_i10(0)}{_i10(0)}")

    # ---- Control: Bulk viscosity
    L.append("*CONTROL_BULK_VISCOSITY")
    L.append("$       Q1        Q2      TYPE     BTYPE")
    L.append(f"{_f10(1.5)}{_f10(0.06)}{_i10(1)}{_i10(0)}")

    # ---- Control: Energy
    L.append("*CONTROL_ENERGY")
    L.append("$     HGEN      RWEN    SLNTEN     RYLEN")
    L.append(f"{_i10(2)}{_i10(2)}{_i10(2)}{_i10(2)}")

    # ---- Database
    dt_out = u["t_end"] / 1000.0
    dt_plot = u["t_end"] / 300.0
    L.append("*DATABASE_GLSTAT")
    L.append(f"{_f10(dt_out)}")
    L.append("*DATABASE_BINARY_D3PLOT")
    L.append(f"{_f10(dt_plot)}")
    L.append("*DATABASE_BINARY_D3THDT")
    L.append(f"{_f10(dt_out)}")

    # ---- Material 1: TNT (High-Explosive Burn + JWL EOS)
    L.append("$")
    L.append("$ =========== MATERIAL 1: TNT ===========")
    L.append("$")
    L.append("*MAT_HIGH_EXPLOSIVE_BURN")
    L.append("$      MID        RO         D       PCJ"
             "      BETA         K         G      SIGY")
    L.append(f"{_i10(1)}{_f10(u['rho0'])}{_f10(u['D_CJ'])}{_f10(u['P_CJ'])}"
             f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    L.append("*EOS_JWL")
    L.append("$    EOSID         A         B        R1"
             "        R2      OMEG        E0        V0")
    L.append(f"{_i10(1)}{_f10(u['A'])}{_f10(u['B'])}{_f10(u['R1'])}"
             f"{_f10(u['R2'])}{_f10(u['omega'])}{_f10(u['E0'])}{_f10(1.0)}")
    L.append("*INITIAL_DETONATION")
    L.append("$      PID         X         Y         Z     BTIME")
    L.append(f"{_i10(1)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")

    # ---- Material 2: Air (Null + Linear Polynomial EOS)
    L.append("$")
    L.append("$ =========== MATERIAL 2: AIR ===========")
    L.append("$")
    L.append("*MAT_NULL")
    L.append("$      MID        RO        PC        MU"
             "     TEROD     CEROD        YM        PR")
    L.append(f"{_i10(2)}{_f10(u['rho_a'])}{_f10(0.0)}{_f10(0.0)}"
             f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    gm1 = u["gamma"] - 1.0
    L.append("*EOS_LINEAR_POLYNOMIAL")
    L.append("$    EOSID        C0        C1        C2"
             "        C3        C4        C5        C6")
    L.append(f"{_i10(2)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}"
             f"{_f10(gm1)}{_f10(gm1)}{_f10(0.0)}")
    L.append("$       E0        V0")
    L.append(f"{_f10(u['E_a'])}{_f10(1.0)}")

    # ---- Section: 3D solid, multi-material ALE (ELFORM=12)
    L.append("$")
    L.append("$ =========== SECTION ===========")
    L.append("$")
    L.append("*SECTION_SOLID")
    L.append("$    SECID    ELFORM       AET")
    L.append(f"{_i10(1)}{_i10(12)}{_i10(0)}")

    # ---- Parts
    L.append("$")
    L.append("*PART")
    L.append("TNT explosive")
    L.append("$      PID     SECID       MID     EOSID"
             "      HGID      GRAV    ADPOPT      TMID")
    L.append(f"{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}"
             f"{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}")
    L.append("*PART")
    L.append("Air")
    L.append("$      PID     SECID       MID     EOSID"
             "      HGID      GRAV    ADPOPT      TMID")
    L.append(f"{_i10(2)}{_i10(1)}{_i10(2)}{_i10(2)}"
             f"{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}")

    # ---- ALE multi-material group (both materials in same group)
    L.append("$")
    L.append("$ =========== ALE MULTI-MATERIAL ===========")
    L.append("$")
    L.append("*ALE_MULTI-MATERIAL_GROUP")
    L.append("$      SID    IDTYPE    GPNAME")
    L.append(f"{_i10(1)}{_i10(1)}")
    L.append(f"{_i10(2)}{_i10(1)}")

    # ---- Nodes
    L.append("$")
    L.append("$ =========== NODES ===========")
    L.append("$")
    L.append("*NODE")
    for nid, x, y, z in nodes:
        L.append(f"{_i8(nid)}{_f16(x)}{_f16(y)}{_f16(z)}")

    # ---- Elements (hex8 solids)
    L.append("$")
    L.append("$ =========== ELEMENTS ===========")
    L.append("$")
    L.append("*ELEMENT_SOLID")
    for eid, pid, n1, n2, n3, n4, n5, n6, n7, n8 in elems:
        L.append(
            f"{_i8(eid)}{_i8(pid)}"
            f"{_i8(n1)}{_i8(n2)}{_i8(n3)}{_i8(n4)}"
            f"{_i8(n5)}{_i8(n6)}{_i8(n7)}{_i8(n8)}"
        )

    # ---- Boundary conditions: symmetry planes
    # All nodes constrained in y and z (radial-only motion).
    # For the small wedge angle, this enforces spherical symmetry with
    # < 0.1% tangential error (sin²(2°) ≈ 0.001).
    L.append("$")
    L.append("$ =========== BOUNDARY CONDITIONS ===========")
    L.append("$")
    L.append("*SET_NODE_LIST_TITLE")
    L.append("All_nodes")
    L.append("$      SID       DA1       DA2       DA3       DA4")
    L.append(f"{_i10(1)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    for i in range(0, n_nodes, 8):
        chunk = list(range(i + 1, min(i + 9, n_nodes + 1)))
        L.append("".join(_i10(n) for n in chunk))

    L.append("*BOUNDARY_SPC_SET")
    L.append("$     NSID       CID      DOFX      DOFY      DOFZ"
             "     DOFRX     DOFRY     DOFRZ")
    L.append(f"{_i10(1)}{_i10(0)}{_i10(0)}{_i10(1)}{_i10(1)}"
             f"{_i10(1)}{_i10(1)}{_i10(1)}")

    # ---- Hourglass
    L.append("$")
    L.append("*HOURGLASS")
    L.append("$     HGID       IHQ        QH")
    L.append(f"{_i10(1)}{_i10(4)}{_f10(0.1)}")

    L.append("$")
    L.append("*END")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ======================= Main =======================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 3D wedge LS-DYNA keyword file for spherical blast",
    )
    ap.add_argument("--config", default="configs/tnt_spherical.yaml")
    ap.add_argument("--output", default="tnt_blast_3d.k")
    ap.add_argument("--n-tnt", type=int, default=50)
    ap.add_argument("--n-air", type=int, default=500)
    ap.add_argument("--wedge-deg", type=float, default=2.0,
                    help="wedge opening angle in degrees (default: 2)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    u = si_to_gmmms(cfg)
    out = Path(args.output)
    write_keyword(out, u, n_tnt=args.n_tnt, n_air=args.n_air,
                  wedge_deg=args.wedge_deg)

    n_elem = args.n_tnt + args.n_air
    n_nodes = (n_elem + 1) * 4
    print(f"[OK] Wrote {out}")
    print(f"     R0 = {u['R0_mm']:.2f} mm, R_far = {u['R_far_mm']:.0f} mm")
    print(f"     {args.n_tnt} TNT + {args.n_air} air hex8 elements = {n_elem} total")
    print(f"     {n_nodes} nodes (4 per radial layer)")
    print(f"     Wedge angle = {args.wedge_deg:.1f} deg")
    print(f"     End time = {u['t_end']:.2f} ms")
    print(f"     ELFORM=12 (multi-material ALE, true spherical symmetry)")


if __name__ == "__main__":
    main()
