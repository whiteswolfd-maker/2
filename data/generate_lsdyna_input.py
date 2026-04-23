"""Generate a complete LS-DYNA keyword file for TNT spherical blast.

Reads physical parameters from ``configs/tnt_spherical.yaml`` and writes
a ready-to-run keyword file with:
- 2D axisymmetric ALE mesh (TNT + air), ELFORM=15
- JWL EOS for TNT, ideal gas EOS for air
- Point detonation at origin
- Symmetry BCs on top/bottom faces

Unit system: g-mm-ms  (length=mm, mass=g, time=ms, pressure=MPa).

Usage
-----
    python -m data.generate_lsdyna_input
    # -> tnt_blast.k
    # Copy to Windows, then: ls-dyna i=tnt_blast.k memory=500m ncpu=4
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
        "R_far_mm": 5000.0,  # 5 m — large enough that blast won't reflect back
        "rho0": rho0_SI * 1e-6,
        "D_CJ": exp["D_CJ"],  # m/s = mm/ms (numerically same)
        "P_CJ": 21.0e9 * 1e-6,  # 21 GPa -> 21000 MPa
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


# ======================= Mesh =======================


def generate_mesh(R0: float, R_far: float, n_tnt: int, n_air: int,
                  r_min: float = 0.5, y_thick: float = 1.0):
    r_tnt = [r_min + (R0 - r_min) * i / n_tnt for i in range(n_tnt + 1)]
    r_air = [R0 + (R_far - R0) * i / n_air for i in range(1, n_air + 1)]
    r_all = r_tnt + r_air
    n_r = len(r_all)

    nodes = []
    for i, r in enumerate(r_all):
        nodes.append((i + 1, r, 0.0, 0.0))
    for i, r in enumerate(r_all):
        nodes.append((n_r + i + 1, r, y_thick, 0.0))

    elems = []
    for i in range(n_r - 1):
        pid = 1 if i < n_tnt else 2
        elems.append((i + 1, pid, i + 1, i + 2, n_r + i + 2, n_r + i + 1))

    return nodes, elems, n_r


# ======================= Keyword writer =======================


def write_keyword(path: Path, u: dict, n_tnt: int, n_air: int) -> None:
    nodes, elems, n_r = generate_mesh(
        R0=u["R0_mm"], R_far=u["R_far_mm"],
        n_tnt=n_tnt, n_air=n_air,
    )

    L = []

    # ---- Header
    L.append("*KEYWORD")
    L.append("*TITLE")
    L.append("TNT Spherical Blast 1kg (g-mm-ms)")
    L.append("$")

    # ---- Control: ALE (2D axisymmetric) - needs 2 cards in R12
    L.append("*CONTROL_ALE")
    L.append("$      DCT      NADV      METH      AFAC      BFAC      CFAC      DFAC      EFAC")
    L.append(f"{_i10(2)}{_i10(1)}{_i10(2)}{_f10(-1.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    L.append("$    START       END     AAFAC     VFACT      PRIT       EBC      PREF   NSIDEBC")
    L.append(f"{_f10(0.0)}{_f10(0.0)}{_f10(1.0)}{_f10(1.0e-6)}{_i10(0)}{_i10(0)}{_f10(0.0)}{_i10(0)}")

    # ---- Control: Termination
    L.append("*CONTROL_TERMINATION")
    L.append("$   ENDTIM    ENDCYC     DTMIN    ENDENG    ENDMAS")
    L.append(f"{_f10(u['t_end'])}{_i10(0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")

    # ---- Control: Timestep
    L.append("*CONTROL_TIMESTEP")
    L.append("$   DTINIT    TSSFAC      ISDO    TSLIMT     DT2MS      LCTM     ERODE     MS1ST")
    L.append(f"{_f10(0.0)}{_f10(0.3)}{_i10(0)}{_f10(0.0)}{_f10(0.0)}{_i10(0)}{_i10(0)}{_i10(0)}")

    # ---- Control: Bulk viscosity
    L.append("*CONTROL_BULK_VISCOSITY")
    L.append("$       Q1        Q2      TYPE     BTYPE")
    L.append(f"{_f10(1.5)}{_f10(0.06)}{_i10(1)}{_i10(0)}")

    # ---- Database
    dt_out = u['t_end'] / 1000.0
    dt_plot = u['t_end'] / 300.0
    L.append("*DATABASE_GLSTAT")
    L.append(f"{_f10(dt_out)}")
    L.append("*DATABASE_BINARY_D3PLOT")
    L.append(f"{_f10(dt_plot)}")
    L.append("*DATABASE_BINARY_D3THDT")
    L.append(f"{_f10(dt_out)}")
    L.append("*DATABASE_TRHIST")
    L.append(f"{_f10(dt_out)}")

    # ---- Material 1: TNT
    L.append("$")
    L.append("$ =========== MATERIAL 1: TNT ===========")
    L.append("$")
    L.append("*MAT_HIGH_EXPLOSIVE_BURN")
    L.append("$      MID        RO         D       PCJ      BETA         K         G      SIGY")
    L.append(f"{_i10(1)}{_f10(u['rho0'])}{_f10(u['D_CJ'])}{_f10(u['P_CJ'])}"
             f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    L.append("*EOS_JWL")
    L.append("$    EOSID         A         B        R1        R2      OMEG        E0        V0")
    L.append(f"{_i10(1)}{_f10(u['A'])}{_f10(u['B'])}{_f10(u['R1'])}"
             f"{_f10(u['R2'])}{_f10(u['omega'])}{_f10(u['E0'])}{_f10(1.0)}")
    L.append("*INITIAL_DETONATION")
    L.append("$      PID         X         Y         Z     BTIME")
    L.append(f"{_i10(1)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")

    # ---- Material 2: Air
    L.append("$")
    L.append("$ =========== MATERIAL 2: AIR ===========")
    L.append("$")
    L.append("*MAT_NULL")
    L.append("$      MID        RO        PC        MU     TEROD     CEROD        YM        PR")
    L.append(f"{_i10(2)}{_f10(u['rho_a'])}{_f10(0.0)}{_f10(0.0)}"
             f"{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    gm1 = u["gamma"] - 1.0
    L.append("*EOS_LINEAR_POLYNOMIAL")
    L.append("$    EOSID        C0        C1        C2        C3        C4        C5        C6")
    L.append(f"{_i10(2)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}"
             f"{_f10(gm1)}{_f10(gm1)}{_f10(0.0)}")
    L.append("$       E0        V0")
    L.append(f"{_f10(u['E_a'])}{_f10(1.0)}")

    # ---- Section: 2D axisymmetric shell
    L.append("$")
    L.append("$ =========== SECTION ===========")
    L.append("$")
    L.append("*SECTION_SHELL")
    L.append("$    SECID   ELFORM      SHRF       NIP     PROPT   QR/IRID     ICOMP    SETYP")
    L.append(f"{_i10(1)}{_i10(15)}{_f10(1.0)}{_i10(2)}{_f10(0.0)}{_i10(0)}{_i10(0)}{_i10(0)}")
    L.append("$       T1        T2        T3        T4      NLOC")
    L.append(f"{_f10(1.0)}{_f10(1.0)}{_f10(1.0)}{_f10(1.0)}{_f10(0.0)}")

    # ---- Parts
    L.append("$")
    L.append("*PART")
    L.append("TNT explosive")
    L.append("$      PID     SECID       MID     EOSID      HGID      GRAV    ADPOPT      TMID")
    L.append(f"{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}")
    L.append("*PART")
    L.append("Air")
    L.append("$      PID     SECID       MID     EOSID      HGID      GRAV    ADPOPT      TMID")
    L.append(f"{_i10(2)}{_i10(1)}{_i10(2)}{_i10(2)}{_i10(0)}{_i10(0)}{_i10(0)}{_i10(0)}")

    # ---- Nodes
    L.append("$")
    L.append("$ =========== NODES ===========")
    L.append("$")
    L.append("*NODE")
    for nid, x, y, z in nodes:
        L.append(f"{_i8(nid)}{_f16(x)}{_f16(y)}{_f16(z)}")

    # ---- Elements
    L.append("$")
    L.append("$ =========== ELEMENTS ===========")
    L.append("$")
    L.append("*ELEMENT_SHELL")
    for eid, pid, n1, n2, n3, n4 in elems:
        L.append(f"{_i8(eid)}{_i8(pid)}{_i8(n1)}{_i8(n2)}{_i8(n3)}{_i8(n4)}")

    # ---- Boundary: Symmetry SPCs (constrain y and z motion)
    L.append("$")
    L.append("$ =========== BOUNDARY CONDITIONS ===========")
    L.append("$")

    # Bottom row (y=0): constrain y, z translation + all rotations
    L.append("*SET_NODE_LIST_TITLE")
    L.append("Bottom_row")
    L.append("$      SID       DA1       DA2       DA3       DA4")
    L.append(f"{_i10(1)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    for i in range(0, n_r, 8):
        chunk = list(range(i + 1, min(i + 9, n_r + 1)))
        L.append("".join(_i10(n) for n in chunk))

    # Top row (y=thick): same constraints
    L.append("*SET_NODE_LIST_TITLE")
    L.append("Top_row")
    L.append("$      SID       DA1       DA2       DA3       DA4")
    L.append(f"{_i10(2)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}{_f10(0.0)}")
    for i in range(0, n_r, 8):
        chunk = list(range(n_r + i + 1, min(n_r + i + 9, 2 * n_r + 1)))
        L.append("".join(_i10(n) for n in chunk))

    # Apply SPC to both rows
    L.append("*BOUNDARY_SPC_SET")
    L.append("$     NSID       CID      DOFX      DOFY      DOFZ     DOFRX     DOFRY     DOFRZ")
    L.append(f"{_i10(1)}{_i10(0)}{_i10(0)}{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}")
    L.append("*BOUNDARY_SPC_SET")
    L.append("$     NSID       CID      DOFX      DOFY      DOFZ     DOFRX     DOFRY     DOFRZ")
    L.append(f"{_i10(2)}{_i10(0)}{_i10(0)}{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}{_i10(1)}")

    # ---- Tracers
    L.append("$")
    L.append("$ =========== TRACERS ===========")
    L.append("$")
    for i, r_mm in enumerate([u["R0_mm"], 100, 200, 400, 800, 1500, 3000], start=1):
        if r_mm > u["R_far_mm"]:
            continue
        L.append("*DATABASE_TRACER")
        L.append("$     TRID     TRACK   AMMG_ID       NID    RADIUS     LOCID")
        L.append(f"{_i10(i)}{_i10(2)}{_i10(0)}{_i10(0)}{_f10(0.0)}{_i10(0)}")
        L.append("$        X         Y         Z")
        L.append(f"{_f10(r_mm)}{_f10(0.5)}{_f10(0.0)}")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tnt_spherical.yaml")
    ap.add_argument("--output", default="tnt_blast.k")
    ap.add_argument("--n-tnt", type=int, default=50)
    ap.add_argument("--n-air", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    u = si_to_gmmms(cfg)
    out = Path(args.output)
    write_keyword(out, u, n_tnt=args.n_tnt, n_air=args.n_air)

    print(f"[OK] Wrote {out}")
    print(f"     R0 = {u['R0_mm']:.2f} mm, R_far = {u['R_far_mm']:.0f} mm")
    print(f"     {args.n_tnt} TNT + {args.n_air} air elements")
    print(f"     End time = {u['t_end']:.2f} ms")


if __name__ == "__main__":
    main()
