#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-radius TNT blast dataset factory.

Generate -> run -> extract -> export a whole family of charge radii with one
command.  Every radius is a Hopkinson-Cranz twin of the 50-mm reference
(``scripts/gen_tnt_spherical_deck.py``): cell size, domain radius, run time and
output cadence all scale with R_0, so the extracted datasets share the same
scaled coverage ``(tau = t/M^(1/3), Z = r/M^(1/3))`` and can be merged into a
single multi-radius neural network.

For each radius the script runs, in order:

    python scripts/gen_tnt_spherical_deck.py --R0 <R0> --out decks/tnt_spherical_<R0mm>mm.k
    scripts/run_lsdyna.bat   decks/tnt_spherical_<R0mm>mm.k runs/<tag> <ncpu>
    python -m data.extract_d3plot --d3plot runs/<tag> --out extracted_<tag> \\
        --rho-hv-slot 0 --R-0 <R0> --x-end <50*R0> --n-x 2500 \\
        --rho-contact-threshold 100 --vf-tnt-slot 1
    python scripts/export_training_dataset.py --input extracted_<tag> \\
        --out dataset_<tag> --times-us <slices>

After all radii finish, ``verify_hopkinson_collapse.py`` checks that every
radius collapses onto the first one in scaled coordinates.

Usage
-----
    python scripts/run_multi_radius.py 0.05 0.1 0.5 1.0
    python scripts/run_multi_radius.py 0.05 0.1 0.5 1.0 --ncpu 8 --extract-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# P(r) slice times (us) that land on the same *scaled* times across radii:
# physical times are chosen proportional to R_0 so tau = t/M^(1/3) is the same.
# E.g. tau = 5.3 us/kg^(1/3)  ->  5 us @ 50 mm, 10 us @ 100 mm, 50 us @ 500 mm.
def _slices_us(R_0: float) -> str:
    lam = R_0 / 0.05
    base = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]  # us @ 50mm
    return ",".join(str(round(v * lam, 3)) for v in base)


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT), shell=sys.platform == "win32")


def process_radius(R_0: float, tag: str, ncpu: int,
                   gen: bool, run: bool, extract: bool, export: bool) -> None:
    r_mm = R_0 * 1000
    deck = ROOT / "decks" / f"tnt_spherical_{r_mm:g}mm.k"
    workdir = ROOT / "runs" / tag
    extracted = ROOT / f"extracted_{tag}"
    dataset = ROOT / f"dataset_{tag}"

    x_end = 50.0 * R_0
    slices = _slices_us(R_0)

    if gen:
        _run([sys.executable, "scripts/gen_tnt_spherical_deck.py",
              f"--R0", f"{R_0:g}", "--out", str(deck)])
    if run:
        bat = ROOT / "scripts" / "run_lsdyna.bat"
        _run([str(bat), str(deck), str(workdir), str(ncpu)])
    if extract:
        _run([sys.executable, "-m", "data.extract_d3plot",
              "--d3plot", str(workdir), "--out", str(extracted),
              "--rho-hv-slot", "0", "--R-0", f"{R_0:g}",
              "--x-end", f"{x_end:g}", "--n-x", "2500",
              "--rho-contact-threshold", "100", "--vf-tnt-slot", "1"])
    if export:
        _run([sys.executable, "scripts/export_training_dataset.py",
              "--input", str(extracted), "--out", str(dataset),
              "--times-us", slices])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("radii", type=float, nargs="+", help="charge radii in metres (e.g. 0.05 0.1 0.5 1.0)")
    ap.add_argument("--ncpu", type=int, default=4)
    ap.add_argument("--tag", default="", help="output tag; defaults to R0 in mm")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="after all radii, run the Hopkinson-collapse check")
    args = ap.parse_args()

    tags = []
    for R_0 in args.radii:
        tag = args.tag or f"{R_0*1000:g}mm"
        tags.append(tag)
        process_radius(R_0, tag, args.ncpu,
                       gen=not args.skip_gen, run=not args.skip_run,
                       extract=not args.skip_extract, export=not args.skip_export)

    if args.verify and len(tags) >= 2:
        spec = []
        for tag, R_0 in zip(tags, args.radii):
            spec += [str(ROOT / f"extracted_{tag}"), f"{R_0:g}"]
        _run([sys.executable, "scripts/verify_hopkinson_collapse.py", *spec])
    return 0


if __name__ == "__main__":
    sys.exit(main())
