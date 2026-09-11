"""Recheck events in an existing extraction, without loading d3plot or torch.

python -m data.detect_events --input extracted/ --out event_check/
Old NPZ files lacking material fractions yield explicit insufficient-data
results. Re-extract locally to obtain the original-cell CJ probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.event_detection import EventConfig, PROBE_FIELDS, analyze_events, write_event_outputs


def detect_directory(in_dir, out_dir=None, config_path=None):
    in_dir = Path(in_dir)
    metadata_path = in_dir/"metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    cfg = {}
    if config_path is not None:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    # Recorded extraction geometry takes precedence: never reinterpret 50 mm
    # data as another radius merely by supplying a different training config.
    R_0 = meta.get("R_0", cfg.get("data", {}).get("R_0"))
    if R_0 is None:
        raise ValueError("R_0 missing: provide extraction metadata or --config")
    cfg_R0 = cfg.get("data", {}).get("R_0")
    if cfg_R0 is not None and not np.isclose(float(R_0), float(cfg_R0), rtol=1e-8, atol=0):
        raise ValueError("Config radius differs from extraction metadata; use the matching case")
    options = dict(meta.get("event_criteria", {}))
    overrides = dict(cfg.get("events", {}) or {})
    ref_override = overrides.pop("cj_reference", None)
    options.update(overrides)
    config = EventConfig(**options)
    reference = meta.get("event_cj_reference") or dict(
        P=meta.get("P_CJ", np.nan), rho=meta.get("rho_CJ", np.nan), u=meta.get("u_CJ", np.nan))
    if ref_override is not None:
        if set(ref_override) != {"P", "rho", "u"}:
            raise ValueError("events.cj_reference must contain P, rho and u")
        reference = {k: float(v) for k, v in ref_override.items()}
    with np.load(in_dir/"raw_line.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    probes = None
    if "cj_probe" in arrays:
        # The compact probe was selected with these criteria. A different band,
        # reaction threshold or reference may require a different original cell.
        selection_keys = ("cj_band_inner_fraction", "cj_vf_min", "burn_complete_min")
        if any(options.get(k) != meta.get("event_criteria", {}).get(k) for k in selection_keys) or \
                reference != meta.get("event_cj_reference"):
            raise ValueError("CJ probe selection/reference changed; rerun data.extract_d3plot locally. "
                             "The compact probe cannot recover discarded original-cell states.")
        if meta.get("cj_probe_fields") != list(PROBE_FIELDS) or \
                arrays["cj_probe"].shape != (len(arrays["t"]), len(PROBE_FIELDS)):
            raise ValueError("Unsupported or damaged compact CJ probe layout")
        probes = [{key: float(v) if np.isfinite(v) else None for key, v in zip(PROBE_FIELDS, row)}
                  for row in arrays["cj_probe"]]
    report, rows, probes = analyze_events(
        arrays["t"], arrays["x"], arrays["rho"], arrays["u"], arrays["P"],
        R_0=float(R_0), P_atm=float(meta.get("P_atm", cfg.get("air", {}).get("P_a", 101325.0))),
        rho_air=float(meta.get("rho_air", cfg.get("air", {}).get("rho_a", 1.225))),
        reference=reference, config=config, vf=arrays.get("vf_tnt"),
        burn=arrays.get("burn_fraction"), sample_count=arrays.get("sample_count"),
        cell_width=arrays.get("cell_width"), state_indices=arrays.get("state_index"),
        cj_probes=probes, source="saved_original_cell_CJ_probes" if probes is not None else "radial_profiles")
    report["provenance"] = meta.get("event_provenance", {"input_directory": str(in_dir.resolve())})
    if "cj_probe" not in arrays:
        report["notes"].append("CJ searched only in averaged radial profiles; short original-cell peaks may be lost.")
    write_event_outputs(out_dir or in_dir/"event_check", report, rows, probes)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing raw_line.npz")
    parser.add_argument("--out", type=Path, default=None, help="Default: INPUT/event_check/")
    parser.add_argument("--config", type=Path, default=None, help="Optional matching YAML configuration")
    args = parser.parse_args(argv)
    try:
        detect_directory(args.input, args.out, args.config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Event detection failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
