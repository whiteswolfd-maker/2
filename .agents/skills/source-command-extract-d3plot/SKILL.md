---
name: "source-command-extract-d3plot"
description: "Migrated source command `extract-d3plot`"
---

# source-command-extract-d3plot

Use this skill when the user asks to run the migrated source command `extract-d3plot`.

## Command Template

Extract a 1D state along the +x axis from LS-DYNA d3plot binaries (in `sim_data/3dTNT1/`) into `extracted/`.  Run with `--debug` first to identify the right history-variable slot for density on your specific *DATABASE_EXTENT_BINARY card.

```bash
# 1) Inspect d3plot structure (lists arrays + history-variable slot stats)
python -m data.extract_d3plot --d3plot sim_data/3dTNT1/ --debug

# 2) Full extraction (replace --rho-hv-slot N with the slot identified above)
python -m data.extract_d3plot \
    --d3plot sim_data/3dTNT1/ \
    --out extracted/ \
    --rho-hv-slot 0 \
    --L0 0.05 \
    --x-end 0.5 \
    --n-x 500 \
    --y-tol 5e-3 \
    --z-tol 5e-3 \
    --state-stride 1

# 3) Sanity-check the extracted dataset (trajectory monotonicity, γ_eff slope)
python -m data.d3plot_dataset --check extracted/
```
