---
name: "source-command-dataset-check"
description: "Migrated source command `dataset-check`"
---

# source-command-dataset-check

Use this skill when the user asks to run the migrated source command `dataset-check`.

## Command Template

Print structural and physical sanity diagnostics for the extracted d3plot dataset.

```bash
python -m data.d3plot_dataset --check extracted/
```

Reports:
* (t, x) shape + ranges and (rho, u, P) min/max
* R_c / R_s monotonicity, P̄_p decay
* Polytropic exponent regression: log P̄_p ~ slope · log R_c.  Slope close to −γ_eff (=−3) supports the Landau-Stanyukovich prior used in `PolytropicLoss`.
