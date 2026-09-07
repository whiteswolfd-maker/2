---
name: "source-command-validate"
description: "Migrated source command `validate`"
---

# source-command-validate

Use this skill when the user asks to run the migrated source command `validate`.

## Command Template

Physics sanity checks + post-training convergence diagnostics.

```bash
# Physics layer (no dataset / no trained model needed)
python -m validation.physics_check

# Convergence vs d3plot dataset (requires trained checkpoints)
python -m validation.convergence --config configs/tnt_spherical_50mm.yaml

# Full unit-test sweep
pytest tests/ -v
```
