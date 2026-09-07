---
name: "source-command-train-pinn"
description: "Migrated source command `train-pinn`"
---

# source-command-train-pinn

Use this skill when the user asks to run the migrated source command `train-pinn`.

## Command Template

Run the spherical two-network PINN training pipeline.

Prerequisite: ``extracted/`` populated by ``/extract-d3plot`` (or
``python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml``).

```bash
# Both phases sequentially: A1 -> A2 -> Sec.4.2.1 gate -> freeze -> B1 -> B2
python -m pinn.trainer --all --config configs/tnt_spherical_50mm.yaml

# Phase A only (DetonationNet)
python -m pinn.trainer --net detonation --config configs/tnt_spherical_50mm.yaml

# Phase B only (AirShockNet -- needs the Phase-A checkpoint already saved)
python -m pinn.trainer --net airshock --config configs/tnt_spherical_50mm.yaml
```

Checkpoints saved to ``checkpoints/{detonation,air_shock}.pt``.  Phase B
auto-loads the Phase-A checkpoint when run alone.
