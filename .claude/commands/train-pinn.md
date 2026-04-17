Run the full three-stage PINN training pipeline (Stage 1 → 2 → 3). Checkpoints are saved to the `checkpoints/` directory automatically.

```bash
python -m pinn.trainer --all --config configs/tnt_spherical.yaml
```

To run individual stages:

```bash
# Stage 1 only (IC warm-up, 5 000 steps)
python -m pinn.trainer --stage 1 --config configs/tnt_spherical.yaml

# Stage 2 only (full losses, 50 000 steps; loads stage1 checkpoint if present)
python -m pinn.trainer --stage 2 --config configs/tnt_spherical.yaml

# Stage 3 only (L-BFGS fine-tune, 8 000 steps; loads stage2 checkpoint if present)
python -m pinn.trainer --stage 3 --config configs/tnt_spherical.yaml
```
