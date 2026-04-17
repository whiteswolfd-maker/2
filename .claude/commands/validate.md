Run physics sanity checks and post-training convergence diagnostics.

```bash
# Physics checks (no GPU, no trained model needed)
python -m validation.physics_check

# Convergence check vs reference (requires trained Stage-3 checkpoint)
python -m validation.convergence --config configs/tnt_spherical.yaml --stage 3

# Unit tests
pytest tests/ -v
```
