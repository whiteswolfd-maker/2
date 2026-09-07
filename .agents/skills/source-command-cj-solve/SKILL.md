---
name: "source-command-cj-solve"
description: "Migrated source command `cj-solve`"
---

# source-command-cj-solve

Use this skill when the user asks to run the migrated source command `cj-solve`.

## Command Template

Run the Chapman-Jouguet detonation solver for TNT and print the CJ state.

```bash
python -c "
import sys; sys.path.insert(0, '.')
from physics import compute_cj_state, TNTParams

cj = compute_cj_state(TNTParams())
print(f'v_CJ   = {cj.cj.v_CJ:.6f}')
print(f'P_CJ   = {cj.P_CJ/1e9:.3f} GPa  (LS-DYNA card target 21.0 GPa)')
print(f'rho_CJ = {cj.rho_CJ:.2f} kg/m^3')
print(f'u_CJ   = {cj.u_CJ:.2f} m/s')
print(f'c_CJ   = {cj.c_CJ:.2f} m/s')
"
```

Defaults match the LS-DYNA *EOS_JWL card used in this project's simulation:
A = 371.2 GPa, B = 3.7471 GPa, R1 = 4.15, R2 = 0.95, omega = 0.30,
E0 = 6.0 GJ/m^3, rho_TNT = 1630 kg/m^3, D_CJ = 6930 m/s.
