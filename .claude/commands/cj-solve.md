Run the Chapman-Jouguet detonation solver for TNT using the default config and print the CJ state.

```bash
python -c "
import sys; sys.path.insert(0, '.')
import yaml, math
from physics import solve_cj
from physics.jwl_isentrope import JWLParams

with open('configs/tnt_spherical.yaml') as f:
    cfg = yaml.safe_load(f)

exp = cfg['explosive']
prep = cfg['preprocessing']
params = JWLParams(
    A=exp['jwl']['A'], B=exp['jwl']['B'],
    R1=exp['jwl']['R1'], R2=exp['jwl']['R2'],
    omega=exp['jwl']['omega'], E0=exp['jwl']['E0'],
    rho0=exp['rho0'],
)
cj = solve_cj(params, exp['D_CJ'], bracket=tuple(prep['v_cj_bracket']))
print(f'v_CJ  = {cj.v_CJ:.6f}')
print(f'P_CJ  = {cj.P_CJ/1e9:.3f} GPa')
print(f'rho_CJ= {cj.rho_CJ:.2f} kg/m^3')
print(f'u_CJ  = {cj.u_CJ:.2f} m/s')
print(f'c_CJ  = {cj.c_CJ:.2f} m/s')
"
```
