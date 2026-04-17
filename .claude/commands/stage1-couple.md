Run §1.5-1.6: contact-face Riemann matching and quasi-steady evolution to t_sep. Prints key separation-state values.

```bash
python -c "
import sys, math; sys.path.insert(0, '.')
import yaml
from physics import solve_cj, match_contact, evolve_quasisteady, find_separation
from physics.jwl_isentrope import JWLParams

with open('configs/tnt_spherical.yaml') as f:
    cfg = yaml.safe_load(f)

exp  = cfg['explosive']
air  = cfg['air']
prep = cfg['preprocessing']

params = JWLParams(
    A=exp['jwl']['A'], B=exp['jwl']['B'],
    R1=exp['jwl']['R1'], R2=exp['jwl']['R2'],
    omega=exp['jwl']['omega'], E0=exp['jwl']['E0'],
    rho0=exp['rho0'],
)
W   = exp['W']
R0  = (3*W/(4*math.pi*exp['rho0']))**(1/3)

cj      = solve_cj(params, exp['D_CJ'], bracket=tuple(prep['v_cj_bracket']))
contact = match_contact(cj, air['gamma'], air['rho_a'], air['P_a'])
series  = evolve_quasisteady(R0, cj, contact, air['gamma'], air['rho_a'], air['P_a'],
                              dt=prep['quasisteady_dt'], t_max=prep['quasisteady_t_max'])
sep     = find_separation(series, air['P_a'])

print(f'Contact face:  P_c* = {contact.P_c/1e6:.3f} MPa,  u_c* = {contact.u_c:.2f} m/s')
print(f'Separation:    t_sep = {sep.t_sep*1e6:.2f} μs')
print(f'               R_c^sep = {sep.R_c*100:.3f} cm')
print(f'               R_s^sep = {sep.R_s*100:.3f} cm')
print(f'               D_s^sep = {sep.D_s:.2f} m/s')
print(f'               v_c^sep = {sep.v_c:.4f}')
print(f'Time steps in series: {len(series.t)}')
"
```
