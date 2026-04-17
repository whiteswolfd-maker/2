Build the PINN initial-condition profile at t_sep using the AnalyticalLoader. Runs the Sedov cross-check and prints deviation.

```bash
python -c "
import sys, math; sys.path.insert(0, '.')
import yaml
from physics import solve_cj, match_contact, evolve_quasisteady, find_separation
from physics.jwl_isentrope import JWLParams
from data.lsdyna_loader import AnalyticalLoader
from data.ic_builder import ICBuilder

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
W  = exp['W']
R0 = (3*W/(4*math.pi*exp['rho0']))**(1/3)

cj      = solve_cj(params, exp['D_CJ'], bracket=tuple(prep['v_cj_bracket']))
contact = match_contact(cj, air['gamma'], air['rho_a'], air['P_a'])
series  = evolve_quasisteady(R0, cj, contact, air['gamma'], air['rho_a'], air['P_a'],
                              dt=prep['quasisteady_dt'], t_max=prep['quasisteady_t_max'])
sep     = find_separation(series, air['P_a'])

loader  = AnalyticalLoader(sep, gamma=air['gamma'], rho_a=air['rho_a'], P_a=air['P_a'])
builder = ICBuilder(loader)

snap    = builder.build()
check   = builder.sedov_check(rho_a=air['rho_a'])

print(f'IC snapshot: {len(snap[\"r\"])} points over [{snap[\"R_c\"]*100:.2f}, {snap[\"R_s\"]*100:.2f}] cm')
print(f'rho range:   [{snap[\"rho\"].min():.3f}, {snap[\"rho\"].max():.3f}] kg/m^3')
print(f'P range:     [{snap[\"P\"].min()/1e3:.2f}, {snap[\"P\"].max()/1e3:.2f}] kPa')
print(f'u range:     [{snap[\"u\"].min():.1f}, {snap[\"u\"].max():.1f}] m/s')
print(f'Sedov check: E_eff={check[\"E_eff\"]/1e6:.3f} MJ, deviation={check[\"deviation_pct\"]:.2f}%')
"
```
