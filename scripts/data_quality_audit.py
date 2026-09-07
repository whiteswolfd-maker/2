#!/usr/bin/env python3
"""
Data quality audit across 3 TNT charge radii (50mm, 80mm, 100mm).

Checks:
  1. Peak pressure at equal scaled distance Z = r / M^(1/3)
  2. Shock arrival time scaled by M^(1/3)
  3. Mesh resolution (dx relative to R0, Rc)
  4. Contact surface in data vs physics
  5. Pressure at physics R_c at t_sep
"""

import numpy as np
import sys, os

# --- Physical constants ---
P_atm = 101_325.0   # Pa
RHO_CONTACT_THEORY = 146.9  # kg/m^3 from §4.2.1

# --- Data for each radius ---
radii = {
    "50mm": {
        "path": "c:/Users/work/Documents/GitHub/2/extracted/raw_line.npz",
        "meta_path": "c:/Users/work/Documents/GitHub/2/extracted/metadata.json",
        "R0_mm": 50.0,
        "M_TNT_kg": 0.853,
        "R_c_physics_mm": 50.0 * 2.2305,  # = 111.525
        "t_sep_us": 10.55e-6,  # s
        "t_sep_tag": "10.55 us",
    },
    "80mm": {
        "path": "c:/Users/work/Documents/GitHub/2/extracted_80mm/raw_line.npz",
        "meta_path": "c:/Users/work/Documents/GitHub/2/extracted_80mm/metadata.json",
        "R0_mm": 80.0,
        "M_TNT_kg": 3.496,
        "R_c_physics_mm": 80.0 * 2.2305,  # = 178.44
        "t_sep_us": 16.87e-6,
        "t_sep_tag": "16.87 us",
    },
    "100mm": {
        "path": "c:/Users/work/Documents/GitHub/2/extracted_100mm/raw_line.npz",
        "meta_path": "c:/Users/work/Documents/GitHub/2/extracted_100mm/metadata.json",
        "R0_mm": 100.0,
        "M_TNT_kg": 6.828,
        "R_c_physics_mm": 100.0 * 2.2305,  # = 223.05
        "t_sep_us": 21.09e-6,
        "t_sep_tag": "21.09 us",
    },
}

# Load data
data = {}
for name, info in radii.items():
    d = np.load(info["path"])
    data[name] = {
        "t": d["t"],        # (N_t,)
        "x": d["x"],        # (N_x,)
        "rho": d["rho"],    # (N_t, N_x)
        "u": d["u"],        # (N_t, N_x)
        "P": d["P"],        # (N_t, N_x)
    }
    print(f"\n=== Loaded {name}: {info['path']} ===")
    print(f"  t: {d['t'].shape}, range [{d['t'].min():.3e}, {d['t'].max():.3e}]")
    print(f"  x: {d['x'].shape}, range [{d['x'].min():.5f}, {d['x'].max():.5f}]")
    print(f"  rho: {d['rho'].shape}, range [{d['rho'].min():.4f}, {d['rho'].max():.4f}]")
    print(f"  u:   {d['u'].shape}, range [{d['u'].min():.4f}, {d['u'].max():.4f}]")
    print(f"  P:   {d['P'].shape}, range [{d['P'].min():.4e}, {d['P'].max():.4e}]")

# Compute derived quantities
for name, info in radii.items():
    info["M13"] = info["M_TNT_kg"] ** (1.0 / 3.0)
    d = data[name]
    info["N_t"] = len(d["t"])
    info["N_x"] = len(d["x"])
    info["x_end"] = d["x"][-1]
    info["dx"] = info["x_end"] / info["N_x"]
    info["R_c_physics_mm"] = info["R0_mm"] * 2.2305
    print(f"\n{name}: M^(1/3) = {info['M13']:.4f}, dx = {info['dx']:.5f} m, "
          f"x_end = {info['x_end']:.4f} m, N_x = {info['N_x']}")

# ============================================================
# 1. PEAK PRESSURE AT EQUAL Z
# ============================================================
print("\n" + "=" * 80)
print("1. PEAK PRESSURE AT EQUAL SCALED DISTANCE Z = r / M^(1/3)")
print("=" * 80)

Z_values = [0.15, 0.2, 0.3, 0.5, 1.0]

peak_table = {}
for Z in Z_values:
    print(f"\n--- Z = {Z} m/kg^(1/3) ---")
    peak_table[Z] = {}
    for name, info in radii.items():
        r_target = Z * info["M13"]  # metres
        d = data[name]
        x = d["x"]
        # find nearest grid index
        idx = np.argmin(np.abs(x - r_target))
        r_actual = x[idx]
        # Peak pressure over all time at this spatial index
        P_over_t = d["P"][:, idx]
        peak_P = np.max(P_over_t)
        peak_P_MPa = peak_P / 1e6
        peak_table[Z][name] = peak_P
        print(f"  {name}: r_target={r_target:.5f} m, r_actual={r_actual:.5f} m, "
              f"peak P = {peak_P_MPa:.4f} MPa")

# Ratio table
print("\nPeak pressure ratios (relative to 50mm):")
for Z in Z_values:
    ref = peak_table[Z]["50mm"]
    r80 = peak_table[Z]["80mm"] / ref
    r100 = peak_table[Z]["100mm"] / ref
    print(f"  Z={Z}: P80/P50={r80:.4f}, P100/P50={r100:.4f}")

# ============================================================
# 2. SHOCK ARRIVAL TIME AT EQUAL Z
# ============================================================
print("\n" + "=" * 80)
print("2. SHOCK ARRIVAL TIME AT EQUAL Z (scaled by M^(1/3))")
print("=" * 80)

arrival_table = {}
for Z in Z_values:
    print(f"\n--- Z = {Z} m/kg^(1/3) ---")
    arrival_table[Z] = {}
    for name, info in radii.items():
        r_target = Z * info["M13"]
        d = data[name]
        x = d["x"]
        t = d["t"]
        idx = np.argmin(np.abs(x - r_target))
        P_over_t = d["P"][:, idx]
        # Find first time P exceeds 1.05 * P_atm
        mask = P_over_t > 1.05 * P_atm
        if np.any(mask):
            t_arrival = t[np.argmax(mask)]  # first True index
            t_arrival_scaled = t_arrival / info["M13"]
        else:
            t_arrival = np.nan
            t_arrival_scaled = np.nan
        arrival_table[Z][name] = {
            "t_arrival": t_arrival,
            "t_arrival_scaled": t_arrival_scaled,
        }
        if np.isnan(t_arrival):
            print(f"  {name}: shock NOT YET ARRIVED at r={r_target:.5f} m")
        else:
            print(f"  {name}: t_arrival={t_arrival*1e6:.4f} us, "
                  f"t_arrival/M^(1/3)={t_arrival_scaled*1e6:.4f} us/kg^(1/3)")

print("\nScaled arrival time invariance:")
for Z in Z_values:
    vals = [arrival_table[Z][n]["t_arrival_scaled"] * 1e6 for n in radii]
    print(f"  Z={Z}: scaled arrival = {[f'{v:.4f}' if not np.isnan(v) else 'NaN' for v in vals]} us/kg^(1/3)")

# ============================================================
# 3. MESH RESOLUTION
# ============================================================
print("\n" + "=" * 80)
print("3. MESH RESOLUTION")
print("=" * 80)

for name, info in radii.items():
    print(f"\n{name}:")
    print(f"  N_x = {info['N_x']}")
    print(f"  x_end = {info['x_end']:.4f} m")
    print(f"  dx = {info['dx']:.5f} m = {info['dx']*1000:.3f} mm")
    print(f"  R0 = {info['R0_mm']} mm = {info['R0_mm']/1000:.4f} m")
    print(f"  R_c_physics = {info['R_c_physics_mm']:.1f} mm = {info['R_c_physics_mm']/1000:.4f} m")
    print(f"  dx / R0 = {info['dx']/(info['R0_mm']/1000):.4f}")
    print(f"  dx / R_c = {info['dx']/(info['R_c_physics_mm']/1000):.4f}")
    print(f"  Cells across R0: {info['R0_mm']/1000 / info['dx']:.1f}")
    print(f"  Cells across R_c: {info['R_c_physics_mm']/1000 / info['dx']:.1f}")

# ============================================================
# 4. CONTACT SURFACE IN DATA
# ============================================================
print("\n" + "=" * 80)
print("4. CONTACT SURFACE IN DATA vs PHYSICS")
print("=" * 80)

for name, info in radii.items():
    d = data[name]
    t = d["t"]
    x = d["x"]
    rho = d["rho"]
    P_data = d["P"]

    # Find closest time frame to t_sep
    t_sep = info["t_sep_us"]
    i_t = np.argmin(np.abs(t - t_sep))
    t_closest = t[i_t]

    print(f"\n{name}:")
    print(f"  Physics t_sep = {t_sep*1e6:.3f} us, closest frame t = {t_closest*1e6:.6f} s = {t_closest*1e6:.3f} us")
    print(f"  Physics R_c = {info['R_c_physics_mm']:.2f} mm")

    # Find outermost r where density >= RHO_CONTACT_THEORY
    rho_row = rho[i_t, :]
    mask_contact = rho_row >= RHO_CONTACT_THEORY
    if np.any(mask_contact):
        idx_contact = np.max(np.where(mask_contact)[0])
        r_contact_data = x[idx_contact]
        rho_at = rho_row[idx_contact]
        print(f"  Data contact r: outermost r where rho >= {RHO_CONTACT_THEORY} kg/m^3 = {r_contact_data*1000:.3f} mm")
        print(f"  rho at that point: {rho_at:.3f} kg/m^3")
        print(f"  Ratio data/physics = {r_contact_data*1000 / info['R_c_physics_mm']:.4f}")
    else:
        print(f"  WARNING: No cell with rho >= {RHO_CONTACT_THEORY} kg/m^3 at t={t_closest*1e6:.3f} us!")
        # Find max density
        max_rho = np.max(rho_row)
        idx_max_rho = np.argmax(rho_row)
        print(f"  Max density = {max_rho:.3f} kg/m^3 at r = {x[idx_max_rho]*1000:.3f} mm")

    # Also: at physics R_c, what is density?
    r_phys = info["R_c_physics_mm"] / 1000.0  # metres
    idx_phys = np.argmin(np.abs(x - r_phys))
    r_phys_actual = x[idx_phys]
    rho_phys = rho_row[idx_phys]
    print(f"  At physics R_c ({r_phys*1000:.2f} mm), rho = {rho_phys:.3f} kg/m^3")

# ============================================================
# 5. PRESSURE AT PHYSICS R_c AT t_sep
# ============================================================
print("\n" + "=" * 80)
print("5. PRESSURE AT PHYSICS R_c AT t = t_sep")
print("=" * 80)

print("\nPhysics expectation: at contact surface, P_x ~ 0.079 GPa = 79 MPa (from JWL Riemann match)\n")

for name, info in radii.items():
    d = data[name]
    t = d["t"]
    x = d["x"]
    P_d = d["P"]

    t_sep = info["t_sep_us"]
    i_t = np.argmin(np.abs(t - t_sep))
    t_closest = t[i_t]

    r_phys = info["R_c_physics_mm"] / 1000.0
    idx_phys = np.argmin(np.abs(x - r_phys))
    r_actual = x[idx_phys]
    P_at_Rc = P_d[i_t, idx_phys]
    P_at_Rc_MPa = P_at_Rc / 1e6

    print(f"{name}:")
    print(f"  t_closest = {t_closest*1e6:.3f} us")
    print(f"  R_c_physics = {r_phys*1000:.2f} mm, grid r = {r_actual*1000:.3f} mm")
    print(f"  d3plot P at (t_sep, R_c_physics) = {P_at_Rc_MPa:.4f} MPa")

    # Also find peak P in the contact region (r within +/-5mm of R_c)
    r_tol = 0.005  # 5 mm
    mask_region = (x >= r_phys - r_tol) & (x <= r_phys + r_tol)
    if np.any(mask_region):
        P_region = P_d[i_t, :][mask_region]
        print(f"  P range near R_c (+/-5mm): [{P_region.min()/1e6:.4f}, {P_region.max()/1e6:.4f}] MPa")

    # Ratio vs physics
    P_x_theory = 79.0  # MPa
    print(f"  Ratio d3plot/theory = {P_at_Rc_MPa / P_x_theory:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
