"""Generate TRAINING_REPORT.md — post-training diagnostic summary."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physics.cj_state import TNTParams, compute_cj_state, compute_separation_state
from data.d3plot_dataset import D3plotLineDataset
from pinn.trainer import _load_cfg
from pinn.networks import build_networks
from pinn.checkpoints import load_checkpoint
from pinn.coupling import air_contact_density


def henrych_overpressure_MPa(Z):
    """Henrych (1979) peak incident overpressure in MPa.

    Original coefficients in bar; 1 bar = 0.1 MPa, so multiply by 0.1.
    Z = R / M_TNT^(1/3)  [m / kg^(1/3)].
    """
    if Z < 0.05 or Z > 10:
        return None
    if Z <= 0.3:
        return 0.14072/Z + 0.05540/Z**2 - 0.003572/Z**3 + 0.0000625/Z**4
    if Z <= 1.0:
        return 0.061938/Z - 0.003262/Z**2 + 0.021324/Z**3
    return 0.00662/Z + 0.0405/Z**2 + 0.03288/Z**3


# ---- data & analytical state ----
tnt = TNTParams()
cj_b = compute_cj_state(tnt)
sep = compute_separation_state(tnt, R_0=0.05, cj_bundle=cj_b)
ds = D3plotLineDataset("extracted/", device="cpu")
cfg = _load_cfg("configs/tnt_spherical_50mm.yaml")
air = cfg["air"]
Pa = air["P_a"]

# ---- gate ----
det, asn = build_networks(
    cfg["networks"],
    rho_ref_A=cj_b.rho_CJ, u_ref_A=cfg["domain"]["u_ref_A"],
    P_ref_A=cj_b.P_CJ,
    rho_ref_B=air["rho_a"], u_ref_B=cfg["domain"]["u_ref_B"],
    P_ref_B=cfg["domain"]["P_ref_B"],
)
det, _ = load_checkpoint("checkpoints/detonation.pt", det)

# Fixed-endpoint data agreement is diagnostic, not learned convergence.
with torch.no_grad():
    t_p = torch.tensor([[sep.t_sep]])
    r_p = torch.tensor([[sep.R_c]])
    rho_p, u_p, P_p = det(t_p, r_p)
    rho_d, u_d, P_d = ds.state_at(t_p, r_p)
re = abs(float(rho_p) - float(rho_d)) / max(abs(float(rho_d)), 1e-10) * 100
ue = abs(float(u_p) - float(u_d)) / max(abs(float(u_d)), 1) * 100
pe = abs(float(P_p) - float(P_d)) / max(abs(float(P_d)), 1e-10) * 100

# ---- coupling ----
cc = 0.0
rho_b = u_b = P_b = None
if Path("checkpoints/air_shock.pt").exists():
    asn, _ = load_checkpoint("checkpoints/air_shock.pt", asn)
    with torch.no_grad():
        rho_b, u_b, P_b = asn(
            torch.tensor([[sep.t_sep]]), torch.tensor([[sep.R_c]]))
    cc = max(
        abs(float(rho_b) - float(air_contact_density(P_p, gamma=air["gamma"],
            rho_a=air["rho_a"], P_a=air["P_a"]))) / max(float(rho_b), 1e-10),
        abs(float(u_b) - float(u_p)) / max(abs(float(u_p)), 1),
        abs(float(P_b) - float(P_p)) / max(float(P_p), 1e-10),
    )

# ---- shock smear ----
t_arr = ds.t_arr.numpy()
x = ds.x_arr.numpy()
mid = len(t_arr) // 4
Ps = ds.P_grid[mid].numpy()
above = np.where(Ps > 1.05 * Pa)[0]
Rs = x[above[-1]]
Pmax = np.max(Ps[max(0, above[-1] - 400):above[-1] + 1])
half = (Pmax + Pa) / 2
ih = above[-1]
for j in range(above[-1], max(0, above[-1] - 400), -1):
    if Ps[j] > half:
        ih = j
        break
smear = (Rs - x[ih]) * 1e3
rho_Rs = ds.rho_grid[mid, above[-1]].item()

# ---- early frames ----
n_early = int((t_arr < 8e-6).sum())
Pg = ds.P_grid.numpy()
n_zero = int((Pg[t_arr < 8e-6] < 1e-6).sum())
total_early = int(Pg[t_arr < 8e-6].size)

# ---- Henrych spot ----
R0 = 0.05
W = 4.0 / 3.0 * np.pi * R0**3 * 1630.0
W3 = W ** (1.0 / 3.0)
k = len(t_arr) // 4
tv = float(t_arr[k])
Rsv = float(np.interp(tv, ds.t_traj.numpy(), ds.R_s_traj.numpy()))
Z = Rsv / W3
dPh = henrych_overpressure_MPa(Z)
dPd3 = h_err = 0.0
if dPh:
    rs = np.linspace(max(0, Rsv - 0.3), Rsv - 1e-4, 50)
    tq = torch.full((50, 1), tv)
    rq = torch.tensor(rs.reshape(-1, 1))
    rhos, _, Ps2 = ds.state_at(tq, rq)
    air_m = rhos.numpy().reshape(-1) < 100
    Ppk = float(Ps2.numpy().reshape(-1)[air_m].max()) if air_m.any() else float(Ps2.max())
    dPd3 = (Ppk - Pa) / 1e6
    h_err = (dPd3 - dPh) / dPh * 100

# ---- build report ----
L = []
def a(s=""):
    L.append(s)

a("# PINN 训练诊断报告")
a()
a("**配置**: configs/tnt_spherical_50mm.yaml  |  "
  "**装药**: R_0=50 mm, M_TNT=%.1f g" % (W * 1e3))
a()

# 1. Analytical
a("## 1. 解析分离参数")
a()
a("| 参数 | 值 | 来源 |")
a("|------|-----|------|")
a("| P_CJ | %.2f GPa | CJ 切线求解 (Brent)" % (cj_b.P_CJ / 1e9))
a("| rho_CJ | %.1f kg/m3 | CJ 求解" % cj_b.rho_CJ)
a("| u_CJ | %.1f m/s | CJ 求解" % cj_b.u_CJ)
a("| c_CJ | %.1f m/s | D_CJ - u_CJ" % cj_b.c_CJ)
a("| V_c | %.4f | Riemann 匹配 Brent: u_p(P)=u_a(P)" % sep.V_x)
a("| P_x | %.2f MPa | Riemann 匹配" % (sep.P_x / 1e6))
a("| u_x | %.1f m/s | Riemann 匹配" % sep.u_x)
a("| rho_x | %.1f kg/m3 | rho_TNT / V_x" % sep.rho_x)
a("| R_c | %.2f mm | R_0 * V_c^(1/3)" % (sep.R_c * 1e3))
a("| t_sep | %.3f us | Riemann不变量积分 (非 Taylor-Sadovsky)" % (sep.t_sep * 1e6))
a("| r_tail(t_sep) | %.1f mm | C-特征线, r<%.1fmm=CJ核心不进PINN"
  % (sep.r_traj_tail[-1] * 1e3, sep.r_traj_tail[-1] * 1e3))
a()
a("> JWL: A=371.2 GPa, B=3.7471 GPa, R1=4.15, R2=0.95, omega=0.30, E0=6.0 GJ/m3")
a("> 空气: gamma=1.4, rho_a=1.225 kg/m3, P_a=101325 Pa")
a()

# 2. Gate
a("## 2. 固定端点与数据的差异")
a("端点值由解析公式固定；下面的误差不是网络学会端点的收敛证据。")
a()
a("DetNet @ (t_sep=%.2f us, R_c=%.1f mm) vs d3plot 数据:" % (sep.t_sep * 1e6, sep.R_c * 1e3))
a()
a("| | PINN 预测 | d3plot 数据 | 误差 | 判定 |")
a("|---|----------|-----------|------|------|")
a("| rho | %.1f kg/m3 | %.1f kg/m3 | %.2f%% | %s | (诊断)"
  % (float(rho_p), float(rho_d), re, "OK" if re <= 5 else "X"))
a("| u | %.1f m/s | %.1f m/s | %.2f%% | %s |"
  % (float(u_p), float(u_d), ue, "OK" if ue <= 5 else "X"))
a("| P | %.3f MPa | %.3f MPa | %.2f%% | %s |"
  % (float(P_p) / 1e6, float(P_d) / 1e6, pe, "OK" if pe <= 5 else "X"))
a()
a("**最大误差: %.2f%% (阈值 5%%)**" % max(re, ue, pe))
a()

# 3. Coupling
a("## 3. 耦合一致性")
a()
if rho_b is not None:
    a("A、B 在出流连接点的压力、速度连续；密度分别为产物侧与空气侧。")
    a()
    a("| | DetNet | AirShock |")
    a("|---|--------|----------|")
    a("| rho | %.1f kg/m3 | %.1f kg/m3 |" % (float(rho_p), float(rho_b)))
    a("| u | %.1f m/s | %.1f m/s |" % (float(u_p), float(u_b)))
    a("| P | %.3f MPa | %.3f MPa |" % (float(P_p) / 1e6, float(P_b) / 1e6))
    a()
    a("**压力/速度连续性及空气密度 RH 一致性最大误差: %.1f%% (目标 < 1%%)**" % (cc * 100))
    if cc > 0.5:
        a("> 连接条件未满足；检查训练版本、检查点约束和空气侧 RH 状态。")

else:
    a("(AirShockNet checkpoint 不存在)")
a()

# 4. Training
a("## 4. 训练状态")
a()
a("| 阶段 | 步数 | LR | 最终 PDE | 目标 |")
a("|------|------|-----|---------|------|")
a("| A2 | 60000 | 5e-5 | 0.00468 | < 1e-4 |")
a("| B2 | 60000 | 5e-5 | 0.223 | < 1e-4 |")
a()
a("Phase A: PDE_A 0.27→0.00468, connect_A~5e-6, data_A~26, IC_A~45")
a("Phase B: PDE_B→0.223, IC_B~63, data_B~22, RH_B~11")
a()

# 5. Data quality
a("## 5. 数据质量")
a()
a("| 指标 | 值 | 判断 |")
a("|------|-----|------|")
a("| d3plot 总帧数 | %d (0-%.0f us) | - |" % (len(t_arr), t_arr[-1] * 1e6))
a("| Phase A 帧数 (t<=%.1f us) | %d | - |"
  % (sep.t_sep * 1e6, int((t_arr <= sep.t_sep).sum())))
a("| 早期帧 (t<8 us) | %d | - |" % n_early)
a("| P=0 格 (t<8 us) | %d/%d (%.1f%%) | **坏数据, t_data_min=8us 已排除** |"
  % (n_zero, total_early, 100 * n_zero / max(total_early, 1)))
a("| 激波半峰宽 (@t=%d us) | %.0f mm | %s |"
  % (int(t_arr[mid] * 1e6), smear,
     "**>50mm=ALE 抹平**" if smear > 50 else "OK"))
a("| rho@激波前沿 | %.2f kg/m3 | **接近环境 (1.225)=无密度跳变** |" % rho_Rs)
if dPh:
    a("| Henrych 抽查 (Z=%.2f) | d3plot=%.2f MPa, 经验=%.2f MPa, 差 %+.0f%% | **ALE 耗散致系统性偏低** |"
      % (Z, dPd3, dPh, h_err))
a()
a("> 早期 P=0: LS-DYNA *MAT_HIGH_EXPLOSIVE_BURN 未烧完的 ALE 格")
a("> 激波抹平: ALE 人工粘性 + 网格太粗 (~10mm), 建议加密到 1-2mm 重算")
a()

# 6. Issues
a("## 6. 已知问题与修复方向")
a()
a("| # | 问题 | 严重度 | 修复 |")
a("|---|------|--------|------|")
a("| 1 | Gate P=%.2f%% > 5%% | 中 | A2 多跑 40000 步, LR 降到 2e-5" % pe)
a("| 2 | 耦合断裂 %.0f%% | **严重** | IC_B 2→10, contact_repeat 16→128, PDE_B 1.0→0.2" % (cc * 100))
a("| 3 | PDE_A=0.00468 未到 1e-4 | 中 | 更小 LR + 更多步数")
a("| 4 | PDE_B=0.223 未到 1e-4 | 中 | 同上")
a("| 5 | ALE 激波抹平 %.0f mm | **严重** | 重算 LS-DYNA 加密网格到 1-2mm" % smear)
a("| 6 | 早期 P=0 垃圾 | 已修复 | t_data_min=8 us")
a()

Path("TRAINING_REPORT.md").write_text("\n".join(L), encoding="utf-8")
print("Done: TRAINING_REPORT.md  (%d lines)" % len(L))
