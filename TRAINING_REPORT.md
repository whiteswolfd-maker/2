# PINN 训练诊断报告

**配置**: configs/tnt_spherical_50mm.yaml  |  **装药**: R_0=50 mm, M_TNT=853.5 g

## 1. 解析分离参数

| 参数 | 值 | 来源 |
|------|-----|------|
| P_CJ | 21.83 GPa | CJ 切线求解 (Brent)
| rho_CJ | 2206.5 kg/m3 | CJ 求解
| u_CJ | 1810.7 m/s | CJ 求解
| c_CJ | 5119.3 m/s | D_CJ - u_CJ
| V_c | 11.0977 | Riemann 匹配 Brent: u_p(P)=u_a(P)
| P_x | 78.89 MPa | Riemann 匹配
| u_x | 7315.5 m/s | Riemann 匹配
| rho_x | 146.9 kg/m3 | rho_TNT / V_x
| R_c | 111.53 mm | R_0 * V_c^(1/3)
| t_sep | 10.545 us | Riemann不变量积分 (非 Taylor-Sadovsky)
| r_tail(t_sep) | 15.1 mm | C-特征线, r<15.1mm=CJ核心不进PINN

> JWL: A=371.2 GPa, B=3.7471 GPa, R1=4.15, R2=0.95, omega=0.30, E0=6.0 GJ/m3
> 空气: gamma=1.4, rho_a=1.225 kg/m3, P_a=101325 Pa

## 2. Gate 检查

DetNet @ (t_sep=10.55 us, R_c=111.5 mm) vs Sec 4.2.1 解析:

| | PINN 预测 | 解析值 | 误差 | 判定 |
|---|----------|--------|------|------|
| rho | 149.9 kg/m3 | 146.9 kg/m3 | 2.08% | OK |
| u | 7263.7 m/s | 7315.5 m/s | 0.71% | OK |
| P | 82.171 MPa | 78.889 MPa | 4.16% | OK |

**最大误差: 4.16% (阈值 5%)**

## 3. 耦合一致性

AirShockNet vs DetNet @ (t_sep, R_c):

| | DetNet | AirShock |
|---|--------|----------|
| rho | 149.9 kg/m3 | 0.0 kg/m3 |
| u | 7263.7 m/s | -9172.5 m/s |
| P | 82.171 MPa | 0.000 MPa |

**差异: 226.3% (目标 < 1%)**

> **严重断裂**: AirShockNet 在分离点跳到了空气侧值。
> 
> **根因**: IC_B 权重 5.0→2.0 降过头，被 PDE_B (权 1.0) + data_B (权 1.0) 在矩形域上的梯度淹没。
> **修复**: IC_B 权重 2→10, contact_repeat 16→128, PDE_B 1.0→0.2

## 4. 训练状态

| 阶段 | 步数 | LR | 最终 PDE | 目标 |
|------|------|-----|---------|------|
| A2 | 60000 | 5e-5 | 0.00468 | < 1e-4 |
| B2 | 60000 | 5e-5 | 0.223 | < 1e-4 |

Phase A: PDE_A 0.27→0.00468, connect_A~5e-6, data_A~26, IC_A~45
Phase B: PDE_B→0.223, IC_B~63, data_B~22, RH_B~11

## 5. 数据质量

| 指标 | 值 | 判断 |
|------|-----|------|
| d3plot 总帧数 | 496 (0-1000 us) | - |
| Phase A 帧数 (t<=10.5 us) | 6 | - |
| 早期帧 (t<8 us) | 5 | - |
| P=0 格 (t<8 us) | 175/12500 (1.4%) | **坏数据, t_data_min=8us 已排除** |
| 激波半峰宽 (@t=247 us) | 104 mm | **>50mm=ALE 抹平** |
| rho@激波前沿 | 1.28 kg/m3 | **接近环境 (1.225)=无密度跳变** |
| Henrych 抽查 (Z=1.31) | d3plot=0.72 MPa, 经验=0.04 MPa, 差 +1575% | **ALE 耗散致系统性偏低** |

> 早期 P=0: LS-DYNA *MAT_HIGH_EXPLOSIVE_BURN 未烧完的 ALE 格
> 激波抹平: ALE 人工粘性 + 网格太粗 (~10mm), 建议加密到 1-2mm 重算

## 6. 已知问题与修复方向

| # | 问题 | 严重度 | 修复 |
|---|------|--------|------|
| 1 | Gate P=4.16% > 5% | 中 | A2 多跑 40000 步, LR 降到 2e-5
| 2 | 耦合断裂 226% | **严重** | IC_B 2→10, contact_repeat 16→128, PDE_B 1.0→0.2
| 3 | PDE_A=0.00468 未到 1e-4 | 中 | 更小 LR + 更多步数
| 4 | PDE_B=0.223 未到 1e-4 | 中 | 同上
| 5 | ALE 激波抹平 104 mm | **严重** | 重算 LS-DYNA 加密网格到 1-2mm
| 6 | 早期 P=0 垃圾 | 已修复 | t_data_min=8 us
