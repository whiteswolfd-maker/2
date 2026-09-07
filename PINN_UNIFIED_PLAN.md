# PINN 球形 TNT 爆炸统一方案（修订版）

## 当前状态

- 三个装药半径（50mm/80mm/100mm）各有一个 DetNet + 一个 AirShockNet
- 每个半径独立训练，不能预测未训练过的半径
- 已完成改动：密度过滤（rho_min_A=100）、删除 connect_A、权重调整（data_A 5.0/2.0, BC_A_slope 0.001）
- 50mm Phase A 达标（PDE_A=7.8e-6, data_A=0.86）
- 80/100mm Phase A 未达标（PDE_A=3.8e-4/4.9e-4），根因是独立训练时数据稀疏且 lr 固定

## 目标

一个 DetNet + 一个 AirShockNet，训练一次，预测任意装药半径 R₀ 的球形 TNT 爆炸流场。

---

## 一、核心原理

### 1.1 Hopkinson-Cranz 缩放

球形 TNT 爆炸在无量纲坐标下自相似。定义：

```
M^(1/3) = (4/3 × π × ρ_TNT)^(1/3) × R₀ = 18.97 × R₀    [kg^(1/3)]

τ = t / M^(1/3)       [s/kg^(1/3)，即 μs/kg^(1/3)]
Z = r / M^(1/3)       [m/kg^(1/3)]
```

验证（实际跑 physics 模块）：

```
        M^(1/3)    τ_sep(μs/kg^1/3)   Z_c(m/kg^1/3)   Z_R0(m/kg^1/3)
50mm:   0.9486     11.12              0.117576         0.052712
80mm:   1.5177     11.12              0.117576         0.052712
100mm:  1.8971     11.12              0.117576         0.052712
20mm:   0.3794     11.12              0.117576         0.052712
200mm:  3.7942     11.12              0.117576         0.052712
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
std:               <0.000001           <0.0000000001
```

τ_sep 和 Z_c 对任意 R₀ 严格恒定。Taylor-Sadovsky ODE 保证 `t_sep ∝ R₀`，`M^(1/3) ∝ R₀`，比值精确抵消。

### 1.2 缩放后 PDE 形式不变

```
(τ, Z) = (t/α, r/α)，其中 α = M^(1/3)

∂/∂t = (1/α) ∂/∂τ      ∂/∂r = (1/α) ∂/∂Z

球形 Euler 质量方程:
  ∂ρ/∂t + (1/r²)∂(r²ρu)/∂r = 0
  → (1/α)∂ρ/∂τ + (1/α²Z²)(1/α)∂(α²Z²ρu)/∂Z = 0
  → ∂ρ/∂τ + (1/Z²)∂(Z²ρu)/∂Z = 0

α 全部消掉，方程形式完全相同。
```

**结论：`spherical_euler_residuals` 等 PDE 算子不需要任何修改**。输入从 (t, r) 换成 (τ, Z)，autograd 自动对 τ 和 Z 求导，物理等价。

### 1.3 硬约束点不变

```
CJ 锚点:    (τ=0,     Z=0.0527)  → 目标 (ρ_CJ, 0, P_CJ)     所有 R₀ 同一点
接触面锚点: (τ=11.12, Z=0.1176) → 目标 (ρ_x, u_x, P_x)   所有 R₀ 同一点

(ρ_x, u_x, P_x) 只依赖 JWL 参数，与 R₀ 无关，验证:
  P_x = 0.079 GPa,  u_x = 7316 m/s,  ρ_x = 147 kg/m³  三个半径全相同
```

---

## 二、统一域定义

### 2.1 Phase A — 爆轰产物域

```
τ ∈ [0, 11.12×10⁻⁶ s/kg^(1/3)]    即 [0, ~11 μs/kg^(1/3)]
Z ∈ [Z_tail(τ), Z_c(τ)]

Z_c(τ):   Taylor-Sadovsky ODE，从 Z=0.0527(τ=0) 到 Z=0.1176(τ=11.12)
Z_tail(τ): CJ 稀疏波尾，从 Z=0.0527(τ=0) 渐缩到 Z=0.0159(τ=11.12)
```

三个半径的 fan 在此坐标下**完全重叠**（验证通过，Z 范围完全相同）。

### 2.2 Phase B — 空气激波域

```
τ ∈ [11.12×10⁻⁶, τ_max]             起始点统一
Z ∈ [0.1176, Z_max]

τ_max: 激波走到 Z=15 的时刻，约 870×10⁻⁶ s/kg^(1/3)
Z_max = 15 m/kg^(1/3)                物理边界，对应 ΔP ≈ 0.01 MPa (<1% 大气压)
```

三个半径在 Z ∈ [0.1176, 1.3178] 范围内**均有数据覆盖**（验证通过）。

### 2.3 Z_max = 15 的依据

```
Henrych (1979) 经验公式:
  Z = 15 → ΔP ≈ 0.009 MPa  (< 1% 大气压，基本回到环境状态)
  Z = 20 → ΔP ≈ 0.005 MPa

选 Z_max=15 即可，更大值无工程意义。
```

### 2.4 数据覆盖分区域图

```
Z=0.118     Z=1.32     Z=1.65     Z=2.64        Z=15
├───────────┼──────────┼──────────┼─────────────┤
│ 三者重叠   │ 50+80mm  │ 只50mm   │ 无数据      │
│ 数据密集   │          │          │ 纯PDE外推   │
│            │          │          │ (弱激波)    │
├───────────┼──────────┼──────────┼─────────────┤
│ data+PDE  │ data+PDE │ PDE+RH   │ 纯 PDE      │
│ +RH       │ +RH      │ (50mm激波│             │
│ (50mm激波)│          │ 轨迹)    │             │
└───────────┴──────────┴──────────┴─────────────┘

激波轨迹: 统一使用 50mm 的 Z_s(τ)
          (80mm和100mm的激波在 τ>137/117 后跑出 2.5m 网格，检测失效)
```

### 2.5 数据量

```
Phase A:  50mm=413 + 80mm=996 + 100mm=1502 ≈ 2900 网格点 (合并后)
Phase B:  每半径约 110 万网格点，合计约 330 万
```

---

## 三、网络架构（不变）

```
DetNet(τ, Z) → (ρ, u, P)
  架构: SIREN 6×128, omega0=30
  输入: 2 维无量纲坐标 τ, Z
  输出: 有量纲 (ρ, u, P)，参考尺度不变 (rho_ref=1630, u_ref=2000, P_ref=21e9)

AirShockNet(τ, Z) → (ρ, u, P)  
  架构: Fourier Feature MLP 6×384, octaves=12
  输入: 2 维无量纲坐标 τ, Z
  输出: 有量纲 (ρ, u, P)，参考尺度不变 (rho_ref=1.225, u_ref=1000, P_ref=1e6)
```

输入从 (t, r) 换成 (τ, Z)，维度不变。网络内部归一化参数需调整：

```
DetNet: 需新增 τ_scale 和 Z_scale 归一化（当前代码无归一化，τ~1e-5 直接
  喂入 omega0=30 的 SIREN，sin(30×1e-5)≈3e-4，第一层在 τ 方向几乎线性）
  τ_scale ≈ 1e-5 (使 τ_norm ∈ [0, ~1.1])
  Z_scale ≈ 0.12 (使 Z_norm ∈ [0, ~1.0])

AirShockNet: t_scale 和 r_scale 需对应 τ 和 Z 的量级
  τ 范围 [0, ~870e-6]  → t_scale ≈ 1e-3 (不变)
  Z 范围 [0, 15]       → r_scale = 2.64 (按数据范围，不按域边界 Z_max=15)
  
  **关键**: r_scale 跟着数据走，不跟域边界走。Fourier 特征本身是周期函数，
  对 r_scale=2.64 时 Z∈[2.64,15] 映射到 [1, 5.7] 完全能处理。若设成 15，
  所有真实数据被压进 [0.008, 0.176]，浪费低频段分辨率。
```

---

## 四、训练数据

### 4.1 数据集合并

新增 `MultiRadiusDataset` 类，加载三个 `extracted_*/` 目录，在 `__getitem__` 中：

```python
1. 随机选 R₀ ∈ {50, 80, 100} mm
2. 从对应数据集中采样 (t, r, ρ, u, P)
3. 缩放: τ = t/M^(1/3)(R₀), Z = r/M^(1/3)(R₀)
4. 几何过滤（见下方域判断）
5. 密度过滤: ρ ≥ 100 kg/m³ (Phase A)，ρ ≥ 1.0 kg/m³ (Phase B)
```

### 4.2 域判断

```python
# Phase A 域
tau_sep = 11.12e-6  # 常数
Z_c = 0.1176        # 常数
Z_tail = interpolate(τ, sep_bundle)  # 各 R₀ 相同（已验证）
in_phase_A = (τ >= t_data_min_scaled) & (τ <= tau_sep) & (Z >= Z_tail) & (Z <= Z_c(τ))

# Phase B 域
Z_max = 15.0        # 物理边界，常数
in_phase_B = (τ >= tau_sep) & (Z >= Z_c) & (Z <= Z_max) & (Z <= x_end/M^(1/3))
#                                          ^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^
#                                          物理边界          数据可用边界
```

### 4.3 现有 extracted/ 目录不需要重新生成

缩放是运行时在 Dataset 里完成的，`raw_line.npz` 保持原样。

---

## 五、Hard Constraint 适配

当前 HardDetNetConstraint 使用物理单位的衰减：

```
tau_t = 1e-6 s, tau_r = 1e-3 m
```

缩放坐标下改为：

```
tau_t = 1.0e-6 s/kg^(1/3)     (≈ 0.1 × τ_sep，约束作用范围 ~5 单位)
tau_r = 1.0e-3 m/kg^(1/3)     (≈ 0.008 × Z_c，约束作用范围 ~5 单位)
```

由于 M^(1/3) 在 0.95~1.90 之间，同一缩放衰减对不同 R₀ 的实际物理范围差不到 2 倍，可以接受。若追求精确，可在 HardDetNetConstraint 中接受 M^(1/3) 参数动态计算，但收益微小。

---

## 六、损失函数与权重（统一，不按半径调整）

### 6.1 Phase A

```
A1 (5,000步, lr=1e-3):
  keys: data_A, IC_A, BC_A_slope
  权重:  5.0     0.1    0.001

A2a (50,000步, lr=5e-5):
  keys: data_A, IC_A, BC_A_slope, PDE_A
  权重:  2.0     0.1    0.001      10.0

A2b (100,000步, lr=1e-5):
  同上 + best checkpoint 跟踪（保存 PDE_A 最低点）
```

### 6.2 Phase B

```
B1 (5,000步, lr=1e-3):   data_B + IC_B
B2a (20,000步, lr=1e-5): data_B + RH_B + BC_B_outflow
B2b (60,000步, lr=1e-5): data_B + PDE_B + RH_B + BC_B_outflow (adaptive + best_ckpt)
B2c (30,000步, lr=5e-6): data_B + PDE_B + RH_B + BC_B_outflow (激波锐化)

权重不变（与当前 config 相同）:
  B1:  data_B=1.0,  IC_B=5.0
  B2a: data_B=5.0,  RH_B=1.0,   BC_B_outflow=0.01
  B2b: data_B=5.0,  PDE_B=0.01, RH_B=1.0,   BC_B_outflow=0.01
  B2c: data_B=5.0,  PDE_B=0.05, RH_B=1.0,   BC_B_outflow=0.01
```

### 6.3 激波轨迹处理

```
L_RH,B 需要激波位置 Z_s(τ):
  统一使用 50mm 的 Z_s(τ)（验证：缩放后 Z_s(τ) 对所有 R₀ 是同一曲线）
  有效范围: Z_s ∈ [0.118, 2.54], τ ∈ [11, 1054]
  Z > 2.54: 关掉 L_RH,B（激波已出训练数据域，且超压 <0.3MPa，RH 不再关键）
```

---

## 七、推理

```python
def predict(R0_mm, t, r):
    R0 = R0_mm / 1000
    M_third = 18.97 * R0
    tau = t / M_third        # t: 秒
    Z = r / M_third          # r: 米
    
    if tau <= 11.12e-6 and Z <= 0.1176:
        rho, u, P = det_net(tau, Z)
    else:
        rho, u, P = asn_net(tau, Z)
    
    return rho, u, P, P - 101325  # 超压
```

### 泛化范围

```
训练过: 50mm, 80mm, 100mm
内插:   60mm, 75mm → 全映射到训练覆盖的 (τ, Z) 范围内 ✅
外推:   20mm → Z ∈ [0.118, 6.6]，部分超出 2.64 数据区，弱激波段 PDE 外推
       200mm → Z ∈ [0.118, 0.66]，完全在数据密集区 ✅
```

---

## 八、改动清单

### 代码改动

| 文件 | 改动 | 量级 |
|---|---|---|
| `data/multi_radius_dataset.py` | **新建**：MultiRadiusDataset，合并多个半径数据，输出无量纲 (τ, Z) | ~100行 |
| `pinn/losses/detonation_loss.py` | 域边界 τ_sep, Z_c 替换 t_sep, R_c；t_data_min 改为缩放单位 | ~10行 |
| `pinn/losses/air_shock_loss.py` | 同上 + Z_max 物理边界；激波轨迹统一用 50mm Z_s(τ) | ~15行 |
| `pinn/networks.py` | AirShockNet 的 r_scale 2.5→15；HardDetNetConstraint tau_t/tau_r 改为缩放单位 | ~5行 |
| `pinn/trainer.py` | 使用 MultiRadiusDataset；Phase A A2→A2a+A2b+best_ckpt | ~30行 |
| `pinn/operators.py` | **不改**（缩放后方程形式不变） | 0 |

### 配置改动

| 文件 | 改动 |
|---|---|
| `configs/tnt_spherical_unified.yaml` | **新建**：三个数据路径、统一域参数、A2a/A2b 分段 |
| `configs/tnt_spherical_50mm.yaml` | 保留不删（向后兼容单半径训练） |
| `configs/tnt_spherical_80mm.yaml` | 保留不删 |
| `configs/tnt_spherical_100mm.yaml` | 保留不删 |

### 不改动

- SIREN / Fourier MLP 网络结构
- 球形 Euler + JWL / 理想气体 EOS operators
- RH 条件、outflow BC
- 密度过滤 rho_min_A
- 现有损失权重值
- 现有 extracted/ 数据文件

---

## 九、合并前验证（必须做）

合并前先确认三个半径缩放后确实塌缩到同一条曲线，否则合并会拉低 50mm 精度。

```python
# 三套数据各自缩放到 (Z, ΔP) 后叠图对比
# 特别注意近场 Z < 0.3（后燃效应可能破坏相似性）
# 若近场散开成束 → 束宽度 = 精度天花板
#   → 近场只用最密的一两个半径，或给不同半径不同采样权重
```

t_data_min 的含义需明确：
```
t_data_min=8μs 截断的原因是"LS-DYNA 前 8μs 有 P=0 伪影"
→ 这是物理时间的 bug，不是无量纲时间的 bug
→ 缩放后应对每个半径各自换算：τ_min(R₀) = 8e-6 / M^(1/3)(R₀)
→ 三个半径的 τ_min 不同（8.4, 5.3, 4.2 ×10⁻⁶），不能统一
```

τ_sep 统一使用 1.112×10⁻⁵（秒），全文档和代码统一，避免 11.12/11.12e-6 混用。

### 训练验证

```
训练数据: 50mm + 80mm + 100mm 全部合并
指标:
  L_PDE,A < 1×10⁻⁴
  L_PDE,B < 1×10⁻⁴ (best checkpoint)
  L_RH,B  < 1×10⁻²
  Gate: 0.00% (HardDetNetConstraint 保证)
```

### 泛化测试

```
内插:   预测 65mm，与 Henrych 公式 / 理论值比较
外推:   预测 150mm，同上
交叉:   训{50, 100}mm，测 80mm → 与 d3plot 对比
       训{50, 80}mm，测 100mm → 与 d3plot 对比
```

### 已知局限

```
1. Z ∈ [2.64, 15] 无训练数据覆盖 → 远场弱激波靠 PDE 外推
2. 80mm/100mm 激波轨迹在 τ>137/117 后失效 → 用 50mm 的 Z_s(τ) 替代
3. DetNet 当前无输入归一化，τ~1e-5 直接喂 SIREN → 需加 τ_scale
4. 近场 Z<0.3 三个半径可能未完全塌缩（后燃效应）→ 合并前必须验证
```

### 补充：80/100 未达标的实际根因

Phase A 数据量 80mm=996, 100mm=1502，比 50mm=413 **更多**，问题不在"数据稀疏"。
真正原因是**超参数按 50mm 调的**（lr schedule、步数、输入未经归一化）套到大半径上不匹配。
合并后不存在"单独训练 80/100"，这个矛盾自然消失。
