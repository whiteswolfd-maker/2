# PINN 爆炸衝擊波求解器

## 項目概述

球對稱 TNT 爆炸空氣衝擊波的 PINN 求解器。LS-DYNA 作為可微代理，四項損失函數物理正確性保證：

$$\mathcal{L}_{total} = w_{IC}\mathcal{L}_{IC} + w_{PDE}\mathcal{L}_{PDE} + w_{BC}\mathcal{L}_{BC} + w_{RH}\mathcal{L}_{RH}$$

三子網路：**MainNet** (SIREN 6×128) + **ContactNet** + **ShockNet**。

## 快速開始

```bash
pip install -r requirements.txt

# 物理驗證（無需 GPU）
python -m validation.physics_check

# 三階段訓練
python -m pinn.trainer --all --config configs/tnt_spherical.yaml

# 收斂診斷
python -m validation.convergence --stage 3

# 單獨各階段
python -m pinn.trainer --stage 1 --config configs/tnt_spherical.yaml
python -m pinn.trainer --stage 2 --config configs/tnt_spherical.yaml
python -m pinn.trainer --stage 3 --config configs/tnt_spherical.yaml
```

**必須從項目根目錄運行**（即 `pinn_tnt_explosion.py` 所在的目錄）：
```bash
cd /path/to/project
python -m pinn.trainer ...
```

## 目錄結構

```
configs/tnt_spherical.yaml   所有物理/網路/訓練超參
physics/                     純 NumPy 解析前處理（CJ/JWL/RH/耦合）
data/                        LS-DYNA loader（AnalyticalLoader 默認，FileLoader 留接口）
pinn/
  networks.py                MainNet(SIREN) + ContactNet + ShockNet
  operators.py               球面 Euler autograd 算子
  losses/                    L_IC, L_PDE, L_BC, L_RH
  trainer.py                 三階段訓練入口
validation/                  物理校驗 + 收斂診斷
tests/                       pytest 單元測試
```

## 斜槓命令

| 命令 | 功能 |
|---|---|
| `/cj-solve` | 運行 CJ 求解，輸出 CJ 狀態 |
| `/stage1-couple` | 運行 1.5-1.6，輸出 t_sep 及耦合曲線 |
| `/build-ic` | 構造 IC 剖面（Sedov 校核） |
| `/train-pinn` | 三階段訓練流水線 |
| `/validate` | 對照參考數據精度報告 + RH 殘差 |

## 單元測試

```bash
pytest tests/ -v
```

**測試錨點**：
- CJ 壓力 ≈ 21 GPa（誤差 < 15%）
- JWL 等熵嚴格遞減
- RH 殘差（Mach-5 精確解）< 1e-6
- 接觸面匹配 |u_p - u_s| / u_s < 1%
- autograd 導數對照解析函數 < 1e-4

## LS-DYNA 數據接入

當你有真實 LS-DYNA d3plot/CSV 數據時：
1. 將數據整理為：`snapshot.csv`（列：r, rho, u, P），`contact.csv`（列：t, P_c, u_c, R_c），`shock.csv`（列：t, R_s, D_s），`metadata.json`（鍵：t_sep, R_c_sep, R_s_sep）
2. 在 `pinn/trainer.py` 中將 `AnalyticalLoader` 替換為 `FileLoader(data_dir="your_data/")`
3. 補完 `data/lsdyna_loader.py` 中 `FileLoader` 的 CSV 解析邏輯

## JWL 參數（Lee-Tarver 1980 TNT 標準）

| 參數 | 值 |
|---|---|
| A | 371.21 GPa |
| B | 3.2306 GPa |
| R1 | 4.15 |
| R2 | 0.95 |
| ω | 0.30 |
| E0 | 7.0 GJ/m³ |

## 訓練收斂判據

| 指標 | 目標 |
|---|---|
| L_PDE (無量綱) | < 1e-4 |
| L_RH 各分量 | < 1% 最大激波後壓力 |
| R_s(t) 誤差 | < 1% (vs 參考解) |
| R_c(t) 誤差 | < 2% |
