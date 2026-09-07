# PINN 球形 TNT 裝藥（R_0 = 50 mm）爆炸衝擊波求解器（d3plot 數據驅動）

## 項目概述

球形 TNT 裝藥（半徑 R_0 = 50 mm，直徑 100 mm，質量 M_TNT = (4/3)π R_0³ ρ_TNT ≈ 0.854 kg）在空氣中爆炸的 1D 球對稱 PINN 求解器；LS-DYNA 用 3D wedge 利用球對稱、PINN 沿 +x 軸切剖面當作徑向 r。**LS-DYNA d3plot 真實數據**作為整域監督，**北理工博論 §4.2.1** 解析 Riemann 匹配 + **Taylor-Sadovsky 均勻膨脹 ODE** 提供物理約束。

兩網路單點耦合 + 順序訓練：

```
                  Phase A（爆轟段）                       Phase B（空氣激波段）
  ┌────────────────────────────────────┐   ┌──────────────────────────────────────┐
  │      DetonationNet(t, r)            │   │      AirShockNet(t, r)                │
  │  三角域 {t∈[0,t_sep], r∈[0,r_c(t)]}  │   │  矩形 {t∈[t_sep,t_end], r∈[R_c,x_end]}│
  │  斜邊 r_c(t) ← Taylor-Sadovsky ODE   │   │  EOS：理想氣體 γ_a = 1.4               │
  │  EOS：JWL 等熵                       │   │      → (ρ, u, P)                      │
  │      → (ρ, u, P)                     │   │                                       │
  └────────┬───────────────────────────┘   └────────▲──────────────────────────────┘
           │ A 訓練 → 凍結 → 取 (t_sep, R_c) 預測值     │
           │  (ρ_x_pred, u_x_pred, P_x_pred) ────→     │
           └→ 作為 AirShockNet 在 (t_sep, R_c) 的單點 IC ┘
```

| 網絡 | 任務 | EOS |
|---|---|---|
| **DetonationNet** SIREN 6×128 | 預測 (ρ, u, P)(t, r) 在三角域 | JWL 等熵 |
| **AirShockNet** SIREN 6×128 | 預測 (ρ, u, P)(t, r) 在矩形域 | 理想氣體 γ=1.4 |

**九項損失**（A 段 4 + B 段 5）：
- A: L_data,A + L_PDE,A + L_IC,A + L_BC,A,slope
- B: L_data,B + L_IC,B + L_PDE,B + L_RH,B + L_BC,B,outflow

**訓練流程**：
* **A1** Adam：`L_data,A + L_IC,A + L_BC,A,slope`
* **A2** Adam：+ `L_PDE,A`（JWL 球形 Euler 殘差）
* **Gate**：DetonationNet(t_sep, R_c) vs d3plot 資料 (u, P) 誤差 < gate_rel_tol；不過則回 A2
  （§4.2.1 解析 u_x=7316 高估球面接觸速度 ~2.4×，僅作 sanity，不作凍結目標）
* **Freeze**：DetonationNet 凍結；快取 (t_sep, R_c) 預測值
* **B1** Adam：`L_data,B + L_IC,B`（IC 用凍結 A 在 (t_sep, R_c) 的輸出）
* **B2** Adam：+ `L_PDE,B + L_RH,B + L_BC,B,outflow`

## 快速開始

```bash
pip install -r requirements.txt

# 物理驗證（無需資料）
python -m validation.physics_check

# 1) 編輯 configs/tnt_spherical_50mm.yaml 的 data 區塊：
#    - data.d3plot_dir：你的 LS-DYNA d3plot 目錄
#    - data.rho_hv_slot：密度 history-variable slot
#    （先跑 --debug 識別 slot；找 median 落在 [0.1, 3000] kg/m^3 那一格）
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml --debug
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml

# 2) 數據健全檢查
python -m data.d3plot_dataset --check extracted/

# 3) 順序訓練（A→Gate→凍結→B）
python -m pinn.trainer --all --config configs/tnt_spherical_50mm.yaml

# 4) 收斂報告（兩階段 + §4.2.1 gate + 連接點耦合一致性）
python -m validation.convergence --config configs/tnt_spherical_50mm.yaml

# 一條命令做完上面所有事：
bash scripts/run_pipeline.sh
```

**必須從項目根目錄運行**（含 `pinn/`、`data/`、`physics/` 的目錄）。

## 目錄結構

```
configs/tnt_spherical_50mm.yaml  所有網路 / 損失 / 訓練超參 + d3plot 路徑 placeholder
data/
  extract_d3plot.py              lasso-python 讀 d3plot → extracted/（含 r_c_taylor_sadovsky.csv）
  d3plot_dataset.py              torch dataset：bilinear (t,r)→(ρ,u,P) + R_c/R_s 軌跡 + Taylor-Sadovsky
physics/
  cj_solver.py                   CJ 切線解
  cj_state.py                    TNTParams + compute_cj_state + compute_separation_state（facade）
  jwl_isentrope.py               JWL 等熵 + Riemann 不變量 u_p(v) + e_iso(v)
  initial_coupling.py            §4.2.1 接觸面 Riemann 匹配 (match_contact)
  uniform_expansion.py           Taylor-Sadovsky 均勻膨脹 ODE（r_c(t), t_sep）
  rh_relations.py                RH 殘差 + 強激波解析（numpy）
pinn/
  networks.py                    DetonationNet + AirShockNet (SIREN 6×128)
  operators.py                   球形 Euler 殘差（含 2/r 源項）+ JWL/理想氣體 EOS helpers
  losses/
    detonation_loss.py           4 項 A 段損失（data/PDE/IC/slope BC）
    air_shock_loss.py            5 項 B 段損失（data/IC/PDE/RH/outflow BC）
  trainer.py                     順序訓練主入口（A1→A2→gate→freeze→B1→B2）
validation/
  physics_check.py               5 項物理 sanity（CJ / 等熵 / RH / 接觸面 / Taylor-Sadovsky）
  convergence.py                 PINN vs d3plot 收斂報告（兩階段 + gate + 耦合一致性）
scripts/
  run_pipeline.sh                端到端一條命令：抽取 → 檢查 → 訓練 → 收斂報告
tests/                           pytest 單元測試
```

## 斜槓命令

| 命令 | 功能 |
|---|---|
| `/cj-solve` | 跑 TNT CJ 求解，輸出 P_CJ ≈ 21 GPa |
| `/extract-d3plot` | 從 `sim_data/3dTNT1/` 抽 1D +x 軸到 `extracted/` |
| `/dataset-check` | 健全檢查 extracted/ |
| `/train-pinn` | 順序訓練 DetonationNet → AirShockNet |
| `/validate` | 物理 sanity + 收斂報告 + pytest |

## 單元測試

```bash
pytest tests/ -v
```

**主要測試錨點**：
- 球形 Euler MMS：靜止狀態零殘差；planar 對照組驗證 2/r 源項存在性
- DetonationNet：u(t, 0) = 0 結構保證；正則性（ρ, P > 0）
- AirShockNet outflow：ambient 殘差 < 1e-10
- §4.2.1 接觸面：|u_p − u_s|/u_s < 1%
- Taylor-Sadovsky ODE：r_c(0)=R_0、r_c(t_sep)=R_c、t 嚴格遞增、u_c 嚴格遞增
- JWL e_iso(v)：torch ↔ numpy 兩條路徑機器精度匹配
- CJ 壓力：21 GPa ± 15%（LS-DYNA 卡片標稱目標）

## d3plot 數據接入細節

LS-DYNA d3plot **不存** 顯式 cell-pressure 或 cell-density 陣列：

| 量 | 來源 |
|---|---|
| 壓力 P | `element_solid_stress` 三軸之 P = −(σ_xx + σ_yy + σ_zz) / 3 |
| 密度 ρ | `element_solid_history_variables` 某 slot；slot 編號取決於 *DATABASE_EXTENT_BINARY 的 NEIPH |
| 速度 u_x | `node_velocity[:, :, 0]` 對 8 個 hex 節點平均到 cell 中心 |
| Cell 中心 | `node_coordinates + node_displacement` 對 hex 節點平均 |

**工作流程**：

1. 編輯 `configs/tnt_spherical_50mm.yaml` 的 `data.d3plot_dir`：填你的 LS-DYNA 結果目錄
2. 識別密度 slot：
   ```bash
   python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml --debug
   ```
   找 median 落在 `[0.1, 3000] kg/m^3` 那一格，回填到 `data.rho_hv_slot`
3. 全抽取：
   ```bash
   python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml
   ```
4. `extracted/` 含：
   - `raw_line.npz`（t, x, ρ, u, P 全格 (N_t, N_x)；x 即徑向 r）
   - `shock.csv`（t, R_s 軌跡）
   - `contact.csv`（t, R_c 軌跡）
   - `product_pressure.csv`（t, P̄_p 軌跡，僅供診斷）
   - `r_c_taylor_sadovsky.csv`（t, r_c(t) 從 Taylor-Sadovsky ODE）
   - `metadata.json`（L_0, t_0, t_end, x_end, ρ_CJ, P_CJ, **R_0, R_c, t_sep, P_x, u_x, ρ_x, V_x**）

## JWL 參數（與本項目 LS-DYNA *EOS_JWL 卡片一致）

| 參數 | 值 |
|---|---|
| ρ_TNT | 1630 kg/m³ |
| D_CJ | 6930 m/s |
| P_CJ（卡片標稱目標）| 21.0 GPa |
| A | 371.2 GPa |
| **B** | **3.7471 GPa** |
| R1 | 4.15 |
| R2 | 0.95 |
| ω | 0.30 |
| **E0** | **6.0 GJ/m³** |
| V0 | 1（無因次 V/V0）|

→ JWL 切線解（self-consistent）：P_CJ ≈ 21.83 GPa（與卡片標稱差 4%，落在 15% 容差內 — 這是 JWL 過約束的標準現象）  
→ ρ_CJ ≈ 2207 kg/m³，u_CJ ≈ 1811 m/s

## 接觸面 §4.2.1 解析狀態（自洽算出，僅作 gate / sanity check）

R_0 = 50 mm（球形裝藥半徑，與 LS-DYNA 模型一致；M_TNT = (4/3)π R_0³ ρ_TNT ≈ 0.854 kg）。

| 量 | 值 |
|---|---|
| P_x | ≈ 0.079 GPa |
| u_x | ≈ 7316 m/s |
| ρ_x | ≈ 147 kg/m³ |
| V_x / V_0 | ≈ 11.10 |
| R_c = R_0 · V_x^(1/3) | ≈ 111.5 mm |
| t_sep（Taylor-Sadovsky 積到 R_c，u_p 不變量 ODE）| ≈ 10.55 μs |

(P_x, u_x, ρ_x, V_x 只依賴 JWL 參數與空氣，不隨 R_0 改變；R_c 與 t_sep 線性放大隨 R_0)。

## 訓練收斂判據

| 指標 | 目標 |
|---|---|
| L_PDE 兩段（無量綱）| < 1e-4 |
| L_RH,B（無量綱）| < 1e-2 |
| L_IC,B（含 (t_sep, R_c) 單點耦合）| < 1e-3 |
| DetonationNet 在 (t_sep, R_c) vs d3plot（gate，u/P）| 誤差 < 8% |
| AirShockNet 在 (t_sep, R_c) vs 凍結 DetonationNet（耦合一致性）| 差 < 1% |
| 各段中位 RMSE（vs d3plot）| < 5% |

## 數據-代碼分工

* **本機（你）**：跑 LS-DYNA → 得 d3plot → 編輯 yaml 中 d3plot_dir → `bash scripts/run_pipeline.sh`
* **此 Linux 環境（我）**：寫程式 + 跑單元測試（無 d3plot 真實資料），最後推到 `Codex/pinn-blast-wave-solver-wkExs` 分支
