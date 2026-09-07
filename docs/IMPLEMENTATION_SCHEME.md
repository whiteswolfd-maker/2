# 球形 TNT 爆炸衝擊波 PINN 求解器 — 實現步驟方案

> 通讀全代碼後整理的端到端實現方案。範圍：`physics/`、`pinn/`、`data/`、
> `validation/`、`configs/`、`scripts/`。
> 說明：程式碼註解/文檔多為繁體中文，本文件沿用。物理量一律 SI（kg, m, s, Pa, J）。

---

## 0. 一句話

用 **兩個前向耦合、順序訓練的 SIREN/Fourier-MLP** 解 1D 球對稱 TNT 空中爆炸的
Euler 方程：
**DetonationNet** 在三角形爆轟產物域 `{0≤t≤t_sep, r_tail(t)≤r≤r_c(t)}`（JWL 等熵），
**AirShockNet** 在矩形空氣域 `{t_sep≤t≤t_end, R_c≤r≤x_end}`（理想氣體 γ=1.4），
在單一連接點 **(t_sep, R_c)** 由凍結的 A 網路的輸出餵給 B 網路的 IC。
兩段皆由 **LS-DYNA d3plot 抽取的 1D 徑向資料**整域監督，解析公式提供物理約束。

---

## 1. 物理問題與建模

### 1.1 幾何與狀態
- 球形 TNT 裝藥，半徑 **R_0 = 50 mm**（本項目主 case；質量 M = ⁴⁄₃π R_0³ ρ_TNT ≈ 0.853 kg）。
- LS-DYNA 用 3D 楔形（八分之一球，`x∈[-5,0], y∈[0,5], z∈[0,5]` m）以球對稱建模；
  PINN 把「任一徑向剖面」當作 1D 徑向座標 r。
- 空氣：γ_a = 1.4，ρ_a = 1.225 kg/m³，P_a = 101325 Pa。
- 爆轟後產物走 **JWL 主等熵線**；空氣走**理想氣體**。

### 1.2 兩個時間段（為何拆兩網路）
| | Phase A（爆轟段） | Phase B（空氣激波段） |
|---|---|---|
| 域 | 三角 `{t∈[0,t_sep], r∈[r_tail(t), r_c(t)]}` | 矩形 `{t∈[t_sep,t_end], r∈[R_c,x_end]}` |
| EOS | JWL 等熵（產物） | 理想氣體 γ=1.4（空氣） |
| 網路 | DetonationNet（SIREN 6×128） | AirShockNet（Fourier-MLP 6×384, 12 octave）|
| 主要物理 | CJ → 等熵膨脹 → Taylor-Sadovsky 扇 | 激波 + RH + 稀疏波 → 環境 |

兩段 EOS 不同、解的尺度差 ~10⁴，單一網路難同時擬合 → 拆兩網路，靠單點耦合串起。

---

## 2. 分析前置：`physics/`（純 NumPy，無 torch）

這層把「幾何 + 材料參數」變成 PINN 需要的所有解析輸入，順序如下：

```
TNTParams(JWL: A,B,R1,R2,ω,E0, ρ0, D_CJ, P_CJ + 空氣 γ,ρ_a,P_a)
   │  cj_solver.solve_cj(params, D_CJ)
   ▼
CJState( v_CJ, P_CJ, ρ_CJ, u_CJ, c_CJ, isentrope )
   │  initial_coupling.match_contact(cj, γ, ρ_a, P_a)     ← §4.2.1 接觸面 Riemann 匹配
   ▼
ContactState( P_c=P_x, u_c=u_x, D_s, v_c=V_x )
   │  R_c = R_0·V_x^(1/3) ;  ρ_x = ρ0/V_x
   ▼
SeparationStateBundle( P_x, u_x, ρ_x, V_x, R_c, t_sep,
                       r_tail(t) 稀疏尾特徵線, ode(UniformExpansionResult) )
```

關鍵模組與角色：

- **`jwl_isentrope.py`**：`JWLIsentrope` 提供 `P_s(v)`、`-dP_s/dv`、`u_p(v)`
  （Riemann 不變量，沿等熵從 CJ 積分）、`v(P)`/`u_p(P)` 反查表、`e_iso(v)`。
  這是 JWL 側一切速度/能量的來源。
- **`cj_solver.py`**：Brent 解切線條件 `-dP_s/dv|_{v_CJ} = D_CJ²·ρ0`，得 CJ 態
  （P_CJ ≈ 21.83 GPa 對卡片標稱 21.0，差 4% 在 15% 容差內）。
- **`rh_relations.py`**：理想氣體激波 `D_s(P)`、`u_s(P)`、`ρ_s/ρ_a(P)`、`rh_residuals`。
- **`initial_coupling.py`**：§4.2.1 接觸面匹配——找 P_c 使
  `u_p_products(P_c) = u_s_air(P_c)`（等熵腿 + 強激波腿）。
  **注意**：此解是 0D 平面強激波值 **u_x≈7316 m/s，高估真實球面接觸速度 ~2.4×**
  （資料 ~3000 m/s ≈ 能量上界 √(2E/M)）。因此它只用於 sanity/gate 參考，
  **不作凍結點目標**（見 §5）。
- **`uniform_expansion.py`**：Taylor-Sadovsky 均勻膨脹 ODE
  `dr_c/dt = u_p((r_c/R_0)³)`，從 r_c(0)=R_0 積到 R_c，得 **t_sep** 與 r_c(t) 斜邊。
  `u_c` 沿此軌跡**嚴格遞增**。當前 50mm 值：**t_sep ≈ 10.55 μs**（舊強激波公式 4.24 μs 偏低 ~2.5×）。
- **`cj_state.py`**：facade `TNTParams` + `compute_cj_state` + `compute_separation_state`，
  打包上面三層成 bundle，同時給 extractor 與 trainer 用。

產生的**關鍵常數/軌跡**（trainer 消費）：
- `R_c ≈ 111.5 mm`、`t_sep ≈ 10.55 μs`、`(P_x≈0.079 GPa, u_x≈7316, ρ_x≈147, V_x≈11.1)`
- 扇下界 `r_tail(t)`：從 (t=0, r=R_0) 以 `u_CJ−c_CJ`（<0）向內走的 C⁻ 特徵線；
  `r<r_tail` 是未擾動 CJ 核心，**排除在 A 域外**。
- `r_c(t)`（Taylor-Sadovsky）當 A 域上界 / slope-BC 軌跡。

---

## 3. 資料管線：`data/`

### 3.1 抽取 `extract_d3plot.py`（LS-DYNA d3plot → extracted/）
1. 依序讀 `d3plot` + `d3plotNN`（lasso-python，**分批 25 state** 控制記憶體）。
2. 自動偵測單位（bbox 跨距 → `mm-Mg-s`/`si`）。
3. **每 state** 算每個 solid cell 的：
   - **質心** = 8 節點平均。關鍵坑：此 d3plot 的 `node_displacement` 存的是
     **絕對座標不是位移** → 用偵測（state0 |disp| > 1% 跨距 = 位置）直接當位置，
     否則質心半徑全翻倍（50mm → 99mm、質量 8.6kg 假象）。
   - **徑向速度** u_r = (cell 速度) · (質心單位向量)（與軸向無關，不再只取 +x）。
   - **壓力** P = −(σ_xx+σ_yy+σ_zz)/3。
   - **密度** = history-variable **slot 0**（用 `--debug` 找 [0.1,3000] kg/m³ 那格；
     slot 0 經質量自檢 0.99× 驗證正確）。
   - **TNT 體積分率 vf** = slot `[2,3,4,7]` 和（tnt/1n1/tnt1/tnt11，含核心 `1n1`）。
4. **球殼平均** `_interpolate_to_grid`：每個 cell 按真實半徑 `|c|` 歸到 1mm 殼，
   殼內平均 → 1D 徑向剖面；空殼用**最近有值殼**雙向填充。
5. 偵測 `R_s(t)`（P>2×P_atm 區最陡負梯度）與 `R_c(t)`（vf>0.01 最外點）。
6. 質量自檢 `[mass]`：`∫4πr²ρ dr` vs `M_TNT`，0.5–1.5× 內 → `OK`。
7. 寫出：`raw_line.npz (N_t×N_x)`、`shock.csv`、`contact.csv`、
   `product_pressure.csv`（僅診斷）、`r_c_taylor_sadovsky.csv`、`metadata.json`
   （含 **R_0, t_sep, R_c, P_x, u_x, ρ_x, V_x, CJ**）。

### 3.2 資料集 `d3plot_dataset.py`
- `D3plotLineDataset(extracted_dir)` 把 npz/csv/metadata 載成 tensor，提供：
  - **雙線性 `state_at(t,r)`→(ρ,u,P)**（訓練取樣核心，searchsorted clamp）。
  - `R_s(t)`/`R_c(t)`/TS `r_c(t)` 線性內插軌跡；`dP̄_p/dt`（僅診斷）。
  - `sample_grid_points(n)` 隨機 (t,r) 格點（供 convergence 抽樣）。
- **單點物理路徑**下座標是物理 (t∈s, r∈m)；`R_0`、`Z_R0=R_0`。

### 3.3 統一資料集 `multi_radius_dataset.py`（HC 縮放；未來統一模型用）
- Hopkinson-Cranz：`τ = t/M^(1/3)`、`Z = r/M^(1/3)`，其中 `M^(1/3) = α·R_0`，
  `α=(4/3π·1630)^(1/3)≈18.97`。
- 縮放座標下所有 R_0 的解析點相同：`TAU_SEP≈1.112e-5`、`Z_C≈0.1176`、`Z_R0≈0.0527`。
- 每 radius 一個 `D3plotLineDataset`，暴露縮放 `state_at(τ,Z,radius_idx)` 等。

---

## 4. 網路與座標

### 4.1 網路 `pinn/networks.py`
- **DetonationNet**：SIREN（sin, ω0=30）6×128，輸入 (τ,Z) **歸一到 O(1)**
  （τ/τ_scale, Z/Z_scale），輸出 (ρ,u,P) 用 `softplus` 保證 ρ,P>0；
  `u×tanh(Z/ε)` 強制球對稱 u(r→0)=0。
- **AirShockNet**：config 選 `fourier_mlp` → **FourierFeatureAirShockNet**
  6×384、12 octave，輸入先投影 cos/sin 多 octave 克服 SIREN 頻譜偏置
  （激波 ~ 近間斷需要高頻）。另有 plain-SIREN / GELU 變體可選。
- **硬約束包裝**（trainer 決定 target，見 §5）：
  - `HardDetNetConstraint`：CJ 錨在 (τ=0,Z_R0)（指數包絡，窄）；`use_contact=False`
    時**不錨**解析接觸點（避免解析 u_x 污染）。
  - `HardContactConstrainedASN`：把 AirShockNet 在 (t_sep,R_c) 硬釘到凍結 DetNet 輸出。

### 4.2 座標系統
- **單點物理**（50/80/100/200mm configs）：loss 直接用物理 (t,r)；
  DetNet 預設 τ_scale=1e-5、Z_scale=0.12 剛好把 50mm 的量級歸一。
- **統一縮放**（`tnt_spherical_unified.yaml`）：一律 (τ,Z)。**trainer 所有 key 都必須用
  同一套座標**——本次修正了 Phase-B 曾混用物理 (t_sep,R_c) 與縮放 (t_end,x_end) 的 bug。

---

## 5. 損失與順序訓練

### 5.1 九項損失
A 段（`pinn/losses/detonation_loss.py`）：
| 項 | 內容 | 採樣 |
|---|---|---|
| `L_data,A` | log ρ、u/u_ref、log P 的 MSE vs d3plot | 扇內稠密抽樣 `(τ∈[首幀,t_sep], Z∈[Z_tail,Z_c])`，`ρ≥ρ_min` 過濾 |
| `L_PDE,A` | 球 Euler 殘差（JWL e(v)）| 扇內均勻 collocation，**跳過 r<r_tail CJ 核心** |
| `L_IC,A` | 單點 CJ 錨 | (τ=0, r_anchor) |
| `L_BC,A,slope` | 沿 Z_c(τ) 對 d3plot | 扇上界軌跡抽樣 |

B 段（`pinn/losses/air_shock_loss.py`）：
| 項 | 內容 | 採樣 |
|---|---|---|
| `L_data,B` | log ρ,u,log P MSE | 矩形均勻 + `shock_frac` 比例的 R_s±band 加權 |
| `L_IC,B` | 兩段 IC：接觸點(重複 contact_repeat) + 環境 | t=t_sep |
| `L_PDE,B` | 球 Euler（理想氣體） | 矩形、**排除 R_s±shock_mask 與 t_margin** |
| `L_RH,B` | RH 跳躍在 R_s−ε vs 解析 | `D_s=dR_s/dt`，**M>2 才用** |
| `L_BC,B,outflow` | `∂R₊/∂r≈0` 在 x_end | r=x_end |

### 5.2 順序
```
A1: data+IC+slope      (Adam 5k, lr 1e-3)
A2: +PDE               (單 A2 或 A2a/A2b 課程)
Gate: 原始 DetNet @(t_sep,R_c) vs d3plot 資料 (u/P) < gate_rel_tol(=0.08)
       ★ 不是對解析 §4.2.1：u_x=7316 高估球面 2.4×，故資料才是凍結目標
Freeze: DetNet 凍結，快取 (t_sep,R_c) 輸出
B1: data+IC（IC 目標 = 凍結 DetNet 輸出，硬約束包裝同時釘同值）
B2: +PDE+RH+outflow（B2 單段或 B2a→B2b→B2c 課程）
```

### 5.3 逐段資料流
```
trainer.train_detonation:
   build_networks → DetonationNet
   包 HardDetNetConstraint(use_contact=False)
   _build_detonation_losses(…, det_train)
   A1 → A2 → gate → save detonation.pt (raw net)
trainer.train_airshock:
   build_networks → AirShockNet(fourier)
   用 frozen det @(t_sep,R_c) 輸出做 target
   包 HardContactConstrainedASN
   _build_air_shock_losses(…, asn_wrapped, frozen_det)
   B1 → B2 → save air_shock.pt + air_shock_hc_meta.pt
```

---

## 6. 驗證 `validation/`

### 6.1 `physics_check.py`（無資料，先跑）
5 項：CJ 壓力容差 / JWL 等熵單調 / RH float64 閉合 (M=5) / §4.2.1 匹配一致 /
Taylor-Sadovsky 端點。→ 全 PASS。

### 6.2 `convergence.py`（訓練後報告）
載入 checkpoint，重包硬約束，檢查：
1. **Phase A gate**：DetNet @(t_sep,R_c) vs **d3plot 資料** u/P（本次修正，原對解析必 FAIL）。
2. **Phase A 域內**中位 (ρ,P) 相對誤。
3. **耦合一致性**：AirShock vs 凍結 DetNet @(t_sep,R_c) 差 <1%。
4. **Phase B 域內**中位誤 + `L_RH,B` 值。

---

## 7. 端到端執行（單點 50mm）

```bash
.venv/Scripts/python -m validation.physics_check            # 物理前置
# 1) configs/tnt_spherical_50mm.yaml data.d3plot_dir 指到 LS-DYNA 目錄
#    (F:/tnt/tnt_files/dp0/SYS/MECH/；rho_hv_slot=0；vf_tnt_slot=[2,3,4,7])
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml   # → extracted/
python -m data.d3plot_dataset --check extracted/            # 健康檢查
python -m pinn.trainer --all --config configs/tnt_spherical_50mm.yaml    # A→B
python -m validation.convergence --config configs/tnt_spherical_50mm.yaml
# 或 scripts/run_pipeline.sh 一條龍
```
全程需從專案根目錄執行（imports 用絕對插入）。

**當前 50mm 資料狀態**：779 幀（t=0→1ms, dt≈0.99μs），質量自檢 0.99×，
t=0 剖面 rho=1630（0–50mm）/1.23（外），t<t_sep 有 11 幀供 Phase A。
**尚未訓練**（等待更多 R_0 仿真資料）。

---

## 8. 統一多半徑模型（未來）

構想：`tnt_spherical_unified.yaml` + `MultiRadiusDataset`，一個 DetNet + 一個
AirShockNet 在縮放 (τ,Z) 上吃多個 R_0 合併資料；推論時任意 R_0 = 把 (t,r)
縮放到 (τ,Z) 再跑網路。預測 R_c/t_sep/強度隨 R_0 依 HC 自動放大。

### ⚠ 已知缺陷（本次審計確認，統一模型開跑前必修）
- **`MultiRadiusDataset.state_at(τ,Z,radius_idx=0)` 預設 radius 0**；所有 data/slope
  loss 都以無 radius_idx 呼叫 → **實際只餵第一個 radius（50mm）資料**。
  `sample_grid_points`（唯一真正跨 radius 隨機採樣的方法）**沒有任何 loss 呼叫**。
  後果：統一訓練的監督訊號沒有跨半徑覆蓋（若 HC 崩潰完美、各 R_0 縮放場完全相同，
  則無害；真實 LS-DYNA 資料有殘差/散射，統一網應吸收它，故這是缺口）。
  修法：給 data/slope/IC loss 一個採樣 radius（各 loss 每 batch 隨機選 radius，
  再用該 radius 的 state_at）。等確定多 radius 資料佈局後實作。

---

## 9. 本次代碼審計修復清單（2026-09-07）

**Bug 修復**
1. unified Phase-B `t_sep/R_c` 混用物理值（第一 radius）vs 縮放 `t_end/x_end` →
   統一改用 `dataset.tau_sep/Z_c`（`pinn/trainer.py`）。
2. `tau_margin_B`（縮放 config）未被讀 → 現在 `t_margin_B`/`tau_margin_B` 都認。
3. 物理路徑 CJ 錨點 `Z_R0` 用縮放值 0.0527（R_0=200mm 時差 4×）→ `D3plotLineDataset.Z_R0=R_0`。
4. `convergence.py` + `make_report.py` 的 §4.2.1 gate（對解析必 FAIL）→ 改對 d3plot 資料。
5. `_shell_avg` 前導空殼填 0 → 雙向最近有值填充。
6. `L_0` 靜默當球半徑 → 顯式 WARN。
7. DetonationICLoss 死參 `u_CJ`（u 目標本為 0）、DetonationPDELoss 死參 `Z_min`。

**死碼刪除**
- imports：trainer(`TAU_SEP,Z_C`)、air_shock_loss(`shock_speed`)、
  detonation_loss(`D3plotLineDataset`)、d3plot_dataset(`sys`)、convergence(`numpy`)。
- functions：`uniform_expansion._u_c`、`rh_relations._hugoniot_pressure_ratio/post_shock_state`、
  `jwl_isentrope.sound_speed/minus_dPdV/v_grid/up_grid`、`d3plot_dataset.Pbar_p`。
- params：`match_contact(P_init_frac,tol,max_iter)`。
- config keys：`r_min_A`/`Z_min_A`（未被讀）。

**文件同步**：CLAUDE.md/AGENTS.md 的 t_sep 4.24→10.55μs、u_c 遞減→遞增、
gate 目標 §4.2.1 解析→d3plot 資料。

**保留/待決（未擅自動）**
- `MultiRadiusDataset` 統一採樣缺口（§8，等資料）。
- `operators.grad`、`spherical_euler_residuals` 的 mu_av 分支、`build_networks` 的
  `eps_origin_A` kwarg、`dPbar_p_dt`（測試錨定）——零呼叫但屬「設計可選」。
- yaml `validation.*` 多數鍵（`Rs_error_tol` 等）未被讀（僅 `separation_gate_tol`）；
  `configs/` 的 80/100/200mm 指向已刪除資料；過時 `/extract-d3plot` slash-command 指向
  `sim_data/3dTNT1/` 與已移除的 `--L0/--y-tol` 旗標。

---

## 10. 檔案對應速查

| 檔案 | 角色 |
|---|---|
| `physics/jwl_isentrope.py` | JWL 等熵 + u_p 不變量 |
| `physics/cj_solver.py` | CJ 切線解 |
| `physics/rh_relations.py` | 理想氣體激波 |
| `physics/initial_coupling.py` | §4.2.1 接觸匹配 |
| `physics/uniform_expansion.py` | Taylor-Sadovsky r_c(t), t_sep |
| `physics/cj_state.py` | facade（TNTParams + bundle）|
| `data/extract_d3plot.py` | d3plot → extracted/（球殼平均 + 質量自檢）|
| `data/d3plot_dataset.py` | 單點資料集 + state_at |
| `data/multi_radius_dataset.py` | 統一 HC 縮放資料集 |
| `pinn/networks.py` | DetNet/ASN + 硬約束包裝 + factory |
| `pinn/operators.py` | 球 Euler 殘差（2/r 源項）+ 自動微分 |
| `pinn/losses/detonation_loss.py` | A 段 4 loss |
| `pinn/losses/air_shock_loss.py` | B 段 5 loss |
| `pinn/trainer.py` | A1→A2→gate→B1→B2 主驅動 |
| `validation/physics_check.py` | 物理 5 檢查 |
| `validation/convergence.py` | 訓練後收斂報告 |
| `configs/*.yaml` | 每 case 超參（data/networks/sampling/weights/schedule）|
