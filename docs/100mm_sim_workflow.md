# 100mm 球形 TNT 爆炸仿真 → 训练数据集 完整工作流

从 LS-DYNA 仿真到任意神经网络训练集的端到端流程。**这台机器上已验证**：
deck 生成 → `lsdyna_dp.exe` 运行 → d3plot 抽取 → 训练集导出，全部跑通。

## 0. 前置（一次性的）

- **许可证**：直接调用 `lsdyna_dp.exe` 需要把 ANSYS 许可模拟器目录加进 PATH 并把
  `LSTC_LICENSE` 设为 `Ansys`。`scripts/run_lsdyna.bat` 已封装好。
- **MCP**（可选，用 Claude 代跑）：`run_lsdyna_batch` 工具已注册（用户级），
  `ANSYS_LSDYNA` 指向 v221 的 `lsdyna_dp.exe`。重启 Claude Code 后生效。

## 1. 生成 deck（球形楔形 2×2 网格 + MMALE，SI 单位）

```bash
python scripts/gen_tnt_spherical_deck.py --R0 0.1 --out decks/tnt_spherical_100mm.k
```

- 参数化：`--R0 0.05 / 0.08 / 0.2` 可生成 50/80/200mm 装药。
- 网格：2×2 横向 × ~540 径向层，近场 1mm → 远端 8mm（装药内 100 个 1mm 单元）。
- 物理：TNT 用 `MAT_HIGH_EXPLOSIVE_BURN` + JWL（A=371.2GPa, B=3.7471GPa,
  R1=4.15, R2=0.95, ω=0.30, E0=6GJ/m³），空气用 `MAT_NULL` +
  `EOS_LINEAR_POLYNOMIAL`（γ=1.4），固定欧拉 MMALE（ELFORM=11, DCT=2），
  中心 `*INITIAL_DETONATION`，外边界非反射，楔形侧面 SPC 对称墙。
- d3plot 每 1µs 一帧，跑 2ms → 2000 帧。

## 2. 运行仿真

### 方式 A：直接脚本（推荐，最稳）

```bat
scripts\run_lsdyna.bat decks\tnt_spherical_100mm.k runs\100mm_full 4
```

输出（d3plot/d3hsp/messag）写到 `runs\100mm_full\`。跑完检查 `messag` 尾部有
`N o r m a l   t e r m i n a t i o n`。约 15 分钟（dt≈5e-9，楔形近轴薄单元驱动）。

### 方式 B：MCP（重启 Claude Code 后）

在对话里让 Claude 调用 `run_lsdyna_batch(input_file="F:/2/decks/tnt_spherical_100mm.k",
workdir="F:/2/runs/100mm_full", ncpu=4)`。

## 3. 抽取 1D 径向数据

```bash
# 先 --debug 确认密度 slot（应为 slot 0；装药 1630 kg/m³、空气 1.225）
python -m data.extract_d3plot --d3plot runs/100mm_full --debug

# 全量抽取（写 extraction 目录）
python -m data.extract_d3plot --d3plot runs/100mm_full \
    --out extracted_100mm_new --rho-hv-slot 0 \
    --R-0 0.1 --x-end 2.5 --n-x 2500 --rho-contact-threshold 100
```

抽取脚本已修复状态计数 bug（旧版按 d3plot 文件数算状态，小模型会丢帧；
现在读 timesteps 数组拿真实帧数）。核对输出：`metadata.json` 里 `n_t` ≈ 2000，
`raw_line.npz` 的 `rho` 满足质量守恒（`[mass] OK: ... ~ M_TNT (1.00x)`）。

## 4. 导出任意框架可用的训练集

```bash
python scripts/export_training_dataset.py --input extracted_100mm_new \
    --out dataset_100mm --times-us 10,20,50,100,500
```

输出 `dataset_100mm/`：
- `dataset.npz` — `t(r) / r(N_x) / rho,u,P(N_t,N_x) / R_s,R_c(N_t)`，全 SI。
- `fields.json` / `metadata.json` / `README.md` — 自描述 schema。
- `slices/` — 指定时刻的 `P(r)` CSV。

任意框架加载（README.md 里也有）：

```python
import numpy as np
d = np.load("dataset_100mm/dataset.npz")
t, r = d["t"], d["r"]
rho, u, P = d["rho"], d["u"], d["P"]   # (N_t, N_x) 各自
R_s, R_c = d["R_s"], d["R_c"]
```

- **PyTorch**：`X = torch.stack(torch.meshgrid(t, r, indexing="ij"), dim=-1)`
- **TensorFlow**：`tf.stack(tf.meshgrid(t, r, indexing="ij"), axis=-1)`
- **JAX**：`jnp.stack(jnp.meshgrid(t, r, indexing="ij"), axis=-1)`

## 5. （可选）喂进本项目 PINN

训练前更新 `configs/tnt_spherical_100mm.yaml` 的 `data.extracted_dir` 指向新抽取目录
（并确认 `rho_hv_slot: 0`、`vf_tnt_slot` 用 `--debug` 扫描结果），然后：

```bash
python -m pinn.trainer --all --config configs/tnt_spherical_100mm.yaml
python -m validation.convergence --config configs/tnt_spherical_100mm.yaml
```

## 数据质量自检（跑完看一眼）

- 峰值压力：装药近场应达 **~8–9 GPa**（ALE 下 CJ 峰值被 1mm 网格展宽；解析 CJ=21.8 GPa）。
- 接触面 R_c(t)：t≈0 时 = R_0=0.1m，t_sep≈21µs 后向外扩展。
- 激波 R_s(t)：约在 1µs 后开始，速度 ~2000-7000 m/s。
- 质量守恒：抽取脚本 `[mass] OK` 应通过（~1.00x M_TNT）。
