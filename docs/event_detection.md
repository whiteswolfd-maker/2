# 在本地识别分离事件与 CJ 候选时刻

不需要上传大体积 `.k` 或 `d3plot` 文件。程序在本机读取已有的 `d3plot` 文件族，输出较小的事件报告。模型/结果里没有的信息不会被理论时间补出来。

## 最直接的运行方法

在 VS Code 打开本项目根目录，在终端执行。先确认使用包含本次修改的代码，且配置文件的 `data.d3plot_dir` 指向自己的结果目录。

```powershell
python -m pip install numpy scipy pyyaml lasso-python
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml --out extracted_events_50mm --batch-size 2
```

不改配置中的目录，也可以通过命令覆盖，路径有空格时保留引号：

```powershell
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml --d3plot "F:/你的结果目录/MECH" --out extracted_events_50mm --batch-size 2
```

目录中应有主文件 `d3plot` 及原有的 `d3plot01`、`d3plot02` 等文件；这些文件不一定与输出帧一一对应。程序按文件内部的真实帧号读取。保留整个文件族。

内存不足可用 `--batch-size 1`。程序限制每批的三维数据量，但仍保存所有所选帧的一维剖面。默认不跳帧。可通过 `--max-time` 指定仿真原始时间上限（单位秒），只提取早期输出；截断过早可能遗漏事件或不足以验证持续性。不要为了省内存把 `--state-stride` 调大后仍声称具有原来的时间精度。

100 mm 算例使用自己的配置和独立输出目录：

```powershell
python -m data.extract_d3plot --config configs/tnt_spherical_100mm.yaml --out extracted_events_100mm --batch-size 2
```

各算例分别识别，保留各自真实时间与位置，不按比例强制重合。对完全留出的 500 mm 工况，用其轨迹指导预测属于额外使用该工况数据；用于事后验证则应与训练/预测输入分开。

## 需要正确配置的信息

程序复用原来的密度、TNT 材料体积分数槽位；槽位含义必须与当前 DYNA 算例一致。50 mm 配置中的槽号是这个项目原有设置，不适用于任意新模型。数值落在 `[0,1]` 只能排除部分错误，不能证明该槽位是什么物理量。

| 配置或参数 | 含义 |
|---|---|
| `data.R_0` | 原始装药半径，米 |
| `data.center` / `--center X Y Z` | 球心坐标，米；默认原点 |
| `data.rho_hv_slot` | 密度的历史变量槽号，从 0 开始 |
| `data.vf_tnt_slot` | TNT 来源材料体积分数槽号，允许多个相加；不重复计数 |
| `data.burn_hv_slot` / `--burn-hv-slot N` | 可选，已核实的反应/燃烧进度槽；范围 `[0,1]`，1 代表完成 |
| `--unit-system si` / `mm-Mg-s` | 覆盖原有单位自动判断；必须按求解器数据的真实单位选择 |
| `--coordinates-mode absolute` / `displacement` | 覆盖节点坐标数组含义；自动模式只在原始第一帧判别一次 |

`burn_hv_slot` 默认是 `null`，程序不会自动把 TNT 体积分数当成燃烧进度。多个材料分别输出的反应变量也不能未经映射就简单相加。数据没有合适的燃烧进度时，仍可输出 P、rho、u 的 CJ 状态候选，但不能证实反应完成。

需要检查槽位时：

```powershell
python -m data.extract_d3plot --config configs/tnt_spherical_50mm.yaml --debug
```

自动单位判断依赖几何尺寸，大尺度 SI 模型可能需要显式指定单位。球心、单位和槽位错误会使事件位置失去意义；程序会记录本次使用的配置与坐标模式，便于核对。

## 输出怎么看

首先打开输出目录的 **`events_summary.txt`**。

| 文件 | 内容 |
|---|---|
| `events_summary.txt` | 中文时间/位置摘要及状态 |
| `events.json` | 完整判据、时间区间、位置区间、状态、来源、诊断说明 |
| `event_trajectories.csv` | 每帧材料界面、空气波前、间隔、分辨率和未通过原因 |
| `cj_candidates.csv` | 每帧近表面区最接近参考状态的原始单元及联合误差；并非每一行都通过 CJ 判据 |
| `shock_raw.csv` | 原有压力检测器的未平滑轨迹，仅供对照，早期不一定为空气激波 |
| `raw_line.npz` | 原有场，加材料体积分数、实际填充数量、径向单元跨度、原始帧号和紧凑 CJ 记录 |

`events.json` 的简明时间字段是：

```python
import json
from pathlib import Path

events = json.loads(Path("extracted_events_50mm/events.json").read_text(encoding="utf-8"))
t_gap = events["t_sep_detected_s"]
t_cj_candidate = events["t_cj_candidate_s"]
print("可分辨分离时刻/s:", t_gap)
print("CJ 候选时刻/s:", t_cj_candidate)
print("分离状态:", events["separation"]["status"])
print("CJ 状态:", events["cj"]["status"])
```

时间全部使用 **DYNA 原始时间**，单位秒。JSON 的 `null`（Python 中为 `None`）表示没有识别到，不能当作 0 使用。CJ 的 `closest_sample` 即使非空，也不表示已经识别成功。

### 分离事件

从中央的 TNT 来源材料区向外识别体积分数过渡，使用 0.9、0.5、0.1 交点及实际径向单元跨度形成接触位置区间。随后在相邻空气一侧，联合检查压力、密度、向外速度，以及波前外部是否恢复环境状态。

空壳层的插值/填充值不计作实际观测。空气层需包含足够的实际填充壳层，且接触面和波前的不确定位置区间已经分开。默认连续 3 帧满足、至少 3 个填充壳层、间隔不少于 2 个有效径向单元尺度。它们都是可调整的诊断阈值，不是论文中的普适物理常数。

`separation.status=detected` 表示在已有网格和输出上满足本判据。缺少原始网格跨度/填充数时降为 `candidate`。体积分数缺失则为 `insufficient_data`；波前超出提取范围、间隔不足或未持续满足等情形为 `not_identified`。

记录的时间是首次满足且由后续帧支持的输出帧时间。`time_interval_s` 是前一帧到这帧的**可分辨判据跨越区间**，不是物理上首次产生空气冲击波的区间。第一帧已满足时下界为空。它也没有自动等同于岳军政论文 §4.2.1 的出流时刻；论文状态和时间定义还要另外核实。

### CJ 候选

默认查看原始表面内侧 `0.9 R_0 <= r <= R_0` 的 TNT 主导单元，在球壳平均之前同时比较 P、rho、径向 u 与参考 CJ 值。每一项相对误差都需不超过默认 15%；没有任何一项可用另一项的匹配抵消。短暂 CJ 状态可能只出现于一个输出帧，因此不强求连续 3 帧。

这是当前空间位置的近表面采样带，不是跟随某个物质点的轨迹。CJ 候选的实际半径一并输出；内侧单元到达状态的时刻不自动等于爆轰面到达精确 `R_0` 的时刻。近表面没有可用单元时会报告缺样，不悄悄扩大搜索到整个装药。

- `state_candidate`：同一原始单元的 P、rho、u 在容差内匹配，材料体积分数通过；没有燃烧进度证据。
- `state_and_burn_candidate`：以上条件满足，且配置的燃烧进度达到阈值。
- `not_identified`：没有通过联合判据；可查看最接近样本的误差，不能把它强行当成 CJ。

即使后一种候选通过，也没有独立检验爆轰波参考系的声速条件，不能声称识别了真实 CJ 形成过程。是否使用程序燃烧、是否解析反应结构、CJ 参数是否与材料卡一致，都影响解释。这个限制不能靠放宽阈值消除。

`cj.time_interval_s` 列出候选帧前后的输出时间，帮助判断时间采样精度；不是 CJ 形成的严格上下界。

参考 CJ 值默认来自本项目已有 CJ 求解器，不使用 `R_0/D` 生成时间。可在配置的 `events.cj_reference` 显式给出一致的 `{P: ..., rho: ..., u: ...}`（SI 单位），只影响检测，不修改训练锚点。修改参考/近表面区域/材料纯度/燃烧阈值后需要从原始数据重新提取，紧凑记录不能恢复之前未保留的单元。

## 已经提取过，想重新检查

```powershell
python -m data.detect_events --input extracted_events_50mm --out event_check_50mm
```

这一入口只需要 NumPy；读取 YAML 时还需 PyYAML。新 NPZ 可以重用原始单元的紧凑 CJ 记录，避免重新读取大文件。旧 NPZ 没有材料体积分数/原始单元记录时，会明确降级，无法凭空补出事件；需要在本地重新运行提取命令。

## 与当前 A/B 训练的关系

本次实现负责**识别与报告**。原有 `metadata.t_sep` 仍是理论膨胀计算时间，`metadata.t_0` 仍是旧材料半径交点；二者已加定义标签。新增 `t_sep_detected`、`t_cj_candidate` 单独保存。

检测结果没有自动平移场坐标，也没有自动覆盖 A/B 硬约束。采用一个已经膨胀的实际数据帧作为新零点时，不能把该帧强制改回 CJ。论文出流状态、检测事件与训练时空域核实一致后，才能把经过审查的起止区间用于采样。

## 验证与依据

测试覆盖已知合成界面的识别、单帧假信号、域边界截断、稀疏网格、原始单元峰值在平均中丢失、压力单项匹配、未完成燃烧、缺失体积分数、坐标批次一致性，以及提取到保存再重读的完整流程。二进制读取接口在集成测试中用合成替身替代；尚未用用户的原始 d3plot 验证槽位与真实事件。

- [LASSO-Python D3plot 接口](https://open-lasso-python.github.io/lasso-python/dyna/D3plot/)：按状态过滤、缓冲读取；实现中再限制每次保留的帧数。
- [Clawpack Euler 方程与接触间断](https://www.clawpack.org/riemann_book/html/Euler.html)：接触面、激波与材料运动的区别。
- [Caltech Shock and Detonation Toolbox 教程](https://shepherd.caltech.edu/EDL/PublicResources/sdt/nb/sdt_intro.slides.html)：CJ 状态条件与反应过程的区别。该教程不提供本程序的经验检测阈值，也不证明这些阈值适用于所有 TNT 算例。
