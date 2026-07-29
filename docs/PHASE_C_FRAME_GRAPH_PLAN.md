# Phase C：整帧 CUDA Graph 蓝图（2026-07-28 起草）

目标形态（用户原话）：整帧图 + 条件节点（CUDA 12.4+，本机 driver=13010 ✓），
增长只许发生在帧边界（discard-grow 铁律已保证大半），异常改为帧尾裁决，
BVH sort n 固定。Phase D 在其上叠 episode 级驻留（动作整段预上传、帧图循环
消费、观测异步双缓冲流回）。

## 现存逐迭代宿主同步清单（B2' 后的全部残余，towel/foldshirt 实测）

| 读点 | 大小 | 宿主消费 | 图化策略 |
|---|---|---|---|
| ccd 标量链 | 72B×0.81/iter | alpha/temp_alpha 起点值 + validate + diag | alpha 已在 slots[5] 设备驻留；LS 起点改读设备（见下）；validate→帧尾旗标；diag→帧尾批量 |
| LS decision | 12B×1/trial | 回溯 while 环控制 | **WHILE 条件节点**：`_global_ls_decide` 已写 status[0]，改写条件句柄；trial 体=stepForward+buildCP+energy+decide 全设备 |
| LS 出口计数刷新 | 24B+4B×1/iter | 下迭代 GH 装配的 offset 算术（global_triplet_offset 等） | 最难点：装配 offset 是宿主算术编入 launch 参数。对策=装配核改设备 offset 读（`d_cpNum` 前缀和已在设备）或每迭代一次 32B 打包读（保留，图外） |
| GH start-ids | 16B×0.97/iter | ABD/FEM contact 段偏移宿主算术 | 同上打包读或设备 offset 化 |
| PCG 状态 | 16B×1/iter | k/h_break（device-loop 后的收尾读） | Newton WHILE 条件节点内消化 |
| Newton 收敛 | （含于上） | distToOpt_PN < threshold break | **WHILE 条件节点**：残差已设备归约，写条件句柄 |

## 分层实施（每层独立可验，沿用 建→锚→套件→指纹→推 环）

**C-1 LS 试探环图化**（最高价值/最低风险）：
- cudaGraphConditionalHandle + WHILE node 包 trial 体；`_global_ls_decide`
  尾部 `cudaGraphSetConditional(handle, status[0]==1 && trial<budget)`。
- 溢出恢复/gdCollapse：decide 已写 status[1]/[2]——条件节点内不可宿主恢复
  → 溢出时条件节点退出 + 帧尾（迭代尾）宿主检查 status[1] 走 legacy 重评
  （罕见路径，语义=今日恢复的迭代尾版）。
- 出口计数刷新保留（图外 1 次），decision 读消失（-1/trial）。
- 前提：trial 体内 buildCP 的 grow 不可发生——B3 已保证（trial defer 模式
  发射 trash+计数器，无 grow）✓；BVH build 的 thrust sort 需换 cub 固定
  scratch（thrust 在 capture 下会 malloc——检查：_mc_sort_active 已是 cub
  预分配（per-env 路径注释），merged 路径 thrust::sort_by_key 需换）。

**C-2 Newton 环图化**：外层 WHILE（收敛/迭代帽双条件），体=GH 装配+solve+
ccd_alpha+LS 子图。前提=C-1 + 装配 offset 设备化（或按迭代 exec-update 注
参）。异常（GeometryError 族）→设备旗标，WHILE 退出后帧尾统一 throw（用户
指定语义）。

**C-3 帧图**：Newton WHILE + 帧首 buildBVH/buildCP + 帧尾 checkpoint 配对
集（帧入口配对集=carrier，检查点门禁 1e-12/逐位已锁）。增长事件=图失效→
帧边界重录（B2' 代数计数器直接复用）。

**C-4 Phase D**：episode 驻留。动作序列 H2D 一次；帧图 WHILE-loop 消费；
观测 async D2H 双缓冲 + event 围栏；宿主整 episode 零阻塞。

## 风险与既得资产
- 条件节点要求 12.4+ runtime/driver：本机 ✓；A800 nsys 2026.3.1 环境待查
  runtime 版本（venv wheel sm_80 需带 12.4+ toolkit 构建——dlto 锚已跨架构）。
- strict 模式：图化后 thrust→cub 替换等必须保锚逐位（cub 定序 ✓）；strict
  可先走非图路径（同 B2' 的 !SPMV_DET 门控先例），锚免疫。
- 已得资产：PCG device-loop（尾launch 自循环图先例）、B2' 容量网格/设备计
  数模式全套、单一代数计数器、15 段门禁+指纹配方。

## C-1 落地战报：LS 回溯 trial 自尾图（2026-07-28）

`STIFF_LS_GRAPH`（默认 0）把首个能量试探失败后的回溯子环录成 device-launch
CUDA Graph：设备 alpha 每轮减半，trial 体执行 step→BVH→CP(defer)→energy→
decide，尾核按 `{decision==1 && trials<budget}` 自重发。宿主每个回溯子环只在
图结束后读一次 24B packed status；首个试探的 12B decision 仍保留，归 C-2
外层 Newton 图吸收。

捕获安全补齐：
- DCD/CCD snapshot 改 stream-ordered async D2D；ABD 固定拓扑 energy workspace
  只在尺寸实际变化时 resize，避开 muda `DeviceBuffer::resize()` 的无条件 wait。
- graph 签名覆盖 `{全局指针代际、budget、cp/gp energy launch bound、snapshot
  长度}`；pair/sort/reduce/snapshot 指针移动均 bump 代际，脏图必销毁重录。
- 捕获体抛异常以及 begin/end/instantiate 任一 API 失败都永久降级到宿主环，
  不留下非法 capture 状态；释放 solver 前先销毁 graph exec。
- strict/default-off 路径不进图；默认 15 段门禁全绿，strict 锚
  `0544461bd82123ae`（53 Newton）逐位原位。knob-on towel 完整 recipe 打印
  `trial self-tail graph active (budget=64)` 后 PASS；foldshirt merged N=4
  30f 兼容冒烟 PASS（该 30f 负载没有发生回溯，故不触图）。

4090 Nsight Systems 2024.6.2 指纹（同一固定种子 towel recipe，`STIFF_NVTX=1`；
SQLite 以 runtime correlation + NVTX `line_search` 区间联查）：

| 模式 | line_search 区间 | 12B decision D2H | 超出首试的 12B | 24B D2H | 额外 packed 24B | host graph launch |
|---|---:|---:|---:|---:|---:|---:|
| graph=0 | 1144 | 1173 | **29** | 1144 | 0 | 0 |
| graph=1 | 1136 | 1136 | **0** | 1160 | 24 | 24 |

merged 的 Newton 数本来即非逐位确定，故两行各自按本行区间数归一：graph=1 时
12B 次数与区间数严格相等，证明所有回溯 trial 决策读均消失；24 个实际进入
回溯的 line search 各付一次 packed 读，而非每 trial 一次宿主往返。

## C-2/C-3 落地战报：Newton→PCG→LS 整帧条件图（2026-07-28）

`STIFF_FRAME_GRAPH=1 STIFF_FRAME_FULL_GRAPH=1`（两个旋钮均默认关闭）启用
整帧条件图候选。录制后的层次为：

```text
root graph
└─ Newton WHILE
   ├─ GH 装配（设备计数/固定容量 launch bound）
   ├─ PCG WHILE
   ├─ convergence IF
   └─ LS WHILE
```

每次物理帧只有一次 root `cudaGraphLaunch`、一次帧尾
`FrameStatus` D2H 和一个宿主帧边界；Newton、PCG、LS 的继续/退出均由设备
条件句柄决定。异常、非有限值和容量不足先写 `FrameDeviceState`，停止条件环，
再由帧尾统一裁决。失败帧通过入口快照逐位恢复 FEM/ABD/κ，容量增长和重录只
发生在合法帧边界。

实现中补齐了以下捕获安全前提：

- 接触分类、稀疏矩阵范围、MAS 范围和 line-search 能量范围改读设备计数；
- merged BVH 与 MAS 的 capture 路径使用预分配 CUB scratch，不在捕获中调用
  Thrust 分配；
- ABD converter 使用边界训练的容量 tier，图内不做 D2H/resize/wait；
- CUDA 捕获异常会安全结束捕获、销毁半成品并对该 Engine 永久降级，不遗留
  非法 capture state；
- 每个嵌套 WHILE 前显式重设条件句柄。仅依赖
  `cudaGraphCondAssignDefault` 会使第二个外层迭代沿用前一轮的终止零值，这一
  CUDA 条件图陷阱已由连续多帧 gate 覆盖。

当前 whole-frame 候选故意只覆盖经过验证的快路径：merged、静态边界、
无 soft target/FEM pin、`skip_all_collision=True`、非 semi-implicit、
单 animation substep，且所有 isolated/strict/per-env overlay 关闭。不满足时
不伪装成功：普通 `step()` 回退到已审计的两图事务路径；捕获失败也只影响当前
Engine 的 whole-frame 候选。ABD 普通逐帧 whole-frame 仍关闭，ABD 支持由
Phase D episode API 显式进入。会执行宿主同步/读回的诊断旋钮（mirror/slot
audit、phase timer、KSUM、MAS dump/validate、stack diagnostic）同样在资格
检查阶段明确拒绝，避免让审计逻辑进入 CUDA capture。

验收入口：

```bash
STIFFGIPC_NATIVE_DIR="$PWD/build" python3 scripts/frame_graph_gate.py
```

该 gate 覆盖 warm-up 降级、正常两图事务、强制逐位回滚、ABD unique-tier
边界增长/重试、重试耗尽、host-audit 请求下的安全降级，以及 whole-frame
开关前后的逐位指纹。当前结果：`FRAME-GRAPH-GATE: PASS`。

## Phase D 落地战报：RL episode 驻留（2026-07-28）

高层 Python API 已加入 `stiff_physics.engine.Engine`：

```python
engine.step()  # 一次同步 warm-up，训练所有 lazy workspace/tier
engine.launch_episode_async(
    frames,
    revolute_actions,   # (frames, revolute_joints, 3)
    prismatic_actions,  # (frames, prismatic_joints, 3)
)

for slot in (0, 1):
    engine.wait_episode_observation(slot)
    observation = engine.get_episode_observation(slot)

successful_frames = engine.finish_episode()
```

动作三元组分别是 `{target, strength, external torque/force}`。两类关节动作先在
宿主打包成一个连续块，整个 episode 只做一次动作序列 H2D；设备上的
`frame_index` 直接索引该序列，逐帧不再回到宿主更新 joint control。

episode 是一次 root graph launch。由于 CUDA 不允许把 host-queryable external
event node 放进 conditional body，root 合法地组织成“两段 WHILE + 两个 root
观测出口”：第一段完成后异步复制 slot 0，第二段完成后异步复制 slot 1。每个
slot 都包含逐帧 positions、velocities、`FrameStatus` 和 attempted-frame 计数，
复制目标是 pinned host memory，并由 external CUDA event 围栏发布。调用方可用
`episode_observation_ready(slot)` 无阻塞轮询，也可只等待需要的 slot。两段仍属于
同一次 root graph launch，episode 内 `host_boundaries == 0`。

任一帧失败会在设备侧停止后续帧，失败帧先回滚到该帧入口状态再发布状态包；
`finish_episode()` 返回成功提交的帧数，调用方可据逐帧状态丢弃 poisoned
episode。销毁/reset 正在运行的 episode 时会先安全收束 stream，再释放借用的
frame snapshot。

可执行图复用策略保持诚实：

- 纯 FEM、形状和 buffer generation 不变时，后续 episode 复用同一个 exec；
- ABD 的中间 converter offset 会随已提交的 q/layout 状态前进，因此当前在
  **episode 边界**重录，绝不在 episode 内重录或同步；
- 不满足 C-3 候选约束、未 warm-up、动作 shape 不符、存在
  `STIFF_DRIVE_SUBSTEP>1` 等情况会明确抛错，不偷偷走逐帧宿主循环。

验收入口：

```bash
STIFFGIPC_NATIVE_DIR="$PWD/build" python3 scripts/episode_graph_gate.py
STIFFGIPC_NATIVE_DIR="$PWD/build" python3 scripts/episode_graph_rl_gate.py
```

第一项连续运行两个纯 FEM episode（默认共 12 帧），覆盖 exec 复用、两个异步
slot、一次发射/零边界计数，并与普通逐帧 whole-frame 路径逐字节一致。第二项
构造 fixed+revolute+prismatic 的 ABD 铰接场景，把 6 帧动作分成两个 episode，
核对逐帧 Newton/PCG/status/观测。ABD 并行归约在普通基线自身就有约
`1e-17` 位置、`1e-15` 速度的末位抖动，因此 gate 先跑两次普通基线建立噪声
包络，再要求 episode 误差不超过该包络和机器精度下限；纯 FEM 的严格逐位 gate
不放宽。当前结果分别为 `EPISODE-GRAPH-GATE: PASS` 和
`EPISODE-RL-GRAPH-GATE: PASS`。

## Phase D 终态落地战报：GPU-native RL 设备 ABI（2026-07-29）

episode API 每段仍付一次动作 H2D 和 pinned 观测 D2H。GPU-native RL 模式把
最后这两笔也删掉：动作、观测、状态、帧计数全部只存在于设备指针后面，宿主在
稳态循环里**一次传输都不做**——真正的闭环（policy 读观测 → 写动作）整体留在
GPU 上，宿主只负责把图发射进流。

```python
engine.step()                       # 一次 warm-up，训练 lazy workspace/tier
engine.prepare_gpu_rl()             # 捕获可复用的一帧图（setup 边界）
abi = engine.get_gpu_rl_device_abi()  # 裸设备指针 + 图审计

for _ in range(horizon):            # 稳态：零宿主同步、零 H2D/D2H
    policy_kernel(abi["positions"], abi["velocities"],
                  abi["revolute_actions"], abi["prismatic_actions"])
    engine.launch_gpu_rl_async(stream)

engine.synchronize_gpu_rl()         # 显式 debug/teardown 边界
engine.end_gpu_rl()                 # 释放并恢复普通 step()
```

契约要点：

- `prepare_gpu_rl()` 是 setup 边界（分配/捕获/上传/同步都允许），复用
  episode 捕获管线但走 `device_native` 分支：跳过 input H2D 与两个 pinned
  slot D2H，图尾改为设备侧 `frame_counter += attempted_frames`；
  **捕获后审计硬性要求 0 host / 0 H2D / 0 D2H 节点**，不满足直接抛错。
- 动作缓冲是紧排 `(joints, 3)` float64（static_assert 钉死布局），语义与
  episode 动作相同：{target, strength, external torque/force}；单帧图恒消费
  slot 0，policy 每步覆写即可。观测 = 提交后的 positions/velocities D2D
  副本 + 208B `FrameStatus` 包 + int64 帧计数，全在设备。
- `frame_id` 在 `PATH_GPU_NATIVE_RL` 下由设备帧计数器推进，跨发射单调；
  失败帧仍走图内回滚 + status 发布，policy 侧读 `result`/`error_code`
  即可丢弃 poisoned 轨迹——检疫裁决权从帧尾宿主移到了消费者。
- 流亲和是 ABI 的一部分：首次发射绑定流，换流必须先 `end_gpu_rl()`；
  `step()`/`launch_episode_async()` 在模式内被锁定，`end_gpu_rl()` 同步、
  释放并恢复普通逐帧路径。
- 资格约束与 C-3 相同（merged、静态边界、单 substep 等）；ABD 铰接场景由
  闭环 gate 实证一帧图跨发射复用合法（converter extents 已设备驻留，
  容量溢出会以 status 错误显形而非静默错误）。

验收入口：

```bash
STIFFGIPC_NATIVE_DIR="$PWD/build" python3 scripts/gpu_rl_gate.py
```

gate 用 cudart memcpy 扮演 GPU policy（写动作/读观测的角色与 torch kernel
等价），双基线建立噪声包络后分两相验证：parity 相（逐发射同步）逐帧对照
公开 setter 基线；zero-sync 相 back-to-back 发射、全程无宿主等待，末尾一次
event 同步后核对帧计数与终态，并断言 stored 观测与引擎提交态逐位相等。
负向断言覆盖 step/episode 锁定、流亲和拒绝、`end_gpu_rl()` 后复活。
4090 实测（fixed+revolute+prismatic ABD 场景）：图 551 节点、**h2d=0、
d2h=0**；parity/final 的位置与速度误差全部 `0.0e+00`（低于基线自身
`1e-17`~`1e-15` 的 ABD 归约抖动包络）。`GPU-RL-GATE: PASS`，套件段位
G17c（`scripts/verify_gates.sh`）；D2D 动作发布变体 =
`scripts/gpu_native_rl_gate.py`（G17d）。

诚实完成度：该 ABI 是 GPU-native RL 的**地基块**，资格约束仍与 C-3 相同
（`skip_all_collision=True` 等）。到"接触丰富场景 + 设备 reward/done/reset +
批量环境 + A800 nsys 证明"的完整定义与四个完成块（C4/D2/D3/D4）见
`docs/GPU_NATIVE_RL_PLAN.md`——四块全过之前，Phase C/D 不标记为最终完成。

## C5 落地战报：isolated 模式整帧图（2026-07-29）

`STIFF_C5_ISOLATED_GRAPH=1`（默认关，且要求 C4 碰撞图同开）让**完整 isolated
bundle**（7 开关）整帧进图。宿主在 isolated 下的逐迭代参与全部消除：

| 原宿主参与 | 设备化方式 |
|---|---|
| 12B `cnt` D2H → `all_env_frozen` → Newton 退出 | `_perenv_newton_decide` 写 `FrameDeviceState::newton_converged`；`_newton_step_predicate` 照常消费 |
| 8B S3 决策 D2H/轮 → 回溯环控制 | S3 录成条件 WHILE：`_s3_round_begin` → per-env step → 重检测 → per-env 能量 → `_s3_decide`（原地减半）→ `_s3_tail_conditional` 置句柄 |
| 4B ground-trial D2H/轮 | `_markGroundTrialInvalid` 的标志字直接门控 `_s3_decide`，并由尾核决定减半重试还是回退 |
| 72B CCD 标量链 + `ccd_cnt` 网格 | 复用 C4 合并链（容量网格 + `_cpNum` 设备计数尾参进 `_per_env_selfAlpha_min`/`_per_env_alpha_compute`） |
| 8B κ 包络 D2H | C4-b 的图内 κ 链 |

**关键结构决策：图内录制合并树 + emission 级跨环境过滤，而非 per-env 树。**
per-env 树是"宿主 env 循环 + 变长发射 + 循环中改写 BVH 对象成员"，结构上不可
capture；而 `set_self_p2g`（isolated 下由 `decouple_thresh` 常开）在发射点滤掉
跨环境对，**产生的接触对集合与 per-env 树路径相同**——树级隔离只是实现手段，
隔离承诺由过滤保证。G19 实测证实：图路径跨环境耦合 1.791e-07，per-env 树基线
1.796e-07，实质相同。

宿主 parity 细节（两处必须照抄否则物理不等价）：
- **S3 耗尽 / ground 塌陷 → 回退而非致命**：宿主在 per-env 搜索无法让所有 env
  下降时会跌落到统一步长线搜索。图内录成 `IF(fallback){统一 LS WHILE}`，且
  回退调用带 `save_temp=false`——宿主也是在 S3 **之前**只存一次 temp，两条搜索
  共用同一起点。
- **边界预热合并管线**：`train_perenv_graph_capacities` 在帧边界跑一次合并
  build+detect。两个理由都会在捕获内致命：合并树的惰性排序 scratch 从未按全量
  N 定尺；EE/canon/self-p2g 等设备符号设置器是**值缓存**的，per-env 与合并路径
  发布的值不同，切换后首次调用会发同步 `cudaMemcpyToSymbol`。

顺带修复：`cal_abd_energy.cu` 的 per-env 能量 bin 用的是同步 `cudaMemset`
（捕获非法），改 stream-ordered。

**strict 仍被资格拒绝**（用户 2026-07-29 决定：strict 不图化）：strict 的承诺
是逐位复现，而容量网格归约合法重排求和序——准入 strict 等于换锚战役。
验收=G19（`scripts/isolated_graph_gate.py`）。

## A800 实测（2026-07-29，sm_80 就地构建 a2c8b33）

容器回收后从零重引导（cmake/ninja/pybind11/numpy 以 wheel 离线解包、
`BUILD_GL_VIEWER=OFF` 无头配置、Release+DLTO+SNK1、
`CMAKE_CUDA_ARCHITECTURES=80`，128 核约 7 分钟）。驱动/运行时=12.8
（≥12.4，条件节点完整支持——蓝图开放问题就此关闭）。判决：

| 项 | 结果 |
|---|---|
| strict 锚 | **PASS `0544461bd82123ae`**（与 4090 单一金值跨架构逐位同值） |
| frame-graph / episode-graph / episode-rl / gpu-rl / gpu-native-rl | **五 gate 全 PASS** |
| gpu-rl 数值 | A800 上该场景基线自身逐位确定（noise=0），GPU-native 路径**逐位相等**（error 全 0.0e+00） |
| nsys 稳态证明 | **PASS：h2d_rows=0 d2h_rows=0 sync_rows=0**（cudaProfilerApi 捕获窗 40 步） |

nsys 捕获窗内 API 分布：40 次 `cudaGraphLaunch`（92.4% API 时间，均值
~109µs 入队）+ 40 笔 D2D 动作发布（共 76µs）——宿主每步的全部工作就是把
一张 551 节点图推进流。闭环 43 帧（3 warm + 40 稳态）全 `result=0`、
`frame_id` 单调至 42。D4 余项=接触丰富负载、设备 policy 集成、长时程/
显存高水位/吞吐延迟度量（见完成块清单）。

## C4-a 落地战报：碰撞进整帧图（2026-07-29）

`STIFF_C4_COLLISION_GRAPH=1`（默认关）把碰撞链整体录进整帧条件图：帧序幕
buildBVH+buildCP（defer 计数模式）、Newton 体内 DCD 快照、`m_ccd_alpha_slots`
全设备 CCD 标量链（初始 ground/self 归约→swept BVH+buildFullCP→CFL+refined
容量网格归约→combine 图内裁决），以及消费真实 barrier/ground 能量的回溯 LS。
无 fallback：碰到不合格场景仍诚实回退，但合格场景**每帧一次 root launch 全
程含碰撞**。

四条工程线支撑：

- **录制期容量镜像**：录制时把 `h_cpNum/h_gpNum` 镜像临时置为满容量
  （tier(MAX_PAIRS)），使装配 extents、triplet offset 步进、能量 launch
  bound 全部按容量成形——否则从 dcd=0 的帧录出的图把接触烤没（Hessian 看
  不见接触而能量看得见 → 首接触帧 LS 活锁，Newton 顶格 1000 迭代实证）。
  能量核活计数改读 `m_pair_snap_cur`（DCD 时刻快照），不再读会被 buildFullCP
  改写的 `_cpNum` 活槽。
- **捕获前干跑训练**：显式 tier 数学（triplet/staging/hash/radix-sort
  workspace/reduce scratch/DCD 快照满容量化）之后，以容量镜像真跑一次
  `computeGradientAndHessian`+`calculateMovingDirection`——converter staging、
  muda 排序 temp、预条件 per-level 缓冲在各自真实路径上于捕获外长到位（三
  次 error-900 捕获期分配逐一实证后收敛到此方案）。干跑必须在
  `arm_full_graph_attempt` 之后（要走与录制相同的 txn 分区分支）；宿主记账
  由快照机制回滚，stats JSON 上下文同样备份恢复。
- **图内溢出裁决**：`_ccd_final_alpha_combine` 增容量参数（swept 超容=
  OVF_CCD_PAIRS→FRAME_RETRY），Newton 环出口 `_pair_tier_guard` 对照录制
  tier 检查五路对计数（OVF_DCD_PAIRS→RETRY）；帧边界 finish 处刷新镜像
  并在 tier 跨越时销毁 exec 强制重录（mask 只能向下，不能向上 launch）。
- **合法漂移档**：容量网格 sum 归约相对精确计数归约合法重排浮点结合序，
  实测位置 ~2e-13/速度 ~1e-11；G18 以 1e-9 相对容差裁决（真接触 bug 是
  1e-3+ 级，六个数量级余量）。

**C4-a 契约**（资格检查强制）：kappa 与 close-set 冻结在帧边界值（图内不跑
postLineSearch）→ sync 等价只在 kappa-quiet 窗口主张；摩擦系数必须为零
（摩擦集只在同步帧重建，非零摩擦被资格拒绝——C4-b/c 的目标=图内 kappa
自适应与摩擦 lagged 集设备重建）。验收=G18（`scripts/collision_graph_gate.py`）。
