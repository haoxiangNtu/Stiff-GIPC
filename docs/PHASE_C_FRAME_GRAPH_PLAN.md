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
