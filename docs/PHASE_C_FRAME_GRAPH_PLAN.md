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
