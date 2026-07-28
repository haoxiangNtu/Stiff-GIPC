# Phase A：全 GPU 驻留战役的定量依据（A800 nsys 剖析，2026-07-28）

场景：ModelScope 盘子 228 帧驻场 episode，dlto rc2 wheel，A800（sm_80）。
工件：/root/workspace/plate_nsys_{merged,isolated}.nsys-rep + stats 日志。

## 主账本（merged）

| 量 | 值 | 折合每帧 |
|---|---|---|
| 仿真墙钟（profiler 下） | 29.2s | 128ms |
| **GPU 内核忙碌总计** | **≈7.0s** | ≈31ms |
| **GPU 等主机（差额）** | **≈22s ≈ 75%** | ≈97ms |
| 阻塞 cudaMemcpy | 24.3s API / 31,475 次 | **138 次/帧** |
| cudaMemcpyToSymbol | 1.2s / 30,029 次 | 132 次/帧 |
| cudaStreamSynchronize | 1.2s / 34,412 次 | 151 次/帧 |
| cudaLaunchKernel | 1.0s / **341,867 次** | **1,500 次/帧** |

阻塞 memcpy 是事实同步点（其耗时含隐式排空等待）：主机每帧把 GPU 排空 ~138 次。
**全驻留（Phase B/C）在此机的理论上限 ≈ 3-4×**（29.2s → ~8-9s，若 GPU 不再排空）。

## 内核侧结构（次要发现）

- 广相查询家族占内核时间 38%+（_selfQuery_ee_lb2 16.8% 领跑、ee_ccd 11.0%、
  vf 10.4%、vf_ccd 6.0%）——EE_LB=2 已在 merged 生效。
- cub RadixSort 47,896 实例（≈210 次/帧，BVH 每帧全重建的排序）——图化时随帧图
  一起驻留，launch 开销同灭。

## 结论 → Phase B/C 靶点排序

1. 138 次/帧阻塞拷贝：计数/α/能量/收敛读回 → 设备驻留谓词（设备线搜索、PCG 自
   尾图、设备 CCD 链三个既有模式连环）。
2. 132 次/帧 ToSymbol：EE 门族每 buildCP 重发布 → 描述符战役 phase-1 的
   `__constant__` 合并单写正好同解（两战役在此汇合）。
3. 1,500 次/帧发射：整帧 CUDA Graph（CUDA 12.4+ 条件节点）一次发射吃掉全部。

isolated 对照：wall 22.9s、newton 545——结构同病，比例相近。

---

# Phase B 现状与设计（2026-07-28 勘察）

## 勘察发现

- **B1（设备线搜索默认化）已是既成事实**：`STIFF_DEVICE_LINESEARCH` 语义为
  非"0"即开（ipc_solver ~41-43）——每 trial 的能量读回早已消灭，每 trial 仅回
  传 1 个决策 int。靶单中它出局。
- **PCG 设备自尾图无罪**：每次求解仅 1 次状态读回（graph_state，~771），不是
  每批。但**每次求解都重新 stream-capture 整张图**（K×body ≈ 百余次发射被反复
  录制 + ExecUpdate + Upload，×~3.4 求解/帧）——1,500 次/帧发射的主要成分之一。
- 138 次/帧阻塞拷贝为长尾结构：per-trial 决策 int、per-buildCP 计数块+
  _gdCollapse、alpha 链标量族（ground/self/refine）、minMax 距离对、kappa、
  检测计数（06 wrappers ×4 D2H）等 ~15 个散点 × 各自倍率。

## B2'：捕获缓存（设计+风险边界）

签名=捕获参数全集 {x/r/z/p/dirs 指针、d_rz/d_break/d_graph_state、n、K、
max_iter、tol 位} + **预条件器缓冲代际计数**。风险：body() 经
apply_preconditioner 吸入 MAS 内部指针（mRbin/mZbin 有增长路径 05:1241、
matbin 每聚合重发布 05:402）——签名漏掉任一指针生命周期，脏命中=越界。
**前置条件：body() 捕获指针族的完整生命周期审计**（枚举每个被捕获参数的
alloc/realloc/publish 点，接显式失效钩子）。审计完成前不实施。

## B3：迭代状态块（设计）

设备侧 struct IterStatus { alpha_ground, alpha_self, alpha_refine, minDist2_g,
minDist2_s, gd_collapse, ls_decision, dirnan_any, ... }，各生产者内核就地写入，
每牛顿迭代**一次** ~64B 读回替代 ~10 个散点 memcpy。改动面=每个散点的消费端
从"独立读回"改为"读状态块字段"，语义逐点位级等价可证。与描述符战役（写侧
__constant__ 合并）互补成对：读侧状态块 + 写侧描述符 = 主机交互每迭代 O(1)。

## 实施顺序（下一轮）

1. B2' 前置审计（Explore 代理枚举捕获指针族生命周期）→ 实施+失效钩子。
2. B3 状态块（散点逐个迁移，每步套件+锚验证）。
3. 复剖析（nsys A800 对照 Phase A 基线：memcpy/帧、launch/帧、GPU 空转比）。

---

# B2' 前置审计结论（2026-07-28，全量指针生命周期普查）

- **朴素签名缓存必然高失效**：triplet_count（converter 每牛顿迭代 D2H 刷新，
  喂 spmv 网格）与 MAS totalNumberClusters（每迭代随接触重聚类，喂 memset
  长度+三处网格）皆 PSV。先决改造=**设备侧计数绑定**：spmv 按容量定网格+核内
  `if(tid>=*d_unique_key_number) return`（设备标量现成 global_matrix.h:255）；
  MAS memset/网格按 m_outputClusterCap 填充+设备簇数界。strict 走 binned 序无
  关沉积→网格改形位级中性；merged/isolated 在非确定性包络内（G9 看守）。
- **收益重估**：捕获+更新+上传 ≈2-3ms/帧（~2%）。**B3 迭代状态块升为主攻**
  （75% 空转在阻塞读回的排空语义）；B2' 缓存降为后续，实施时按审计的
  PcgGraphSig 结构体+三代际计数器（GIPCTripletMatrix::m_alloc_gen /
  MASPreconditioner::m_alloc_gen / PCGSolver::m_seg_alloc_gen，ABA 防护非可选）
  +STIFF_PCG_GRAPH_CACHE 开关+CACHE_VERIFY 自证模式。
- **捎回两个潜伏 bug（入错误分类学队列）**：①cub_temp 惰性分配后**从不增长**
  且 cub 返回码被丢弃（pcg_solver.cu:528-533）——n 增长时静默 InvalidValue；
  ②reserve_discard free-立即-malloc 的 ABA（global_matrix.h:81-83）。
  另：pcg_solver 文件级 getenv 静态族（s_graph_env 等）为进程稳定缓存，跨引擎
  语义已由 phase-0a 值追踪先例覆盖 s_seg_binned_host，余者低危。
- 完整表（每指针 owner/alloc/realloc 点/稳定性判定）见会话审计工件。

---

# B3 精确归因（2026-07-28，NVTX 相位 + sqlite 联查，4090 foldshirt 4env 20f）

每牛顿迭代阻塞拷贝census（~1000 迭代归一）：
linear_solve **5.4**（6.46s 总）> line_search **6.2**（2.82s）> ccd_alpha
**4.2**（3.98s）> GH_assembly 4.8 + **ToSymbol 5.4**（1.36s）。
**20s 墙钟中 13.3s 为阻塞拷贝**（4090 尚且如此，A800 按往返延迟放大）。

静态普查三次落空的教训：大头在被调函数内部（buildFullCP 的 ccd 计数镜像刷新、
MAS ReorderRealtime 读回、DeviceOut 助手隐藏读回），顶层代码只见 1 次
h_ccd_state。工具链已入库：STIFF_NVTX=1（默认零开销）+ 六相位区间 +
scratchpad sqlite 联查模板。

B3 手术顺序（按耗时）：①linear_solve 内部（converter 计数→设备侧绑定与 B2'
前置改造同解；MAS 重排读回合批）②ccd_alpha 的隐藏 3.2（buildFullCP 计数族
合批/延迟）③line_search 每 trial 族。每步：改后重跑本归因剖析对照 + 套件+锚。

---

# B3 子相位判决与解读修正（2026-07-28）

**解读修正**：阻塞拷贝的"耗时"含生产性等待（ls_pcg 每迭代 1 次 × 8.9ms =
等 PCG 图完成，GPU 在干活）。诚实浪费度量 = GPU 忙碌 vs 墙钟：**4090 忙
74%/闲 26%（14.9s/20.0s）；A800 闲 75%**——同结构，主机延迟是放大器。
手术优化目标因此是**往返次数**（每次在服务器付 ~百µs 空转），不是拷贝毫秒数。

每迭代阻塞往返 census（616 迭代归一）：
- **ls_mas_setup 8.0 次**（MAS 装配小读回族——count 之王，手术①头号）
- line_search 6.2 · GH ToSymbol 5.4（=描述符 phase-1 靶）· ccd 隐藏 3.2
- ls_convert 1.0（×1.1ms：unique-key 归约+计数刷新，半生产性）
- ls_pcg 1.0（×8.9ms 生产性等待，非靶）

合计 ~22 往返/迭代。A800 期望：每削一次往返 ≈ 省百µs 级空转 × 迭代数。
下一刀：MAS setPreconditioner/ReorderRealtime 内部 8 次读回的合批/设备驻留。

---

# 手术①战报：MAS 层级读回设备驻留（2026-07-28）

单环境路径（盘子/towel 类=ModelScope 与 RL 主战场）：ReorderRealtime 的
per-level h_clevelSize 读回（levelnum-1 次/迭代）消灭——层级尺寸本就由
_prefixSumLx 在设备上写出，主机副本纯属镜像。改造=_dev 内核变体核内读
d_levelSize[level]（容量网格；两核原生 inRange/无早退纪律=填充惰性由设计保证）
+ 定容 thrust scan（前缀缓冲每层零化到容量，填充尾扫描为常量）+ number<1
设备侧写穿。多环境（其环填充算术需主机 warp 计数）保留旧路。

指纹验证（8 字节 D2H × ls_mas_setup 相位）：towel 1env=**1.0/迭代**（原 4）；
foldshirt 4env=5.0（旧路,按设计）。锚 0544461bd82123ae 逐位、MAS oracle、
15 段、diag 三模式全绿。剩余 mas 段 ~3/迭代为其他站点（下一刀），line_search
6.2 与 GH ToSymbol 5.4（描述符）在队。多环境版设备绑定（环填充算术下设备）
为后续项。

---

# 手术②战报 + 手术③依赖链（2026-07-28）

**手术② ToSymbol 值缓存**：14 setter（八门族/emit_caps/envmajor/part/两指针/
bar_targets；reset 类设备写计数器刻意不缓存；租约下单写者=sound）。指纹：
GH 8426→全程 5、line_search 19805→**0**（约 20 次/迭代→0）。锚逐位、15 段绿。

**累计**：手术①+② 消灭 ~23 往返/迭代（原 census ~42 的过半）。

**手术③（line_search trial 块）侦察结论**：per-trial 的 {decision, cpNum[6],
gdCollapse} 三读合一受阻于**能量核网格消费 h_cpNum[0]**（decide 前主机已读计
数）——先决改造=能量核容量网格+设备计数界（_dev 同款；与 B2' 的 spmv 设备绑
定同族）。改造顺序：①能量/装配核 grid 设备绑定（同时解锁 B2' 缓存与 trial
合块）→②trial 三读合一→③gdCollapse 延迟消费（仅 line-search 路径，其余
buildCP 调用者保留即时校验；NaN→backtrack 语义已天然兜底）。ccd 隐藏 3.2 同
族（buildFullCP 计数镜像）。

---

# 手术③战报：trial 计数延迟 + 设备计数界（2026-07-28）

能量核容量网格改造落地（merged 路径）：barrier/ground 能量核 +`d_live` 尾参
（空=旧语义，全调用点零改动）；trial 期 buildCP 跳过计数 D2H（镜像保持
invalid，MIRROR_AUDIT 武装态即收口证明）；能量网格用迭代首界（×1.25+64 松
弛）+核内设备活计数；溢出走发射路径的**永增计数器**（trash 分支冷路径零热
成本）随 decision 同一次 8 字节读回捎带，触发即退回 legacy buildCP 复用全部
grow+redo 机器再重评（假阳性=良性重跑）。零填充归约位级中性（temp=0）。
strict/per-env 路径不启用（其管线自带刷新），锚逐位原位。

同场景对照（towel）：line_search 阻塞拷贝 **9.87→8.29/迭代**（-1.6=2 拷贝
×0.8 trial，精确吻合）；GH 7.88 不变。多 trial 场景收益按 2×trials/iter 放大。
解锁：trial 决策+溢出已合一读；余 gdCollapse 1/trial（下刀）；B2' 图缓存的
triplet_count PSV 同法可解；ccd 隐藏族同族。

---

# 手术④战报：gdCollapse 搭乘 decision 读回（2026-07-28）

塌陷响应从 `throwIfGroundDistanceInvalid()` 拆出 `handleGroundCollapse(int)`；
trial 期 buildCP 跳过 4 字节塌陷回读，`_global_ls_decide` 把 `_gdCollapse`
写进 status[2]（decision 缓冲 2→3 int，同一次读回捎带），宿主在**同一
trial**内消费——GeometryError 契约不变，NaN→backtrack 仍是天然前哨。非
trial 调用者（post-LS buildCP、FEM 包装器）保留即时阻塞校验。

验证：15 段绿，锚 0544461bd82123ae 逐位。towel（空中揉布 gpNum=0）按设计
无变化（GH 7.88 / ls 8.27）；foldshirt merged N=4 字节指纹：12B×769=新 3-int
decision 读（延迟激活证明），4B 族 806≈非 trial 人口（trial 期塌陷读若存活
应翻倍）。

**foldshirt 分解普查的意外收获**（新目标入列）：
- 48B×2/trial：buildBVH 根 AABB 回读（bvh_f+bvh_e 各一）→ 手术⑥
- ls_mas_setup 8B×5/iter：多环境 merged 走 MAS 层循环 legacy 分支（手术①只
  盖单环境）→ 多环境刀的精确账
- ls_convert 4B×1/iter：triplet_count（B2' 阻塞点实锤）
- ls_pcg 16B×1/iter：收敛判定（有效等待）

---

# 手术⑤战报：ccd 计数搭乘标量链读回（2026-07-28）

merged buildFullCP 不再阻塞刷新 h_ccd_cpNum：refined min 归约改容量网格+核内
活计数掩码（**min 精确无舍入 ⇒ 重排网格位级中性**；OOB 线程持恒等元 1.0，
统一尾部见全网格）；combine 核从设备计数器读配对门并把原始计数写进
slots[8]（标量链 8→9 double，同一次读回）。**过容量溢出信号=原始原子计数
本身**（trash 发射也计入，无需借永增计数器）：读回处发现 count>capacity 即
退回 legacy grow-redo 机器重算 refined 项再重读——子集 alpha 消费前即弃。
per-env 保留即时刷新；迭代内下游镜像读全部改用捎带值（审计武装收口）。

验证：15 段绿（MIRROR_AUDIT 全程），锚逐位。towel ccd_alpha **6.52→5.69/
迭代**，4B 刷新族**整族消失**；残余=48B×1.63（swept 根 AABB→⑥）+72B×0.81
（9-double 标量链=宿主决策读，Phase C 整帧图目标）。GH 对照 7.88 不变。

---

# 手术⑥战报：merged BVH 根 AABB 设备驻留（2026-07-28）

per-env 路径早备好的 `calcMaxBV_async`（设备写 `_bvs[0]`，morton 哈希核本就
从设备指针读）直接接管 merged Construct/ConstructFullCCD 全部 4 个热站——根
盒 48B 回读归零。**第一版翻车实录**：宿主 `scene` 成员靠阻塞版 Construct 的
副作用喂 `GIPC::init()` 的 dHat 推导；副作用一除，不走 getSceneSize 的流程
init 吃到未初始化值（bboxDiagSize2=1.2e65 → dHat=3.5e29 → 2200 万接触对 →
OOM/越界，一根线炸 7 段门禁；锚与 foldshirt 恰好走了另一条填充路径而幸免）。
修复 v2 更本质：init 改为 boot 期单次从设备 `_bvs[0]` 直读根盒，彻底解除对
构建副作用的暗依赖。

验证：15 段绿，锚 0544461bd82123ae 逐位。towel：**ccd_alpha 5.69→0.81/迭代**
（=纯 72B 标量链决策读，理论地板）、**line_search 8.25→3.27/迭代**；GH 7.88
不变（非 bbox 族，下一分解目标）。战役累计：ls 9.87→3.27，ccd 6.52→0.81。

---

# 手术⑦战报：摩擦期 gp 计数 D2D 恢复（2026-07-28）

摩擦 Hessian gd 路径每迭代把 h_gpNum_last 阻塞 H2D 回传 `_gpNum`——GH 相唯一
逐迭代 H2D。改为摩擦集构建时设备侧 stash（`m_scr_gp_friction`，boot memset 0；
此刻 `_cpNum+5` 恰持镜像快照值）+ 异步 D2D 恢复。时序审计：恢复后 `_gpNum`
持 last 值直到下次检测重写——与 legacy 全同；trial 能量核只在 trial buildCP
重写后读 `_cpNum+5`。towel：GH **7.88→6.88/迭代**（−1.0 精确），4B H2D 族消失，
实传输仅剩 16B contact start-ids（B2' 块）。ls 3.28 / ccd 0.82 对照不动。

---

# 手术⑧战报：多环境 MAS 层循环设备驻留（2026-07-28）

关键发现：`_mas_envSegN` 的 `warpNum % n_env` 整除性守卫使 segN **非**场景常
量（齐次批量恒成立，分歧层静默退全局 bank）——判定随 warpNum 整体搬进
`_mas_env_base_dev` 核内（写 `d_segwpe` 设备槽，apply/setx 自守卫），宿主只留
env-var/4096 静态解析（`_mas_envSegN_static`）。PrefixSumLx 多环境改容量扫描
+零填充（手术①已证位级中性）+ `_prefixSumLx_dev`；NextLevelCluster 多环境同
走 `_dev` 容量网格；ReorderRealtime 无条件容量 memset（396 else-读删除）。
L1 的 warpNum 是 totalNodes 派生（场景常量）保持宿主判定。

验证：15 段绿（mas-oracle strict 20-cube 4-level=本刀位级覆盖），锚逐位。
foldshirt merged N=4：**ls_mas_setup 8B×5/iter → 8B×1/iter**（−4，逐层回读全
灭；余=终层 totalNumberClusters 1 读 + 4B×1 未名读，双双归 B2' 块）。

---

# B2' 战报：跨 solve PCG 图缓存（2026-07-28）

三段落地：
- **a**（eba4871）：converter 设备侧 finalize 唯一键数（+1 上核，写入 07-24 事
  故后专位保留却从未被写的 `d_unique_key_number`），宿主镜像改从该槽 D2H；
  spmv 加 `d_live` 尾参——网格仍用等值镜像（今日逐位同），核内界读设备。
- **b-1**（77b4fe2）：三个 SchwarzLocalXSym 核 +`(d_lvl,levelnum)` 尾参核内取
  `d_levelSize[levelnum].y`；apply 路径 4 个 memset 改 `m_outputClusterCap` 容
  量字节数（活区等同、跨 Newton 稳定）。
- **b-2**：`pcg_capacity_mode.h` 双全局（容量网格模式 + 单一缓冲代数计数器，
  gt 三处 realloc + MAS ensureOutputClusterCapacity 皆 bump）；签名
  {generation, dofs, K, tol_bits} 命中即跳过逐 solve 重录（录制窗口内网格切容
  量派生）；`STIFF_PCG_GRAPH_CACHE`（默认 0）+`STIFF_PCG_CACHE_VERIFY`（强制
  重录，existing exec-update 路径持续证明拓扑不漂移）双知识库行。strict 因
  `!STIFF_SPMV_DET` 门控从不捕获——锚免疫本构。

验证：默认套件 15 段绿锚逐位；cache=1 towel/foldshirt merged 双 HIT+PASS；
VERIFY 模式如期无 HIT。桌面 4090 towel 墙钟 7.48→7.70s（小场景容量网格开销
≈录制节省；本刀目标=A800 慢主机链路场景，默认关、待 A800 实测定夺翻转）。

---

# Phase C-1 战报：LS 试探环设备自尾launch 图（2026-07-28）

trial 体（halve→step→BVH→CP(defer)→energy→decide→tail）录成 DeviceLaunch 图，
decide 尾核未收敛即 `cudaGraphLaunch(cudaGetCurrentGraphExec(), TailLaunch)`
自我重launch（PCG device-loop 先例复用，无需条件节点）；宿主出环单次 6-int 打包读
{decision, overflow, collapse, trials, alpha_bits}。跨 LS/帧缓存于
pcg_buffer_generation 签名；`STIFF_LS_GRAPH`（默认 0）+DIAG 行。

**七连环调试实录**（每环都以工具实证收口）：
1. muda `DeviceBuffer::resize` 无条件 `.wait()` → 能量路径 6 处尺寸守卫；
2. 同步 memcpy 捕获非法 → snapshot 改设备计数核（容量网格+`_cpNum` 掩码）；
3. **c1m 逐迭代 / κ 逐帧被烤死**（E1 用陈旧 Armijo 斜率与 κ 评判→64 减半全烧）
   → `_ls_seed` 按值下发 {c1m, κ} 至设备槽，decide/combine/barrier/ground 核
   读 `c1m_dev/kappa_dev`；
4. 能量网格界烤死 → 录制期容量网格（barrier=MAX_PAIRS、ground/gd=surfN、
   fric=cap；`m_ls_recording` 开关 sizing）；
5. sum 尾部末块 warpNum 负值跳写 → d_live 时尾部按全网格参与（0 填充和中性）；
6. 摩擦 lastH 族 `resize_discard` 无代数 bump → 悬垂（sanitizer:
   `_getFrictionEnergy_gd` 越界读 308B cap 后 1 字节实锤）→ ensure_frictionBuffers
   + snapshot grow 补 bump；
7. **stash 从设备 `_cpNum+5` 原始拷贝与镜像脱钩**（LS 出口刷新后设备槽被后续
   检测漂移，缓冲却按镜像尺寸增长）→ `_stash_fric_counts` 镜像值按值下发。

验证：15 段绿锚 0544461bd82123ae 逐位（默认关）；knob=1 towel 全程
PASS/0 耗尽/drift 1.94e-3（健康区间）、foldshirt merged N=4 图激活 PASS；
指纹=逐 trial 12B 决策读消失（余：首评 1 读+出环 24B 打包+设计内出口刷新）。
副产物：thrust→cub 排序（a606c79）、摩擦计数镜像化根治一处真实潜伏竞态。

---

# C-2 ②/Phase D 战报 + 统一验证战役（2026-07-28）

**C-2 ②**（7bc910a）：GH 起点快照扩 12-int {contact_total, fricgd_start,
ground_start, rsv, cd4..2, f4..2}；soft/barrier 融合装配（活类型计数——其核
自身向 _cpNum 再发射，消费者必须用快照）/摩擦 cp（6 值参→双紧凑指针视图）/
摩擦 gd/ground（设备段基址）全链尾参；`_c2_offset_dev()` 宏上移 GIPC.cuh
（12 在 13 前含入）。converter 容量化（pad-sentinel 键语义）列入 ③ 蓝图。

**Phase D 原语**（eaa6e39）：`fetch_obs_async(slot)` 顶点缓冲 D2H 进钉页双缓
冲（专用非阻塞流，PTDS 事件栅栏定序——宿主永不同步仿真流）；
`obs_ready/obs_wait/get_obs`（零拷贝 numpy 视图）；`upload_episode_actions`
整段一次 H2D 返回设备指针（gpu-direct 行消费）；`engine.episode_loop` =
标准零停顿环（第 i 帧观测回读与第 i+1 帧计算重叠）。

**统一验证战役**（实现先行后的批量收口）：
- 15 段门禁绿，锚 0544461bd82123ae 逐位；
- towel 三 knob 齐开（LS_GRAPH+C2_OFFSET_DEV+PCG_GRAPH_CACHE）：PASS、
  offset 对账 0 分歧、0 耗尽、drift 1.942e-3（健康带）；
- foldshirt LS+CACHE：PASS、0 耗尽；
- episode API 冒烟（examples/episode_obs_smoke.py）：10 帧零拷贝观测、
  动作预上传、布料坠落物理正常。

**至 Phase C-3 全图的剩余蓝图**（前置全部就绪）：converter pad-sentinel 容量
化 → ext 段设备递延 → Newton WHILE 尾launch（图内设备重算 offset 快照替换
镜像下发）→ 帧图。A800 部署实测裁决四 knob 默认值。
