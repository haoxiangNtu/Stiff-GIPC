# StiffGIPC 手册 · 原理分册:求解、并行与执行通道

> **PRINCIPLES_EXECUTION** — 线性系统装配与归并、PCG/MAS 预条件、多环境三模式的实现机制、
> 确定性栈与金锚、帧事务与整帧 CUDA Graph、episode 图与 GPU 驻留 RL、执行通道选型、诊断工具箱。

---

## 目录

- [0. 适用版本与标注约定](#0-适用版本与标注约定)
- [1. 线性系统:triplet 装配、unique-block 归并与容量档](#1-线性系统triplet-装配unique-block-归并与容量档)
  - [1.1 两层求解栈](#11-两层求解栈)
  - [1.2 solve_linear_system 流程](#12-solve_linear_system-流程)
  - [1.3 GIPCTripletMatrix:SymGH 块上三角存储与增长合同](#13-gipctripletmatrixsymgh-块上三角存储与增长合同)
  - [1.4 装配流布局与接触四类分区](#14-装配流布局与接触四类分区)
  - [1.5 Converter:unique-block 归并](#15-converterunique-block-归并)
  - [1.6 容量档 assembly_capacity_tier](#16-容量档-assembly_capacity_tier)
- [2. PCG 与 MAS 预条件](#2-pcg-与-mas-预条件)
  - [2.1 PCGSolver 配置与收敛判据](#21-pcgsolver-配置与收敛判据)
  - [2.2 迭代体与三级执行策略](#22-迭代体与三级执行策略)
  - [2.3 分段(块对角)PCG:seg_pcg](#23-分段块对角pcgseg_pcg)
  - [2.4 预条件器框架与 P_type 分支](#24-预条件器框架与-p_type-分支)
  - [2.5 MASPreconditioner:多级加性 Schwarz](#25-maspreconditioner多级加性-schwarz)
  - [2.6 METIS 排序(MAS 前置)](#26-metis-排序mas-前置)
  - [2.7 实测:MAS 应用占比与 diag A/B 结论](#27-实测mas-应用占比与-diag-ab-结论)
- [3. 多环境实现机制](#3-多环境实现机制)
  - [3.1 两轴三模式与 flag bundle](#31-两轴三模式与-flag-bundle)
  - [3.2 merged:合场景与"为何非 run-to-run 确定"](#32-merged合场景与为何非-run-to-run-确定)
  - [3.3 isolated:per-env 束与检疫铁律](#33-isolatedper-env-束与检疫铁律)
  - [3.4 strict:跨环境逐位机制](#34-strict跨环境逐位机制)
  - [3.5 S4 active mask 与线性系统掩码](#35-s4-active-mask-与线性系统掩码)
- [4. 确定性与门禁](#4-确定性与门禁)
  - [4.1 det 栈原理:binned 指数分箱归约](#41-det-栈原理binned-指数分箱归约)
  - [4.2 金锚(bitwise anchor)机制](#42-金锚bitwise-anchor机制)
  - [4.3 strict 锚:跨架构与批次不变性的契约地位](#43-strict-锚跨架构与批次不变性的契约地位)
  - [4.4 图门禁的两层验证](#44-图门禁的两层验证)
  - [4.5 checkpoint 与确定性](#45-checkpoint-与确定性)
- [5. 帧事务与整帧 CUDA Graph【仅 phase-cd】](#5-帧事务与整帧-cuda-graph仅-phase-cd)
  - [5.1 两图事务(root + terminal)](#51-两图事务root--terminal)
  - [5.2 整帧图(whole-frame conditional graph)](#52-整帧图whole-frame-conditional-graph)
  - [5.3 设备快照集:fem + abd + kappa](#53-设备快照集fem--abd--kappa)
  - [5.4 OVF 回退协议:尝试→回滚→宿主重解→扩容→重录](#54-ovf-回退协议尝试回滚宿主重解扩容重录)
  - [5.5 容量档训练与 headroom](#55-容量档训练与-headroom)
- [6. episode 图与 GPU 驻留 RL【仅 phase-cd】](#6-episode-图与-gpu-驻留-rl仅-phase-cd)
  - [6.1 两种形态与 GPU-native 契约](#61-两种形态与-gpu-native-契约)
  - [6.2 动作预上载与观测双缓冲(split-frame)](#62-动作预上载与观测双缓冲split-frame)
  - [6.3 API 面(GIPC / SimEngine / Python)](#63-api-面gipc--simengine--python)
  - [6.4 零 D2H 契约的实证](#64-零-d2h-契约的实证)
- [7. 执行通道选型结论](#7-执行通道选型结论)
- [8. 诊断工具箱](#8-诊断工具箱)
- [附录 A. 本分册环境变量速查](#附录-a-本分册环境变量速查)

---

## 0. 适用版本与标注约定

本手册覆盖两条产品线:

| 线 | 仓库/分支 | 版本锚点 | 布局 |
|---|---|---|---|
| **稳定线** | `Stiff-GIPC-stable-08`,分支 `release/stable-0.8` | HEAD `b8e27a1` = tag **v0.8.5.3**(2026-08-11 发布;公开仓 `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel,CUDA 架构 sm_80/89/120) | v0.8.6 重构前布局:`StiffGIPC/GIPC.cu` 为 16884 行单体、`MASPreconditioner.cu` 为 3126 行单体;但 `linear_system/` 模块目录(PCG/预条件器抽象)在稳定线**已存在** |
| **工程线(phase-cd)** | `Stiff-GIPC-c1-ls-graph`,分支 `codex/phase-cd` | HEAD `b3ab747` | v0.8.6 模块化重构(`gipc_modules/`、`mas_modules/`、`frame_fsm/` 等复合 TU 拆分)+ 整帧 CUDA Graph、GPU 驻留 RL、episode、checkpoint v2 等全部 v0.8.5 后工作 |

标注约定(每个 API/特性/旋钮后缀):

- **【稳定线+phase-cd】** 两线都有;
- **【仅 phase-cd】** 工程线专属(v0.8.6 重构线产物);
- **【仅稳定线】** 只进了 v0.8.5.3(如 tactile 线的两个修复);
- **【实验性,默认关】** 存在但默认不启用,须显式旋钮打开。

行号约定:`文件:行` 未注明线时指 **phase-cd @ b3ab747**;注明"稳定线"时指 v0.8.5.3(b8e27a1)。
本分册在稳定线上核对代码时使用的工作树 checkout 位于 v0.8.5.3 之后数个提交,经
`git diff --stat b8e27a1..HEAD` 亲验共 8 个文件(GIPC.cu/GIPC.cuh/sim_engine.cu/.h/
engine.py/bindings/pystiffgipc.cu/CHANGELOG.md/pyproject.toml),内容为摩擦锚/静摩擦线,
其中含正式发布提交 `0894958` **release(v0.8.5.4): default-on true static friction**
——一个默认行为变更的新版本,不只是零散修复。但这些提交**不触及**本分册讨论的
`linear_system/`、MAS、多环境机制——故文中"稳定线也有"的判定对 v0.8.5.3 成立;
摩擦默认行为以本分册锚点 v0.8.5.3 为准(v0.8.5.4 的 static friction 默认开不在其内)。

> **重要差异(tactile 线)**:接触力读数中摩擦分量恒零的修复、`reset_transient_contact_state()`
> API,**只进了稳定线 v0.8.5.3**,phase-cd 尚未移植——phase-cd 上
> `get_vertex_contact_forces` 的 `friction_lagged`/`total` 分量仍受恒零 bug 影响。
> 详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

实测数字的权威出处:[../OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md)(2026-08-11 成本结构)、
[../SIMULATOR_EXECUTION_DESIGN.md](../SIMULATOR_EXECUTION_DESIGN.md)(执行架构定稿与 v0.8.5 横测)、
[../A800_ALLEXAMPLES_TIMING_2026-08-01.md](../A800_ALLEXAMPLES_TIMING_2026-08-01.md)(C6 系列战役全记录)。
本分册只引用,不重造。

---

## 1. 线性系统:triplet 装配、unique-block 归并与容量档

### 1.1 两层求解栈

**【稳定线+phase-cd】** 引擎内并存两层:

1. **legacy 壳层 `PCG_Data`**(`StiffGIPC/PCG_SOLVER.cuh:15-27`):只剩
   `double* squeue`(标量归约 scratch,尺寸 max(vertexNum, tetrahedraNum))、
   `double3* dx`(vertexNum)、`MASPreconditioner MP` 成员和 `int P_type = 1`
   (默认 1 = MAS,`PCG_SOLVER.cuh:22`)。真正的 PCG 循环**不在**这里;
   `FREE_DEVICE_MEM()` 在 P_type==1 时调 `MP.FreeMAS()`(`PCG_SOLVER.cu:34-37`)。
2. **现役求解栈 `gipc::GlobalLinearSystem`**(`linear_system/linear_system/global_linear_system.h:15`),
   聚合(注册点 `gipc/gipc.cu:188-215`):
   - 子系统:`ABDLinearSubsystem`(先注册,dof offset 0)+ `FEMLinearSubsystem`(在后);
   - 求解器:`gipc::PCGSolver`;
   - 预条件器:见 [§2.4](#24-预条件器框架与-p_type-分支)。

每个 Newton 迭代的入口:
`GIPC::calculateMovingDirection(device_TetraData&, int cpNum, int preconditioner_type)`
(`gipc_modules/14_energy_linesearch_solver.inl:362`)→
`m_global_linear_system->solve_linear_system()`(同文件 :406)。
注意 `cpNum`/`preconditioner_type` 形参在现行函数体内不再被使用(历史签名残留;
预条件器选择改由注册期 `P_type` 分支决定)。返回的 PCG 迭代数写入 statistics JSON
`["newton"].back()["pcg"]["iterations"]`(:422-427;录图时跳过)。

### 1.2 solve_linear_system 流程

`GlobalLinearSystem::solve_linear_system()`(`global_linear_system.cu:201-219`)每次调用依序执行:

| 步 | 动作 | 出处 |
|---|---|---|
| 1 | `build_linear_system()`:各子系统 `report_subsystem_info()`;对 RHS dof 做 exclusive_scan 得 `m_rhs_offset_per_subsystem`(ABD 在前,FEM 在后) | global_linear_system.cu:11-38 |
| 2 | 空系统(`global_triplet_offset==0 或 total_rhs_count==0`)打印并返回 false | :44-49 |
| 3 | `m_b/m_x.resize`,`m_x` memsetAsync 清零(PCG 初值必须有限) | :52-60 |
| 4 | **求解前容量保障** `ensure_capacity_preserve(length, 2*layout)`;layout 在 device-count 模式下取 `assembly_capacity_tier(length)`,否则取 length | :75-106 |
| 5 | 各子系统 `do_assemble(rhs_view)`(填 b + 把 Hessian 三元组写入全局 triplet 流);ABD 预条件器(preconditioner_id==0)在 convert **之前** assemble(块取自子系统自身数据) | :114-119 |
| 6 | `convert_new()`:triplet → BCOO 唯一块归并(§1.5) | :267-343 |
| 7 | 全局预条件器 assemble(若有);其余 local(含 MAS,id==1)在 convert **之后** assemble——MAS 消费归并后的唯一块 | :126-132 |
| 8 | `[S4]` 若装了 env mask:`_s4_zero_masked_rhs` 清零被屏蔽 env 的 b 分量 | :192-213 |
| 9 | `m_solver->solve(m_x, m_b)` | :215 |
| 10 | `distribute_solution()`:各子系统 `do_retrieve_solution`(无全设备同步) | :138-149 |

第 4 步从 `discard` 改为 `preserve` 是 towel-strict 崩溃(v0.8.5.1)的根因修复:
pre-solve 的销毁式增长会摧毁**活跃的**矩阵内容,SpMV 随即越界。
`ensure_capacity_preserve` 在 live 超容量时直接 throw(拒绝求解已损坏的矩阵),见 §1.3。

**公开签名**(`global_linear_system.h`):

```cpp
gipc::SizeT solve_linear_system();            // 返回 PCG 迭代数;空系统返回 0
void set_env_mask(const int* active, const int* dof_to_group, int ng);
bool arm_precond_seg_dot(double* partials, const int* d2g, int ng);
bool precond_graph_capturable() const;        // 全部预条件器都声明 graph_capturable() 才 true
void train_converter_capacity(int capacity);  // 【仅 phase-cd】图容量训练入口
```

### 1.3 GIPCTripletMatrix:SymGH 块上三角存储与增长合同

**【稳定线+phase-cd】** `linear_system/linear_system/global_matrix.h`。

- **SymGH 块级上三角存储恒开**(`global_matrix.h:8`,`#define SymGH`):每个接触对最多写
  `M12_Off=10` 个 3×3 块(12×12 的 4×4 块网格上三角 4·5/2)、`M9_Off=6`、`M6_Off=3`
  (:9-17;非 SymGH 时为 16/9/4)。相对全存储 **−37.5% triplets**(v0.8.2 引入,CHANGELOG)。
- 缓冲:`block_values`(`Eigen::Matrix3d`)/row/col + 归并 scratch(64 位 hash、sort hash、
  index、sort index、temp)(:36-43)。
- **增长合同**(两条路径,语义截然不同):

| API | 语义 | 约束 | 出处 |
|---|---|---|---|
| `ensure_capacity_discard(need)` | **销毁式**(free→malloc,无拷贝、无双驻留);margin = need×30%,上限 512MB | 唯一合法调用点 = 帧起点 offset 刚清零处;由单发窗口 `open_discard_window` + `assert_discard_window_or_throw` 审计强制(误用即 towel-strict SpMV OOB) | global_matrix.h:78-92, 171-178 |
| `ensure_capacity_preserve(live_count, need)` | 保留 `[0:live)` 数据;live > capacity 时 **throw**(runtime_error) | 求解前容量保障的唯一合法形式 | :101-133, 119-125 |

  两者都会 `++pcg_buffer_generation()`(指针可能搬动 → 使跨 solve 的 PCG 图缓存失效,§2.2)。
- **计数层**:`global_triplet_offset`(装配流长)、`d_unique_key_number`(设备真值;
  v0.8.5.1 修复:绝不与 scratch 计数别名,:288-295)、`h_unique_key_number`(HostMirror 审计镜像)、
  接触四类 start id 连续 5 块单次 D2H(:341-357)。
- **【仅 phase-cd】容量档训练成员**:`m_abd_unique_tier[2]`、`m_contact_class_tier[4]`、
  `m_observed_class_count[4]`(帧 0 训练,越档时设备上报 `FRAME_RETRY`,:254-281)。
- **【仅 phase-cd】`device_count_mode()`**(:328-339)决定布局宽度是否走容量档,优先级:
  `s_layout_override_off`(C6-m fallback,thread_local)>
  `s_layout_force_on`(GPU-RL 自持,thread_local + `LayoutForceOnScope`)>
  `STIFF_CONVERT_DEVICE_COUNT` > `STIFF_FRAME_GRAPH`,默认 false。
- 槽审计 `STIFF_SLOT_AUDIT=1`【仅 phase-cd】:sentinel 填充 row=-1,装配后扫描存活
  sentinel 即"保留未写"槽,发现即 throw(:158-170)。

### 1.4 装配流布局与接触四类分区

**【稳定线+phase-cd】**(phase-cd 行号,`gipc_modules/13_kappa_partition_gradhess.inl`):

- 流内顺序:**BARRIER**(`h_cpNum[4]·M12_Off + h_cpNum[3]·M9_Off + h_cpNum[2]·M6_Off`)→
  **FRICTION**(lagged `h_cpNum_last` 同式 + `h_gpNum_last`)→ **GROUND**(`h_gpNum`)→
  **FEM 内部**(:188 附近;探针打印的 tri-bounds 见 `14_energy_linesearch_solver.inl:373-376`)。
- 帧起点 `[P1-dyn]` 可证上界增长(唯一 discard 合法点,:924-1013):

  ```
  bound = m_fixed_triplet_base + fem_point_num
        + M12_Off·h_cpNum[0] + M6_Off·h_gpNum (+ 摩擦项) + 4096
  conv_pred = 2.7 × prev_len        // 2×1.35 跳变余量(towel 触发帧 +26% 单帧跳)
  target = max(bound, conv_pred)
  ```

- `partitionContactHessian()`(:258+):碰撞三元组按 64 位 hash radix sort 分**四类**
  (ABD/ABD、ABD/FEM、FEM/ABD、FEM/FEM);【仅 phase-cd】类档(class tier)固定段布局供整帧图;
  纯 ABD/零接触训练态的档矛盾钳制([D4-b],:307-361)。ABD 类后续 **16× 展开**
  (12×12 body 块 → 16 个 3×3,`global_matrix.h:255-258` 注释)。

### 1.5 Converter:unique-block 归并

**【稳定线+phase-cd】**(binned 归并;跳零 deposit 为【仅 phase-cd】)。
`Converter::convert(global_triplets, start, length, capacity, out_start_id, layout)`,
layout ∈ {AbdContact, AbdFinal, FinalGlobal}(`linear_system/utils/converter.h:7-29`):

1. `_radix_sort_indices_and_blocks`(converter.cu:192-248):
   key = `(uint64(row)<<32) | uint32(col)`;pad 槽 key=~0 排最后;
   `cub::DeviceRadixSort::SortPairs` 按 capacity 宽;按 sort_index 重排块值,pad 置零。
2. `_make_unique_block_warp_reduction`(:250-507+):相邻 key 不等 → partition flag;
   ExclusiveSum 得唯一块序号;首元素写 row/col([C6-t] 该 pass 发射宽度收窄到 length,
   曾是整帧图最大单项开销,:280-306);`_finalize_unique_count<<<1,1>>>` 写
   `d_unique_key_number`;`_publish_unique_frame_state` 越档记 `OVF_UNIQUE_BLOCKS` +
   `FRAME_RETRY_REQUIRED`(:68-91)。
3. **binned 归并**:`m_mergebin`(每唯一块 9 double × BINNED_K bins)deposit-then-combine,
   顺序无关 ⇒ 归并值确定(converter.h:53-57;取代按发射序求和的 FastSegmentalReduce 旧路线)。
4. **[C6-w] 跳零 deposit,默认开【仅 phase-cd】**(`STIFF_SKIP_ZERO_DEPOSIT=0` 恢复,
   converter.cu:464-498):tier 布局下约一半 payload 是 (0,0) 键的零 pad,全部去重到同一索引
   → 同 9 个 bin 的**原子长龙**(单 kernel 实例 2999µs vs 133µs,22×)。跳零后整帧图 −27%
   (forcegrip 60f 13.34→9.78s)。对 det 栈**严格中性**(+0.0 不改 bin);对 merged 只是
   ~2 ULP 重排(merged 本就非 run-to-run 确定,§3.2)。实现为**两个独立 lambda** 而非
   运行时分支——[C6-v] 教训:同一 scatter lambda 加一个永不触发的分支就扰动原子到达序。
5. `ensure_capacity(capacity)`:mergebin 增长,<32MB 翻倍、大缓冲 +12.5%
   (beaker 576MB bins 翻倍曾 OOM 24GB 卡,:112-154)。
6. 计数落地:FinalGlobal + device_count → `h_unique_key_number = length`(上界布局);
   AbdFinal/AbdContact 在事务内用训练档 `m_abd_unique_tier`(clamp 到 length,防 16× 展开
   吃 padding,:337-347);否则精确 D2H(capture 中到达此处直接 throw,:349-377)。

上层 `convert_new()`(global_linear_system.cu:267-343):FinalGlobal;宽度默认 tier 宽
(`STIFF_CONVERT_EXACT_WIDTH=1` 收窄到 exact——[C6-u] 实测噪声内,:303-332)。
`Converter::convert` 在 capacity < length 时 MUDA_ASSERT(converter.cu:166)。

### 1.6 容量档 assembly_capacity_tier

**【仅 phase-cd】**(`linear_system/utils/capacity_tier.h:10-90`):

```
blocks = ceil(count / 256) 上取 2 的幂(≤ 2^22)
tier   = blocks × 256
```

即 **256 粒度的 2 的幂阶梯**,阶梯本身的放大上界数学上 **<2×**——按此公式
971162 → blocks=3794 → 4096 → tier=1,048,576(仅 1.08×)。代码注释中的
"实测 971162 → 2097152 = 2.16×"(capacity_tier.h:18-19)必含阶梯之外的另一因子
(headroom 加倍时代的测量),不可当作阶梯本身的例证引用。
容量档是整帧图录制的 launch 常量宽度来源(§5.5)。

- `STIFF_TIER_STEPS=n` 在幂之间插入 n−1 个几何档(n=2 加 1.5× 档……)。
  **2026-08-07 判定 REJECTED 为默认**:A/A 校准矩阵 24 配对 −1.0% 中位 = null,
  且档边界回退 6-8 次 vs 3-5 次增多;早先"−7.9% 赢"是单批幻影(A/A 噪声地板 5-9%)
  (:19-44 注释)。【实验性,默认关】
- `STIFF_ASSEMBLY_TIER_SHIFT`:手动提档(诊断)。

---

## 2. PCG 与 MAS 预条件

### 2.1 PCGSolver 配置与收敛判据

**【稳定线+phase-cd】** 文件:`linear_system/solver/pcg_solver.{h,cu}`。

**配置**(`PCGSolverConfig`,pcg_solver.h:22-32):

| 参数 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `max_iter_ratio` | 0.3 | — | 最大迭代数 = `0.3 × DOF 数`(solve() :728-729) |
| `global_tol_rate` | 1e-4 | — | 相对收敛阈值(被引擎配置覆盖) |
| `use_bsr` | true | — | 声明存在,本文件内未见消费点(疑为死字段,**待核实**) |

引擎链路:`SimEngineConfig.pcg_tol = 1e-4`(sim_engine.h:39)→
`ipc.pcg_threshold = cfg.pcg_tol`(engine_modules/01_config_upload.inl:17)→
`cfg.global_tol_rate = pcg_threshold` 后 `create<gipc::PCGSolver>(cfg)`(gipc/gipc.cu:196-198)。
运行时覆盖:`STIFF_PCG_TOL`、`STIFF_PCG_CHECK_K`(默认 **K=8**;pcg_solver.cu:782-783,1418-1420)。

**收敛判据**(数学形式):记预条件残差 z = M⁻¹r,rz = rᵀz(即 ‖r‖²_{M⁻¹});初始 rz₀ = r₀ᵀz₀。

```
收敛:  |rz| ≤ tol_rate · rz₀            (M⁻¹-范数平方相对下降 1e-4)
护栏:  pᵀAp ≤ 0 或非有限  →  d_break = 2  (SPD 破坏 / NaN)
```

(判据 :407-417;护栏 :362-383,:420-433;beta 侧 rz_old==0/非有限则跳过 p 更新 :387-402。)

**检查节奏 K**:每 K 次迭代做一次收敛检查(host 路径 = 阻塞 D2H 读 break 旗;
device-loop 图 = 尾核自决)。固定 K ⇒ 迭代数确定(strict 安全);代价是收敛后最多多跑
K−1 次(:779-782 注释)。**分段 PCG 要求 K 为正偶数**(rz 指针乒乓的奇偶性;
违反抛 runtime_error,:1498-1501)。

**签名**:

```cpp
PCGSolver(const PCGSolverConfig& cfg);
void config(const PCGSolverConfig&);
const PCGSolverConfig& config() const;
protected:
SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;
// 返回:迭代数(host 路径);录图/Phase-C 条件图返回 0(计数走 FrameDeviceState.pcg_iter_total)
// 抛错:分段+条件图下 K 非正偶数;条件图下 K==0(:860-862)
```

### 2.2 迭代体与三级执行策略

初始化:`r = b`;`z = M⁻¹r`;`rz = r·z`(cub 融合 dot:`Cub_PCG_DotReduction`,
TransformInputIterator 把乘法融进树规约,1 次 launch,:633-653);`p = z`;x 初值填 0(:694)。

每次迭代 body(:819-853,6-8 个 kernel):

1. `pcg_iteration_begin<<<1,1>>>`:`iteration_active = iteration < max_iteration`;
2. `spmv(p, Ap)`(§3.4 的 det/fast 两路,作用于归并后唯一块);
3. `dot(p, Ap) → d_dot_res`;
4. `update_vector_dx_r_fused`:每线程重算 `α = rz/dot_res`,`x += αp; r −= αAp`
   (dot 非法时 thread0 置 break=2 并全员跳过 axpy);
5. `z = M⁻¹r`;
6. `dot(r, z) → d_rz_new`;
7. `update_vector_c_fused`:`β = rz_new/rz`,`p = z + βp`;
8. `post_iter_swap_and_check<<<1,1>>>`:`rz = rz_new`;判收敛;`++iteration`。

设备侧标量全家(免每迭代 2 次 D2H):`d_rz/d_rz0/d_rz_new/d_dot_res/d_alpha/d_beta/d_break` +
`PCGDeviceState* d_graph_state`(pcg_solver.h:11-20, 56-67;alignas(16),字段
iteration(从 1 起)/max_iteration/iteration_active/converged/terminal/segmented/
`frame_fsm::FrameDeviceState* frame`)。
【仅稳定线差异】稳定线的 d_graph_state 只是 `unsigned long long*` {iteration, break} 快照,
无 FrameDeviceState 发布(diff 亲验)。

**三级执行策略**(按优先级):

| 级 | 条件 | 行为 | 适用线 |
|---|---|---|---|
| 1. **Phase-C 条件图内嵌** | `frame_fsm::ConditionalGraphRecorder::current()` 活跃(整帧图录制中) | 把 K 次 body + `pcg_graph_tail_conditional` 录成嵌套 WHILE 条件节点;返回 0,迭代数设备侧累计进 `FrameDeviceState.pcg_iter_total`(:858-881) | 【仅 phase-cd】 |
| 2. **device-loop 自尾图** | `STIFF_PCG_DEVICE_LOOP`(默认 1)且 driver ≥ 12000 | K 次迭代 capture 成 device-launchable graph;尾核 `pcg_graph_tail_relaunch` 用 `cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch)` 自我续跑至收敛/超限,中途零 D2H(:535-555, :917-1103) | 【稳定线+phase-cd】 |
| 3. **host 回放图 / 纯循环** | capture 失败或图被 gating 关闭 | capture K 次 body 为普通 graph,回放一批后 D2H 读 break;capture 失败永久回退纯循环(:1116-1157) | 【稳定线+phase-cd】 |

图路径 gating(:889-897):
`use_graph = STIFF_PCG_GRAPH(默认1) && !STIFF_SPMV_DET && !STIFF_KSUM && !STIFF_MAS_FUSE_VALIDATE && !STIFF_MAS_DUMP && precond_graph_capturable()`。
即 **strict(SPMV_DET)不走标量 PCG 图**——但分段 PCG 图对 strict 开
("graphs are arithmetic-neutral",:1515-1526),且 det-SpMV 的惰性 ybin 分配在 capture 前
预热(capture 内 cudaMalloc 非法,:1546-1556)。

**[B2'-b] 跨 solve 图缓存【仅 phase-cd,实验性,默认关】**:`STIFF_PCG_GRAPH_CACHE=1`;
签名 = {pcg_buffer_generation, dofs, K, tol bits}(:938-948),命中跳过整个 re-record。
录制时 `pcg_grid_capacity_mode()=1` 让 spmv/Schwarz 网格按容量取宽、kernel 内按设备侧
活跃计数掩蔽(pcg_capacity_mode.h:14-18);任何搬动已烘焙指针的 realloc
`++pcg_buffer_generation()` 使缓存失效。

> **注**:标量路径的 `Cub_PCG_DotReduction` 依赖 cub::DeviceReduce 的固定归约树来获得
> run-to-run 确定性——这是基于 CUB 实现惯例的判断,代码内无显式声明(**待核实**)。
> strict 的正门主要走分段路径的 binned dot(§2.3),不依赖此假设。

### 2.3 分段(块对角)PCG:seg_pcg

**【稳定线+phase-cd】**(`pcg_solver.cu:1204-1706`;multi-env P3 产物)。

- **触发**:`ModeConfig::env_on("STIFF_SEGMENTED_PCG")` 且 `m_s4_dof_to_group/m_s4_ng` 就位
  (solve() :722-729)。dof_to_group 按 3×3 块索引:DOF i → block i/3 → group d2g[i/3]。
- **数学原理**(pcg_solver.h:87-94):P1 隔离后矩阵按 env 块对角、预条件 intra-env;
  标量 PCG 中唯一的跨 env 耦合是全局 dot。换成 **per-env dot + per-env α/β/收敛**后,
  各 env 数学独立 ⇒ 相同 env 跨 env 逐位一致。ng=1 也走分段核
  (**批大小不得改变数值算法**,:717-719)。
- **per-env dot 三条路径**:
  1. **strict(binned)**:`_seg_dot_deposit`(块内 shared bins,`binned_deposit`
     精确指数分箱)+ `_seg_dot_combine(_ck)`(:44-103, :229-244)。
     `g_seg_binned` 由 `STIFF_SPMV_DET || STIFF_SEG_BINNED` 值跟踪设置(:1211-1218)。
  2. **warp 预求和**(`g_seg_warp`;merged/isolated 默认 1,strict 自动 0):同 env 整 warp
     shuffle 树预求和再一次 deposit,**7.9× 提速**,但 warp 分组依赖全局 DOF 布局
     (env k 起点 Σ Nⱼ 非 32 对齐)→ 破坏跨 env/跨 batch 位同一;它保 run-to-run
     (布局一次运行内固定)(:1218-1230)。`STIFF_SEG_WARP=1` 可强制(A/B 用,破确定性)。
  3. **spmv 融合 dot(fast 路径)**:spmv 核内顺带累加 per-env `p·Ap` 到
     `d_dot_partials[ng*256]`(block 哈希槽,spmv.cu:181-188),`_seg_partial_combine(_ck)`
     折叠 + 顺带收敛检查;`arm_precond_seg_dot` 让预条件器顺带累加 r·z
     (Diag/ABD capable,**MAS 不 capable** → 回退独立 dot,:1470-1476)。
- **warm start**【实验性,默认关】(`STIFF_PCG_WARM=1`,非 binned 限定):x₀ = 上次解,
  r₀ = b − A·x₀(修正旧 bug:旧版设 r=b 使解 = x_stale + H⁻¹b);per-env 护栏
  ‖r₀_g‖² ≤ ‖b_g‖² 否则回零启动;rz₀_g 保持 b 基准(:1296-1327)。
- **Eisenstat–Walker 自适应 forcing**【实验性,默认关】(`STIFF_PCG_EW=1`,非 binned 限定):
  `η_g = γ·rz₀ₖ/rz₀ₖ₋₁`,`T_g = η² ∈ [tol, η_max²]`;默认 γ=0.9、η_max=0.5,
  可调 `STIFF_PCG_EW_GAMMA/STIFF_PCG_EW_ETAMAX`(:283-304, :1361-1381)。
- 迭代 kernels:`_seg_axpy_xr`(冻结已收敛 env,:143-158)、`_seg_axpy_p`(:159-172);
  rz 交换 = host 指针乒乓(K 偶数保持奇偶,:1439-1491)。

### 2.4 预条件器框架与 P_type 分支

**【稳定线+phase-cd】** 接口(`linear_system/linear_system/i_preconditioner.h`):

- `IPreconditioner`:`do_apply(r,z)` / `do_assemble(GIPCTripletMatrix&)` 纯虚;
  `seg_dot_capable()` 默认 false;`arm_seg_dot(partials, d2g, ng)`;
  `graph_capturable()` 默认 false(:31-41)。
- `LocalPreconditioner`(绑定一个 DiagonalSubsystem):`do_apply` 按子系统 dof 区间取
  subview 再调 `apply`(i_preconditioner.cu:122-129);
  `calculate_subsystem_bcoo_indices(number, keep_count_device)` 用 DeviceSelect 从归并后
  唯一块筛出 row/col 均落在本子系统块区间的索引(:33-105;device_count 模式发射宽度 =
  `assembly_capacity_tier(host_count)`,精确数留设备)。
- `GlobalPreconditioner`:apply 作用全向量。

**注册分支**(`gipc/gipc.cu:194-215`,`GIPC::create_LinearSystem`):

| P_type | 注册 | apply 组合 | 说明 |
|---|---|---|---|
| 1(默认) | `ABDPreconditioner`(local,id=0,ABD 12-dof 段)+ `MAS_Preconditioner`(local,id=1,FEM 段) | 全局阶段 = identity(`z=r` 的 memcpyAsync,global_linear_system.cu:244-257),随后 ABD local 覆写 ABD 段、MAS local 覆写 FEM 段 | `SimEngineConfig.preconditioner_type = 1`(sim_engine.h:67) |
| 0 | `ABDPreconditioner` + `DiagPreconditioner`(Global) | Diag 全向量 `z = D⁻¹r`,再 ABD local 用 12×12 块逆覆写 ABD 段(:259-262) | Python 可配:`bindings/pystiffgipc.cu:124` |

- **DiagPreconditioner**(diag_preconditioner.{h,cu}):assemble 遍历唯一块,`i==j` 时
  `diag(i) = inverse(H)`;`H.isZero(0.0)`(中性 (0,0) pad)不反转(:9-39, :89-95)。
  apply:`z_i = D_i⁻¹ r_i` 逐 3×3 块;armed 变体顺带 per-env r·z。
  `seg_dot_capable`/`graph_capturable` = true。
- **ABDPreconditioner**(abd_preconditioner.{h,cu}):per-body 12×12 系统对角块求逆
  (`_cal_abd_system_preconditioner`);apply
  `z.segment<12>(i·12) = inv(i) · r.segment<12>(i·12)`;armed 时用修正累加
  `r·(z_new − z_old)` 使 per-env Σ 等于最终 z 的 r·z(:28-33 注释)。

### 2.5 MASPreconditioner:多级加性 Schwarz

**【稳定线+phase-cd】**(稳定线为 3126 行单体 `MASPreconditioner.cu`;phase-cd 拆为
`mas_modules/00..05` 顺序 #include 的复合 TU,拼接与拆分前字节级相等,
sha256 见 `mas_modules/ORIGINAL_SHA256.txt`;**include 顺序禁止重排**)。

**结构常量**:`BANKSIZE 16`、`DEFAULT_BLOCKSIZE 256`(eigen_data.h:17-19);
每 cluster(bank)= 16 节点 = 48×48 对称块,存为 `MAS_NB = 16·17/2 = 136` 个 3×3 上三角块;
装配/求逆输入为 fp64(`MasMatrixSymT{Matrix3d M[136]}`),**应用矩阵为 fp32**
(`MasMatrixSymf{Matrix3f M[136]}`,eigen_data.h:116-127)。层数上限 **6**
(`computeNumLevels`,levelSz 每层 /16 再 16 对齐,05_envseg:394-413)。

**生命周期 API**(完整声明 `MASPreconditioner.cuh:89-142`;phase-cd 行号):

1. `void initPreconditioner_Neighbor(int vertNum, int mCollision_node_offset, int totalNeighborNum, int4* m_collisonPairs, int partMapSize)`(05_envseg:1328-1412)
   - `vertNum` = **FEM 顶点数**(`ipc.vertexNum − abd_vertexOffset`);
     `mCollision_node_offset` = abd_vertexOffset;`partMapSize` = METIS 分区数 × 16。
   - 层数按 **per-env 节点数**定深(`segN>1 ? maxNodes/segN : maxNodes`)——否则批大小 N
     改变层深 → 同一 env 在不同 batch 拿到不同预条件器 → ~1e-9 批漂移(:1341-1352)。
   - 一次性分配聚合 scratch 与 CUB scan workspace(capture 内 Thrust 临时 cudaMalloc 非法)。
2. `void initPreconditioner_Matrix()`(:1414-1460):以 cpNum=0 跑一次 `ReorderRealtime(0)`
   得零接触 cluster 总数;输出层缓冲按 `totalCluster×1.05` 分配;binned 累加器
   `g_mRbin/g_mZbin/g_matbin` 经 `cudaMemcpyToSymbol` **一次性**绑定
   (每次 apply 内绑定会破坏图 capture,:1447-1456)。
3. `int ReorderRealtime(int cpNum)`(:460-531):**每次 MAS assemble 都重建整个层级**
   (接触改变连通性)。步骤:L0 连通掩码(CSR 邻接表 → 同 bank 位掩码)→ 碰撞连接
   (int4 对经 `_real_map_partId` 映射进分区序,跨 bank 记入 coarse 表)→ bank 内连通闭包 +
   per-bank cluster 计数 → `BuildLevel1`(CUB ExclusiveSum + fine→coarse 链 `d_goingNext`;
   per-env 时每 env 的 L1 段对齐到 BANKSIZE 独立 bank)→ 层循环 level=1..levelnum−1
   (投影邻接 → 碰撞连接 → NextLevelCluster → PrefixSumLx → ComputeNextLevel;
   【仅 phase-cd】[B3 s8] `_mas_env_base_dev` 等设备驻留 env 分段,消灭每层 host 读回)→
   `AggregationKernel`(逐顶点逐层粗节点路径 `d_coarseTable`)。
   【仅 phase-cd】`deviceExtentActive()` 时总 cluster 数取分配上界作发射宽度,精确数留设备
   (`device_cluster_count()` 暴露给图 resizer);否则 D2H 读 `d_levelSize[levelnum]`。
4. `void setPreconditioner_bcoo(Matrix3d* vals, int* rows, int* cols, uint32_t* indices, int offset, int triplet_num, const int* d_triplet_num, int cpNum)`(:1009-1103):
   恢复邻接表副本 → `ReorderRealtime(cpNum)` → `ensureOutputClusterCapacity`
   (lens-A:帧内 cluster 数可超 init 零接触分配)→ `PrepareHessian_bcoo`。
   `STIFF_MAS_DUMP=<dir>` 首次装配 dump 原始三元组 + 分区前缀,供
   `examples/test_mas_oracle.py` CPU oracle 全链重建。
5. `PrepareHessian_bcoo`(:534-891):
   - 三元组落地:**同 bank** → 直写 `_invMatrix[cPid].M[16·bvRid − bvRid(bvRid+1)/2 + bvCid]`;
     **跨 bank** → 沿 goingNext 爬层,首次同粗 bank 时按上/下三角转置 `binned_deposit`
     进 g_matbin(对角块双计 H+Hᵀ,:607-673)。
   - bank 稠密化 + 粗层聚合:`prefix==1` 快路 = warp shuffle 树 + 每 warp 一次 deposit
     (修复历史 partial-bank 集体操作 UB);**strict(`g_det_reduce`)= 逐对 binned deposit**
     (顺序无关 ⇒ 布局/批大小不变,:721-847)。
   - `__inverse6_P96x96`:逐 cluster 48×48 **Gauss-Jordan 就地求逆**(shared 双缓冲,
     96 线程算 2 个 48×48;零对角补 1;lens-B:无 early-return,越界线程驻留 phantom
     单位阵防 barrier divergence;02_inverse:1-112),结果写 **fp32** `d_precondMatMas`。
6. `void preconditioning(const double3* R, double3* Z)`(:1116-1326)= PCG 每迭代 apply:
   - 容量断言:`totalNumberClusters ∈ [0, m_clusterCap·levelnum]` 且 16 对齐,否则 throw;
   - `BuildMultiLevelR(R)`:细层 mlR = R(fp64→fp32),cluster 残差和沿 goingNext
     binned_deposit 到各粗层(fast=warp 段树;strict=升序 lane 确定性求和);
   - `SchwarzLocalXSym_block3` → `_schwarzLocalXSym6`:每线程一 (bank行, bank列) 对,
     fp32 `M[idx]·smR[col]`,warp 段树按行归约,行边界 lane deposit 进 g_mZbin;
   - `CollectFinalZ(Z)`:每顶点 z = 自身 cluster z + 逐层 coarseTable 粗 z 之和,回写 double3。
   - `STIFF_MAS_FUSE=1`【实验性,默认关】:融合核族(`_schwarzLocalXSym6_fused8` +
     `__collectFinalZ_binned_new`);`STIFF_MAS_FUSE_VALIDATE` 与 legacy 路径一次性 bit 对比,
     不一致 throw(05_envseg:1188-1270)。
   - `[mas-apply-resize]`【仅 phase-cd,实验性,默认关】:Schwarz 发射是图内 PCG 的主宰成本
     (**实测占图内 PCG body 63-65%,3.3× 于 spmv**,:918-927 注释);
     `STIFF_MAS_APPLY_RESIZE=1` armed 设备 resizer 以 `d_levelSize[levelnum].y` 收窄录制网格。
7. `void FreeMAS()`(:1516-1571)。

壳层 `MAS_Preconditioner`(preconditioner/fem_mas_preconditioner.cu):assemble 读
`*cpNum` → `calculate_subsystem_bcoo_indices` → `setPreconditioner_bcoo(...)`;
apply = `preconditioning((double3*)r, (double3*)z)`(r/z 已是 FEM 段 subview)。
`graph_capturable = true`(symbol 绑定已移至 alloc 期 + memset 全 Async)。

**MAS 确定性(strict)**:三个非确定 atomicAdd 目标(d_multiLevelR / d_multiLevelZ /
d_inverseMatMas 粗聚合)全部换 binned 可重现浮点(00_binned_accum.inl:33-40);
`g_det_reduce=0`(默认)时 `binned_deposit` 退化为 bins[0] 单 atomic(§4.1)。

**per-env MAS 分段**:`_mas_envSegN`(05_envseg:106-139)——多 env 默认开;
`STIFF_MAS_SEG=0` 强关/其它值只接受验证过的 env 数;上限 4096。
启用前置是 **bank 级同质性守卫**(01_config_upload.inl:656-731):FEM 顶点须 env-major
连续、每 env 等顶点数、每 bank 单 group 且 group 的 bank 区间连续等长
(分区数随图结构不随顶点数),否则回退全局层级并打印警告。

### 2.6 METIS 排序(MAS 前置)

**【稳定线+phase-cd】**(`MeshProcess/metis_partition/src/metis_sort.cpp`):

- `block_size = 16`(:74);nPart 自适应循环
  `nPart = (V + block_size − metis_offset − 1)/(block_size − metis_offset)`,反复
  `METIS_PartGraphKway` 直到最大分区 ≤ 16(:152-192)。
- 产物:排序序→输入序置换 + 分区文件(`sorted_mesh/`);sidecar 缓存
  `<mesh>_sorted.16.idx`(合同见 metis_sort.h:1-25;陈旧缓存检测 :130-135)。
- 引擎侧映射:`tetMesh.partId`(每 FEM 顶点分区 id)、`part_offset`(分区总数)、
  `partId_map_real[part_offset·16]`(分区槽→真实顶点,−1=padding)与反向表
  (load_mesh.h:56-60);用户可见顶点序经 `vertex_metis_to_input` 透明还原(:169-180)。
- 部分场景以 `P_type = 0` 绕过 metis_sort(gl_main.cu:1668, 1864)。

### 2.7 实测:MAS 应用占比与 diag A/B 结论

权威出处 [../OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md)(2026-08-11,A800,
foldshirt 4-env merged 回放 1551 帧,宿主通道 410s = 263ms/帧;
设备 globaltimer 打点 `STIFF_GRAPH_PHASE_TIME`,416.8s 设备时间):

| 相位 | 时间 | 占比 |
|---|---|---|
| PCG(线性求解) | 237.1s | **56.9%** |
| — 其中 precond 应用(MAS) | 112.3s | **27%(全仿真)** |
| — 其中 SpMV | 34.1s | 8% |
| — 其中 mix1/mix2 | 31.0s | 7% |
| GH(装配) | 91.8s | 22.0% |
| CCD+LS | 82.9s | 19.9% |

- **MAS 应用占全仿真 27%** 是该场景的实测,不是普适比例(finray 2env 上 diag 反胜)。
  fp32 应用降精度立项的理论上限 ~13%(带宽减半)。
- **对角替代 MAS 的 A/B**(同文档):diag 两轮 r0=501s/Newton 10881、r1=401s/10088;
  MAS 基线 410-411s/Newton 8846-8850(极紧致)。对角**每迭代便宜 ~13%** 但
  Newton +14~23% 且轨迹方差巨大 → 净账高方差平手偏负,列入死路表
  ("慢 22%,Newton +23%")。结论:**MAS 的收敛性 + fp32 应用降本 = 两头通吃**;
  预条件选型**场景相关**(finray 2env diag 胜 / 大批量 MAS 优)。
- 图内 Schwarz apply = 图内 PCG body 的 63-65%、3.3× spmv
  (05_envseg_host_pipeline.inl:918-927,mas-apply-resize 的动机)。
- 分段 binned seg-dot 核曾占 strict 帧 18.5% → warp 预求和 7.9×(pcg_solver.cu:33-41)。
- "MAS levelnum 扫描"用户裁定不做(OPTIMIZATION_ROADMAP.md 死路表)。

---

## 3. 多环境实现机制

> 本节讲**实现机制**;三模式的用户契约、模式解析器(`resolve_multienv_mode`)、进程级
> 模式锁与遥测 API 属于多环境使用分册。促记:模式 = 两条独立轴(耦合度 × 确定性)上的
> 三个点,单一权威 `StiffGIPC/multienv/mode_contract.h`【仅 phase-cd 有此成文契约头】。

### 3.1 两轴三模式与 flag bundle

flag bundle 单源(`stiff_physics/engine.py:194-217`,两线一致):

```
isolated = STIFF_BVH_ENVDET  STIFF_PERENV_BVH  STIFF_DECOUPLE_THRESH  STIFF_PERGROUP_KAPPA
           STIFF_SEGMENTED_PCG  STIFF_PERENV_ALPHA  STIFF_PERENV_PAR          (7 个)
strict   = isolated + STIFF_EE_CANON  STIFF_EE_DETGATE  STIFF_CCD_CANON  STIFF_SPMV_DET  (11 个)
merged   = (none)      # merged 额外 setdefault STIFF_EE_LB=2
```

| 模式 | 承诺 | 不承诺 |
|---|---|---|
| MERGED(默认) | 最大吞吐;一个全局求解(全局 line-search α、全局 Newton 收敛、共享 kappa);热路径零 per-env 机制 | 隔离(病态 env 抛异常杀整批,fail-fast 是立场;fail-isolate 扩展被评估并**拒绝**);可复现性 |
| ISOLATED | per-env 公平(per-env α、per-env 收敛冻结、per-group kappa、env 本地 broadphase);**铁律(有前置条件)**:病态 env 中途检疫为完全惰性,健康 env 继续——但**中途检疫要求 host 遥测路径开启**(`env_newton_iter_cap>0` 或 STIFF_PERENV_TELEM,后者由 `per_env_exit=True` 设置;两者均**不在** isolated 的 7 旗标 bundle 内且默认全关,§3.3 可用性门)。默认配置的 `mode="isolated"` 走纯设备快路径,**蓄意没有 NaN 防御**,拿不到中途检疫 | 可复现性 |
| STRICT | isolated 全部 + **逐位(bitwise)可复现:run-to-run 且跨架构**(已证 sm_80 ≡ sm_89) | —(代价:canonical 发射序 + 布局固定策略) |

C++ 运行时真相【仅 phase-cd】:`ModeConfig`(multienv/mode_config.h)= finalize 时刻对
STIFF_* env 的一次性快照;命名规则
`mode = (iso_bundle==7) ? (strict_extras==4 ? Strict : Isolated) : Merged`——只有**完整**
bundle 才命名;`env_on()` 值感知(空/未设/"0" = 关)。半配置组合由 `warn_if_incoherent()`
告警。稳定线无 ModeConfig,机制散落在单体 GIPC.cu 内直接 getenv。

### 3.2 merged:合场景与"为何非 run-to-run 确定"

merged 把所有 env 放进**一个**全局求解:一次 BVH(env-major Morton,§3.4)、一个全局
line-search α、一个全局 Newton 收敛判据、共享 kappa。它非 run-to-run 确定,机制链条
(全部有代码/文档出处):

1. **裸 atomicAdd 归约**:`binned_deposit` 默认路径是一条
   `atomicAdd(&bins[0], val)`(binned_reduce.cuh:26-30)——浮点加法不可结合,
   原子到达顺序是调度属性。
2. **双流配对发射**:DCD 检测 face 树在默认流、edge 树在 `m_aux_stream` 并行重叠,
   两者 atomicAdd 进同一 `_cpNum` 与 `_collisionPair`;**pair 集合确定、槽排列竞态**
   (消费者按 `0..h_cpNum[0]` 遍历不受影响;10_ccd_buildcp_quarantine.inl:1988-2008)。
3. **LS 能量按槽序求和**继承该排列竞态(:2136-2142)。
4. **档位边界截断竞态**【仅 phase-cd 图路径】:发射超容量的 emit 被重定向 trash slot,
   哪些 pair 被截取决于原子到达序(frame_transaction.cu:4598-4605)。
5. **实测**:plain release 路径 towel **从 frame 2 就与自己分歧(2.2e-14 relative),
   frame 119 达 1.1e-4**(converter.cu:474-477;A800_ALLEXAMPLES_TIMING:300-305)。
   Newton 计数倒是稳定(G9 用它做包络门禁)。

**推论**:任何依赖逐位重放 merged 的测试都在测调度巧合;merged 的回归验证用
Newton 包络 + 跨模式等价容差(G9),不用逐位哈希。

### 3.3 isolated:per-env 束与检疫铁律

isolated 解决的是**物理隔离/公平**,不是复现。核心机制
(phase-cd 布局 `multienv/isolation.{cuh,cu}` + `gipc_modules/11_perenv_machinery.inl`;
稳定线机制在单体 GIPC.cu,有 quarantine、**无 reviveEnv**):

- **可用性门** `GIPC::perEnvIsolationLive()`:multi-env groups 已声明
  (`m_active_group_count > 1`)AND host 遥测路径开(`env_newton_iter_cap > 0` 或
  STIFF_PERENV_TELEM)AND per-env alpha 开。**纯设备快路径蓄意没有 NaN 防御——
  隔离承诺依赖此门**(isolation.cuh:8-13)。
- **状态码** `m_env_status`:0 running / 1 converged / 2 timeout-frozen(per-solve)/
  3 quarantined;`m_env_quarantined` 是跨帧持久旗标,带设备镜像(isolation.cuh:14-17)。
- **被检疫 env 完全惰性**:位置冻结(α==0 保持最后接受状态)、每迭代方向清零
  (`_zero_dir_quarantined`)、body 从 ground 检测与 ground-CCD α 中移除(skip 表)、
  solve 循环槽位钉状态 3。init 时刻的违规仍抛异常(isolation.cuh:18-22)。
- **三个 mid-frame 入口**:帧首 flag-only 不可行探针(teleport 可在帧间使 env 不可行)、
  detection 时降级、post-PCG 非有限方向扫描(`_scan_dir_nonfinite`)(isolation.cuh:23-27)。
- **API**:

  ```cpp
  bool GIPC::quarantineEnv(int env, int vertex, double distance);   // 【稳定线+phase-cd】
  bool GIPC::quarantineEnvOfVertex(int vertex, double distance);    // 【稳定线+phase-cd】
  bool GIPC::reviveEnv(int env);                                    // 【仅 phase-cd】
  ```

  `reviveEnv` 是 episode reset 的逆操作(清旗标 + 清 ground-skip + status=0);
  **复活自纠错**:仍坏的 env 一帧内被重新检疫(帧首探针/每迭代检测器)
  (isolation.cu:184-223)。调用点:`SimEngine::teleport_fem_vertices` 触及被检疫 env 时
  视为 episode reset,复活后强制重建帧入口 pair 集(04_teleport_checkpoint.inl:95-123)。

**per-env 求解决策链**(与 merged 路径的数学对齐是 strict 的前提,详见
11_perenv_machinery.inl):

- per-env kappa:`_per_env_kappa_deposit/_combine/_finalize`
  (`K_g = clamp(max(−gsum_g/gsnorm_g, suggested), 0, kmax)`,suggested/kmax 为 env 无关
  host 标量,逐位等于 host 循环);postLineSearch 的 per-group κ 设备加倍
  `_per_group_kappa_double`(冻结 env 连接触参数一起冻结——否则伙伴的额外迭代继续加倍
  该组 kappa,使 strict env0 依赖 batch size,11:243-246)。
- per-env line-search α:`_per_env_groundAlpha_min` / `_per_env_selfAlpha_min`
  (canonical 顶点序见 §3.4)+ 设备侧 `_per_env_alpha_compute`
  (公式与 host 循环逐位一致:`ta = min(ground, narrow)`;CFL `a = min(ta, 0.5·sq/hmx)`;
  refinement 门 `gate_lhs > 2·gate_rhs` 用 **per-env** ta/acfl——全局门会让 env₀ 是否进
  refined CCD 取决于伙伴,破坏 batch-invariance;冻结阈值
  `thr_g = vtol_dt>0 ? vtol_dt : ntol_dt·sqrt(env_bbox2[g])` 用 env 自身 bbox,
  batch-invariant;11:117-172)。
  不变量(已验证):`min_g(m_env_alpha) == 全局可行 α`;N=1 时 env₀ α == 全局 α。
- per-env 严格 line search(S2/S3):每 env 按自己的 CCD 可行 α 步进,强制 **per-env 能量
  下降**(E_g 上升的 env 单独减半重试,maxBT=8 次内不降则回退 uniform search);
  per-env 能量分解精确(Σ_g E_g == global,机器精度)(ipc_solver.inl:266-362)。
- per-env swept-CCD 搜索膨胀:每 env 用自己的 `ta_e = min(ground_e, narrowSelf_e)` 而非
  全局 α(全局 α 使 env e 的 swept-BVH 搜索/refined pair 集/hr 依赖伙伴——已确认的批漂移
  根因);ta_e ≥ 全局 α ⇒ 保守超集(11:426-436)。
- `STIFF_PERENV_PAR`:per-env swept build+query 在 K-stream scratch 池上并发
  (默认 K=8,`STIFF_PERENV_K`);2026-07-04 已 **QUALIFIED FOR STRICT**
  (run-to-run + cross-env + batch N=2/4/8 全 0.000——canon/order-free 栈吸收了 stream
  交错的 pair 发射)。

### 3.4 strict:跨环境逐位机制

strict 的逐位一致要求整条链每层都"序无关或序规范"。逐层拆解:

**(a) 场景构造层:局部帧(co-located local frames)**
所有 env 加载在同一原点(局部坐标位级相同),分隔交给
`SimEngine::set_env_offsets(per_group_xyz)`(sim_engine.h:229-233,**必须 finalize()
之后调用**;up 轴分量保持 0 使地面接触共享):**BVH 建在 `_vertexes + offset` 上
(env 空间分离),narrow-phase 仍在局部 `_vertexes` 上**——消除使相同 env 分歧的
world-offset 浮点。示例层惯例 `CASE39ME_LOCAL_FRAME=1`(examples/umi_finray_lib.py:835-850;
这是示例层旋钮,不是引擎旋钮)。这一层是 **cross-env 位级一致**的前提;
run-to-run 一致不依赖它。【稳定线+phase-cd】

**(b) BVH 构建层:`STIFF_BVH_ENVDET`**
per-env build 的 Morton 排序换成稳定排序(cub LSD radix SortPairs,天然稳定):
等 Morton 码平局保持 env-local-canonical 顺序 ⇒ 相同 env 建出相同树 ⇒ 相同遍历 ⇒
相同候选集(mlbvh_modules/00_gates_globals.inl:936-977)。【稳定线+phase-cd】

**(c) 配对发射层:`STIFF_EE_CANON` + vloc 总序 + `STIFF_EE_DETGATE`**
- `g_vloc`:global→env-local 顶点 id 映射(镜像 env 间不变);
  全序 `_vless(a,ia,b,ib)` = 位置字典序 (x,y,z),再按 `g_vloc` 破平局
  (00_gates_globals.inl:70-75)。
- EE canon:每条边内部端点按 `_vless` 交换;边对顺序(self/obj)按更小端点交换——
  使 `_dType_EE` 分类与能量计算的输入顺序与 env/线程布局无关
  (03_pair_emission.inl:122-159)。EE once-only dedup 的比较键从全局索引
  (非 env 镜像)换成 env-local 键 `_edge_lkey`(00_gates_globals.inl:921-930)。
- `g_ee_detgate`:emit 判据 `d < dHat` 改用**序不变的真实 segment-segment 几何距离**
  `_seg_seg_d`,而非 dtype 分类选出的子距离 ⇒ 近阈值接触在镜像 env 做出相同 emit 决定
  ("the last bit-identity layer",:931-935)。
- vloc 表在首次 buildCP 时按 p2g 在 host 编 env-local 序号并上传;`g_vloc` 是进程全局
  CUDA symbol,每次 EE kernel 前无条件重发布本 Engine 视图(防跨 Engine 陈旧指针,
  10_ccd_buildcp_quarantine.inl:1958-1972)。【稳定线+phase-cd】

**(d) CCD 层:`STIFF_CCD_CANON`**
`_per_env_selfAlpha_min` 内:PT 对按 env-local id 对三角形三顶点排序(3 次比较交换网络)、
EE 对边内端点+边间顺序按 vloc 规范化 ⇒ 镜像对以相同顺序喂
`point_triangle_ccd`/`edge_edge_ccd` ⇒ TOI 逐位一致(refined-CCD `hr` 的 1-ULP 种子修复;
11_perenv_machinery.inl:343-409)。per-env α 扫的是 **DCD 时刻的快照**
`_dcd_ccd_snapshot`,绝不扫 live CCD buffer(其前缀是上一迭代 swept 发射的竞态有序切片
——曾是 strict 跨 env 不对称的根因;ipc_solver.inl:1789-1800)。【稳定线+phase-cd】

**(e) 归约层:binned 精确指数分箱(§4.1)+ det SpMV(`STIFF_SPMV_DET`)**
det SpMV(spmv.cu):行贡献**逐 entry `binned_deposit`,完全不做 warp segmented reduce**
(warp 归约按 lane 序求和,镜像行在不同全局三元组位置 → lane 序不同 → 跨 env 1-ULP 种子;
:190-198);下三角贡献 deposit 进 ybin;`spmv_binned_combine` 固定序合并且读后清零
(无每 call memset,bit-identical by construction,:250-264)。
fast 路径(merged/isolated)保留 `cub::WarpReduce::HeadSegmentedReduce` + 直接 atomicAdd
(省 ~8MB/call 的 ybin memset)。【稳定线+phase-cd】

**(f) 消费序层:canon-slots(C6-l)【仅 phase-cd】**
`GIPC::canonicalizePairSlots()`(10_ccd_buildcp_quarantine.inl:2133-2347):发射后对
pair 槽(+ ground 列表)做一次稳定两趟字典序 radix sort(先次键 (z,w) 后主键 (x,y),
key 经符号位翻转的 `_canon_ord32`)⇒ **槽序成为 pair SET 的纯函数**。动机:DCD 发射用
atomicAdd 分配槽 → 同一 pair 集的槽**排列**逐跑不同;LS 能量按槽序求和 → recorded
whole-frame replay 的调度抖动能翻掉刀锋 LS 决定。调用点:buildCP 内
`if(m_mode_config.ee_canon) canonicalizePairSlots();`(:2058-2060)。
非图 strict 的 G1/G2 锚多年逐位说明宿主路径槽序在同配置下可重复,但这是经验性质
(**待核实**其机制保证);canon-slots 是图模式下的根治。

**per-env / strict 相关 SpMV 与 PCG 的配套**:strict 关闭标量 PCG 图(§2.2 gating)、
自动 `set_seg_warp(0)`(§2.3)、分段 dot 走 binned。

### 3.5 S4 active mask 与线性系统掩码

**【稳定线+phase-cd】**(S4-dev 设备派生 mask 全套机制——`_mask_from_env_alpha`、
`STIFF_PERENV_MASK_DEV` 门、周期性 all-active recheck——在稳定线 v0.8.5.3 单体
GIPC.cu 中已完整存在,:10904-10916, 15467-15470, 16301-16305;与 phase-cd 的
`11_perenv_machinery.inl:215-232` 同源,**非** phase-cd 新工作)

- `_mask_from_env_alpha`:active = (α != 0);设备派生 mask,零新增 D2H——冻结决策已在
  `m_env_alpha`(来自真实 solve);被 mask env 下一迭代 moveDir=0(RHS 清零)→
  持续 mask 直至周期性 all-active recheck 重解以检测 bounce-back。固定节奏、per-env 决策 ⇒
  strict/batch-invariance 安全(11:216-232)。
- 注入线性系统:`set_env_mask` → `solve_linear_system` 清被 mask env 的 RHS(§1.2 步 8),
  SpMV 跳其 triplets(spmv.cu:114-122)。
- **segmented PCG 即使 mask 关闭也注册 DOF→group 映射**(active=nullptr ⇒ 无 RHS masking);
  scalar 与 segmented kernel 永不混用,改 batch 数绝不静默改变数值算法
  (ipc_solver.inl:1299-1306;pcg_solver.cu:715-722)。

---

## 4. 确定性与门禁

### 4.1 det 栈原理:binned 指数分箱归约

**【稳定线+phase-cd】** 核心原语 `linear_system/utils/binned_reduce.cuh`:

常量 `BINNED_K=4, BINNED_W=30, BINNED_E0=60`(:10-14)。K 个指数 bin,每个是**锚对齐的
精确定点切片**;向一个 bin 的原子沉积是**精确的**(无舍入)⇒ **顺序无关 ⇒ 与线程调度
无关、位级一致**,同时保有全动态范围。`__dadd_rn/__dsub_rn` 阻止编译器重结合。

```
binned_deposit(bins, val):
  g_det_reduce == 0(非 strict 快路径): atomicAdd(&bins[0], val)      // 一条裸原子
  g_det_reduce == 1(strict):
     for k in 0..K-1:
        M  = ldexp(1.5, E0 − k·W)          // 锚
        hi = ((val + M) − M)               // 该指数窗内的精确部分
        atomicAdd(&bins[k], hi);  val −= hi
binned_combine(bins):  细 bin 在前、固定顺序求和
```

(:24-51;ABD 12 向量/12×12 矩阵版本 `bin_add12/bin_add144` :54-76。
K/W/E0 覆盖窗之外的溢出/下溢行为未从代码推导,**待核实**;
验证器 `tools/binned_reduction_test.cu`。)

**中央门 `g_det_reduce`**(energy/03_barrier_fused_assembly.inl:36-42):
定义 `__device__ int g_det_reduce = 1`——**保守默认 1**:首个 computeGradientAndHessian
之前发出的沉积(frame-0 init 能量等)必须已经序无关,否则 strict 的 run-to-run 位一致在
第一次 line search 就被播下破坏(实测 default 0 在 foldshirt strict f0k2 即分歧)。
**闩锁只会为 merged/isolated 松到 0**。设置点
(13_kappa_partition_gradhess.inl:798-810,computeGradientAndHessian 顶部):

```cpp
int det = (m_mode_config.spmv_det || getenv("STIFF_DIAG_BINNED_GRAD")) ? 1 : 0;
// 值变化时 set_binned_on(det); set_det_reduce(det);
```

即**正门控:由 STIFF_SPMV_DET(strict 设)统一驱动全部 binned 用户(fem/MAS/ABD)**;
值追踪而非 once-latch(旧 once-latch 使同进程第二个 engine 继承第一个的模式)。
梯度 combine `_gfxToGrad` **必须求和全部 K 个 bin**(沉积者是混合的;只读 bin0 曾造成
κ 建议是"舍入到 ulp(2^60·1.5)=256"的垃圾 → line-search 死亡螺旋,case_26 家族,:63-69)。

det 栈成员一览(strict 正门统一驱动):FEM/ABD 能量与梯度沉积、MAS 三处聚合(§2.5)、
det SpMV(§3.4e)、分段 binned dot(§2.3)、converter binned 归并(§1.5,天然序无关)。

### 4.2 金锚(bitwise anchor)机制

**【仅 phase-cd 的门禁脚本;锚语义两线共享】**

- 锚场景 `scripts/anchor_scene.py`:2 ABD cubes + 2 FEM cubes,groups [0,1,0,1],
  strict 旗标,50 帧;打印 `VHASH = sha256(V.tobytes())[:16]`(最终顶点 buffer)。
  Config 钉死:dt=0.01、density=1e3、young=1e6、fric=0.4、relative_dhat=1e-3、
  `absolute_dhat=1e-3`(钉绝对 dHat:merged-bbox 相对 dHat 会让接触宽度随全 env
  包围盒变化,破坏跨批次镜像)。
- G1 门(`scripts/verify_gates.sh`):`GOLD_ANCHOR="0544461bd82123ae"`;strict 模式跑锚场景,
  VHASH 与金值**严格相等**;全套门禁在 `STIFF_MIRROR_AUDIT=1 STIFF_SLOT_AUDIT=1` 武装下跑
  (审计已证 bit-transparent)。金值同时钉在 mode_gates.py/bvh_candidate_gate.py/
  mode_bench.py 三处。
- G9(mode_gates.py):三承诺——resolver 路径的 strict VHASH == gold(与 G1 的显式旗标
  路径互证);merged/isolated NEWTON 包络 == 记录基线;跨模式等价
  max|Δpos| < 1e-3(实测 isolated-strict 2.26e-9、merged-others 6.02e-5,16× 裕度)。

**锚的世系**(两个 hash 的关系,权威出处 ../RELEASE_NOTES_v0.8.6-rc1.md):

| 锚 | 世代 | 说明 |
|---|---|---|
| `f7fb5a786c2d7935` | dlto 采纳前(v0.8.5 世系;v0.8.6 重构全程不动,含跨架构 4090 sm_89 ≡ A800 sm_80) | 稳定线 v0.8.5.3 属此世系(推断,稳定线树内未跑门禁验证金值,**待核实**) |
| `0544461bd82123ae` | dlto(device link-time optimization)采纳后的新不动点(45d74f0 起,phase-cd 现行) | 全引擎换锚 + 跨架构重验:A800 sm_80 同值、两跑逐位稳定 |

**锚的用途模式**:任何改动后门禁全绿 + 锚逐位原位 = "确定性栈没被碰";若改动**合法地**
移动数值(如 dlto),则换锚并跨架构重验——这是显式工程决策,不是门禁豁免。
例:C6-w 跳零默认开后锚不动(strict 跑 det 栈,该变更在其上可证中性)。

### 4.3 strict 锚:跨架构与批次不变性的契约地位

- **跨架构**:已验证的架构对是 sm_80 ≡ sm_89(dlto 前后各一次锚证据)。
  wheel 同时发 sm_120,但 **sm_120 无锚证据**——不应外推为任意架构承诺。
- **批次不变性(cross-batch-size identity)**:`stiff_physics/engine.py:187-190` 的公开
  措辞是"strict 保证固定场景 + 固定 batch layout 的可重复性;**cross-batch-size identity
  仍是验证目标而非公开契约**"。稳定线的 `examples/test_strict_quadgate.py` 五门自检
  (r2r@N=2、r2r@N=8、cross-env@N=8、batch N=2-vs-N=8、对角 PC 交叉)在默认 MAS 下
  bit 级持续断言它——工程上成立,契约上不承诺。文档与下游使用请保持这个双层表述。
- **strict 性能代价的口径**:mode_contract.h 说"anchor 类场景个位数 %";整场景矩阵
  (THREE_MODE_MATRIX_rc2)给出 towel 8.1s vs merged 6.8s、foldshirt 61.6 vs 40.6、
  A800 盘子 45.0 vs 26.1s(+20%~2×)。两个口径不同(前者近于单项 canon 开销),
  引用时须注明场景。另一面:strict 的峰值 Newton **多数场景最低**
  (MODE_MATRIX_REPORT_2026-07-27:towel 61 vs 94/98、盘子 15 vs 22/19——canonical 序
  降低最坏帧),但**非普适**:rc2 重跑的同一 A800 盘子场景 merged 峰值 14@fr164 <
  strict 15@fr202(THREE_MODE_MATRIX_rc2:31-33)。两批实测数字不同,引用时注明出处批次。

### 4.4 图门禁的两层验证

**【仅 phase-cd】**(`scripts/frame_graph_gate.py`,G16;C6-w 重构后):

1. **det 层**:图-对-宿主逐位 digest 比较在 `STIFF_SPMV_DET=1`(merged 模式、其余 det
   旗标显式=0)下跑——binned 级联使"graph 不得改变任何一个 bit"成为**关于代码的数学命题**
   而非 GPU 调度巧合(:287-299)。
2. **默认层**:非 det 栈上,图结果须落在**基线自身 run-to-run 包络**内:3 次基线跑,
   noise = 两两 max|Δ|,`budget = max(4·noise, 1e-11·scale)`(:356-386)。
   实测 G16 error=0.0 vs noise=3.3e-16;G17a error=0.0 vs 4.5e-15。

**残余分歧的定位**:即使 SPMV_DET 开,towel 上 full-graph-vs-release 仍在 frame 31 出现
1.816e-07 分歧——那是**容量网格归约的合法浮点重结合**(不违反任何契约:strict 本尊被挡
在图外,§5.2 资格规则)。同构门禁:episode_graph_gate.py(G17a)、gpu_rl_gate.py(G17c)。

### 4.5 checkpoint 与确定性

**【仅 phase-cd】**(v2 STIFFCP2 语义整体只在 phase-cd 存在。稳定线的
`save/load_checkpoint` 是**调试级格式**——magic `0x53544B50` "STKP",仅存 FEM 4 数组 +
ABD q/q_prev/q_v + Kappa + total_Frames;无模式旗标位图、无格式版本号、无换模式恢复
硬拒;且 load 端 magic/计数不匹配时只 printf `[ckpt] MISMATCH` 后**静默 return**,
继续用未还原状态(稳定线 GIPC.cu:16591-16631,GIPC.cuh 注释自称 "[decouple debug]
Full-state checkpoint")。本节的模式入档/硬拒/~5e-17 重启差**全部只在 phase-cd 成立**)
(`checkpoint/checkpoint_io.cu`):

- **模式入档**:14 个模式旗标打进 u32 位图,连同 mode 写头(:292-309, 874);
  载入时 `saved_mode != 当前 mode || 旗标位图不等` → format_error
  ("execution-mode configuration differs from checkpoint")——**换模式恢复硬拒**(:970-972)。
- **帧入口配对集重建**(:1224-1242):每帧第一个 Newton 迭代跑在上一帧最后一次
  line-search buildCP 留下的接触对集上(跨帧状态,有意不入档)。恢复时从刚还原的位置
  重建(同位置 → 同 SET)⇒ 重启差从 ~5e-6 降到 **~5e-17**;与摩擦锚/Kappa 同一
  "恢复时重构"原则。同时强制一次 BVH 质量重建。
- 门禁 G11:rc2 起收紧为 "strict 逐位 / 其余 1e-12"。

---

## 5. 帧事务与整帧 CUDA Graph【仅 phase-cd】

> 本节全部内容(两图事务、整帧图、条件图录制器、设备 resizer、容量训练)是 phase-cd
> 专属,稳定线不存在 `StiffGIPC/frame_fsm/` 目录(亲验)。
> 所有相关旋钮默认**关**(`STIFF_FRAME_GRAPH`/`STIFF_FRAME_FULL_GRAPH` 双 opt-in)。

### 5.1 两图事务(root + terminal)

`STIFF_FRAME_GRAPH=1` 时 `SimEngine::step()` 分流到 `IPC_Solver_FrameGraph`
(engine_modules/03_step_getters_export.inl:72-78)。事务把每个物理帧包进
"帧首快照 → 求解(宿主或图)→ 帧尾验证/还原/序列化"的三段协议:

- **稳定 ABI**(frame_fsm/frame_status.cuh):
  - `FrameResult`:`FRAME_OK=0, FRAME_RETRY_REQUIRED=1, FRAME_FATAL=2, FRAME_RUNTIME_ERROR=3`;
  - `FrameInvalidBits` 低位 = 物理无效类(`INV_CCD_GROUND=1<<0 … INV_NAN_STATE=1<<8`),
    高位 = 容量溢出类(`OVF_DCD_PAIRS=1<<16, OVF_CCD_PAIRS, OVF_TRIPLETS,
    OVF_UNIQUE_BLOCKS, OVF_MAS_CLUSTERS=1<<20`);
  - `FrameStatus`(alignas(16),`static_assert(sizeof <= 256)`)= **帧尾唯一 D2H 包**;
  - `FrameDeviceState` = 图执行期 GPU 独占状态;`fsm_record_error` 用 atomicCAS **只保首错**
    (该语义影响 OVF 轴掩码的可得性,§5.4)。
- **root graph**(capture_root_graph,frame_transaction.cu:1055-1133):
  h_begin H2D + `frame_begin_init`(重置 FrameDeviceState)+ **设备快照 D2D 拷贝**(§5.3)。
  root 图内零 D2H。
- **terminal graph**(:1135-1212):`frame_terminal_apply`(合入宿主报告的计数)→
  `frame_validate_finite`(逐顶点有限性,非有限 → ERR_NONFINITE_STATE + FRAME_FATAL)→
  `frame_restore_fem/abd/kappa`(**当且仅当 result != FRAME_OK** 时从快照还原;OK 帧
  核内 early-return,保证成功帧逐位不变)→ `frame_serialize_status` → status D2H
  (事务唯一 D2H)。
- 宿主三段:`frame_graph_begin`(容量保障 + `snapshot_host_attempt` 快照 ~25 个宿主镜像 +
  launch root)→ `frame_graph_enqueue_terminal`(重复发射 throw)→
  `frame_graph_finish_terminal`(读 h_status;**OVF 裁决/容量增长的唯一合法点**;
  失败帧 `restore_host_attempt`)。审计断言:root 图 d2h==0、terminal 图 d2h==1,
  违反 throw(:3913-3915)。

### 5.2 整帧图(whole-frame conditional graph)

`STIFF_FRAME_FULL_GRAPH=1` 叠加层:整帧(Newton WHILE、PCG WHILE、LS WHILE、收敛 IF)
录成一张条件图,每物理帧 = 一次 `cudaGraphLaunch` + 一次 FrameStatus D2H + 一个宿主帧边界。

**资格检查 `full_graph_eligible`**(frame_transaction.cu:251-439),拒绝理由逐条:

| 拒绝条件 | 说明 |
|---|---|
| 诊断旋钮活跃(MIRROR/SLOT_AUDIT、PHASE_TIME、KSUM、MAS_DUMP、MAS_FUSE_VALIDATE、STACK_DIAG、NAN_DIAG) | capture-incompatible host diagnostics |
| ABD 存在且普通 step() 无 `STIFF_C6_ABD_STEP_GRAPH=1`;ABD unique tier 未训 | episode 路径经 `allow_abd=true` 放行 |
| FEM-to-ABD pins;移动边界;宿主持有的 soft target;semi-implicit;多子步动画 | 非驻留态 |
| 碰撞开启但无 `STIFF_C4_COLLISION_GRAPH=1` **且非** episode 捕获 | episode/GPU-RL 捕获自带资格([self-contained prepare]) |
| **Strict 模式 / ee_detgate / ccd_canon** | "strict determinism controls are active"——strict 的承诺是位级复现,而容量网格归约合法重结合求和;**准入 strict 将是换锚决定而非驻留决定**(:365-388;用户 2026-07-29 决定) |
| Isolated 无 `STIFF_C5_ISOLATED_GRAPH=1` 等 | isolated 完整 bundle + 旋钮才可入图 |
| merged 上的**部分** per-env overlay | 半配置拒绝(:427-437) |

**准入的 det 旋钮**(C6-l):`spmv_det`(数值路径选择而非 strict 锚;整帧配置的确定性
杠杆——det-SpMV 杀掉 armed-layout SpMV 序竞态)与 `ee_canon`(canon-slots 使图内接触
能量求和序无关;没有它 towel 两跑之一在首次接触爆炸)。
想要"整帧图 + run-to-run 确定"的配方是 **merged + `STIFF_SPMV_DET=1 STIFF_EE_CANON=1`**,
不要开 EE_DETGATE/CCD_CANON(会触发资格拒绝)。

**小场景保护**:`tiny_scene_for_full_graph`,默认 `min_verts=1024`
(`STIFF_FULL_GRAPH_MIN_VERTS` 覆盖,0=关):sub-1k 场景整帧图付 10-80×;
只谢绝 FULL 图,两图事务保留(tier 训练与 episode 契约依赖它)(:2376-2394)。

**录制结构**(`capture_full_graph`,:1214-1485):`graph_resize::begin()`(每次录制
**独占 16 槽环中一个设备槽数组**——[C6-aa] 共享数组版曾让 episode 重放 resize 到 frame 图
的节点)→ 序幕(h_begin H2D + frame_begin_init + **FEM 4 数组 + ABD q 族 4 数组 +
kappa 的 D2D 快照**)→ 帧体(`ConditionalGraphRecorder` scope 内
`enqueue_frame_graph_body`)→ 帧尾(updateVelocities + computeXTilta + validate +
restore 三件套 + serialize + 唯一 D2H)→ EndCapture 后 `graph_resize::publish()`、
传输审计(**硬断言 d2h==1 && host==0**)、Instantiate + Upload、
`full_generation = pcg_buffer_generation()`。捕获抛异常 → 永久 `full_capture_failed`,
回退两图事务。

**帧体**(`GIPC::enqueue_frame_graph_body`,core/ipc_solver.inl:896-1146):
收敛阈值 `threshold = newton_velocity_tol>0 ? newton_velocity_tol·dt :
sqrt(Newton_solver_threshold²·bbox2·dt²)`(与宿主 `_newton_thr` 必须逐位一致);
Newton WHILE 体 = `enqueue_pair_tier_guard`(**每个 Newton 迭代顶部**——帧尾-only 时代,
帧中 DCD 溢出让循环在截断 pair 集上饥饿自旋,同帧两跑实测 369 vs 1000 迭代,曾是整帧
replay 的主导非确定源)→ snapshot → computeGradientAndHessian →
calculateMovingDirection(PCG WHILE)→ 收敛判定(merged:`calcMinMovement_DeviceOut` +
decide;isolated:per-env α 链 + 全 env 冻结判定)→ IF(step)内 CCD α 链 +
lineSearchConditional(isolated 为 per-env S3 WHILE + uniform 回退 IF)+
post-LS kappa 条件加倍 → `_newton_tail_conditional`。

**条件图录制器与设备 resizer**:
- `ConditionalGraphRecorder`(conditional_graph.h:22-75):`while_loop`/`if_then` 在捕获
  前沿插条件节点;一线程一 recorder(thread_local)。CUDA 陷阱:每个嵌套 WHILE 前须
  显式重设条件句柄,仅靠 default assign 会沿用上一轮的终止零值。
- `gipc::graph_resize`(graph_node_resize.h):[C6-v] kernel 节点标记
  `cudaLaunchAttributeDeviceUpdatableKernelNode`,1 线程 resizer 节点在**同一次重放内**
  读设备计数并 `cudaGraphKernelNodeSetGridDim`(sm_89/CUDA 12.8 验证);
  `STIFF_GRAPH_DEVICE_RESIZE` 默认**开**(C6-y 后)。

### 5.3 设备快照集:fem + abd + kappa

事务快照集(root 图与 FULL 图序幕必须一致):

| 快照 | 内容 | 还原核 |
|---|---|---|
| FEM 4 数组 | `fem_vertexes / o_vertexes / velocities / x_tilta` | `frame_restore_fem` |
| ABD q 族 4 数组 | `abd_q / q_prev / q_v / q_tilde`(`gipc::Vector12`) | `frame_restore_abd` |
| kappa | `kappa_snapshot`(+ per-group `kappa_group`) | `frame_restore_kappa` |
| 宿主镜像 | `snapshot_host_attempt` ~25 项(计数、h_cpNum/h_gpNum 及 lagged、triplet offsets 等) | `restore_host_attempt` |

历史上 `capture_full_graph` 曾**漏抄 ABD q 族 + kappa**(只快照 FEM 顶点):失败回滚只还原
顶点、q 保持 attempt 中值 → 破坏引擎级不变量 `x ≡ J·q` → 宿主重解的 E0 读还原顶点而
每次线搜索由污染 q 重导出 vert=J·q → 回退帧 line search 必然耗尽。修复(4e49fda)后
图通道 ls_exhaust 2-3 → 0。修复史详见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。

### 5.4 OVF 回退协议:尝试→回滚→宿主重解→扩容→重录

`IPC_Solver_FrameGraph`(frame_transaction.cu:4440-4703)的重试循环:

1. **帧 0 恒走 release 求解器**(warm-up 边界:分配/惰性工作区安定后图才缓存指针)。
2. attempt 循环(`STIFF_FRAME_MAX_RETRIES` 默认 3,clamp [0,16]):
   - 尝试 `try_launch_full_graph`(资格/tiny/曾永久失败 → 回两图路径);
   - 图内溢出:发射按容量截断,设备写 OVF 位 + `required_*` 需求、**拒绝提交坏帧**;
     [C6-o] solve_subIP 逐迭代 12B 异步状态轮询立即中止(否则空转烧 150k PCG 迭代);
   - 帧边界 `frame_graph_finish_terminal` 裁决:失败帧从快照**逐位还原**;
   - **[C6-i] 溢出帧不在图内重试**:默认 `capacity_fallback=true`,从还原状态用 release
     求解器解完本帧——重录变 grid 形状与归约序、重试是否发生取决于贴档 racy 原子计数,
     ~1e-6 扰动被混沌放大(towel crumple 0.77..1.11 图内 vs 宿主确定 0.905);
     本帧边界裁决的增长塑形**下一帧**的重录(`STIFF_GRAPH_INGRAPH_RETRY=1` 恢复旧行为,
     【实验性,默认关】);
   - 宿主回退帧的两个位级保证:[C6-m] `LayoutOverrideOff` RAII 关掉容量布局机器 ⇒
     fallback 帧逐位等于图关帧;[C6-l] attempt>0 先 `buildBVH()+buildCP()` 重建对集
     (aborted 录制留下的是截断 racy 子集;相同位置给出相同 SET,canon 槽序使数组位级
     可复现;`STIFF_FALLBACK_NO_REBUILD` 诊断关);
   - **扩容**(帧边界,宿主,唯一合法点):按 OVF 位分轴增长——
     `OVF_UNIQUE_BLOCKS` 按 `required_unique_blocks << escalation` 取档;
     DCD/CCD 各在自己的轴上长(needed=trained+1 再乘 headroom 取档;设备报告的跨越轴
     掩码打包在 `err_primitive`,拿不到时**只长 cp[0]**——"grow every axis" 曾致
     512k→2.77M→11.8M→OOM);`OVF_TRIPLETS` 按类长档后在合法 discard 窗口重定容量;
     增长必 `++pcg_buffer_generation()`(不 bump 代际的增长对录制路径是死码——retry
     重放同一 exec 同一 baked extents);
   - **增长 streak 升级**([C6-o/C6-aa]):同一轴 8 帧内再次跨档 → 每连击多左移一档,
     **封顶额外 2 档**(`escalation_peek` 返回 `min(streak+1, 2)`,即"每连击 2×,
     至多 4×",frame_transaction.cu:4165-4176;非无界升级)
     (`escalation_peek/commit` 拆分,防未增长路径推进 streak);
   - retryable = `(RETRY_REQUIRED || tier 真长了) && (invalid_bits & OVF 位) && 非诊断注入`
     ("tier 真长了才重试"是终止性保证);预算尽 → `ERR_RETRY_EXHAUSTED` + throw。
3. **重录**触发条件:`!full_exec || full_generation != pcg_buffer_generation()`——
   即下一帧发现代际变化时以新容量重录。

行搜索预算耗尽(INV_LS_BUDGET)**非致命**:计数 + WARN,与宿主策略一致(接受非下降步)。

### 5.5 容量档训练与 headroom

- **契约**(`train_collision_for_capture`,:2032-2374):**所有图内碰撞缓冲必须在捕获前
  达最终容量**——先显式 tier 数学(`train_collision_graph_capacities`),再以容量镜像
  **干跑**装配+求解的宿主 builder(converter staging、radix-sort temp、预条件 per-level
  缓冲、ABD 装配档在真实路径上于捕获外长到位);垃圾 triplet/梯度由图序幕覆盖,宿主记账
  由快照回滚。step 整帧路径与 episode/GPU-RL 捕获共用。
- **训练档成员**(GIPC.cuh:951-1025):`m_graph_train_cp[5]`(分 arity 的 DCD 档——
  triplet 流大小 = tier(n4)·M12 + tier(n3)·M9 + tier(n2)·M6,单一 pairs 数定不了);
  `m_graph_train_ground`(**地面轴恒取最坏 surf_vertexNum**——地面无 OVF 位,截断不可
  检测);`m_graph_train_ccd` 独立训练(foldshirt DCD ~27k vs 扫掠 ~324k,按比例推导
  曾迫使 4× DCD 膨胀爆显存)。tier 按**峰值**而非帧末训练(foldshirt 帧末 ~58k vs
  峰值 ~390k)。
- **headroom**:`graph_train_headroom_num()` 默认 **1**([C6-p] 从 2 翻转;tier 宽是录制
  图内的 launch 常量,2× headroom 曾使 converter unique-block 归并单项 86ms/帧@8.4M 槽
  对 4.2M payload;实测图开倍率 forcegrip 4090 2.08×→1.39×、A800 3.25×→1.75×;
  代价 = 每 run 1-2 个额外增长帧,覆盖 97%→94%)。`STIFF_GRAPH_TIER_HEADROOM` 覆盖
  [1,64]。[C6-g]:headroom 同时是**确定性**旋钮——贴档边界的 racy 接触计数有时跨档
  有时不跨 → retry 非物理中性(headroom=2 症状 = 门禁边缘化通过/失败)。
- [D4 fix] `converter.ensure_capacity(2×payload_tier)` 无条件 reserve(干跑与录制 pass 的
  exact baked counts 漂移几个百分点);retier 后再干跑一遍(一次收敛)。
- **不再从容量膨胀的 payload 重推 class tier**(曾致 246k→2.6M→21M 复利爆炸);
  唯一窄例外:纯 ABD 场景按实测 payload 重定 abd_abd 档([D4-b])。

---

## 6. episode 图与 GPU 驻留 RL【仅 phase-cd】

### 6.1 两种形态与 GPU-native 契约

`EpisodeGraphContext`(frame_transaction.cu:170-240)按 `device_native` 分两形态:

| 形态 | 用途 | 特征 |
|---|---|---|
| **open-loop episode**(`launch_episode_async` 家族) | 回归/吞吐实验;**不满足** GPU-native 定义 | 动作整段预上载;观测 pinned 双缓冲 + event 围栏;`finish_episode_graph()` 收尾 |
| **GPU-native RL**(`device_native=true`) | 稳态 RL 循环 | `completion_event`;`d_joint_obs`(图自身刷新);in-stream reset 快照;`d_frame_counter`(int64) |

**GPU-native 定义**([../GPU_NATIVE_RL_PLAN.md](../GPU_NATIVE_RL_PLAN.md)):稳态转移
`device action → sim step → device obs/done → device reset → next action` 全留流上;
正常 RL 步不得含 H2D/D2H payload 拷贝、宿主等待、决定控制流的宿主决策、拓扑分配或图
重捕获。setup/诊断/checkpoint/渲染可跨宿主边界。reward 按契约归 policy 侧(任务相关,
不属物理引擎)。**捕获审计 fail-closed**:device_native 图硬性要求
`host==0 && h2d==0 && d2h==0`,违反 throw(:1957-1990)。

### 6.2 动作预上载与观测双缓冲(split-frame)

- **动作布局**:`d_actions` 一块——revolute 打包区 + `prismatic_action_offset`
  (= revolute_bytes)起的 prismatic 区;`RevoluteDrivingControlPacked` /
  `PrismaticDrivingControlPacked` 均 static_assert 为 3×Float 紧排(:2931-2939)。
  语义:revolute = {target angle, strength ratio, external torque};
  prismatic = {target distance, strength ratio, external force}(float64[joints,3])。
  设备端按 `frame_index` 索引动作序列(`enqueue_episode_driving_targets`),逐帧不回宿主。
- **观测双缓冲**:`split_frame = (frame_count+1)/2`(:2900);图组织为
  "两段 WHILE + 两个 root 观测出口"——CUDA 不允许 host-queryable external event 节点
  进 conditional body(PHASE_C_FRAME_GRAPH_PLAN.md:159-165)。每个出口
  `enqueue_episode_slot_copy`:positions/velocities/statuses → pinned;
  `cudaEventRecordWithFlags(slot_event, cudaEventRecordExternal)`(capture 内普通 event
  record 只是图内依赖,不可宿主查询);最后 `d_ready_one → h_ready[slot]` 发布哨兵。
- **每帧结构**(`enqueue_iteration`):`episode_frame_begin`(GPU-native 下
  `frame_id = *frame_counter + frame_index`)→ FEM/ABD/kappa D2D 快照(同 full 图)→
  设备驱动目标 → `enqueue_frame_graph_body` → updateVelocities/computeXTilta →
  validate/finalize/restore → `episode_store_frame`(提交态写进
  `episode_positions[frame·V+v]`、per-frame FrameStatus、推进 attempted/successful)→
  `episode_tail`(OK 且 next<frame_count 时 frame_index++)。
- **[C6-aa] resize 槽独占**:episode 构造从前不 publish 自己的 resizer 槽,其 resizer
  读到的是上一个已发布构造(frame 图)的节点句柄并 resize 了**那些**——16 槽环独占 +
  episode 图首次真正武装是该缺陷的修复(:1654-1657 注释)。

### 6.3 API 面(GIPC / SimEngine / Python)

**GIPC 层核心**(frame_transaction.cu;签名与抛错条件全部亲验):

```cpp
void GIPC::prepare_episode_graph(device_TetraData& mesh, int frame_count,
    const RevoluteDrivingControlPacked* revolute_actions, int revolute_count,
    const PrismaticDrivingControlPacked* prismatic_actions, int prismatic_count,
    bool device_native);
// 抛错:frame_count<=0 / joint 计数为负 → invalid_argument;
//       m_total_frames==0 → logic_error("one synchronous warm-up step is required ...");
//       另一图事务活跃 → logic_error;STIFF_DRIVE_SUBSTEP>1 → runtime_error;
//       资格不符 → "[episode-graph] unsupported scene: <reason>";
//       动作 shape 与关节数不符 → invalid_argument。
// exec 复用:frame_count/vertex_count/关节数/device_native/generation 全同 → 只 upload_actions。

void launch_episode_graph_async(device_TetraData&, int64_t base_frame_id);
// device_native → logic_error(用 gpu_rl 版);in_flight → logic_error;
// 代际变化 → runtime_error("graph buffers changed; prepare the episode again")

void launch_gpu_rl_graph_async(uintptr_t cuda_stream);
// 要求 device_native 且 frame_count==1("closed-loop execution requires a one-frame graph");
// 流亲和:in_flight 后换流 → logic_error("repeated launches must use the same CUDA stream;
// end_gpu_rl() before changing streams");每发射后 cudaEventRecord(completion_event)

void launch_gpu_rl_episode_graph_async(uintptr_t);   // device_native 多帧一次发射整段
void launch_gpu_rl_reset_async(uintptr_t stream);    // [D2] 纯 D2D 重放 prepare 时提交态
                                                     // (FEM 4 数组 + ABD q 族);帧计数器故意不清
void launch_gpu_rl_reset_masked_async(uintptr_t d_env_mask, uintptr_t stream);
// [D3] 按 p2g/b2g 查 env 组,mask[g]!=0 才还原;组=-1 永不经掩码路径;
// mask 空指针 → invalid_argument;无 p2g 且有顶点 → logic_error

int  finish_episode_graph();     // 等 slot1、逐帧累加成功帧(遇非 OK 停)、宿主计数合入;返回成功帧数
void destroy_episode_graph();    // in_flight 时先 drain 整条流再释放
// 查询/指针导出:gpu_rl_graph_prepared/ready/bound_stream/synchronize_gpu_rl_graph;
// gpu_rl_{revolute,prismatic}_actions_device_ptr / {positions,velocities,statuses,
// frame_counter,joint_observations}_device_ptr;joint_observation_count = 2·rev + 2·pri
```

**SimEngine / Python 表面**(engine_modules/03_step_getters_export.inl;
stiff_physics/engine.py:1205-1373):

- `SimEngine::prepare_gpu_rl()`:[self-contained prepare] Isaac 式契约——warm-up 在纯宿主
  布局跑过时,`LayoutForceOnScope` 强制容量布局并**内部多跑一帧 step() 训档**
  (python docstring 声明 prepare 会推进一帧仿真),再
  `prepare_episode_graph(mesh, 1, nullptr, rev, nullptr, pri, true)`。
  `prepare_gpu_rl_episode(frame_count)` 同理(要求 >1)。
- **step() 自动分流**:`gpu_rl_graph_prepared()` 时 step() 变薄发射(宿主 joint 目标
  异步 memcpy 到设备动作 slab + `launch_gpu_rl_graph_async(bound)`);episode in flight
  时 step() → LifecycleError。实测薄发射 3.14ms ≈ 手工 launch 3.18ms(表面统一零开销)。
- Python:`prepare_gpu_rl() / prepare_gpu_rl_episode(frames) / launch_gpu_rl_async(cuda_stream=0)
  / launch_gpu_rl_episode_async / gpu_rl_prepared() / gpu_rl_ready() / synchronize_gpu_rl()
  / end_gpu_rl() / launch_gpu_rl_reset_async / launch_gpu_rl_reset_masked_async /
  get_gpu_rl_device_abi() -> dict / gpu_rl_tensors() -> dict`。
  `gpu_rl_tensors()` 返回 **torch 零拷贝视图**(`__cuda_array_interface__` v3 包装裸指针;
  键 actions_revolute/actions_prismatic/positions/velocities/joint_observations/
  statuses(uint8)/frame_counter)。**流纪律**:引擎在 per-thread default stream 入队,
  torch 默认 legacy stream 隐式有序;非默认 torch 流须先 `synchronize_gpu_rl()`。
- 设备指针有效期至 `end_gpu_rl()`/reset/析构;**重复发射必须同一 CUDA 流**
  (流序替代宿主栅栏)。

### 6.4 零 D2H 契约的实证

([../GPU_NATIVE_RL_PLAN.md](../GPU_NATIVE_RL_PLAN.md):53-128;节点数依场景而变,勿引单一数)

| 平台 | 证据 |
|---|---|
| 4090(2026-08-05,Nsight `--cuda-graph-trace=node`) | 1057 节点图(contact+friction);40 步窗:40 次 cudaGraphLaunch、40 非阻塞 event、80 笔 24B D2D 动作发布、5655 图节点事件、**0 H2D / 0 D2H**、2.461ms 提交窗内 0 同步 API |
| A800(2026-08-06,sm_80 Release+DLTO) | 同 1057 节点、43/43 帧 result=0;11.571851ms 提交窗内 0 传输 0 同步;工件 `artifacts/a800-fa6e03e/` |

精确表述是"**稳态提交环内零同步**"(profiler 停机边界外有一次 cuCtxSynchronize);
"CPU 不在场"是错的:每步仍有 1 次图入队 + 2 笔小 D2D 发布(CPU **在场**但不**参与**)。

---

## 7. 执行通道选型结论

定稿架构(权威出处 [../SIMULATOR_EXECUTION_DESIGN.md](../SIMULATOR_EXECUTION_DESIGN.md)):
**一个表面(Isaac 式 `step()`)、两个内核**——

| 通道 | 构成 | 主场 |
|---|---|---|
| **step 通道**(默认) | 宿主帧骨架 + PCG 自发射图岛(K=8 设备收敛,双通道默认开)+ 每帧边界事务 | **大帧回放最优** |
| **residency 通道** | 整帧图 + 设备 resizer + 多帧连发 + 段边界事务;`prepare_gpu_rl()/end_gpu_rl()` 显式声明,永不按帧自动猜测 | **RL 稳态微步最优** |

**实测总表**(引用出处内注;60 帧 3 轮中位/交错多轮):

| 负载 | v0.8.5 | phase-cd step(默认) | phase-cd 图开/驻留 |
|---|---|---|---|
| forcegrip 60f 4090 | 115.2ms/帧 | **105.5ms(−8%)** | 图开 +14%~43%(战役期不同口径) |
| beaker 60f 4090 | 160.8ms | 166.8ms(+4%;**差 = 纯启动段 ~0.6s,每帧持平**——"退 7%"已破案) | 图开 +35% |
| finray 60f 4090 | 289.5ms | **270.7ms(−7%),方差 233–315 → 267–280 收窄** | — |
| RL 微步 D4 4090(ms/步) | 4.00 | 3.14(1.27×,step() 薄发射) | 3.18(手工 launch,一致) |
| RL 微步 D4 A800(ms/步) | 19.3 | **9.7(2.0×)** | **驻留 3.85(5.0×)** |

- **大帧回放:宿主(step 通道、图关)最优**。图开 = 墙钟 **+3%~75% 场景相关**
  (OPTIMIZATION_ROADMAP 图通道结论);大帧整帧图列入死路表(结构性负收益:发射开销
  占比过小,容量成形 + 事务保险 + 轨迹效应为净成本)。图开的唯一附带价值:尾延迟可预测
  (方差 ±8% vs 宿主 ±21%)。
- **RL 微步:residency 通道唯一主场,vs v0.8.5 收益 ~1.3×(4090)到 5.0×(A800)**
  (即本节表:D4 4090 4.00→3.18ms/步,A800 19.3→3.85ms/步)。另一口径的
  1.8–2.8×(A800_ALLEXAMPLES)是 **HEAD 内部** episode-vs-宿主对比,勿与 vs-v0.8.5
  区间混拼。收益 ∝ 宿主往返延迟,是**平台的函数**
  (τ:A800 集群 ~15ms/步、4090 ~1ms/步)——"决定胜负的是 regime,不是特性"。
- **容量平稳性第二判据**:接触逐帧升级的轨迹(foldshirt 抓握)会击穿训练档——第 15 帧
  auto-prepare 后 25/25 帧 OVF_TRIPLETS 失败(fail-closed 成立)。**真实大帧操作轨迹
  必须走 step 通道**:这是驻留合同与工作负载的结构矛盾,不是速度问题。
- **成对审计 +1% 的证据链**(`STIFF_FRAME_FORCE_ROLLBACK=2`,commit 4219f37,
  A800 foldshirt 4-env merged 1551 帧):图解完整帧后边界翻转 RETRY → eager 重放 restore
  核 → 宿主用 release 求解器把同帧再解一遍 ⇒ 每帧一个**同状态成对样本**。结论:
  - 同状态图 vs 宿主 Newton **+1.0%**(869/1550 帧逐位相等)——图内求解器语义
    **不是**生产差距;
  - 纯宿主 8808±60 | 仅事务壳(无 C4/C6 无图)9342±230 | 生产整帧图 9697±35 ⇒
    **+10% 生产盈余的主体(~+6%)= 事务壳的 ULP 级逐帧扰动 × merged 接触混沌轨迹效应**
    (15σ 聚类分离排除 run-to-run 运气),另 ~+4% 为图化增量;
  - 回滚兜底循环本身非物理中性(强制逐帧回滚的 D 臂零图参与也让轨迹成本翻倍,
    尽管 1550 次 fallback 的顶点+q 还原全部逐位审计通过)。
- **钉死参数勿动**(附录 D 定稿):headroom=1、容量档 2 的幂阶梯、PCG K=8、
  CFL 下限/swept 触发 = 上游原值。
- **测量方法论**:全轨迹 Newton 总数为主判据(±1%);60 帧窗墙钟噪声 ±10pp 不可裁
  <10pp 效应;凡裁决必带 A/A 对照;混沌场景必须交错多轮取中位。

---

## 8. 诊断工具箱

以下旋钮均为诊断用途;凡标注"资格拒绝"者会把整帧图关掉(§5.2),
故**诊断态与生产态的执行路径不同**,定位问题时先明确自己在哪条路径上。

| 旋钮 | 适用线 | 用法与输出 |
|---|---|---|
| `STIFF_FRAME_GRAPH_DIAG=1` | 【仅 phase-cd】 | 帧图诊断打印族:资格拒绝原因(tiny/eligible reason)、`[restore] max\|live-snap\|`(干跑污染检查)、`[rollback-audit]`(attempt>0 时对帧起始状态逐位比对——起始状态在重试循环**外**抓取,对 context.fem_vertexes 比对是自证)、fallback/重录事件。第一优先诊断旋钮:图路径任何异常先开它 |
| `STIFF_GRAPH_PHASE_TIME=1` | 【仅 phase-cd】 | 设备 globaltimer 打点分相计时:帧级 `[graph-phases] bvh/dcd0/gh/pcg/ccd_ls/postls` + PCG 子相(spmv/mix1/precond/mix2,buf[11..14])。**§2.7 的 27% 成本结构即由它产出**;宿主通道也可用(pcg_solver.cu:799-814 自备 buf)。注意:它是资格拒绝项——开着它测的是两图/宿主路径 |
| `STIFF_PCG_EXIT_DIAG=1` | 【仅 phase-cd】 | 每 solve 一次阻塞读:iters/brk(1=收敛,2=SPD 破坏)/rz/rz0/rel。诊断 PCG 不收敛/提前断 |
| `STIFF_LSX_DIAG=1` | 【仅 phase-cd】 | **line-search 耗尽取证**:E0 时刻把能量项槽(15 槽:FEM_kinetic/FEM_elastic/membrane/bending/soft/ground/barrier/friction/gfriction/ABD_*)+ pair 计数 + drive rate 快照;耗尽时最后一次 trial 跑在数值零 α 上,其状态应等于 E0 状态——**逐槽打印差异,直接点名回滚泄漏的缓冲**(`[lsx-diag] slot <name> E0=... trial=... ratio=...`;core/ipc_solver.inl:196-214, 742-760)。§5.3 的 ABD q 族漏抄即由它定位 |
| `STIFF_FRAME_FORCE_ROLLBACK=2` | 【仅 phase-cd】 | **成对审计(paired audit)**:整帧图解完 → 边界注入 RETRY → restore 核翻转生效 → 宿主 release 求解器把同帧再解一遍;轨迹端到端宿主形状,`[graph-frame]` 诊断行保留每次图 attempt 的真实设备计 Newton/PCG ⇒ **每帧一个同状态图/宿主成对样本**(§7 的 +1% 证据即此法产出)。`=1` 是 G16 故障注入(两图 terminal 把 OK 改判 RETRY,注入帧最终 throw) |
| `STIFF_GRAPH_DIRPROBE=1` / `STIFF_GRAPH_TIERPROBE=1` | 【仅 phase-cd】 | 干跑期探针:同一状态下 exact-count 装配 vs capacity-mirror 装配的搜索方向对照 / tier 不变性四变体对照(cp4 与 cp4×8、有/无预清零)。验证"容量镜像不改数值" |
| `STIFF_PCG_WIDTH_DIAG` / `STIFF_GRAPH_RESIZE_DIAG` | 【仅 phase-cd】 | PCG 容量网格宽度 / 设备 resizer 槽活动打印 |
| `STIFF_XENV=1`(+`STIFF_XENV_ID`) | 【稳定线+phase-cd】 | 跨 env 镜像诊断:把 per-vertex buffer 按 env0/env1 的 local-id 摆到可比布局,报告 max\|env0−env1\| 与最坏 lid;pair 分类 intra/CROSS-env。证明镜像(call#0==0)与追杀跨 env 泄漏的工具 |
| `STIFF_MAS_DUMP=<dir>` | 【稳定线+phase-cd】 | 首次 MAS 装配 dump 原始三元组+分区前缀,供 `examples/test_mas_oracle.py` CPU oracle 全链重建(层级/聚合/求逆独立复算)。资格拒绝项 |
| `STIFF_MAS_FUSE_VALIDATE=1` | 【稳定线+phase-cd】 | 融合 Schwarz 路径与 legacy 路径一次性 bit 对比,不一致 throw |
| `STIFF_MIRROR_AUDIT=1 STIFF_SLOT_AUDIT=1` | 【仅 phase-cd】 | HostMirror 审计 / 装配槽 sentinel 审计(保留未写槽 throw);已证 bit-transparent,门禁全程武装。资格拒绝项 |
| `STIFF_SEG_DIAG` / `STIFF_SWEPT_DIAG` / `STIFF_PENV_STATS` / `STIFF_S1_DEBUG` / `STIFF_ALPHA_DBG` | 【稳定线+phase-cd】 | 分段 PCG / swept-CCD / per-env 遥测统计打印。注意 per-env 诊断旋钮会把 S1 路由到 host 遥测路径(改变执行路径但不改数值公式) |
| `STIFF_KNOB_STRICT=1` | 【仅 phase-cd】 | 环境中未注册的 STIFF_* 旋钮直接抛 `ConfigurationError`(默认仅 stderr 警告;"typo = silent no-op" 防线)。正式实验建议常开。旋钮单源 `StiffGIPC/config/knob_registry.h`,G14 静态核查 |
| `[mode-config]` 行(非旋钮) | 【仅 phase-cd】 | `[mode-config] resolved=<mode> iso[...] strict[...]` 由 `g_gipc_log_level` 门控,默认值 1 ⇒ **默认即打印**,唯一改变途径是 `SimEngine::set_log_level()` API(00_impl_api_surface.inl:325-327)——判定当前进程真实模式;半配置打 WARNING。**`STIFF_LOG_LEVEL` 是死旋钮,勿设**:两线引擎代码均无人读取(仅 scripts/examples 残留 setdefault),且不在 knob_registry.h,`STIFF_KNOB_STRICT=1` 下设置它直接抛 `ConfigurationError` |

**典型排查剧本**:

1. *图通道结果与宿主不同/怀疑图路径有 bug*:`STIFF_SPMV_DET=1` 下重跑——det 层上
   图-对-宿主应**逐位相等**(§4.4);仍分歧 = 真 bug,分歧消失 = 容量网格重结合(合法)。
2. *line-search 每帧耗尽*:`STIFF_LSX_DIAG=1` 看哪个能量槽在零 α trial 上与 E0 不等——
   点名的槽就是状态泄漏源(回滚漏抄/soft target 漂移/drive rate 未还原)。
3. *图通道慢于预期*:`STIFF_GRAPH_PHASE_TIME=1` 分相 + `STIFF_FRAME_GRAPH_DIAG=1` 看
   fallback/重录频率;重录风暴通常是容量档反复增长(看 OVF 位与轴)。
4. *怀疑图对轨迹的统计影响*:`STIFF_FRAME_FORCE_ROLLBACK=2` 成对审计,比较
   `[graph-frame]` 行的图 Newton 与宿主实际 Newton。

---

## 附录 A. 本分册环境变量速查

(全量旋钮注册表见 `StiffGIPC/config/knob_registry.h`【仅 phase-cd】;
值语义一律**值感知**:空/未设/"0" = 关。)

**线性求解 / PCG**【除注明外 稳定线+phase-cd】

| 变量 | 默认 | 作用 |
|---|---|---|
| STIFF_PCG_TOL | config(1e-4) | 覆盖收敛率 |
| STIFF_PCG_CHECK_K | 8 | 收敛检查周期 K(分段要求正偶数) |
| STIFF_PCG_GRAPH | 1 | PCG 图 capture 总开关 |
| STIFF_PCG_DEVICE_LOOP | 1 | 自尾 device graph |
| STIFF_PCG_GRAPH_CACHE / STIFF_PCG_CACHE_VERIFY | 0 / 关 | 跨 solve 图缓存 / 持续验证【仅 phase-cd,实验性】 |
| STIFF_SEGMENTED_PCG | 模式束 | 分段块对角 PCG(isolated/strict bundle 成员) |
| STIFF_SEG_BINNED / STIFF_SEG_WARP | strict 联动 / strict=0 否则 1 | binned per-env dot / warp 预求和(=1 强开破确定性) |
| STIFF_PCG_WARM / STIFF_PCG_EW(±GAMMA/ETAMAX) | 0 / 0 | warm start / Eisenstat-Walker【实验性,默认关】 |
| STIFF_SPMV_DET | strict 束成员 | det SpMV + g_det_reduce 总闸 + binned seg-dot + 关标量 PCG 图 |
| STIFF_MAS_SEG | auto(多 env 开) | per-env MAS 分段 0/1/N |
| STIFF_MAS_FUSE / STIFF_MAS_FUSE_VALIDATE | 关 | 融合 Schwarz / bit 校验【实验性】 |
| STIFF_MAS_APPLY_RESIZE | 0 | 图内 Schwarz 网格设备 resizer【仅 phase-cd,实验性】 |
| STIFF_SKIP_ZERO_DEPOSIT | 1 | 归并跳零 deposit(=0 恢复原子长龙)【仅 phase-cd】 |
| STIFF_CONVERT_DEVICE_COUNT / STIFF_CONVERT_EXACT_WIDTH / STIFF_CONVERT_HOST_BOUND | 关 / 0 / 0 | 设备计数布局 / convert 宽度收窄 / host-bound 免 D2H【仅 phase-cd】 |
| STIFF_TIER_STEPS / STIFF_ASSEMBLY_TIER_SHIFT | 1 / 0 | 容量档细分(判 REJECT)/ 手动提档【仅 phase-cd】 |

**帧图 / episode / RL**【全部 仅 phase-cd】

| 变量 | 默认 | 作用 |
|---|---|---|
| STIFF_FRAME_GRAPH | 关 | 两图事务(step() 分流) |
| STIFF_FRAME_FULL_GRAPH | 关 | 整帧图叠加层 |
| STIFF_C4_COLLISION_GRAPH / STIFF_C5_ISOLATED_GRAPH / STIFF_C6_ABD_STEP_GRAPH | 关 | 碰撞进图(step 路径;episode 免)/ isolated 进图 / 普通 step() 的 ABD 进图 |
| STIFF_FRAME_MAX_RETRIES | 3(clamp 0..16) | OVF 重试预算 |
| STIFF_FULL_GRAPH_MIN_VERTS | 1024(0=关) | tiny 场景保护 |
| STIFF_GRAPH_TIER_HEADROOM | 1 | 训练档 headroom [1,64] |
| STIFF_GRAPH_DEVICE_RESIZE | 开 | 设备 grid resize(16 槽环) |
| STIFF_GRAPH_INGRAPH_RETRY | 关 | 恢复图内重试溢出帧【实验性】 |
| STIFF_GUARD_FRAME_END_ONLY / STIFF_FALLBACK_NO_REBUILD / STIFF_C6_RERECORD | 关 | 诊断:tier guard 仅帧尾 / fallback 不重建对集 / ABD 逐帧强制重录 |
| STIFF_DRIVE_SUBSTEP | — | >1 时 episode prepare 拒绝(runtime_error) |
| STIFF_LS_GRAPH | 关 | C-1 LS 回溯自尾图【实验性】 |

**诊断**:见 §8 表。

---

## 相关分册与文档

- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — 已知问题与修复史(tactile 修复的线间差异、
  整帧图回滚泄漏 4e49fda、towel-strict 根因等)
- [../OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md) — 成本结构与已裁决优化(死路表)
- [../SIMULATOR_EXECUTION_DESIGN.md](../SIMULATOR_EXECUTION_DESIGN.md) — 执行架构定稿
  (一个表面、两个内核;五层动态性 L1–L5;附录 A/D 实测)
- [../GPU_NATIVE_RL_PLAN.md](../GPU_NATIVE_RL_PLAN.md) — GPU-native RL 契约与 nsys 证据
- [../PHASE_C_FRAME_GRAPH_PLAN.md](../PHASE_C_FRAME_GRAPH_PLAN.md) — Phase C/D 蓝图与战报
- [../GRAPH_DEFAULT_ON_EVIDENCE.md](../GRAPH_DEFAULT_ON_EVIDENCE.md) — 整帧图不默认开的裁决
  (倍率表已过时,结论方向存活)
- [../CI.md](../CI.md) — 门禁体系(G1/G9/G11/G14/G16/G17/G18/G19)
- [../PHYSICS_VALIDATION.md](../PHYSICS_VALIDATION.md) — 物理验证与 checkpoint v2

---

*本分册基于 phase-cd @ b3ab747 与稳定线 v0.8.5.3 @ b8e27a1 的代码与仓内文档写成;
所有"待核实"标注表示提取材料与代码均未闭环确认的点,禁止在下游引用时当作事实。*
