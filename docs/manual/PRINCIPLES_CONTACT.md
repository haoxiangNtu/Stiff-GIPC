# StiffGIPC 手册 · 原理分册:接触 / CCD / 摩擦

> **PRINCIPLES_CONTACT** — 障碍接触模型、碰撞检测管线、连续碰撞检测(CCD)、光滑化库仑摩擦、地面接触与接触力读出的完整原理与实现出处。
>
> 相邻分册:求解器主循环与 line search 见 [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) §2–4;能量项总表与组合公式见同册 §1(组合公式)与 §5(逐槽详解)。

## 适用版本与行号约定

| 线 | 仓库 | 分支 / HEAD | 说明 |
|---|---|---|---|
| **稳定线** | `/home/ps/Downloads/Stiff-GIPC-stable-08` | `release/stable-0.8`,tag **v0.8.5.3**(`b8e27a1`,2026-08-11 发布) | 重构前单体布局(`StiffGIPC/GIPC.cu` 1.6 万+ 行);公开仓 `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel(CUDA `sm_80/89/120`) |
| **工程线(phase-cd)** | `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph` | `codex/phase-cd`,HEAD `b3ab747` | v0.8.6 模块化重构 + 整帧 CUDA Graph + GPU 驻留 RL 等全部 v0.8.5 后工作 |

- 未注明的 `文件:行号` 均指 phase-cd 树 `StiffGIPC/` 下的路径;稳定线行号写作 `stable:GIPC.cu:NNNN`(指向该树 `StiffGIPC/GIPC.cu` 的 v0.8.5.3 内容)。
- 适用线标注:**【稳定线+phase-cd】**(两线同在)、**【仅 phase-cd】**、**【仅稳定线】**、**【实验性,默认关】**。
- 关键差异预告(§7 详述):**接触力读数的摩擦分量修复(pre-commit 快照)与 `reset_transient_contact_state` API 只进了稳定线 v0.8.5.3;phase-cd 尚未移植**——phase-cd 上 `get_vertex_contact_forces` 的 `friction_lagged`/`total` 分量仍受"恒零" bug 影响。

## 目录

1. [IPC 障碍接触](#1-ipc-障碍接触)
   - 1.1 记号与平方距离约定 · 1.2 log-barrier 形式(RANK=2) · 1.3 dHat/dTol/fDhat 派生与 absolute_dhat 换算 · 1.4 kappa 生命周期与 per-group kappa
2. [碰撞检测管线](#2-碰撞检测管线)
   - 2.1 帧内时序 · 2.2 LBVH 广相 · 2.3 DCD 窄相与配对编码 · 2.4 配对缓冲与容量 · 2.5 确定性槽序 · 2.6 EE mollifier 现状
3. [CCD 与无相交保证](#3-ccd-与无相交保证)
   - 3.1 ACCD 算法 · 3.2 每次 Newton 迭代的 alpha 链 · 3.3 slack/CFL 旋钮 · 3.4 line search 中的 CCD 过滤 · 3.5 构造性保证的准确表述 · 3.6 图内形态
4. [摩擦](#4-摩擦)
   - 4.1 滞后摩擦模型 · 4.2 λ 的数学 · 4.3 光滑化 f0/f1/f2 与 fricDHat/eps · 4.4 三处分支阈值的一致性 · 4.5 buildFrictionSets · 4.6 per-body μ · 4.7 真静摩擦(v0.8.5.4)
5. [摩擦保真实测](#5-摩擦保真实测)
6. [地面接触](#6-地面接触)
   - 6.1 平面模型与配置 · 6.2 检测与 d-floor fail-fast · 6.3 能量/梯度/Hessian · 6.4 检疫(quarantine)协议
7. [接触力读出原理](#7-接触力读出原理)
   - 7.1 单位与符号 · 7.2 摩擦读数为何需要 pre-commit 快照 · 7.3 API:get_vertex_contact_forces · 7.4 API:reset_transient_contact_state · 7.5 其它接触导出 API
8. [参数与环境变量速查](#8-参数与环境变量速查)
9. [待核实清单](#9-待核实清单)

---

# 1. IPC 障碍接触

## 1.1 记号与平方距离约定

StiffGIPC 全库统一使用**平方距离**:

- `d` = 一个接触对(点-点/点-边/点-三角形/边-边)的**平方**最近距离(`_d_PP/_d_PE/_d_PT/_d_EE` 均返回平方值,`mlbvh_modules/02_distances_dtypes.inl:1-29`);
- `dHat`(成员 `GIPC::dHat`)= 接触激活阈值的**平方**(§1.3);
- 势垒仅对 `0 < d < dHat` 的对生效。`d ≤ 0` 的**类型化 fail-fast 仅存在于地面路径**(线性距离,§6.2;塌陷旗 `_gdCollapse` 只由地面检测与帧首探针写入,`06_kinetic_soft_ground.inl:36`、`multienv/isolation.cu:74`);自接触四类距离(`_d_PP/_d_PE/_d_PT/_d_EE`)按构造返回平方值恒 ≥ 0,**不存在**"检测到 d≤0 即抛类型化错误"的对应机制——body-body 的几何不可行表现为网格穿插(平方距离仍为正),其防线是 CCD α + 势垒发散(§3.5;整网格相交复查 `isIntersected` 默认关闭,§3.4)。

**【稳定线+phase-cd】** 该约定两线一致。

## 1.2 log-barrier 形式(RANK = 2)

势垒幂次的单一真源是 `contact/barrier_rank.h:7` 的 `#define RANK 2`(**编译期常量,两线相同**)。RANK=2 时,一个非退化接触对的势垒能量为(`energy/02_contact_energy_device.inl:31-37`,即 GIPC 论文的 rank-2 对数势垒):

```
I5   = d / d̂                    (d、d̂ 均为平方距离)
E(d) = κ · (d − d̂)² · log²(I5)
```

性质:`d → d̂⁻` 时 E、∂E/∂d 光滑趋 0(C¹ 接入);`d → 0⁺` 时 `log²(d/d̂) → ∞`,能量发散——这是"势垒阻止穿透"的数学来源。κ(Kappa)在 device 函数内乘入(所以能量组合时该槽裸加,见 [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) §1.2)。

- RANK 1..6 的完整家族形式(交替正负号:`−κ(d−d̂)²log(I5)`、`κ(d−d̂)²log²(I5)`、…)全部保留在源码里(`02_contact_energy_device.inl:34-47`),但**只有 RANK==2 是活代码**,其余为编译死支。
- **mollified(平行边)变体**(`02:49-91` "pee"、`:98-142` "pp"、`:169-216` "ppe"):
  `E = κ · (−I1²/eps_x² + 2·I1/eps_x) · (d̂ − d̂·I2)² · log²(I2)`,其中 `I1 = ‖(v1−v0)×(v3−v2)‖²`、`eps_x = 10⁻³·‖v0−v1‖²_rest·‖v2−v3‖²_rest`(`_compute_epx`,`mlbvh_modules/02_distances_dtypes.inl:45-49`)。这些分支**在能量端仍在但为死代码**——发射端从不产生 mollified 编码,见 §2.6。
- 势垒能量为负时打印诊断 `"I am pee/pp/ppe"`(`02:88-89` 等)。

**【稳定线+phase-cd】** RANK=2、mollified 死支冻结("frozen smooth branches travel VERBATIM, still dead, still frozen",`02:6-7` 头注释)两线一致。

## 1.3 dHat 派生:relative_dhat、absolute_dhat 换算、dTol、fDhat

派生全部发生在 `GIPC::init`(`gipc_modules/09_friction_sets_host_mem.inl:580-622`;稳定线同款逻辑 `stable:GIPC.cu:9448-9463`)。

**第一步:场景包围盒。** 非 `m_skip_all_collision` 时从设备直读 BVH 根盒 48 字节(`09:594-595`,[B3 bbox-async fix]:旧代码消费 `bvh_f.scene`——阻塞构建的副作用,异步构建后不再刷新,曾导致 `bboxDiagSize2=1e65` → 2200 万接触对 → OOM,`09:588-593` 注释)。`bboxDiagSize2 = ‖upper − lower‖²`(`09:597-598`)。

**第二步:absolute_dhat 有效包围盒。** 场景 bbox 对角线随环境数量/间距增长,会放大 bbox 派生的 dHat——这是多环境接触对超线性增长的物理性根因。当 `absolute_dhat > 0 && relative_dhat > 0` 时构造**有效 bbox**(`09:599-607`):

```
eff_bboxDiagSize2 = absolute_dhat² / relative_dhat²
```

代入下面的派生公式后恰好使 `dHat == absolute_dhat²`,且 dTol/fDhat 与"该接触尺度的单环境场景"一致。**【稳定线+phase-cd】**(稳定线在 `stable:GIPC.cu:9448-9463`)。

**第三步:派生量(数值照抄代码)。**

| 派生量 | 公式 | 出处 | 量纲 | 作用 |
|---|---|---|---|---|
| `dTol` | `1e-18 · eff_bboxDiagSize2` | `09:608` | 平方距离 | close 集"过近"阈值(`_computeGroundCloseVal`/`_calSelfCloseVal` 的 dTol 参数;close 集机制见 [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) §2.3–2.4) |
| `minKappaCoef` | `1e11`(常量) | `09:609` | — | κ 建议值系数(§1.4) |
| `dHat` | `relative_dhat² · eff_bboxDiagSize2` | `09:612` | **平方**距离 | 势垒激活阈值;absolute_dhat 设置时 = `absolute_dhat²` |
| `fDhat` | `1e-4 · eff_bboxDiagSize2` | `09:613` | **平方速度 (m/s)²**(= IPC 的 epsv²;数值上从 m² 的 bbox 量派生,系数 1e-4 吸收 1/s²) | 摩擦静止阈值基量(§4.3):调用侧 `fricDHat = fDhat·dt²` [m²] 与 ‖u‖² 直接比较、`eps = √fDhat·dt` [m](`16:830`)——把 fDhat 读作 m² 会使 eps 变 m·s,量纲链只在 (m/s)² 读法下自洽 |

**配置入口:**

| 参数 | 默认值 | 单位 | 含义 | 出处 |
|---|---|---|---|---|
| `Config.relative_dhat` | `1e-3` | 无量纲(相对场景对角线) | `dHat_sqrt = relative_dhat × eff_diag`;场景侧统一使用 1e-3 | `sim_engine.h:40`;GIPC 成员默认 0,由 `engine_modules/01_config_upload.inl:20-21` 拷入 |
| `Config.absolute_dhat` | `0.0`(= 用场景 bbox 派生) | m | >0 时钉死 `dHat_sqrt = absolute_dhat`,与环境数/间距无关;多环境建议设为单环境的 dHat_sqrt | `sim_engine.h:66` 及注释;Python 侧 `engine.py:352,408-411` |

日志:全局 `g_gipc_log_level >= 1` 时打印 `[dhat] ... dHat_sqrt=...(ABSOLUTE|scene-bbox)`(`09:614-617`)。该门是全局 int(默认 1,即**默认就打印**,`00_prelude_common.inl:151`),只能经 `SimEngine::set_log_level` / Python `engine.set_log_level` 设置(`engine_modules/00_impl_api_surface.inl:327`)——**不存在读 `STIFF_LOG_LEVEL` 环境变量的代码路径**(个别脚本/示例设置该名字,但两树库代码均无消费者,是无效设置)。`STIFF_SEED_DIAG` 打印 17 位精度全量种子(`09:618-622`)。参考尺度:beaker 场景 dHat_sqrt ≈ 2.39 mm、foldshirt ≈ 1.9 mm(relative 1e-3 下的场景实测值)。

## 1.4 kappa 生命周期与 per-group kappa

κ(接触刚度,成员 `Kappa`,初值 0.0,`GIPC.cuh:580`)是**逐帧重初始化的量,不在帧内自适应**。帧首生命周期(`core/ipc_solver.inl:2559-2564`;稳定线同构:`suggestKappa`@`stable:GIPC.cu:12651`、`upperBoundKappa`@12682、`initKappa`@12704):

1. `upperBoundKappa(Kappa)` — 钳到上限;
2. `Kappa < 1e-16` 时 `suggestKappa(Kappa)` — 首帧/未初始化时给建议值;
3. `initKappa(TetMesh)` — 梯度投影初始化。

**数学形式**(`gipc_modules/13_kappa_partition_gradhess.inl:1-255`):

- `suggestKappa`:`bb = bboxDiagSize2`,但 `absolute_dhat>0` 时 `bb = (absolute_dhat/relative_dhat)²`(abs-kappa 一致性:κ 尺度跟随物理接触尺度;merged bbox 派生会随环境数稀释势垒——ablation 实测 258 vs 489 Newton 全因于此;诊断逃生旗 `STIFF_DIAG_KAPPA_MERGEDBB`)。公式:
  ```
  compute_H_b(1e-16·bb, dHat, H_b)
  κ_suggest = minKappaCoef · meanMass / (4e-16 · bb · H_b)     (meanMass==0 时去掉质量因子)
  ```
- `upperBoundKappa`:`κ_max = 100 × κ_suggest` 同式(`13:32-51`),超上限钳回。
- `initKappa`(每帧,`13:54-255`):装配非接触梯度 `_GE`(kinetic+FEM elastic+soft)与**单位刚度**接触梯度 `_gc`(`computeGroundGradient(_gc, 1.0, false)` + `calBarrierGradient(..., 1.0, ...)`——必须用单位接触梯度,复用上一帧 per-group 刚度会让估计依赖历史且首帧读未初始化显存,`13:110-114` 注释);取
  ```
  minKappa = −⟨gc, GE⟩ / ⟨gc, gc⟩
  ```
  若 >0 则采纳,再以 `suggestKappa` 为下限、`upperBoundKappa` 为上限(`13:160-170`)。

**帧内 κ 翻倍是两线刻意退役的死路径**:宿主 `postLineSearch` 的 `checkCloseGroundVal/checkSelfCloseVal` 门在 `h_close_gpNum/h_close_cpNum`,而这两个 host 镜像**全树无写点**(仅声明+初始化 0:`GIPC.cuh:577-578`;稳定线 `stable:GIPC.cuh:233-234` 同)→ 恒 false → `Kappa *= 2.0` 永不触发。上游注释:"legacy close-contact Kappa-doubling ... intentionally disabled: its raw restore destabilizes coupled ABD contact"(`09:918-920`)、"do not revive ... without a separately validated adaptive-contact redesign"(`12_host_wrappers_fem.inl:344-345`)。

**per-group(per-env)κ【稳定线+phase-cd,isolated 组件,默认关】**:全局 κ 是跨所有 env 的归约 → batch 依赖;开 `STIFF_PERGROUP_KAPPA`(或 phase-cd `m_mode_config.pergroup_kappa`)后每 env 独立 κ(`m_kappa_group`),在 `initKappa` 内 per-env 梯度投影(`13:171-238`),摩擦 λ 同样按 per-group κ 取值(§4.2),使 env0 的接触法向力 batch 不变。

**图内 κ 条件路径【仅 phase-cd】**:整帧图的 `enqueue_post_ls_kappa_conditional`(`gipc_modules/10_ccd_buildcp_quarantine.inl:3098-3206`)**默认 host-equivalent(不倍增)**——图形态读的是设备计数器(重建确实会填充),曾忠实"复活"退役路径并偏离宿主:foldshirt 抓合帧 κ 连倍 6 次(41.73→2670.70),line search 塌到 α=2.4e-31、127 次试探杀帧;宿主 κ 钉在 41.73 用 31 次 Newton 过帧。倍增 kernel 只留在 `STIFF_GRAPH_LEGACY_KAPPA_DOUBLE` 旋钮后(`10:3168-3170`)**【实验性,默认关】**。图内 κ 设备标量经 `graph_kappa_dev()`(`10:3070-3076`)注入各接触 kernel 尾参。

编译期开关:`ADAPTIVE_KAPPA` 恒定义(`CMakeLists.txt:93`,两线同)。

---

# 2. 碰撞检测管线

## 2.1 帧内时序

每次 Newton 迭代接受一步之后(line search 的每个 trial 也一样),碰撞状态按以下顺序重建:

```
buildBVH()          广相:face 树 + edge 树 LBVH 构建
buildCP()           窄相:DCD 自碰撞检测(双流并行)+ 地面检测 + 溢出重跑 + 快照
```

CCD 阶段(每次 Newton 迭代取步长前)另行构建 swept 结构:`buildBVH_FULLCCD(α)` + `buildFullCP(α)`(§3.2)。摩擦滞后集只在帧首/子步首重建(§4.5),Newton 迭代内不变。

## 2.2 LBVH 广相

`buildBVH`(`gipc_modules/11_perenv_machinery.inl:640-652`)**【稳定线+phase-cd】**:

- 两棵树:`bvh_f`(surface face 树)与 `bvh_e`(edge 树),LBVH(30-bit Morton;实现在 `mlbvh.cu`/`mlbvh_modules/`)。每次调用都是**全建**(排序+建叶+传播);"通常只 refit"对该实现为假(`docs/BVH_ACCELERATION_VALIDATION.md:39-61`)。精确拓扑 refit(`STIFF_BVH_REFIT_INTERVAL`)、PLOC++、查询排序等均为**【仅 phase-cd,实验性,默认关】**旋钮(裁决见 `docs/BVH_ADVANCED_VALIDATION.md`,注意架构×模式矩阵无普适默认)。
- 构建前先 `_addEnvOffset`:`d_bvh_vertexes = _vertexes + d_env_offset`(`11:630-638,648-649`)——多环境确定性:BVH 建在**偏移分离**的顶点上,窄相仍用局部 `_vertexes`;offset=0 时为 no-op。
- per-env 模式(`m_perenv_bvh && groups>0 && !m_graph_merged_detect`)下树在 buildCP 的 per-env 循环里建;整帧图强制 merged 树(per-env 是宿主循环+BVH 对象变异,不可捕获;隔离性由发射处 `set_self_p2g` 过滤保持,对集相同,`10:1974-1979`)**【仅 phase-cd】**。

## 2.3 DCD 窄相:buildCP 与配对编码

`buildCP`(`gipc_modules/10_ccd_buildcp_quarantine.inl:1897-2127`)主流程(merged 路径):

1. host 镜像失效(`h_cpNum/h_gpNum/h_ccd_cpNum.invalidate()`,`10:1899-1904`)——DCD 重发射改写 `_cpNum/_gpNum` 并**覆写 `_ccd_collisonPairs` 前缀**(文档化危害,见 §3.2 快照);
2. 发射门设置(`10:1921-1972`):`set_ee_nodedup/detgate/canon/nomollify`、`set_bvh_envpart`、`set_self_p2g`(per-vertex 跨 env 跳过)等;
3. 计数清零 → `bvh_f.SelfCollitionDetect(dHat)`(默认流)与 `bvh_e.SelfCollitionDetect(dHat, m_aux_stream)`(aux 流)**并行**,双方 `atomicAdd` 同一 `_cpNum`/`_collisonPairs`(`10:2005-2048`)→ `GroundCollisionDetect()`(`10:2051`,§6.2)→ 事件 join;
4. (可选)`canonicalizePairSlots()`(§2.5)**【仅 phase-cd】**;
5. 一次 6-int D2H 刷新计数 → **溢出 grow-redo**:`h_cpNum[0] > MAX_COLLITION_PAIRS_NUM` 时 `newcap = 1.5x+1`、`pair_buffers_grow_dcd`(DCD≤CCD 容量 lockstep)、重设 BVH 指针、**重跑检测**(`10:2083-2114`);
6. `snapshotDcdCcdPairs()`(`10:2121`,§3.2)→ **仅非 defer 模式**同步 `throwIfGroundDistanceInvalid()`(`if(!m_ls_defer_counts)` 门控,`10:2122-2126`,§6.4)。默认 merged 设备 line-search 的 trial-defer 模式下 buildCP 尾部**不抛**:地面塌陷旗改由 line-search 决策读回捎带,在消费端 `handleGroundCollapse` 处理(同一 trial、幂等,`core/ipc_solver.inl:481-485`)——防线语义不变(§3.4 第 3 条),但抛错/检疫时序在消费端而非 buildCP 内。

**配对类型与 int4 编码**(发射端 `mlbvh_modules/03_pair_emission.inl` ↔ 能量端 `energy/02_contact_energy_device.inl:12-268` 的二进制约定)**【稳定线+phase-cd】**:

| 类型 | 编码 `(x,y,z,w)` | 判别式(能量端) | 出处 |
|---|---|---|---|
| EE(边-边,4 顶点) | `(id0, id1, id2, id3)` 全非负 | `x≥0 && w≥0` | 发射 03:513;解码 02:21-48 |
| PT(点-三角形) | `(−idP−1, t0, t1, t2)` | `x<0`,z、w ≥0 | 02:241-266 |
| PE(点-边) | `(−idP−1, idE0, idE1, add_e)` | `x<0 && w<0 && y≥0` | 发射 03:276/283(dtype2/3)、390/397(dtype4/5);判别 02:167,解码 02:217-239 |
| PP(点-点) | `(−id0−1, id2, −1, add_e)` | `x<0 && z<0 && y≥0` | 03:199/206;02:143-165 |
| EE-M / PP-M / PE-M(mollified 变体) | `w<0` / 全负 / `y<0` | **能量端在但死代码**(§2.6) | 02:49-91 / 98-142 / 169-216 |

- EE 检测按 `_dType_EE` 分 9 个退化 dtype 发射;PP/PE 是 EE/PT 的退化情形。
- `_MatIndex` 记录每对在 rank 槽内的序号,供 Hessian 装配 tier 布局。

## 2.4 配对缓冲、计数块与容量策略

**【稳定线+phase-cd】**(行号为 phase-cd):

- `_collisonPairs`(int4,DCD 对)、`_ccd_collisonPairs`(int4,swept CCD 对,4 顶点原始 id 未编码)、`_MatIndex`。分配于 `pair_buffers_alloc`(`09:398`),**+1 trash slot**:发射溢出被重定向到 `index==cap` 的垃圾槽,检测 kernel 永不越界写,宿主随后按 1.5× 增长重跑(`09:394-397`)。
- `_cpNum` 是连续 6×uint32 块:`[0]`=DCD 总对数(CCD 发射计数复用同一槽)、`[2]`=PP、`[3]`=PE、`[4]`=PT/EE(4 顶点 rank)、`[5]`=`_gpNum`(地面对数,别名 `_gpNum = _cpNum + 5`);一次 6-int D2H 同取 cp+gp(`09:402-407`,`10:2073-2079`)。`[1]` 未见独立消费(见 §9 待核实)。
- 宿主镜像 `h_cpNum/h_gpNum/h_ccd_cpNum` 带 invalidate/refresh 纪律(phase-cd 有 `STIFF_MIRROR_AUDIT` 审计)。
- 初始容量策略(`GIPC::init`,`09:629-689`):`Minimum = 100000·buffScale`;`b4/b3/b2 = max(2·(surf_vertexNum+edge_Num), Minimum)`、`b1 = 2·surf_vertexNum`;碰撞 triplet 上限 = `b4·16 + b3·9 + b2·4 + b1`;[P1-dyn] 非 hybrid 场景按实际接触数动态增长而非最坏值预分配(`09:657-675`)**【稳定线+phase-cd】**(稳定线同款注释与 `m_dynamic_triplet` 逻辑:`stable:GIPC.cu:9503-9509`,另 12932-12937、13227-13233 两处帧首 provable-upper-bound grow)。
- `m_pair_snap_cur/m_pair_snap_last`:6×uint32 设备侧计数快照(当前 DCD / 滞后摩擦集时刻),供整帧图/tier 布局在设备上读活跃计数(`09:407-412`)**【仅 phase-cd】**。

## 2.5 确定性槽序 canonicalizePairSlots【仅 phase-cd】

DCD 发射用 atomicAdd 分配槽位 → 相同对集的**槽排列**逐次运行不同;line search 能量归约按槽序求和 → 曾在录制的整帧回放中于首接触帧翻转 towel 的刀锋 line-search 决策而爆炸(newton=1000、ls=20586)。修复(`10:2270-2347`,动机注释 2133-2151):发射后对配对槽(+地面表)做**稳定两趟字典序 radix 排序**(`cub::DeviceRadixSort::SortPairs`,minor 键 `(z,w)` 先、major 键 `(x,y)` 后),全部按 MAX 容量固定形状+设备侧活跃掩码(pad 键 `~0ull` 沉尾),使录制 launch 每帧形状相同、捕获中零分配。键序转换 `_canon_ord32(v) = uint(v) ^ 0x80000000`(有符号→无符号序)。由 `m_mode_config.ee_canon` 门控(strict 模式成套开启)。

## 2.6 EE mollifier 现状:上下游 smooth=false

**【稳定线+phase-cd】结论:mollify 请求被计数但从不执行。** 这是上游(KemengHuang)的决定,**非回归**。

- 发射端每个 EE dtype 都计算 mollify 请求:`eeSqureNCross = ‖(v1−v0)×(v3−v2)‖²` 与 `eps_x = 1e-3·‖v0−v1‖²_rest·‖v2−v3‖²_rest`(`mlbvh_modules/02_distances_dtypes.inl:51-58`),`add_e = g_ee_nomollify ? −1 : ((eeSqureNCross < eps_x) ? −obj_idx−2 : −1)`(`03:185,224,...,491`);
- 但发射函数中 `bool smooth = false;` **硬编码**(`03:171`),所有 `if(smooth)` 的 mollified 编码分支(`03:191-198、501-505` 等)都是死代码:dtype8 的 `add_e<=-2` 分支与 else 分支发射**完全相同**的 `(id0,id1,id2,id3)`(`03:495-515`);PP/PE 退化 dtype 的 add_e 仅存进 `.w`,而能量端 PP/PE 判别只看符号不读 `.w`(`02:96,167`);
- 能量端 mollified 分支同样冻结(`02:6-7` 头注释:"frozen smooth branches travel VERBATIM, still dead, still frozen");
- `STIFF_EE_NOMOLLIFY=1` 只是把请求计数也关掉(`mlbvh_modules/00_gates_globals.inl:97-98`);
- 实测量级:folded 布料场景 44 万次 mollify 请求 / 0 次执行。

**影响**:接近平行的边-边接触使用非 mollified 距离势垒——势垒仍在 `d→0` 发散,**非穿透性质不受影响**;受影响的是该几何退化附近的梯度光滑性(mollifier 本用于消除 EE 距离在平行构型的 C⁰ 缺口),可能表现为平行边密集场景的收敛质量波动。滞后摩擦集对 mollified 编码(`w<0`)的 EE 对直接跳过(§4.5)——由于发射端从不产生该编码,此分支实际也不触发。

> 稳定线的 mlbvh 发射代码与 phase-cd 同谱系(smooth=false 同款),未逐行复核稳定线副本——见 §9 待核实。

---

# 3. CCD 与无相交保证

## 3.1 ACCD 算法(Additive CCD)

`ACCD.cu` 实现 Codimensional IPC(Li et al. 2021)的 Additive CCD:逐段保守推进的迭代式 CCD。核心函数 `point_triangle_ccd`(`ACCD.cu:456-524`)与 `edge_edge_ccd`(`ACCD.cu:373-454`)**【稳定线+phase-cd】**:

```
1. 位移去均值:dp,dt* ← dp,dt* − ¼·Σ(位移)          (456-477;EE 版 388-393)
2. max_disp_mag = ‖dp‖ + max‖dt*‖                    (余下位移模上界)
3. gap = η · (d₀² − ξ²)/(d₀ + ξ)                     (停止判据;491)
4. 迭代:
   toc_lower_bound = (1−η)·(d² − ξ²) / ((d + ξ)·max_disp_mag)     (502-503)
   顶点沿位移推进 toc_lower_bound;重算 d
   若 toc>0 且 (d² − ξ²)/(d + ξ) < gap → 停止
   toc += toc_lower_bound;toc > 1 → 返回 1(整步安全)
   迭代上限 50000                                      (497)
```

其中 `d` 为当前(线性插值轨迹上的)距离、`ξ = thickness`。调用侧传参:`η = CCDDistRatio = 1 − slackness`、`thickness = 0`,位移传 `−moveDir`(Newton 方向取负)(`gipc_modules/07_energy_alpha_reductions.inl:130,156-178`;per-env 版 `11:356,370-395`)。EE 版对 `dFunc ≤ 0` 回退用 4 端点最小距离(`ACCD.cu:404-413`)。

`ACCD.cu` 后半还有基于三次方程/Newton 检查的精确 VF CCD(`IntersectVF/doCCDVF`,`ACCD.cu:711/761` 起),仅供 `isIntersected` 家族的交叉检测使用(默认关,§3.4),不在 alpha 主链上。

## 3.2 每次 Newton 迭代的 alpha 链

宿主路径 `core/ipc_solver.inl:1730-1903`**【稳定线+phase-cd】**(稳定线同构:`stable:GIPC.cu:15850-15935`)。设备标量链缓冲 `m_ccd_alpha_slots`:phase-cd **9×double**(`09:480`),稳定线 **8×double**(`stable:GIPC.cu:9377`,无 slot[8])。槽位含义(`GIPC.cuh:637-639` + `_ccd_final_alpha_combine` 写入):

| slot | 含义 |
|---|---|
| 0 | ground alpha(ACCD 解析式,§6) |
| 1 | narrow-self alpha(对 **DCD 时刻快照** 的 ACCD) |
| 2 | `temp_alpha = min(slot0, slot1)` |
| 3 | max speed(CFL 用) |
| 4 | refined-self alpha(swept 对集的 ACCD) |
| 5 | 最终 alpha |
| 6 | alpha_CFL |
| 7 | 有效 invalid 位快照 |
| 8 | [B3 ccd-defer] 原始 swept 对数**【仅 phase-cd】** |

流程:

1. 清 invalid 掩码 → `ground_largestFeasibleStepSize_DeviceOut(slack_a, …, slots+0)`:每 surf 顶点若 `coef = n·moveDir > 0`,候选 `α = slackness · dist/coef`(`07:60`);`dist≤0`/非有限 → atomicOr invalid 位;skip-body 顶点中立 1.0;级联 min 归约。
2. `self_largestFeasibleStepSize_DeviceOut(slack_m, …, m_dcd_snap_count, slots+1)`:**narrow-phase self CCD 扫 `_dcd_ccd_snapshot`**——DCD 时刻的 CCD 镜像快照,免受后续 `buildFullCP` 覆写(`GIPC.cuh:561-570` 注释;`snapshotDcdCcdPairs()` `10:2349-2368` 把前缀 D2D 进专用缓冲)。该快照机制**稳定线也有**(stable 树 6 处引用)。
3. `_ccd_initial_alpha_combine<<<1,1>>>`:`slots[2] = min(ground, self)`,非 `(0,1]`/非有限记 invalid(`10:1-15`)。
4. **swept 阶段**:`buildBVH_FULLCCD(1.0, slots+2)` + `buildFullCP(1.0, slots+2)`——swept BVH/检测的 α 直接从设备槽 `slots[2]` 读(`alpha_dev` 参数),宿主不回读;buildFullCP = 双流 `SelfCollitionFullDetect` + 溢出 1.5× grow-redo(`11:540-627`)。
5. `cfl_largestSpeed_DeviceOut`(surf 顶点 moveDir 最大速)→ slots+3;`self_full_largestFeasibleStepSize_DeviceOut(slack_m, …, slots+4)` 对 swept 对集做精细 ACCD。
6. `_ccd_final_alpha_combine<<<1,1>>>`(`10:17-126`)核心裁决:
   ```
   cfl_base  = sqrt(dHat) / max_speed        (dHat 为平方距离 ⇒ sqrt(dHat) 是线性激活距离)
   alpha_cfl = cfl_base · cfl_factor
   cfl_floor = cfl_base · 0.5                (floor 系数固定,不随旋钮)
   alpha = min(temp_alpha, alpha_cfl)
   若 temp_alpha > 2·cfl_floor(refined 咨询触发):
       alpha = max( min(temp_alpha, refined·ccd_size), cfl_floor )     (ccd_size = 1.0)
   ```
   稳定线为固定 `0.5` 系数的同一公式(`stable:GIPC.cu` `_ccd_final_alpha_combine`,`__dmul_rn(__ddiv_rn(__dsqrt_rn(d_hat), max_speed), 0.5)`),此时 cap 与 floor 两角色重合。
7. 宿主一次 9-double(稳定线 8-double)D2H 读整链;phase-cd 的 defer 分支在 `slots[8] > MAX_CCD` 时回退 legacy 重算(`1869-1902`)。
8. `validateFinalCcdStateOrThrow`(`10:3376-3396`):invalid 位或 `alpha ∉ (0,1]` → `std::runtime_error`(fail-fast,不带 NaN 继续)。invalid 位定义 `contact/ccd_invalid_bits.h:12-20`(GlobalGround/GlobalNarrow/GlobalRefined/PerEnv* 共 6 位)。

per-env S1 alpha(每 env 独立 ground/narrow α、per-env CCD 搜索用各 env 自己的 `ta_e`)由 `STIFF_PERENV_ALPHA`/`m_mode_config.perenv_alpha` 门控**【稳定线+phase-cd,isolated 组件,默认关】**(`11:343-448`;stable:15893-15921)。四面体反转守卫 `InjectiveStepSize` 在主链中被注释掉(`ipc_solver.inl:1759`;stable:15878)。

## 3.3 slack / CFL 旋钮:三个旋钮与 CFL 的三重角色

| 旋钮 | 作用 | 默认 | 合法域 | 出处 | 适用线 |
|---|---|---|---|---|---|
| `STIFF_CCD_SLACK_A` | ground ACCD slackness(η = 1−slack) | **0.9** | (0,1) | `GIPC.cuh:66-72` | 【仅 phase-cd】(稳定线硬编码 0.9,`stable:GIPC.cu:15850`) |
| `STIFF_CCD_SLACK_M` | self/swept ACCD slackness | **0.8** | (0,1) | `GIPC.cuh:83-84` | 【仅 phase-cd】(稳定线硬编码 0.8,同上) |
| `STIFF_CCD_CFL_FACTOR` | CFL **cap** 系数(floor 恒 0.5) | **0.5** | (0,8] | `GIPC.cuh:95-98` | 【仅 phase-cd】(稳定线固定 0.5) |

CFL 量在裁决中扮演**三重角色**(`10:44-73` [alpha-tune] 注释):

1. **作 cap**(`alpha_cfl = cfl_base·factor`):swept-BVH 膨胀守卫,**不是正确性界**——实测 fs4 重接触场景 22.6% 迭代被 CFL 压低 α(中位压低 1.48×),可安全调高;
2. **作 floor**(`cfl_floor = cfl_base·0.5`,固定):refined 路径的下限。floor 若放大会接受未经 ACCD 认证的位移,有穿薄风险,故系数钉死;
3. **作 refined 咨询触发**(`temp_alpha > 2·cfl_floor`):只有初步 α 足够大时才咨询 swept 精细值(保持 legacy 阈值)。

**调参建议(实测背书,`docs/SIMULATOR_EXECUTION_DESIGN.md` 附录 C/D)**:重接触场景 `STIFF_CCD_SLACK_M=0.9 STIFF_CCD_CFL_FACTOR=1.0` → 全轨迹 Newton −7.9%(5 批配对全负)、墙钟 −3~4%(A800 fs4-4env);轻接触场景零收益或小亏,**勿开;默认全关**。

## 3.4 line search 中的 CCD 过滤与相交回退

CCD 链给出的 α 是 line search 的**初始步长**;line search 内部还有三层与穿透相关的过滤(`core/ipc_solver.inl:181-880`;详见 [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) §4)**【稳定线+phase-cd】**:

1. **ground 域回退**(`:370-383`):首个 trial 后 `groundTrialStatus() != 0`(trial 状态跌出严格势垒域)→ `alpha *= 0.5` 循环,预算 `line_search_max_iter`(默认 64)耗尽 → **抛** `"ground trial step remained outside the strict barrier domain after line-search backtracking"`。
2. **网格相交回退**(`:385-411`):`while(checkInterset && isIntersected(...))` 每次 `alpha/=2; alpha=min(cfl_alpha, alpha)`,耗尽 → 抛 `"mesh intersection persists after line-search backtracking (start state likely already intersecting; ...)"`。**但 `isIntersected` 默认直接 `return false`**——edge-tri 精确复查默认关闭(假阳性曾致 26 万次打印+死循环;`GIPC_FORCE_CCD_SANITY=1` 才启用,`14_energy_linesearch_solver.inl:547-590`)。该回退环默认惰性。
3. **buildCP 的 d-floor fail-fast**(§6.2):每个 trial 的 `buildCP()` 都会在地面距离塌陷(`dist ≤ 0` 或非有限)时记录并触发检疫/异常。

移动边界预处理(帧首,`m_update_boundary` 时,`:2503-2555`)有独立的 FULLCCD + intersect 回退循环,预算同 64,耗尽抛 `"boundary-move intersection persists after backtracking"`。

## 3.5 无相交保证的准确表述

StiffGIPC 的"无相交"是 **IPC 式的构造性保证**,由四重机制叠加,而非某一次显式相交测试:

1. **势垒发散**:任何激活对的能量随 `d→0` 发散(§1.2),能量下降的优化器不会主动走向穿透;
2. **CCD 认证步长**:每次 Newton 迭代的 α 以 ACCD 认证值为主界(§3.1-3.2),线性化轨迹上不发生穿越。slack 的进入方式:self/swept 路径以 `η = 1−slack` 进入 ACCD **内部推进**(`07:130`),不是对认证值的外乘;仅 ground 候选是字面 `slackness·dist/coef`(`07:60`)。slackness(0.8/0.9)保留安全边距吸收 ACCD 的保守推进误差。**已知例外**:refined 咨询分支的 `cfl_floor` 会把小于 floor 的 swept 认证值顶起(`10:65-72`,§3.3 角色 2)——见下方边界条目;
3. **trial 域检查**:line search 的每个 trial 重建碰撞状态,ground 域违规立刻回退(耗尽即抛错,不接受);
4. **d-floor fail-fast**:`d ≤ 0` 一经检测(buildCP/地面 CCD)即走类型化错误或环境检疫(§6.4),**不带非法状态继续**。

**必须如实陈述的边界**:

- 保证的对象是"每个被接受的步内,线性插值轨迹不发生穿越"——它依赖浮点算术下 ACCD 的保守性与 slackness 边距,是**工程性构造保证**,不是精确算术意义上的数学证明;
- **cfl_floor 覆盖窗口**:当 swept 精细 ACCD 认证值小于 `cfl_floor` 时,最终 α 被启发式的 sqrt(dHat) 尺度 floor 顶起、**超过**认证值(`_ccd_final_alpha_combine`,`10:65-72`;代码注释自认:"As a FLOOR it OVERRIDES a tiny certified refined value with a heuristic sqrt(dHat)-scale step ... risks tunneling through thin geometry",`10:49-53`)。swept 通道又是"从 DCD 邻域之外快速逼近"的唯一认证者(`10:61-64`)——即对 swept-only 快速逼近对,存在一个 floor 尺度的**未认证位移窗口**(穿薄风险);floor 系数因此钉死 0.5 不随旋钮(§3.3 角色 2);
- line search **能量预算耗尽的政策是 WARN+接受**(响亮 stderr 警告,`ipc_solver.inl:732-738`;详见 [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) §4.6)。被接受的最终候选仍在 CCD 认证 α 之内、且通过了 ground 域检查,**无相交性质不因此破坏**;受损的是能量单调性(可表现为后续帧接触漂移/势垒距离塌缩/迭代爆增,警告文本原话);
- 显式的整网格相交复查默认关闭(§3.4 第 2 条),防线的实际构成是 CCD α + ground 域检查 + d-floor throw;
- 实测口径:全部演示轨迹逐帧 `min-separation > 0`(对外报告建议采用"penetration-free throughout all N demonstrations"的构造性表述)。

## 3.6 图内 enqueue 形态【仅 phase-cd】

`enqueue_ccd_alpha_conditional`(`10:3210-3267`)是 merged 标量链的整帧图版:**无宿主读、无 grow-redo**——校验与超容量重试全走 `FrameDeviceState`。次序与宿主路径同构;差异:narrow-self 用容量网格掩码版(`self_..._DeviceOut_Masked`,`10:2936-2972`),swept 网格按训练容量 `graph_trained_ccd_extent()`,超出 → 图内裁决 `OVF_CCD_PAIRS` → 边界重试再录更大档。line search 的图形态、starved-search 政策见 [PRINCIPLES_EXECUTION.md](PRINCIPLES_EXECUTION.md) §5。

---

# 4. 摩擦

编译期开关 `USE_FRICTION` 恒定义(`CMakeLists.txt:93`,两线同)。运行时系数:

| 参数 | 默认值 | 单位 | 含义 | 出处 |
|---|---|---|---|---|
| `Config.friction_rate` | **0.4** | 无量纲 | 自接触全局 μ | `sim_engine.h:24`;GIPC 成员默认 0,`01_config_upload.inl:5-6` 拷入 |
| `Config.gd_friction_rate` | **0.4** | 无量纲 | 地面接触全局 μ | `sim_engine.h:25` |

## 4.1 滞后(lagged)摩擦模型总览

IPC 标准半隐式摩擦:在**帧首(及每个驱动子步首)的当前 DCD 接触集**上冻结三样东西,供该(子)步全部 Newton 迭代的摩擦能量/梯度/Hessian 使用:

- **λ**(`lambda_lastH`):势垒法向力幅,`λ = −κ·2√d·∂b/∂d`;
- **T**(`tanBasis`,Matrix3x2d):接触点切平面基;
- **β**(`distCoord`,double2):最近点重心坐标(PP 无、PE 单坐标、PT/EE 双坐标)。

摩擦能量对**当前步内位移**可微:`relDX3D` = 当前位置相对帧首位置(`o_vertexes`)在滞后重心坐标下的加权相对位移,`u = Tᵀ·relDX3D` 为 2 维切向滑移。冻结使摩擦项成为标准势能(可进 Newton),代价是法向力/切向基滞后一步——正是"lagged"标签与 §5.3 暂态、§7.2 读数问题的共同根源。

## 4.2 λ 的数学

自接触(`_calFrictionLastH_DistAndTan`,`09:33-151`,RANK 分支;**RANK==2 生效**):

```
RANK=1: λ = −κ_eff · 2√d · [ −2·t·log(d/d̂) − t²/d ],  t = d − d̂
RANK=2: λ = −κ_eff · 2√d · [ log²(d/d̂)·(2d − 2d̂) + 2·log(d/d̂)·(d − d̂)²/d ]      (09:143-147)
```

即 `λ = −κ·2√d·∂b/∂d`(势垒对平方距离导数的模,恒 ≥0,因 `d < d̂` 时括号内为负)。`κ_eff` 支持 per-group κ(`09:52-56`:取对代表顶点 `gv` 的 `kappa_grp[p2g[gv]]`,带 −1 guard)。

地面(`_calFrictionLastH_gd`,`09:1-31`):

```
g_b = −2·t·log(d²/d̂) − t²/d²,  t = d² − d̂       (d 为线性距离,d² 为平方)
λ_gd = −κ_p · 2·√(d²) · g_b                        (09:24-29)
```

> **注意**:地面 λ 用的是 RANK-1 型公式(单 log),**不随 RANK 宏**;自接触 λ 是 RANK-2(双 log)。这与地面势垒能量本身是 RANK-1 形式一致(§6.3),但"地面/自接触 rank 不一致是否上游有意设计"未见文档说明——见 §9 待核实。

**【稳定线+phase-cd】** λ 公式两线一致(稳定线在 `stable:GIPC.cu:9572-9591` 一带)。

## 4.3 光滑化函数 f0/f1/f2 与 fricDHat/eps

库仑摩擦耗散 `μλ‖u‖` 在 `u=0` 不可导;IPC 用光滑化头替换小滑移区。选择宏 `FrictionUtils.cuh:13` **`#define SFCLAMPING_ORDER 1`**(C1 钳制生效,两线同)。三个函数族(`FrictionUtils.cuh:417-499`,x² 即 `relDXSqNorm = ‖u‖²`,ε 为静止阈值):

| 阶 | f0(能量头) | f1/‖u‖(梯度头) | f2(Hessian 头) | 状态 |
|---|---|---|---|---|
| C0 | `x²/(2ε) + ε/2` | `1/ε` | `1/ε` | 死支(仅地面能量用其闭式,见下) |
| **C1(生效)** | `x²·(−√x²/3 + ε)/ε² + ε/3` | `(−√x² + 2ε)/ε²` | `2(ε − √x²)/ε²` | `FrictionUtils.cuh:434-448` |
| C2 | `x²(¼x² − (√x²−1.5ε)ε)/ε³ + ε/4` | … | … | 死支 |

C1 形式等价于 IPC 论文的 `f0(y) = y²(−y/(3ε) + 1)/ε + ε/3` 的平方距离参数化。

**阈值派生**(调用侧口径,`energy/16_friction.inl:827-856`):

```
fricDHat = fDhat · dt²          (能量分支阈值,‖u‖² 与之比较)
eps      = sqrt(fDhat) · dt     (光滑化 ε;梯度/Hessian 分支阈值为 eps²)
```

由 `fDhat = 1e-4·eff_bboxDiagSize2`(§1.3)可得 `sqrt(fDhat) = 1e-2·eff_diag`,故 **ε = 1e-2·eff_diag·dt = 一步内允许的"静止"切向位移**,对应场景派生的 `epsv ≈ 1e-2·eff_scene_diag [m/s]`。absolute_dhat 设置时随 eff_diag 一并钉死。
**v0.8.5.3 与 phase-cd 两线均只有这一条场景派生路径**;稳定线 v0.8.5.4 引入 `absolute_epsv` 把 epsv 钉成绝对值(并默认开),连同持久摩擦锚一起构成"真静摩擦"两件套——机制、代价与批不变性冲突见 **§4.7**。

**能量分支**(`__cal_Friction_energy`,`02:297-375`):

```
‖u‖² > fricDHat(滑动区): E = λ·‖u‖                (线性库仑)
否则(静止区):            E = λ·f0_SF(‖u‖², ε)     (C1 光滑头)
```

组合时再乘 `frictionRate`(全局 μ;per-body μ 见 §4.6)。

**梯度**(`_calFrictionGradient`,`16:615-802`):`g = μ·λ·T·û`;滑动区 `û = u/‖u‖`,静止区 `û = f1_SF_div_relDXNorm(‖u‖², ε)·u`(光滑过渡),经 `liftRelDXTanToMesh_*` 抬回网格 DOF,`_gfxAdd` 原子/分箱沉积。**Hessian**(`_calFrictionHessian(_gd)`,`16:91-570`)用 f1_div 与 f2_SF 组合并投影到 PSD。

**地面摩擦能量**(`__cal_Friction_gd_energy`,`02:272-294`):切向投影 `VProj = Δx − n(n·Δx)`;`‖VProj‖² > ε²` → `E = λ(‖VProj‖ − ε/2)`,否则 `E = λ‖VProj‖²/(2ε)`——**用的是 C0 闭式**,与自接触的 f0_SF(C1)不同(代码事实;设计意图未查证,见 §9)。地面摩擦梯度/Hessian 与自接触同型(`16:91-` 起,`eps = sqrt(eps2)`,`16:112`)。

## 4.4 三处分支阈值的一致性

能量分支阈值 `fricDHat = fDhat·dt²`;梯度/Hessian 分支阈值 `eps²`(形参名 eps2)。数值上 `eps² = (√fDhat·dt)² = fDhat·dt² = fricDHat`——**二者相等,只是拼写不同**。因此能量/梯度/Hessian 的滑动-静止切换点严格一致,line search 的能量模型与下降方向所描述的势能不冲突。

## 4.5 滞后集构建:buildFrictionSets

`GIPC::buildFrictionSets()`(`09:732-804`;稳定线 `stable:GIPC.cu:9572-9608`)**【稳定线+phase-cd】**。

**调用时机**(`core/ipc_solver.inl:2566-2568` 帧首,initKappa 之后;`:2614-2616` 每个驱动子步循环内)——即**每帧+每个 animation 子步重建一次**,Newton 迭代内不变。前置 `ensure_frictionBuffers()`(grow-only,无每帧 malloc)。

流程:

1. `cudaMemset(_cpNum, 0, 5×uint32)`——`[0..4]` 复用为滞后集计数器(`09:734`);
2. **[multi-env determinism] 先零 distCoord**(`09:745-751` 注释):PP 滞后对写 tanBasis 但**不写 distCoord**(点-点无重心坐标),槽内留垃圾;槽位置又由 atomicAdd 非确定 → 滞后摩擦数据逐次运行非确定 → 摩擦 Hessian → 整个求解。memset 后 PP 的 distCoord 确定性为 0(PP 摩擦 Hessian 不消费它;EE/PE/PT 会覆写)。**这就是"PP-distCoord 置零修复"**——多环境确定性战役查明的"残余非原子源"之一;kernel 内另有 `distCoord[i] = (0,0)` 显式置零双保险(`09:90-91`)。稳定线同款(`stable:GIPC.cu:9585-9591`);
3. `_calFrictionLastH_DistAndTan`(`09:33-151`):对每个 DCD 对解码类型并写入滞后槽:
   - EE(`x≥0 && w≥0`):计数 `[0]`+`[4]`;`_d_EE` → `computeClosestPoint_EE`(双坐标)+ `computeTangentBasis_EE`;**mollified 编码(`w<0`)的 EE 直接跳过不入摩擦集**(外层 if 不匹配;因发射端从不产生该编码,实际不触发);
   - PP(`x<0,z<0,y≥0`):计数 `[0]`+`[2]`;distCoord=(0,0);`computeTangentBasis_PP`;
   - PE(`x<0,w<0,y≥0`):计数 `[0]`+`[3]`;closestPoint 只写 `.x`,`.y=0`;
   - PT(其余):计数 `[0]`+`[4]`;双坐标;
   - 末尾按 §4.2 公式写 λ,原始对存 `_collisonPairs_lastH`;
4. `h_cpNum_last` 5-int D2H 刷新(`09:765`);phase-cd 另做 `m_pair_snap_last` 设备快照——其 `[5]`(地面数)从 `m_pair_snap_cur[5]` 取而**不是**从 `_gpNum`:地面 Hessian 装配会把 `_gpNum` 再自增 rank,若 Newton 提前退出,槽 5 会是 `2×h_gpNum`;DCD 快照才是滞后地面计数的不可变精确源(`09:766-780` 注释)**【仅 phase-cd】**;
5. 地面滞后集 `_calFrictionLastH_gd`(`09:1-31`):按 §4.2 地面公式写 `lambda_gd`,`h_gpNum_last = h_gpNum`(`09:798`)。

装配侧:摩擦 G/H 的 launch 数量 = `h_cpNum_last[0]` / `h_gpNum_last`(滞后集,而非当前 DCD 集,`12_host_wrappers_fem.inl:241-`)。整帧图录制时能量核网格按容量界、滞后计数从 `m_pair_snap_last` 设备读(`16:813-856`)**【仅 phase-cd】**;其余路径保持宿主镜像 launch(逐位一致的发布语义)。

## 4.6 per-body μ:代表顶点规则与几何平均

**【稳定线+phase-cd】**(per-body μ 修复在 v0.8.5.2 之内)。

- 数据:per-vertex μ 表 `d_vert_mu`(自接触)/`d_vert_mu_gd`(地面);初始化默认填 `cfg.friction_rate`(`engine_modules/01_config_upload.inl:214-215`)。
- **代表顶点规则**(`_pair_mu`,`02:377-396`):一个接触对两侧各取一个代表顶点——EE 取 `pair.x` 与 `pair.z`;点类(PP/PE/PT)取 `−pair.x−1`(点侧)与 `pair.y` 解码(对方侧首顶点)。
- **组合律 = 几何平均**:`μ_pair = √(μ_a·μ_b)`(PhysX 风格)。`d_vert_mu == nullptr` 时回退全局标量。
- 实现细节:能量核内乘 `μ_pair/μ_global`,宿主组合(`host_energy += frictionRate·slots[7]`)乘回全局 μ 后恰好落在 μ_pair(`16:35-40` 注释)——保证 per-body 关闭时逐位等于历史行为。
- API:`SimEngine::set_body_friction(int body_offset, double mu, double ground_mu = -1.0)`(`sim_engine.h:278-283`;稳定线 `stable:sim_engine.h:282`)。`ground_mu < 0` = 该 body 地面摩擦沿用全局。

## 4.7 真静摩擦(v0.8.5.4):absolute_epsv + 持久摩擦锚

**【仅稳定线 v0.8.5.4】** §4.3 的光滑化头让"静摩擦"其实是一段**低速蠕变**而非真正粘住。稳定线 v0.8.5.4 用两件套修掉它:`absolute_epsv`(把 epsv 从场景尺度里拔出来)与**持久摩擦锚 friction anchor**(把每步重置的弹簧原点变成跨步锚点)。两者在 `0894958` 一起翻成默认开——**改变一切含摩擦场景的轨迹**。

> **出处口径(本小节特殊)**:v0.8.5.4 的代码**不在任一磁盘工作树里**——稳定线工作树被未提交的回退改动钉在 v0.8.5.3 内容上,phase-cd HEAD 全树 grep `absolute_epsv`/`friction_anchor` 零命中(已亲验)。本小节行号一律取自 git 对象 `c0339c8`(= tag `v0.8.5.4`),记作 `stable@c0339c8:GIPC.cu:NNNN`,与本册其它 `stable:` 前缀(指工作树 = `b8e27a1` 内容)**不是同一坐标系**。相关提交:`dc1a297`(特性)、`0894958`(默认开 + CHANGELOG)、`c0339c8`(strict 抑制)。

### 4.7.1 (a) legacy epsv:静摩擦精度被场景包围盒绑架

legacy 分支就是 §4.3 那一行(`stable@c0339c8:GIPC.cu:9500` 的 `:` 右支;phase-cd 同款 `09:613`):

```
fDhat = 1e-4 · eff_bboxDiagSize2            // eff_bboxDiagSize2 是对角线的平方 [m²]
⇒ epsv = √fDhat = 1e-2 · eff_diag  [m/s]
⇒ ε    = epsv · h                  [m]      (§4.3 的光滑化阈值)
```

关键在于**平方口径**:系数 1e-4 开方成 1e-2,而 `eff_bboxDiagSize2` 开方成 eff_diag,于是 epsv 与场景对角线**线性**成比例。1.9 m 的 flask_cap 场景给出 `epsv ≈ 19 mm/s`——**是 IPC 论文默认 `1e-3·l` 的 10 倍、其静摩擦精度建议值 `1e-5 m/s` 的 1900 倍**(`stable@c0339c8:GIPC.cuh:244-248` 注释;IPC 原文建议:静摩擦精度取 1e-5 m/s,抓取场景 1e-4 是好默认)。

这条派生的结构性错误是**量纲归属**:epsv 是**接触物理**(静→动摩擦转捩速度),不是几何。把它挂在包围盒上意味着——

- 多环境批量:env 数/间距一变,`bboxDiagSize2` 就变,**静摩擦精度随 batch 形状漂移**;
- 单场景:把机器人挪远 1 m,同一个抓取的静摩擦精度就变了;
- `absolute_dhat` 已经把 `eff_bboxDiagSize2` 改写成 `(absolute_dhat/relative_dhat)²`(§1.3),于是钉 dhat 会**顺带钉住 epsv**——但钉到的是那个比值碰巧算出来的数,不是物理上挑的数。

这是**包围盒派生参数家族的第三个成员**,前两个是 `dHat`(已由 `absolute_dhat` 治好)与 `κ`(已由 per-group κ / uipc 式物理接触尺度治好)——见 §1.3、§2.1。

`absolute_epsv > 0` 时改走绝对支:`fDhat = epsv²`(`stable@c0339c8:GIPC.cu:9498-9500`),与场景无关;环境变量 `STIFF_EPSV` 在 `GIPC::init` 里再覆盖一层(`:9499`)。`0` = legacy,逐位不变。

### 4.7.2 (b) 每步重置锚点 → 保持期蠕变(量级关系)

**机制**。由 §4.3 的 C1 头,静止区摩擦力幅为 `μλ·f1_SF_div·‖u‖ = μλ·‖u‖(2ε−‖u‖)/ε²`,在 `‖u‖ ≪ ε` 时线性化:

```
F_fric ≈ (2μλ/ε)·‖u‖        ⇒  切向弹簧刚度 k = 2μλ/ε
```

即光滑化头把静摩擦实现成一根**切向弹簧**。问题出在它的原点:`u` 由 `relDX3D` 算出,而 `relDX3D` 量的是**当前位置相对帧首 `o_vertexes`** 的位移(§4.1),帧末 `updateVelocities` 又执行 `o_vertexes = vertexes`(§7.2)——**弹簧的自然长度每步归零**。于是一个被稳定托住的接触,为了在本步重新生成同样的抵抗力 `F_t`,必须**每步重新滑够** `u*`:

```
u*      = F_t·ε/(2μλ)
creep_v = u*/h = F_t·(epsv·h)/(2μλ·h) = (F_t/(μλ))·epsv/2   ~  (load/(μ·N))·epsv
```

(`N ≡ λ` = §4.2 那个滞后一步的法向力幅;`F_t` = 该接触实际承担的切向载荷。代码注释与 `engine.py:313` 写作 `(load/(mu*N))*epsv`,略去线性化带来的 1/2 —— 本式按量级读。)

两个推论值得单独记住:

1. **蠕变速度与 dt 无关**——`ε = epsv·h` 里的 h 与除以 h 恰好抵消。减小步长不治蠕变(只把每步滑移等比例减小,步数等比例变多)。
2. **它与库仑锥无关**——蠕变发生在锥内深处,`F_t/(μλ)` 越小蠕变越慢,但永不为零。所以"受力远在摩擦锥内却在滑"不是锥违例,是光滑化头 + 锚点重置的构造性质。

**flask_cap 实测对账**(双臂 finray 抓取-提升-保持,400 步 ≈ 6 s 保持段,μ=3.5):

| 量 | 实测 | 折合速率 |
|---|---|---|
| 烧瓶平移滑移 | 3.7 mm / 6 s | ≈ **0.5 mm/s** |
| 瓶盖锥体转动 | 11–20° / 6 s | ≈ **3 °/s** |
| 每步位移漂移(newton-trace) | 10 µm/frame | — |

代回量级关系自洽:`creep_v/epsv = 0.5/19 ≈ 2.6%`,即 `F_t/(μλ) ≈ 5%`——**确实深在 μ=3.5 锥内**,与"受力远在摩擦锥内"的现象描述一致。这就是把 epsv 从 19 mm/s 降下来就能压蠕变的原因。

**但缩放不是严格线性**:`epsv=1e-5`(降 1900×)只把保持滑移从 3.71 mm 压到 0.05 mm(74×,`dc1a297`)。说明蠕变还有第二个地板(弹性顺应、接触对增删、λ 滞后),纯靠调 epsv 到不了零——这正是需要锚点的理由。

### 4.7.3 (c) 持久摩擦锚:携带与失效

**模型**。每个滞后配对携带一个累积切向弹性偏移 `e`(世界系,米);所有摩擦核改用

```
u_total = relDX_step + e
```

求能量/梯度/Hessian(核内注入点:`stable@c0339c8:GIPC.cu:1289`、`:1370`、`:1432`、`:1558/1640/1725/1809`、`:5295-5471`、`:7254/7332`,一律 `if(g_fric_anchor_d)` 守卫)。于是弹簧原点不再每步归零:静止接触稳定在**恒定 `u_total`**,`relDX_step → 0`,蠕变消失。设备侧句柄是两个 `__device__` 指针 `g_fric_anchor_d`(体-体对,lastH 槽序)/ `g_fric_anchor_gd_d`(地面对)(`:9623-9624`)。

**帧首携带 `GIPC::carryFrictionAnchors()`**(`stable@c0339c8:GIPC.cu:9801`;调用点在 `buildFrictionSets()` 末尾 `:9999`——即 §4.5 每帧/每驱动子步重建滞后集之后,把上一步的锚点匹配进本步新槽位):

- **体-体对按规范键匹配**。键 = `int4` 配对编码的原始位打包(`_fricPairKey`,`:9626-9636`;保负值,同一物理对 → 同一键)。上一步的键已排好序,核内做 `lower_bound` 二分(`_fric_anchor_carry`,`:9710-9735`)。**这是确定性的**:不依赖槽序、不依赖原子序,strict 安全。
- **地面对不用键**:锚点存在一个**按顶点号索引的稠密数组** `fric_anchor_gd_dense`(size = `vertexNum`),按槽 gather 即可(`_fric_anchor_carry_gd`,`:9737-9746`)。
- **携带即重投影**:取回的 `e` 立刻用**本步的** `tanBasis` 做切平面投影并重新钳幅(`_anchorProjectCap`,`:9645-9689`)。所以锚点跟随旋转的接触标架走,**离开切平面的分量当场丢弃**。

**帧末提交 `GIPC::commitFrictionAnchors()`**(`:9900`;调用点 `:17189`,在 `snapshotFrictionForce` 之后、`updateVelocities` 之前——**必须赶在 `o_vertexes = x` 归零本步位移之前**):`e_new = relDX_step + e_in` → 投影+钳幅 → 写出键;随后 `cub::DeviceMergeSort::SortPairs` 按键排序供下一步二分(`:9928-9938`,持久 scratch,不做每步 `cudaMalloc/Free`——thrust 默认分配器会同步设备)。地面侧先 memset 稠密数组再 scatter(带法向投影与钳幅,`_fric_anchor_commit_gd`,`:9769-9790`)。

**钳幅半径 = 滑动边界**:`‖e‖ ≤ eps = √fDhat·h = epsv·h`(`:9869`、`:9904`)。这个半径**恰好是 §4.3 光滑化头饱和到完整库仑力 μλ 的那个 ε**——所以"锚点顶到帽"就等价于"该接触已在滑动",帽上的径向回拉(radial return)**就是库仑滑动本身**,不是信息损失。

**锚点失效(= 回到 legacy 行为)的完整条件**:

| 条件 | 结果 | 出处 |
|---|---|---|
| 键未命中(新接触、配对编码/类型改变) | `e = 0` 重新起步——**与 legacy 逐位同义** | `:9731-9734` |
| 切平面变了 | 面外分量投影丢弃(面内保留) | `:9645-9689` |
| `‖e‖ > epsv·h` | 径向回拉到帽 = 库仑滑动 | 同上 |
| `clearFrictionAnchors()` | `fric_prev_count = 0` + 地面稠密数组 memset | `:9793-9799` |
| ↑ 调用者:`reset_transient_contact_state()` | teleport / episode 复位**必须**清锚(不能继承上一 episode 的接触) | `stable@c0339c8:sim_engine.cu:3640` |
| 特性关(config / `STIFF_FRIC_ANCHOR=0` / strict 默认) | 两个符号置 `nullptr`,所有核走 `if(g_fric_anchor_d)` 的 legacy 分支,**逐位等于 v0.8.5.3** | `:9884-9896` |

**反向条件(什么时候绝不能失效)**:容量增长必须**拷贝保留**已携带的集合(`:9830-9856` copy-grow)。理由写在注释里且是 batch 不变性的硬约束——直接丢弃会在**依赖 N 的帧**上抹掉锚点(不同 env 数的配对计数按不同节奏增长),制造批间差异。符号上传另有 `m_fric_sym_last` 缓存(`:9884-9896`),指针没变就不重传。

**效果**(`0894958` CHANGELOG + `dc1a297` 验证段,flask_cap 400 步,锚 + `epsv=1e-4`):烧瓶保持滑移 **3.7 mm → 0.00 mm**;瓶盖倾角**钉在 0.7–1.4°** 全程平直不再爬升(legacy 11° 且仍在爬;纯 `epsv=1e-5` 仍有 3.9°);每步漂移 **10 µm/frame → 5e-12 m/frame**。**注意**:只降 epsv 不加锚,压不住转动——转动分量对锚点的依赖比平移更强。

### 4.7.4 (d) 代价:Newton 变贵,以及 strict 的批不变性冲突

**epsv 越小 Newton 越贵**,机制就是 §4.7.2 的刚度式 `k = 2μλ/ε`:ε 正比于 epsv,epsv 缩小 ⇒ 切向弹簧刚度成反比放大 ⇒ 摩擦 Hessian 条件数变差 ⇒ Newton/PCG 迭代变多。实测标定:

| 配置 | step 成本 | 出处 |
|---|---|---|
| `epsv=1e-5` | **+12%** | `dc1a297` |
| `epsv=1e-4`(发布默认) | **无可测代价**(49 ms/step 不变) | `dc1a297` / `engine.py:318-319` |
| 锚(优化前) | +30% | `dc1a297` |
| 锚(发布态,抓取场景) | **+9%** | `0894958` CHANGELOG |

锚的 +9% 花在**动态 stick-slip 段**(预加载的切向弹簧带来额外 Newton/PCG 功),**保持段迭代数不变**——即代价与"真在滑"的时长成正比,而不是常数税。

**strict 冲突(`c0339c8`)**。`epsv=1e-4` + `h=0.01` 把钳幅半径收到 `epsv·h ≈ 1 µm`(legacy 在 1.9 m 场景是 `19 mm/s × 0.01 s ≈ 190 µm`,**收紧 190×**)。这么小的半径把摩擦能量推到**依赖 N 形状的 line-search 能量归约的 ulp 比较边界**上:一个离散的 accept 决策翻转,随后混沌放大。foldshirt strict 实测:**前 17 帧逐位相同,帧 18 一步之内 96% 顶点分歧**。bisect 干净利落——**epsv-only 绿、anchor-only 绿、组合红**。

处置:strict(以 `STIFF_SPMV_DET` 为签名)**默认压掉锚**,epsv 保留(它本身 batch 安全);merged / isolated 两个默认都开;`STIFF_FRIC_ANCHOR=1` 可在 strict 里强制开(`stable@c0339c8:GIPC.cu:9803-9823`,首次触发打印一行说明)。`c0339c8` 的门禁:strict 跨 env moveDir 0.0、逐次运行 vhash 一致(25 帧)、batch env0 max-delta 0.000。

> **这条冲突后来被根治了,但不在稳定线上**——见 §4.7.6。

### 4.7.5 (e) 与 §5「摩擦保真实测」的关系:那些数据是 legacy 路径测的

**§5 全部实测(剪切台 `0.600 ± 0.008`、暂态峰值 `0.792`)都在稳定线 v0.8.5.3 上完成,即 legacy 摩擦路径**:场景派生 epsv + 每步重置锚点。v0.8.5.4 默认开后**两个前提都变了**,引用时须分清:

- **§5.1 剪切台 μ 定标**:测的是**滑动段**的 `Ft/Fn` 饱和比。滑动段 `‖u‖ > ε` 走线性库仑支(§4.3),既不经过光滑化头也不受锚点影响(锚早已顶帽做径向回拉)——**结论 `0.600 ± 0.008` 与 §4.6 组合律在 v0.8.5.4 下依然成立**。
- **§5.2 的理论预期需要打折**:那里说"实测滑动比应略低于设定 μ,因为光滑化静止头在 `‖u‖ ≤ ε` 内欠饱和"。这个欠饱和量**正比于 ε**;v0.8.5.4 在 1.9 m 量级场景把 ε 收了约 190×,**欠饱和项随之几乎消失**。用 v0.8.5.4 复现 §5.2 那张待测的 μ 饱和曲线时,预期贴 μ 会比 v0.8.5.3 更紧——**不要拿 v0.8.5.3 的欠饱和幅度当 v0.8.5.4 的验收基线**。
- **§5.3 暂态峰值 0.792 不受影响**:它的成因是 **λ 滞后一步**(分子用旧 λ 封顶、分母是新法向力),与 epsv、与锚点都无关。v0.8.5.4 下该暂态照旧存在,逐帧 `Ft/Fn ≤ μ` 断言仍须留暂态余量。
- **§5 的"蠕变"话题在 v0.8.5.3 时点根本没被测过**:v0.8.5.3 的战役记录是"beaker 摩擦实验待开",而蠕变是 flask_cap 上另一条线发现的(§4.7.2)。**别把 §5.2 的"无存档实测"与本节的 flask_cap 数据混为一谈**——前者是滑动饱和(未测),后者是静止蠕变(已测),两个现象、两套指标。

**一句话口径**:§5 的读数结论在 v0.8.5.4 下继续有效,§5.2 的**理论预期**要按新的 ε 重算;凡引用 §5 数字,注明"v0.8.5.3 legacy 摩擦路径"。

### 4.7.6 移植状态:phase-cd 未合入,但移植分支已越过稳定线

**这是继「摩擦读数恒零修复」(§7.2)与 `reset_transient_contact_state`(§7.4)之后的第三个未移植项**——`codex/phase-cd` HEAD 全树 grep `absolute_epsv`/`friction_anchor` **零命中**(亲验)。

但**同仓分支 `port/friction-anchor-086` 已经把这批工作移到 0.8.6 线,并且在两个方向上超过了稳定线**(均亲验提交消息):

| 提交 | 内容 |
|---|---|
| `c735e13` | port(dc1a297/epsv):`absolute_epsv` 旋钮落 0.8.6 线 |
| `57015da` | port(anchors):持久摩擦锚落 0.8.6 线(参数化) |
| `9fd2905` | **根治 §4.7.4 的 strict 冲突**:真因定位到 `_penv_energy_accum` 用裸 `atomicAdd(double)` 按线程调度序累加 per-env 能量 → env0 能量带 N 依赖的末位抖动 → 翻转 S3 backtrack 决策(取证链 E1–E7:梯度、rz0、整个 PCG 循环跨 N 逐位相同,翻转发生在 PCG 之后)。改走 `binned_deposit`(Demmel-Nguyen 精确分箱,定序 combine 尾声)后**strict 里锚点可以默认开**:cross-env moveDir 0、run-to-run vhash 逐位、batch env0 N=2 vs N=4 **带锚**逐位 0.000e+00 |
| `1040edb` | **checkpoint 序列化**:摩擦锚是本引擎**第一个真正跨步的摩擦状态**(lastH 是每步从序列化位置重建的快照,锚点**重建不出来**)。未序列化时 merged 重启漂移 3.7e-07;新增 `kStateFricAnchor` 段(排序后的上一步配对键 + 切向锚 + 地面稠密锚),缺段标志 → 逐字节兼容旧布局且载入即清锚 |

**合入前的坑**(与 §7.4 同源):该分支基于 `6b0e02e`,**落后 phase-cd HEAD 148 个提交**(亲验 `git rev-list --count 6b0e02e..codex/phase-cd`),须先前移(rebase/merge)再重跑 22 段门禁;且 `9fd2905` 的 binned per-env 能量与整帧图路径(`frame_fsm/`)的交互未验。

---

# 5. 摩擦保真实测

> **版本前提(v0.8.5.4 读者必读)**:本章全部实测在**稳定线 v0.8.5.3 的 legacy 摩擦路径**上完成——场景派生 epsv(§4.3)+ 每步重置摩擦锚点。稳定线 v0.8.5.4 默认开 `absolute_epsv=1e-4` 与持久锚后这两个前提都变了:**§5.1 的定标结论与 §5.3 的暂态照旧成立,但 §5.2 的理论预期须按新 ε 重算**——逐条对照见 **§4.7.5**。

以下实测全部在**稳定线 v0.8.5.3**(摩擦读数修复后)完成;phase-cd 因读数分量恒零(§7.2)**不能**复现这些读出验证。

## 5.1 剪切台 μ 定标:0.600 ± 0.008

双关节剪切台(设定 gel μ=1.0 / bar μ=0.6,稳定线 `CHANGELOG.md:20-24` 原文),三个压深下实测滑动比:

```
|Ft| / |Fn| = 0.600 ± 0.008
```

与 bar 侧设定 μ=0.6 精确吻合(稳定线 `CHANGELOG.md:20-24`,v0.8.5.3 发布验证)。

**组合律注意(勿引用"主导侧"说法)**:CHANGELOG 只记录了设定值与实测吻合,**没有**给出组合机制解释。按 §4.6 的组合律,若被测滑动界面是 gel/bar 的自接触对,则 `_pair_mu = √(1.0×0.6) ≈ 0.775`(`02:384-396`),不是 0.6——几何平均不存在"由一侧主导"的语义。实测恰为 0.600 说明被测滑动界面的**有效 μ 本身就是 0.6**:或两侧同为 μ=0.6,或走地面摩擦路径(地面 μ 取顶点自身 `μ_gd`,不做配对平均,`16:115`)。该实验的具体界面构成未在仓内存档(见 §9);做 per-body μ 标定时一律按 §4.6 组合律推算——按"主导侧"机制预测会错约 29%。

## 5.2 μ 饱和曲线:无存档实测

对滑动接触验证 `Ft/Fn` 是否贴设定 μ(库仑锥饱和)的 beaker 抓取 μ 扫描,**在两个代码树、稳定线 CHANGELOG 与实测权威文档中均检索不到任何数据**;v0.8.5.3 时点的战役记录是"beaker 摩擦实验待开"。稳定线发布验证中唯一的摩擦保真数据点是 §5.1 的剪切台 `0.600 ± 0.008`。本册早先版本曾载有一张 μ∈[0.02,0.2] 的四点饱和表与一个 14 臂抓取矩阵——**全部无出处,已撤下,勿引用**。

理论预期(由 §4.1/§4.3 可推,供未来实验对照):实测滑动比应略低于设定 μ——光滑化静止头在 `‖u‖ ≤ ε` 内摩擦力低于饱和值,λ 又滞后一步,时间平均读数轻微欠饱和;暂态可短暂越过 μ(§5.3)。

## 5.3 单步滞后暂态:峰值 0.792 的解释

同一剪切台实验中,**粘滞(stick)期库仑锥不超限,但暂态峰值出现 `|Ft|/|Fn| = 0.792 > μ=0.6`**。这不是库仑锥违例,而是滞后摩擦的构造性质:

- 摩擦力上界是 `μ·λ`,其中 **λ 是上一(子)步冻结的法向力**(§4.1);
- 读出的 `|Fn|` 则是**当前**势垒梯度;
- 当法向力在一步内明显变化(压深增减、接触刚建立)时,分子按旧 λ 封顶、分母是新法向力,二者错位一拍——比值可短暂越过 μ,实测峰值 0.792;
- 下一(子)步 `buildFrictionSets` 重建 λ 后比值回落到锥内。

对下游意味着:用 `Ft/Fn ≤ μ` 做逐帧断言的验证脚本必须允许单步暂态(或用滑动稳态段做判定);暂态幅度随 dt 减小、随法向力变化率减小。

> 出处:剪切台实验记录(2026-08-11 战役);0.792 的原始日志未在主仓存档——见 §9 待核实。

## 5.4 方法论教训(写给复现者)

1. **任何早于 v0.8.5.3 读数修复的"摩擦为零"诊断都不可信**(§7.2 恒零 bug;phase-cd 上今天依然如此);
2. 轴向约定要先查场景:beaker 家族 **Y 轴朝上**(`ground_offset=0.75` 施加于 y),z 是水平轴——曾有把 z 当高度而误读出"提升段摩擦骤降"的教训,已撤回;
3. UMI 实录条件 μ=0.4 时物体本来就滑("recorded μ=0.4, which let the object slip — 1.0 holds better",库内注释);仿真 μ=1.0 是刻意加强,不是失真。

---

# 6. 地面接触

## 6.1 平面模型与配置

**【稳定线+phase-cd】** 半空间地面:`dist(x) = n·x − offset`(线性距离)。存储上有 5 个平面槽(`_groundNormal/_groundOffset`,`09:413-423`):槽 0 = 配置地面,槽 1-4 = ±x/±z 墙(offset −1,1,−1,1)——**但检测与能量只用槽 0**(kernel 解引用 `*g_normal/*g_offset`)。

| 参数 | 默认值 | 单位 | 含义 | 出处 |
|---|---|---|---|---|
| `Config.ground_normal` | `(0, 1, 0)` | 单位向量 | 地面法向(上方向) | `sim_engine.h:95`;上载 `01_config_upload.inl:15`;GIPC 成员 `GIPC.cuh:598` |
| `Config.ground_offset` | `-1.0` | m | 平面方程偏移:`n·x = offset` 处为地面 | `sim_engine.h:96`;`GIPC.cuh:599` |
| `add_ground(height=0.0)` | — | m | 场景 API,按高度加地面 | `sim_engine.h:218` |
| `add_ground_collision_skip(body_id)` | — | — | 把 body 加入 ground-skip 表(finalize 后调用;保持上轴分量 0 使地面接触共享) | `sim_engine.h:232-234` |

Python 侧 `Config(ground_normal=(0,1,0), ground_offset=-1.0)`(`engine.py:373-374`)。

## 6.2 检测与 d-floor fail-fast

`GroundCollisionDetect`(`09:807-817` → `_GroundCollisionDetect`,`06_kinetic_soft_ground.inl:9-44`),在 buildCP 内执行:

- 跳过 `_ground_skip_body` 标记的 body(检疫/skip 表,`:26-31`);
- `dist = n·x − offset`;
- **d-floor fail-fast**:`!isfinite(dist) || dist <= 0` → `atomicMin(_gdCollapse, −(svI+1))` 记录违规顶点(仍入表,`:33-39`)——log 势垒需要严格 `d > 0`,零/负距离是几何不可行而非"深接触";
- `dist² <= dHat` 才计入 `_environment_collisionPair[atomicAdd(_gpNum,1)] = svI`(`:40-43`)。

buildCP 尾部 `throwIfGroundDistanceInvalid()`(`10:3269-3277`)读 `_gdCollapse` 并交 `handleGroundCollapse`(§6.4)。

## 6.3 能量 / 梯度 / Hessian

能量(`_computeGroundEnergy_Reduction`,`energy/17_ground.inl:7-42`):

```
d² = (n·x − offset)²
E_raw = −(d² − d̂)² · log(d²/d̂)          (17:36;核内不乘 κ)
总贡献 = Kappa · E_raw                     (组合核乘入,见 PRINCIPLES_DYNAMICS.md §1.2)
```

> 形式上是 **RANK-1 型**(单 log),与自接触的 RANK-2(双 log,§1.2)不同——亲验代码事实;是否上游有意设计未查证(§9)。

梯度/Hessian(`_computeGroundGradientAndHessian`,`17:46-119`):

```
t = d² − d̂
g_b  = −2t·log(d²/d̂) − t²/d²
grad = κ · g_b · 2·dist · n
H    = κ · param · nnᵀ,   param = 4·H_b·d² + 2·g_b
```

**PSD clamp:`param < 0 → 0`**(`17:97-107`)——上游注释掉的 `if(param>0)` 曾造成不定的秩-1 块,此为修复;`STIFF_GROUND_HESS_LEGACY=1` 还原旧行为**【实验性,默认关】**(`09:446-450`)。支持 per-group κ 与图内设备 κ/计数(后者【仅 phase-cd】)。

## 6.4 检疫(quarantine)协议

多环境场景下,单个环境的地面不可行(RL teleport/驱动可在帧间制造)**不应杀死整个 batch**。协议第 1-3 步**【稳定线+phase-cd】**、第 4 步复活**【仅 phase-cd】**(行号为 phase-cd;稳定线在 `stable:GIPC.cu:6767-6803、10403-10419` 一带):

1. **归因与降级**(`handleGroundCollapse`,`10:3282-3337`):`_gdCollapse < 0` → 解出违规顶点,回读位置/body/normal/offset 算距离;**[iron-law demotion]** 运行中且 per-env 冻结机制在场时,地面不可行顶点是 env 可归因的 → `quarantineEnvOfVertex` 检疫该 env,而不是抛错杀掉健康 env;不可归因(初始化期)→ 抛类型化 `gipc::GeometryError`(Python 面 `pystiffgipc.GeometryError`),消息含 17 位精度 vertex/body/position/normal/offset/distance 与"log 势垒需要严格 d>0"提示。
2. **检疫动作**(`quarantineEnvOfVertex` → `quarantineEnv`,`multienv/isolation.cu:120-175`):置宿主+设备检疫旗、打印 `[per-env][QUARANTINE]`、懒分配 `_ground_skip_body` 并把该 env 全部 body 标进 ground-skip 表(此后该 env 对地面检测/势垒完全惰性)。
3. **帧首扫描**(`quarantineGroundInfeasibleAtFrameStart`,`isolation.cu:230-264`;调用点 `ipc_solver.inl:2577-2579`):在**任何 CCD-alpha 工作之前**运行纯旗探测 `_probe_ground_infeasible`(无对表副作用),循环至多 `active_group_count+1` 轮(atomicMin 一轮只报一个赢家),逐个检疫;没有这个探针,ground-CCD fail-fast 会在检测看到帧间新状态之前触发,绕过降级。`STIFF_SKIP_GRND=1` 可跳过(诊断)。
4. **复活【仅 phase-cd】**(`reviveEnv`,`isolation.cu:192-223`;**稳定线全树无此函数**——不是"teleport 未接线"而是机制整体不存在,稳定线 episode 重置的对应物是 `reset_transient_contact_state`,§7.4):[rl-reset] episode 重置的 teleport 触到被检疫 env 时清检疫态(宿主+设备旗、unmark ground-skip、清 env status),帧首扫描按新构型重新裁决;复活是自纠错的——仍不可行的 env 一帧内会被再次检疫。`teleport_abd_bodies/teleport_fem_vertices` 自动调用 reviveEnv(`engine_modules/03_step_getters_export.inl:2554-2564`、`04_teleport_checkpoint.inl:95-110`)。

CCD 侧配套:被检疫 env 的方向被 `_zero_dir_quarantined` 清零、ground alpha 归约对 skip-body 顶点中立(§3.2)。

---

# 7. 接触力读出原理

## 7.1 单位与符号

引擎内部缓冲存的是**增量势梯度** `∂E/∂x`(量纲 = 力·dt²):`dE/dx = −force·dt²`。物理接触力(牛顿):

```
F = −(∂E/∂x) / dt²
```

(`03_step_getters_export.inl:1753-1759` 注释)。符号 bug 史:首个发布按 `+1/dt²` 缩放,所有向量反向;静置方块回归用 `|Fy|` 把它藏掉了;修复后验证:静置方块净竖直力 = `+mg`(向上支持力)——两线同注释(修复自 v0.8.4.1)。

## 7.2 摩擦读数为何需要 pre-commit 快照:位移提交归零问题

滞后摩擦梯度是**本步位移**的函数:kernel `_calFrictionGradient(_vertexes, _o_vertexes, ...)` 消费 `x − o_vertexes`(`composite_externs.cuh:187-199`;host 包装 `12_host_wrappers_fem.inl:495-519` 传 `TetMesh.o_vertexes`)。而帧末提交 `updateVelocities` 执行 `o_vertexes = vertexes`(`gipc_modules/08_step_update_topology.inl:89,94`)——**提交后步内位移恒等于零**。

因此,任何在 `step()` 返回之后"重算摩擦梯度"的 accessor 数学上必然得到 0——尽管求解内部摩擦集活着(实测 266–270 个滞后配对)。这就是 `get_vertex_contact_forces(components=friction_lagged|total)` 的**摩擦分量恒零 bug**。

**稳定线 v0.8.5.3 的修复【仅稳定线】**:求解器在提交**之前**快照——`GIPC::snapshotFrictionForce(TetMesh)`(`stable:GIPC.cu:16504-16527`)镜像 accessor 的 binned-gradient 协议(`zeroBinnedGrad → calFrictionGradient → combineBinnedGrad`)把摩擦梯度存进 `m_d_fric_force_snap` 并置 `m_have_fric_snap`;调用点在 `IPC_Solver` 帧收尾、`updateVelocities` 之前(`stable:GIPC.cu:16800`,紧邻注释:"capture the lagged-friction force of THIS step before the commit below zeroes the in-step displacement it is computed from");accessor 读快照叠加(`stable:sim_engine.cu:3451-3496`)。

**phase-cd 现状【未修】**:全树 grep 无 `snapshotFrictionForce/m_have_fric_snap`;其 accessor 用 `have_friction = (h_cpNum_last[0]>0 || h_gpNum_last>0)` 判定后**在读回时重算** `calFrictionGradient`(`03:1713-1746`)——由于 phase-cd 的 `_updateVelocities` 同样步末提交 `o = x`,**phase-cd 上 step() 之后读 `friction_lagged`/`total` 仍得到零摩擦分量**(静态代码推理结论,未运行验证——§9)。`normal` 分量(components=0)两线均正确。**结论:摩擦力读数当前仅稳定线可信。**

## 7.3 API:get_vertex_contact_forces

**【稳定线+phase-cd】(摩擦分量仅稳定线可信,见 §7.2)**

```cpp
int SimEngine::get_vertex_contact_forces(double* out3, int n,
                                         bool include_ground = true,
                                         int components = 0);
```

(声明 `sim_engine.h:295-296`;实现 phase-cd `03:1690-1787`,稳定线 `stable:sim_engine.cu:3451-3505` 一带。)

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `out3` | `double*` | — | 输出缓冲,`min(n, vertexNum)` 个 (fx,fy,fz) 三元组,**牛顿** |
| `n` | `int` | — | 缓冲容量(顶点数) |
| `include_ground` | `bool` | `true` | normal 分量是否叠加地面势垒力 |
| `components` | `int` | `0` | `0` = **normal**(body-body 势垒 [+ground]);`1` = **friction_lagged**(solver 本步实际使用的摩擦势梯度:位置当前、λ/切基滞后一步——半隐式摩擦的诚实标签);`2` = **total** = normal + friction_lagged |

- **返回**:实际写入的三元组数 `nw = min(n, vertexNum)`;请求的分量不存在时输出全零并返回 nw。
- **单位/符号**:牛顿;`F = −gradient/dt²`(§7.1)。
- **顶点序**:phase-cd 按 `vertex_metis_to_input` 解扰为 input 序(与 `get_vertices()` 一致,`03:1770-1785`);稳定线版无 perm 解扰(直接引擎序拷贝缩放)——MAS/metis 重排场景下两线输出序不同,见 §9。
- **注意/陷阱**:
  1. `want_normal` 时会**就地重建** BVH+CP(post-step 状态,`03:1716-1720`)——每次调用有真实成本(~百 ms 量级于大场景),控制回路优先用 `get_stitch_max_stretch`(~70ms 量级)或批量版;
  2. friction 路径**绝不**重建摩擦集(会扰动下一步求解状态;读的是不可变 lastH 集,`03:1701-1702`);
  3. 早退语义经过审计:normal 集为空时 friction-only 请求不被吞掉(`03:1721-1725`);
  4. **phase-cd 上 components=1/2 的摩擦分量恒零**(§7.2);
  5. `m_skip_all_collision` 场景返回全零。
- Python:`get_vertex_contact_forces(include_ground=True, components="normal") -> (N,3)`,字符串映射 `{"normal":0, "friction_lagged":1, "total":2}`(`engine.py:1641-1661`)。

## 7.4 API:reset_transient_contact_state

**【仅稳定线】(v0.8.5.3 新增;phase-cd 全树无命中)**

```cpp
void SimEngine::reset_transient_contact_state();
```

(实现 `stable:sim_engine.cu:3620-3636`;pybind `bindings/pystiffgipc.cu:422`;Python `engine.py:941-952`。)

- **参数**:无。**返回**:无。
- **语义**:就地 episode 重置用。执行:`h_cpNum[0..4]=0; h_cpNum_last[0..4]=0; h_gpNum=0; h_gpNum_last=0; m_have_fric_snap=false; Kappa=0.0`。
- **动机**(稳定线 `CHANGELOG.md:31-38`):teleport 恢复了位姿,但当前/滞后接触配对镜像与自适应 κ 仍描述**上一个 episode**;第一次新 solve 会用陈旧配对表施加**幽灵摩擦**。实测:不调用时 vs 全新构建 24 µm 状态发散;调用后 6.7e-9 m。κ 清零使下一次 `IPC_Solver` 走首解路径(`Kappa<1e-16 → suggestKappa`)重新推导(κ 携带上一 episode 的接触历史:仅清镜像仍剩 0.9 µm 残差,来源即 κ)。
- **注意/陷阱**:
  1. **刻意不并入 teleport API**——中途 teleport 单个 body 不能清掉其它接触的摩擦状态;
  2. 正确用法:in-place episode reset = `teleport_abd_bodies` + `teleport_fem_vertices` 全部完成之后调用**一次**;
  3. phase-cd 的对应机制不同:其 teleport 自带帧入口配对集重建 + reviveEnv(`04_teleport_checkpoint.inl:112-123`),但**没有**滞后镜像/κ 清理 API——从稳定线迁移 episode-reset 代码时注意语义差异。

## 7.5 其它接触导出 API(简表)

| API | 单位 | ground | friction | 备注 | 适用线 |
|---|---|---|---|---|---|
| `get_vertex_contact_force_sum(offset,count)`(Python 名 `get_body_contact_force`) | **legacy:raw 梯度**(`dE/dx = −F·dt²`) | 含(C++ 实现调了 `computeGroundGradient`,`03:1174`;Python docstring 称"no ground"——以代码为准,见 §9) | 无 | pre-0.8.4 兼容;每调用重建 BVH+CP | 【稳定线+phase-cd】 |
| `get_body_contact_force_batched(starts,counts)` | raw 梯度 | 无 | 无 | 一 block 一 segment,env 间零共享;整批 1 kernel + 1 D2H(`03:1584-1688`) | 【稳定线+phase-cd】 |
| `get_pair_contact_force(A,B)` | raw 梯度(Python 侧乘 −1/dt²) | 无 | 无 | 临时把配对表指到子集重跑 barrier 梯度(`03:1350-1433`) | 【稳定线+phase-cd】 |
| `compute_contacts(rebuild=false)` + `get_contacts()/get_contacts_device()` | **牛顿**(核内乘 1/dt²) | 含(bodyB=−1) | 无 | per-contact (bodyA,bodyB,force);默认复用上一 step 的接触集,零重建(`03:1304-1348`,`12:410-472`) | 【稳定线+phase-cd】 |
| `get_collision_pairs_clean()` / `get_ccd_pairs_clean(move,α)` | — | — | — | UIPC 风格解码配对导出(PP/PE 以 −1 padding;`03:1186-1302`) | 【稳定线+phase-cd】 |

---

# 8. 参数与环境变量速查

## 8.1 配置参数(本册相关)

| 参数 | 默认值 | 单位 | 含义 | 出处 |
|---|---|---|---|---|
| `relative_dhat` | 1e-3 | 相对 | dHat_sqrt = relative_dhat × eff_diag | `sim_engine.h:40` |
| `absolute_dhat` | 0.0(关) | m | >0 钉死 dHat_sqrt;多环境推荐 | `sim_engine.h:66` |
| `friction_rate` | 0.4 | — | 自接触全局 μ | `sim_engine.h:24` |
| `gd_friction_rate` | 0.4 | — | 地面全局 μ | `sim_engine.h:25` |
| `ground_normal` | (0,1,0) | — | 地面法向 | `sim_engine.h:95` |
| `ground_offset` | −1.0 | m | 地面偏移 | `sim_engine.h:96` |
| `line_search_max_iter` | 64 | — | LS 预算(也是 ground/intersect 回退预算) | `sim_engine.h:80` |
| 派生:`dHat` | relative_dhat²·eff | m² | §1.3 | `09:612` |
| 派生:`dTol` | 1e-18·eff | m² | close 集阈值 | `09:608` |
| 派生:`fDhat` | 1e-4·eff | (m/s)²(= epsv²,§1.3 量纲说明) | 摩擦阈值基量 | `09:613` |
| 派生:`eps` | √fDhat·dt | m | 摩擦静止位移阈值 | `16:830` |
| 编译期:`RANK` | 2 | — | 势垒 log 幂次 | `contact/barrier_rank.h:7` |
| 编译期:`SFCLAMPING_ORDER` | 1 | — | 摩擦光滑化阶(C1) | `FrictionUtils.cuh:13` |
| 编译期:`USE_FRICTION`/`ADAPTIVE_KAPPA` | 恒开 | — | — | `CMakeLists.txt:93` |

## 8.2 环境变量(本子系统;默认全 off 除注明)

| 变量 | 作用 | 默认 | 适用线 |
|---|---|---|---|
| `STIFF_CCD_SLACK_A` / `STIFF_CCD_SLACK_M` | ground / self+swept ACCD slackness | 0.9 / 0.8 | 【仅 phase-cd】(稳定线硬编码同值) |
| `STIFF_CCD_CFL_FACTOR` | CFL cap 系数(floor 恒 0.5) | 0.5,域 (0,8] | 【仅 phase-cd】 |
| `GIPC_FORCE_CCD_SANITY` | 启用 line search 精确相交复查 | off | 【稳定线+phase-cd】 |
| `STIFF_EE_NOMOLLIFY` | 连 mollify 请求计数也关掉 | off | 【稳定线+phase-cd】 |
| `STIFF_EE_NODEDUP` / `STIFF_EE_CANON` / `STIFF_EE_DETGATE` / `STIFF_CCD_CANON` | strict 确定性组件(python resolver 成套设置) | off | 【稳定线+phase-cd】 |
| `STIFF_PERENV_BVH` / `STIFF_DECOUPLE_THRESH` / `STIFF_PERGROUP_KAPPA` / `STIFF_PERENV_ALPHA` / `STIFF_PERENV_PAR` | isolated 组件 | off | 【稳定线+phase-cd】 |
| `STIFF_SKIP_GRND` | 跳过帧首地面检疫扫描 | off(诊断) | 【稳定线+phase-cd】 |
| `STIFF_GROUND_HESS_LEGACY` | 地面 Hessian 撤销 PSD clamp | off | 【稳定线+phase-cd】 |
| `STIFF_GRAPH_LEGACY_KAPPA_DOUBLE` | 图内复活 κ 倍增 | off(=host-equivalent) | 【仅 phase-cd,实验性】 |
| `STIFF_ALPHA_TYPE_SPLIT` / `STIFF_ALPHA_RESIZE` | CCD 归约实验(前者实测 4-env foldshirt 慢 17%) | off | 【仅 phase-cd,实验性】 |
| `STIFF_SKIP_F` / `STIFF_SKIP_E` | 只跑 EE / 只跑 PT 检测 | off(诊断) | 【稳定线+phase-cd】 |
| `STIFF_ALPHA_STATS` / `STIFF_SWEPT_DIAG` / `STIFF_CCD_VALIDATE` / `STIFF_MCDUMP` / `STIFF_CONTACT_DBG` / `STIFF_FRICTION_DBG`(稳定线) | 诊断打印/宿主对照 | off | 见 `config/knob_registry.h`(【仅 phase-cd】的单一真源,G14 门禁) |

---

# 9. 待核实清单

以下各点为代码事实已确认、但**设计意图或旁证未闭环**的项,写作时均已在正文标注;引用前请人工确认:

1. **地面 λ 与自接触 λ 的 rank 不一致**(地面恒 RANK-1 型单 log,自接触 RANK-2 双 log;§4.2、§6.3)——代码事实,是否上游 GIPC 论文惯例/有意设计未查证。
2. **地面摩擦能量用 C0 闭式而自接触用 f0_SF(C1)**(§4.3)——同上,设计意图未见注释。
3. **`_cpNum[1]` 槽位**:代码中未见独立消费(rank 槽按 2/3/4 使用;§2.4)。
4. **稳定线 mlbvh 发射代码的 smooth=false**:与 phase-cd 同谱系推断,未逐行复核稳定线副本(§2.6)。
5. **phase-cd 摩擦读数恒零**为静态代码推理结论(kernel 依赖 `x−o`、步末 `o=x`、无快照;§7.2),未实际运行验证;亦未发现 phase-cd 有任何 pre-commit 的 accessor 调用路径。
6. **0.792 暂态峰值**:出自剪切台实验记录(2026-08-11),原始脚本/日志未在主仓存档(§5.3)。0.600±0.008 有稳定线 CHANGELOG 背书;**μ 饱和扫描与 14 臂抓取矩阵无任何存档出处,原表已从 §5.2 撤下**。另:剪切台实测 0.600 与 §4.6 几何平均组合律(gel/bar 自接触对应得 √0.6≈0.775)之间的界面构成疑点未闭环——实测吻合 0.6 意味着被测滑动界面有效 μ 就是 0.6(两侧同 μ,或地面路径取顶点自身 μ_gd),实验几何未存档(§5.1)。
7. **`get_body_contact_force` 的 ground 口径**:C++ 实现含 ground 梯度(`03:1174`),Python docstring 称 "no ground"——文档按代码口径写,docstring 待上游修正(§7.5)。
8. **稳定线 `get_vertex_contact_forces` 无 perm 解扰**:MAS/metis 重排场景下与 `get_vertex_positions` 的序一致性未逐行核对稳定线 getter(§7.3)。
