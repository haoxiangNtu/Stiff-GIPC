# StiffGIPC 手册 · 更新日历（v0.8.5 → 2026-09-07）

> 本分册是 StiffGIPC 双线（稳定线 / 工程线）自 v0.8.5 以来的完整版本与战役编年。
> 所有 commit hash、日期、数字均经 `git log` / `git show` 或对应仓库文档亲验；
> 无法亲验的少数数字逐条标注"待核实"。引用倍率/毫秒数时**必须连同测量口径**
> （场景 / 帧数 / 轮数 / 显卡 / 默认集）一起引用——本文每个数字都带口径。
>
> 相关既有证据文档（工程线仓库内，相对本文的路径）：
> [`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md) ·
> [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md) ·
> [`../A800_ALLEXAMPLES_TIMING_2026-08-01.md`](../A800_ALLEXAMPLES_TIMING_2026-08-01.md) ·
> [`../GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md) ·
> [`../PHASE_C_FRAME_GRAPH_PLAN.md`](../PHASE_C_FRAME_GRAPH_PLAN.md) ·
> [`../BVH_ADVANCED_VALIDATION.md`](../BVH_ADVANCED_VALIDATION.md) ·
> [`../RELEASE_NOTES_v0.8.6-rc1.md`](../RELEASE_NOTES_v0.8.6-rc1.md) ·
> [`../CI.md`](../CI.md)

---

## 目录

1. [双线总览与 tag 链总图](#1-双线总览与-tag-链总图)
   - 1.1 [两条线是什么、分叉在哪里、为什么分叉](#11-两条线是什么分叉在哪里为什么分叉)
   - 1.2 [tag 链总图](#12-tag-链总图)
   - 1.3 [tag 落点惯例说明](#13-tag-落点惯例说明)
   - 1.4 [两线关键行为差异（升级前必读）](#14-两线关键行为差异升级前必读)
2. [逐版本条目](#2-逐版本条目)
   - 2.1 [v0.8.5](#21-v085--2026-07-23--稳定线phase-cd-共同祖先)
   - 2.2 [v0.8.5.1](#22-v0851--2026-07-24--稳定线phase-cd)
   - 2.3 [v0.8.5.2](#23-v0852--2026-07-25--稳定线phase-cd分叉点)
   - 2.4 [v0.8.5.3](#24-v0853--2026-08-11--仅稳定线)
   - 2.5 [v0.8.5.4](#25-v0854--2026-08-12--仅稳定线)
   - 2.6 [v0.8.6-rc1-internal](#26-v086-rc1-internal--2026-07-27--仅-phase-cd内部)
   - 2.7 [v0.8.6-rc2-internal](#27-v086-rc2-internal--2026-07-28--仅-phase-cd内部)
   - 2.8 [phase-cd HEAD（未打 tag）](#28-phase-cd-head-b3ab747--2026-08-11--未打-tag)
3. [工程线主题时间线（按战役）](#3-工程线主题时间线按战役)
   - 3.1 [v0.8.6 结构重构 Phase 1–4 + P5](#31-v086-结构重构-phase-14--p5725)
   - 3.2 [mode 加固 + 能量层 E1–E3.6](#32-mode-加固--能量层-e1e36726727)
   - 3.3 [rc1 工程化（rc1 本体 + rc1→rc2 窗口）](#33-rc1-工程化727rc1-本体--rc1rc2-窗口)
   - 3.4 [rc2 + dlto 采纳](#34-rc2--dlto-采纳728)
   - 3.5 [Phase A/B：宿主往返手术](#35-phase-ab宿主往返手术728)
   - 3.6 [Phase C1–C3：条件图组装](#36-phase-c1c3条件图组装728)
   - 3.7 [Phase C4/C5 + Phase D：GPU-native RL](#37-phase-c4c5--phase-dgpu-native-rl729)
   - 3.8 [C6 战役：整帧图从"能跑"到"确定 + 可用"](#38-c6-战役整帧图从能跑到确定--可用72983)
   - 3.9 [Phase D episode/RL 吞吐 + Isaac 表皮](#39-phase-d-episoderl-吞吐--isaac-表皮73184)
   - 3.10 [跨版本横测 vs v0.8.5](#310-跨版本横测-vs-v08583)
   - 3.11 [BVH 战役](#311-bvh-战役8486)
   - 3.12 [resize/PCG 诊断与宽度/tier 战役](#312-resizepcg-诊断与宽度tier-战役8587)
   - 3.13 [alpha-tune（CCD 步长帽）](#313-alpha-tuneccd-步长帽8789)
   - 3.14 [回滚泄漏破案 + 成对审计 + 优化路线图（收官）](#314-回滚泄漏破案--成对审计--优化路线图收官810811)
4. [里程碑提交速查表](#4-里程碑提交速查表)
5. [金锚与门禁演化](#5-金锚与门禁演化)
6. [稳定线 v0.8.5.3 之后的提交明细与工作树状态](#6-稳定线-v0853-之后的提交明细与工作树状态)
7. [未推送/未发布状态说明](#7-未推送未发布状态说明)
8. [本文未决点（待核实清单）](#8-本文未决点待核实清单)

---

## 1. 双线总览与 tag 链总图

### 1.1 两条线是什么、分叉在哪里、为什么分叉

| 项 | 稳定线 | 工程线 |
|---|---|---|
| 仓库（本机） | `/home/ps/Downloads/Stiff-GIPC-stable-08` | `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph` |
| 分支 | `release/stable-0.8` | `codex/phase-cd` |
| 当前权威版本 | **最新 = v0.8.5.4**（tag `v0.8.5.4` = `c0339c8`，2026-08-12；`pyproject.toml:7` = `0.8.5.4`，亲验 `git show HEAD:pyproject.toml`）。**本手册的行号/行为引用基线仍是 v0.8.5.3**（`b8e27a1`，2026-08-11）——行号按该 tag 的 blob（复核用 `git show v0.8.5.3:<文件>`；工作树已于 2026-09-08 恢复为 v0.8.5.4 内容，此前误留的 8 文件暂存回退已清），v0.8.5.4 独有行为见 §2.5 | HEAD = `b3ab747`（2026-08-11，未打 tag；`pyproject.toml:7` 版本号 `0.8.6rc2`） |
| 发布形态 | 公开仓 `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel，CUDA 架构 sm_80/89/120 | 仅本地源码构建，**分支未推送远端**（`git branch -a` 无 `origin/codex/phase-cd`，亲验） |
| 文件布局 | **重构前单体**：`StiffGIPC/GIPC.cu` 16,884 行、`sim_engine.cu` 4,534 行、`mlbvh.cu` 3,171 行、`MASPreconditioner.cu` 3,126 行（`wc -l` 实测） | v0.8.6 模块化：四大单体拆成 `gipc_modules/`(现存 14，`00..14` 缺 `04`——barrier 融合装配已随能量层 E1d `7983be1` 迁至 `energy/03_barrier_fused_assembly.inl`；phase1 拆分时点为 15，见 §3.1) + `engine_modules/`(5) + `mlbvh_modules/`(7) + `mas_modules/`(6) + `energy/` 独立 TU + `core/` + `frame_fsm/` 等 |
| 服务对象 | 需要稳定行为、复现实验、wheel 安装的用户 | 引擎开发、GPU 驻留 RL、整帧 CUDA Graph、门禁基建 |

**分叉点**：`05c3f75`（= tag `v0.8.5.2`，2026-07-25，"feat(iron-law): quarantined envs go fully inert"；`git merge-base` 亲验）。

**分叉理由**：v0.8.5.2 是 **v0.8.6 大规模工程化（模块化重构、整帧图、GPU 驻留）开始之前的最后一个纯修复版**。稳定线从这里分出，只接收行为保守的修复（v0.8.5.3 的两个 contact-I/O 修复即来自 tactile 传感器线），承诺"不调用新 API 则轨迹与上一版完全一致"（稳定仓 `CHANGELOG.md:7-13`；该承诺的 teleport 例外见 §2.4 行为承诺条）；工程线则在 `codex/phase-cd` 上滚动推进全部 v0.8.5 之后的结构与性能工作。

**提交总量**（工程线，`git log v0.8.5..HEAD` 亲验）：**209 条**（2026-07-24 → 2026-08-11）。按日分布：7/24: 2｜7/25: 19｜7/26: 10｜7/27: 26｜**7/28: 38（峰值日）**｜7/29: 10｜7/30: 10｜7/31: 15｜8/1: 9｜8/2: 4｜8/3: 16｜**8/4: 23**｜8/5: 8｜8/6: 8｜8/7: 5｜8/9: 1｜8/10: 4｜8/11: 1。

### 1.2 tag 链总图

```text
                    v0.8.5      v0.8.5.1     v0.8.5.2
 (共同历史) ────────  2efaa72 ──── b6c1f09 ──── 05c3f75 ──┬──────────────────────────────►  工程线 codex/phase-cd
                    2026-07-23   2026-07-24   2026-07-25 │
                    numpy 2.x    SpMV OOB     iron-law   │   v0.8.6-rc1-internal   v0.8.6-rc2-internal        HEAD
                    修复等        preserve     检疫        │   6c9d730 (2026-07-27)  6b0e02e (2026-07-28)  b3ab747 (2026-08-11, 未打 tag)
                                                         │   重构+19-demo 自动化    hooks+dlto 决策包      优化路线图
                                                         │   ... 中间 209 条提交: 重构 phase1-4 / E1-E3.6 /
                                                         │       Phase A/B/C/D / C6 / BVH / alpha-tune / 收官审计 ...
                                                         │
                                                         └──► 稳定线 release/stable-0.8（分叉理由: 工程化前最后修复版）
                                                                │
                                                                ├── 1bc13ef (2026-07-31) contact-IO: 摩擦读数恒零修复 + reset API
                                                                ├── 1d05c7a (2026-07-31) teleport ABD 表面立即刷新
                                                                │
                                                              v0.8.5.3 = b8e27a1 (2026-08-11)  ◄── 本手册的稳定线引用基线（按此 tag 的 blob；工作树现为 v0.8.5.4）
                                                                │
                                                                ├── dc1a297 (8/11) absolute_epsv 旋钮 + 持久摩擦锚（真静摩擦）
                                                                ├── d7ab5bf (8/11) STIFF_NEWTON_TRACE 逐迭代 move-norm 诊断
                                                                ├── 0894958 (8/12) release 本体: 两者默认开
                                                                │
                                                              v0.8.5.4 = c0339c8 (2026-08-12)  ◄── 稳定线最新版本（strict 多环境抑制 anchor）
                                                                                                  ⚠ 改变所有含摩擦场景的轨迹, 见 §2.5
```

要点：

- `v0.8.5.3` / `v0.8.5.4` **不在** phase-cd 历史里（`git merge-base --is-ancestor v0.8.5.3 HEAD` 在工程线返回 NOT ancestor，亲验）。
- `v0.8.6-rc1-internal` / `v0.8.6-rc2-internal` 是**内部 tag**，只在私仓，未对外发布。
- 工程线 HEAD `b3ab747` 未打 tag，v0.8.6 **尚未发布**（见 §7）。

### 1.3 tag 落点惯例说明

本仓有一个可观察到的惯例（亲验三例，未向 owner 确认动机——见 §8）：**tag 常打在 release 提交之后一条的收尾/hook 提交上**：

| tag | release 提交本体 | tag 实际落点 |
|---|---|---|
| `v0.8.6-rc1-internal` | `3810f54`（"release(internal): v0.8.6-rc1"，2026-07-27） | `6c9d730`（同日，replay-trajectory 入 demo_verify） |
| `v0.8.6-rc2-internal` | `7e39591`（"release: v0.8.6rc2 internal"，2026-07-28） | `6b0e02e`（同日，pre-push hook 两缺陷修复） |
| `v0.8.5.4`（见 §2.5） | `0894958`（"release(v0.8.5.4)"，2026-08-12） | `c0339c8`（strict 抑制 friction_anchor 提交） |

引用版本内容时以 release 提交本体为准，引用"该 tag 可达的代码"时以 tag 落点为准。

### 1.4 两线关键行为差异（升级前必读）

> **最重要的一条：tactile 线的两个修复只进了稳定线 v0.8.5.3，phase-cd 尚未移植。**
>
> 1. **接触力读数中摩擦分量恒零的修复**【仅稳定线】：`get_vertex_contact_forces(components=friction_lagged|total)`
>    在稳定线 v0.8.5.3 通过 `GIPC::snapshotFrictionForce`（稳定仓 `StiffGIPC/GIPC.cu:16504` 定义、
>    `:16800` 在 IPC_Solver 提交前调用；注释 `GIPC.cuh:332-341`）在 `updateVelocities` 前把 lagged-friction
>    梯度快照到持久设备缓冲——因为该梯度是步内位移 `(x − o_vertexes)` 的函数，步末提交后位移恒 0，
>    事后重算恒为零。**phase-cd 没有此快照机制**（`engine_modules/03_step_getters_export.inl:1690-1748`
>    仍是 post-step 现算 `calFrictionGradient`，全树 grep `snapshotFrictionForce` 零命中，亲验）——
>    phase-cd 上 `friction_lagged`/`total` 分量**仍受恒零 bug 影响**（代码结构证实；运行时读数未实测，见 §8）。
> 2. **`reset_transient_contact_state()` API**【仅稳定线】：就地 episode 重置用，清零**三族**状态：
>    current/lagged contact-pair host 镜像 + lagged-friction 快照 + **自适应 Kappa 归零**（`g.Kappa=0.0`，
>    使下个 solve 走 fresh-process 的 `suggestKappa` 路径；稳定仓 `sim_engine.h:296-312`、`engine.py:941-952`、
>    实现 `sim_engine.cu:3620-3636`）。
>    修复的现象：in-place teleport 后首个 solve 用**上一 episode**的 pair 列表构建摩擦集
>    （实测 1116 个 stale pair → 幻影摩擦、24 µm 状态分歧）。分层实测（提交 `1bc13ef`）：只清
>    pair/摩擦镜像后仍残留 **0.9 µm**（自适应 kappa 携带上一 episode 接触历史）；连 kappa
>    一并清零才到 **6.7e-9 m**——复刻此数字必须包含 Kappa 归零。
>    phase-cd 没有该 API，但其 `teleport_abd_bodies` **内建**了 pair 集重建
>    （`03_step_getters_export.inl:2566-2582`：`invalidateRefitTopology` + `buildBVH()` + `buildCP()`，
>    与 `load_checkpoint` 同契约）——语义近似，**不逐位等价**，且**不复位 Kappa**（按稳定线
>    分层实测推定至少残留 ~0.9 µm 量级的跨 episode 状态携带）。

其余同名 API 行为差异（进程级模式锁、checkpoint v1/v2 格式互不兼容、teleport 强度差异等）不属于本分册主题，详见本手册的 API 分册与《稳定线 vs 工程线差异》勘探（工程线 checkpoint v2：`checkpoint/checkpoint_io.cu:33-53`，magic `STIFFCP2`/CRC-64；稳定线 v1：稳定仓 `GIPC.cu:16591-16631`，magic `0x53544B50`，无校验和）。

---

## 2. 逐版本条目

条目格式：日期 / tag 落点（提交）/ 适用线 / 内容 / 关键数字。CHANGELOG 行号除注明外均指**稳定线仓库**的 `CHANGELOG.md`（phase-cd 树的 CHANGELOG 止于 0.8.5.1，两树 CHANGELOG 自 0.8.5.2 起分叉，亲验 grep）。

### 2.1 v0.8.5 — 2026-07-23 — 【稳定线+phase-cd 共同祖先】

- **tag 落点**：`2efaa72`（2026-07-23 23:13 +0800，"fix(examples): numpy 2.x — ndarray.ptp() removed"）。CHANGELOG 记发布日期 2026-07-24（CHANGELOG.md:105）。
- **内容**（CHANGELOG.md:58-98 同段内容在稳定线树 :105 起）：

| 变更 | 说明 | 关键数字 |
|---|---|---|
| 摩擦 Hessian rank 修复 | ⚠ 行为变更——0.8.3 起装配丢失摩擦块，修复后所有含摩擦场景轨迹改变 | — |
| GPU 控制流快路径 | 每步宿主往返大幅削减 | **~150k → 数千** 次/步 |
| A800 wheel 支持 | wheel 默认 CUDA 架构扩为 `80;89;120` | 新增 sm_80（A800） |
| 多环境生产推荐 | `per_env_exit=True, env_newton_iter_cap=100`；冻结 status 2 / 检疫 status 3 | 验证口径：峰值 Newton ≤28 正常、>100 持续=不稳定信号 |

- **发布物**：公开仓 `haoxiangNtu/stiff-physics` wheel（cp311/cp312，sm_80/89/120）。

### 2.2 v0.8.5.1 — 2026-07-24 — 【稳定线+phase-cd】

- **tag 落点**：`b6c1f09`（2026-07-24 13:10 +0800）。
- **主修复：towel-strict SpMV 越界（OOB）**。真因：`GlobalLinearSystem::build()` 的 pre-solve 扩容用了 `ensure_capacity_discard()`，但扩容点在装配之后、buffer 正持有本迭代的活矩阵；towel_scramble（30×30 布）strict 模式第 26 帧触发 `2×168469 = 336938 > 336199` → 行列索引全 0 → `cudaErrorIllegalAddress`。**修复 = `ensure_capacity_preserve`（preserve-grow）**；strict 金锚 `f7fb5a786c2d7935` 不变（CHANGELOG.md:54-104 段；提交 `b6c1f09`）。
- **后续配套**（同窗口 `03e70f6`，7/24，**分叉点之前——两线共有**：`git merge-base --is-ancestor 03e70f6 HEAD` 在稳定仓为真，稳定线 `GIPC.cu:13255` 保有同一 `2.7 = 2×1.35` 注释、CHANGELOG 0.8.5.1 条目记录同机制，亲验）：常规增长搬到帧首 discard-grow（`max(装配界, 2.7×上迭代精确长度)`，2.7 = 2×1.35 留 35% 单迭代跳变余量），build 点 preserve 留作正确性后盾——避免 24GB 卡双驻留（[P0-mem]）。
- **附带 3 项防御加固**：`if(I1==0)` early-out 误分类 abd_abd、`d_unique_key_number` 别名、SPLIT_GH 16/9/4 与 SymGH 10/6/3 布局不兼容检查（提交 `b6c1f09` 正文）。

### 2.3 v0.8.5.2 — 2026-07-25 — 【稳定线+phase-cd；分叉点】

- **tag 落点**：`05c3f75`（2026-07-25 09:39 +0800，iron-law P1 完结）。v0.8.5..v0.8.5.2 共 6 条提交。
- **主题：iron-law 检疫（quarantine）**——中途 ground-infeasible 的 env **隔离而非杀进程**：
  - 隔离 env 完全惰性化：方向清零、位置冻结（alpha==0）、逐 env NaN 扫描、freeze-loop status 3 优先（`77dcb7a` + `05c3f75`）。
  - 新门禁 `test_env_midrun_quarantine.py`：4 个 strict env，env0 传送到地下 10 cm，验证其余 env **drift 恰为 0.0**。
- **同窗口审计尾款**（`fa0cd63`，7/25）：NaN line-search 缺口（NaN 与任何数比较皆为假，非有限试探能量落到 status 0 = **被当作"已接受的下降"**——发散 env 的 NaN 唯一能溜过检疫的路径；修复 = 非有限判 status 1 继续回溯，预算耗尽时响亮告警，`14_energy_linesearch_solver.inl:52-57` 注释）、`set_emit_caps` 集中化、revolute 限位 atan2 越界警告、`ExclusiveSum` 未初始化尾元素、`engine.py` `os.environ.setdefault` 进程全局串扰。
- **稳定线 CHANGELOG 条目**：仅存于稳定线树（`CHANGELOG.md:40-53`）。

### 2.4 v0.8.5.3 — 2026-08-11 — 【仅稳定线】

- **tag 落点**：`b8e27a1`（2026-08-11 18:41 +0800，"release(v0.8.5.3): stable-line patch — contact-IO fixes from the tactile line"）。**当前稳定线权威版本；工作树钉在此处。** 公开仓挂 cp311/cp312 wheel（sm_80/89/120）。
- **行为承诺**：无 solver-dynamics 改动；不调用新 API 时轨迹与 0.8.5.2 一致（稳定仓 `CHANGELOG.md:7-13`）。**⚠ 该承诺有一个上游 CHANGELOG 未列的例外**：`1d05c7a` 修的是**既有** API `teleport_abd_bodies` 的行为（v0.8.5.2 已存在，`git grep` tag 树 2 命中；修复前 teleport 后表面顶点滞留旧位姿），调用它的 0.8.5.2 脚本在 0.8.5.3 上轨迹**必然不同**——"完全一致"只对不调用 teleport 的脚本成立。
- **内容**（组成提交 `1bc13ef`、`1d05c7a`，均 2026-07-31）：

| # | 修复/新增 | 机制 | 验证数字 |
|---|---|---|---|
| 1 | `get_vertex_contact_forces` **摩擦分量恒零修复**（`friction_lagged` 恒零；`total` 读数本身非零、只是退化为 normal-only——勿用 `total==0` 作 bug 特征） | accessor 在步末提交后重算 lagged-friction 梯度，此时步内位移恒 0 → 恒零；修复 = `updateVelocities` 前 `snapshotFrictionForce` 快照到持久设备缓冲（稳定仓 `GIPC.cu:16504`、`:16800`；`GIPC.cuh:332-341`） | 解内 `h_cpNum_last`=266..270 对活跃；双关节剪切台（gel μ=1.0 / bar μ=0.6）实测滑动比 \|Ft\|/\|Fn\| = **0.600 ± 0.008**（三个压深） |
| 2 | 新增 `SimEngine.reset_transient_contact_state()` | 就地 episode 重置：清零 current/lagged contact-pair host 镜像 + 摩擦快照 + **自适应 Kappa 归零**（`g.Kappa=0.0` → 下个 solve 走 fresh-process `suggestKappa`；`sim_engine.cu:3620-3636`）；**刻意不并入 teleport API**（teleport 单 body 不应清其它接触的摩擦状态） | 陈旧配对幻影摩擦 24 µm 状态分歧 → 只清镜像/快照仍残留 **0.9 µm**（kappa 携带接触历史）→ 连 kappa 清零后 **6.7e-9 m**（`1bc13ef` 分层实测） |
| 3 | `teleport_abd_bodies` 表面顶点滞留旧位姿修复（`1d05c7a`） | teleport 写 q 族后立即对 ABD 点区间跑 `cal_x_from_q` 并同步 `o_vertexes`（修复前首个 line search 的 E0 用旧顶点评估 → 实测 40/40 帧 LS 预算耗尽） | — |

- **⚠ 移植状态**：以上 #1/#2 **未移植到 phase-cd**（见 §1.4）；#3 在 phase-cd 有独立实现（teleport 内建 BVH+CP 重建）。

### 2.5 v0.8.5.4 — 2026-08-12 — 【仅稳定线】

**稳定线最新版本。** 单一主题：**真静摩擦（true stiction）**——把持握场景的静摩擦蠕滑（stiction creep）从"结构性缺陷"变成"默认修好"。

- **tag 落点**：`c0339c8`（2026-08-12 01:05 +0800，strict 抑制 anchor 的收尾提交；符合 §1.3 的落点惯例）；**release 提交本体** `0894958`（2026-08-12 00:12 +0800，"release(v0.8.5.4): default-on true static friction"）。版本号 `pyproject.toml:7` = `0.8.5.4`、CHANGELOG 条目 `CHANGELOG.md:7-44`。
- **对外发布**：公开仓 `github.com/haoxiangNtu/stiff-physics` 的 Release `v0.8.5.4`（标题 "v0.8.5.4 — true static friction (default-on)"）**已正式发布**，`published: 2026-08-11T17:07:37Z`（= 2026-08-12 01:07 +0800，紧跟 tag 提交），`draft:false`/`prerelease:false`；资产 **cp311/cp312 双 wheel**：`stiff_physics-0.8.5.4-cp311-cp311-linux_x86_64.whl`、`stiff_physics-0.8.5.4-cp312-cp312-linux_x86_64.whl`。公开仓 README 的安装 URL 已由 `a38ede4` 指向 v0.8.5.4（`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` 亲验；OPEN_POINTS OP-001 已关闭）。
- ⚠ **本条目的全部稳定仓行号取自 `git show HEAD:<file>`**（勘探当时工作树被误留的 8 文件暂存回退钉在 v0.8.5.3 内容上，磁盘副本里看不到这些代码；该状态已于 2026-09-08 清除，现在磁盘文件与 `git show HEAD:` 一致，见 §6）。
- **⚠ 行为警示（升级必读）**：两个新默认值**改变所有含摩擦场景的轨迹**——不是"不调用新 API 就一致"的那类补丁版（对比 v0.8.5.3 的兼容承诺，§2.4）。逐位回到 0.8.5.3 需显式关掉两者。

**组成提交**（4 条，`git log v0.8.5.3..v0.8.5.4` 亲验）：

| 提交 | 时间 | 内容 | 关键数字 |
|---|---|---|---|
| `dc1a297` | 8/11 22:47 | **真静摩擦两件套**（当时**双双默认关**）：① `absolute_epsv` 旋钮（m/s；`fDhat = epsv²` 钉死，旧编码 `fDhat = 1e-4·eff_bboxDiagSize²` ⇒ epsv = 1e-2×有效场景对角线，1.9 m 场景 = **19 mm/s**，是 IPC 论文默认 `1e-3·l` 的 10×、静摩擦精度值 1e-5 m/s 的 **1900×**；稳定仓 `GIPC.cu:9490-9500`、`GIPC.cuh:244-249`）② 持久摩擦锚：每个 lagged 配对携带累计切向弹性偏移 `e`，能量/梯度/Hessian 一律评估 `u_total = relDX_step + e`（稳定仓 `GIPC.cu:1289`、`:1370`、`:1558` 等），`‖e‖` 在 `eps = √fDhat · h = epsv·h` 处做径向回拉截断（`_anchorProjectCap`，稳定仓 `GIPC.cu:9645-9657`；`eps` 计算点 `:9869`）——**打到帽子就是 Coulomb 滑动**；跨步携带按规范化配对键（bit-packed int4 + cub merge-sort + 二分查找，确定性、strict 安全），地面配对走逐顶点稠密数组；未匹配的新接触从 `e=0` 起（= legacy 行为） | 动机实测 flask_cap（双臂 finray 抓取-提升-保持 400 步）：烧瓶 6 s 内滑 **3.7 mm**、瓶盖锥体转 **11–20°**，而受力**远在 μ=3.5 摩擦锥内**。机理：`creep_v ≈ (load/(μ·λ))·epsv`。分档验证：`epsv=1e-5` 把保持段滑移 3.71 → **0.05 mm**（step 成本 +12%）；`epsv=1e-4` **零成本**（49 ms/步不变）；anchor + `epsv=1e-4`：滑移 **0.00 mm**、cap 倾角全程钉在 0.7°（legacy 11° 且仍在爬；纯 `epsv=1e-5` 仍 3.9°） |
| `d7ab5bf` | 8/11 23:19 | `STIFF_NEWTON_TRACE`：逐 Newton 迭代 move-norm 踪迹（默认关，开时每迭代一次 D2H；稳定仓 `GIPC.cu:16000-16011`） | 用作 anchor A/B 显微镜（flask_cap 保持段 f=150–400）：anchor **ON r1 中位 5.4e-12**（真不动点）vs **OFF 1.0e-5 m/帧** = 每步 10 µm 蠕滑（×50 fps = 0.5 mm/s，与观测滑移吻合）；动态段 f=80–150 anchor ON 多付 **+1.9 Newton 迭代/帧**（解真实 stick-slip 转换） |
| `0894958` | 8/12 00:12 | **release 本体：两者翻默认开**——`Config.absolute_epsv = 1e-4`（m/s，原 0.0 = legacy 场景派生）、`Config.friction_anchor = True`（稳定仓 `engine.py:318-326`、`:386-391`；C++ 侧 `sim_engine.h:66-74`、`sim_engine.cu:952-953`、绑定 `bindings/pystiffgipc.cu:47-48`） | **改变所有含摩擦场景轨迹**；step 成本 **+9%**（release 口径；`dc1a297` 提交正文的 +30% 是优化前读数）。⚠ **两侧默认值不对称**：C++ `SimEngineConfig::absolute_epsv` 仍为 `0.0`（legacy），1e-4 由 Python `Config` 写入（稳定仓 `sim_engine.h:70` vs `engine.py:321`）——绕过 Python 层直接用 C++ 引擎的调用者拿到的是 legacy epsv，anchor 则两侧同为 `true` |
| `c0339c8` | 8/12 01:05 | **strict 多环境默认抑制 anchor**：`m_fric_anchor_on = e ? (atoi(e)!=0) : (m_fric_anchor_cfg && !strict)`（strict 签名 = `STIFF_SPMV_DET`；稳定仓 `GIPC.cu:9803-9822`，抑制时打印 `[fric-anchor] strict mode: ... suppressed for batch invariance`）。真因：anchor 与 `absolute_epsv=1e-4` 组合后收紧的 cap 半径（`epsv·h ≈ 1 µm`）把摩擦能量推到 **N 形状相关的 line-search 能量和的 ulp 比较边界**上，翻掉一次 accept 决策 → 批不变性破坏。tag `v0.8.5.4` 落在此提交 | bisect：**epsv-only 绿、anchor-only 绿、组合红**；foldshirt strict env0 逐位 N=2 vs N=4：**前 17 帧逐位相同，第 18 帧一次 accept 翻转 → 单步 96% 顶点分歧**。本提交门禁：strict 跨 env moveDir 0.0、run-to-run vhash 25 帧一致、batch env0 max-delta 0.000。根修（N 不变能量归约）**排入 0.8.6** |

**默认值与逃生阀速查**：

| 旋钮 | v0.8.5.3 | v0.8.5.4 默认 | 逃生阀（逐位回 0.8.5.3） |
|---|---|---|---|
| `Config.absolute_epsv` | 不存在（epsv 恒场景派生） | **1e-4 m/s**（Python 层；C++ 结构体仍 0.0） | `absolute_epsv=0` 或 `STIFF_EPSV=0` |
| `Config.friction_anchor` | 不存在（摩擦锚每步重置） | **True**；strict 多环境**自动关**（`STIFF_FRIC_ANCHOR=1` 可强开） | `friction_anchor=False` 或 `STIFF_FRIC_ANCHOR=0` |

**IPC 原文口径**：静摩擦精度建议取 **1e-5 m/s**，抓取场景 **1e-4 是好默认**；epsv 越小 Newton 迭代越贵（稳定仓 `GIPC.cu:9490-9497` 注释、`engine.py:310-317`）。

**配套语义**：`reset_transient_contact_state()` 在 v0.8.5.4 里**多清一族**——`g.clearFrictionAnchors()`（稳定仓 `sim_engine.cu:3622-3641`，其中 `:3640`）：传送/回合重置的物体不得继承上一 episode 的摩擦锚。

- **⚠ 移植状态**：**这批工作完全不在工程线 `codex/phase-cd`**（两树 grep `absolute_epsv`/`friction_anchor` 均零命中，亲验）——继"摩擦读数恒零修复"与 `reset_transient_contact_state`（§2.4）之后的**第三个未移植项**。三项合并清单与移植路径见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1.0 / §1.6。

### 2.6 v0.8.6-rc1-internal — 2026-07-27 — 【仅 phase-cd，内部】

- **tag 落点**：`6c9d730`（2026-07-27 09:58 +0800，replay-trajectory 家族入 demo_verify，19-demo 自动化面完成）；release 提交本体 `3810f54`（同日，"release(internal): v0.8.6-rc1"，版本号 0.8.5.1→0.8.6rc1）。
- **定位**：内部测试 RC，不对外。主线 = v0.8.6 模块化重构 + 能量层分离 + 门禁体系（详见 §3.1–3.3）。
- **验收面**：12 段武装门禁套件全绿；strict 金锚 `f7fb5a786c2d7935`；A800 四核+矩阵（[`../RELEASE_NOTES_v0.8.6-rc1.md`](../RELEASE_NOTES_v0.8.6-rc1.md):7-27）；demo_verify 19-run 与 owner 13-demo UI 走查（同文档 :44）。

### 2.7 v0.8.6-rc2-internal — 2026-07-28 — 【仅 phase-cd，内部】

- **tag 落点**：`6b0e02e`（2026-07-28 01:59 +0800，pre-push hook 两个实弹缺陷修复：可重入 flock、GIT_DIR 泄漏）；release 提交本体 `7e39591`（同日，15 门禁绿，dlto 决策包归档）。
- **rc2 增量**（rc1→rc2 窗口 11 条提交，**全部带 15 段门禁绿 + 锚 `f7fb5a786c2d7935` 不动实据**；[`../RELEASE_NOTES_v0.8.6-rc1.md`](../RELEASE_NOTES_v0.8.6-rc1.md):96-141）：

| 增量 | 关键数字 |
|---|---|
| checkpoint 续跑 5e-6 破案（帧入口配对集为"第三类不入档状态"）→ load 后重建 | restart delta **5e-6 → 5e-17**；门禁收紧 strict 逐位 / 其余 1e-12 |
| ABD teleport 顶点失同步修复；跨引擎串态三族清除 | 21 处静态 device scratch 实例化 |
| RL episode-reset 契约（teleport 尾部重建配对集、检疫复活、`get_ls_exhausted_count`/`get_ls_nonfinite_count` 健康 getter、merged 穿地 reset 即抛 `GeometryError`） | 门禁 12 → **15 段**（+G13 geometry / G14 knob-registry / G15 rl-reset） |

- **rc2 tag 之后紧接的采纳**（时序亲验：release 本体 `7e39591` 01:43 → tag `6b0e02e` 01:59 → `19b257a` 03:12 → `40c9f11`/`45d74f0` 08:10，三者 `git merge-base --is-ancestor` 均**非** rc2 祖先——**checkout rc2 tag 得到的仍是旧锚 `f7fb5a786c2d7935`、dlto 未开**）：

| 变更（rc2 后，同日 7/28） | 关键数字 |
|---|---|
| **dlto（设备链接时优化）默认开**（`45d74f0`/`40c9f11`，RN:129 明言"rc2 后首个变更"） | 怪兽核 254 reg/23.8KB 栈 → **178 reg/2KB**；跨 TU ABI 调用 320→76；SASS −21%；运行时 towel −10~12% 全模式、foldshirt merged **−15.2%**、strict 几乎不动；构建代价全量 +76%、增量 ~60s 串行 dlink |
| strict 金锚迁移（随 dlto 采纳） | `f7fb5a786c2d7935` → **`0544461bd82123ae`**（run-to-run 逐位；4090 sm_89 ≡ A800 sm_80 跨架构同值） |
| 三模式全矩阵复验（`19b257a`，tag 后 docs） | 本地 19-demo × 3 模式 **56/56 绿**；A800 盘子 228 帧 3/3（strict peak 15 / 45.0s 与前役逐字复现） |

### 2.8 phase-cd HEAD `b3ab747` — 2026-08-11 — 未打 tag

- **提交**：`b3ab747`（2026-08-11 16:44 +0800，"docs: optimization roadmap anchored in the measured phase breakdown"）。`git describe --tags` = `v0.8.6-rc2-internal-147-gb3ab747`（rc2 后 147 条提交）；裸 `git describe` 输出 `v0.8.6-rc1-internal-164-gb3ab747`——`v0.8.6-rc2-internal` 是 lightweight tag，默认只认 annotated（rc1-internal 是 annotated），`git cat-file -t` 亲验。
- **内容**：优化路线图定稿（见 §3.14）。rc2 → HEAD 之间的实质工作全部按战役收录于 §3.5–3.14：整帧 CUDA Graph（Phase C/C6）、GPU 驻留 RL 与 episode（Phase D）、确定性栈、BVH 战役、回滚泄漏破案、成对审计。
- **相对稳定线的公开 API 增量**（名字级 diff 亲验）：`get_frame_status`/`FrameStatus`、episode 族（`launch_episode_async` 等 7 个）、GPU-RL 族（`prepare_gpu_rl` 等 12+ 个）、`gpu_rl_tensors()` 零拷贝 torch 视图、类型化异常（`LifecycleError` 等）、`fem_model()`；稳定线独有 `reset_transient_contact_state`。**注意** `save_checkpoint`/`load_checkpoint` **不是名字级增量**——稳定线绑定早已暴露同名 API（v1 格式，稳定仓 `bindings/pystiffgipc.cu:496-497`，magic `0x53544B50`）；工程线增量是 **v2 格式**（`STIFFCP2`/CRC-64，v1/v2 互不兼容，见 §1.4）——用 `hasattr` 探测两线时勿用此对名字。

---

## 3. 工程线主题时间线（按战役）

每个战役给：时间窗 / 代表提交 / 一句话成果 / 关键数字。全部提交在 `codex/phase-cd` 可达历史内（BVH 战役经 merge `5054fee` 进入）。

### 3.1 v0.8.6 结构重构 Phase 1–4 + P5（7/25）

单日 19 条提交。技术路线：**字节等同复合 TU**——拆分脚本断言"有序拼接 == 原文件 sha256"，编译产物不变，strict 位级锚原理上不可能变（`StiffGIPC/GIPC.cu:1-42` 自述；[`../V086_REFACTOR_PLAN.md`](../V086_REFACTOR_PLAN.md)）。

| 提交 | 阶段 | 成果 | 数字 |
|---|---|---|---|
| `8d649d9` | phase1 | 16,849 行 `GIPC.cu` → 15 个语义模块 `gipc_modules/00..14`；同提交带 Phase 0 安全网 `scripts/verify_gates.sh`（8 门禁）+ `scripts/anchor_scene.py` | GIPC.cu 瘦身为 include 壳（该文件 churn 16,874 行 ≈ +27/−16,847；全提交 21 文件 +17,163/−16,847，`git show --stat` 亲验） |
| `1a4a129` | phase1b | `mlbvh.cu`(3,171)→7 模块、`MASPreconditioner.cu`(3,126)→6 模块、`sim_engine.cu`(4,475)→5 模块——四大单体 27.6k 行全部模块化 | 27 文件，+10847/−10772 |
| `283db2e`+`1d57eeb` | phase2a | 15 个 block-SUM 归约尾统一为 `gipc_block_sum_to` 模板；再统一 7 个 MIN/MAX 尾 | +97/−588 |
| `dca94d1` | phase2b | pair-buffer 增长机制单一 owner（`contact/pair_buffers.cuh`）；**顺手抓到潜伏 OOB**（per-env DCD grow 无条件把 CCD 镜像重分配到 newcap+1，首次 per-env 溢出时分配缩到 published cap 之下） | — |
| `b08b65e`；`11538cb`/`29fe6ec` | phase2c/2d | env-isolation 单一 owner；帧编排入 `core/`，`core/ipc_solver.cu` 成为**第一个物理分离 TU**（host-only） | 20 个 kernel extern 声明 |
| `12ab87d`/`0f51e8c`/`0fe6e0a`/`e1694d6` | phase3 | `contact/encoding.h` int4 契约单源；`HostMirror<T>` 审计镜像（`STIFF_MIRROR_AUDIT=1` 武装）；RAII device-buffer owner 试点；槽位契约审计 | — |
| `9174e3e` | phase4 | `core/solver_stats.h` 跨 TU 计数器单一声明源；finalize 失败可命名（曾烧掉一次 A800 部署的 nlohmann 死 ifstream → 带路径 throw） | — |
| `4c96b31` | P5 | 删除验证死码（ge2sym、手写树归约 PCG 家族等）；冻结区（smooth/mollifier、close-set 链）不动 | −363 行 |
| `161e66e` | G0 | **门禁必须自带构建**——发现当日 P5→phase4 的门禁全跑在陈旧 `build/` 上（"武装验证空跑"事件）；修复后 8 门禁对累积树首次真实全绿 | exit 2 硬停 |

### 3.2 mode 加固 + 能量层 E1–E3.6（7/26–7/27）

| 提交 | 内容 | 数字/出处 |
|---|---|---|
| `afd9af1`/`0c9f56a`/`fda5106`（7/26） | 武装审计转常驻、`ModeConfig` 快照、mode promise 表（`multienv/mode_contract.h`）、冻结区 tripwire、G9 mode 包络+等价门禁、sanitizer runner | — |
| `eb3caf7`（7/26） | merged 契约按 owner 决定降级；能量/本构分离蓝图 | [`../ENERGY_SEPARATION_PLAN.md`](../ENERGY_SEPARATION_PLAN.md) |
| E1：`cf74065`→`b2b601a`→`d3335e1`→`7983be1`（7/26） | host-dispatch 入 `energy/`（12 类型 switch + `energy_terms.h` 术语表）→ 每项归约文件 → G/H 装配归位 → barrier 家族 | E1 完 |
| `0065489`（7/27） | **E2：X-macro 注册表 `GIPC_ENERGY_TERMS`**——"新增本构项 = 一个文件 + 一行注册"落地；死类型 11 真删 | 注册表 11 行（type 0..10） |
| E3：`19dd069`→`e393738`→`e6f6a75`→`cc0e8a3`→`1babc85`（7/27） | kinetic 首个 FP-kernel TU 分离 → ground+soft+delta → bending+triangle_membrane（跨 TU femEnergy 调用证明位级中性）→ friction TU（RANK 单源）→ **barrier TU，E3 完结** | 35+6 个头文件函数补 inline；nvlink/ld 重复符号由 G0 硬停抓住；fused barrier G/H 留复合 TU = 成文例外（受控抽离实验：254 reg/23,832B 栈 → 255/33,688（+41%）/SASS +9.5%，静态性能否决，[`../ENERGY_SEPARATION_PLAN.md`](../ENERGY_SEPARATION_PLAN.md):44-52） |

### 3.3 rc1 工程化（7/27，rc1 本体 + rc1→rc2 窗口）

> **版本归属提醒**（时序亲验）：rc1 tag `6c9d730` 打在 07-27 09:58；本表 12 行中**只有 `d0c0071`（02:00）与 release 行在 rc1 tag 内**，其余 11 行（`6c562bc` 19:51、`ce11d36` 11:37、`07abb93` 11:38、`16746b9` 11:44、`c86a1ea` 23:08、`93a68b6` 23:13、`07f2254` 23:23、`de893e9` 23:40、`7eac2e8` 次日 00:58、`b6d6b80` 次日 01:24 等）都落在 **rc1 tag 之后的 rc1→rc2 窗口**（`git merge-base --is-ancestor` 亲验非 rc1 祖先），其成果由 §2.7 的 rc2 条目收录发布——本表按主题聚合当日战役，与 §2.7 是同一事实的两个视角，勿按"通往 rc1 的工作"读。

| 提交 | 内容 | 数字 |
|---|---|---|
| `6c562bc` | "Harden simulator release gates and runtime state" 大聚合：exact-SHA push 验证、生命周期/资源所有权（单进程单 Engine `RuntimeOwnerLease`）、checkpoint v2、本构/FD 验证、可复现性能/sanitizer 门禁 | **68 文件 +5870/−1435** |
| `c86a1ea` | **checkpoint 载体破案**：load 后重建帧入口 contact pair set；诊断排除 8 个候选机制（PCG 容差 1e-6..1e-14、预条件器、warm start、BVH 拓扑等），最终 `get_collision_pairs_clean` 差 15 对定案 | restart delta **5e-6 → 5e-17** |
| `93a68b6`/`07f2254`/`b6d6b80` | descriptor phase-0a/0b/0.3：模式旗闩锁改值跟踪（跨引擎继承 bug：strict-then-merged 永不放松）、函数静态 device scratch 改实例持有 | v0.9 蓝图 = [`../DEVICE_DESCRIPTOR_PLAN.md`](../DEVICE_DESCRIPTOR_PLAN.md)（31 个可变 `__device__` 全局、34 个 ToSymbol 写点清点） |
| `de893e9`/`7eac2e8` | rl-reset：teleport 尾部重建 buildBVH+buildCP（与 load_checkpoint 同契约）、ABD teleport 从 q 同步顶点、隔离复活（`reviveEnv`）、G15 门禁 | — |
| `d0c0071` | **case-26 ABD 质量奇异修复**（owner 在 UI 亲手抓到"布不落"）：PSD clamp 把负特征值钉在 0 → 奇异 affine mass → NaN q_tilde；xarm7 body 8 反绕面网格触发 | 修复 = 特征值正下限（min eig −0.095 案例） |
| `ce11d36`；`06a6710`；`07abb93`→`16746b9` | FD 门禁（E↔G 有限差分一致性）；错误分类学 + STIFF_* 旋钮注册表（G13/G14）；CI 从 cron 轮询改 push 时门禁（pre-push hook 12 段武装套件；`SKIP_GATES=1` 为记录在案的逃生口） | [`../CI.md`](../CI.md) |
| `3810f54` / `6c9d730`(tag) | = v0.8.6-rc1-internal（见 §2.6） | — |

### 3.4 rc2 + dlto 采纳（7/28）

见 §2.7 表。补充：`e7ea834` cmake urdfdom 命名空间 target（A800 wheel 构建）；`e774fdd` `BENCH_IDLE_ALLOWLIST`（GPU 非空闲 fail-closed 的白名单机制）。

### 3.5 Phase A/B：宿主往返手术（7/28）

7/28 是峰值日（38 条），主体为本战役与 C1–C3。证据文档：[`../PHASE_A_GPU_RESIDENCY_PROFILE.md`](../PHASE_A_GPU_RESIDENCY_PROFILE.md)。

**Phase A 剖析**（`eee2f64`；A800 盘子 228 帧 merged，dlto rc2 wheel）：

| 指标 | 值 |
|---|---|
| 墙钟 | 29.2 s（128 ms/帧） |
| GPU 内核忙碌 | ≈7.0 s（31 ms/帧） |
| **GPU 等宿主** | **≈22 s ≈ 75% 帧时间** |
| 阻塞 memcpy | 31,475 次 = **138 次/帧** |
| `cudaMemcpyToSymbol` | 30,029 次 = 132 次/帧 |
| stream sync | 34,412 次 = 151 次/帧 |
| kernel launch | 341,867 次 = **1,500 次/帧** |
| 全驻留理论上限 | ≈3–4×——**该口径针对 A800 盘子 228 帧大帧回放**（29.2s→~8-9s，`PHASE_A…PROFILE.md:19`），此负载上图开实测反而 +35~43%，**上限从未兑现**；Phase D 的 A800 5.0× 是**另一负载**（RL 微步 300 步），regime 不同不可混接 |

**Phase B 勘察与手术**：

- `e46226d` 勘察（B1 设备线搜索已默认开）；`23d9c74` B2' 指针生命周期审计判死 naive cache，B3 升主。
- `c782f8d`/`16c4fce` B3 归因：4090 74% busy vs A800 25% busy 同一引擎结构——**往返次数是乘数**（每次 ~百 µs），非拷贝字节数。每迭代往返普查：`ls_mas_setup` 8.0 次（王者）、`line_search` 6.2、GH ToSymbol 5.4、ccd 3.2、convert 1.0、pcg-wait 1.0（productive）。
- **手术 ①–⑧**（`7e3246a` 同日系列；代表 `7e424bf` surgery-2：ToSymbol setter 值缓存）：260 帧 towel 全程 `GH_assembly` ToSymbol 8,426→5、`line_search` **19,805→0**；后续手术把 gdCollapse/ccd 计数/BVH 根 AABB/gp 计数/MAS level 循环逐一改设备驻留（⑥ 根 AABB 设备驻留：ccd 5.69→**0.81** 往返/迭代 = 纯 72B 标量链理论地板）。
- `7b7698b`/`2add5f5`/`752e9bd` B2'：spmv 读设备活 triplet 数、MAS apply 读设备层级、跨 solve PCG graph cache（`STIFF_PCG_GRAPH_CACHE`，默认关）。

### 3.6 Phase C1–C3：条件图组装（7/28）

| 提交 | 阶段 | 成果 | 数字 |
|---|---|---|---|
| `815c269` | C-1 chunk1 | merged BVH morton 排序 thrust→cub 预分配 scratch（消 capture-blocker 的内部 cudaMalloc/Free；位级同序） | 全宽 LSD radix |
| `dc0eb88` | C-1 chunk2 | line-search 回溯自尾图（`STIFF_LS_GRAPH`，默认关）：每回溯付一次 24B 打包读代替每试探 12B 决策读 | NVTX 指纹 12B D2H **29→0**（towel recipe，4090） |
| `389b530`…`d611138` | C2 | 帧状态 + capture-safe 控制暂存 → 事务性帧图边界 | 首提交 `389b530` 15 文件 +432/−41；收尾 `d611138` 6 文件 +985/−8（`git show --stat` 亲验） |
| `d6c16a5` | C3 | PCG 组成嵌套条件图 | +690 行 |

图层次（[`../PHASE_C_FRAME_GRAPH_PLAN.md`](../PHASE_C_FRAME_GRAPH_PLAN.md):85-116）：root → Newton WHILE → {GH, PCG WHILE, 收敛 IF, LS WHILE}；每帧一次 root launch + 一次 `FrameStatus` D2H；失败帧从入口快照逐位恢复。CUDA 条件图陷阱（成文）：嵌套 WHILE 前必须显式重设条件句柄。

### 3.7 Phase C4/C5 + Phase D：GPU-native RL（7/29）

| 提交 | 内容 | 数字 |
|---|---|---|
| `0e1b3f9` | "complete Phase C graphs and Phase D episodes"——C/D 骨干落地 | 31 文件 +4144/−109 |
| `9a592c5` | **Phase D 设备 ABI**：`prepare_gpu_rl()` 捕获可复用单帧条件图；捕获审计硬性要求 **0 host / 0 H2D / 0 D2H 节点**（fail-closed）；外部 CUDA policy 写 packed `(joints,3)` float64 action，同流自 launch | 13 文件 +1746；API 面 `prepare_gpu_rl`/`launch_gpu_rl_async`/`synchronize_gpu_rl`/`end_gpu_rl`/`get_gpu_rl_device_abi` 等 |
| `0735481` | A800 sm_80 验证 | 金锚 `0544461bd82123ae` 跨架构位级相同；nsys 稳态 40 步 **0 H2D / 0 D2H / 0 host sync**（宿主仅 40 次 `cudaGraphLaunch` + **80 笔 24B D2D action 发布**——ABI 每步一图两笔小 D2D；提交正文原写 40 笔，2026-08-06 干净 sm_80 复测与 4090 节点级 Nsight 均定格 80，[`../GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md):46,66-67,109） |
| `5f746cf` | C4-a：碰撞、CCD、line search 进整帧图（`STIFF_C4_COLLISION_GRAPH=1`，默认关；录制时容量镜像钉 `h_cpNum`/`h_gpNum`） | +911 行 |
| `1d389bc` | C4-b..d：kappa 与摩擦进图（`kappa_dev` 尾参、`FrameDeviceState::kappa`、快照回滚） | kappa 轨迹逐值对上基线（3.3586e6/3.3804e6/3.3757e6/3.3743e6） |
| `502f324` | A800 接触负载验证 | 碰撞+摩擦+关节驱动下 0 同步；A800 慢链 ~214 µs 平均 enqueue |
| `f5308e3` | **C5：isolated 模式整帧图化**（`STIFF_C5_ISOLATED_GRAPH=1`）——逐 env line search 变条件 WHILE | +926 行；12B/8B/4B 每迭代读全消；G19 隔离门禁：图路径跨环境耦合 1.791e-07 vs per-env 树基线 1.796e-07（零新增耦合） |
| `e1788a8` | merged vs isolated 默认路径基准 | towel isolated 快 13%/22%（4090/A800），foldshirt isolated 慢 69%/74%——算法性质非机器噪声 |

**strict 模式刻意不图化**：用户 2026-07-29 决定——容量网格归约会合法重排求和序，准入等于换锚战役（[`../PHASE_C_FRAME_GRAPH_PLAN.md`](../PHASE_C_FRAME_GRAPH_PLAN.md):292-294）。

### 3.8 C6 战役：整帧图从"能跑"到"确定 + 可用"（7/29–8/3）

约 45 条提交。全记录：[`../A800_ALLEXAMPLES_TIMING_2026-08-01.md`](../A800_ALLEXAMPLES_TIMING_2026-08-01.md)。开门提交 `7eee806`（7/29）：容量训练三重错（按 worst-case 而非 observed、按 per-arity 而非 per-pair、按 per-class 而非 per-payload；foldshirt MAX_PAIRS 737k 的 triplet 包络本需 ~44 GB）。

| 子阶段 | 提交 | 要点 | 实测 |
|---|---|---|---|
| C6-b | `81b60e5` | 4 个容量训练缺陷（普查自馈 242k→524k→1048k OOM 24GB 卡；tier 按帧尾而非帧峰训练等）；`0f0347e` 图内 kappa 更新复活宿主已退休策略（"图/宿主策略漂移"案例） | foldshirt merged N=1 92% 帧进图 |
| C6-c/d | `7baa0c8`/`0475c1a` | 地面塌陷可恢复、tier 增长瞄准失败轴（err_primitive 位掩码只长报告越界的轴） | 全轴齐长曾致 512k→2.77M→11.8M triplet 三连爆 OOM |
| C6-e | `65da2e5` | foldshirt 60 帧全图内；两个"重试是新帧"缺陷（attempt==0 门控的帧边界块 → 无条件） | 61/62 = 98% 帧录入且快于宿主 |
| C6-g/h | `48b5db3`/`124b3c5` | towel 是确定性缺陷非物理缺失（graph-on crumple 散布 0.77..1.03 vs 关图 0.905 三连）；beaker 60 帧过（重试 kappa 从失败尝试的垃圾梯度导出 **614.4→8.63 = 71× 过软** → 17.3M 对雪崩；修复 = 重试继承 attempt-0 的 post-initKappa 值） | — |
| C6-i | `b58a0b7` | **溢出帧改在 release solver 收尾**：增长决策回宿主帧边界，图只覆盖帧内部（iron law 的重试版；~1e-6 扰动源除） | 默认 `capacity_fallback=true`；`STIFF_GRAPH_INGRAPH_RETRY=1` 恢复旧行为（实验） |
| C6-k | `8bba75d` | 武装态非确定性根因定位：frame 2 梯度位级相同、首个 moveDir 差 1e-16 → raciness 在 `calculateMovingDirection` | — |
| C6-l | `1a5769b`/`454fb59` | spmv_det + ee_canon 准入整帧图；**det 旋钮下整帧图位级确定** | towel 220 帧逐帧哈希三连全同；4 个叠加缺陷修复（含 canonical pair-slot 排序 cub 两趟字典序 radix） |
| C6-m/n | `ff35cc4`/`b14e6fd` | 容量回退跑真正关图帧（`s_layout_override_off` RAII）；`STIFF_FULL_GRAPH_MIN_VERTS`（默认 1024）以下谢绝整图机器 | A800 全例配对计时（`f09596b`）：15 例 13 过、graph-on 过例区间 **1.09–2.75×**（源文档散文段自写 1.09–1.94× 有误——其自己的表把 forcegrip 2.75×、beaker 2.07× 均计为 pass，`A800_ALLEX:16,18` vs `:30-31`；4090 为 1.6–3.3×）；towel 82× 异常：C6-n 解剖把 468 s 中的 **219 s** 归给 19 个重录风暴帧（A800 每次 capture 20–30 s），其余为容量宽度固定成本——**但 C6-o 的 nsys 归因随即推翻了重录假说**：two-graph towel 全程**零** capture/instantiate 调用，真因是 OVF 已标记而宿主帧末才看、导致空转**烧 150k PCG 迭代**（`A800_ALLEX:35-44,50-53`）。止血后 towel 图 env 6s vs off 7s |
| C6-o | `fe83618`/`1ed5073` | **根治空转**——nsys 归因**推翻重录假说**（two-graph towel 全程**零** capture/instantiate 调用，A800_ALLEX:50-53）：真因 = OVF 已在设备标记而宿主帧末才看；修复 = 宿主 Newton 循环逐迭代轮询 pinned 快照，见 OVF 位立即中止尝试（否则截断梯度空转烧 150k PCG 迭代 = 8× 无效功）；每轴增长连击升级（8 帧窗内重越级则 2×，上限 4×） | towel full-graph **502→51 s** |
| C6-p | `ed2617f`/`47c37d6`/`edbc6e1` | 稳态开销法证（pre-launch 12.5ms 非主因；PCG 迭代数与宽度 graph/host 相同）→ **默认 headroom 2→1** | forcegrip 4090 2.08×→**1.39×**、A800 3.25×→**1.75×**、beaker 1.61×→1.21×（vs 关图）；代价 97%→94% 全图覆盖率。**口径注**：该轮（8/1）未记帧数/轮数；两天后 C6-w 轮（8/3）同场景为 1.96→1.46×——两轮为独立测量口径（混沌带 + 负载差），不可串成同一时间序列读作"回退"，中间不存在已解释/未解释的性能事件 |
| C6-q | `575539f` | v0.8.5 标准复审抓 3 个本 session bug（C6-n 全旁路吞了 tier 训练 → 收窄为只谢绝 FULL 图） | 套件 **22/22** |
| C6-r/s/t | `79f88f4`/`e460207`/`68f1e16` | episode 通道经济学、CDP 探针（cub CDP 可用但图不 fence CDP 子核）、**设备侧 grid-dim 更新可行**（修正"烘焙宽度是结构性"论断） | episode 通道大帧最慢：beaker 94 帧 4090 host 156.4 / step 图 244.4（1.56×）/ episode 319-326 ms（~2.05×） |
| C6-t step1 | `dda5430` | converter 唯一性趟按 length 而非 capacity 发射 | forcegrip 4.2M 载荷曾按 8.4M 发射；nsys 记其为整帧图最大单项开销（2248 ms 中的 1284 ms） |
| C6-v | `c74071a`/`f92f4a0` | **真动态图 `gipc::graph_resize`**（单线程 resizer 节点读设备计数器调 `cudaGraphKernelNodeSetGridDim`；replay/条件体内/capture 下均生效；sm_89/CUDA 12.8 端到端验证）+ **原子长龙归因**：真代价是零 pad 三元组全 hash 到 (0,0) 同挤 9 个 bin 的 atomic convoy，**不是发射宽度** | 干净卡对照：`binned_block_merge_scatter` on 93×@2999 µs vs off 62×@133 µs = **22× 每实例**，gap 的 50%（270.7 ms）；此前归因被共卡 20GB 训练任务污染 |
| C6-w | `b842d2f` | **跳零默认开（`STIFF_SKIP_ZERO_DEPOSIT`）+ 门禁两层化**：det 层 = `STIFF_SPMV_DET` 下逐位（数学性质）；default 层 = 基线自身 3 次 run-to-run 噪声包络（budget = max(4×noise, 1e-11×scale)）——比原逐位门禁更强 | 跳零修复轮（`c74071a` 正文/A800_ALLEX:278-279）：forcegrip 图开 13.34→9.78 s（−27%，对照 graph-off 6.81 s，step 图比 **1.96→1.44×**）；出货复测轮（`b842d2f`，4090 干净卡 60f 中位 3：on 10.37 s / off 7.10 s）step 图比 **1.96→1.46×**——两轮口径不同勿混拼；实证 merged 本就非 run-to-run 确定（release 自身帧 2 分歧 2.2e-14 → 帧 119 达 1.1e-4），跳零扰动 ~2 ULP（3.6e-16）低两个数量级、det 栈上精确中性；22/22 绿、金锚不动 |
| C6-x/y/z | —/`f5ddca7`/`ef27556` | 图开剩余差距解剖（图在所有该赢的轴上赢但让 GPU 多做 34% 功：memset 节点 36%、干跑 BVH 18%、padded 数据 26%）；RL 门禁速度地板补 1/dt 因子（速度 = 位置差分/dt，噪声放大恰 100×@dt=0.01）；**`STIFF_GRAPH_DEVICE_RESIZE` 转默认开**；pad-class allowance 落为 opt-in（问题源已被 `~0ull` 哨兵修掉） | device-resize：all-kernel GPU −10%、memset32 136→83 ms；pad 削减四次实测 ~0% 墙钟（vs 跳零 −27%）——"padded WORK 贵、padded LAUNCH WIDTH 近乎免费" |
| C6-aa | `686a94d` | 收官复审 4 缺陷全修：① graph_resize 全局槽数组**跨图串扰**（episode 重放改全图节点 → 欠清零静默错物理）→ 16 槽环独占 + episode 图首次真正武装；② 钉页 static 跨引擎 → 成员；③ layout 旗 → `thread_local`；④ 滞回 streak 副作用 → peek/commit 拆分 | — |
| C6-j 系列 | `37775f0`/`2c5ded8`/`af37df3`（7/31） | G18 包络从 N 条基线取最大成对散布（默认 4，`STIFF_G18_BASELINE_RUNS`）、误差取到最近基线距离；A800 修正：速度地板带 1/dt、kappa 等价地板 1e-6×scale | — |

### 3.9 Phase D episode/RL 吞吐 + Isaac 表皮（7/31–8/4）

| 提交 | 内容 | 数字 |
|---|---|---|
| `0d167ac`/`12fbac6` | D4：两个 capture 内分配 bug（converter3x3 merge bins 只在 retier 分支 reserve → error-900）；纯 ABD 类守卫不可满足（pad hash(0,0) 全归 abd_abd → `cap3≥cap3+其余` 永假） | 修复后 D4 接触稳态 episode 门禁 PASS（43 帧） |
| `dece015` | G17e：per-env 掩码 GPU-RL reset（`launch_gpu_rl_reset_masked_async`） | env0 位级回 prepare 快照（0.0）、env1 保留演化态、混合态可续步；掩码在设备内存，done-flag kernel 可零传输翻转 |
| `c1dd071` | episode vs host 吞吐（4090 接触场景） | N=150：7.63→4.34 ms/帧 = 1.76×；N=600：11.49→4.14 = **2.78×**；episode 平坦 ~4.1–4.3 ms/帧 |
| `a01cf27`（8/3） | 单图发射多帧 GPU RL episode（`prepare_gpu_rl_episode`/`launch_gpu_rl_episode_async`） | — |
| `9a314ac`…`ec1f1a3`（8/3） | full-dynamic-graph 契约、重放证据、持久 GPU 运行时设计与回退边界文档 | [`../FULL_DYNAMIC_GRAPH_DESIGN.md`](../FULL_DYNAMIC_GRAPH_DESIGN.md)、[`../PERSISTENT_GPU_RUNTIME.md`](../PERSISTENT_GPU_RUNTIME.md) |
| `5e7a887`（8/4） | **Isaac 风格 step() 表皮：一面两通道**——用户只见 `step()`；`prepare_gpu_rl()` 自足（`LayoutForceOnScope` thread_local RAII、内部一次 tier 训练帧、零环境旋钮）；prepare 后 `step()` 自动改道 RL 图（薄异步入队）；`gpu_rl_tensors()` 经 `__cuda_array_interface__` 零拷贝 torch 视图 | [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md)（总纲）；step() 表面统一零开销：3.14 ≈ 手工 `launch_gpu_rl_async` 3.18 ms/步（4090） |
| `5117e2f`（8/4） | prepare-timeline 试验台 + **容量平稳性第二判据**：驻留契约要求 capacity stationarity | 真实 foldshirt 抓握轨迹第 15 帧 auto-prepare 后 **25/25 帧 OVF_TRIPLETS 失败**（fail-closed 成立；接触升级轨迹必须走 step 通道——结构矛盾，不是"慢一点"） |

GPU-native RL 证据链定格数字：铰接一帧图 551 节点（无碰撞，出处 [`../PHASE_C_FRAME_GRAPH_PLAN.md`](../PHASE_C_FRAME_GRAPH_PLAN.md):248,312）/ **1057 节点**（contact+friction，以下均出自 [`../GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md):53-128）；4090 Nsight `--cuda-graph-trace=node` 稳态 40 步：40 launch、80 笔 24B D2D、5655 节点事件、**0 H2D / 0 D2H**、2.461 ms 提交窗零同步；A800 同构验证 11.571851 ms 提交窗零同步（工件 `artifacts/a800-fa6e03e/`）。

### 3.10 跨版本横测 vs v0.8.5（8/3）

提交 `b9b48c5`（文档追加于 [`../A800_ALLEXAMPLES_TIMING_2026-08-01.md`](../A800_ALLEXAMPLES_TIMING_2026-08-01.md):341-372）。平台口径逐行标注：大帧两行 = **4090 干净卡**（该节标题）；RL 微步行的 19.3/9.7/3.85 = **A800**（[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):131-137 双列表定标，4090 同负载为 4.25→3.72→3.18 ms/步 = 1.14×/1.34×——**⚠ 源文档口径冲突**：A800_ALLEX 把该 RL 表放在 "(4090, clean GPU)" 标题节内（300 步）而 SIM_EXEC 同数字标 A800（150 步），本文从 SIM_EXEC/工作记忆取 A800，冲突记入 §8 待核实）：

| 负载 | v0.8.5 | HEAD 默认（step/宿主） | HEAD 图开 |
|---|---|---|---|
| forcegrip 60 帧总墙钟（中位×3） | 7.26 s | 7.10 s（−2%） | 10.37 s（+43%） |
| beaker 60 帧总墙钟 | 10.09 s | 10.78 s（+7%）* | 13.57 s（+35%） |
| RL 微步 D4 关节 ABD 链压地（300 步，dt=0.01，A800，ms/步） | 19.3 | 9.7（**2.0×**） | gpu_rl 驻留 **3.85（5.0×）** |

结论原句："**The regime, not the feature, decides.**"（决定胜负的是 regime，不是特性。）

\* **beaker +7% 已破案**：差 = 纯启动段 ~0.3–0.6 s，**每帧持平**——step 通道"每帧 == v0.8.5"的承诺成立（[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):123 的 §6 交错复测：beaker 10.42 / 11.11，差 = 0.6 s 启动段；下限 0.3 s 出自同文档 :167 附录 A 终局对比 beaker 行"启动 +0.3s"）。破案证据在文档而非独立提交（见 §8 待核实 #2）。引用 "+7% 回归/open item" 的旧措辞时必须带此更正。

同期 4090 附录 A 终局对比（2026-08-04，3 轮交错，门禁 21/21 绿，[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):154-178）：forcegrip 115.2→**105.5 ms/帧（−8%）**；finray 289.5→**270.7 ms（−7%）且方差 233–315 → 267–280 大幅收窄**；RL 微步 4090 4.00→**3.14 ms/步（1.27×）**；finray 逐核归因：barrier GH **2.9×**、frictionH 2.3×、FEM GH 1.8×（融合装配战果）——HEAD GPU 总功 1.25× 于 v0.8.5 却墙钟更快 = 宿主/同步开销大降。

### 3.11 BVH 战役（8/4–8/6）

在独立分支 `codex/bvh-full-campaign`（worktree 隔离）上做，`5054fee`（8/6）合回 phase-cd（32 文件 +9820/−187，**全部旋钮 opt-in 态**）。两份证据文档必须合读：[`../BVH_ACCELERATION_VALIDATION.md`](../BVH_ACCELERATION_VALIDATION.md) + [`../BVH_ADVANCED_VALIDATION.md`](../BVH_ADVANCED_VALIDATION.md)，再加 [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md) 附录 B 的 merged-4env 复验。

| 日期 | 代表提交 | 内容 |
|---|---|---|
| 8/4 | `96b52fb`（候选总装，25 文件 +1996）、`04689f3`（GPU PLOC builder）、`a803c1d`（BVH8）、`1c02653`（VF pair cache）、`9416d15`（PLOC 拓扑摊销+精确 refit）等 "validation:" 系列 | 树质量/构建策略/查询排序/缓存全候选矩阵验证 |
| 8/5–8/6 | `fa6e03e`（关 4090 战役）、`e0476ac`（release/managed 显存验证加固）、`3be83d9`（关 A800 战役） | 跨架构收口 |
| 8/6 | **`474d1dc` 裁决** | 两个赢家候选在主战场 fs4 merged-4env 复测全输 |

裁决数字（A800 fs4 4env 60 帧，4 轮墙钟中位，[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):191-214）：host base 945.6 ms/帧｜+`STIFF_BVH_QUERY_ORDER` 960.9（+1.6%）｜+`STIFF_BVH_PLOC`+refit-128 1001（+5.9%）｜两者 1062.7（**+12.4% 更差**）｜图+PLOC **8510 ms（8 倍崩塌）**——PLOC 的变宽度迭代聚类被图捕获烤进最坏宽度 × O(log N) 轮，**对图路径有毒**。

对照 4090 1550 帧长轨迹（graph off）：merged 1env 基线 96.05 → PLOC/refit **87.75 ms/帧（−8.64%，merged 赢家）**；isolated 4env 组合束 −3.92%；A800 merged 组合束 −5.13% 但 **A800 isolated 全部候选倒退（+1.5~4.9%）**。

**结论**：架构 × 模式 × 环境数三维都不允许统一默认——全部 BVH 候选保持显式 opt-in（`474d1dc`）。同期否证：`7acba8f` host-bound class converts 第五次否证；`c92344f`/`9037154` alpha type-split 与 margin-table sizing 均 REFUTED。唯一早期接受的精确优化：`STIFF_EE_RANGE_PRUNE=1`（EE original-index subtree pruning，物理候选多重集逐位不变；EE-CCD kernel −13.14%、全核 −0.74%，opt-in）。

### 3.12 resize/PCG 诊断与宽度/tier 战役（8/5–8/7）

| 提交 | 内容 | 裁决/数字 |
|---|---|---|
| `f4cc4c8`/`2654264`（8/5） | 图内 globaltimer 相位戳（`STIFF_GRAPH_PHASE_TIME`）+ stage-kernel pad 卫生（masked lane 写零 triplet 恢复 pad-is-zero 不变量）；**顺手发现宿主 PCG 早退 bug**（每配置恒 8.0 iters/Newton 在首个 K-batch 边界退出） | — |
| `0f4075f`/`8c4497f`/`0742495`；`516a71a` | PCG 退出码/spmv 宽度审计/body 子相位戳**定罪 MAS apply**；apply-node resizer 实测 | apply-node resizer 零结果 |
| `ccf6735` → `0d2ad7e`（8/6） | alpha-reduction resizer + **resizer 成本模型**：resize 节点每 replay 都跑（`<<<1,1>>>` × 每 PCG 迭代），三轮 wall-clock 一致反向 → **迭代级 resizer 解除武装、保留帧级**；staging/compact 类核**不可**收窄（pad lane 写零是语义） | 1099/1027/1059 vs 1121/1070/1141 ms，~6% 更差 |
| `ff7ae09`/`dbc2c33` | width-fit（训练宽度跟随负载）与滞回 tier shrink（opt-in） | 第一版探针测了个寂寞（`m_peak_cpNum` 镜像从不复位）；五个宽度旋钮清扫**全零收益**（shrink −1.3%≈0） |
| `531211b`（8/7）→ `a46852f` | 细容量阶梯 `STIFF_TIER_STEPS`（headroom 扫描确立 padding 线性成本：1×/2×/4× = +11.7%/+19.1%/+67.7%）曾被记为"图路径首个稳健正收益"；随后 **A/A 校准矩阵否决**：同配置成对差中位 7.3%(host)/9.1%(graph)/4.8%(t4)——单批 <10pp 效应不可分辨（解释此前所有正负号翻转） | graph vs host 24 对 median **+17.1%**（17/7 方向一致）定格为当期图开成本 |

方法论定版（[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):233-234）：全轨迹 Newton 总数为主判据（±1%）；60 帧窗墙钟噪声 ±10pp 不可裁 <10pp 效应；凡裁决必须 A/A 对照。

### 3.13 alpha-tune（CCD 步长帽）（8/7–8/9）

| 提交 | 内容 | 数字 |
|---|---|---|
| `0ee5387` | CCD 步长帽旋钮（`STIFF_CCD_SLACK_M` + `STIFF_CCD_CFL_FACTOR`，opt-in）；fs4 重段 1845 迭代普查 | α 束缚者三分天下：ground 34.6% / CFL 32.8% / refined ACCD 32.4%；CFL 在 22.6% 迭代把 α 压到 ACCD 认证值之下（中位 1.48×），另 10.2% 充当有益地板故不可直接删 |
| `5759acd` | 修复：单一 `cfl_factor` 承载三种语义（CAP 可放大 / FLOOR 放大 = 未认证位移的隧穿风险且无 fail-fast / refined 咨询阈值）——**只放 CAP** | — |
| `affb479`；`8a60d08`（8/9） | 附录 C 最终配方与安全分析；附录 D 最快验证总配置 | 推荐配置（仅重接触）：`STIFF_CCD_SLACK_M=0.9 STIFF_CCD_CFL_FACTOR=1.0` → Newton **−7.9%**（5 批配对全负）、墙钟 −3~4%（A800 fs4-4env 全轨迹）；轻接触零收益或小亏勿开；**默认全关** |

### 3.14 回滚泄漏破案 + 成对审计 + 优化路线图（收官）（8/10–8/11）

图开 +10~15% 盈余的最终解剖，8/10–8/11 四条提交收官：

| 提交 | 内容 | 关键数字 |
|---|---|---|
| `322c243` | 审计定性：+10–15% 图差主要是**可修的 rollback-leak bug** 而非图理念（graph 臂 Newton +5–10%；LS 预算耗尽 2/2 轨迹 vs host 0/2；烟枪 = 耗尽都落在 graph-fallback 帧，64 次减半后 α=1e-21 仍超 E0 64–395%——零步不可能升能量） | — |
| `369ac25` | 三段探针链 `STIFF_LSX_DIAG`（E0-vs-trial 位级逐项、FINAL ARBITER 活参重发能量分派）；A800 判 **STATE-DESYNC**：位级 + 13 槽未动 → vertex 镜像与 q 在 LS 入口不一致，只有 stitch/soft 跨双空间读所以只有它爆 3e5..6e7×；`STIFF_POSTLS_FREEZE` 否证冻结语义假说（−1.6%≈0） | — |
| **`4e49fda`** | **正确性修复**：`capture_full_graph` 快照/还原**漏抄 ABD q 族（q/q_prev/q_v/q_tilde）+ per-group kappa**（两图事务有、整图没有）——OVF 帧破 `x == J·q` 不变量 → 回退帧 LS 必然耗尽；修复 = 镜像补齐（+73 行；FRAME_OK 早退保证接受帧位级不动） | ls_exhaust **2→0** 双轮验收 + 22/22 门禁 |
| **`4219f37`** | **成对审计** `STIFF_FRAME_FORCE_ROLLBACK=2`：整图解完整帧 → 设备态翻 RETRY → 急切重放还原核 → release solver 重解同帧——同态成对样本 | 1551 个同态成对样本/趟（A800 foldshirt 4-env）；**同态 graph vs host Newton 仅 +1.0%**，869/1550 帧位级相等——图内求解语义不是生产差距 |
| **`b3ab747`**（HEAD） | 优化路线图定稿：[`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md) | A800 foldshirt 4-env merged 1551 帧宿主 410 s（263 ms/帧）相位剖面：**PCG 56.9%（MAS apply 单项 = 全仿真 27%、SpMV 8%）、GH 装配 22.0%、CCD+LS 19.9%、DCD 1.0%、BVH 0.2%**——全部检测合计 1.3%（检测类优化 = 死路）；优先队列：fp32 MAS apply（上限 ~13%）→ α research 档（−16% Newton，前置穿透审计）→ per-env converged-mask 早退 → ABD 12×12 块预条件；附死胡同表防止重诉 |

图开盈余最终解剖（`OPTIMIZATION_ROADMAP.md:75-81` + 4219f37/322c243）：**图内求解语义 +1%；主体 +6% = 事务壳 ULP 扰动 × merged 混沌轨迹效应；+4% 图化增量**；回滚兜底循环非物理中性（D' 臂 +100%）。根治留"首分歧帧二分"（低优先）。（注："B' 9342±230 vs 宿主 8808±60，15σ" 的 ±/σ 细分出自提交正文与工作记忆合读，未逐字全文核对——见 §8 待核实 #3。）

**通道定论**（同文档 :75-81）：RL 微步/episode 驻留 **2.8–5× 胜，是图的唯一主场**；大帧回放图开 = 墙钟 +3%~75% 场景相关 → **大帧走宿主（step 通道），图只用于 RL 微步**。

---

## 4. 里程碑提交速查表

按时间排序；hash 均在对应仓库亲验可 `git show`。

| 主题 | 日期 | 代表 hash | 一句话成果 | 关键数字 | 适用线 |
|---|---|---|---|---|---|
| v0.8.5 | 7/23 | `2efaa72`(tag) | 摩擦 rank 修复 + GPU 控制流快路径 + sm_80 wheel | 往返 ~150k→数千/步 | 共同祖先 |
| v0.8.5.1 | 7/24 | `b6c1f09`(tag) | towel-strict SpMV OOB：discard-grow 毁活矩阵 → preserve | 2×168469>336199；锚 `f7fb5a786c2d7935` 不变 | 稳定线+phase-cd |
| v0.8.5.2 | 7/25 | `05c3f75`(tag) | iron-law：隔离 env 完全惰性 | drift 恰 0.0；**分叉点** | 稳定线+phase-cd |
| 重构 phase1 | 7/25 | `8d649d9` | GIPC.cu 16,849 行 → 15 模块，字节等同 | 8 门禁绿 | 仅 phase-cd |
| 重构 phase1b | 7/25 | `1a4a129` | 四单体 27.6k 行全模块化 | +10847/−10772 | 仅 phase-cd |
| pair-buffer owner | 7/25 | `dca94d1` | 单一 owner 顺手抓潜伏 OOB | — | 仅 phase-cd |
| G0 build | 7/25 | `161e66e` | 门禁自带构建（空跑事件封口） | exit 2 硬停 | 仅 phase-cd |
| 能量 E2 | 7/27 | `0065489` | 新本构项 = 一文件 + 一行注册 | 11 行 X-macro | 仅 phase-cd |
| 能量 E3.6 | 7/27 | `1babc85` | barrier TU，E3 完结 | 41 处 inline 修 | 仅 phase-cd |
| 大聚合 harden | 7/27 | `6c562bc` | 门禁 + 运行时状态加固 | 68 文件 +5870/−1435 | 仅 phase-cd |
| checkpoint 破案 | 7/27 | `c86a1ea` | load 后重建帧入口配对集 | 5e-6→5e-17 | 仅 phase-cd |
| ABD 质量奇异 | 7/27 | `d0c0071` | PSD clamp 钉零 → 特征值正下限 | min eig −0.095 案 | 仅 phase-cd |
| rc1 | 7/27 | `3810f54` / `6c9d730`(tag) | 内测 RC | 12 段套件 + 19 demo | 仅 phase-cd |
| rc2 | 7/28 | `7e39591` / `6b0e02e`(tag) | 内测 rc2 | 15 门禁；56/56 矩阵（`19b257a`，tag 后复验，见 §2.7） | 仅 phase-cd |
| dlto 默认开 | 7/28 | `40c9f11` | 设备 LTO，锚重钉 | `0544461bd82123ae`；254→178 regs；SASS −21%；foldshirt merged −15.2% | 仅 phase-cd |
| Phase A 剖析 | 7/28 | `eee2f64` | 75% A800 帧时间 = 宿主往返 | 138 memcpy + 132 ToSymbol + 151 sync + 1500 launch/帧 | 仅 phase-cd |
| B3 手术 | 7/28 | `7e424bf` | ToSymbol 值缓存 | line_search ToSymbol 19,805→0 | 仅 phase-cd |
| C1 LS 图 | 7/28 | `dc0eb88` | line-search 自尾图 | 12B D2H 29→0 | 仅 phase-cd（`STIFF_LS_GRAPH` 默认关） |
| C3 | 7/28 | `d6c16a5` | PCG 嵌套条件图 | +690 行 | 仅 phase-cd |
| Phase C/D 骨干 | 7/29 | `0e1b3f9` | 图 + episode 骨干落地 | 31 文件 +4144 | 仅 phase-cd |
| D 设备 ABI | 7/29 | `9a592c5` | 零传输闭环 | 0 H2D/0 D2H/0 sync（`0735481` nsys 证） | 仅 phase-cd |
| C4-a | 7/29 | `5f746cf` | 碰撞/CCD/LS 进整帧图 | +911 行 | 仅 phase-cd（默认关） |
| C5 | 7/29 | `f5308e3` | isolated 整帧图化 | +926 行；G19 零新增耦合 | 仅 phase-cd（默认关） |
| C6-e | 7/30 | `65da2e5` | foldshirt 60 帧全图内 | 98% 帧录入 | 仅 phase-cd |
| C6-i | 7/31 | `b58a0b7` | 溢出帧回 release solver 收尾 | ~1e-6 扰动源除 | 仅 phase-cd |
| C6-l | 7/31 | `454fb59` | 整帧图位级确定（det 栈） | towel 220 帧 ×3 全同 | 仅 phase-cd |
| C6-p | 8/1 | `47c37d6` | 默认 headroom 2→1 | forcegrip 4090 2.08→1.39×、A800 3.25→1.75× | 仅 phase-cd |
| C6-q | 8/1 | `575539f` | v0.8.5 标准复审 | 套件 22/22 | 仅 phase-cd |
| C6-v | 8/3 | `c74071a` | 真动态图 graph_resize + 原子长龙定罪 | scatter 22×/实例 = 零 pad convoy | 仅 phase-cd |
| C6-w | 8/3 | `b842d2f` | 跳零默认开、逐位门禁入 det 栈 | 1.96→1.46×；merged 自噪 1.1e-4 | 仅 phase-cd |
| C6-aa | 8/3 | `686a94d` | 复审 4 缺陷（16 槽环独占等） | — | 仅 phase-cd |
| 横测 | 8/3 | `b9b48c5` | vs v0.8.5：regime 定胜负 | A800 19.3→9.7→3.85 ms/步（2.0×/5.0×） | — |
| Isaac 表皮 | 8/4 | `5e7a887` | 一面两通道，step() 自动改道 | `gpu_rl_tensors()` 零拷贝；3.14≈3.18 ms/步 | 仅 phase-cd |
| BVH 战役 | 8/4–8/6 | `96b52fb`→`3be83d9`→`474d1dc`；`5054fee`(merge) | 全候选实测裁决，opt-in 归档 | merged-4env +both = +12.4% 更差；图+PLOC 8× 崩塌 | 仅 phase-cd（全 opt-in） |
| resizer 成本模型 | 8/5–8/6 | `ccf6735`→`0d2ad7e` | 迭代级 resizer 反收益，帧级保留 | ~6% 更差 | 仅 phase-cd |
| tier 阶梯裁决 | 8/7 | `531211b`→`a46852f` | A/A 校准否决；graph-host 差定格 | 噪底 7–9%；median +17.1% | 仅 phase-cd |
| alpha-tune | 8/7 | `0ee5387`/`5759acd` | CCD 帽旋钮 opt-in；CFL 三角色拆分 | ground 34.6%/CFL 32.8%/ACCD 32.4% | 仅 phase-cd（默认关） |
| 回滚泄漏修复 | 8/10 | **`4e49fda`** | 整图回滚补 ABD q 族 + kappa | ls_exhaust 2→0；soft 项爆 3e5..6e7× 的真因 | 仅 phase-cd |
| 成对审计 | 8/10 | **`4219f37`** | 同态 graph vs host Newton +1.0% | 869/1550 位级相等；1551 样本/趟 | 仅 phase-cd |
| 路线图（HEAD） | 8/11 | `b3ab747` | 相位剖面 + 优先队列 | PCG 57%、MAS apply 27%、检测 1.3% | 仅 phase-cd |
| 稳定 v0.8.5.3 | 8/11 | `b8e27a1`(tag) | contact-IO 修复（摩擦读数恒零 + reset API + teleport 表面刷新） | 滑动比 0.600±0.008；24 µm→6.7e-9 m | **仅稳定线** |
| 稳定 v0.8.5.4 | 8/12 | `0894958` / `c0339c8`(tag) | 真静摩擦默认开：`absolute_epsv=1e-4` + 持久摩擦锚（strict 多环境自动抑制 anchor） | flask_cap 保持段滑移 3.7→**0.00 mm**、cap 11–20°→0.7°；每步蠕滑 1.0e-5→5.4e-12 m；step +9%。**⚠ 改变所有含摩擦场景轨迹**（§2.5） | **仅稳定线** |

---

## 5. 金锚与门禁演化

"金锚"= strict 模式场景哈希，逐位复现的法定标识（run-to-run + 跨架构 sm_89 ≡ sm_80）。

| 项 | 值 | 生效窗口 | 出处 |
|---|---|---|---|
| strict 金锚（dlto 前） | `f7fb5a786c2d7935` | v0.8.5 世代 → v0.8.6 重构全程（字节等同拆分的证明物） | [`../RELEASE_NOTES_v0.8.6-rc1.md`](../RELEASE_NOTES_v0.8.6-rc1.md):17；稳定线 CHANGELOG.md（0.8.5.1 条目） |
| strict 金锚（dlto 后，工程线现行） | `0544461bd82123ae` | `45d74f0`/`40c9f11`（**rc2 tag 后首个变更**——dlto 默认开；checkout rc2 tag 仍是旧锚）起 | RELEASE_NOTES:132-137；[`../PHASE_C_FRAME_GRAPH_PLAN.md`](../PHASE_C_FRAME_GRAPH_PLAN.md):305（A800 跨架构 PASS） |
| 门禁段数演化 | 8 段（phase1 `verify_gates.sh`）→ 12 段（rc1）→ 15 段（rc2，+G13/G14/G15）→ 21/21（Isaac 表皮附录 A 时代）→ **22/22**（C6-w/BVH 战役至今） | 2026-07-25 → 今 | RELEASE_NOTES:27,116；SIM_EXEC:154；A800_ALLEX:337-339 |
| 门禁纪律 | 门禁必须自带构建（G0 build；G0.2 realpath 断言）；push 时执行（分支 = full，tag = heavy）；`SKIP_GATES=1` 逃生口入 bypass.log | `161e66e` 起 | [`../CI.md`](../CI.md) |
| STIFF_* 旋钮治理 | `StiffGIPC/config/knob_registry.h` 单一真源（G14 落地提交 `06a6710` 时点共 **84** 条：diag 42/solver 15/perf 11/mode_isolated 7/mode_strict 4/python 3/audit 2，`git show 06a6710:…knob_registry.h` 亲验；现 170 条）；未登记 WARN，`STIFF_KNOB_STRICT=1` 升 `ConfigurationError` | rc2 起 | `git show 06a6710:StiffGIPC/config/knob_registry.h`；knob_registry.h:23 起 |

**注意**：稳定线不承载以上门禁基建（无 `scripts/` 目录，亲验 `ls`）；其正确性承诺依赖"逐版本轨迹兼容声明"（CHANGELOG 条目）而非武装门禁。

---

## 6. 稳定线 v0.8.5.3 之后的提交明细与工作树状态

> **本节定位已更新**：这 4 条提交构成正式版本 **v0.8.5.4**，其版本条目见 **§2.5**；本节保留为**提交级明细 + 工作树状态记录**。
> 仍需注意的两件事：① **工作树状态已更正**：勘探时磁盘工作树被误留的 8 文件暂存回退按在 v0.8.5.3 内容上（owner 确认非本意），2026-09-08 已 `reset --hard HEAD` 恢复为 v0.8.5.4 内容（回退补丁留有备份）；本手册的稳定线**行号**仍一律按 v0.8.5.3 blob（复核用 `git show v0.8.5.3:<文件>`），v0.8.5.4 独有行号在 §2.5 里按 `stable@c0339c8` 标注、现与磁盘一致。② **wheel 资产已确认挂出**（§8 #1 / OPEN_POINTS OP-001 已关闭）：Release `v0.8.5.4` 2026-08-11 正式发布、cp311/cp312 双 wheel 在架——"版本存在"与"发布物存在"这次两件都成立，工作树也已跟上。

组成提交明细（`CHANGELOG.md:7-44` 的 0.8.5.4 条目只存在于 `git show HEAD:CHANGELOG.md`，磁盘副本因回退而无此条目）：

| 提交 | 日期 | 内容 | 关键数字 |
|---|---|---|---|
| `dc1a297` | 8/11 | 真静摩擦两件套：`absolute_epsv` 旋钮（m/s；旧编码 epsv = 1e-2×eff_scene_diag ≈ 19 mm/s，是 IPC 论文默认 1e-3·l 的 10×、静精度值 1e-5 m/s 的 1900×）+ 持久摩擦锚（friction anchor） | flask_cap 双臂抓持 400 步实测：flask 滑 3.7 mm、cap 转 11–20°@μ=3.5 锥内——stiction creep 根因 |
| `d7ab5bf` | 8/11 | `STIFF_NEWTON_TRACE` 逐迭代移动范数踪迹（诊断） | — |
| `0894958` | 8/12 | "release(v0.8.5.4)" 提交本体：默认开 `absolute_epsv=1e-4 m/s` + `friction_anchor=True` | **改变所有含摩擦场景轨迹**；`STIFF_EPSV=0`/`STIFF_FRIC_ANCHOR=0` 位级回 0.8.5.3；flask 滑 3.7→0.00 mm，step 成本 +9% |
| `c0339c8` | 8/12 | strict 多 env 默认压掉 friction_anchor——batch 不变性破坏（N=2 vs N=4 env0：17 帧位级相同后帧 18 一次 accept 翻转 → 96% 顶点单步分歧；bisect：epsv-only 绿、anchor-only 绿、组合红）；根修（N 不变能量归约）记为排入 0.8.6。tag `v0.8.5.4` 实际落在此提交 | — |

这些特性（`absolute_epsv`/`friction_anchor`/`STIFF_NEWTON_TRACE`）**不在两个磁盘工作树的任何一个里**（grep 双树零命中，亲验），也不在 phase-cd 历史里——即工程线的**第三个未移植项**，清单见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1.0。

---

## 7. 未推送/未发布状态说明

| 项 | 状态 | 依据（亲验） |
|---|---|---|
| 工程线分支 `codex/phase-cd` | **已于 2026-09-08 首次推送远端**（`origin/codex/phase-cd`，含手册）。此前的备份状况也非"全在本地"：其中 **62 条**早已随 `origin/internal/v0.8.6-rc1` 在远端 | `git ls-remote origin codex/phase-cd`；62 条：`comm -12 <(rev-list v0.8.5..origin/internal/v0.8.6-rc1) <(rev-list v0.8.5..HEAD)` |
| v0.8.6 | **未发布**——HEAD `b3ab747` 未打 tag；`v0.8.6-rc1-internal`/`rc2-internal` 是内部 tag，仅私仓；`pyproject.toml` 版本号 `0.8.6rc2` 是内部标识 | `git tag --merged HEAD`；pyproject.toml:7 |
| 稳定线最新版本 | **v0.8.5.4**（tag `c0339c8`，2026-08-12，`pyproject.toml:7` = `0.8.5.4`）——真静摩擦默认开，**改变所有含摩擦场景轨迹**（§2.5）。**已正式发布**：公开仓 `github.com/haoxiangNtu/stiff-physics` Release `v0.8.5.4`（2026-08-11 发布）挂 cp311/cp312 双 wheel，README 安装 URL 已指向它；v0.8.5.3 的 wheel 仍在架可回退（cp311/cp312，sm_80/89/120） | 稳定仓 tag `v0.8.5.4`/`v0.8.5.3`；`git show HEAD:pyproject.toml`；`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` |
| 两线 Python 包名 | 同为 `stiff-physics`/`stiff_physics`，**不能并存于同一环境**；探测：`hasattr(engine, "reset_transient_contact_state")` = 稳定线；`hasattr(engine, "prepare_gpu_rl")` = phase-cd；或 `importlib.metadata.version` = `0.8.5.3` vs `0.8.6rc2` | 两树 pyproject.toml / 绑定名字级 diff |
| 稳定线 → phase-cd 未移植项（**三项**） | ① 接触力摩擦读数修复（`snapshotFrictionForce`）② `reset_transient_contact_state` ③ **v0.8.5.4 真静摩擦**（`absolute_epsv` + `friction_anchor`，§2.5）——从稳定线迁到工程线的用户须知：`get_vertex_contact_forces` 的 `friction_lagged`/`total` 分量在 phase-cd 上仍恒零，且长时保持抓取会按 legacy 场景派生 epsv 蠕滑。清单与移植路径见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1.0 | grep `snapshotFrictionForce` / `absolute_epsv` / `friction_anchor` 在 phase-cd 零命中 |
| 工程线 CHANGELOG | phase-cd 树的 `CHANGELOG.md` 止于 0.8.5.1——0.8.5.2/0.8.5.3 条目只在稳定线树；v0.8.6 发布时需合并两树 CHANGELOG 并补 0.8.5.2+ 条目 | 两树 CHANGELOG grep |

---

## 8. 本文未决点（待核实清单）

1. ~~**v0.8.5.4 的 wheel 资产状态**~~ **已关闭（2026-09-08）**：公开仓 Release `v0.8.5.4` 已正式发布（`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` 亲验：`published: 2026-08-11T17:07:37Z`，`draft:false`/`prerelease:false`），`stiff_physics-0.8.5.4-cp311/cp312-linux_x86_64.whl` **两个 wheel 均已挂出**，公开仓 README 的安装 URL 也已由提交 `a38ede4`（"Bump install URLs to v0.8.5.4 (true static friction default-on)"）指向它——**未撤回**。README §3.1 的安装指令已改按 v0.8.5.4 给（OPEN_POINTS OP-001 已关闭）。版本本体此前即已亲验：tag `v0.8.5.4` 落在 `c0339c8`、`pyproject.toml` = `0.8.5.4`、CHANGELOG 有完整 [0.8.5.4] 条目（`git show HEAD:CHANGELOG.md:7-44`）。（磁盘工作树被回退到 v0.8.5.3 内容是另一回事，见 §6。）
2. **beaker +7% 破案的证据锚**：结论（纯启动段 ~0.3–0.6 s、每帧持平）见 [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md):123 与工作记忆，但 v0.8.5..HEAD 提交史中**没有对应的破案提交**；每帧持平的原始数据文件未定位到具体行。
3. **成对审计的分布细分**："B' 9342±230 vs 宿主 8808±60（15σ）"与"+6% 壳效应 / +4% 图化增量"的拆分出自 `4219f37` 提交正文与工作记忆合读；提交正文亲验部分只确认"+1.0% 同态差"与"869/1550 位级相等"。引用 ± 数字前应 `git show 4219f37` 核对完整 body。
4. **tag 打在 release 提交后一条的惯例**（§1.3 三例）：是仓库惯例还是三次巧合，未向 owner 确认。
5. **phase-cd 摩擦读数恒零的运行时确认**：代码结构证实 phase-cd 无 `snapshotFrictionForce`、post-step 现算（`engine_modules/03_step_getters_export.inl:1744-1747`），按稳定线 `1bc13ef` 的诊断逻辑推定恒零；但未在 phase-cd 上运行实测读数。
6. **BVH 战役分支内提交归属**：`codex/bvh-full-campaign` 分支的 ~20 条 "validation:" 提交经 merge `5054fee` 计入 209 条总量，但未逐条核对哪些在分支侧、哪些在主线侧。
7. **约 30 条纯 docs 提交的正文细节**（如 `79f88f4` episode 通道经济学的完整数值表）只核对了标题或摘要，引用其内部数字时需按 hash 回查。
8. **RL 微步 19.3/9.7/3.85 ms/步的平台标签在两份源文档间冲突**：`A800_ALLEXAMPLES_TIMING_2026-08-01.md:342` 把该表放在 "(4090, clean GPU)" 标题节内（口径 300 步），而 `SIMULATOR_EXECUTION_DESIGN.md:131-137` 的双列表把同数字标为 A800（口径 150 步，4090 列 = 4.25/3.72/3.18）。本文取 A800（SIM_EXEC 双列表 + 工作记忆），但 A800_ALLEX 该节标题是否笔误未向 owner 确认。

---

*本分册由文档工程流水线生成于 2026-09-07（2026-09-08 补入 v0.8.5.4 条目，§2.5）；事实基线：稳定线 `b8e27a1`（v0.8.5.3，行号基线）+ `c0339c8`（v0.8.5.4，最新版本，内容取自 `git show HEAD:`）、工程线 `b3ab747`（codex/phase-cd HEAD）。发现与仓库现状不符处，请以 `git log`/`git show` 与 §8 清单为准修订。*
