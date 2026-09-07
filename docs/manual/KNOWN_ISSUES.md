# StiffGIPC 手册 — 已知问题与潜在风险(KNOWN_ISSUES)

> 手册分册。姊妹分册:多环境三模式见 [API_EXECUTION.md](API_EXECUTION.md) §1 · 整帧图与 GPU 驻留 RL 见 [PRINCIPLES_EXECUTION.md](PRINCIPLES_EXECUTION.md) §5–6 · 接触 / CCD / 摩擦见 [PRINCIPLES_CONTACT.md](PRINCIPLES_CONTACT.md)。
> 性能路线图与死路表的权威出处:[../OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md);执行架构定稿:[../SIMULATOR_EXECUTION_DESIGN.md](../SIMULATOR_EXECUTION_DESIGN.md);门禁体系:[../CI.md](../CI.md)。

本册收录两条产品线上**已确认存在**的缺陷、语义陷阱、性能坑与运维风险。每条按固定格式给出:现象 / 根因 / 影响面 / 规避方法 / 修复状态,并标注适用线:

| 标签 | 含义 |
|---|---|
| 【稳定线+phase-cd】 | 两条线都存在 |
| 【仅 phase-cd】 | 只在工程线(`codex/phase-cd`,v0.8.6rc2)存在 |
| 【仅稳定线】 | 只在稳定线(`release/stable-0.8`,内容 = tag `v0.8.5.3`)存在 |
| 【实验性,默认关】 | 需显式旋钮开启的路径,默认行为不受影响 |

**版本勘定**(2026-09-07 亲验):

- 稳定线 = `/home/ps/Downloads/Stiff-GIPC-stable-08`,分支 `release/stable-0.8`。磁盘工作区内容与 tag `v0.8.5.3`(2026-08-11 发布)**逐字节一致**(`git diff v0.8.5.3 --stat` 为空);公开仓 `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel,CUDA 架构 sm_80/89/120。C++ 布局为重构前单体(`StiffGIPC/GIPC.cu` 16884 行)。**稳定线最新版本是 tag `v0.8.5.4`**(2026-08-12 = `c0339c8`,`pyproject.toml:7` = `0.8.5.4`;真静摩擦默认开,**改变所有含摩擦场景轨迹**,见 §1.5/§1.6 与 [CHANGELOG_TIMELINE.md](CHANGELOG_TIMELINE.md) §2.5)——但**磁盘工作区内容被回退到 v0.8.5.3**,故本册的稳定线行号一律是 v0.8.5.3 内容,v0.8.5.4 独有行号按 `git show HEAD:` 单独标注。
- 工程线 = `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph`,分支 `codex/phase-cd`(HEAD `b3ab747`,`pyproject.toml` 版本 `0.8.6rc2`),含 v0.8.6 模块化重构、整帧 CUDA Graph、GPU 驻留 RL、episode、checkpoint v2 等全部 v0.8.5 后工作。
- 两线分叉点 = `05c3f75`(git merge-base),**早于 v0.8.5.3**——稳定线 v0.8.5.3 的两个接触 I/O 修复提交(`1bc13ef`、`1d05c7a`)与 v0.8.5.4 全部提交都不在 phase-cd 历史里——**三个未移植项**,合并清单见 §1.0(详条 §1.1、§1.2、§1.6;分支拓扑见 §5)。
- 文中行号:未注明仓库的 `文件:行号` 指 phase-cd 树 `StiffGIPC/` 下路径;`stable ...` 前缀指稳定线树。

---

## 目录

- [1. 正确性类](#1-正确性类)
  - [1.0 未移植项清单(稳定线 → phase-cd,三项)](#10-未移植项清单稳定线--phase-cd三项)
  - [1.1 phase-cd:接触力读数中摩擦分量恒为零](#11-phase-cd接触力读数中摩擦分量恒为零)
  - [1.2 phase-cd:缺 `reset_transient_contact_state()`(就地回合重置的幽灵摩擦)](#12-phase-cd缺-reset_transient_contact_state就地回合重置的幽灵摩擦)
  - [1.3 EE 近平行 mollifier 全关(边-边接触硬切换)](#13-ee-近平行-mollifier-全关边-边接触硬切换)
  - [1.4 硬钉(USE_HARD_PIN)弹性链式法则缺失](#14-硬钉use_hard_pin弹性链式法则缺失)
  - [1.5 静摩擦蠕滑:场景派生 epsv 过大](#15-静摩擦蠕滑场景派生-epsv-过大)
  - [1.6 phase-cd:未移植 v0.8.5.4 真静摩擦(长时保持抓取蠕变)](#16-phase-cd未移植-v0854-真静摩擦长时保持抓取蠕变)
- [2. 语义陷阱类](#2-语义陷阱类)
  - [2.1 legacy 接触力 API 返回 −F·dt² 而非牛顿](#21-legacy-接触力-api-返回-fdt-而非牛顿)
  - [2.2 `Config(**kwargs)` 静默忽略未知键](#22-configkwargs-静默忽略未知键)
  - [2.3 `collision_detection_buff_scale` 双默认值(Python 6.0 / C++ 1.0)](#23-collision_detection_buff_scale-双默认值python-60--c-10)
  - [2.4 `newton_iter_cap` 耗尽照常接受(且无专用警告)](#24-newton_iter_cap-耗尽照常接受且无专用警告)
  - [2.5 line-search 预算耗尽:响亮 WARN 后接受非下降步](#25-line-search-预算耗尽响亮-warn-后接受非下降步)
  - [2.6 merged 模式非 run-to-run 确定](#26-merged-模式非-run-to-run-确定)
  - [2.7 `set_vertex_velocities_gpu` 不重建 xTilta](#27-set_vertex_velocities_gpu-不重建-xtilta)
  - [2.8 关节能量刻意不乘 dt²(改 dt 隐式改关节刚度)](#28-关节能量刻意不乘-dt改-dt-隐式改关节刚度)
  - [2.9 其他语义陷阱速查表](#29-其他语义陷阱速查表)
- [3. 性能类](#3-性能类)
  - [3.1 MAS 预条件应用占全仿真 27%](#31-mas-预条件应用占全仿真-27)
  - [3.2 整帧图在大帧 regime 是结构性负收益](#32-整帧图在大帧-regime-是结构性负收益)
  - [3.3 事务壳 +6% 轨迹效应(成对审计结论)](#33-事务壳-6-轨迹效应成对审计结论)
  - [3.4 cap=50 饥饿帧(CCD 钳制"长征帧")](#34-cap50-饥饿帧ccd-钳制长征帧)
  - [3.5 已证死路速查(勿再投入)](#35-已证死路速查勿再投入)
- [4. 工程 / 运维类](#4-工程--运维类)
  - [4.1 OVF 回退税与容量重录成本(A800 一次 20–30 s)](#41-ovf-回退税与容量重录成本a800-一次-2030-s)
  - [4.2 显存:容量档与配对缓冲只增不减](#42-显存容量档与配对缓冲只增不减)
  - [4.3 strict 模式的性能代价](#43-strict-模式的性能代价)
  - [4.4 CI 金锚:默认路径受逐位约束,改代码须过 22 段门禁](#44-ci-金锚默认路径受逐位约束改代码须过-22-段门禁)
  - [4.5 进程级全局状态:单进程单模式 / 单 Engine](#45-进程级全局状态单进程单模式--单-engine)
- [5. 上游依赖(KemengHuang/StiffGIPC)与分叉点](#5-上游依赖kemenghuangstiffgipc与分叉点)
- [6. 待核实清单](#6-待核实清单)

---

## 1. 正确性类

### 1.0 未移植项清单(稳定线 → phase-cd,三项)

**适用线:【仅 phase-cd】**(三项都是"稳定线已修、工程线未合入";两线分叉点 `05c3f75` 早于三者的全部提交)

本节把散落的三条**同源缺陷**并成一张表:它们的共同点是**修复只落在稳定线**,而 `codex/phase-cd`(HEAD `b3ab747`)的历史里连提交都没有。三者都与**摩擦**有关,升级/迁线的用户请整体评估,不要逐条踩。

| # | 未移植项 | 稳定线修复落点 | phase-cd 上的后果 | 详条 |
|---|---|---|---|---|
| 1 | **接触力摩擦读数恒零修复**(`snapshotFrictionForce`) | v0.8.5.3 `1bc13ef`(stable `GIPC.cu:16504`、调用点 `:16800`) | `get_vertex_contact_forces(components="friction_lagged"\|"total")` 的摩擦贡献**恒零**(求解本身不受影响,只是读不出来) | [§1.1](#11-phase-cd接触力读数中摩擦分量恒为零) |
| 2 | **`reset_transient_contact_state()`** 就地回合重置 API | v0.8.5.3 `1bc13ef`(stable `sim_engine.h:296-312`、`sim_engine.cu:3620-3636`;v0.8.5.4 在同函数追加 `clearFrictionAnchors()`,HEAD 内容 `:3622-3641`) | 无等价入口:绕过 teleport API 的自定义重置会带上一 episode 的陈旧配对(幻影摩擦),且没有 Kappa 归零入口(推定残留 ~0.9 µm 量级) | [§1.2](#12-phase-cd缺-reset_transient_contact_state就地回合重置的幽灵摩擦) |
| 3 | **v0.8.5.4 真静摩擦**(`absolute_epsv` + 持久摩擦锚 `friction_anchor`) | v0.8.5.4 `dc1a297`/`0894958`/`c0339c8`(2026-08-11~12;`0894958` 起两个旋钮**默认开**) | epsv 仍由场景尺度派生(1.9 m 场景 = 19 mm/s)且摩擦锚每步重置 → **长时保持抓取持续蠕变**(稳定线实测 6 s 滑 3.7 mm、瓶盖转 11–20°) | [§1.6](#16-phase-cd未移植-v0854-真静摩擦长时保持抓取蠕变) |

**共同规避**:需要以上任一能力的项目**用稳定线**(第 3 项要 v0.8.5.4;第 1/2 项 v0.8.5.3 起即可);必须留在 phase-cd 的,按各详条的"规避方法"降级使用,并在项目文档里写明缺陷。

**共同移植路径**(亲验,2026-09-08):同仓分支 **`port/friction-anchor-086`** 已把三项**全部**按 v0.8.6 模块化布局落位——`3ec2734` "port(1bc13ef): contact-IO fixes onto the 0.8.6 refactored line"(第 1/2 项)、`c735e13` "port(dc1a297/epsv): absolute_epsv knob onto the 0.8.6 line"、`57015da` "port(anchors): persistent friction anchors onto the 0.8.6 line (parameterized)"(第 3 项)。该分支共 5 条提交,均为 2026-08-12。其后 `9fd2905` 给出了 **strict 的正解**(稳定线 `c0339c8` 只是把 anchor 在 strict 下关掉,根修排入 0.8.6——见 §1.6):真因是 `_penv_energy_accum` 用裸 `atomicAdd(double)` 按**线程调度顺序**累加逐 env 能量,顺序是 batch 形状相关的 → env0 能量带 N 依赖的末位 ulp 抖动 → 在边界帧翻掉一次 S3 回溯决策 → alpha 发散(E1–E7 取证链:梯度、rz0、整个 PCG 环在不同 N 下逐位相同,翻转发生在 PCG 之后)。修法 = 改走 `binned_deposit`(Demmel-Nguyen 精确分箱,`g_det_reduce` 下顺序无关)+ 固定顺序 `_penv_bins_combine` 收尾,kernel 签名不变;**能量和 N 不变之后 strict 不再需要抑制 anchor**,`carryFrictionAnchors` 在所有模式下都听配置(`STIFF_FRIC_ANCHOR=0` 逃生阀保留)。该提交的净室门禁(实测二进制):跨 env moveDir 0、run-to-run vhash 逐位、**batch env0 N=2 vs N=4 逐位 0.000e+00(anchor 开着)**、merged 冒烟干净。`1040edb` 再补 checkpoint 序列化摩擦锚状态 + 逐 env LS 能量恒精确分箱 + 重钉 strict 门禁。

> ⚠ **合入前必须做的两件事**:① 该分支基于 `6b0e02e`(2026-07-28),**落后 phase-cd HEAD 147 个提交**且 `git merge-base --is-ancestor port/friction-anchor-086 codex/phase-cd` 返回 NOT ancestor(亲验)——须先前移(rebase/merge)到 `b3ab747` 并重跑 22 段门禁;② `3ec2734` 未触及 `frame_fsm/`——§1.1 移植要点第 5 条(整帧图/episode 路径的快照捕获)在该分支上**仍未解决**。
> 合入第 3 项等于**改变所有含摩擦场景的默认轨迹**(与稳定线 v0.8.5.4 同性质),必须换锚立项,不能当"补丁合并"处理。

### 1.1 phase-cd:接触力读数中摩擦分量恒为零

**适用线:【仅 phase-cd】**(稳定线 v0.8.5.3 已修复;修复未移植)

| 项 | 内容 |
|---|---|
| 现象 | `Engine.get_vertex_contact_forces(components="friction_lagged")` 与 `components="total"` 中的摩擦贡献在 `step()` 之后**恒为 0**;`components="normal"`(默认)不受影响。 |
| 根因 | IPC 滞后摩擦的梯度是**步内位移**的函数(当前位置相对摩擦集构建时刻位置的切向相对位移 `relDX`,见 `energy/02_contact_energy_device.inl:297-375`)。帧提交(`updateVelocities`,`core/ipc_solver.inl:2624`)之后步内位移归零,accessor 在提交后现算 `calFrictionGradient`(`engine_modules/03_step_getters_export.inl:1744-1746`,读只读 lastH 集)——数学上精确等于零。 |
| 影响面 | phase-cd 上所有依赖摩擦力读数的传感/评估代码:抓取滑动检测(`|Ft|/|Fn|` 滑比)、斜面静置合力校验(文档示例 "total ≈ 0" 在 phase-cd 上**不成立**——total 退化为 normal)、RL 观测里的摩擦通道。求解本身**不受影响**(摩擦在求解器内正常施加,只是读数为零)。 |
| 规避方法 | ① 需要摩擦读数的项目使用稳定线 v0.8.5.3(该修复经双关节剪切台验证:gel μ=1.0 / bar μ=0.6,三个压深实测滑比 0.600 ± 0.008,stable `CHANGELOG.md:7-41`)。② phase-cd 上暂用 `components="normal"` + 法向力 × μ 上界做保守判据,并在文档/代码注释中标注读数缺陷。 |
| 修复状态 | 稳定线已修(v0.8.5.3 提交 `1bc13ef`);`codex/phase-cd`(HEAD `b3ab747`)未合入——**未移植项清单第 1 项,见 §1.0**。**移植成品已存在**:同仓分支 `port/friction-anchor-086` 首提交 `3ec2734` "port(1bc13ef): contact-IO fixes onto the 0.8.6 refactored line" 已把摩擦快照 + `reset_transient_contact_state`(§1.2)按模块化布局落位(GIPC.cuh 成员、`09_friction_sets_host_mem.inl` 快照实现、`ipc_solver.inl` 提交前钩子、`engine_modules/03` accessor、bindings、engine.py;提交注明 `1d05c7a` 无需移植——本线已有等价的 [rl-reset] `x=J·q` 整块重推),同分支后续 `c735e13`/`57015da` 还移植了 v0.8.5.4 的 `absolute_epsv`/friction anchors(§1.6)。**注意**(细节见 §1.0):该分支基于 `6b0e02e`(2026-07-28,落后 phase-cd HEAD 147 个提交),合入前须前移(rebase/merge)到 `b3ab747` 并重跑 22 段门禁;且 `3ec2734` 未触及 `frame_fsm/`——下述移植坑 5(整帧图路径)在该分支上仍未解决。 |

**移植要点**(下述 1–4 步在 `port/friction-anchor-086@3ec2734` 已实现,列此供前移时 review 对照;第 5 步在该分支上也**尚未解决**):

1. 成员:向 `GIPC.cuh` 加 `m_d_fric_force_snap`(double3*)、`m_fric_snap_cap`、`m_have_fric_snap`(参照 stable `GIPC.cuh:332-341` 注释块)。
2. 快照函数:移植 `GIPC::snapshotFrictionForce(device_TetraData&)`(stable `GIPC.cu:16504-16527`)——`zeroBinnedGrad → calFrictionGradient → combineBinnedGrad` 写入持久设备缓冲,与 accessor 的分箱梯度协议逐一镜像;落位建议 `gipc_modules/12_host_wrappers_fem.inl`。
3. 调用点:宿主路径在 `core/ipc_solver.inl:2624` 的 `updateVelocities(TetMesh)` **之前**调用(stable 调用点 `GIPC.cu:16800`,置于 `#ifdef USE_FRICTION` 内)。
4. accessor:`engine_modules/03_step_getters_export.inl:1690-1759` 的 `want_friction` 分支改为拷贝快照(参照 stable `sim_engine.cu:3454`、`:3491`),不再现算。
5. **phase-cd 特有的移植坑**:整帧图路径的提交发生在图内(`frame_fsm/frame_transaction.cu:1329` 全帧图、`:1797` episode 迭代)——快照 kernel 必须一并录进捕获序列(注意零 H2D/D2H 审计,快照是纯设备工作可以过审),或者第一版先声明"摩擦快照仅宿主 step 路径有效、图/episode 路径返回上一宿主帧快照"并写进 docstring。`reset_transient_contact_state` 移植(§1.2)时须同步清 `m_have_fric_snap`(stable `sim_engine.cu:3630`)。

### 1.2 phase-cd:缺 `reset_transient_contact_state()`(就地回合重置的幽灵摩擦)

**适用线:【仅 phase-cd】缺此 API;对应缺陷在稳定线 v0.8.5.3 以新 API 修复**

| 项 | 内容 |
|---|---|
| 现象 | 就地(in-place)teleport 回合重置后,首次 solve 使用**上一个 episode 遗留的接触对镜像**构建滞后摩擦集:稳定线修复前实测 1116 个陈旧配对 → 幻影摩擦力 → 重置后轨迹出现 **24 µm 状态分歧**;清镜像后仍余 **0.9 µm** 残差,归因于自适应 Kappa 携带上一回合的接触历史(stable `CHANGELOG.md:7-41`、stable `sim_engine.cu:3620-3636` 注释)。 |
| 根因 | 每帧第一个摩擦集构建读的是宿主接触对镜像(`h_cpNum/h_cpNum_last/h_gpNum/h_gpNum_last`),teleport 单点写位姿不会使这些镜像失效;标量 `Kappa` 同为跨帧携带态。 |
| 影响面 | RL 就地重置工作流。**注意区分**:phase-cd 的官方重置通道**没有此缺陷**——`teleport_fem_vertices` / `teleport_abd_bodies` 在尾部强制 `invalidateRefitTopology + buildBVH() + buildCP()` 重建帧入口配对集,并对被触及 env 调 `reviveEnv`(`engine_modules/03_step_getters_export.inl:2484-2585`、`04_teleport_checkpoint.inl:95-123`);GPU 驻留 RL 的 `launch_gpu_rl_reset_async` 回放 prepare 快照。**真正暴露的**是绕过 teleport API 的自定义重置(如手写 `set_vertex_positions_gpu` + `set_vertex_velocities_gpu` 拼重置),以及"想只清接触/摩擦历史、不动几何"的场景——phase-cd 没有等价入口。另注意语义差:phase-cd 的重建是"当前几何的**新**配对集",稳定线 API 是"清空镜像、下一 solve 从零建",两者近似但不逐位等价;phase-cd 也没有稳定线那样的 Kappa 归零入口(stable `sim_engine.cu:3635`:`Kappa=0` 使下一 solve 走 `suggestKappa` 全新派生)。 |
| 规避方法 | phase-cd 上重置一律走 `teleport_fem_vertices` / `teleport_abd_bodies` / `launch_gpu_rl_reset_async`,不要拼裸 setter;需要 fresh-process 等价的 Kappa 时用 `save_checkpoint`/`load_checkpoint`(v2 checkpoint 含 Kappa 族)或重建 Engine。 |
| 修复状态 | 稳定线 v0.8.5.3 新增 `SimEngine::reset_transient_contact_state()`(stable `sim_engine.h:296-312`、`sim_engine.cu:3620-3636`;实测重置分歧 24 µm → 6.7e-9 m;**刻意不并入 teleport API**——teleport 单个 body 不应清掉其它接触的摩擦状态)。phase-cd 主线未合入;移植量很小——清 5+5 个 `h_cpNum[i]`/`h_cpNum_last[i]` 配对镜像 **加 `h_gpNum`/`h_gpNum_last` 两个地面配对镜像**(漏掉这两个会把幽灵摩擦换成地面通道复现)+ 摩擦快照旗标 `m_have_fric_snap=false` + `Kappa=0.0`(stable `sim_engine.cu:3620-3636`)。该 API 连同 §1.1 已在分支 `port/friction-anchor-086`(`3ec2734`,基 `6b0e02e`,须前移到 HEAD)按模块化布局移植完成;合入时接入 `scripts/rl_reset_gate.py`(G15)验收。**未移植项清单第 2 项,见 §1.0**;若一并合入 v0.8.5.4 真静摩擦(§1.6),此处还须补清持久摩擦锚(稳定线 v0.8.5.4 在同一函数里加了 `clearFrictionAnchors()`,stable `sim_engine.cu:3640`)。 |

### 1.3 EE 近平行 mollifier 全关(边-边接触硬切换)

**适用线:【稳定线+phase-cd】**(上游决定,非回归)

| 项 | 内容 |
|---|---|
| 现象 | 近平行 edge-edge 接触不做 IPC 论文式的 mollified(平滑化)势垒过渡:势垒在 EE↔退化(PP/PE)分类切换处是**硬切换**。折叠布料(folded)场景实测产生约 44 万次 mollify 请求、0 次执行(数字出自战役记录,本次代码勘探未复测——见 §6)。 |
| 根因 | 发射端 `mlbvh_modules/03_pair_emission.inl:171` 硬编码 `bool smooth = false;`,所有 `if(smooth)` 的 4 点 mollified 配对发射分支(如 `:191-198`、`:501-505`)永不执行。近平行分类本身仍在计算:`eeSqureNCross = ‖(v1−v0)×(v3−v2)‖²` 与阈值 `eps_x = 1e-3·‖v0−v1‖²_rest·‖v2−v3‖²_rest`(`mlbvh_modules/02_distances_dtypes.inl:51-58`),`add_e = (eeSqureNCross < eps_x) ? −obj_idx−2 : −1` 只被存进 int4 的 `.w` 而能量端不消费(能量端 PP/PE 分支只看符号,`energy/02_contact_energy_device.inl:96,167`);能量端 mollified 解码分支保留为冻结死代码("frozen smooth branches travel VERBATIM, still dead, still frozen",`energy/02:6-7`)。 |
| 影响面 | 理论上近平行 EE 接触的势垒梯度在分类切换处不 C¹ 连续,可能给 Newton 增加迭代;实践中两线全部验收场景(含折叠布料)在此配置下通过,无已知由此引发的失稳案例。 |
| 规避方法 | 无需规避(此为两线的基准行为,亦是全部金锚/门禁的锚定行为)。诊断旋钮 `STIFF_EE_NOMOLLIFY=1` 只把请求计数也关掉(`mlbvh_modules/00_gates_globals.inl:95-98`),用于跨 env 对称性诊断,不改变物理。 |
| 修复状态 | **不修**——上游 KemengHuang 的决定,两线跟随(上游同步压力使 smooth/mollifier 属冻结项,`../V086_REFACTOR_PLAN.md:27-38`)。任何复活 smooth 分支的尝试等价于换锚战役,须单独立项验证。 |

### 1.4 硬钉(USE_HARD_PIN)弹性链式法则缺失

**适用线:【稳定线+phase-cd】**(2026-09-07 复核:稳定线单体 GIPC.cu 含**逐字相同**的 KNOWN LIMITATION 注释块与活代码路径,stable `GIPC.cu:13894-13906`;且经公开 API 可达——`add_fem_pin_to_abd` / `add_fem_pins_with_local_pos` 在 stable `bindings/pystiffgipc.cu:250-292`、`stiff_physics/engine.py:699-722` 均暴露,`n_fem_pins>0` 即走该路径,stable 示例 `case_27_mobile_s1_softgripper_cup.py` 同样经 `USE_HARD_PIN` 环境变量启用)

| 项 | 内容 |
|---|---|
| 现象 | FEM 顶点以硬钉(substitution method)固定到 ABD 刚体、且刚体被关节**动态驱动**时,Newton 不收敛——撞 `newton_iter_cap=1000` 上限(注释点名经 `m5_drive_joint2_test` 验证,k=1000)。静态夹爪闭合场景正常(Newton k=1–2)。 |
| 根因 | 只有惯性项 `mass·I` 被链式法则到 ABD 的 q-DOF;FEM 弹性 Hessian 的交叉块 `H_fp·J_p`(自由行 × ABD 列)及其转置被 BoundaryType 清零逻辑丢弃,未做链式法则 → PCG 中 FEM 与 ABD 解耦:ABD 移动 q 时忽略自由 FEM 顶点的弹性回拉(`gipc_modules/13_kappa_partition_gradhess.inl:1644-1655`,注释原文 "KNOWN LIMITATION" / "TODO M3.5")。 |
| 影响面 | **两线**的 `USE_HARD_PIN` + 动态关节运动场景。另一红旗:硬钉扩展装配容量不足时打印 `[M3.5] WARN ext_count=%d > capacity=%d (truncated; expect Newton instability)`(`13_kappa:1535-1540`;稳定线同款 stable `GIPC.cu:13787`)——截断即丢 Hessian 贡献,应视为硬钉容量不足的硬信号。 |
| 规避方法 | 动态关节驱动场景设 `USE_HARD_PIN=0`,改用 stitch spring 软钉(两线的默认耦合手段)。 |
| 修复状态 | 有编号待办(TODO M3.5"full elasticity chain-rule"),未排期。 |

### 1.5 静摩擦蠕滑:场景派生 epsv 过大

**适用线:【稳定线+phase-cd】**(两条线的**磁盘内容**都是场景派生 epsv;**修复在稳定线 v0.8.5.4**——该版本的代码不在磁盘工作树里,只在稳定仓 HEAD,见 §6#2。phase-cd 侧的未移植后果单列 §1.6)

| 项 | 内容 |
|---|---|
| 现象 | 持握/静置物体在摩擦"静止区"内缓慢蠕滑,量级 ~0.5 mm/s(稳定仓 v0.8.5.4 提交说明中的实测;flask_cap 持握-滑移案例 3.7 mm 漂移)。 |
| 根因 | IPC 摩擦平滑化阈值(epsv·h)由场景尺度派生:能量端一步内"静止"切向位移阈值 `eps = √fDhat · dt`,而 `fDhat = 1e-4 · eff_bboxDiagSize²`(`gipc_modules/09_friction_sets_host_mem.inl:613`、`energy/16_friction.inl:830,852`),即 **epsv ≈ 1e-2 × 有效场景对角线 [m/s]**。IPC 论文对静摩擦精度的建议量级是 1e-5;场景越大阈值越松,静止区内二次能量对切向滑移的抵抗越弱。 |
| 影响面 | 所有依赖长时间静摩擦持握精度的场景(抓取保持、堆叠)。 |
| 规避方法 | ① 减小 `absolute_dhat`/场景有效对角线可等比例收紧 eps(副作用是接触整体变硬,须回归验证)。② 缩短评估窗口,或在控制层做位置伺服补偿。 |
| 修复状态 | **稳定线 v0.8.5.4 已修复并默认开**(提交 `dc1a297`/`0894958`):`absolute_epsv` 旋钮(Python `Config` 默认 1e-4 m/s)+ 持久摩擦锚 `friction_anchor`(默认 True,真静摩擦:flask_cap 保持段滑移 3.7 mm→0.00 mm,step 成本 +9%;后续 `c0339c8` 在 strict 多环境默认抑制 anchor)。详见 [CHANGELOG_TIMELINE.md](CHANGELOG_TIMELINE.md) §2.5。**但该修复不在 v0.8.5.3 磁盘内容、也不在 phase-cd**(两树 grep `absolute_epsv`/`friction_anchor` 均 0 命中)——工程线侧的影响与规避见 §1.6;v0.8.5.4 已正式发布并挂出 cp311/cp312 双 wheel(§6#2)。 |

### 1.6 phase-cd:未移植 v0.8.5.4 真静摩擦(长时保持抓取蠕变)

**适用线:【仅 phase-cd】**(§1.5 的缺陷本体两线共有;**修复只在稳定线 v0.8.5.4**,工程线未移植——§1.0 清单第 3 项)

| 项 | 内容 |
|---|---|
| 现象 | 工程线上**长时保持抓取会持续蠕变**:物体在摩擦锥内、受力远未达滑动条件,却以恒定速率滑移/转动,且**不收敛**——越保持越偏。稳定线修复前的同源实测(flask_cap 双臂 finray 抓取-提升-保持 400 步):烧瓶 6 s 内滑 **3.7 mm**,瓶盖锥体转 **11–20°**,而受力**远在 μ=3.5 摩擦锥内**;逐步分辨率下每步漂移 **1.0e-5 m**(×50 fps = 0.5 mm/s,与观测滑移吻合)。phase-cd 代码路径同构,故同样成立(未在工程线上复测,见 §6#6)。 |
| 根因 | 两条,**都还在 phase-cd 里**:(a) **静摩擦阈值 epsv 由场景尺度派生**——`fDhat = 1e-4 · eff_bboxDiagSize²` ⇒ epsv = √fDhat = **1e-2 × 有效场景对角线**(`gipc_modules/09_friction_sets_host_mem.inl:613`、`energy/16_friction.inl:830,852`),1.9 m 场景即 **19 mm/s**,是 IPC 论文默认 `1e-3·l` 的 10×、静摩擦精度值 1e-5 m/s 的 1900×;蠕滑速度 `creep_v ≈ (load/(μ·λ))·epsv`,于是**场景 bbox 泄漏进了静摩擦精度**(继 dhat/kappa 之后 bbox 派生参数族的第三个成员)。(b) **摩擦位移锚点每步重置**——lagged 摩擦只度量"相对本步起始位置"的切向滑移,每步归零,静接触因此永远停在"刚开始滑"的状态,无法形成真正的静摩擦弹簧。稳定线的解法是 `u_total = relDX_step + e`(跨步累计切向弹性偏移 `e`,`‖e‖` 在 `eps = epsv·h` 处径向回拉截断 = Coulomb 滑动),phase-cd 无对应实现(grep `friction_anchor`/`absolute_epsv` 零命中)。 |
| 影响面 | 工程线上**所有长时保持类任务**:抓取-提升-保持(RL 操作回放、装配、递交)、堆叠静置、任何以"保持 N 秒后位姿"为成功判据的评测。**RL 训练尤其危险**——蠕变是**系统性偏置**而非噪声,策略会学到补偿它;换到稳定线 v0.8.5.4 上评估时行为不迁移。求解稳定性不受影响(不是发散,是**物理上错的"稳定"**)。另注意与 §1.1 的叠加效应:phase-cd 上既蠕变、又读不出摩擦力,滑移检测两头落空。 |
| 规避方法 | ① **需要真静摩擦就用稳定线 v0.8.5.4**(默认即开;逃生阀 `absolute_epsv=0` / `friction_anchor=False`,或 `STIFF_EPSV=0` / `STIFF_FRIC_ANCHOR=0`)。② 必须留在 phase-cd 时的**部分缓解**:缩小场景有效 bbox(等比例收紧 epsv,副作用是接触整体变硬,须回归验证)、缩短保持窗口、在控制层做位置伺服补偿——**都补不上锚点每步重置这条**,只能减速不能归零。③ 手工移植:见 §1.0 的 `port/friction-anchor-086` 路径(该分支连 strict 的正解一并给了,`9fd2905` 逐 env 分箱能量 → anchor 在 strict 下也能默认开)。④ 评测纪律:凡是拿 phase-cd 数据做保持类结论,须标注"legacy 场景派生 epsv + 每步锚重置"口径。 |
| 修复状态 | **稳定线已修(v0.8.5.4,默认开)**;`codex/phase-cd`(HEAD `b3ab747`)**未合入,提交都不在历史里**(分叉点 `05c3f75` 早于 `dc1a297`)。移植成品在 `port/friction-anchor-086`(`c735e13` epsv + `57015da` anchors + `9fd2905` strict 正解 + `1040edb` checkpoint 序列化),**须先前移到 `b3ab747` 并重跑 22 段门禁**(§1.0)。合入即**改变所有含摩擦场景的默认轨迹**——等于换锚立项,不是补丁合并。 |

---

## 2. 语义陷阱类

### 2.1 legacy 接触力 API 返回 −F·dt² 而非牛顿

**适用线:【稳定线+phase-cd】**(两树 Python Config/Engine 该区段一致)

| 项 | 内容 |
|---|---|
| 现象 | `get_body_contact_force(vertex_offset, vertex_count)` 返回的不是牛顿,而是增量势梯度和 `Σ ∂E/∂x`,单位关系为 **`gradient = −F · dt²`**;且只含 body-body 势垒项(无地面、无摩擦)。`get_pair_contact_force(...)` 同为 raw IP scaling。 |
| 根因 | 兼容 0.8.4 之前的调用方而保留的 legacy 单位(`stiff_physics/engine.py:1590-1605` docstring 明文 ".. warning:: … `-force x dt^2`, NOT Newtons")。物理力换算:**F = −gradient / dt²**(与 0.8.4.1 的符号修复约定一致:静置立方体净竖向力 = +mg 向上支持力,`CHANGELOG.md:211-215`;换算注释亦见 `engine_modules/03_step_getters_export.inl:1753-1759`)。 |
| 影响面 | 直接把 `get_body_contact_force` 读数当牛顿用的代码:dt=0.01 时数值偏小 1e4 倍且**符号相反**。 |
| 规避方法 | 新代码一律用 `get_vertex_contact_forces()`(牛顿,已含 −1/dt² 与符号约定,可选地面项)+ `get_load_records()` 顶点区间聚合;或 `get_contacts()`/`get_contacts_device()`(逐接触,牛顿)。legacy API 只做旧脚本兼容。 |
| 修复状态 | 保持现状(改单位会破坏兼容);docstring 已加醒目警告。 |

### 2.2 `Config(**kwargs)` 静默忽略未知键

**适用线:【稳定线+phase-cd】**(两树 `class Config` 区段逐字节相同)

| 项 | 内容 |
|---|---|
| 现象 | `Config(newton_toll=1e-3)`(拼写错误)或传入当前构建不存在的参数,**不报错、不告警、静默丢弃**。 |
| 根因 | 透传实现:`for k, v in kwargs.items(): if hasattr(self._cfg, k): setattr(self._cfg, k, v)`(`stiff_physics/engine.py:442-444`)——native `_C.Config` 没有的属性直接跳过。此外多个具名参数也用 `hasattr` 守卫(如 `newton_velocity_tol`、`absolute_dhat`、`max_*_step_per_frame`,`engine.py:404-419`),在旧 wheel 上同样静默失效。 |
| 影响面 | 全部 Python 配置面。典型事故形态:调参实验里 typo 的旋钮"看起来生效了"(实际跑的是默认值)。 |
| 规避方法 | ① 构造后回读断言:`assert cfg.native.newton_tol == expected`。② 优先用具名参数而非 kwargs。③ 注意与环境变量的差别:phase-cd 对 `STIFF_*` 环境变量有 typo 治理(未注册的 `STIFF_*` 在 finalize 时 stderr WARN,`STIFF_KNOB_STRICT=1` 升级为 `ConfigurationError`,`config/knob_registry.h:198-239`)——**但 Config kwargs 不在该治理范围内**;稳定线连环境变量治理也没有。 |
| 修复状态 | 未修;有意保留的宽松透传(跨版本脚本兼容)。 |

### 2.3 `collision_detection_buff_scale` 双默认值(Python 6.0 / C++ 1.0)

**适用线:【稳定线+phase-cd】**

| 名称 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `stiff_physics.Config(collision_detection_buff_scale=…)` | **6.0** | 倍率 | 初始 DCD 配对缓冲容量乘数(`stiff_physics/engine.py:380`) |
| `SimEngineConfig::collision_detection_buff_scale`(C++) | **1.0** | 倍率 | 同一字段的 native 默认(`sim_engine.h:53`) |

| 项 | 内容 |
|---|---|
| 现象 | 同名参数两个默认值:经 Python `Config` 走的是 6.0;直接构造 `pystiffgipc.Config()`(或 C++ 嵌入)拿到 1.0。 |
| 根因 | Python 包装层在构造时无条件覆写 native 值(`engine.py:431`);容量语义 `Minimum = 100000 × buffScale` 进入 `GIPC::init` 的 triplet/配对缓冲初始容量(`gipc_modules/09_friction_sets_host_mem.inl:629-689`)。 |
| 影响面 | 绕过 Python `Config` 的集成方:初始容量小 6 倍 → 启动期更频繁触发 grow-redo(`[DCD-grow]`/`[CCD-grow]` 自愈重检测,见 §4.2)。**正确性不受影响**(溢出自愈,不丢配对),纯启动性能与日志噪声差异。反向陷阱:读 C++ 头以为默认 1.0 去做容量估算,实际 Python 用户全在 6.0。 |
| 规避方法 | 文档/复现脚本注明取值路径;嵌入 C++ 的用户显式设 6.0 对齐 Python 行为。 |
| 修复状态 | 未统一;docstring(`engine.py:377-379`)已说明"溢出自愈,该参数只权衡启动期 grow-redo 次数 vs 显存"。 |

### 2.4 `newton_iter_cap` 耗尽照常接受(且无专用警告)

**适用线:【稳定线+phase-cd】**

| 名称 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `newton_iter_cap` | 1000 | 迭代 | 全局 Newton 迭代上限(`sim_engine.h:87`、`GIPC.cuh:756`) |
| `env_newton_iter_cap` | 0(关) | 迭代 | per-env 迭代预算,到点冻结该 env 为 status 2 TIMEOUT(`sim_engine.h:76`) |

| 项 | 内容 |
|---|---|
| 现象 | Newton 循环 `for(; k < iterCap; ++k)`(`core/ipc_solver.inl:1356`)跑满上限即退出,**未收敛状态照常提交为该帧结果**——不抛错、不设失败位,而且(与 line-search 耗尽不同)**没有专用 WARN 打印**;每帧例行的 `Kappa: … iteration k: …` 行**默认就打印**(守卫是 `g_gipc_log_level >= 1` 且默认值即 1,`core/ipc_solver.inl:2477-2478`、`gipc_modules/00_prelude_common.inl:151`;该值唯一写点是 `Engine.set_log_level(n)` API——**不存在** `STIFF_LOG_LEVEL` 环境变量,全树无 getenv、knob_registry 无该条目)。 |
| 根因 | 帧预算语义:cap 是墙钟保护,不是收敛保证。 |
| 影响面 | 撞 cap 的帧其残差未达阈值——后续帧可能继承欠收敛状态(接触漂移、能量注入)。已知会系统性撞 cap 的构型:硬钉+动态关节(§1.4);半配置的 per-env bundle(`STIFF_DECOUPLE_THRESH` 无 `STIFF_PERENV_ALPHA` → 冻结判据永假,每帧跑满,`engine.py:531-541` 警告)。 |
| 规避方法 | 监控每帧迭代数:phase-cd 用 `get_frame_status().newton_iters`;两线可用 `get_per_env_newton_iters()`(须 host per-env 路径)。运营口径(v0.8.5 验收标准,`CHANGELOG.md:64-66`):**峰值 Newton ≤28 正常;>100 持续出现 = 场景不稳定信号**,应排查而非提 cap。多环境生产配置推荐 `per_env_exit=True, env_newton_iter_cap=100`(`CHANGELOG.md:94-98`)使坏 env 被冻结而不拖垮 batch。 |
| 修复状态 | 行为按设计;无计划加失败语义。 |

### 2.5 line-search 预算耗尽:响亮 WARN 后接受非下降步

**适用线:【稳定线+phase-cd】**(WARN+接受两线一致;typed throw / starved-LS 重试仅 phase-cd)

| 名称 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `line_search_max_iter` | 64 | 次二分 | 能量回溯与相交回溯共用预算(`sim_engine.h:80`、`GIPC.cuh:494`;stable `sim_engine.h:79`) |

机制:回溯 `while(energy_decision==1 && numOfLineSearch < budget)` 每次 α/=2(`core/ipc_solver.inl:694-738`);预算尽且能量未降 → stderr:

```
[line-search][WARN] budget exhausted (%d halvings, alpha=...): energy did NOT decrease
(E=... > E0=...). Step accepted anyway -- POTENTIAL SOLVER ERROR: expect contact drift /
collapsed barrier distances / iteration blow-up in later frames.
Raise Config.line_search_max_iter, reduce dt, or soften the drive.
```

(`ipc_solver.inl:732-738`;稳定线同款 stable `GIPC.cu:15221`。)

| 项 | 内容 |
|---|---|
| 现象/策略 | **WARN + 接受最终候选步**(loud but non-fatal)——由调用方决定是否中止(`:689-693` 注释)。后果如警告文本所述:接触漂移、势垒距离塌陷、后续帧迭代爆炸。 |
| 例外(仅 phase-cd) | ① 帧 0 + 非有限增量势 → 抛 `gipc::GeometryError`("initial configuration is infeasible for IPC — bodies interpenetrating at spawn",`ipc_solver.inl:837-846`;稳定线无类型化异常面)。② 整帧图内耗尽同样非致命,记 `INV_LS_BUDGET` 位接受(`gipc_modules/14_energy_linesearch_solver.inl:201-214`);**但**耗尽且同时挂容量 OVF 位 = "被截断配对集饿死的搜索"而非难帧,转 `FRAME_RETRY_REQUIRED` 在更大容量档重试(`14_energy:177-199`,towel 实证 inv=0x10040)。③ LS 后仍相交的二级回溯共用同一预算,耗尽**抛** `std::runtime_error("mesh intersection persists after energy line-search backtracking…")`(`ipc_solver.inl:853-862`)。 |
| RL 契约 | merged 模式"WARN 并继续"是**成文契约**;RL 循环应跨 step 差分健康计数 `get_ls_exhausted_count()` / `get_ls_nonfinite_count()`(经 `engine.native.…`,`sim_engine.h:728-737`,计数累加点 `ipc_solver.inl:828-830`)来发现并丢弃中毒 episode。 |
| 规避方法 | 按警告文本三选:提 `line_search_max_iter`、减 `dt`、软化驱动(降 `*_driving_strength_ratio` 或 `max_*_step_per_frame`);帧 0 抛错则修 spawn 位形(IPC 要求初始严格无穿透)。诊断:`STIFF_LSX_DIAG=1` 打印耗尽帧逐 slot 能量对照与 STATE-DESYNC/BAKED-ARGS 仲裁(`ipc_solver.inl:742-825`)。 |
| 修复状态 | 按契约保持;NaN 防漏已修(非有限试探能量与任何比较皆 false,旧逻辑会落进"接受下降"分支;现 `!isfinite(e1) ⇒ 继续回溯`,`14_energy:51-57`,宿主对应 `ipc_solver.inl:518`)。 |

### 2.6 merged 模式非 run-to-run 确定

**适用线:【稳定线+phase-cd】**(模式契约一致)

| 项 | 内容 |
|---|---|
| 现象 | 默认(merged)模式同一场景两次运行轨迹**不逐位相同**,且分歧随混沌接触指数放大:towel release 构建自身对照,帧 2 分歧 2.2e-14 → 帧 119 达 1.1e-4(实测,`../A800_ALLEXAMPLES_TIMING_2026-08-01.md` C6-w 节)。宏观表现为每帧 Newton 数有 ± 散布、混沌场景(布料揉皱)终态形貌可不同。 |
| 根因 | 原子发射与归约顺序不定(接触对槽位由 `atomicAdd` 分配、浮点求和序随调度漂移)——merged 契约明文"无可复现承诺"(`multienv/mode_contract.h:22`)。 |
| 影响面 | 以 merged 轨迹做逐位回归、以单次运行裁决 <10pp 性能效应、用固定随机种子期望可复现 RL rollout——全部无效。 |
| 规避方法 | ① 需要逐位可复现用 **strict** 模式(run-to-run + 跨架构 sm_80≡sm_89,金锚见 §4.4;代价见 §4.3)。② merged 上做测量遵守法定协议:混沌场景交错多轮取中位、全轨迹 Newton 总数为主判据(±1%)、凡裁决必带 A/A 对照(`../SIMULATOR_EXECUTION_DESIGN.md` 附录 C/D)。③ 回归容差按噪声包络而非零(门禁 default 层即如此:budget = max(4×基线自噪声, 1e-11×scale),C6-w)。 |
| 修复状态 | 按设计;确定性是 strict 的付费特性,不是 merged 的缺陷修复目标。 |

### 2.7 `set_vertex_velocities_gpu` 不重建 xTilta

**适用线:【稳定线+phase-cd】**(方法集合一致)

| 项 | 内容 |
|---|---|
| 现象 | 只调 `set_vertex_velocities_gpu(v)` 后 `step()`,物体**并不会**按新速度运动。 |
| 根因 | 该 API 只写速度缓冲,不重建惯性预测子 `xTilta`(`x̃ = x_prev + dt·v + dt²·a`)——增量势的运动学目标仍是旧的(docstring 明文警告,`stiff_physics/engine.py:1451-1462`)。同族陷阱:`set_vertex_positions_gpu` 只写当前位置,下一步会经旧 xTilta"弹回"(`engine.py:1464-1469` 注释)。 |
| 影响面 | 手写状态注入/重置代码。 |
| 规避方法 | 设完整运动学状态用 `teleport_fem_vertices(positions, velocities=…)`——一致地写 `_vertexes` / `o_vertexes` / `xTilta`(phase-cd 还重建配对集,见 §1.2)。裸 setter 仅用于知道自己在做什么的诊断场景。 |
| 修复状态 | 按设计(低层 setter 语义);警告已入 docstring。 |

### 2.8 关节能量刻意不乘 dt²(改 dt 隐式改关节刚度)

**适用线:【稳定线+phase-cd】**(同源设计,对齐 rbs-uipc)

| 项 | 内容 |
|---|---|
| 机制 | 关节罚项刚度 `K = strength_ratio · ctrl.strength_ratio · (m_parent + m_child)`,**不乘 dt²**(`abd_system/abd_system_function/setup_abd_system_gradient_and_hessian.cu:1388-1391`,注释:"Matches rbs-uipc … Joint energies are intentionally NOT scaled by dt²")。增量势中弹性/接触项均带 dt² 而关节项不带 → 关节相对其他能量项的等效刚度 ∝ 1/dt²。 |
| 影响面 | 调参可移植性:**改 dt 会隐式改变关节相对刚度**——dt 减半等效关节刚 4×。dt 扫描实验(README 的 dt U 形曲线)中关节场景的最优 `*_strength_ratio` 不能跨 dt 平移。 |
| 规避方法 | 换 dt 后重标定 `joint_strength_ratio` 族;关节调参基准表(`../JOINT_TUNING_v0.8.4.2_zh.md`)默认锚定 dt=0.01。 |
| 修复状态 | 按设计(IP 公式中相对动能极硬的罚项);文档义务而非代码修复。 |

### 2.9 其他语义陷阱速查表

| # | 陷阱 | 适用线 | 一句话 + 出处 |
|---|---|---|---|
| 1 | 多环境请求 isolated/strict 但未 `set_body_groups` → **静默降级 merged-equivalent**(打印 `[multienv] WARNING … per-env machinery disabled`) | 两线 | 用户裁定 fallback+警告不自动分组;单 env 也要显式声明(全部 body→0)才激活 per-env 路径(`engine_modules/01_config_upload.inl:974-986`;N=1 通配表会**零自碰撞→静默错物理**,故 `h_groups_present` 是硬门) |
| 2 | `set_env_offsets()` 在 `finalize()` **之前**调用 → WARNING 后直接 return,静默无效 | 两线 | `engine_modules/00_impl_api_surface.inl:571-575`;`../FEATURE_GUIDE_v0.8.4.2_zh.md` 同述 |
| 3 | per-env BVH 下未分组图元(env<0)被**排除出检测** | 两线 | `[perenv-bvh] WARNING %d/%d prims ungrouped — excluded`(`gipc_modules/11_perenv_machinery.inl:783,835-837`);若非刻意共享几何即等于漏检 |
| 4 | revolute 关节限位被 `initial_angle_offset` 平移出 atan2 域 (−π,π] → 该侧限位**永不触发** | 两线 | init 期一次 `[revolute-limit][WARN] … Re-express the limits relative to the initial pose.`(`setup_abd_system_gradient_and_hessian.cu:1363-1379`;v0.8.5.1 审计发现) |
| 5 | stitch 过刚 → step ~131 类 PCG NaN;有预警 `*** WARNING: stitch system may be too stiff ***`(ratio>1000 触发) | 两线 | `engine_modules/02_finalize_nandiag.inl:73-93`(实测 ratio 130 稳 / 1.3e4 NaN);处置:soft_motion_rate /100 或提 young_modulus |
| 6 | CFL/ground 距离非有限或 ≤0 → **刻意 fail-fast 抛错**,不是 bug | 两线 | `validateFinalCcdStateOrThrow`(`gipc_modules/10_ccd_buildcp_quarantine.inl:3376-3396`)、地面塌陷 `gipc::GeometryError`(`10:3314-3335`);设计依据:钳制-继续会把健康 µm 级平衡两帧压到 1e-23 m 永久钉死(`multienv/isolation.cu:87-102`)。视为场景/配置 bug 去排查 |
| 7 | GPU-RL ABI 的 positions/velocities 是**引擎内部顶点序**(非 METIS 反排的用户序,与 `get_vertices()` 不同) | 仅 phase-cd | `engine.py:1288-1306` docstring;直接按用户序索引 = 读错顶点 |
| 8 | `prepare_gpu_rl()` 在纯宿主布局下会**内部推进一帧仿真**(训练容量档) | 仅 phase-cd | `engine.py:1210-1214`、`03_step_getters_export.inl:381-419`;逐帧对齐外部记账时须计入 |
| 9 | 两线 checkpoint 文件**互不兼容**:稳定线 magic `STKP`(无版本/无校验/不匹配仅 printf 后 return);phase-cd `STIFFCP2` v2(CRC-64,不匹配抛 `CheckpointError`) | 两线各自 | stable `GIPC.cu:16591-16631` vs `checkpoint/checkpoint_io.cu:33-53`;phase-cd 承诺收窄为"帧边界积分器 checkpoint",绕过 load 后配对集重建的自定义恢复会引入 ~5e-6 轨迹漂移(`checkpoint/checkpoint_io.cu:1224-1242`) |
| 10 | 多 env + semi-implicit 全局早退 = batch 耦合结果(所有 env 同迭代停) | 两线 | `[semi-implicit] WARNING`(`ipc_solver.inl:2450-2454`);per-env 收敛用 `STIFF_DECOUPLE_THRESH=1 STIFF_PERENV_ALPHA=1` |
| 11 | `STIFF_*` 环境变量 typo:稳定线**完全静默**;phase-cd 默认仅 WARN | 两线程度不同 | phase-cd `config/knob_registry.h:198-239`(`STIFF_KNOB_STRICT=1` 升抛错);CI 层 G14 强制新 getenv 同 commit 注册 |
| 12 | 融合 MAS 路径不物化 mlR/mlZ 诊断缓冲(默认路径下两缓冲无效) | 仅 phase-cd | `mas_modules/05_envseg_host_pipeline.inl:1224-1227`;诊断需 oracle/dump 模式 |

---

## 3. 性能类

> 本节引用的实测口径:A800 foldshirt 4-env merged 1551 帧、宿主通道 410 s(263 ms/帧),设备 globaltimer 相位分解(`../OPTIMIZATION_ROADMAP.md:3-16`,2026-08-11);大帧/RL 通道对照见 [../SIMULATOR_EXECUTION_DESIGN.md](../SIMULATOR_EXECUTION_DESIGN.md) §6/附录 A/D。测量法定协议:全轨迹 Newton 总数为主判据(±1%);60 帧窗墙钟噪声 ±10pp 不可裁 <10pp 效应;凡裁决必带 A/A 对照。

### 3.1 MAS 预条件应用占全仿真 27%

**适用线:【稳定线+phase-cd】**(成本结构在两线的宿主路径同型;实测数字出自 phase-cd)

| 项 | 内容 |
|---|---|
| 现象 | 成本结构(上述口径):**PCG 56.9%**(其中 MAS precond 应用 112.3 s = **全仿真 27%**、SpMV 8%、mix1/mix2 7%)> GH 装配 22.0% > CCD+LS 19.9% > DCD 1.0% > BVH 0.2%。另一口径:MAS apply 占 PCG body 的 65%(SIM_EXEC 附录 D)。 |
| 根因 | MAS(多级 additive Schwarz)预条件每 PCG 迭代的稠密块应用是纯带宽/算力项。 |
| 影响面 | 重接触多环境场景的墙钟大头;也是"检测类优化天花板不足 1%"的镜像(检测合计仅 1.3%)。 |
| 规避方法 | 无用户侧规避;**不要**换对角预条件"省 MAS"——同场景实测对角每迭代便宜 ~13% 但 Newton +14~23% 且方差巨大,净账平手偏负(roadmap 死路表)。预条件选型是场景相关的:finray 2env diag 胜 / 大批量 MAS 优(候选:场景自适应,未实施)。 |
| 修复状态 | 路线图立项 ①:MAS 应用降精度 fp32(理论上限 ~13%),未实施;立项 ④:ABD per-body 12×12 块预条件(kick×uipc 案,兼治稳定性),未实施。 |

### 3.2 整帧图在大帧 regime 是结构性负收益

**适用线:【仅 phase-cd】【实验性,默认关】**(`STIFF_FRAME_FULL_GRAPH`,默认 0)

| 项 | 内容 |
|---|---|
| 现象 | 大帧回放(百 ms 级/帧)开整帧图**倒贴**,幅度**场景相关 +3%~75%**(`../OPTIMIZATION_ROADMAP.md` 图通道结论)。收官测量(4090 干净卡,60 帧中位 3)vs v0.8.5——forcegrip 图开 +43%、beaker +35%(同构建图开/图关比 1.46×/1.24×,跳零修复后);早期未修状态下六场景无一胜出:五场景 1.62×~10.8× 全慢,第六个 beaker_finray 直接 OVF_CCD 重试预算耗尽跑不完(`../GRAPH_DEFAULT_ON_EVIDENCE.md`,其倍率表已过时但结论方向存活)。附录 D 的"图开 +10~15%"是**锚场景口径**(A800 foldshirt 4-env merged 1551 帧),不是上限——引用任何幅度必须钉住场景/基线集/中位轮数口径,拿 +10~15% 做通道决策会把 forcegrip 类场景的实测代价低估约 3 倍。 |
| 根因 | 大帧发射开销占比过小,图付出的容量成形(padded 工作)、事务保险(快照/验证/还原)与轨迹效应(§3.3)成为净成本。破案要点:主导项曾是零 pad 三元组全 hash 到 (0,0) 的原子长龙(`binned_block_merge_scatter` 22×/实例),跳零默认开后仍余结构性成本;四次"收窄发射宽度"实测 ~0%——**padded WORK 贵、padded LAUNCH WIDTH 近乎免费**(C6-v/w/z)。 |
| 影响面 | 把 `STIFF_FRAME_FULL_GRAPH=1` 当"免费加速"开在大帧回放/操作轨迹上的用户。**接触升级轨迹更是合同矛盾**:foldshirt 抓握第 15 帧 auto-prepare 后 25/25 帧 OVF_TRIPLETS 失败(fail-closed 正确拒绝;首次冒烟的"2.2× 加速"是空转假象)。 |
| 规避方法 | 按 regime 选通道(附录 D 定稿):**大帧 = step 通道(宿主,图关,即默认)**;**RL 稳态微步 = residency 通道**(`prepare_gpu_rl`,A800 5.0×、4090 1.27×);接触升级轨迹禁止 prepare。图开的唯一非速度理由:尾延迟可预测(方差 ±8% vs 宿主 ±21%)。 |
| 修复状态 | 判决为结构性(死路表"大帧整帧图"),不再投入;正确性侧的回滚漏抄 bug 已修(commit `4e49fda`:`capture_full_graph` 漏抄 ABD q 族+kappa 快照/还原 → 破 `x≡J·q` → 回退帧 LS 必然耗尽;修后 ls_exhaust 2→0,门禁 22/22)。 |

### 3.3 事务壳 +6% 轨迹效应(成对审计结论)

**适用线:【仅 phase-cd】**(commit `4219f37`,`STIFF_FRAME_FORCE_ROLLBACK=2` 成对审计,A800 foldshirt 4-env merged 1551 帧)

| 项 | 内容 |
|---|---|
| 现象 | 图开 +10% Newton 盈余的解剖:同状态图 vs 宿主逐帧成对比较,**图内求解器语义仅 +1.0%**(869/1550 帧 Newton 逐位相等);全轨迹 Newton:纯宿主 8808±60 | 仅事务壳(无图)9342±230 | 生产整帧图 9697±35 —— **主体 +6% 来自事务壳本身**(快照/验证/还原包裹的 ULP 级逐帧扰动 × merged 混沌接触放大;15σ 聚类分离排除 run-to-run 运气),再 +4% 图化增量。 |
| 根因 | 事务壳改变了浮点求值上下文(ULP 级),merged 的混沌轨迹把逐帧微扰积成 Newton 成本;并且强制逐帧回滚兜底循环即使零图参与也让轨迹成本翻倍(D 臂 +100%)——**回滚兜底循环非物理中性**(尽管 1550 次 fallback 的顶点+q 还原全部逐位审计通过)。 |
| 影响面 | 任何"把事务壳当免费保险"的推理;以及用图开轨迹与图关轨迹做同物理对比的实验设计(两者是不同轨迹族)。 |
| 规避方法 | 性能对比一律成对审计或 A/A;大帧不开图(§3.2)。 |
| 修复状态 | 根治留"首分歧帧二分"(`_dbg_ksum` 哈希管线),低优先(roadmap:78-81)。 |

### 3.4 cap=50 饥饿帧(CCD 钳制"长征帧")

**适用线:【仅 phase-cd 实测记档;机制两线同族】**

| 项 | 内容 |
|---|---|
| 现象 | `env_newton_iter_cap=50` 的多环境轨迹中,每轨迹出现 16–25 个被 TIMEOUT 冻结的"饥饿帧",约占总 Newton 12%(`../OPTIMIZATION_ROADMAP.md:37-38`)。 |
| 根因 | CCD α 钳制(CFL cap 等)下的"长征帧":单帧需要大量小步 Newton 才能走完位移,cap 到点被冻结。本质是 α 配方保守,不是发散。 |
| 影响面 | `env_newton_iter_cap>0` 的生产配置;被冻结 env 该帧欠收敛(与 §2.4 同后果,但按 env 局部化)。 |
| 规避方法 | ① 提 cap(代价墙钟)。② 重接触场景可试 α 配方 opt-in:`STIFF_CCD_SLACK_M=0.9 STIFF_CCD_CFL_FACTOR=1.0` → Newton −7.9%、墙钟 −3~4%(A800 fs4-4env 全轨迹;**轻接触零收益或小亏勿开**,默认全关;SIM_EXEC 附录 C)。 |
| 修复状态 | 等 roadmap 立项 ② α research 档(−16% Newton,前置=穿透审计基建),未实施。 |

### 3.5 已证死路速查(勿再投入)

以下方向已被实测否决,列此防止重复投入(逐条证据见 [../OPTIMIZATION_ROADMAP.md](../OPTIMIZATION_ROADMAP.md):62-73):

| 方向 | 裁决证据 |
|---|---|
| 检测类优化(BVH/broadphase/DCD 节流) | 成本合计仅 1.3%,天花板不足 1% |
| 对角替代 MAS(本场景族) | 慢 22%,Newton +23% |
| 大帧整帧图 | 结构性负收益(§3.2) |
| 融合核 TU 抽取 | 255 寄存器、栈 +41%、SASS +9.5%、460 B/线程溢出,双盲复现否决 |
| 图发射宽度收窄 / pad 削减 | 三/四次实测 ~0%(跳零后宽度近乎免费) |
| 容量档 headroom 提档 | 回退减半但 Newton 不动,墙钟 +15% |
| 容量档持久化 | 绑场景,改场景即作废 |
| MAS levelnum 扫描 | 用户裁定不做 |

---

## 4. 工程 / 运维类

### 4.1 OVF 回退税与容量重录成本(A800 一次 20–30 s)

**适用线:【仅 phase-cd】【实验性,默认关】**(整帧图/residency 通道;默认宿主路径无此项)

机制链(详见 [PRINCIPLES_EXECUTION.md](PRINCIPLES_EXECUTION.md) §5):图内碰撞缓冲按训练容量档(tier)烘焙;帧内实际需求超档 → 设备写 OVF 位(`OVF_DCD_PAIRS/OVF_CCD_PAIRS/OVF_TRIPLETS/OVF_UNIQUE_BLOCKS/OVF_MAS_CLUSTERS`,`frame_fsm/frame_status.cuh:46-50`)+ 拒绝提交坏帧 → 宿主帧边界裁决增长 → **该帧从逐位恢复态用 release 宿主求解器完成**(C6-i 默认 `capacity_fallback`,`frame_transaction.cu:4491-4503`)→ 增长塑形**下一帧**的重录。

| 项 | 内容 |
|---|---|
| 现象 | ① 回退税:溢出帧整帧走宿主(付一次宿主帧成本 + 帧内已烧掉的图 attempt);中帧轮询兜底(`[C6-o] capacity retry flagged … aborting attempt at Newton iter %d`)防止截断梯度空转满 1000 迭代(towel 曾烧 8.5× PCG 工作量,`core/ipc_solver.inl:1325-1370`)。② 重录成本:容量档变化触发下一帧 re-record,**一次 capture:A800 ~20–30 s,4090 ~1–3 s**(C6-n 实测;towel 502 s 异常即由此)。③ 增长风暴:同轴 8 帧内复越 → streak 升级 2×/封顶 4×(`frame_transaction.cu:4165-4182`);耗尽 `STIFF_FRAME_MAX_RETRIES`(默认 3,clamp 0–16)抛 `capacity retry budget exhausted`。 |
| 根因 | CUDA Graph 的 L5 硬边界:`cudaMalloc` 仅 CPU,发射宽度是录制常量——容量变化必须跨帧边界处置(SIM_EXEC 五层动态性)。 |
| 影响面 | residency/图通道用户,尤其 A800 类慢主机(单次重录抵掉数十帧节省——这也是 pad-class allowance 削减保持 opt-in 的原因,C6-z);接触量逐帧升级的轨迹会连环溢出(§3.2 合同矛盾)。 |
| 规避方法 | ① 遵守通道判据:容量平稳(RL 稳态微步)才 prepare。② prepare 前用真实峰值负载 warm-up(prepare 的训练帧机制即为此)。③ 观测:`STIFF_FRAME_GRAPH_DIAG=1` 打印 `[graph-train]` 增长行与 fallback 原因;`get_frame_status()` 读 `required_*` vs `hw_*`。④ `STIFF_GRAPH_TIER_HEADROOM`(默认 1,可 1–64)是**确定性↔速度双重旋钮**:贴档的 racy atomicAdd 计数骑边界时过时不过,重试非物理中性(~1e-6 扰动被混沌放大:towel crumple 0.77..1.03 vs 宿主确定 0.905,`GIPC.cuh:980-1008`);headroom=2 曾使容量宽 pass 翻倍(forcegrip 2.08×→1.39× 的反向教训)。⑤ `STIFF_GRAPH_INGRAPH_RETRY=1` 恢复图内重试为实验行为(曾放大非确定性,不推荐)。 |
| 修复状态 | C6 系列收官后的定型行为;帧 0 恒走 release 求解器(warm-up 边界)。 |

### 4.2 显存:容量档与配对缓冲只增不减

**适用线:【稳定线+phase-cd】(grow-redo);容量档族【仅 phase-cd】**

| 项 | 内容 |
|---|---|
| 现象 | 长时间运行的常驻进程显存单调上行至峰值负载水位:① 宿主 grow-redo——配对缓冲溢出自愈(`[DCD-grow]/[CCD-grow] … redo detection`,增长 1.5×+1,重跑检测;两线同款,stable `GIPC.cu:10318/11185/11246/11637`),且 CCD 镜像与 DCD 锁步、**从不收缩**(`contact/pair_buffers.cuh:1-50`;该头 10-15 行记载了曾经"收缩-越界"潜伏 bug 的修复——收缩被明令禁止)。② 图容量档按**峰值**训练(帧末计数会低估:foldshirt 帧末 ~58k vs 峰值 ~390k,C6-b 教训)且逐轴单调增长;地面轴恒按最坏 `surf_vertexNum` 烘焙(无 OVF 位,截断不可检测)。 |
| 根因 | 收缩会使已发布指针/已录图失效,且历史上引入过越界;峰值训练是欠清零/欠检测事故的疫苗。 |
| 影响面 | 显存预算紧张的部署(多引擎共卡、A800 共享集群);频繁 grow 行还提示初始容量偏小(性能:每次多付一遍检测)。 |
| 规避方法 | ① 初始容量调 `collision_detection_buff_scale`(注意 §2.3 双默认值)。② 图路径勿盲目提 headroom(§4.1)。③ 历史教训已内建:OVF 增长按设备报告的轴掩码只长越界轴——"全轴齐长"曾致 triplet 512k→2.77M→11.8M 三连爆 OOM(`frame_transaction.cu:4277-4284` 注释)。④ 需要回收显存:销毁重建 Engine(注意 §4.5 进程级约束)。 |
| 修复状态 | 按设计;容量档持久化(跨进程复用训练档)已裁定不做(绑场景)。 |

### 4.3 strict 模式的性能代价

**适用线:【稳定线+phase-cd】**

| 项 | 内容 |
|---|---|
| 现象 | strict(逐位可复现)相对 merged 的代价分两层:① 锚级场景实测**个位数 %**(`multienv/mode_contract.h:36-37`,随负载而变);② 三模式全矩阵里 strict 均步时可达 **+20%~2×**(4090:finray_beaker 16.6→35.0 s;A800 盘子 228f:merged 26.9 s / strict 45.0 s,`../MODE_MATRIX_REPORT_2026-07-27.md`)。同时 strict 峰值 Newton 系统性**最低**(towel 61 vs 94/98;盘子 15 vs 22/19)——canonical 序降低了最坏帧,但均帧付序化成本。 |
| 根因 | canonical 发射顺序(EE_CANON/EE_DETGATE/CCD_CANON)+ 顺序无关 SpMV(SPMV_DET)+ layout 固定策略。 |
| 影响面 | 把 strict 当默认生产模式的部署;另注意 **strict 不图化**——整帧图资格检查明确拒绝 strict(容量网格归约会重排求和序 = 换锚战役;用户 2026-07-29 决定),strict 用户没有 residency 加速可拿。 |
| 规避方法 | 确定性需求分级:逐位复现(CI 锚、审计)才用 strict;隔离需求用 isolated;吞吐用 merged。单 env 场景 strict 仍完全有意义(canonical 序 ⇒ 逐位可复现),isolated 单 env ≈ merged+env-flag。 |
| 修复状态 | 代价按设计;strict 图化无计划。 |

### 4.4 CI 金锚:默认路径受逐位约束,改代码须过 22 段门禁

**适用线:【仅 phase-cd】**(稳定线无 scripts/ 门禁体系)

| 项 | 内容 |
|---|---|
| 现象 | 对 phase-cd 改任何热路径代码,推送时触发 **22 段门禁**(push hook 在临时 detached worktree **从零构建**后跑,见 [../CI.md](../CI.md));strict 金锚为逐位断言:当前锚 `0544461bd82123ae`(dlto 后;run-to-run 逐位 + 4090 sm_89 ≡ A800 sm_80 跨架构同值),历史锚 `f7fb5a786c2d7935`(dlto 前世代)。**逐位门禁天生脆**:实测给 scatter lambda 加一个永不触发的分支就让 gate 3/3→1/3(编译器调度扰动 ULP;A800_ALLEXAMPLES C6-v)。 |
| 根因 | 逐位可复现是 strict 的产品承诺,锚是它的可执行契约;nvcc `-fmad=true` 下跨 TU 移动代码即可漂移浮点结果(V086 重构约束)。 |
| 影响面 | 引擎开发节奏:dlto 全量构建 +76%、增量 ~60 s 串行 dlink;"看起来无关"的重构也可能碰锚。门禁历史教训:**门禁必须自带构建**——曾有门禁跑陈旧二进制导致武装验证空跑(G0 build + G0.2 realpath 断言已结构性关死)。 |
| 规避方法 | ① 分层理解容差:det 层(`STIFF_SPMV_DET` 确定栈)逐位断言是数学性质;default 层用基线自噪声包络(budget = max(4×noise, 1e-11×scale))——改默认路径先看落在哪层。② 碰锚即换锚是**战役**不是提交:须 run-to-run + 跨架构重认证(参照 dlto 换锚流程,`../RELEASE_NOTES_v0.8.6-rc1.md`)。③ 绝不静默换锚(V086 计划明文)。逃生口 `SKIP_GATES=1` 会记入 bypass.log。 |
| 修复状态 | 体系按设计;门禁两层化(C6-w)后比单层逐位**更强**且更抗良性扰动。 |

### 4.5 进程级全局状态:单进程单模式 / 单 Engine

**适用线:锁与类型化报错【仅 phase-cd】;底层全局符号问题【稳定线+phase-cd】**

| 项 | 内容 |
|---|---|
| 现象 | phase-cd:同进程第二个 Engine 请求不同 `multienv_mode` / 不同 `per_env_exit`,或首个 Engine 之后手改 `STIFF_*` 模式旗标再触发 step/finalize → 抛 `LifecycleError`(进程级三锁,`engine.py:489-514`);`finalize()` docstring 明文单进程仅一个 finalized Engine(legacy 求解器缓冲经进程级 CUDA symbol 发布,`engine.py:983-991`、`sim_engine.h:419-422`);rc1 后更引入 RuntimeOwnerLease 强制。稳定线:**无锁**——先 strict 后 merged 的混用不报错,但底层同样是 31 个进程级 `__device__` 全局 + host 端 last-value 缓存(单引擎租约假设),混用可静默腐蚀状态。 |
| 根因 | 原生热路径对 `STIFF_*` 首用即缓存;设备符号进程级。 |
| 影响面 | 多模式对比脚本、pytest 同进程多 fixture、常驻服务内重建引擎。 |
| 规避方法 | 跨模式/跨配置对比一律用**子进程**;两线皆然(phase-cd 只是把静默腐蚀升级成响亮报错)。 |
| 修复状态 | 结构性解(单 `__constant__ EngineDeviceDescriptor` 描述符,drain-then-publish,买到串行多引擎)已立项为 v0.9 phase-1(`../DEVICE_DESCRIPTOR_PLAN.md`);phase-0 的三处跨引擎串态真 bug(模式闩锁/静态 scratch)已清。 |

---

## 5. 上游依赖(KemengHuang/StiffGIPC)与分叉点

本项目是上游 **KemengHuang/StiffGIPC**(GIPC 论文作者实现谱系)的长期分叉。与已知问题相关的结构性事实:

1. **分支拓扑**:`main` = 上游只读镜像(`CLAUDE.md:49-57`);`release/stable-0.8` = 稳定发布线;`codex/phase-cd` = 工程线。两条产品线的 git merge-base = `05c3f75`,早于 v0.8.5.3 —— 因此稳定线 v0.8.5.3 的接触 I/O 修复(`1bc13ef` 摩擦读数快照、`1d05c7a` teleport 表面顶点同步)与 v0.8.5.4 全部提交(`dc1a297`/`0894958`/`c0339c8` 真静摩擦族)**均不在 phase-cd 历史**;phase-cd 对 teleport 问题有自己的(不同实现的)解法(§1.2),对摩擦读数**没有**(§1.1),对真静摩擦也**没有**(§1.6)——三项合并清单见 §1.0。
2. **上游冻结项**(为保持可同步性,两线均不改动):EE mollifier smooth 分支(§1.3)、close-set 帧内 Kappa 倍增路径。后者是**刻意退役的死路**:宿主的帧内倍增门在 `h_close_gpNum/h_close_cpNum` 宿主镜像上,而这两个镜像树内**从无写点**,恒 false → Kappa 帧内永不动,每帧由梯度投影策略重播种(`gipc_modules/09_friction_sets_host_mem.inl:917-919`、`12_host_wrappers_fem.inl:344-345`:"do not revive … without a separately validated adaptive-contact redesign")。整帧图曾如实读设备计数把死路复活并引发 kappa 41.73→2670.70 六连倍增杀帧,现图路径默认宿主等价,倍增 kernel 仅留在 `STIFF_GRAPH_LEGACY_KAPPA_DOUBLE` 后供重设计对照(`10_ccd_buildcp_quarantine.inl:3151-3170`)【实验性,默认关】。
3. **上游三方库遗留**:muda 的点-边 CCD `toc = roots[i]*(1-eta)` 带 `//TODO: distance eta` 注释(`muda/src/muda/ext/geo/distance/details/ccd.inl:462`)——eta 语义近似,非本仓可控,主 CCD 链(ACCD)不经过它。
4. **上游算法基线**:CCD = Additive CCD(Li et al. 2021 Codimensional IPC 谱系,`ACCD.cu`);势垒 RANK=2 对数势垒 `E = κ(d−d̂)²log²(d/d̂)`(d 为平方距离,`contact/barrier_rank.h:7`);这些与上游一致,handbook 级对照修改都属换锚战役。
5. **发布面**:公开仓 `haoxiangNtu/stiff-physics`(镜像 README/wheel);公开 commit 作者铁律与发布流程见 `STIFF_PHYSICS_RELEASE_HANDBOOK.md`(注意其三处过时项:CUDA 架构默认已是 `80;89;120`、GIPC.cu 行数、"自适应 Kappa 未启用"表述对 phase-cd 已过时——phase-cd 有图内 kappa 链但默认 host-equivalent)。

---

## 6. 待核实清单

以下陈述在本册标注"待核实",不作为承诺引用:

1. **"folded 布料 44 万 mollify 请求 / 0 执行"** 的具体计数出自战役记录,本次代码勘探确认了机制(请求被计数、`smooth=false` 使执行为零)但未复测该数字(§1.3)。
2. ~~**v0.8.5.4 的 wheel 资产状态**~~ **已关闭(2026-09-08)**:公开仓 Release `v0.8.5.4` 已正式发布(`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` 亲验:`published: 2026-08-11T17:07:37Z`、非 draft/prerelease),`stiff_physics-0.8.5.4-cp311/cp312-linux_x86_64.whl` **两个 wheel 均已挂出**,公开仓 README 安装 URL 也已由提交 `a38ede4` 指向它(OPEN_POINTS OP-001 已关闭)。版本本体此前即已确认(tag `c0339c8`、`pyproject.toml` = `0.8.5.4`、`git show HEAD:CHANGELOG.md:7-44` 有完整 [0.8.5.4] 条目)。**仍需注意**(与发布状态无关):磁盘工作区被回退到 v0.8.5.3 内容,故本册稳定线行号仍是 v0.8.5.3 口径;而拿到 v0.8.5.4 wheel 的用户默认就在真静摩擦轨迹上(§1.5、§1.6)。
3. **phase-cd 摩擦读数恒零**已由代码结构证实(无 snapshotFrictionForce、accessor 提交后现算)并与稳定线 `1bc13ef` 诊断同构,但未在 phase-cd 上跑剪切台复测数值(§1.1);移植后应以稳定线的滑比验证(0.600±0.008 @ μ=0.6)为验收基准。
4. ~~**硬钉链式法则缺失(§1.4)在稳定线的存在性**~~ **已核实(2026-09-07)**:稳定线单体 GIPC.cu 含逐字相同的 KNOWN LIMITATION 注释与活代码路径(stable `GIPC.cu:13894-13906`、`[M3.5] WARN` 在 `:13787`),并经 `add_fem_pin_to_abd` / `add_fem_pins_with_local_pos` 公开绑定可达——§1.4 已改标【稳定线+phase-cd】。
5. 稳定线 v0.8.5.3 的 line-search WARN 行号(stable `GIPC.cu:15221`)与 grow-redo 行号(stable `GIPC.cu:10318` 等)出自勘探笔记的当日核对,复引前建议 grep 复核(单体文件行号对补丁敏感)。
6. **phase-cd 上的保持段蠕变未运行时复测**(§1.6):代码结构已证两条根因都在(场景派生 `fDhat`、无摩擦锚),蠕变数字(3.7 mm / 11–20° / 每步 1.0e-5 m)全部取自稳定线 v0.8.5.4 提交的 flask_cap 实测;工程线上跑一次同构保持实验即可闭合,移植后以稳定线的"保持段滑移 0.00 mm、cap 倾角 0.7° 恒定"为验收基准。
