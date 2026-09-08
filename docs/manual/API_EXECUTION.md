# StiffGIPC 手册 · 执行与高级 API 参考(API_EXECUTION)

> 本分册覆盖:多环境三模式、可复现性、GPU 驻留 RL、异步 episode、checkpoint 与帧状态、
> 步健康遥测、`STIFF_*` 环境变量全清册、性能指南与示例配方。
> 场景构建 / 求解器核心 / 基础 Python API 见本手册同目录其他分册;
> 背景设计文档见 [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md)、
> [`../GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md)、
> [`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md)。

## 适用版本与标注约定

本分册同时覆盖两条线,所有条目按下列标签标注适用性:

| 标签 | 含义 |
|---|---|
| 【稳定线+phase-cd】 | 两条线都有,语义一致(差异会单独注明) |
| 【仅 phase-cd】 | 只在工程线 `codex/phase-cd` 存在;稳定线设了/调了无效或不存在 |
| 【仅稳定线】 | 只在稳定线 v0.8.5.3 存在(tactile 线修复),phase-cd 尚未移植 |
| 【仅稳定线 v0.8.5.4+】 | 只在稳定线 v0.8.5.4(`dc1a297`..`c0339c8`)存在的摩擦线工作;v0.8.5.3 与 phase-cd 均无(两树 grep `absolute_epsv`/`friction_anchor` 0 命中) |
| 【实验性,默认关】 | 存在但默认关闭,非生产承诺,开启前读对应小节的风险说明 |

- **稳定线** = 仓库 `Stiff-GIPC-stable-08`,分支 `release/stable-0.8`,
  发布点 **tag `v0.8.5.3` = commit `b8e27a1`**(2026-08-11 发布;公开仓
  `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel,CUDA 架构 sm_80/89/120)。
  文件布局为重构前单体(`StiffGIPC/GIPC.cu`)。
  注:该仓库工作树在 v0.8.5.3 之后还有摩擦线提交(tag `v0.8.5.4`
  "default-on true static friction" 等);本分册的**默认承诺面仍是 v0.8.5.3**,
  v0.8.5.4 的静摩擦/friction-anchor 行为一律按【仅稳定线 v0.8.5.4+】单独标注
  (旋钮见 §7.4/§7.5,strict 交互见 §2.1,参数与升级影响见 API_CORE §2.1)。
  v0.8.5.4 **已正式对外发布**(2026-08-11T17:07:37Z,cp311/cp312 双 wheel 均已挂出;
  附录 A#2 / OPEN_POINTS OP-001 已关闭)——读者拿到的 wheel 可能已是 v0.8.5.4,
  本分册按 v0.8.5.3 给的摩擦默认值届时需按【仅稳定线 v0.8.5.4+】标注改读。
  更正一处旧勘探口径:CHANGELOG `[0.8.5.4]` 条目在提交 `0894958`/`c0339c8` 里
  **是存在的**;当时树上看不到是因为工作树被误留的 8 文件暂存回退按在 v0.8.5.3,
  该状态已于 2026-09-08 清除,现在磁盘 CHANGELOG 即含该条目。
- **工程线(phase-cd)** = 仓库 `Stiff-GIPC-c1-ls-graph`,分支 `codex/phase-cd`,
  HEAD `b3ab747`;含 v0.8.6 模块化重构、整帧 CUDA Graph、GPU 驻留 RL、episode、
  checkpoint v2 等全部 v0.8.5 之后的工作。
- 文中 `文件:行号` 出处:未加前缀指 phase-cd 树;前缀 `stable` 指稳定线树
  (行号按 2026-09-07 勘探时的文件内容)。
- **重要行为差异提前声明**:tactile 线的两个修复——接触力读数中摩擦分量恒零的修复
  (`snapshotFrictionForce`,stable GIPC.cu:16504)与 `reset_transient_contact_state`
  API(stable sim_engine.cu:3620)——**只进了稳定线 v0.8.5.3**;phase-cd 上
  `get_vertex_contact_forces` 的 `friction_lagged`/`total` 分量仍受恒零 bug 影响
  (滞后摩擦梯度是 `x − o_vertexes` 的函数,步末 `o = x` 提交后重算恒为零)。
  **摩擦力读数当前仅稳定线可信。**

## 目录

- [1. 多环境三模式(merged / isolated / strict)](#1-多环境三模式merged--isolated--strict)
  - [1.1 保证–代价矩阵](#11-保证代价矩阵)
  - [1.2 模式解析:`resolve_multienv_mode` 与 flag bundle](#12-模式解析resolve_multienv_mode-与-flag-bundle)
  - [1.3 进程级模式锁与半配置警告](#13-进程级模式锁与半配置警告)
  - [1.4 C++ 运行时真相:`ModeConfig` 快照【仅 phase-cd】](#14-c-运行时真相modeconfig-快照仅-phase-cd)
  - [1.5 `set_body_groups`:env 分组规则](#15-set_body_groupsenv-分组规则)
  - [1.6 `set_vertex_env_ids`:逐顶点接触隔离](#16-set_vertex_env_ids逐顶点接触隔离)
  - [1.7 `per_env_exit` overlay](#17-per_env_exit-overlay)
  - [1.8 per-env 遥测 API](#18-per-env-遥测-api)
  - [1.9 隔离契约:quarantine / revive](#19-隔离契约quarantine--revive)
- [2. 可复现性配方](#2-可复现性配方)
- [3. GPU 驻留 RL 全家桶【仅 phase-cd】](#3-gpu-驻留-rl-全家桶仅-phase-cd)
- [4. 异步 episode 七件套【仅 phase-cd】](#4-异步-episode-七件套仅-phase-cd)
- [5. checkpoint 与 get_frame_status](#5-checkpoint-与-get_frame_status)
- [6. 步健康遥测](#6-步健康遥测)
- [7. STIFF_* 环境变量全清册](#7-stiff_-环境变量全清册)
- [8. 性能指南](#8-性能指南)
- [9. 示例配方](#9-示例配方)
- [附录 A. 本分册未决事项(待核实)](#附录-a-本分册未决事项待核实)

---

# 1. 多环境三模式(merged / isolated / strict)

三模式的承诺表(promise table)的单一权威是纯文档头文件
`StiffGIPC/multienv/mode_contract.h`(mode_contract.h:1-57,【仅 phase-cd】;
稳定线行为同构但无此契约文件)。模式是**两条独立轴**上的三个点
(mode_contract.h:5-9):

- **axis A(coupling)**:各 env 共享多少求解器状态;
- **axis B(determinism)**:承诺什么可复现性;
- 第四角(coupled + deterministic)没有用户故事,契约明文禁止在无用户故事时新增模式。

## 1.1 保证–代价矩阵

| | **MERGED**(默认) | **ISOLATED** | **STRICT** |
|---|---|---|---|
| 设计目标 | 相对裸单仿真的**最小开销**(owner decision 2026-07-26)——"和普通 GIPC 一样快" | per-env 公平与隔离 | isolated 的全部 + 逐位可复现 |
| 求解结构 | 一个全局求解:全局 line-search α、全局 Newton 收敛、共享 κ;热路径零 per-env 机制 | per-env line-search α、per-env 收敛冻结、per-group κ、env 本地 broadphase(无跨 env 接触) | 同 isolated |
| 病态 env | **无隔离承诺**:病态 env 抛异常并杀死整个 batch(fail-fast 是找 bug 的立场;fail-isolate 扩展被评估并**拒绝**用于 merged) | **IRON LAW**:病态 env 中途被检疫(quarantine)为完全惰性(INERT),健康 env 继续跑。**前提:检疫可用性门开着**(§1.9,isolation.cu:78-85)——除 groups>1 外还要求 host 遥测路径(`env_newton_iter_cap>0` 或 `STIFF_PERENV_TELEM=1`);**isolated/strict 的 flag bundle 不含 `STIFF_PERENV_TELEM`,`env_newton_iter_cap` 默认 0**,纯默认配置走蓄意无 NaN 防御的纯设备快路径,`quarantineEnv` 直接返回 false、调用者保留 throw——病态 env 仍杀死整个 batch(isolation.cu:112-115)。**要拿到铁律,须配 `env_newton_iter_cap`(如 100)或 `STIFF_PERENV_TELEM=1`** | 同 isolated |
| 可复现性 | **无承诺**(原子发射/规约顺序) | 无承诺 | **逐位(BITWISE):run-to-run 且跨架构**(已证 sm_80 ≡ sm_89;当前发布面锚 `0544461bd82123ae`,dlto 采纳后 A800 sm_80 复验同值,见 §2.1 与 ../RELEASE_NOTES_v0.8.6-rc1.md:132-137;mode_contract.h:35 写的 `f7fb5a786c2d7935` 是 dlto **前**世代锚,已陈旧。前提同 §2.1:同一 wheel/同一编译配置——换链接优化即换锚) |
| 代价 | 最快 | 中 | canonical 发射顺序 + layout 固定策略;锚级场景实测个位数 %,随负载而变 |
| 行为门禁 | kick + foldshirt-smoke + G9 包络(Newton 计数)+ G9 跨模式等价性 | midrun + startup quarantine 门禁 | G1 bitwise anchor(armed MIRROR+SLOT) |

出处:mode_contract.h:13-38。

补充口径:

- Python 侧注释另有表述(engine.py:187-190):strict 保证**固定场景 + 固定 batch layout**
  的可重复性;**cross-batch-size identity 仍是验证目标,不是公开契约**。
- 预条件器与模式的经验配对(foldshirt 研究,mode_contract.h:53-55):MAS 最配 MERGED,
  diagonal 配 ISOLATED/STRICT;尚未自动选择。
- 单 env 场景的模式语义(../UI_DEMO_MODE_TEST_CHECKLIST.md:42-47):isolated 的 per-env
  机制要求 groups>1 才激活,单 env 时 isolated ≈ merged + env-flag;
  **strict 单 env 仍完全有意义**(canonical 序 ⇒ 逐位可复现)。

## 1.2 模式解析:`resolve_multienv_mode` 与 flag bundle

【稳定线+phase-cd】模式在 **Python 层解析**、C++ 层只消费低层 `STIFF_*` 旗标。
bundle 定义位于 `stiff_physics/engine.py`:

```python
def resolve_multienv_mode(mode: str = "merged") -> str   # engine.py:287-320
```

| 参数 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `mode` | `"merged"` | — | 模式名或别名;`STIFF_MULTIENV_MODE` 环境变量覆盖此实参 |

- **别名表**(engine.py:213-217):`0/merged/a → merged`;`1/isolated/decoupled/b → isolated`;
  `2/strict/deterministic/c → strict`。未知值抛
  `ValueError("unknown multienv_mode ...; use merged/isolated/strict (or 0/1/2)")`。
- **setdefault 语义**:解析器只对未设的 `STIFF_*` 键写值——**显式设置的 `STIFF_*`
  环境变量永远赢**;对不再属于所选模式的旗标,解析器**回收(`del`)自己设过的键**
  (`_retract_our_flags`,engine.py:277-284),**从不写显式 `"0"`**;用户改过的键
  归用户,不回收。
- **flag bundle**(engine.py:194-212):

```
isolated = STIFF_BVH_ENVDET  STIFF_PERENV_BVH  STIFF_DECOUPLE_THRESH
           STIFF_PERGROUP_KAPPA  STIFF_SEGMENTED_PCG  STIFF_PERENV_ALPHA
           STIFF_PERENV_PAR                                  (7 旗)
strict   = isolated + STIFF_EE_CANON  STIFF_EE_DETGATE
           STIFF_CCD_CANON  STIFF_SPMV_DET                   (7+4=11 旗)
merged   = (无;额外 setdefault STIFF_EE_LB=2)
```

各旗含义见 [§7.2 表](#72-模式旗mode_isolated--mode_strict稳定线phase-cd)。
`_MULTIENV_ISOLATED_ONLY` 为**空表**(engine.py:212)——当前不存在
"只在 isolated 合法、strict 非法"的旋钮(`STIFF_PERENV_PAR` 于 2026-07-04 通过
strict 资格并入基础表:run-to-run / cross-env / batch-size N=2/4/8 全 0.000;
`STIFF_PERENV_PAR=0` 可单独退出)。merged 的 `STIFF_EE_LB=2`(selfQuery_ee 的 lb2
变体)是 EE-DCD 实测 −1.69% 的微优化;strict 不默认(engine.py:312-319)。

**使用方式**:一般不直接调用 `resolve_multienv_mode`——
`Engine(Config(multienv_mode="strict"))` 会在任何 load/finalize/step 之前自动解析
(engine.py:469-474);优先级 `STIFF_MULTIENV_MODE` env > `Config.multienv_mode`。

## 1.3 进程级模式锁与半配置警告

【仅 phase-cd】`STIFF_*` 是进程全局的,且原生热路径**首次读取即缓存**——跨模式必须用
子进程(engine.py:220-224)。三个进程级锁(engine.py:489-514):

1. `_PROCESS_MULTIENV_MODE`:同进程第二个 Engine 请求不同模式 → `LifecycleError`;
2. `_PROCESS_PER_ENV_EXIT`:`per_env_exit` 同为进程级;
3. `_PROCESS_MODE_SIGNATURE`:首个 Engine 之后 `STIFF_*` 模式旗标被改动 →
   `LifecycleError`(签名键集 = 全部 bundle 旗 + per_env_exit 旗 + `STIFF_EE_LB` +
   `STIFF_PERENV_MASK_DEV`,engine.py:236-246);finalize / step /
   launch_episode_async / prepare_gpu_rl / prepare_gpu_rl_episode 五个入口
   再次断言(`_assert_process_mode_signature`,engine.py:259-267;
   teleport 与 checkpoint 入口**不**调用它)。

锁在原生构造 + CUDA init 成功后才提交(失败的构造不毒化后续 Engine 尝试,
engine.py:547-552)。稳定线**没有**这些锁(grep 无命中),但同样的静态缓存事实存在——
在稳定线上跨模式复用进程属于未定义行为,同样应当用子进程。

**半配置警告**(Python 侧 engine.py:531-541;C++ 侧另见 §1.4):

- `STIFF_DECOUPLE_THRESH` 无 `STIFF_PERENV_ALPHA`:per-env 冻结无法运行 →
  Newton 循环每帧跑满迭代上限(**比默认更糟**);
- `STIFF_PERGROUP_KAPPA` 无 `STIFF_DECOUPLE_THRESH`:per-group kappa 只是全局
  kappa 的广播 stub——env 间**没有** kappa 隔离(代码根据:
  gipc_modules/13_kappa_partition_gradhess.inl:241-249 的 stub 广播分支)。

## 1.4 C++ 运行时真相:`ModeConfig` 快照【仅 phase-cd】

`struct ModeConfig`(multienv/mode_config.h:22-140)是 finalize 时刻对 `STIFF_*`
环境的**一次性快照**,"C++-side SINGLE RUNTIME TRUTH":捕获点
`GIPC::m_mode_config = ModeConfig::capture_from_env()` 后跟 `warn_if_incoherent()`
(gipc/gipc.cu:88-92);新代码必须读快照不读环境;mid-run 改环境变量无任何支持。

- **env 读法 `env_on(k)`**(mode_config.h:40-44):**值感知**——非空且首字符非 `'0'`
  才算 on。原因:用户可手工设 `STIFF_X=0` 关某旗(显式设置永远赢,§1.2),
  presence-only 检查会把这种关掉的旗重新打开。(mode_config.h:36-39 注释里
  "Python 解析器写显式 `"0"`"的说法已陈旧——解析器只回收自己设的键,从不写
  `"0"`;mode_contract.h:47 已明文纠正为 "the resolver retracts only its own
  flags"。)
- **模式命名规则**(mode_config.h:63-77):

  ```
  mode = (iso_bundle == 7) ? (strict_extras == 4 ? Strict : Isolated) : Merged
  ```

  只有**完整** bundle 才命名为 isolated/strict。merged 上的 per_env_exit overlay
  **绝不能被报告(或 checkpoint)为 isolated tier**(mode_config.h:68-72)。
- **`warn_if_incoherent()`**(mode_config.h:86-124)对三类半配置打 stderr 警告:
  partial strict(非 0/4 个 extras)= "determinism is NOT promised in this state";
  strict extras 无 `STIFF_PERENV_ALPHA` = "no known mode";
  partial isolated(非 0/7 且不是 exit_overlay)= "per-env fairness/isolation
  promises may not hold"。exit_overlay 识别式:
  `decouple_thresh && perenv_alpha && !bvh_envdet && !perenv_bvh &&
  !pergroup_kappa && !segmented_pcg && !perenv_par`(mode_config.h:111-113)。
- 稳定线无 `ModeConfig`(grep 无命中),旗标经散落的 `getenv` 消费
  (如 stable GIPC.cu 中 `getenv("STIFF_DECOUPLE_THRESH")` 有 17 处直接读点)。

## 1.5 `set_body_groups`:env 分组规则

【稳定线+phase-cd】

```python
Engine.set_body_groups(groups)          # engine.py:715-726
```

| 参数 | 类型 | 含义 |
|---|---|---|
| `groups` | 长度 = collision body 数的整数序列 | per-collision-body 的组(环境)id;ABD body 在前 `[0, n_abd)`,FEM 在后 |

规则(全部在 finalize 时校验,非法声明**抛错**而非静默降级物理):

1. 不同组(均 ≥0)的 body **绝不碰撞**——finalize 时折进 collision-skip 矩阵;
   空间平铺的环境无论 spacing 多小都保证隔离。
2. 非负 id 必须**稠密** `0..N-1`,且 **N ≤ 256**(`kEnvAlphaSlots = 256`,
   GIPC.cuh:664 → device_fem_data.cuh:20 `kGroupSlotCapacity`)。
3. `-1` 是 merged 模式的 wildcard(与所有 env 交互)。
4. **isolated / strict 要求每个 collision body 都属于某组**(不允许 `-1`)。

## 1.6 `set_vertex_env_ids`:逐顶点接触隔离

【稳定线+phase-cd】

```python
Engine.set_vertex_env_ids(env_ids)      # engine.py:728-743
```

| 参数 | 类型 | 含义 |
|---|---|---|
| `env_ids` | 长度 = 引擎顶点数的整数序列 | per-**vertex** env id;传空序列释放 |

与 `set_body_groups` 的区别:

| | `set_body_groups` | `set_vertex_env_ids` |
|---|---|---|
| 粒度 | per collision body | per vertex |
| 作用面 | 碰撞 skip 矩阵 + per-env 求解机制(κ/α/PCG 分组)的输入 | **仅接触过滤**:宽相跳过两顶点 env id 不同(均 ≥0)的接触对 |
| 空间要求 | 无(skip 矩阵保证) | 无需空间分离即可跨 env 接触隔离 |
| 典型场景 | 常规多环境平铺 | 单个 FEM body 的粒子横跨多个 env |
| 调用时机 | finalize 前声明 | **finalize 后任意时刻可调** |
| 共享几何 | `-1` wildcard | `id < 0` = 与所有 env 碰撞的共享几何 |
| 其他 | — | 底层 CUDA symbol 是进程级的,同时只有一个 Engine 可拥有 |

## 1.7 `per_env_exit` overlay

【稳定线+phase-cd】`Config(per_env_exit=True)` 是 merged 模式上的**合法叠加**
(productized 开关,engine.py:381-394):每个 env 按**自己的**准则收敛并被冻结/掩出
(资源释放),不与 batch 耦合;仅多 env 场景有意义。解析为 tracked-setdefault 四旗
(engine.py:230-235):

```
STIFF_DECOUPLE_THRESH  STIFF_PERENV_ALPHA  STIFF_PERENV_MASK  STIFF_PERENV_TELEM
```

`STIFF_PERENV_TELEM=0` 可显式退回零 D2H 的纯设备快路径(无遥测,engine.py:516-526)。
该 overlay **不承诺** collision/kappa/PCG 隔离(mode_config.h:68-72)。
生产推荐(CHANGELOG 0.8.5,稳定线 CHANGELOG.md:141-145):多环境生产配置
`per_env_exit=True, env_newton_iter_cap=100`。

## 1.8 per-env 遥测 API

【稳定线+phase-cd】

```cpp
std::vector<int> SimEngine::get_per_env_newton_iters() const;  // sim_engine.h:307
std::vector<int> SimEngine::get_per_env_status() const;        // sim_engine.h:310
```

```python
Engine.get_per_env_newton_iters() -> np.ndarray   # engine.py:1668-1672,长度 256
Engine.get_per_env_status() -> np.ndarray         # engine.py:1674-1677,长度 256
```

- **返回**:均为 256 槽 int 数组(`kEnvAlphaSlots`),拷贝语义,const、无参、不抛。
- `get_per_env_newton_iters`:每个 env 上次 solve 冻结时的 Newton 迭代号
  (converged / timeout / diverged;`-1` = 跑到循环末尾或 absent)。
- `get_per_env_status`:上次 solve 的 per-env 状态码:

| 状态码 | 含义 |
|---|---|
| 0 | active / absent |
| 1 | converged(per-env 冻结接受) |
| 2 | timeout(被 `env_newton_iter_cap` 强制冻结) |
| 3 | diverged(NaN,已检疫) |

- 数据每次 `solve_subIP` 开头重置(core/ipc_solver.inl:1256-1257)。
- **注意/陷阱**:数据只在 **host S1 遥测路径**运行时被填(路由条件 `_s1diag_` =
  `STIFF_PENV_STATS/A0_DUMP/S1_DEBUG/ALPHA_DBG` 之一存在,或
  `env_newton_iter_cap > 0`,或 `STIFF_PERENV_TELEM` 开;ipc_solver.inl:2033-2040)。
  纯设备快路径下不填,返回值保持 reset 值(-1 / 0)——该结论由代码路径推理得出,
  未跑程序验证(待核实)。实践口径:**要用遥测,配
  `per_env_exit=True` + `env_newton_iter_cap`(或 `STIFF_PERENV_TELEM=1`)**。
  **owner 决策(2026-09-08)**:两种契约保持分离——isolated/strict 只管物理隔离,故障隔离
  要 `per_env_exit=True` 另开(遥测走宿主 S1 路径,整帧图/gpu_rl 驻留通道随之失格,
  `frame_transaction.cu:416-423`,所以不能默认带上)。作为补偿,【仅 phase-cd】`finalize()`
  在 isolated/strict + 分组数>1 + 未开任何遥测时打印一次
  `[stiff-physics][WARN] … this mode is physical isolation only`(`engine.py`
  `_warn_isolated_without_quarantine`;六种组合的触发矩阵见 `examples/test_isolated_quarantine_warn.py`)。
- 相关配置:`SimEngineConfig::env_newton_iter_cap = 0`(int,默认 0=off;per-env
  Newton 迭代预算,到 cap 仍活跃的 env 被强制冻结为 status 2,其他 env 不受影响;
  仅 host per-env 路径;sim_engine.h:73-76,pybind bindings/pystiffgipc.cu:130)。

## 1.9 隔离契约:quarantine / revive

契约文本的单一权威是 `StiffGIPC/multienv/isolation.cuh`(:1-28,【仅 phase-cd 布局】;
稳定线机制在单体 GIPC.cu 中,**有 quarantine、无 revive**)。

- **产品铁律**:病态 env 绝不得扰动健康 env(isolation.cuh:4-5)。
- **可用性门 `GIPC::perEnvIsolationLive()`**(isolation.cuh:8-13;实现
  isolation.cu:78-85):
  multi-env groups 已声明(`m_active_group_count > 1`)**AND**
  host 遥测路径开(`env_newton_iter_cap > 0` 或 `STIFF_PERENV_TELEM`)**AND**
  per-env alpha 开(`STIFF_PERENV_ALPHA`;isolated/strict 模式会设)。
  **纯设备快路径蓄意没有 NaN 防御:隔离承诺依赖此门。**
- 状态存储:`m_env_status` 每次 solve 重置(0 running / 1 converged /
  2 timeout-frozen / 3 quarantined);`m_env_quarantined` 是**持久**旗标(跨帧存活),
  带设备镜像供 direction-zero kernel(isolation.cuh:14-17)。
- 被检疫 env 的惰性语义(isolation.cuh:18-22):位置冻结(α==0 逐字保持最后接受状态)、
  每迭代 Newton 方向清零、其 body 从 ground 检测与 ground-CCD α 中移除(skip 表)、
  solve 循环中槽位钉状态 3。**init 时刻的违规仍然抛异常。**
- 三个 mid-frame 检疫入口(isolation.cuh:23-27):帧首 flag-only 不可行探针
  (teleport 可在帧间使 env 不可行,而 CCD fail-fast 先于任何 detection 运行);
  detection 时降级(`throwIfGroundDistanceInvalid`);solve_subIP 中 post-PCG
  非有限方向扫描。

### `GIPC::quarantineEnv`【稳定线+phase-cd】

```cpp
bool GIPC::quarantineEnv(int env, int vertex, double distance);   // isolation.cu:112-166
bool GIPC::quarantineEnvOfVertex(int vertex, double distance);    // isolation.cu:168-175
```

- **参数**:`env` 组 id(`0 <= env < 256`);`vertex` 触发顶点(≥0 = ground 不可行,
  <0 = 非有限 Newton 方向);`distance` 诊断用距离。
- **返回**:`true` = 已(或已经)检疫;`false` = per-env 机制不 live(调用者保留
  throw 权)。**merged 模式无隔离机制 → 恒 `false`**(isolation.cu:103-111)。
- 首次检疫打印 `[per-env][QUARANTINE] env %d ... healthy envs continue`。
- 稳定线对应物:stable GIPC.cuh:208-209、GIPC.cu:10411+。

### `GIPC::reviveEnv`【仅 phase-cd】

```cpp
bool GIPC::reviveEnv(int env);            // isolation.cu:184-223
```

- quarantine 的逆,用于 **episode reset**:清 host+device 旗标、撤销 ground-skip
  标记、`m_env_status[env] = 0`(reset 后立刻查询 status 即反映复活)。
- **复活是自纠错,不是走私通道**:仍坏的 env 一帧内被重新检疫(ground:铁律帧首探针;
  非有限方向:每迭代检测器)。
- **返回**:未检疫的 env 返回 `false`;前置门同 quarantine。
- **自动调用点**(用户一般不直接调):
  - `SimEngine::teleport_fem_vertices`:teleport 触及被检疫 env 视为 episode reset,
    对与被 teleport 顶点区间重叠的 FEM body 的组调 `reviveEnv`
    (engine_modules/04_teleport_checkpoint.inl:95-110),之后强制重建帧入口 pair 集
    (`invalidateRefitTopology + buildBVH + buildCP`,04:112-123);
  - `SimEngine::teleport_abd_bodies`:同理(03_step_getters_export.inl:2554-2564)。
- 稳定线**没有** `reviveEnv`(grep 无命中)——稳定线上被检疫 env 无法在同进程内复活。

### 帧首检疫扫描【仅 phase-cd 形态】

`quarantineGroundInfeasibleAtFrameStart()`(isolation.cu:230-264):`STIFF_SKIP_GRND`
可跳过;循环至多 `m_active_group_count+1` 轮(atomicMin 每轮只报一个赢家 → 每轮检疫
一个新 env);不可归因时 break 留给常规 throw 站点。

### per-env 求解机制速览(供排查用,机制细节)

- **per-env line-search α**(gipc_modules/11_perenv_machinery.inl):5 槽 scratch
  `[ground, narrow-self, refined-self, surface-cfl-max, all-vertex-max-move]`
  (11:119-120);设备决策 `_per_env_alpha_compute`(11:117-172):

  ```
  ta_g   = min(ground_g, narrowSelf_g)
  a_g    = min(ta_g, 0.5 * sq / hmx)            (CCD 存在且 hmx>0)
  精化门  gate_lhs > 2*gate_rhs  →  a_g = max(min(ta_g, hr*ccd_size), acfl_g)
  冻结阈  thr_g = vtol_dt > 0 ? vtol_dt : ntol_dt * sqrt(env_bbox2[g])
  decouple && nmx_g < thr_g  →  a_g = 0        (按全部顶点冻结)
  ```

  与 host 循环逐位一致(同公式、无规约)→ 保持 strict 跨 env 逐位一致
  (11:121-123)。不变量(已验证):`min_g(m_env_alpha) == 全局可行 α`;
  N=1 时 `m_env_alpha[g0] == 全局 α`(ipc_solver.inl:2002-2005)。
- **per-env S3 line search**(ipc_solver.inl:266-362):每 env 按自己的 CCD 可行 α
  步进后强制 **per-env 能量下降**(E_g 上升的 env 单独减半重试,maxBT=8;per-env
  能量分解已验证 `Σ_g E_g == global` 至机器精度);无法下降 → 回退标准 uniform search。
  帧内 S3 backtrack 只把非零 α 减半(**从不 →0**),冻结集在 S1 后即最终
  (ipc_solver.inl:2276)。
- **per-env κ**:`initKappa` 的 per-env 收尾
  `K_g = clamp(max(-gsum_g/gsnorm_g, suggested), 0, kmax)`(11:259-269);
  postLineSearch 的设备侧 per-group κ 加倍只对命中 close-set 且未冻结的组,capped
  `kappaMax`(11:238-254)。冻结 env 连接触参数一起冻结——否则 strict env0 会依赖
  batch size(11:243-246)。
- **Newton 退出(per-env)**:`decouple_thresh && all_env_frozen && ...` → break
  (ipc_solver.inl:2264-2272);动机:全局 gradVanish 可能在某 env 仍欠收敛时触发
  (ipc_solver.inl:1311-1316)。
- **S4 active mask**:`active = (alpha != 0)` 设备派生(11:227-232);被 mask env
  的 RHS 清零、SpMV 跳其 triplets(ipc_solver.inl:1281-1298);固定节奏 all-active
  recheck 检测 bounce-back → strict/batch-invariance 安全(11:216-221)。
- **跨 env 诊断**:`STIFF_XENV`/`STIFF_XENV_ID`(11:656-744)按 env-local id 摆平
  buffer 报告 `max |env0-env1|` 与最坏 lid——证明 mirror(call#0==0)与追杀跨 env
  泄漏的工具。

---

# 2. 可复现性配方

## 2.1 想逐位复现,该怎么配

**基础配置**(【稳定线+phase-cd】):

```bash
STIFF_MULTIENV_MODE=strict python your_script.py
# 或
eng = Engine(Config(multienv_mode="strict", ...))
```

strict = isolated 7 旗 + 确定性 4 旗(`STIFF_EE_CANON` 规范 EE 发射序、
`STIFF_EE_DETGATE` 确定性 EE 去重门、`STIFF_CCD_CANON` 规范 CCD 发射序、
`STIFF_SPMV_DET` 序无关 binned SpMV)。

**多环境 strict 的布局三件套**(v0.8.5 mode-driven 固定策略,
examples/replay_foldshirt_multienv.py:176-188):

1. `SPACING=0`:所有 env **物理共址**(消除世界偏移的浮点差异);
2. finalize **之后** `eng.native.set_env_offsets(per_group_xyz)`:把分隔网格
   (如 4.0 间距)只喂给 BVH 广义相——BVH 仍高效分隔,数值在局部坐标;
3. 显示期偏移(渲染时平移,物理坐标不动;`CASE39ME_DISPLAY_SPLIT=1`,仅 phase-cd
   的 GUI 便利)。

**必须同时满足的边界条件**:

- **固定场景 + 固定 batch layout**(engine.py:187-190):batch 大小、加载顺序、
  分组声明全部固定。cross-batch-size identity 是验证目标而非契约。
- **同一进程状态干净**:`STIFF_*` 首读即缓存,一个进程一个模式(§1.3 进程锁,
  仅 phase-cd 强制;稳定线上是使用者纪律)。
- **同一 wheel / 同一编译配置**:dlto 换锚史(../RELEASE_NOTES_v0.8.6-rc1.md:132-137)
  说明改变链接优化会改变逐位值。跨**架构**逐位已证明(strict 锚
  `0544461bd82123ae` 在 4090 sm_89 与 A800 sm_80 同值;dlto 前世代锚为
  `f7fb5a786c2d7935`)。
- **不要开整帧图**:strict / `STIFF_EE_DETGATE` / `STIFF_CCD_CANON` 开着时整帧图
  **资格拒绝**(frame_transaction.cu:384-388)——strict 不图化是 2026-07-29 的用户
  决定(容量网格归约会重排求和序 = 换锚战役,../GPU_NATIVE_RL_PLAN.md:44)。
- **不要动 strict 栈的确定性旋钮**:`STIFF_SEG_WARP=1` 对 strict 强开 warp 预求和
  会**破坏确定性**(仅 A/B 用,pcg_solver.cu:1226-1244);`STIFF_GRAPH_INGRAPH_RETRY=1`
  注入 ~1e-6 run-to-run 扰动(frame_transaction.cu:4491-4503)。
- **strict 会自动关掉持久摩擦锚**【仅稳定线 v0.8.5.4+】:v0.8.5.4 的
  `friction_anchor` 虽默认开,但引擎一看到 strict 签名(`STIFF_SPMV_DET`)就把它
  按回 legacy 并打印 `[fric-anchor] strict mode: friction_anchor suppressed for
  batch invariance`(stable@c0339c8 GIPC.cu:9804-9822)。理由是**批不变性**:锚与
  `absolute_epsv=1e-4` 组合后,收紧的锚封顶半径 `eps = epsv·h`(≈1 µm)把摩擦能量
  推到线搜索能量求和的 ulp 比较边界上,而该求和的规约形状随 env 数 N 变——一次
  accept 判定翻转即被混沌放大(foldshirt strict:前 17 帧逐位相同,第 18 帧一步内
  96% 顶点分歧),表现为 env0 逐位 N=2 ≠ N=4。二分定位:**只开 epsv 绿、只开 anchor
  绿、两者同开红**,故 strict 只交易掉 anchor(`absolute_epsv` 保留——它本身批安全);
  merged / isolated 保持两个默认。`STIFF_FRIC_ANCHOR=1` 可在 strict 强制开回,
  但那时不要再指望 batch 不变性。根治(N 不变的能量归约)记在 0.8.6:工程线未合入的
  移植分支 `port/friction-anchor-086` 已有 `9fd2905`"order-free (binned) per-env
  line-search energy;anchors default-ON in strict",可作移植时的参照。
- 自检工具:`examples/test_strict_quadgate.py`(strict 五门逐位:run-to-run、
  跨 env、batch 不变性等;自足无金锚)。

## 2.2 merged 为何不承诺

- merged 的接触发射与规约依赖原子操作到达序(mode_contract.h:23),**本身就不是
  run-to-run 确定的**:实测 release 求解器自身在 towel 场景从帧 2 起分歧 2.2e-14,
  帧 119 放大到 1.1e-4(../A800_ALLEXAMPLES_TIMING_2026-08-01.md:294-339,C6-w 四实证)。
- 因此 merged 通道的回归门禁用**基线自噪声包络**
  (`budget = max(4×noise, 1e-11×scale)`)而非逐位;逐位门禁只住在
  `STIFF_SPMV_DET` 确定栈上(数学性质)。
- `STIFF_SKIP_ZERO_DEPOSIT`(默认开)在 merged 栈上引入 ~2 ULP 扰动(在自噪声内),
  在 det 栈上**位级中性**(converter.cu:470-500)——即它不改变 strict 的承诺。
- isolated 同样不承诺逐位(mode_contract.h:30):它只承诺物理隔离与公平,
  确定性内核不在其 bundle 内。

## 2.3 相关的复现工具

| 工具 | 用途 | 适用线 |
|---|---|---|
| `save_checkpoint`/`load_checkpoint` | 跨进程精确续跑(v2 格式续跑 delta ~5e-17,见 §5) | 【仅 phase-cd】(稳定线 legacy 格式无此精度承诺) |
| `CASE39ME_DUMP_VERTS` | 末态顶点 dump,strict 共址下逐位可比 | 示例层 |
| `STIFF_VERT_HASH=1` | 逐帧顶点 md5(umi_finray_lib headless) | 示例层 |
| `scripts/frame_graph_gate.py` / G1 bitwise anchor | 门禁级锚校验 | 【仅 phase-cd】 |

---

# 3. GPU 驻留 RL 全家桶【仅 phase-cd】

## 3.0 概念与契约

"GPU-native RL" 的定义(../GPU_NATIVE_RL_PLAN.md:3-26):稳态转移
`device action → sim step → device obs/reward/done → device reset → next action`
全部留在 CUDA 流上;正常 RL 步**不得**含 H2D/D2H payload 拷贝、任何宿主等待、
决定控制流的宿主决策、拓扑相关分配或图重捕获。setup / 诊断 / checkpoint / 渲染
允许跨宿主边界。执行架构定稿是"一个表面(Isaac 式 `step()`)、两个内核":
step 通道(宿主帧骨架 + PCG 自发射图岛,大帧最优)与 **residency 通道**
(整帧图 + 设备 resizer + 多帧连发,微步最优);**模式切换 = `prepare_gpu_rl()` /
`end_gpu_rl()` 显式声明,永不按帧自动猜测**
(../SIMULATOR_EXECUTION_DESIGN.md:8-27)。

捕获审计是 fail-closed 的:device-native 图**硬性要求 host==0 && h2d==0 && d2h==0**,
违反即 throw(frame_transaction.cu:1957-1990)。实测证据:contact+friction 铰接一帧图
1057 节点、40 步 nsys 窗内 0 H2D / 0 D2H / 0 同步(4090 与 A800 各一次独立复核,
../GPU_NATIVE_RL_PLAN.md:53-128;图节点数依场景而变:551 无碰撞版 / 1057 / 1089
D4 场景)。

## 3.1 prepare 族

### `Engine.prepare_gpu_rl()`

```python
def prepare_gpu_rl(self) -> None          # engine.py:1205-1223
```

- 捕获可复用的**一帧** GPU-native RL 图(等价 frame_count=1 的 device_native episode)。
- **[self-contained prepare]**:无需任何环境旋钮——进程未开 `STIFF_FRAME_GRAPH` 时,
  引擎在 `LayoutForceOnScope` 强制容量布局下**内部多跑一帧 `step()` 训练容量档**再捕获
  (此时 prepare 会使仿真前进一帧;03_step_getters_export.inl:381-419)。
  episode 捕获本身即碰撞进图的资格(`m_episode_capture`),不需要
  `STIFF_C4_COLLISION_GRAPH`(frame_transaction.cu:319-332)。
- **前置**:必须先跑过至少一次同步 `step()`(`m_total_frames==0` →
  `logic_error("one synchronous warm-up step is required before graph capture")`,
  frame_transaction.cu:2794-2797);warm-up 应带**有代表性的接触峰值**,
  容量档按 warm-up 实测训练。
- **抛错**:场景资格不符 → `"[episode-graph] unsupported scene: <reason>"`
  (资格 = `full_graph_eligible(allow_abd=true)`,见 §3.6);`STIFF_DRIVE_SUBSTEP>1`
  → runtime_error(非驻留,frame_transaction.cu:2809-2812);另一图事务活跃 → throw。
- prepare 之后:`step()` 变**薄异步入队**(§3.5),或直接经设备 ABI 驱动
  (§3.3/§3.4)。`launch_episode_async` 被锁定直至 `end_gpu_rl()`。

### `Engine.prepare_gpu_rl_episode(frames)`

```python
def prepare_gpu_rl_episode(self, frames: int) -> None    # engine.py:1225-1235
```

- 捕获**多帧** device_native episode(frames > 1,否则 invalid_argument):
  一次 `launch_gpu_rl_episode_async` 发射,图的外层条件循环消费全部 frames,
  无每帧宿主发射。动作 slab 直接在设备上写。

## 3.2 launch / 查询 / 结束族

| API(Python,engine.py:1237-1290) | 语义 |
|---|---|
| `launch_gpu_rl_async(cuda_stream: int = 0)` | 入队一步,无宿主等待。`0` = 引擎的 per-thread default stream |
| `launch_gpu_rl_episode_async(cuda_stream: int = 0)` | 发射整段多帧 episode 一次 |
| `gpu_rl_prepared() -> bool` | 图是否已捕获武装(纯 host 查询) |
| `gpu_rl_ready() -> bool` | 非阻塞完成查询(cudaEventQuery);**不属于稳态 RL 环** |
| `synchronize_gpu_rl()` | 阻塞等最后一步完成(debug / teardown / 非默认 torch 流前) |
| `end_gpu_rl()` | 同步、销毁图与资源、恢复常规 host-driven step();销毁前**先 drain 整条流**(GPU-native 消费者可能在 completion event 后追加了 obs/reward/policy 核,frame_transaction.cu:2755-2777) |

**流纪律(设备 ABI 的一部分)**:

- 重复发射**必须用同一 CUDA 流**;in_flight 后换流 → `logic_error("repeated
  launches must use the same CUDA stream; end_gpu_rl() before changing streams")`
  (frame_transaction.cu:3225-3287)。
- 动作写与观测读必须发在同一流上(流序替代宿主栅栏)。
- **代际检查**:任何可能移动设备指针的 realloc bump `pcg_buffer_generation()`;
  发射前比对,变了 → `runtime_error("[gpu-rl] graph buffers changed; prepare
  the graph again")`(单帧 launch,frame_transaction.cu:3235-3237;episode 发射
  路径报 "... prepare the episode again",frame_transaction.cu:3185-3187 与
  3302-3304;代际机制 linear_system/utils/pcg_capacity_mode.h:20)。

## 3.3 reset 族

```python
launch_gpu_rl_reset_async(cuda_stream: int = 0)                       # engine.py:1268-1277
launch_gpu_rl_reset_masked_async(env_mask_device_ptr: int, cuda_stream: int = 0)
```

- **整场景 reset**:纯 D2D 重放 prepare 时刻的提交态快照(FEM 4 数组 + ABD q 族),
  零宿主同步;下一步在自己的序幕里重建碰撞状态。**设备帧计数器故意不清**
  (episode 记账是调用方政策;frame_transaction.cu:3550-3601)。
- **masked reset**(per-env 选择性):`env_mask_device_ptr` 指向设备 int32 数组
  (按 env 组 id 索引),非零项的 env 回到 prepare 快照,其余不动;mask 可由设备端
  done-flag kernel 写 → **零宿主传输的选择性重置**。组=-1(无主)永不经掩码路径重置;
  mask 空指针 → invalid_argument;无 p2g 且有顶点 → logic_error
  (frame_transaction.cu:3506-3548)。
- 多环境配套设备句柄(ABI 内):`point_to_group`(int32/顶点,-1=wildcard)、
  `env_quarantined`(int32/env)、`env_count`。
- 注意:GPU-RL reset 走**快照重放**,与 host 侧 `teleport_*` reset(§1.9,自动
  reviveEnv + 重建 pair 集)是两条不同的 reset 通道;驻留循环内用前者。

## 3.4 设备 ABI 与零拷贝 torch 契约

### `Engine.get_gpu_rl_device_abi() -> dict`(engine.py:1292-1311)

返回裸设备指针与审计过的图 ABI。键表(docstring + 代码亲验):

| 键 | 内容 |
|---|---|
| `revolute_actions` / `prismatic_actions` | 打包 `(joints, 3)` float64 动作 slab:每 joint 三元组 **{target, strength_ratio, external torque/force}**(紧排布局由 `static_assert` 钉死,frame_transaction.cu:2931-2939) |
| `positions` / `velocities` | `(vertices, 3)`(一帧 API)或 `(frames, vertices, 3)`(多帧)float64,**引擎内部顶点序** |
| `statuses` | 每捕获帧一个 `status_bytes` 大小的 `FrameStatus` 包 |
| `frame_counter` | int64 设备帧计数器(图内自增) |
| `episode_frame_count` | 捕获帧数 |
| `graph_nodes` / `graph_h2d` / `graph_d2h` | 捕获审计计数(device-native 恒 h2d=0/d2h=0) |
| `joint_observations` / `joint_observation_count` | 图自身刷新的 float64 关节观测块:每 revolute 驱动关节 {angle, rate},随后每 prismatic 驱动关节 {displacement, rate};count = 2·revolute + 2·prismatic(frame_transaction.cu:2976-2977) |
| `point_to_group` / `env_quarantined` / `env_count` | 多环境句柄(§3.3) |
| `revolute_joints` / `prismatic_joints` / `vertices` / `status_bytes` | 形状元数据(`gpu_rl_tensors` 消费) |

- **指针有效期**:至 `end_gpu_rl()` / reset(重 prepare)/ 引擎析构
  (../GPU_NATIVE_RL_PLAN.md:130-156)。
- **顶点序陷阱**:obs slab 由图内 `episode_store_frame` 直接从引擎内部数组拷入
  (frame_transaction.cu:1864-1877)——**engine 内部序,无 metis 解扰**;host getter
  (`get_vertices()` 等)输出 input 序。MAS 预条件(`preconditioner_type != 0`)对
  FEM/布料做过 metis 重排时两路混用会错位;ABD 顶点与无 MAS 场景是恒等置换,不受影响。

### `Engine.gpu_rl_tensors() -> dict`(engine.py:1313-1373)

零拷贝 **torch** 视图:用 `__cuda_array_interface__` v3 包装 ABI 裸指针,
`torch.as_tensor(..., device="cuda")`,**无 `.numpy()`、无拷贝**。键:

| 键 | dtype/形状 |
|---|---|
| `actions_revolute` / `actions_prismatic` | float64 `(frames, joints, 3)`;一帧 API squeeze 成 `(joints, 3)`。**直接往里写 target,替代 memcpy** |
| `positions` / `velocities` | float64 `(vertices, 3)`(多帧捕获带前导 frames 维) |
| `joint_observations` | float64 `(joint_observation_count,)` |
| `statuses` | uint8 `(frames, status_bytes)` |
| `frame_counter` | int64 标量(shape `(1,)`) |

- **流纪律**:引擎在 CUDA per-thread default stream 入队;torch 默认(legacy)流
  隐式有序;**若使用非默认 torch 流,读观测前必须 `synchronize_gpu_rl()`**。
- torch 是可选依赖(函数内局部 import)。

## 3.5 `step()` 自动分流(表面统一)

`SimEngine::step()`(03_step_getters_export.inl:1-60):`gpu_rl_graph_prepared()`
为真时,step() 变成"薄入队"——把 host 侧 joint 目标打包为
`RevoluteDrivingControlPacked{target_angle, strength_ratio, ext_torque}` /
`PrismaticDrivingControlPacked{...}`,`cudaMemcpyAsync` 到动作 slab(绑定流上,异步),
然后 `launch_gpu_rl_graph_async(bound)` 返回——**不等待**。流亲和:外部曾用
`launch_gpu_rl_async` 绑定流则沿用,否则 `cudaStreamPerThread`。
实测表面统一零开销:自动薄发射 3.14 ms/步 ≈ 手工 launch 3.18 ms
(../SIMULATOR_EXECUTION_DESIGN.md:170-178)。

episode(非 gpu_rl)在飞时 `step()` 抛
`LifecycleError("step() is unavailable while an episode graph is in flight")`。

## 3.6 资格与拒绝

prepare 走 `full_graph_eligible(..., allow_abd=true)`(frame_transaction.cu:251-439),
不合格**明确抛错不偷跑**。常见拒因(reason 字符串在代码中逐条可查):

| 拒因 | 说明 |
|---|---|
| capture 不兼容诊断旋钮 | `STIFF_MIRROR_AUDIT / SLOT_AUDIT / PHASE_TIME / KSUM / MAS_DUMP / MAS_FUSE_VALIDATE / STACK_DIAG` 任一开着(frame_transaction.cu:276-292) |
| FEM pins | `n_fem_pins != 0`(FEM-to-ABD pin) |
| 移动边界 / host-owned soft targets | `m_update_boundary`;非设备驻留 soft target(设备驻留 stitch 弹簧合法) |
| semi-implicit / 多子步动画 | `semi_implicit_enabled`;`animation_subRate < 0.999999` |
| strict 确定性栈 | `Strict` 模式、`STIFF_EE_DETGATE`、`STIFF_CCD_CANON`(`STIFF_SPMV_DET` 与 `STIFF_EE_CANON` 被准入) |
| isolated 附加条件 | 需 `STIFF_C5_ISOLATED_GRAPH=1` + 碰撞在图内 + 已声明 env 组;`STIFF_PERENV_TELEM`/`env_newton_iter_cap>0`(host S1 诊断路径)→ 拒 |
| merged 上部分 per-env overlay | 任一 per-env 旗单独开 → 拒(只有完整 bundle 或纯 merged 合法) |
| `STIFF_DRIVE_SUBSTEP>1` | 驱动 ramp 非设备驻留 |
| ABD tier 未训练 | warm-up 未跑出 ABD final assembly tier |

**容量平稳性判据**(结构性限制,不是速度问题):接触**逐帧升级**的轨迹
(如 foldshirt 抓握)会击穿 prepare 训练档——实测第 15 帧 auto-prepare 后
25/25 帧 `OVF_TRIPLETS` 失败(fail-closed 正确拒绝提交坏帧)。
**真实大帧操作轨迹必须走 step 通道**(../SIMULATOR_EXECUTION_DESIGN.md:81-90)。

## 3.7 `STIFF_AUTO_PREPARE_AT` 自动恢复协议

【仅 phase-cd,诊断/基准用】`STIFF_AUTO_PREPARE_AT=k`:在本 Engine 的第 k 次
`step()` 前自动调 `prepare_gpu_rl()`(自足 prepare);之后每步 thin-route + 同步
(保持逐帧计时诚实);不合格场景打印拒因一次并留在宿主路径(engine.py:1005-1035)。

自动 **OVF 恢复协议**(engine.py:1036-1126,边界协议 §5 的参考实现):

1. 每步同步后经 ABI 直读一帧 `FrameStatus` 包(ctypes cudaMemcpy,读
   result/invalid_bits/error_code 三字段);
2. `result==FRAME_RETRY_REQUIRED && error_code==ERR_CAPACITY`(且恢复次数 <64)时:
   `end_gpu_rl()` → 临时把 `STIFF_FRAME_GRAPH / STIFF_FRAME_FULL_GRAPH /
   STIFF_C4_COLLISION_GRAPH / STIFF_C6_ABD_STEP_GRAPH` 四旗设 "1",
   用**一个 step 事务帧**重放失败帧(它回滚、按真实 required 增长容量档、完成物理),
   随后恢复原环境值 → `prepare_gpu_rl()` 在长大的档上重捕获 → 继续 thin;
3. 打印 `[auto-prepare] OVF recovery #N ...`;其他失败打印
   `frame FAILED (no recovery)`;atexit 打印健康统计
   `[auto-prepare] health: ok/total frames`。

配套:`STIFF_MS_DUMP=path` 逐步墙钟 ms 落盘(np.save,atexit)。

## 3.8 观测/审计辅助 getter(SimEngine 层)

`get_gpu_rl_{revolute,prismatic}_actions_device_ptr` /
`get_gpu_rl_{positions,velocities,statuses,frame_counter,joint_observations}_device_ptr` /
`get_gpu_rl_joint_observation_count` / `get_gpu_rl_status_size_bytes`
(= `sizeof(FrameStatus)`)/ `get_gpu_rl_graph_node_count/h2d_count/d2h_count/episode_frame_count`
(03_step_getters_export.inl:514-616)。非 device_native 或未 prepare 时一律
`logic_error("[gpu-rl] no GPU-native RL graph is prepared")`
(frame_transaction.cu:3406-3468)。

---

# 4. 异步 episode 七件套【仅 phase-cd】

open-loop episode 通道:动作**整段预上传**、观测 pinned **双缓冲**异步拷出。
定位:回归 / 吞吐实验用;它有 H2D/D2H(≥4 个 D2H 观测节点),
**不满足** GPU-native 定义(../GPU_NATIVE_RL_PLAN.md:48-51)。闭环 RL 用 §3。

## 4.1 `launch_episode_async`

```python
def launch_episode_async(self, frames: int,
                         revolute_actions=None,
                         prismatic_actions=None) -> None    # engine.py:1154-1172
```

| 参数 | 类型 | 含义 |
|---|---|---|
| `frames` | int > 0 | episode 帧数 |
| `revolute_actions` | float64 `(frames, revolute_joints, 3)` 或 None | 每 joint 三元组 **[target, strength, external torque]** |
| `prismatic_actions` | float64 `(frames, prismatic_joints, 3)` 或 None | 每 joint 三元组 **[target, strength, external force]** |

- **动作张量语义**:`(frames, joints, 3)`,第 3 维依次是驱动目标、强度比、外力矩/外力
  ——与 GPU-RL 动作 slab 同一打包(§3.4)。图内由设备端按 `frame_index` 索引动作序列,
  逐帧**不回宿主**(`enqueue_episode_driving_targets`,frame_transaction.cu:1707-1886)。
- **校验与抛错**(03_step_getters_export.inl:126-231):未 finalize / gpu_rl 模式
  (提示先 `end_gpu_rl`)/ 已在飞 → `LifecycleError`;`frames<=0`、关节维度与场景
  不符、非空动作缺指针 → `invalid_argument`;形状尺寸溢出 → `overflow_error`;
  逐元素非有限 → `invalid_argument`。
- **前置**:先跑一次常规 `step()`(训练 lazy CUDA workspace);场景资格同 §3.6。

## 4.2 其余六件

| API | 签名/语义 |
|---|---|
| `episode_in_flight() -> bool` | 是否还有未 finish 的 episode(engine.py:1174-1176) |
| `episode_observation_ready(slot) -> bool` | 非阻塞查询 pinned 观测槽 0/1 是否就绪(volatile 哨兵 + cudaEventQuery 双检,frame_transaction.cu:3635-3658) |
| `wait_episode_observation(slot)` | 只等一个观测槽及其 CUDA event 围栏(自旋 yield + eventSynchronize) |
| `get_episode_observation(slot)` | 返回就绪槽:`first_frame`、`positions`/`velocities`(`(slot_frames, vertices, 3)`)、逐帧 `statuses`。**host 读回按 metis perm 解扰为 input 序**(与 `get_vertices()` 一致;03:268-325) |
| `get_episode_attempted_frame_count() -> int` | 已发布观测槽覆盖的尝试帧数 |
| `finish_episode() -> int` | 等终局槽,**返回成功帧数**,`step_count += successful`;并把**最后一帧成功执行的动作回写** host 侧 joint controls(保证 host 镜像与设备终态一致;03:336-377) |

## 4.3 双缓冲结构与失败语义

- `split_frame = (frames+1)/2`:图组织为"两段 WHILE + 两个 root 观测出口"
  ——CUDA 不允许 host-queryable external event node 进 conditional body
  (../PHASE_C_FRAME_GRAPH_PLAN.md:159-165)。slot 0 覆盖 `[0, split_frame)`,
  slot 1 覆盖其余;先等 slot 0 可实现半程流水。
- 每帧图内 `episode_store_frame` 写入 slab 并序列化 per-frame `FrameStatus`;
  某帧失败 → `episode_tail` 停止推进(keep_running=0),后续帧不执行;
  `finish_episode` 逐帧累加成功帧(遇非 OK 停),`m_last_frame_status` = 最后尝试帧。
- exec 复用:同 frame_count/顶点数/关节数/device_native/代际时只重传动作即返回
  (frame_transaction.cu:2872-2893)。
- **episode 无逐帧 fallback**:必须 overflow-proof 烘焙(C6-l;GIPC.cuh:1041-1046)
  ——容量溢出即整段失败,回 step 通道重放(对照 §3.7 的自动恢复)。

---

# 5. checkpoint 与 get_frame_status

版本标注:`save_checkpoint` / `load_checkpoint` 本身【稳定线+phase-cd】——
稳定线存在且可用(legacy 无版本格式,stable GIPC.cu:16593-16655,见 §5.1 末
"稳定线对照");**v2 格式、~5e-17 续跑精度与 `get_frame_status`【仅 phase-cd】**。

## 5.1 `save_checkpoint` / `load_checkpoint`

```python
Engine.save_checkpoint(path)    # engine.py:1396-1407;未 finalize 抛 LifecycleError
Engine.load_checkpoint(path)    # engine.py:1409-1418
```

**v2 版本化格式**(magic `"STIFFCP2"`,version=2;checkpoint/checkpoint_io.cu):

| 部分 | 内容 |
|---|---|
| header(120 B 定长) | magic、version、header_size、endian_marker(0x01020304)、scalar_size(8)、mode、mode_flags、state_flags、constitutive_model(1=SNK1/2=SNK2/3=ARAP,编译期)、vN、nb、env_count、tet/tri/bending/soft 计数、scene_signature、payload_size、**crc64**(checkpoint_io.cu:33-53, 867-888) |
| payload | current / previous(o_vertexes)/ velocity / xTilta / targetVert + ABD 五组 12·nb double(**q, q_prev, q_v, q_tilde, ext_force**)+ Kappa + 按 state_flags 可选:per-group kappa、env_active、quarantined、direction_nan、ground_skip 表 + recheck_counter + m_total_frames(:731-860, 1001-1048) |

- **保存端**:全量 `require_finite`(非有限 → `CheckpointError`);原子写
  (写完才替换目标文件)。
- **载入端**:先读 header;**legacy magic `0x53544B50` 显式拒收**并提示重新生成;
  校验 version/endian/scalar/未知 state 位/本构模型/mode 配置/拓扑计数/env 布局/
  payload 长度/CRC64/scene_signature,**全部通过后才动第一个设备字节**
  (:929-934, 1080-1081);随后恢复全部状态含 per-env 镜像与 ground-skip 表。
- **[frame-entry pair set] 载入后重建帧入口配对集**(:1224-1242):每帧第一次
  Newton 跑在上一帧最后一次 line-search buildCP 继承的配对集上——这是刻意
  **不入 checkpoint** 的跨帧状态;从刚恢复的位置重建可精确重构源的入口集
  (续跑 delta 从 ~5e-6 降到 **~5e-17**,实测接触场景);附带
  `invalidateRefitTopology` 强制一次树质量重建。摩擦锚与 Kappa 同一
  "restore 时重构"原则(../PHYSICS_VALIDATION.md:42-76)。
- **错误类型**:`gipc.CheckpointError`(checkpoint_io / checkpoint_format)。
- 门禁口径:续跑 strict 逐位、merged 1e-12(../PHYSICS_VALIDATION.md)。

**稳定线对照**【仅稳定线】:legacy 无版本格式(magic `0x53544B50`,
stable GIPC.cu:16593-16655):只存 vN/nb + 4 组顶点 double3 + ABD 三组
(q/q_prev/q_v,**无 q_tilde/ext_force**)+ Kappa + total_Frames;
**无校验和、无场景签名、无原子写**;打不开/不匹配只 printf 返回(**不抛错**);
载入后**不重建配对集**。两代格式互不兼容(v2 拒 legacy,legacy 读 v2 报 magic
MISMATCH)。

## 5.2 `get_frame_status` 与 `FrameStatus`

```python
Engine.get_frame_status()       # engine.py:1422-1429;返回 FrameStatus 对象
```

`frame_fsm::FrameStatus`(frame_fsm/frame_status.cuh:80-132,`alignas(16)`,
`static_assert(sizeof <= 256)`——帧尾**唯一**的小 D2H 包)。pybind 导出全部标量字段
为 readonly(bindings/pystiffgipc.cu:38-93;`contact_class_count[4]` 数组**未**导出)。

**枚举**(frame_status.cuh:13-78):

| 枚举 | 值 |
|---|---|
| `FrameResult` | `FRAME_OK=0, FRAME_RETRY_REQUIRED=1, FRAME_FATAL=2, FRAME_RUNTIME_ERROR=3` |
| `FramePhase` | `IDLE=0, FRAME_BEGIN=1, ASSEMBLY=2, PCG=3, NEWTON_DECIDE=4, CCD=5, LINE_SEARCH=6, POST_LS=7, COMMIT=8, ROLLBACK=9` |
| `FrameInvalidBits`(低位物理无效) | `INV_CCD_GROUND=1<<0 … INV_LS_BUDGET=1<<6, INV_START_INTERSECTING=1<<7, INV_NAN_STATE=1<<8` |
| `FrameInvalidBits`(高位容量溢出) | `OVF_DCD_PAIRS=1<<16, OVF_CCD_PAIRS=1<<17, OVF_TRIPLETS=1<<18, OVF_UNIQUE_BLOCKS=1<<19, OVF_MAS_CLUSTERS=1<<20` |
| `FramePathFlags` | `PATH_GRAPH_REQUESTED=1<<0, PATH_GRAPH_ACTIVE=1<<1, PATH_LEGACY_FALLBACK=1<<2, PATH_HOST_PHASE_BRIDGE=1<<3, PATH_FULL_CONDITIONAL_GRAPH=1<<4, PATH_TERMINAL_ROLLBACK=1<<5, PATH_TEST_INJECTION=1<<6, PATH_RETRIED=1<<7, PATH_PCG_DEVICE_CONTINUATION=1<<8, PATH_LS_DEVICE_LOOP=1<<9, PATH_EPISODE_RESIDENT=1<<10, PATH_GPU_NATIVE_RL=1<<11` |
| `FrameErrorCode` | `ERR_NONE=0, ERR_CAPACITY=1, ERR_NONFINITE_STATE=2, ERR_SOLVER_EXCEPTION=3, ERR_RETRY_EXHAUSTED=4, ERR_GRAPH_LAUNCH=5, ERR_CCD_INVALID=6` |

**字段表**(pybind 导出名,bindings:38-93):

| 组 | 字段 |
|---|---|
| 裁决 | `result`、`phase`、`invalid_bits`、`launch_status`、`error_code`、`path_flags` |
| 首错定位 | `err_env`、`err_primitive`、`err_newton_iter`、`err_ls_iter`(atomicCAS 只保**首个**错误,frame_status.cuh:190-204) |
| 工作量 | `substeps`、`newton_iters`、`pcg_iters`、`ls_trials`、`graph_launches`、`host_boundaries` |
| 容量高水位/需求 | `hw_dcd_pairs / hw_ccd_pairs / hw_triplets / hw_unique_blocks / hw_mas_clusters` 与对应 `required_*`(OVF 裁决输入) |
| 图统计 | `root_graph_nodes / root_d2h_nodes / terminal_graph_nodes / terminal_d2h_nodes` |
| 末态标量 | `final_alpha`、`final_energy`、`max_movement`、`cfl_alpha`、`kappa`(double) |
| 记账 | `frame_id`(int64)、`attempt`、`retry_count`、`retry_invalid_bits` |

**典型用法**(整帧图覆盖率审计,examples/replay_foldshirt_multienv.py:388-409):

```python
fs = eng.native.get_frame_status()
in_full_graph = bool(fs.path_flags & (1 << 4))        # PATH_FULL_CONDITIONAL_GRAPH
overflowed    = bool((fs.invalid_bits | fs.retry_invalid_bits) & (0x1F << 16))
```

非图路径每帧也会记录 legacy 状态(`record_legacy_frame_status`,03:71-89),
所以 `get_frame_status` 在 `STIFF_FRAME_GRAPH` 关闭时同样可读。

---

# 6. 步健康遥测

【稳定线+phase-cd,差异注明】:§6.1 表中 6 个累计计数 getter **稳定线也有**且已绑
pybind——`get_total_newton_iters` / `get_total_pcg_iters` /
`get_total_collision_pairs` / `get_max_collision_pairs` / `get_total_frames_done` /
`get_total_energy_tolerance_accepts`(stable sim_engine.h:642-647,
bindings/pystiffgipc.cu:435-447);**仅 phase-cd** 的是
`get_ls_exhausted_count` / `get_ls_nonfinite_count` 与 `get_frame_status`
(稳定线 grep 无 `m_ls_exhausted_total` / `FrameStatus` 命中)。两线真正的差异:
稳定线这些计数器是 **TU 全局**(如 stable sim_engine.cu:4515 返回全局 `totalNT`,
有跨引擎串台风险),phase-cd 已成员化。

设计意图(GIPC.cuh:111-115):**RL 循环对 `step()` 前后差分这些计数器,检测并丢弃
被静默降级的 episode**(merged 模式的政策是 line-search 耗尽时响亮 WARN 后接受
非下降步——轨迹继续但质量降级,靠计数器可见)。

## 6.1 计数器与 getter

| Python/pybind getter(bindings:750-762;SimEngine 实现 04_teleport_checkpoint.inl:156-175) | 底层计数器 | 语义 |
|---|---|---|
| `get_total_newton_iters()` | `m_total_newton_iters` | 累计 Newton 迭代 |
| `get_ls_exhausted_count()` | `m_ls_exhausted_total` | line-search 预算(`line_search_max_iter`,默认 64)耗尽次数 |
| `get_ls_nonfinite_count()` | `m_ls_nonfinite_total` | 耗尽且 trial/E0 非有限的次数 |
| `get_total_pcg_iters()` | `m_total_pcg_iters` | 累计 PCG 迭代(double) |
| `get_total_collision_pairs()` / `get_max_collision_pairs()` | — | 累计/峰值接触对 |
| `get_total_frames_done()` | `m_total_frames` | 完成帧数 |
| `get_total_energy_tolerance_accepts()` | `energy_tolerance_accept_count` | 容差协助接受次数(uint64) |
| `get_frame_status()` | `m_last_frame_status` | §5.2 |

计数器均为 **GIPC 成员**(曾是 TU 全局导致跨引擎串台,已成员化;GIPC.cuh:107-178),
轻量 host 读,随 checkpoint 恢复、随帧事务回滚还原(frame_transaction.cu:1514-1548)。

## 6.2 增量点与失败语义

- host 路径:LS 预算耗尽 `++m_ls_exhausted_total`;若 trial 能量或 lastEnergyVal
  非有限再 `++m_ls_nonfinite_total`(ipc_solver.inl:826-830)。
- **帧 0 + 非有限增量势 = 初始构型不可行 → 抛 `gipc.GeometryError`**
  (出生即穿插/穿地;中途政策不变:WARN+接受 / isolated 隔离;ipc_solver.inl:831-848)。
- 整帧图路径:图内 LS 耗尽只置 `INV_LS_BUDGET` 位(非致命,接受该步,与 host WARN
  政策一致);host 在帧边界读终端状态包补 `++m_ls_exhausted_total`
  (frame_transaction.cu:4073-4085)。

## 6.3 RL 回合筛除配方

```python
before = (eng.native.get_ls_exhausted_count(),
          eng.native.get_ls_nonfinite_count())
run_episode(...)                                   # N 步
after = (eng.native.get_ls_exhausted_count(),
         eng.native.get_ls_nonfinite_count())
if after != before:
    discard_episode()      # 有静默降级步:该回合样本不可信
```

配合 per-env 遥测(§1.8)可进一步定位:status==2(timeout)/ 3(diverged)的 env
单独丢弃;`get_frame_status().invalid_bits & INV_LS_BUDGET` 给逐帧粒度。
`STIFF_ITER_LOG=1`(【稳定线+phase-cd】,engine.py:1142-1150)可在任意 example
免改代码打印每帧 Newton 遥测。

---

# 7. STIFF_* 环境变量全清册

## 7.0 注册表机制【仅 phase-cd】

- 单一事实源:`StiffGIPC/config/knob_registry.h` 的 X-macro 表(:22-195,170 行),
  分类 7 种:`mode_isolated / mode_strict / perf / solver / audit / diag / python`。
- **维护铁律**(:6-7):任何新增 `getenv("STIFF_...")` 必须同 commit 注册,否则
  `scripts/knob_gate.py`(G14 门禁)FAIL。
- **finalize tripwire**(`stiff_check_unknown_knobs()`,:209-238):进程环境中
  不在注册表的 `STIFF_*` 名字默认打印
  `[knob-registry][WARN] unknown STIFF_* knob ...`(治拼错=静默 no-op);
  设 **`STIFF_KNOB_STRICT=1`** 升级为抛 `ConfigurationError`。
- 稳定线**没有**注册表与 tripwire(`config/` 目录不存在)。

## 7.1 读值语义(五种,写配置前必读)

| 语义 | 判定 | 使用者 |
|---|---|---|
| `env_on`(值感知) | 非空且首字符非 `'0'` | **全部模式旗**(mode_config.h:40-44;python `_env_enabled`,engine.py:249-252) |
| `knob_enabled` | `atoi(value) != 0` | frame-graph 家族(frame_transaction.cu:245-249) |
| **presence-only** | `getenv != nullptr`——**设 `=0` 也算开!** | 大量 diag 旋钮(KSUM/XENV/PHASE_TIME/MIRROR_AUDIT/GRAPH_STATS/SKIP_* 等)。**要关就 unset,不要设 0** |
| 数值参数 | atoi/atof 带默认与钳制 | TIER_STEPS[1,8]、CCD_SLACK 越界回落默认等 |
| 特殊拼写 | 只认精确值 | `STIFF_MAS_FUSE` 只认 `"1"`;`STIFF_URDF_PRIM_PROXY` 只认 `"0"` 为关;`STIFF_NVTX` 只认首字符 `'1'`;`STIFF_DEVICE_LINESEARCH` 默认开、仅首字符 `'0'` 关 |

几乎所有读点用 `static` 局部缓存首读结果(进程级一次性)——**运行中改环境变量无效**,
这也是 §1.3 进程锁的根源。

**默认开的旋钮**(容易被误当 opt-in;其余 160+ 个全部默认关/不改行为):
`STIFF_DEVICE_LINESEARCH`、`STIFF_PCG_GRAPH`、`STIFF_PCG_DEVICE_LOOP`、
`STIFF_GRAPH_DEVICE_RESIZE`、`STIFF_SKIP_ZERO_DEPOSIT`、`STIFF_MAS_SEG`(多 env 时)、
`STIFF_URDF_PRIM_PROXY`(proxy 生成默认开)。

**capture 黑名单**(7 个 diag/audit 旋钮开着时整帧图自动不合格,是"图覆盖率突然
掉 0"的常见原因;frame_transaction.cu:276-292):`STIFF_MIRROR_AUDIT`、
`STIFF_SLOT_AUDIT`、`STIFF_PHASE_TIME`、`STIFF_KSUM`、`STIFF_MAS_DUMP`、
`STIFF_MAS_FUSE_VALIDATE`、`STIFF_STACK_DIAG`。

## 7.2 模式旗(mode_isolated / mode_strict)【稳定线+phase-cd】

| 名称 | 取值 | 作用 | 默认 | 风险/备注 |
|---|---|---|---|---|
| `STIFF_MULTIENV_MODE` | merged/isolated/strict 及别名 0/1/2/a/b/c/decoupled/deterministic | 模式选择,覆盖 `Config.multienv_mode`;唯一消费点在 Python 层(engine.py:295, 471-474) | merged | 未知值 ValueError;进程级锁(§1.3) |
| `STIFF_BVH_ENVDET` | env_on | env-id 碰撞隔离(无跨 env 接触对) | 0 | isolated bundle;单独设=dev override |
| `STIFF_PERENV_BVH` | env_on | per-env 宽相 BVH | 0 | isolated bundle |
| `STIFF_DECOUPLE_THRESH` | env_on | 绝对 dHat + per-group κ gating + N 不变 meanMass;Newton 退出阈值用 env 自身 bbox(ipc_solver.inl:1462-1471) | 0 | 单开无 PERENV_ALPHA → Newton 每帧跑满上限(警告) |
| `STIFF_PERGROUP_KAPPA` | env_on | per-env barrier/friction/init κ | 0 | 单开无 DECOUPLE_THRESH = 广播 stub,非隔离 |
| `STIFF_SEGMENTED_PCG` | env_on | 块对角 per-env PCG(自有 α/β/收敛;pcg_solver.cu:722)。scalar 与 segmented kernel 永不混用 | 0 | isolated bundle |
| `STIFF_PERENV_ALPHA` | env_on | per-env line-search α(S1→S3→冻结;§1.9) | 0 | 隔离可用性门的一半 |
| `STIFF_PERENV_PAR` | env_on;`=0` 单独退出 | K-stream 并发 per-env BVH build+query | 0(模式设 1) | 已通过 strict 资格(N=2/4/8 全 0.000) |
| `STIFF_EE_CANON` | env_on | 规范 edge-edge 发射顺序 | 0 | strict extra;被整帧图**准入**(C6-l) |
| `STIFF_EE_DETGATE` | env_on | 确定性 EE 去重门 | 0 | strict extra;开着整帧图不合格 |
| `STIFF_CCD_CANON` | env_on | 规范 CCD 发射顺序 | 0 | strict extra;开着整帧图不合格 |
| `STIFF_SPMV_DET` | env_on | 序无关(binned cascade)SpMV(spmv.cu:52/82;pcg_solver.cu:892/1213/1555) | 0 | strict extra;det 开则 PCG 图关闭;被整帧图准入 |

正交输入(不参与模式命名):

| 名称 | 取值 | 作用 | 默认 | 备注 |
|---|---|---|---|---|
| `STIFF_PERENV_TELEM` | env_on | host S1 遥测路径(per-env iters/status、NaN 检疫);`=0` 显式退回零 D2H 快路径 | per_env_exit 设 1 | 注册表标 "python" 但 C++ 也读(mode_config.h:60);开着 isolated 整帧图不合格 |
| `STIFF_PERENV_MASK` | env_on | per-env 活跃掩码(冻结已收敛 env 掩出线性系统) | 0 | solver 类 |
| `STIFF_PERENV_MASK_DEV` | env_on | 设备派生掩码(从 m_env_alpha,零 D2H);需 perenv_alpha | 0 | solver 类 |
| `STIFF_PERENV_K` | int | per-env 并发流数上限 | 8 | 11_perenv_machinery.inl:455 |

## 7.3 perf 类(性能;重点旋钮各带说明)

### 整帧 CUDA Graph 家族【全部仅 phase-cd】

| 名称 | 取值 | 作用 | 默认 | 风险 |
|---|---|---|---|---|
| `STIFF_FRAME_GRAPH` | knob_enabled | 两图事务(root 快照图 + terminal 验证/还原/序列化图包住宿主帧体);step() 每帧分流(03:72-78);未设 `STIFF_CONVERT_DEVICE_COUNT` 时同时打开 device-count 布局 | 关 | 大帧倒贴(§8);帧 0 恒走释放版求解器 |
| `STIFF_FRAME_FULL_GRAPH` | knob_enabled | 叠加:整帧收进单条件图(一次 cudaGraphLaunch + 一次边界同步);失败/不合格回落两图事务 | 关 | 资格表见 §3.6;strict 拒绝 |
| `STIFF_C4_COLLISION_GRAPH` | knob_enabled | 碰撞(BVH+CP+CCD)进整帧图 opt-in(step 路径;episode/prepare_gpu_rl **不需要**它);同时打开帧边界 kappa 重初始化(attempt 0 only,C6-h) | 关 | — |
| `STIFF_C5_ISOLATED_GRAPH` | knob_enabled | isolated 完整 bundle 进整帧图 opt-in;图内录**合并树 + 发射级 env 过滤**(与 per-env 树同 pair 集;14_energy:715-720) | 关 | 需碰撞在图内 + env 组已声明 |
| `STIFF_C6_ABD_STEP_GRAPH` | knob_enabled | ABD 刚体在普通 step() 整帧图 opt-in(GPU-RL 路径早已证明,此旗只管 step()) | 关 | — |
| `STIFF_FULL_GRAPH_MIN_VERTS` | int | 小场景只婉拒 FULL 图(两图事务保留——tier 训练/ABD warm-up 住在那里);`<=0` 关门 | 1024 | sub-1k 场景整帧图付 10-80x |
| `STIFF_GRAPH_TIER_HEADROOM` | int [1,64] | 容量训练档余量倍数。**确定性旋钮不只是内存旋钮**(接触计数出自跨流 racy atomicAdd,近档边界会致有的 run 重试有的不)(GIPC.cuh:980-1005) | 1(C6-p 从 2 降) | 2x 曾使每容量宽 pass 翻倍(forcegrip A800 3.25x→1.75x) |
| `STIFF_TIER_STEPS` | int [1,8] | 容量档梯每倍频细分级数 | 1 | 细分实测 −1.0% 中位=null 且边界回退增多,否决为默认 |
| `STIFF_GRAPH_TRAIN_SHRINK` | int(hold 帧数) | 训练宽度收缩(滞回;宽度变化主动 re-record ~25ms) | 0=legacy 单调 | — |
| `STIFF_GRAPH_INGRAPH_RETRY` | knob_enabled | 恢复旧的图内溢出重放 | 关(Scheme-1 契约) | 【实验性】注入 ~1e-6 run-to-run 扰动(towel crumple 0.77..1.11 vs 宿主确定 0.905) |
| `STIFF_GRAPH_DEVICE_RESIZE` | knob_enabled | 图节点网格宽度的 device 端 resizer(16 槽环独占;`cudaGraphKernelNodeSetGridDim` 同重放生效) | **开**(C6-y 起) | `=0` 关;memset32 136→83ms,forcegrip 全核 −10% |
| `STIFF_ALPHA_RESIZE` / `STIFF_MAS_APPLY_RESIZE` | knob_enabled | α 归约 / MAS apply 网格窄化 | 关 | 位级中性需 identity fill;实验 |
| `STIFF_NO_PAD_CLASS_ALLOWANCE` | knob_enabled | 去 pad-class 容量补贴 | 关(保留补贴) | convert 发射长度 −42% 但墙钟不动、覆盖 94%→92%(C6-z) |
| `STIFF_ABD_TIER` | 非零 | 直接宣布 ABD 装配 tier 可信(跳过两图训练) | 关 | 测试用 |
| `STIFF_FRAME_MAX_RETRIES` | int [0,16] | 帧重试预算(注册表分类 solver) | 3 | 耗尽抛 `capacity retry budget exhausted` |
| `STIFF_LS_GRAPH` | int | C-1:回溯 line-search 自尾发图(宿主循环换一次 24B packed 读) | 0 | capture 内抛错则本会话永久回退宿主循环 |

### PCG / 线搜索 / 装配家族

| 名称 | 取值 | 作用 | 默认 | 适用线 |
|---|---|---|---|---|
| `STIFF_PCG_GRAPH` | int | PCG 迭代体 CUDA 图(merged 与段式路径) | **1** | 两线;det 栈/KSUM/MAS_DUMP 等自动关 |
| `STIFF_PCG_DEVICE_LOOP` | int | device 自尾发收敛循环(需 use_graph、驱动≥12000、K>0;段式还要 K 偶数) | **1** | 两线 |
| `STIFF_PCG_GRAPH_CACHE` | int | 跨 solve 图缓存(签名=代际+形状+tol bits) | 0 | 仅 phase-cd |
| `STIFF_PCG_CHECK_K` | int > 0 | 每 K 次迭代做一次收敛检查(注册表标 diag,实为节律参数) | 8 | 两线;<=0 抛错 |
| `STIFF_PCG_TOL` | float | 覆盖 PCG 收敛容差 | `Config.pcg_tol`(默认 1e-4) | 两线;solver 类 |
| `STIFF_PCG_WARM` | int | warm-start(per-env 安全门;与 binned/det 互斥) | 0 | 两线 |
| `STIFF_PCG_EW` (+`_GAMMA`=0.9, `_ETAMAX`=0.5) | int/float | per-env Eisenstat–Walker 强迫项(solve 内常量→graph-safe) | 0 | 两线;solver 类 |
| `STIFF_SEG_BINNED` | env_on | 非 strict 强开 binned seg-dot | det 时自动 | 两线 |
| `STIFF_SEG_WARP` | int | warp 预求和覆盖:det 默认 0、非 det 默认 1;`=2` 审计 | 条件默认 | 两线;`=1` 对 strict 强开会**破坏确定性** |
| `STIFF_DEVICE_LINESEARCH` | 首字符 '0' 关 | 能量归约出 device(`computeEnergy_DeviceOut`) | **开** | 两线 |
| `STIFF_MAS_SEG` | 未设=auto/0/1/N | per-env 分段 MAS;多 env 默认全模式开(块对角正确预条件;strict 借它拿跨 env 位一致);上限 4096 env | 多 env 自动开 | 两线 |
| `STIFF_MAS_FUSE` | 仅精确 "1" | MAS 融合内核 opt-in | 关 | 两线;融合 TU 曾被双盲否决(255reg 溢出) |
| `STIFF_EE_LB` | 0/2/3 | selfQuery_ee 内核变体(2=lb2) | 0;merged 解析器 setdefault 2 | 两线;`=3` 实测 +52% 勿用 |
| `STIFF_SKIP_ZERO_DEPOSIT` | knob_enabled | merged-deposit 跳零块(b842d2f 真因案:零 pad 原子长龙 22×/实例;forcegrip 60f 13.34→9.78s) | **开** | 仅 phase-cd;`=0` 恢复;merged 栈 ~2 ULP 扰动(自噪声内),det 栈位级中性 |
| `STIFF_CONVERT_DEVICE_COUNT` | env_on | device-count 布局模式总开关 | 未设时跟随 FRAME_GRAPH | 仅 phase-cd |
| `STIFF_CONVERT_HOST_BOUND` | int | 事务外允许宿主上界 convert(消 per-convert D2H) | 0 | 仅 phase-cd |
| `STIFF_CONVERT_EXACT_WIDTH` | int | convert 发射宽度收到精确 length | 0(保持 tiered) | 仅 phase-cd;收益在噪声内 |
| `STIFF_ALPHA_TYPE_SPLIT` | int | α 归约按 PT/EE 分两 pass | 0 | 仅 phase-cd;实测 −17%(慢),留实验 |
| `STIFF_DRIVE_SUBSTEP` | int S | 驱动目标沿前 S 个 Newton 迭代 ramp(治全局 LS 的 kick;注册表标 perf,实改求解语义) | 0 | 两线;episode 图拒绝 >1 |

### CCD 步长旋钮(solver;【仅 phase-cd】;GIPC.cuh:61-101)

| 名称 | 默认 | 合法域 | 作用 |
|---|---|---|---|
| `STIFF_CCD_SLACK_A` | 0.9 | (0,1) | ACCD 保守推进分数,**ground** CCD |
| `STIFF_CCD_SLACK_M` | 0.8 | (0,1) | ACCD 保守推进分数,**self + swept** |
| `STIFF_CCD_CFL_FACTOR` | 0.5 | (0,8] | swept-BVH 膨胀护栏 `sqrt(dHat)/maxspeed × factor`——性能护栏非正确性界(实测在 22.6% 重段迭代把 α 压到 ACCD 认证值下,中位 1.48×) |

推荐配方(**仅重接触场景**,../SIMULATOR_EXECUTION_DESIGN.md:228-231):
`STIFF_CCD_SLACK_M=0.9 STIFF_CCD_CFL_FACTOR=1.0` → Newton −7.9%(5 批配对全负)、
墙钟 −3~4%;轻接触零收益或小亏勿开;默认全关。

### BVH 实验家族【全部仅 phase-cd;全部默认关,显式 opt-in】

| 名称(族) | 作用 | 实测要点 |
|---|---|---|
| `STIFF_BVH_PLOC`(+`_RADIUS`=16/`_CHUNK`=256/`_MASK`=0xF) | PLOC/PLOC++ 构建 | 4090 merged-1env 赢家之一;**A800 isolated 全退、merged-4env +5.9%、图内 8× 崩塌**——无普适默认(../BVH_ADVANCED_VALIDATION.md:792-796;../SIMULATOR_EXECUTION_DESIGN.md:191-214) |
| `STIFF_BVH_REFIT_INTERVAL`(默认 1)+`_MASK` | 精确拓扑 refit 间隔 | interval-128 与 PLOC 组合是 4090 merged 赢家(1550 帧 −8.64%) |
| `STIFF_BVH_QUERY_ORDER` | RBS 式查询排序(mask 1=VF-DCD) | merged −5.93%(4090);VF-CCD 排序倒亏 |
| `STIFF_EE_RANGE_PRUNE`(1/2) | EE 子树所有权剪枝 | 档 1 精确、EE-CCD kernel −13.14%;档 2 改编码,strict 下不可用 |
| `STIFF_BVH_WIDE8`(0..3)+`_MASK`、`STIFF_BVH_MORTON14`、`STIFF_BVH_SAH_ROTATIONS` 族、`STIFF_BVH_PAIR_CACHE` 族、`STIFF_BVH_BODY_MAJOR`(+_MASK)、`STIFF_BVH_FRONT_REUSE`、`STIFF_BVH_PERENV_QUERY_SUBSET` | 各类实验 | 全部维持 opt-in;引用裁决前读 ../BVH_ADVANCED_VALIDATION.md 与 ../SIMULATOR_EXECUTION_DESIGN.md 附录 B 三合一 |

## 7.4 solver 类(改物理/求解语义——**不是安全默认关的生产旋钮**)

| 名称 | 取值 | 作用 | 适用线 |
|---|---|---|---|
| `STIFF_SKIP_E` / `STIFF_SKIP_F` / `STIFF_SKIP_GRND` / `STIFF_SKIP_FRIC` | presence-only | 分别跳过 edge 自碰 / face 自碰 / 地面碰撞(连帧首地面隔离扫描一起关)/ 摩擦贡献。**物理消融用,开了直接改物理** | 两线 |
| `STIFF_EE_NODEDUP` / `STIFF_EE_NOMOLLIFY` | presence-only | 关 EE 去重 / 关 EE mollifier(device 旗) | 两线 |
| `STIFF_GROUND_HESS_LEGACY` | env_on | 退回未 PSD clamp 的地面 Hessian 投影 | 两线 |
| `STIFF_ABD_PRECOND_LEGACY` | env_on | 恢复旧 racy 赋值式 ABD 对角预条件(新默认 atomic 累加,治轻质刚体 kick/flip) | 两线 |
| `STIFF_SPLIT_GH` | presence-only | 梯度/Hessian 装配拆分 | 两线 |
| `STIFF_NO_REFINE` | presence-only | 跳过 per-env α 的 refine 二次门(注册表标 perf,实改 α 语义;验证 hr 泄漏的测试旗) | 两线 |
| `STIFF_EPSV` | float(m/s);`atof`,`>0` 才生效,`=0`/负 → legacy | 钉死 IPC 摩擦 stiction 阈值 epsv:`fDhat = epsv²`;否则 legacy 场景派生 `epsv = 1e-2·eff_scene_diag`(1.9 m 场景 ≈ 19 mm/s)。**覆盖 `Config.absolute_epsv`**(env 赢),读点在 `init` 一次性(stable@c0339c8 GIPC.cu:9498-9500);日志等级 ≥1 时 `[dhat]` 行会打印 `epsv=<值> (ABSOLUTE)`/`(scene-scale)`,是确认旋钮生效的唯一自证。默认 = `Config.absolute_epsv` = **1e-4(v0.8.5.4 起)**;`STIFF_EPSV=0` 是回 legacy 的逃生阀。IPC 原文:静摩擦精度取 1e-5,越小 Newton 越贵(1e-5 +12% step,1e-4 无可测成本) | **【仅稳定线 v0.8.5.4+】**;phase-cd 无读点,设了不但无效,还会被 knob-registry tripwire 报 `unknown STIFF_* knob`(§7.0) |
| `STIFF_FRIC_ANCHOR` | `atoi != 0`;显式 `0`/`1` 均被尊重(存在即定值,不是 presence-only) | 持久跨步摩擦锚(真静摩擦)总开关,**覆盖 `Config.friction_anchor`**(stable@c0339c8 GIPC.cu:9811)。默认 = Config = **True(v0.8.5.4 起)**,但 **strict 下自动抑制**(检测到 `STIFF_SPMV_DET` 即关,并打印 `[fric-anchor] strict mode: friction_anchor suppressed for batch invariance`;`STIFF_FRIC_ANCHOR=1` 可强制开回,§2.1)。`=0` 回 legacy 每步重置锚。开着改物理:flask_cap 保持段滑移 3.7 mm→0.00 mm,代价 +9% step | **【仅稳定线 v0.8.5.4+】**;同上,phase-cd 无读点/未注册 |

## 7.5 diag / audit 类(诊断;多为 presence-only)

| 名称 | 作用 | 备注 |
|---|---|---|
| `STIFF_FRAME_GRAPH_DIAG` | 整帧图机制主 verbose(不合格原因/rollback-audit/resize 等十余处) | 混用两种门,**建议统一写 `=1`**;仅 phase-cd |
| `STIFF_GRAPH_STATS` | teardown 打印 `[graph-stats] frames=N full_graph=... two_graph=... legacy=...` 覆盖率 | presence-only;仅 phase-cd |
| `STIFF_GRAPH_PHASE_TIME` / `STIFF_FRAME_SECTION_TIME` | 设备 globaltimer 相位打点 `[graph-phases] bvh/dcd0/gh/pcg/ccd_ls/postls` / 边界三段耗时 | 仅 phase-cd;§8 相位分解即由此测得 |
| `STIFF_FRAME_FORCE_ROLLBACK` | `=1` G16 故障注入(OK 帧改判 RETRY+ERR_CAPACITY);**`=2` paired-audit 成对审计**(图解完整帧→丢弃→宿主重解同帧;逐帧位级同态 graph-vs-host 成本对照) | 仅 phase-cd;frame_transaction.cu:258/2676/4045/4473 |
| `STIFF_FRAME_FORCE_UNIQUE_TIER` / `STIFF_C6_RERECORD` / `STIFF_GRAPH_DIRPROBE` / `STIFF_GRAPH_TIERPROBE` / `STIFF_GRAPH_RESIZE_DIAG` / `STIFF_GRAPH_TRAIN_SHRINK_DIAG` / `STIFF_GRAPH_UNTIER_PAIR_SLOT0` / `STIFF_GRAPH_LEGACY_KAPPA_DOUBLE` / `STIFF_GUARD_FRAME_END_ONLY` / `STIFF_FALLBACK_NO_REBUILD` / `STIFF_GRAPH_ACCEPT_STARVED_LS` | 图机制探针族 | 仅 phase-cd |
| `STIFF_KSUM` | 内核级 checksum(大量 deviceSync) | presence-only;capture 黑名单;关 PCG 图;两线 |
| `STIFF_MIRROR_AUDIT` / `STIFF_SLOT_AUDIT` | 镜像/槽位审计 | audit 类;capture 黑名单;仅 phase-cd |
| `STIFF_PHASE_TIME` | 宿主路径相位计时 | presence-only;capture 黑名单;两线 |
| `STIFF_STACK_DIAG` | per-pop 栈深探针 | capture 黑名单;两线 |
| `STIFF_LSX_DIAG` | LS 耗尽诊断(点名哪个能量项/计数在零步长下变了=状态泄漏) | 仅 phase-cd |
| `STIFF_NEWTON_TRACE` | 逐 Newton 迭代打印 `[newton-trace] f=<帧> k=<迭代> move=<move-norm>`(取 `calcMinMovement` 归约出的移动范数,作残差代理;帧号由 `k==0` 自增,全局非 per-env;已判全局退出的那次迭代不打印) | **【仅稳定线 v0.8.5.4+】**(`d7ab5bf`,stable@c0339c8 GIPC.cu:16000-16011);presence-only(设 `=0` 也算开);开着**每迭代一次 D2H**,只作诊断用。用途示范:摩擦锚 A/B 显微镜——flask_cap 保持段 anchor ON 的 r1 中位 5.4e-12(真不动点)vs OFF 1.0e-5 m/步(= 10 µm/帧 creep,×50 fps ≈ 0.5 mm/s,与实测滑移吻合) |
| `STIFF_POSTLS_FREEZE` | 跳过 LS 后帧内 kappa/close-set 自适应(因果测试) | 仅 phase-cd |
| `STIFF_DEVICE_LINESEARCH_VALIDATE` / `STIFF_CCD_VALIDATE` / `STIFF_S3_VALIDATE` / `STIFF_ENERGY_VALIDATE` / `STIFF_MAS_FUSE_VALIDATE` / `STIFF_PCG_CACHE_VERIFY` | 各对照校验 | 两线为主 |
| `STIFF_PENV_STATS` / `STIFF_A0_DUMP` / `STIFF_S1_DEBUG` / `STIFF_ALPHA_DBG` / `STIFF_ALPHA_STATS` / `STIFF_SWEPT_DIAG` / `STIFF_SEED_DIAG` | per-env α/S1/S3 打印族(其中前四个会把 S1 路由到 host 遥测路径,§1.8) | 两线为主 |
| `STIFF_XENV` / `STIFF_XENV_DUMP` / `STIFF_XENV_ID` | 跨 env 通道差探针(2-env 复制场景验证 env0/env1 位对应) | 两线 |
| `STIFF_DUMP_FRAME` / `STIFF_PROBE_K` / `STIFF_GRAD_PRE` / `STIFF_GRAD_PROBE` / `STIFF_H_DUMP` / `STIFF_Z_DUMP` / `STIFF_ABD_DUMP` / `STIFF_MCDUMP` / `STIFF_MAS_DUMP` / `STIFF_SHAPE_STAGE` / `STIFF_HESS_ENV0` / `STIFF_BAR_TRACE` / `STIFF_BAR_TGT0/1` / `STIFF_EE_TRACE` / `STIFF_MERGED_DIAG_FRAME` / `STIFF_CONTACT_DBG` / `STIFF_ABD_DBG` / `STIFF_XSKIP_DBG` / `STIFF_BVH_ENVPART` | dump/trace 选择器族 | 两线为主 |
| `STIFF_PCG_GRAPH_DIAG` / `STIFF_PCG_EXIT_DIAG` / `STIFF_PCG_WIDTH_DIAG` / `STIFF_SEG_DIAG` / `STIFF_DIAG_BINNED_GRAD` / `STIFF_DIAG_KAPPA_MERGEDBB` | PCG/κ 诊断 | 两线为主 |
| BVH audit:`STIFF_BVH_TRAVERSAL_AUDIT` / `STIFF_BVH_PAIR_WORK_AUDIT(_DETAIL)` / `STIFF_BVH_COHERENCE_AUDIT` / `STIFF_BVH_CCD_COHERENCE_AUDIT` / `STIFF_BVH_MARGIN_SCALE` / `STIFF_BVH_PAIR_CACHE_STATS` / `STIFF_BVH_REFIT_STATS` / `STIFF_BVH_SAH_ORACLE` | 遍历/配对审计 | 仅 phase-cd;部分需 `-DSTIFFGIPC_BVH_*_AUDIT_BUILD` 编译宏(环境变量与同名宏是两回事) |
| `STIFF_NVTX` | NVTX range(仅认 '1') | 仅 phase-cd |
| `STIFF_ASSEMBLY_TIER_SHIFT` | tier 左移 N 档(测试) | 仅 phase-cd |

## 7.6 python 类

| 名称 | 作用 | 适用线 |
|---|---|---|
| `STIFF_ITER_LOG=1` | 每帧 Newton 迭代遥测,任意 example 免改代码(engine.py:1142-1150) | 两线 |
| `STIFF_AUTO_PREPARE_AT=k` | §3.7 自动 prepare + OVF 恢复 | 仅 phase-cd |
| `STIFF_MS_DUMP=path` | 逐步墙钟 ms np.save | 仅 phase-cd |

## 7.7 已知死行 / 漏网 / 陈旧名

1. **注册表死行(2 个)**:`STIFF_GRAPH_TRAIN_FIT`、`STIFF_SPMV_RESIZE`——两棵树全部
   代码(含 scripts)无任何消费点,设了完全无效(tripwire 不会警告,因为在注册表里)。
2. **漏网旋钮 `STIFF_URDF_PRIM_PROXY`**:被 `urdf_scene_importer.cpp:882` 真实消费
   (默认开启 primitive→proxy mesh;仅精确 `"0"` 恢复 ≤0.8.4 的静默跳过),但**不在
   注册表**(`knob_gate.py` 的 grep 不含 `.cpp`)。后果:设它会触发 finalize tripwire
   的 unknown-knob WARN(误报);`STIFF_KNOB_STRICT=1` 下直接抛错。两线都读它。
3. **陈旧注释名 `STIFF_CONVERT_TIER_WIDTH`**:global_linear_system.cu:318 注释提到,
   代码实际读的是 `STIFF_CONVERT_EXACT_WIDTH`;不存在叫 TIER_WIDTH 的变量。
4. 稳定线独有(phase-cd 已删):`STIFF_FRICTION_DBG`(stable GIPC.cu:9606, 11981)。
   另有三个**稳定线 v0.8.5.4 独有、phase-cd 从未有过**的摩擦线旋钮:`STIFF_EPSV`、
   `STIFF_FRIC_ANCHOR`(§7.4)、`STIFF_NEWTON_TRACE`(§7.5)。它们不在 phase-cd
   注册表内,因此**不计入下条的 170**;在 phase-cd 上设置只会换来 unknown-knob WARN。
5. 稳定线/phase-cd 覆盖对照:注册表 170 个旋钮中 **82 个稳定线也有消费点,88 个仅
   phase-cd**(整帧图全家族、BVH 实验族、converter 布局族、CCD 步长旋钮、audit 族、
   NVTX、AUTO_PREPARE_AT/MS_DUMP 均仅 phase-cd)。**给稳定线用户的配置不得出现这些。**

---

# 8. 性能指南

## 8.1 通道 regime 结论(决定胜负的是 regime,不是特性)

定稿架构(../SIMULATOR_EXECUTION_DESIGN.md):

| regime | 正确选择 | 依据 |
|---|---|---|
| **大帧回放/操作轨迹**(百 ms 级/帧) | **step 通道,图关**(默认即是) | 整帧图开 = **墙钟 +3%~75%(场景相关)**——本文档 §8.2 实测 forcegrip +43%、beaker +35%。伴随的 **+10% 是 Newton 计数盈余**(非墙钟;9342 vs 8808),收官解剖(paired-audit 成对审计,A800 foldshirt 4-env 1551 帧,869/1550 帧逐位相等):图内求解语义仅 +1.0%,主体 +6% 为事务壳 ULP 扰动 × merged 混沌轨迹效应,+4% 图化增量(../OPTIMIZATION_ROADMAP.md:75-81) |
| **RL 稳态微步**(ms 级/步) | **residency 通道(`prepare_gpu_rl`)** | A800 D4 关节接触,**300 步**(contact priming 后,dt=0.01;C6-w 轮 2026-08-06,../A800_ALLEXAMPLES_TIMING_2026-08-01.md:356-364):v0.8.5 宿主 19.3 ms/步 → HEAD 宿主 9.7(**2.0×**)→ 驻留 **3.85(5.0×)**;4090(150 步,2026-08-04 轮):4.25→3.72→3.18(1.34×),终局 step() 薄发射 3.14 | 
| **接触逐帧升级轨迹**(抓握) | **禁止 prepare**,走 step 通道 | 容量平稳性判据:foldshirt auto-prepare 后 25/25 帧 OVF_TRIPLETS 失败(合同矛盾非速度问题) |
| 需要**可预测尾延迟** | 图开可接受 | 方差 ±8% vs 宿主 ±21%(唯一理由) |

- 判据 τ/T:τ = 每步宿主固定开销,是**平台属性**(A800 集群 ~15 ms/步、桌面 4090
  ~1 ms/步);收益 ∝ 宿主往返延迟。
- 发射经济学一句话:**串行化贵、宽度免费**——痛苦 ∝ 发射次数×延迟÷单核计算量。
- 钉死参数勿动:headroom=1、档梯 2 的幂、PCG K=8、CFL 下限=上游原值
  (../SIMULATOR_EXECUTION_DESIGN.md 附录 D)。

## 8.2 场景规模实测数字表

大帧(60 帧,4090,3 轮交错中位;../SIMULATOR_EXECUTION_DESIGN.md 附录 A):

| 负载 | v0.8.5 | phase-cd 默认(step,图关) | 图开 |
|---|---|---|---|
| forcegrip(每帧中位) | 115.2 ms(8.7 fps) | **105.5 ms(9.5 fps,−8%)** | +43%(C6-w 口径 7.10→10.37 s 总墙钟) |
| beaker | 160.8 ms | 166.8 ms(+4%,**差 = 纯启动段 ~0.6 s,每帧持平**——step 通道每帧 == v0.8.5 承诺成立) | +35% |
| finray(每帧中位) | 289.5 ms | **270.7 ms(−7%),方差 233–315 → 267–280 收窄** | — |

RL 微步(D4 铰接接触场景,ms/步):

| 平台 | v0.8.5 宿主 | phase-cd 宿主 | phase-cd 驻留(gpu_rl) |
|---|---|---|---|
| 4090 | 4.25(另一轮口径 4.00) | 3.72 | **3.18;终局 step() 自动薄发射 3.14(1.27×)** |
| A800 | **19.3** | **9.7(2.0×)** | **3.85(5.0×)** |

(两组 4090/A800 数字来自不同轮次:§6 版 2026-08-04 与附录 A 终局/C6-w 版
2026-08-06,引用时注明轮次;../A800_ALLEXAMPLES_TIMING_2026-08-01.md:341-372。)

多环境容量参考(RTX 4090 24GB,foldshirt 抓取回放,examples docstring 历史实测):
~6 GB 基座 + ~0.8 GB/env;N=20@22.4 GB 369 ms、N=22 边缘、N≥24 OOM;
merged 吞吐 ~55 env-steps/s 饱和(串行 merged 解只摊薄固定开销)。

## 8.3 相位成本分解(引 ../OPTIMIZATION_ROADMAP.md,勿重测)

基准:foldshirt 4-env merged 回放 1551 帧,A800,宿主通道 410 s(263 ms/帧);
设备 globaltimer 打点(`STIFF_GRAPH_PHASE_TIME`,416.8 s 设备时间):

| 相位 | 时间 | 占比 |
|---|---|---|
| PCG(线性求解) | 237.1 s | **56.9%** |
| — 其中 precond 应用(MAS) | 112.3 s | **27%(全仿真)** |
| — 其中 SpMV | 34.1 s | 8% |
| — 其中 mix1/mix2 | 31.0 s | 7% |
| GH(梯度+Hessian 装配) | 91.8 s | 22.0% |
| CCD+LS(线搜索) | 82.9 s | 19.9% |
| DCD | 4.0 s | 1.0% |
| BVH | 0.8 s | 0.2% |
| postLS | 0.3 s | 0.1% |

优化原则:候选项对照它作用的相位占比 = 收益天花板(Amdahl)。
大帧另一口径(4090 finray):BVH selfQuery ~29%、PCG 微核链 ~20%;
MAS apply 占 PCG body 65%。

## 8.4 预条件器(precond)选择

| 场景 | 推荐 | 依据 |
|---|---|---|
| merged 多环境臂+布(foldshirt 家族) | MAS(`preconditioner_type=1`) | MAS 最配 MERGED(mode_contract.h:53-55);对角替代实测每迭代便宜 ~13% 但 Newton +14~23% 且方差巨大,净账平手偏负(../OPTIMIZATION_ROADMAP.md:20-33) |
| isolated / strict | diagonal(`=0`)起步 | diag 配 ISOLATED/STRICT(mode_contract.h);0.8.3 数据:strict N=8 分段 MAS 608.9 vs diag 711.2 ms/帧(−14%)——大批量下 MAS 仍可赢,需 A/B |
| 接触轻场景(beaker) | `=0` | examples 默认:beaker precond=0,foldshirt/cupshirt=1 |
| finray 2-env | diag 胜 | 场景相关(foldshirt 研究);A/B 工具 `examples/precond_ab.py` |

多 env 时 `STIFF_MAS_SEG` 默认自动开启 per-env 分段 MAS(块对角正确预条件,§7.3)。

## 8.5 已裁决优化速查(做/不做;详表见 ../OPTIMIZATION_ROADMAP.md:62-81)

| 候选 | 裁决 |
|---|---|
| 跳零 scatter(`STIFF_SKIP_ZERO_DEPOSIT`) | **默认开**(图开 forcegrip −27%) |
| 设备 grid resize(`STIFF_GRAPH_DEVICE_RESIZE`) | **默认开** |
| headroom 2→1 | **默认** |
| CCD α 配方(SLACK_M=0.9 + CFL=1.0) | opt-in,仅重接触 |
| PLOC/refit-128、QUERY_ORDER、EE_RANGE_PRUNE | opt-in(架构×模式矩阵无普适默认) |
| 大帧整帧图默认开 | **否决** |
| 对角替代 MAS | 否决(本场景) |
| 融合核 TU 抽取 | 否决(255reg/栈+41%/SASS+9.5%) |
| 图宽度收窄/pad 削减(四次) | 否决(~0% vs 跳零 −27%:墙钟成本是串行化不是 padded 功) |
| 检测类优化(BVH/broadphase/DCD 节流) | 死路(合计 1.3% 天花板) |
| Persistent GPU Runtime 全量重写 | 不做(维持 3 原型裁决门槛) |

## 8.6 测量方法论(引用数字前必读)

- 混沌场景必须**交错多轮取中位**;单次测量在 ±20% 轨迹分岔前无意义。
- **全轨迹 Newton 总数为主判据**(±1%);60 帧窗墙钟噪声 ±10pp 不可裁 <10pp 效应;
  凡裁决必带 A/A 对照。
- median-of-3;首跑污染 ~10%;GPU 非空闲 fail-closed(`BENCH_REQUIRE_IDLE_GPU=1`)。
- 相隔时间的两次手测可有 3× 机器态漂移——**只有配对运行可用**
  (../GRAPH_DEFAULT_ON_EVIDENCE.md:25-29)。

---

# 9. 示例配方

运行前提:两树均无 `./run` 包装,标准跑法 **`PYTHONPATH=. python examples/<file>.py`**;
资产目录自动探测 `assets/` 或 `Assets/`。命名习惯:`CASE39ME_*`=多环境族、
`CASE39_*`=case39 场景族、`GRIP_*`=夹爪控制、`STIFF_*`=引擎旗标。
通用:`STIFF_BENCH_STATS=1`(逐帧 `[bench] frame N newton X ms Y`)、`CASE39_QUIET=1`(默认静音)。
示例 docstring 里常见的 `STIFF_SKIP_CCD_SANITY=1` 是历史残留,**勿再设置**:
引擎已不读该变量(CCD sanity 复检默认已跳过),phase-cd 上触发 unknown-knob WARN,
`STIFF_KNOB_STRICT=1` 下直接 `ConfigurationError`(口径与 README §6 一致)。

## 9.1 finray 抓取族:GRIP_MODE 三模式

核心库 `examples/umi_finray_lib.py`(三场景 × 单/多环境 × replay/UI 共 9+1 个薄入口)。
场景:`foldshirt`(衬衫布,连续 grip)/ `beaker`(ABD 烧杯,连续 grip)/
`cupshirt`(ABD 杯 + 衬衫,二值 grip)。

三种夹爪控制模式(`GRIP_MODE`,umi_finray_lib.py:461-573):

| 模式 | 机制 | 适用 |
|---|---|---|
| `pos`(默认) | 纯位置控制:`set_prismatic_position(pi, cl + s*(op-cl))`,无反馈 | 布料场景最干净 |
| `force` | 软刚度阻抗抓取:target 指向闭合端、刚度用软的 `GRIP_K_GRIP`;布 → 轻柔闭死,刚体 → 有界抓力 `F = k_grip*(cl − q_obj)` 持住(自动重抓/防滑)。默认 `GRIP_PINCH=1`:finray 接触力和达 `GRIP_TARGET` 即以 `GRIP_LOCK_K` 锁在当前开度(按直径捏住);接触力跌破 `target×GRIP_PINCH_RESUME_FRAC` 解锁重闭。真实抓力用 `get_prismatic_drive_force` 观测 | 刚体 + 布通吃,一条规则 |
| `stitch` | 缝线形变门控位控:`get_stitch_max_stretch_batched` 超 `GRIP_STITCH_THRESH` 且已闭过 70%(`GRIP_STITCH_MIN_S=0.3`)才 latch;跌破 `thresh×GRIP_STITCH_RESUME_FRAC` 解锁重闭(滞回) | 布抓取的低成本反馈(~70ms/帧 vs 接触力 ~200ms/帧) |

关键旋钮默认值(`_drive_params`,umi_finray_lib.py:641-669):`POS_K=15.0`、
`GRIP_K_GRIP=3.0`、`GRIP_TARGET=0.03`(**单位不是牛顿**——示例代码注释里的
"(N)"是错的,这正是 `-force·dt²` 单位陷阱:比较对象是
`get_body_contact_force(_batched)` 范数之和(umi_finray_lib.py:517-521),该
getter 返回**原始增量势垒梯度 `dE/dx = -force·dt²`,LEGACY UNITS、无 `-1/dt²`
换算**(engine.py:1591-1601 与 1783-1793 docstring 明写 "NOT Newtons";
03_step_getters_export.inl:1141-1145;只有 `get_vertex_contact_forces` 路径才
换算成牛顿,03:1758-1764)。按家族默认 dt=0.02,阈值 0.03(梯度范数)≈ 75 N
物理力;**换 dt 后同一数值的物理含义随 dt² 漂移,需重调**。0.03 在默认步长帽下
对 beaker+cup 给 ~17-18mm 捏合)、`GRIP_LOCK_K=15.0`、`GRIP_PINCH_FRAMES=1`、
`GRIP_PINCH_RESUME_FRAC=0.4`、`GRIP_FORCE_STEP_CAP=0.012`(m/帧,≈4 帧闭合)、
`GRIP_CLOSE_DS=0.03`、`GRIP_STITCH_THRESH=2e-5`(m)、`GRIP_URDF=obb`。

可运行命令:

```bash
# 单环境 replay(GUI;三场景同构)
PYTHONPATH=. GRIP_MODE=pos   python examples/replay_foldshirt_finray.py
PYTHONPATH=. GRIP_MODE=force python examples/replay_beaker_finray.py
PYTHONPATH=. GRIP_MODE=force python examples/replay_cupshirt_finray.py

# headless
PYTHONPATH=. CASE39ME_HEADLESS=1 GRIP_MODE=stitch python examples/replay_foldshirt_finray.py

# 多环境(default 4 env;多环境默认 strict)
PYTHONPATH=. CASE39ME_NUM_ENVS=4 GRIP_MODE=pos python examples/replay_foldshirt_finray_multienv.py
PYTHONPATH=. CASE39ME_NUM_ENVS=4 CASE39ME_HEADLESS=1 GRIP_MODE=stitch \
    python examples/replay_beaker_finray_multienv.py

# 交互 UI(单环境;OPEN/CLOSE 按钮 + 模式实时切换 + 150 步预沉降)
PYTHONPATH=. python examples/ui_foldshirt_finray.py

# headless 抓取判定(每次调用独立进程跑,避免同进程多场景踩 metis sort-cache)
PYTHONPATH=. SC=cupshirt GRIP_MODE=force CASE39ME_NUM_ENVS=2 python examples/diag_finray_grip.py
```

注意:`examples/UMI_FINRAY_NOTES.md` 的 force 段描述的是旧实现(硬 barrier /
`GRIP_BARRIER_KAPPA` / force-lock),这些旋钮在当前 lib 已不存在(grep 0 命中);
force 模式以代码为准(impedance + pinch + step cap)。

## 9.2 多环境 CASE39ME_*

正主 `examples/replay_foldshirt_multienv.py`(混合软爪 + 双 panda 抓/叠衬衫回放):

```bash
PYTHONPATH=. CASE39ME_HEADLESS=1 CASE39ME_NUM_ENVS=8 python examples/replay_foldshirt_multienv.py [episode.hdf5]
```

| 旋钮 | 默认 | 含义 |
|---|---|---|
| `CASE39ME_NUM_ENVS` | 4 | env 数(≤256;显存 ~0.8 GB/env) |
| `CASE39ME_MULTIENV_MODE`(lib 入口)/ `STIFF_MULTIENV_MODE` | 多环境→strict,单环境→merged | 三模式;strict 自动配布局三件套(§2.1) |
| `CASE39ME_SPACING` | 4.0(strict 未显式给时自动 0) | env 网格间距 |
| `CASE39ME_BVH_OFFSET` | strict 自动 4.0 | `set_env_offsets` 广义相分隔 |
| `CASE39ME_ISOLATE` | 1 | `set_body_groups` 均分声明 |
| `CASE39ME_PHASE` | 0 | 逐 env 时间相位偏移(异构难度;per-env α 有收益的条件) |
| `CASE39ME_TRAJ_GLOB` / `CASE39ME_TRAJ_JITTER` / `CASE39ME_PIN_ENV0` | — | 异构轨迹池 / 合成关节偏移 / env0 钉基准。**仅 finray 族消费**(umi_finray_lib.py:769/784/787);本脚本(replay_foldshirt_multienv.py)不读这三个,设了静默无效 |
| `CASE39ME_BUFF_SCALE` / `LSYS_SCALE` / `TRIPLET_MARGIN` / `ABS_DHAT` | 4.0 / 2.0 / 4.0 / 0.0019 | 缓冲与接触参数(abs_dhat 钉单环境值,接触不随 N 膨胀) |
| `CASE39ME_LOAD_CHECKPOINT` / `SAVE_CHECKPOINT` | — | checkpoint 快启/保存【仅 phase-cd】 |
| `CASE39_GRAPH_STATS=1` | — | 逐帧 `get_frame_status()` 整帧图审计:`[fs-graph-audit] full/fallback/overflow`【仅 phase-cd】 |
| `CASE39_RECORD_MP4` / `CASE39_RECORD_USD` / `CASE39_RECORD_STRIDE` | — | 录制(基准循环不变)【仅 phase-cd】 |
| `CASE39ME_DUMP_VERTS` / `CASE39ME_DUMP_PAIRS` / `CASE39ME_DUMP_CCD_PAIRS` | — | release-gate 工具(strict 共址下逐位可比) |

机理说明(必读 docstring,examples/replay_case39_multienv.py:1-33):StiffGIPC 的
multi-env 不是批式/向量化 solver——唯一世界、唯一 BVH、唯一 Newton/PCG;
merged 下一个 env 顶到 newton cap 全体一起慢(隔离要 isolated/strict 或
per_env_exit)。

其他多环境示例:`duck_multienv.py`(`DUCK_N=64` 上限压测)、
`duck_grasp_multienv.py`(FR3 抓鸭,uipc 512-env 对标,`GRASP_N=4`)。

## 9.3 RL 用法(GPU 驻留)【仅 phase-cd】

```python
import os
os.environ.setdefault("STIFF_MULTIENV_MODE", "merged")   # residency 支持 merged/isolated;strict 刻意排除
from stiff_physics import Config, Engine

eng = Engine(Config(dt=0.01, ...))
# ... load_urdf / load_mesh / set_body_groups / finalize ...
eng.finalize()

# 1) warm-up:带有代表性接触峰值的同步 step(容量档按此训练)
for _ in range(warmup_frames):
    eng.step()

# 2) 捕获一帧 GPU-native 图(此调用可能内部再推进一帧,见 §3.1)
eng.prepare_gpu_rl()

# 3) 零拷贝 torch 稳态环(全程无宿主同步)
t = eng.gpu_rl_tensors()
for i in range(steps):
    t["actions_revolute"][:, 0] = policy(t["positions"], t["joint_observations"])
    eng.launch_gpu_rl_async()          # 或直接 eng.step():自动薄发射,等价
    # done 判定 / reward 在设备上算;需要选择性重置:
    # eng.launch_gpu_rl_reset_masked_async(done_mask_tensor.data_ptr())

eng.synchronize_gpu_rl()               # 边界:读回前的显式栅栏(非默认 torch 流必须)
eng.end_gpu_rl()                       # 回到常规 host-driven step()
```

- 回合健康筛除:差分 `get_ls_exhausted_count()` 等(§6.3);逐帧粒度读
  `statuses` uint8 张量按 `FrameStatus` 布局解(result 偏移 0、invalid_bits 偏移 8、
  error_code 偏移 32;engine.py:1044-1060 的 ctypes 参考实现)。
- 基准/自动化:`STIFF_AUTO_PREPARE_AT=k` + `STIFF_MS_DUMP=path`(§3.7)。
- 开环吞吐实验:`launch_episode_async(frames, rev_actions, pri_actions)` 七件套(§4)。

## 9.4 初始状态变异(towel scramble)

`examples/recipe_towel_scramble.py`(标准配方,docstring 全录于文件):
Phase1 SCRAMBLE(随机速度 kick 揉毛巾——**必须走
`teleport_fem_vertices(positions, velocities)`** 注入速度+xTilta;
`set_vertex_velocities_gpu` 单独用不重建 xTilta,body 不会动)→
Phase2 SAVE(.npy + `save_checkpoint`)→ Phase3 REUSE(新 engine 同场景 teleport
到揉皱位形)。rest shape(DmInverses)不受 teleport 影响 → 弹性对皱态仍正确;
跨进程精确复现用 `save/load_checkpoint`。phase-cd 附加:验证段前 `eng.reset()`
(solver buffer 走进程级全局 CUDA symbol,须先释放第一个 engine)。

## 9.5 力/关节控制教学(9 连)

`case_force_*.py` 9 个 GUI 小场景逐一演示:`set_body_external_force` /
`set_body_external_wrench`(12-DOF)/ `set_revolute_torque` / `set_prismatic_force` /
位置驱动→力控两阶段 / 关节速度控制(target=当前角+ω·dt)/ 双臂混合控制。
docstring 里的 `STIFF_SKIP_CCD_SANITY=1` 前缀勿再设置(见 §9 开头)。
headless 回归:`test_force_control.py`
(x(t)=½at² 验证)等。

## 9.6 回归/门禁测试清单(特性有测试背书)

多环境/确定性:`test_env_isolation.py`、`test_env_quarantine.py`(per_env_exit +
env_newton_iter_cap:中毒 env 冻结不炸进程)、`test_env_midrun_quarantine.py`
(运行中 ground-infeasible env 单独隔离 status=3,健康 env 继续)、
`test_perenv_telemetry.py`、`test_strict_quadgate.py`。
接触/应力:`test_contact_force_stress.py`。求解器:`test_kick_abd_precond.py`、
`test_mas_oracle.py`。哨兵:`test_sentinel_*.py` 4 个。
(完整 30 个 `test_*.py` 见 examples/ 目录。)

---

# 附录 A. 本分册未决事项(待核实)

以下各点在写作时无法从代码/文档闭环确认,文中相应位置已标注"待核实",
引用前需人工复核:

1. **per-env 遥测在纯设备快路径下的返回值**:`get_per_env_newton_iters/status` 在
   `STIFF_PERENV_TELEM=0` 且 `env_newton_iter_cap=0` 时返回 reset 值(-1/0)——
   由"host S1 路径才填"的代码路径推理得出,未运行程序验证(§1.8)。
2. **稳定线 v0.8.5.4 的对外发布状态(已核实关闭,2026-09-08)**。提交 `dc1a297`
   (`absolute_epsv` + persistent friction anchors)、`0894958`(两者默认开,含
   CHANGELOG `[0.8.5.4]` 条目)、`c0339c8`(strict 抑制 anchor)在稳定仓 git 历史里,
   行为与旋钮已按【仅稳定线 v0.8.5.4+】写入 §2.1 / §7.4 / §7.5 与 API_CORE §2.1。
   **发布物已确认**:公开仓 `github.com/haoxiangNtu/stiff-physics` 的 Release
   `v0.8.5.4` 已正式发布(`published: 2026-08-11T17:07:37Z`,`draft:false`/
   `prerelease:false`),`stiff_physics-0.8.5.4-cp311/cp312-linux_x86_64.whl`
   两个资产均已挂出,公开仓 README 安装 URL 也已由 `a38ede4` 指向 v0.8.5.4——
   **未撤回、未重做**(OPEN_POINTS OP-001 已关闭)。稳定仓工作树曾被误留的 8 文件
   暂存回退按在 v0.8.5.3,已于 2026-09-08 恢复为 v0.8.5.4 内容;那只是工作树状态,与发布无关。
3. **`get_body_contact_force` 的 ground 口径矛盾**:Python docstring 称
   "no ground contact",但 C++ 实现(03_step_getters_export.inl:1174)实际调用了
   `computeGroundGradient`——倾向以代码为准(含 ground),文档口径待项目定夺。
4. **稳定线 getter 的 metis 解扰**:稳定线 v0.8.5.x 的 `get_vertex_positions` 是否
   与 phase-cd 一样带 perm 解扰未逐行核对(勘探只精读了 stable 的接触力/teleport/
   checkpoint 段)。
5. **`m_avg_env_bbox2` 的填充点与 merged-bbox 回退分支**:per-env 冻结阈值的
   env-bbox 来源只见消费点,`h_env_bbox2` 为空时的标量回退分支在什么配置下走到未确认。
6. **gpu_rl `d_joint_obs` 的字节级打包**:{angle, rate}/{disp, rate} 的顺序有
   docstring 与 GIPC.cuh:1126-1129 注释背书,但生成 kernel 未逐行读,字节序细节
   以 `gpu_rl_tensors()["joint_observations"]` 的扁平 float64 视图为准。
7. **`fix_obj_winding.py` 位置(已核实关闭)**:位于 `tools/fix_obj_winding.py`
   (两树均有);`case_26_perf_tuned.py` docstring 的 `examples/` 前缀已过时,文件未丢失。
