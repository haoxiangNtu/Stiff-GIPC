# StiffGIPC 仿真手册 — 总入口

> 手册基准日期:2026-09-07。
> 覆盖两条产品线:**稳定线 v0.8.5.3**(`release/stable-0.8`,tag `v0.8.5.3`,2026-08-11 发布)与
> **工程线 codex/phase-cd**(`0.8.6rc2`,HEAD `b3ab747`)。
> 除非另有标注,文件:行号出处指工程线工作树 `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph`;
> 标注 `[stable]` 前缀者指稳定线工作树 `/home/ps/Downloads/Stiff-GIPC-stable-08`。

---

## 目录

- [0. 阅读约定](#0-阅读约定)
- [1. StiffGIPC 是什么](#1-stiffgipc-是什么)
- [2. 版本导览:稳定线 vs 工程线](#2-版本导览稳定线-vs-工程线)
  - [2.1 怎么选](#21-怎么选)
  - [2.2 两线差异速览表](#22-两线差异速览表)
  - [2.3 运行时探测你在哪条线](#23-运行时探测你在哪条线)
  - [2.4 性能定位:regime 决定胜负](#24-性能定位regime-决定胜负)
- [3. 安装](#3-安装)
  - [3.1 wheel 安装(稳定线)](#31-wheel-安装稳定线)
  - [3.2 从源码构建(两线)](#32-从源码构建两线)
  - [3.3 安装排障](#33-安装排障)
- [4. 五分钟上手](#4-五分钟上手)
- [5. 手册分册导读](#5-手册分册导读)
  - [5.1 维护者:发布与门禁](#51-维护者发布与门禁)
- [6. 示例索引](#6-示例索引)
- [7. 本篇待核实项](#7-本篇待核实项)

---

## 0. 阅读约定

每个 API / 特性 / 环境变量按适用线标注,全手册统一:

| 标注 | 含义 |
|---|---|
| 【稳定线+phase-cd】 | 两线均有且行为一致(或差异已在条目内说明) |
| 【仅 phase-cd】 | 只存在于工程线 `codex/phase-cd` |
| 【仅稳定线】 | 只存在于稳定线 v0.8.5.3 |
| 【实验性,默认关】 | 存在但默认关闭,需环境变量 / 构建开关显式开启,不承诺稳定契约 |

机制描述附数学形式与 `文件:行号` 出处;性能数字一律引自仓库内证据文档
([`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md)、
[`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md)、
[`../GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md) 等),不在手册中重造。
未经代码核实且原始材料中缺失的陈述,标注"待核实"或不写。

---

## 1. StiffGIPC 是什么

StiffGIPC 是一个 **GPU 常驻的 IPC(Incremental Potential Contact)物理仿真引擎**:
整个世界——刚体铰接、四面体软体、布料/壳、接触与摩擦——在单张 NVIDIA GPU 上以
增量势能优化(投影牛顿 + 预条件共轭梯度 + 连续碰撞检测 line search)统一求解,
保证**无穿透、无翻转**的接触轨迹;Python 侧以 `stiff_physics` 包
(`Config` / `Engine` / `Robot`)提供 Isaac 风格的 `step()` 接口
(`stiff_physics/engine.py:455-465`)。上游谱系为 KemengHuang 的 GIPC
(`CLAUDE.md:49-57` 分支表:`main` = 上游只读镜像)。

### 能力清单

| 能力 | 内容 | 适用线 |
|---|---|---|
| 刚柔统一 | ABD 仿射刚体(12-DOF `q=[p; vec(A)]`)、FEM 四面体软体(稳定 Neo-Hookean;phase-cd 可编译期选 SNK1/SNK2/ARAP)、`dimensions=2` 三角壳/布料、闭合表面网格直接作 ABD(散度定理算质量,免四面体化,`examples/demo_cube_abd_obj.py`) | 【稳定线+phase-cd】 |
| IPC 接触 | log-barrier 无穿透接触、LBVH 广义相(VF/EE × DCD/CCD 四查询族)、ACCD 连续碰撞、滞后摩擦(lagged friction)、解析地面 | 【稳定线+phase-cd】 |
| 机器人 | URDF 加载(revolute/prismatic/fixed 关节)、位置/速度/力-力矩三类控制可共存、逐关节强度、primitive 碰撞代理(box/sphere/cylinder) | 【稳定线+phase-cd】 |
| 软爪抓取 | stitch 缝合弹簧 + fixed joint 组合的 finray/hybrid 混合爪;pos / stitch(形变门控)/ force(阻抗 + pinch)三种夹爪控制配方(`examples/umi_finray_lib.py`) | 【稳定线+phase-cd】 |
| 多环境 | 一个 GPU 世界装 N 份空间分隔场景,一次 `step()` 全步进;`merged`(吞吐)/ `isolated`(per-env 物理隔离 + 中途检疫)/ `strict`(逐位复现金锚;跨架构位级一致仅 phase-cd dlto,见确定性行)三执行模式 | 【稳定线+phase-cd】 |
| 触觉读数 | `get_vertex_contact_forces`(normal / friction_lagged / total,单位 N)、`get_fem_von_mises_stress`;摩擦分量修复见 §2.2 第 4 行 | 【稳定线+phase-cd】(摩擦分量修复【仅稳定线】) |
| 整帧 CUDA Graph | `STIFF_FRAME_GRAPH=1` = 审计两图事务:root 图 → 宿主边界 → terminal 图,一帧默认 2 次 `cudaGraphLaunch` + 1 个宿主边界(`frame_fsm/frame_transaction.cu:64-65`,回退重试帧追加发射);再加 `STIFF_FRAME_FULL_GRAPH=1` 才是单发射整帧图(Newton/PCG/LS 全在图内条件节点,capture 不可用自动回落两图事务,`frame_transaction.cu:2581,4583-4585`);`FrameStatus` 状态包 | 【仅 phase-cd】【实验性,默认关】 |
| episode 驻留回放 | `launch_episode_async`:整段动作序列一次上传、单次图发射消费全部帧,pinned 双槽观测 | 【仅 phase-cd】 |
| GPU 驻留 RL | `prepare_gpu_rl` 族:稳态循环 **0 H2D / 0 D2H / 零宿主同步**(nsys 佐证,`../GPU_NATIVE_RL_PLAN.md:65-71`),裸设备指针 ABI + `gpu_rl_tensors()` torch 零拷贝视图、in-stream(masked)reset | 【仅 phase-cd】 |
| checkpoint | 帧边界积分器状态存取;phase-cd 为 v2 格式(CRC-64、原子写、类型化报错),稳定线为 v1 格式,两者互不兼容 | 两线各自有(见 §2.2) |
| 确定性 | strict 模式 run-to-run 逐位金锚,两线各有锚值(稳定线为**跨环境**锚 `f7fb5a786c2d7935`,[stable] `CHANGELOG.md:58`);**跨架构**(sm_89 ≡ sm_80)位级一致仅在 phase-cd dlto 构建上实测(锚 `0544461bd82123ae`,`../RELEASE_NOTES_v0.8.6-rc1.md:132-137`)——稳定线以含 PTX 的 `"80;89;120"` 构建(JIT SASS 随驱动/架构而变),**不承诺**跨架构逐位一致 | 【稳定线+phase-cd】(run-to-run 逐位);跨架构金锚【仅 phase-cd,dlto】 |

规模与速度的量级参考(详见 [PRINCIPLES_EXECUTION.md](PRINCIPLES_EXECUTION.md)):
RTX 4090 24GB 上满抓取多环境轨迹 ~20 env(~0.8GB/env,`examples/replay_case39_multienv.py` docstring 实测);
A800 上 RL 微步经 GPU 驻留通道从 19.3 → 3.85 ms/步(**5.0×**,
`../A800_ALLEXAMPLES_TIMING_2026-08-01.md:358-364`)。

---

## 2. 版本导览:稳定线 vs 工程线

### 2.1 怎么选

| 你是谁 | 用哪条线 | 理由 |
|---|---|---|
| wheel 用户 / 需要稳定复现实验 | **稳定线 v0.8.5.3** | 唯一发布 wheel 的线;CHANGELOG 明言"不调用新 API 时轨迹与 0.8.5.2 完全一致"([stable] `CHANGELOG.md:7-13`) |
| 依赖接触**摩擦力读数**(触觉)的用户 | **稳定线 v0.8.5.3** | 摩擦读数恒零 bug 的修复(`snapshotFrictionForce`)与 `reset_transient_contact_state()` **只进了稳定线**;phase-cd 的 `get_vertex_contact_forces` friction_lagged/total 分量仍受恒零 bug 影响(未移植,见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)) |
| 引擎/求解器开发 | **工程线 phase-cd** | v0.8.6 模块化布局、knob 注册表治理、22 段门禁基建 |
| RL 训练(小帧微步、多环境吞吐) | **工程线 phase-cd** | episode / GPU 驻留 RL / torch 零拷贝只在这条线 |
| 大帧操作回放(抓布、抓杯) | 两线皆可 | phase-cd 默认(宿主)通道与 v0.8.5 每帧持平(`../SIMULATOR_EXECUTION_DESIGN.md:120-123`);**不要**在大帧场景开整帧图(墙钟 +3%~75% 场景相关,`../OPTIMIZATION_ROADMAP.md:78`;4090 典型 +10~15%,`../SIMULATOR_EXECUTION_DESIGN.md:240` 附录 D;A800 实测 +35~43%,`../A800_ALLEXAMPLES_TIMING_2026-08-01.md:366`) |

两条线的 Python 包名与模块名相同(`stiff-physics` / `stiff_physics` / `pystiffgipc`),
**不能并存于同一环境**;区分只能靠版本号或特性探测(§2.3)。

### 2.2 两线差异速览表

| # | 维度 | 稳定线 v0.8.5.3 | 工程线 phase-cd(0.8.6rc2) |
|---|---|---|---|
| 1 | 获取方式 | 发布 wheel + `release/stable-0.8` 源码 | 仅源码构建,本地分支 `codex/phase-cd`(未推送远端) |
| 2 | `importlib.metadata.version("stiff-physics")` | `0.8.5.3` | `0.8.6rc2` |
| 3 | C++ 布局 | 重构前单体:`GIPC.cu` 16884 行、`sim_engine.cu` 4534 行(wc -l 实测) | v0.8.6 模块化:`GIPC.cu` 42 行组合 TU,按序 include `gipc_modules/*.inl`(14 个,编号 00–14 缺 04,原 04 号内容已迁入 `energy/`);另有 `engine_modules/`、`energy/`、`core/`、`frame_fsm/`、`checkpoint/`、`config/knob_registry.h` 等 |
| 4 | 接触力摩擦读数 | **已修复**:`snapshotFrictionForce` 在 `updateVelocities` 前快照([stable] `GIPC.cu:16504`、`GIPC.cuh:332-341`),`get_vertex_contact_forces(components=1|2)` 返回快照值(剪切台实测滑比 0.600±0.008,[stable] `CHANGELOG.md`) | **未修复**:friction 分量步后现算 `calFrictionGradient`(`engine_modules/03_step_getters_export.inl:1690-1748`),lagged 摩擦梯度是步内位移的函数,提交后位移为零 → friction_lagged(components=1)**恒零**;total(components=2)仍重建 BVH/CP 并算 barrier 梯度,**退化为 normal-only**(friction 贡献为零,接触存在时非零——勿用 total==0 判断此 bug) |
| 5 | `reset_transient_contact_state()` | **有**:episode 就地重置清 pair/friction 镜像(`h_cpNum`/`h_cpNum_last`/`h_gpNum_last`/fric_snap)**并归零自适应 Kappa**——仅清镜像仍有实测 0.9µm 状态发散([stable] `sim_engine.cu:3620-3636`;声明 `sim_engine.h:296-312`) | **无,且无等效替代**:teleport 只重建当前 pair 集(第 6 行),不清 `*_last` lagged 镜像、不归零 Kappa(`engine_modules/03_step_getters_export.inl:2484-2585` 无一行触碰)→ phase-cd 就地 episode 重置达不到"如新进程"语义 |
| 6 | `teleport_abd_bodies` 强度 | 写 q 族 + 立即 `cal_x_from_q` 重导顶点;stale-pair 留给用户手动 reset | 同上,**外加** 检疫复活(`reviveEnv`)+ BVH 失效 + `buildBVH()`/`buildCP()` 帧入口 pair 集重建(`engine_modules/03_step_getters_export.inl:2484-2585`) |
| 7 | checkpoint 格式 | v1:magic `0x53544B50`,无版本号/校验和,不匹配 printf 后静默 return([stable] `GIPC.cu:16591-16631`) | v2:magic `STIFFCP2`、CRC-64、原子替换、本构编码进 ABI、失败抛 `CheckpointError`(`checkpoint/checkpoint_io.cu:33-53`);Python 包装 `Engine.save/load_checkpoint`。**两格式互不兼容** |
| 8 | 多引擎/多模式进程语义 | 同进程可先 strict 后 merged(模块撤回旧旗标,[stable] `engine.py:219-227`) | **模式为进程级锁**:第一个 Engine 提交模式签名,换模式构造 / 事后改 STIFF_* 旗标 → `LifecycleError`(`stiff_physics/engine.py:219-268, 486-553`);单进程单 finalized Engine 成文(`engine.py:983-991`) |
| 9 | 类型化异常 | 无(printf / `std::runtime_error`) | `StiffGIPCError` 层级:`ConfigurationError` / `GeometryError` / `CheckpointError` / `LifecycleError`(`StiffGIPC/errors.h`;`stiff_physics/__init__.py:21-33`) |
| 10 | 帧状态包 | 无 | `Engine.get_frame_status()` → `FrameStatus`(Python 绑定 39 只读字段 + `FrameResult` 枚举,`frame_fsm/frame_status.cuh:81-120`、`bindings/pystiffgipc.cu:38-93`;struct 的 `contact_class_count[4]` 未绑定到 Python) |
| 11 | episode / GPU 驻留 RL | 无 | `launch_episode_async` 族 + `prepare_gpu_rl` 族(`sim_engine.h:425-503`) |
| 12 | STIFF_* 旗标治理 | 无(typo 静默无效) | `config/knob_registry.h` 唯一登记表(170 个,另加元旋钮 `STIFF_KNOB_STRICT`),未登记旗标 stderr WARN,`STIFF_KNOB_STRICT=1` 升级为 `ConfigurationError`(`knob_registry.h:198-239`) |
| 13 | FEM 本构 | 硬编码 SNK1 | CMake `STIFFGIPC_FEM_MODEL=SNK1|SNK2|ARAP`(默认 SNK1);运行时 `stiff_physics.fem_model()` 查询(`bindings/pystiffgipc.cu:16-17`) |
| 14 | 构建 | `CMAKE_CUDA_ARCHITECTURES "80;89;120"`(含 PTX) | `STIFFGIPC_DLTO` 默认 ON → `80-real;89-real;120-real`(无 PTX JIT 回退),运行时最高 −15%、全量构建 +76%(`../RELEASE_NOTES_v0.8.6-rc1.md:124-141`) |
| 15 | `Config` 类 | 两线 Python `Config` **逐字节相同**(diff 实测)——phase-cd 未增删任何 Config 参数 | 同左 |

> 分叉点:`git merge-base` = `05c3f75`。phase-cd 分叉于 v0.8.5.3 **之前**,
> 故稳定线 v0.8.5.3 的两个 contact-IO 修复提交不在 phase-cd 历史里。
> 另:稳定线仓库 git 历史中存在 tag `v0.8.5.4`(absolute_epsv / friction_anchor 真静摩擦),
> 但**磁盘工作树内容与 tag v0.8.5.3 逐字节一致**(`git diff v0.8.5.3 --stat` 为空),
> 且 CHANGELOG 无 0.8.5.4 条目——本手册以 **v0.8.5.3** 为稳定线基准,
> v0.8.5.4 特性是否已对外发布 wheel 待核实,不在手册覆盖范围。

### 2.3 运行时探测你在哪条线

```python
# 版本号探测仅对 pip 安装的发行版有效(wheel,或源码 pip install)。
# phase-cd 的标准跑法是源码树 PYTHONPATH=.(§3.2/§6,无 pip install 步骤),
# 此时无 dist-info,本行抛 importlib.metadata.PackageNotFoundError——
# 请用下面的 hasattr 特性探测,两线两种跑法都可靠。
import importlib.metadata
print(importlib.metadata.version("stiff-physics"))  # "0.8.5.3" 或 "0.8.6rc2"

from stiff_physics import Engine, Config
eng = Engine(Config())
hasattr(eng, "reset_transient_contact_state")  # True → 稳定线 v0.8.5.3
hasattr(eng, "prepare_gpu_rl")                 # True → phase-cd
```

phase-cd 另有模块级 `stiff_physics.fem_model()` 与可导入异常类
`from stiff_physics import LifecycleError`(`stiff_physics/__init__.py:19-33`),
稳定线 `__all__` 仅 `["Engine", "Config", "Robot", "JointInfo", "Pipeline"]`。

### 2.4 性能定位:regime 决定胜负

来自配对基准的定论(出处:`../SIMULATOR_EXECUTION_DESIGN.md` §6/附录 A/D、
`../OPTIMIZATION_ROADMAP.md:75-81`):

| 负载 regime | 建议 | 依据 |
|---|---|---|
| 大帧操作回放(百 ms 级/帧) | 两线持平,走默认宿主通道,**勿开整帧图** | forcegrip 60 帧 4090:v0.8.5 115.2 → phase-cd 默认 105.5 ms/帧(−8%);图开 +10~15%(4090,附录 A/D;场景相关最高 +3%~75%,`../OPTIMIZATION_ROADMAP.md:78`;A800 +35~43%) |
| 小帧 RL 微步(ms 级/帧) | phase-cd,宿主通道已 2.0×;上 `prepare_gpu_rl` 驻留通道 5.0×(A800) | D4 铰接微步 A800:19.3 → 9.7 → **3.85 ms/步** |
| 接触升级轨迹(抓握由松到紧) | **禁止** prepare 驻留(容量档必被击穿,fail-closed 拒帧),走宿主通道 | foldshirt 第 15 帧 auto-prepare 后 25/25 帧 OVF_TRIPLETS(`../SIMULATOR_EXECUTION_DESIGN.md:81-90`) |
| 需要尾延迟可预测 | 图开可接受:方差 ±8% vs 宿主 ±21% | 附录 D(`../SIMULATOR_EXECUTION_DESIGN.md:236-257`) |

---

## 3. 安装

### 3.1 wheel 安装(稳定线)

【仅稳定线】发布 wheel 挂在公开仓 `github.com/haoxiangNtu/stiff-physics` 的 GitHub Release。
URL 模板(按 v0.8.4 发布页格式,[stable] `README.md:24-27`;v0.8.5.3 资产文件名按同模板,见 §7 待核实项 1):

```bash
# 模板
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v<版本>/stiff_physics-<版本>-cp<ABI>-cp<ABI>-linux_x86_64.whl

# v0.8.5.3,Python 3.11
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v0.8.5.3/stiff_physics-0.8.5.3-cp311-cp311-linux_x86_64.whl
# v0.8.5.3,Python 3.12
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v0.8.5.3/stiff_physics-0.8.5.3-cp312-cp312-linux_x86_64.whl
```

| 要求 | 值 | 出处 |
|---|---|---|
| OS | Linux x86_64(Ubuntu 20.04+) | [stable] `README.md:11-16` |
| Python | 3.11 / 3.12(`requires-python >= 3.11`,wheel 只出 cp311/cp312) | [stable] `pyproject.toml:10` |
| GPU 架构 | **sm_80(A800/A100)、sm_89(RTX 4090)、sm_120(RTX 5090)** | [stable] `CMakeLists.txt:16`;`CHANGELOG.md` 0.8.5 条目(wheels +sm_80)。⚠ 稳定线 README 系统需求表仍只写 sm_89/120,系 0.8.5 之前的过时残留 |
| 驱动 | 支持 CUDA 12.x 的 NVIDIA 驱动(wheel 自带引擎,无需 CUDA SDK / C++ 工具链) | [stable] `README.md:5, 14` |
| 系统库 | **`liburdfdom`**(运行时动态链接):`sudo apt install liburdfdom-dev` | [stable] `README.md:280-281`;`STIFF_PHYSICS_RELEASE_HANDBOOK.md:377` |
| Python 依赖 | `numpy`(wheel 自带声明);可视化另装 `pip install "polyscope>=2.4,<2.6" scipy`(polyscope 2.6 删除了示例用到的 imgui 接口,勿装 2.6+) | [stable] `pyproject.toml:12-19` |
| 示例附加依赖 | `pip install stiff-physics[examples]`(trimesh / meshio / usd-core / urdf-parser-py) | [stable] `pyproject.toml:22-23` |

### 3.2 从源码构建(两线)

开发构建命令(两线一致,`CLAUDE.md:19-26`):

```bash
cmake -B build -DBUILD_PYTHON_BINDINGS=ON -DBUILD_GL_VIEWER=OFF
cmake --build build -j --target pystiffgipc
# 产物:build/pystiffgipc.cpython-<ver>-x86_64-linux-gnu.so + build/libstiffgipc_core.so
PYTHONPATH=. python examples/test_fem_3d_minimal.py   # 冒烟
```

`stiff_physics` 的原生模块加载顺序:installed `_native/` → `./build_venv/` →
`./build_311/` → `./build/`(`CLAUDE.md:25-27`;可用 `STIFFGIPC_NATIVE_DIR` 覆盖)。

构建依赖:CUDA 12.x toolkit、CMake、pybind11(scikit-build-core 拉取,`pyproject.toml:1-3`)、
**liburdfdom-dev**。urdfdom 探测:优先 CMake config(要求 `TARGET urdfdom::urdf_parser`),
否则 pkg-config 回退——phase-cd 修复了 Debian liburdfdom-dev(≤3.0.x)config 缺 target 的坑
(`CMakeLists.txt:75-85`);稳定线只检查 `urdfdom_FOUND`([stable] `CMakeLists.txt:31-37`),
在该 Debian 版本上可能需手动走 pkg-config。GL viewer(`BUILD_GL_VIEWER`,默认 ON)
另需 GLEW/GLUT/OpenGL/ImGui(`CLAUDE.md:36`),Python 用户建议 OFF。

phase-cd 独有 CMake 开关:

| 开关 | 默认 | 含义 | 适用线 |
|---|---|---|---|
| `STIFFGIPC_DLTO` | ON | CUDA device LTO;架构变为 `80-real;89-real;120-real`(无 PTX 回退,新架构 GPU 需自行改架构列表重编);运行时最高 −15%,全量构建 +76%(`CMakeLists.txt:22-35`;`../RELEASE_NOTES_v0.8.6-rc1.md:124-141`) | 【仅 phase-cd】 |
| `STIFFGIPC_FEM_MODEL` | `SNK1` | 编译期四面体本构 SNK1/SNK2/ARAP(`CMakeLists.txt:57-64`) | 【仅 phase-cd】 |
| `STIFFGIPC_ENABLE_DIAGNOSTICS` | OFF | 编出 `debug_fd_*` 侵入式有限差分诊断 API | 【仅 phase-cd】【实验性,默认关】 |
| `STIFFGIPC_BVH_STACK_CAP` / `*_TRAVERSAL_AUDIT` / `*_COHERENCE_AUDIT` | 2048 / OFF / OFF | BVH 栈容量与审计构建 | 【仅 phase-cd】【实验性,默认关】 |

### 3.3 安装排障

来自 [stable] `README.md:274-281`(sm 支持列表已按 §3.1 更正):

1. **`ImportError: libstiffgipc_core.so: cannot open shared object file`** —
   wheel 自带核心库;出现此错通常说明只 clone 了仓库没有装 wheel(或源码构建的
   `build/` 不在加载路径上)。
2. **`CUDA error: no kernel image is available for execution on the device`** —
   GPU 架构不在构建列表内。wheel 支持 sm_80/89/120;其它架构需从源码构建并改
   `CMAKE_CUDA_ARCHITECTURES`(phase-cd 默认 `-real` 无 PTX 回退,更严格)。
3. **`ImportError: liburdfdom_model.so: cannot open shared object file`** —
   `sudo apt install liburdfdom-dev`。

---

## 4. 五分钟上手

【稳定线+phase-cd】最小可运行场景:一个 ABD 刚性方块叠在一个 FEM 软方块上落地。
生命周期为 **`Config` → `Engine` → `load_mesh`(可多次)→ `finalize()`(一次,之后不能再加载)
→ `step()` 循环 → getters**(`stiff_physics/engine.py:455-465` 类 docstring 同款)。
代码取材自 `examples/test_fem_3d_minimal.py`(仓库内可直接运行的回归脚本):

```python
import numpy as np
from pathlib import Path
from stiff_physics.engine import Engine, Config

# 资产目录:源码树用仓库根的 Assets/;wheel 安装则可留空(自动指向包内数据目录,
# stiff_physics/engine.py:433-437)。发布手册要求脚本显式传 assets_dir。
ASSETS_DIR = str(Path("Assets").resolve()) + "/"

cfg = Config(
    dt=0.020,                # 步长(s);默认 0.01(engine.py Config 签名)
    ground_offset=-0.5,      # 地面高度(y);默认 -1.0
    assets_dir=ASSETS_DIR,
    preconditioner_type=0,   # 0=diagonal, 1=MAS(默认);小场景 diagonal 足够
)
eng = Engine(cfg)

# 1 个 ABD 刚体方块(上,young 只影响 ABD 的仿射刚度)
T1 = np.eye(4); T1[:3, :3] *= 0.1; T1[1, 3] = 0.5
eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="ABD",
              transform=T1, young_modulus=1e8)

# 1 个 FEM 软体方块(下)
T2 = np.eye(4); T2[:3, :3] *= 0.1; T2[1, 3] = 0.0
eng.load_mesh("sim_data/tetmesh/cube.msh", dimensions=3, body_type="FEM",
              transform=T2, young_modulus=1e6)

eng.finalize()                      # 装配上 GPU;此后拓扑冻结
print("verts =", len(eng.get_vertices()))

for i in range(100):
    eng.step()                      # 一帧:Newton+PCG+CCD line search,保证无穿透

verts = eng.get_vertices()          # (V,3) float64,用户顶点序
faces = eng.get_surface_faces()     # 表面三角,配合 polyscope/USD 渲染
print("y-range:", verts[:, 1].min(), verts[:, 1].max())
```

运行(源码树,仓库根目录):

```bash
PYTHONPATH=. python examples/test_fem_3d_minimal.py
```

要点与陷阱:

- `load_mesh(mesh_path, dimensions, body_type, transform, young_modulus, boundary_type, density)`:
  `.msh` 四面体网格用 `dimensions=3`;`.obj` 布料/壳用 `dimensions=2`;
  `body_type="ABD"|"FEM"`;相对路径先对 `assets_dir` 解析(`stiff_physics/engine.py`
  `load_mesh` 实现)。闭合三角 `.obj` 也可直接 `body_type="ABD"`(免四面体化,
  `examples/demo_cube_abd_obj.py`)。
- **`finalize()` 之后不能再加载 body**;phase-cd 上一个进程同时只能有一个
  finalized Engine(`stiff_physics/engine.py:983-991`,违者 `LifecycleError`)。
- 冲击/动态接触场景**不要开 `semi_implicit_enabled`**(会在接触完全解出前提前
  终止 Newton,`examples/demo_cube_abd_obj.py:70-74` 注释)。
- 常用 Config 起步值(完整参数表见 [API_CORE.md](API_CORE.md);默认值均核对自
  `stiff_physics/engine.py` Config 签名,两线逐字节相同):

| 参数 | 默认值 | 单位 | 含义 |
|---|---|---|---|
| `dt` | 0.01 | s | 步长;抓取家族示例常用 0.02(U 形扫描峰值吞吐 1.21×,见 [stable] `README.md:213-224`;更高的 1.43× 系 case_26 β_tol+newton_tol+dt 三项累积调参的总吞吐,[stable] `README.md:176-209`,非 dt 单项) |
| `density` | 1e3 | kg/m³ | FEM 体密度 |
| `young_modulus` | 1e7 | Pa | 全局默认杨氏模量(`load_mesh` 逐 body 覆盖) |
| `friction_rate` | 0.4 | — | Coulomb 摩擦系数(`gd_friction_rate=None` 时地面跟随此值) |
| `newton_tol` | 1e-2 | — | Newton 退出容差 |
| `newton_iter_cap` | 1000 | — | Newton 上限(示例常压到 50) |
| `pcg_tol` | 1e-4 | — | PCG 容差(0.8.2 起回退默认;1e-6 在 N=1 代价 ~22%) |
| `relative_dhat` / `absolute_dhat` | 1e-3 / 0.0 | — / m | 接触激活距离(相对场景对角线 / 绝对值;多环境建议钉 absolute,见 [API_CORE.md](API_CORE.md)) |
| `preconditioner_type` | 1 | — | 1=MAS,0=diagonal(场景相关,`../OPTIMIZATION_ROADMAP.md:50-60`) |
| `gravity` | (0, −9.8, 0) | m/s² | Y-up |
| `ground_offset` | −1.0 | m | 解析地面高度 |
| `multienv_mode` | `"merged"` | — | `merged` / `isolated` / `strict`(见 [API_EXECUTION.md](API_EXECUTION.md)) |

下一步:
- 带 GUI 的机器人 + 布料:`python examples/case_26_arm_cloth_semi_implicit.py`
- 软爪抓取主线(推荐):§6 的 UMI finray 套件
- 多环境与 RL:[API_EXECUTION.md](API_EXECUTION.md)

---

## 5. 手册分册导读

| 分册 | 一句话导读 |
|---|---|
| [API_CORE.md](API_CORE.md) | 核心 API 逐条参考:`Config` 全参数表、场景构建(`load_mesh` / `load_urdf` / stitch / fixed joint / 碰撞排除)、关节与力控、状态读取(顶点/接触力/应力/BodyView)、teleport 与 per-body 参数 |
| [API_EXECUTION.md](API_EXECUTION.md) | 执行面 API:`step()` 语义、多环境三模式与 per-env 检疫/遥测、`FrameStatus`、episode 驻留回放、GPU 驻留 RL 设备 ABI 与 torch 零拷贝、checkpoint(v1/v2)、异常层级 |
| [PRINCIPLES_DYNAMICS.md](PRINCIPLES_DYNAMICS.md) | 动力学原理:增量势能形式、ABD 12-DOF 仿射动力学、FEM 本构(SNK1/SNK2/ARAP)、布料/弯曲、关节约束能量与外力注入(q_tilde 路径) |
| [PRINCIPLES_CONTACT.md](PRINCIPLES_CONTACT.md) | 接触原理:IPC log-barrier 与 dHat/kappa 自适应、LBVH 四查询族、ACCD 与 line search、滞后摩擦模型及其读数陷阱、地面接触 |
| [PRINCIPLES_EXECUTION.md](PRINCIPLES_EXECUTION.md) | 执行架构:一个表面两个内核(step 通道 vs residency 通道)、整帧 CUDA Graph 与容量档边界协议、确定性栈与 strict 金锚、性能成本结构与已裁决优化 |
| [CHANGELOG_TIMELINE.md](CHANGELOG_TIMELINE.md) | 版本时间线:v0.8.5 → v0.8.5.3 / 0.8.6rc2 的行为变更与性能里程碑(合并两树 CHANGELOG,标注每项落在哪条线;更早历史见下方指路) |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已知问题与坑:phase-cd 摩擦读数恒零、进程级模式锁、checkpoint 不兼容、EE mollifier 关闭、图开负收益场景、诊断类 API 的破坏性等 |
| [OPEN_POINTS.md](OPEN_POINTS.md) | 待核实项唯一登记表:全手册约 45 处"待核实"标记收拢为 OP-001 起的 37 条编号条目(含验证方法建议);关闭时引用编号 |

引擎内部证据文档(手册数字的一手出处,在 `../`):
[`OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md)(成本结构与立项/死路表)、
[`SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md)(执行架构总纲)、
[`GPU_NATIVE_RL_PLAN.md`](../GPU_NATIVE_RL_PLAN.md)(GPU-native RL 契约与 nsys 证据)、
[`PHYSICS_VALIDATION.md`](../PHYSICS_VALIDATION.md)(物理/导数/checkpoint 验证)、
[`CI.md`](../CI.md)(22 段门禁体系)。

pre-v0.8.5 早期版本史([CHANGELOG_TIMELINE.md](CHANGELOG_TIMELINE.md) 覆盖窗 v0.8.5 起,更早不重述):
0.1.0 → 0.8.5.1 的完整逐条目在工程线树根 [`CHANGELOG.md`](../../CHANGELOG.md),
0.8.5.2 / 0.8.5.3 条目在稳定线树根 [stable] `CHANGELOG.md`。

### 5.1 维护者:发布与门禁

手册暂无独立发布章,维护者直接用仓库根的既有材料:

- **发布 / wheel 打包**:[`STIFF_PHYSICS_RELEASE_HANDBOOK.md`](../../STIFF_PHYSICS_RELEASE_HANDBOOK.md)
  ——§2 公开仓 `haoxiangNtu/stiff-physics` 的 commit 作者铁律(release 必须以 `haoxiang002` 身份提交)、
  §3 wheel 构建(scikit-build-core 后端,`pip wheel . --no-build-isolation -w dist/`;
  cp311 / cp312 双 ABI 各自在对应 Python 虚拟环境重跑一次)、
  §4.2 发布流程(强制 **commit → tag → 从 tag `git worktree` 干净构建 → `gh release create`**;
  只有 `release/stable` / `release/stable-X.Y` 允许打 release tag)。
  ⚠ 引用前注意其**三处已知过时**(见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §5 第 5 条):
  CUDA 架构默认已是 `80;89;120`;GIPC.cu 行数是重构前旧值;
  "自适应 Kappa 未启用"的表述对 phase-cd 已过时(phase-cd 有图内 kappa 链,默认 host-equivalent)。
- **门禁**:改代码后跑 [`scripts/verify_gates.sh`](../../scripts/verify_gates.sh)
  (22 段全量;`verify_gates.sh quick` 跳过慢速 towel recipe,**不足以支撑 push**,脚本头注同述);
  pre-push 钩子一次性安装 [`scripts/install_hooks.sh`](../../scripts/install_hooks.sh)
  (装入 git common dir 并设 `core.hooksPath`)。体系说明见 [`../CI.md`](../CI.md),
  金锚约束见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §4.4。

---

## 6. 示例索引

示例在仓库根 `examples/`(phase-cd 91 个 `.py`,稳定线 89 个;
仅 phase-cd:`replay_foldshirt_stats.py`、`test_abd_badmesh_kinetic.py`,
另有 3 个文件带 phase-cd 增量旋钮——`umi_finray_lib.py`、`replay_foldshirt_multienv.py`、
`recipe_towel_scramble.py`;其余 86 个两线字节级相同)。

**标准跑法**:仓库根目录 `PYTHONPATH=. python examples/<file>.py`
(个别老 docstring 里的 `./run` 前缀是历史工作树残留,已不存在)。
示例 docstring 里常见的 `STIFF_SKIP_CCD_SANITY=1` 是历史残留,**勿再设置**:
line-search 尾部 CCD sanity 复检两线**默认已跳过**,引擎不再读取该变量
(唯一读取的是反向调试旋钮 `GIPC_FORCE_CCD_SANITY=1`,
`gipc_modules/14_energy_linesearch_solver.inl:556-574`;[stable] `GIPC.cu:14894-14899` 同),
设了纯属无效——且它不在 phase-cd knob 登记表内,finalize 时触发 stderr WARN,
`STIFF_KNOB_STRICT=1` 下直接 `ConfigurationError`(§2.2 第 12 行)。
逐帧计时加 `STIFF_BENCH_STATS=1`(示例脚本 Python 侧读取,有效;但同样未登记,
phase-cd 上会 WARN);静音默认开(`CASE39_QUIET=1`)。

### 按任务分组

| 任务 | 入口 | 说明 |
|---|---|---|
| **最小上手** | `test_fem_3d_minimal.py` → `demo_cube_abd_obj.py` → `demo_bunny_abd_obj.py` → `demo_body_view.py` → `demo_loglevel_reset.py` | FEM+ABD 最小场景;表面 OBJ 直接作 ABD;BodyView 逐 body 读取;`set_log_level` / `reset()` |
| **机器人 + 布料** | `case_26_arm_cloth_semi_implicit.py`、`case_26_perf_tuned.py`、`case_27_ridgeback_panda_cloth.py`、`case_27_mobile_s1_*.py` | GUI 滑条交互;调参版(`joint_strength_ratio=100`);移动双臂 + 布 |
| **软爪抓取(推荐主线)** | `replay_{foldshirt,beaker,cupshirt}_finray.py`(单环境)、`..._finray_multienv.py`(默认 4 env)、`ui_{...}_finray.py`(交互)、`diag_finray_grip.py`(headless 判定);共享库 `umi_finray_lib.py` | 三场景 × replay/UI × 单/多环境;`GRIP_MODE=pos|stitch|force` 三种夹爪控制;`diag_finray_grip.py` 每次调用须独立进程 |
| **多环境正主** | `replay_foldshirt_multienv.py`(重点,`CASE39ME_*` 全旋钮)、`replay_case39_multienv.py`(docstring 是多环境机理说明)、`replay_foldshirt_stats.py`【仅 phase-cd】 | strict 位移三件套、checkpoint 快启、MP4/USD 录制、整帧图审计(`CASE39_GRAPH_STATS=1`【仅 phase-cd】) |
| **力/力矩控制教学** | `case_force_*.py` 共 9 个(cube 外力、12-DOF wrench、revolute 扭矩、prismatic 力、速度控制、混合控制、双臂) | 每个演示一个 API 概念;headless 回归对应 `test_force_control.py` 等 |
| **多环境规模基准** | `duck_multienv.py`(uipc 512-env 对标)、`duck_grasp_multienv.py`(FR3 抓鸭) | env 上限 / 显存曲线;PD 目标驱动(`set_revolute_target` / `set_prismatic_target`,`stiff_physics/engine.py:1755,1761`;duck_grasp docstring 里的 `set_joint_target` 是二者的口头简称,引擎无此 API 名) |
| **初始状态变异配方** | `recipe_towel_scramble.py` | SCRAMBLE→SAVE→REUSE 三段;必须用 `teleport_fem_vertices(positions, velocities)` 注入速度(单独 `set_vertex_velocities_gpu` 不重建 xTilta) |
| **性能/调参复现** | `bench_case26_simple.py`、`bench_obb_perf.py`、`bench_s1_hybrid.py`、`precond_ab.py`、`repro_*.py` 5 个 | wall-time 基准;MAS vs diagonal A/B;joint strength 扫描 |
| **回归门禁(特性有测试背书)** | `test_*.py` 共 30 个(稳定线 29;差 = `test_abd_badmesh_kinetic.py`) | 多环境隔离/检疫/strict 五门(`test_strict_quadgate.py`)、per-body 参数、力控、闭环四杆、接触力/应力、M0 物理哨兵等 |
| **旧线对照(force-control 演化史)** | `replay_case39_UMI_*.py`、`case_umi_finray_*ui*.py` | 保留作对照;其 barrier 方案已被现行 impedance+pinch 取代,勿作新代码范本 |

完整逐文件清册、GRIP_* 旋钮全表与三模式选择指引见
[API_EXECUTION.md](API_EXECUTION.md) 与 `examples/UMI_FINRAY_NOTES.md`
(⚠ NOTES 的 force 模式细节段描述旧实现,以 `umi_finray_lib.py` 代码为准)。

已知不一致(引用示例文档时注意):
- `case_26_perf_tuned.py` docstring 要求先跑 `examples/fix_obj_winding.py`,
  路径前缀过时:该文件实际在两树的 `tools/fix_obj_winding.py`(亲验存在),
  只是从未搬进 `examples/`。
- `replay_case39_UMI_obb_cup_shirt_forcegrip.py` 的 `CASE39_GRIP_MODE` 代码默认
  `"trackgrip"`,与 docstring 的五模式列表不一致(待核实)。

---

## 7. 本篇待核实项

1. v0.8.5.3 wheel 的 Release 资产文件名按 v0.8.4 发布页模板推断
   (cp311/cp312 已由发布记录确认),下载 URL 未逐字节验证。
2. 稳定线仓库历史中的 tag `v0.8.5.4`(absolute_epsv / friction_anchor 真静摩擦)
   是否已对外发布 wheel 未验证;磁盘工作树 = v0.8.5.3,本手册不覆盖 v0.8.5.4 行为。
3. ~~`examples/fix_obj_winding.py` 的实际位置~~ **已关闭**:文件在两树的
   `tools/fix_obj_winding.py`(亲验存在);`case_26_perf_tuned.py` docstring 的
   `examples/` 前缀过时(§6)。
4. `replay_case39_UMI_obb_cup_shirt_forcegrip.py` 默认模式 `"trackgrip"` 的语义。

---

*手册维护约定:数字一律引用 `docs/` 内证据文档并附行号;两线行为差异一律进 §2.2
速览表并在对应分册展开;新增 API 必须标注适用线。*
