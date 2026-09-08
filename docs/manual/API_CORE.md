# StiffGIPC 手册 · 核心 API 参考(API_CORE)

> 本分册是 StiffGIPC 用户手册的核心 API 卷:分层模型、`Config` 全参数、场景构建、关节与耦合、per-body 材料、步进与生命周期、状态读取、接触力读出、teleport 族、native-only API 清册。
> 求解器内部机制(Newton/PCG/CCD/κ)、多环境三模式契约、CUDA Graph 与 GPU 驻留 RL、环境变量旋钮全表等主题在同目录其余分册中展开;性能定位与相位分解请引用 [`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md) 与 [`../SIMULATOR_EXECUTION_DESIGN.md`](../SIMULATOR_EXECUTION_DESIGN.md)。

## 版本口径与标注约定

本手册同时覆盖两条产品线,所有 API/特性条目都标注适用线:

| 标注 | 含义 |
|---|---|
| 【稳定线+phase-cd】 | 两条线都有,签名与语义一致(个别行为差异在条目内单独说明) |
| 【仅 phase-cd】 | 只在工程线 phase-cd 存在 |
| 【仅稳定线】 | 只在稳定线 v0.8.5.3 存在 |
| 【实验性,默认关】 | 存在但默认关闭,需环境变量/显式调用启用,契约可能变动 |

两条线的定义:

| | 稳定线 | 工程线 |
|---|---|---|
| 仓库 | `/home/ps/Downloads/Stiff-GIPC-stable-08`,分支 `release/stable-0.8` | `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph`,分支 `codex/phase-cd`,HEAD `b3ab747` |
| 版本 | **tag `v0.8.5.3`**(commit `b8e27a1`,2026-08-11 发布;本册 `[stable]` 行号按该 tag 的 blob,复核用 `git show v0.8.5.3:<文件>`;tag 树 `pyproject.toml` version=`0.8.5.3`)。稳定线仓分支 HEAD = `c0339c8` = tag `v0.8.5.4`(2026-08-12,较 v0.8.5.3 +511 行:`absolute_epsv` 旋钮、持久摩擦锚/默认开真静摩擦,**摩擦行为有变**);工作树已于 2026-09-08 恢复为 HEAD(v0.8.5.4)内容——此前 8 个文件被某次会话误留的暂存回退按在 v0.8.5.3,非 owner 本意,已 `reset --hard`(回退补丁留有备份),**勿按磁盘行号复核 v0.8.5.3 引用** | `0.8.6rc2`(`pyproject.toml:7`) |
| 发布形态 | 公开仓 `github.com/haoxiangNtu/stiff-physics` 挂 cp311/cp312 wheel,CUDA 架构 sm_80/89/120 | 仅源码构建(本地分支,未推送远端) |
| C++ 布局 | 重构前单体(`StiffGIPC/GIPC.cu` 16884 行、`sim_engine.cu` 4534 行) | v0.8.6 模块化:`GIPC.cu`/`sim_engine.cu` 仅为按序 include `gipc_modules/`、`engine_modules/` 等 `.inl` 的组合 TU |
| 内容差异 | 含 tactile 线两个修复(§8.1、§9.5) | 含 v0.8.5 之后全部工作:整帧 CUDA Graph、GPU 驻留 RL、episode、checkpoint v2、类型化异常、进程级模式锁等 |

两条线分叉点为 `05c3f75`(v0.8.5.3 之前)——**v0.8.5.3 的两个 contact-IO 修复(摩擦读数快照、`reset_transient_contact_state`)不在 phase-cd 历史里,phase-cd 未移植**(§8.1、§9.5 详述)。两个包的 Python 包名/模块名相同(`stiff_physics` / `pystiffgipc`),不能并存于同一环境;运行时判别:

```python
import stiff_physics, importlib.metadata
importlib.metadata.version("stiff-physics")        # "0.8.5.3" vs "0.8.6rc2"
hasattr(eng, "reset_transient_contact_state")      # True = 稳定线
hasattr(eng, "prepare_gpu_rl")                     # True = phase-cd
```

**行号约定**:未注明 "stable" 的 `文件:行号` 一律指 phase-cd 树;稳定线出处以 `stable <文件>:<行号>` 标注。两树 Python `Config` 类经 diff 验证逐字节相同,`SimEngineConfig` 全部字段与默认值一致(stable 行号 = phase-cd 行号 − 44,对 engine.py 的 Config 段)。

---

## 目录

1. **分层模型** — Config/Engine/BodyView 与 `eng.native`;原生模块加载;异常体系
2. **Config 完整参数表** — 34 显式参数 + kwargs 透传 native-only 字段;进引擎后的派生公式;陷阱
3. **场景构建** — load_urdf / load_mesh / load_mesh_from_data / load_mesh_instanced;body_type×dimensions 组合语义;boundary_type;load records 与 BodyView;碰撞域控制
4. **关节与耦合** — fixed/revolute/prismatic 关节;drive 目标与 strength 族;限速;stitch spring 与 FEM-pin
5. **per-body 材料** — 摩擦/密度/质量/惯量/逐 tet 杨氏模量
6. **步进与生命周期** — Engine 构造、finalize、step、reset、进程级模式锁、checkpoint、异常与 GIL
7. **状态读取** — 顶点/速度/表面;ABD 位姿速度;关节角三件套;遥测
8. **接触力读出** — get_vertex_contact_forces 三分量与单位契约;legacy −f·dt² 族;两线摩擦读数差异
9. **teleport 族与瞬态接触状态** — xTilta 语义;teleport_abd_bodies/teleport_fem_vertices;reset_transient_contact_state(仅稳定线)
10. **native-only API 清册** — 全部须 `eng.native.xxx` 直呼的方法,按功能分组
11. **便利层:Robot 与 Pipeline** — 按名/序号存取关节;deg/mm 单位换算与限位夹紧;polyscope 运行循环 `run()`/`user_gui()`
12. **资产与场景工具链** — `stiff_physics` 辅助模块(USD/URDF/轨迹/网格)与 `tools/` 脚本的存在与支持状态;`build_hybrid_mesh.py` 断链
13. **附录** — 抛错条件汇总;engine.py 环境变量清单;调用时序速查

---

## 1. 分层模型

### 1.1 三层结构 【稳定线+phase-cd】

```
┌─────────────────────────────────────────────────────────┐
│ stiff_physics.Engine / Config / BodyView   (Python 包装) │  engine.py
│   便捷签名、字符串枚举映射、numpy 视图、纯 Python 增值功能   │
├─────────────────────────────────────────────────────────┤
│ pystiffgipc (pybind11 模块, 别名 _C)                     │  bindings/pystiffgipc.cu
│   Config / SimEngine / JointInfo / BodyLoadRecord /      │
│   MeshAsset / InstancedLoadResult (+phase-cd: FrameStatus│
│   / FrameResult / 异常类 / fem_model)                    │
├─────────────────────────────────────────────────────────┤
│ gipc::SimEngine (native C++, pImpl, 不可拷贝)            │  StiffGIPC/sim_engine.h:144-758
└─────────────────────────────────────────────────────────┘
```

- `Engine` 内部持 `self._engine = _C.SimEngine()`;属性 **`Engine.native`**(engine.py:555-557)裸露 pybind 对象。**凡包装层没有的方法,一律写 `eng.native.xxx(...)`** —— 完整清册见 §10。
- `Config.native`(engine.py:446-448)同理裸露 `_C.Config`;Python 构造器没有的 native 字段既可经 `Config(**kwargs)` 透传(§2.3),也可事后 `cfg.native.xxx = v` 直改。
- `BodyView`(§3.6)是某 body 在全局数组中切片的只读视图,经 `Engine.get_bodies()/get_abd_body()/get_fem_body()` 获取,不要直接构造。

### 1.2 包级导出与 fem_model

phase-cd `stiff_physics/__init__.py:4-49` 用模块级 `__getattr__` 惰性导出:

| 导出 | 来源 | 适用线 |
|---|---|---|
| `Engine`, `Config` | `stiff_physics.engine` | 【稳定线+phase-cd】 |
| `Robot`, `JointInfo` | `stiff_physics.robot` | 【稳定线+phase-cd】 |
| `Pipeline` | `stiff_physics.pipeline` | 【稳定线+phase-cd】 |
| `fem_model` | `_C.fem_model` — **函数**,返回编译期四面体本构模型名 `"SNK1"`/`"SNK2"`/`"ARAP"`(bindings/pystiffgipc.cu:16-17;CMake 选项 `STIFFGIPC_FEM_MODEL`,默认 SNK1;稳定线硬编码 USE_SNK1) | 【仅 phase-cd】 |
| `StiffGIPCError`, `ConfigurationError`, `GeometryError`, `CheckpointError`, `LifecycleError` | `_C`(异常类) | 【仅 phase-cd】 |

稳定线 `__all__ = ["Engine", "Config", "Robot", "JointInfo", "Pipeline"]`(stable `__init__.py:22`),无 fem_model、无异常类导出。

### 1.3 原生模块加载 `_import_native()` 【稳定线+phase-cd】

engine.py:16-82,优先级:

1. 环境变量 **`STIFFGIPC_NATIVE_DIR`** 指定目录(engine.py:27-37;指错抛 `ImportError`);
2. 已安装 wheel:`stiff_physics._native.pystiffgipc`(engine.py:40-45,置 `_INSTALLED_MODE=True`,此时包内 `data/` 目录成为默认 assets_dir);
3. 开发构建目录:项目根下 `build_{py_tag}`(如 `build_312`)优先于 `build/`,且要求扩展文件匹配当前解释器 `EXT_SUFFIX`(engine.py:54-76;防止 py3.11/3.12 混存时加载陈旧扩展破坏 `SimEngineConfig` 内存布局);
4. 全部失败抛 `ImportError`,提示安装 wheel 或 `cmake -DBUILD_PYTHON_BINDINGS=ON .. && make pystiffgipc`。

### 1.4 异常体系 【仅 phase-cd】

`StiffGIPC/errors.h` + pybind 注册(bindings/pystiffgipc.cu:18-27):

```
gipc::StiffGIPCError : std::runtime_error      (带 ErrorCode code())
 ├── ConfigurationError    (ErrorCode::configuration)
 ├── GeometryError         (ErrorCode::geometry)
 ├── CheckpointError       (checkpoint_io / checkpoint_format)
 └── LifecycleError        (ErrorCode::lifecycle)
```

Python 侧四个子类均以 `StiffGIPCError` 为基类,可 `from stiff_physics import LifecycleError` 捕获。

> **陷阱(稳定线)**:稳定线的 pybind 绑定**完全没有** `register_exception`(grep 验证 0 命中)——native 错误只能以 pybind 默认异常(通常 `RuntimeError`)浮出,部分错误路径是 `printf` + 返回甚至 `exit(-1)`(见 §3.2 文件缺失)。跨线可移植的错误处理请勿依赖异常类型精确匹配。

---

## 2. Config 完整参数表

`stiff_physics.Config`(engine.py:323-451)镜像 native `gipc::SimEngineConfig`(StiffGIPC/sim_engine.h:18-99)。**两树 Python Config 类逐字节相同,native 结构体字段与默认值也完全一致**(header diff 为空)。

### 2.1 Python 构造器显式参数(34 个两线共有 + 2 个仅稳定线 v0.8.5.4)【稳定线+phase-cd】

| 参数 | 默认值 | 单位 | 物理含义 / 进引擎后的派生 |
|---|---|---|---|
| `dt` | `0.01` | s | 时间步长 → `ipc.IPC_dt`(engine_modules/01_config_upload.inl:13)+ ABD parms(02_finalize_nandiag.inl:34) |
| `density` | `1e3` | kg/m³ | 全局 FEM 体密度;tet 质量装配 `m += vlm·ρ/4`(01:60-70)。布料另有 `cloth_density`(kwargs) |
| `young_modulus` | `1e7` | Pa | 全局 FEM 杨氏模量 E;派生 Lamé:μ = E/(2(1+ν)),λ = Eν/((1+ν)(1−2ν));`lengthRate = 4μ/3`、`volumeRate = λ + 5μ/6`(01:41-45)。**实际逐 tet 存储**(per-tet 数组,01:77-81),全局值只是兜底 |
| `poisson_rate` | `0.49` | — | 泊松比 ν |
| `friction_rate` | `0.4` | — | 体-体摩擦系数 μ;同时是 per-body 摩擦表的默认底色(01:214-215) |
| `gd_friction_rate` | `None` | — | 地面摩擦系数;**None = 跟随 `friction_rate`**(历史行为;A2 wrapper bug 修复,engine.py:333-338, 401-402)。native 默认 0.4 |
| `newton_tol` | `1e-2` | — | legacy Newton 退出:`sqrt(newton_tol²·bbox²·dt²)` 位移判据(core/ipc_solver.inl:1561) |
| `newton_velocity_tol` | `0.0` | m/s | 【opt-in,uipc 风格】物理退出:max 步位移 ≤ tol×dt(uipc 参考默认 0.05,来自注释,待核实);0 = legacy 判据。开启后 `relative_dhat` 对退出判据惰性(sim_engine.h:33-38) |
| `pcg_tol` | `1e-4` | — | PCG 容差。[0.8.2] 回到 0.6.x 默认;1e-6 在 N=1 代价 ~22% 无精度收益;刚接触场景可 1e-8(env `STIFF_PCG_TOL` 亦可) |
| `relative_dhat` | `1e-3` | — | 相对接触距离(乘场景 bbox 对角);dHat 派生见 §2.4 |
| `absolute_dhat` | `0.0` | m | >0 时 **dHat = absolute_dhat²** 钉死,不随合并 bbox/env 数膨胀(多环境正确性关键);0 = legacy bbox 派生。**仅当 `relative_dhat > 0` 时生效**(§2.4) |
| `absolute_epsv` 【仅稳定线 v0.8.5.4+;phase-cd 无】 | **`1e-4`**(v0.8.5.4 起,stable@c0339c8 engine.py:321;dc1a297 首版为 `0.0`) | m/s | IPC 摩擦 stiction 阈值 epsv 钉成绝对值:`fDhat = eff_epsv²`,eff_epsv ≤ 0 时才回落 legacy 的 `fDhat = 1e-4·eff_bboxDiagSize2`(即 epsv = sqrt(fDhat) = 1e-2·eff_scene_diag,1.9 m 场景 ≈ 19 mm/s)——stable@c0339c8 GIPC.cu:9498-9500。epsv 是接触物理不是几何:legacy 编码使**静摩擦精度随场景 bbox 变化**,持握 creep ≈ (load/(μ·λ))·epsv(§2.2 的 `fDhat = 1e-4·eff` 是这条 legacy 分支)。IPC 原文口径:默认 ~1e-3·l、静摩擦精度取 1e-5 m/s,epsv 越小 Newton 越贵(实测 1e-5 +12% step、1e-4 无可测成本)。逃生阀:`absolute_epsv=0` 或 `STIFF_EPSV=0` |
| `friction_anchor` 【仅稳定线 v0.8.5.4+;phase-cd 无】 | **`True`**(v0.8.5.4 起,stable@c0339c8 engine.py:326;dc1a297 首版默认关) | — | 持久跨步摩擦锚(真静摩擦):每个 lagged 接触对携带累计切向弹性偏移 e,能量/梯度/Hessian 按 `u_total = relDX_step + e` 求值,静接触收敛到常值 u_total 而不再每步重新滑移;‖e‖ 封顶在 stiction 边界 `eps = sqrt(fDhat)·h`,触顶即 Coulomb 滑动(径向回拉)。锚按规范 pair key 跨步匹配(bit-pack int4 + cub merge-sort + 二分,确定性),地面对走逐顶点稠密数组;未匹配的新接触从 e=0 起(即 legacy 行为);`reset_transient_contact_state()` / teleport 清锚(stable@c0339c8 sim_engine.cu:3638-3640)。实现出处 stable@c0339c8 GIPC.cu:9615-9622(设计注释)、:9793(`clearFrictionAnchors`)、:9801(`carryFrictionAnchors`)、:9999(建对后每步调用)。**strict 多环境下默认被抑制**(批不变性,见 API_EXECUTION §2.1)。逃生阀:`friction_anchor=False` 或 `STIFF_FRIC_ANCHOR=0` |
| `joint_strength_ratio` | `100.0` | — | fixed/URDF 关节约束刚度系数:`kappa = ratio·(m_parent+m_child)`,**无 dt² 因子**(setup_abd_system_gradient_and_hessian.cu:1245-1260) |
| `revolute_driving_strength_ratio` | `100.0` | — | revolute 位置伺服刚度系数;有效 K = ratio·per-joint strength·(m_p+m_c)(§4.5) |
| `semi_implicit_enabled` | `False` | — | 半隐式模式开关 |
| `semi_implicit_beta_tol` | `1e-3` | — | 半隐式 β 容差 |
| `semi_implicit_min_iter` | `1` | — | 半隐式最小迭代数 |
| `newton_iter_cap` | `1000` | — | Newton 迭代上限 |
| `energy_abs_tol` | `0.0` | — | 线搜索能量接受带:E1 ≤ E0 + abs + rel·\|E0\|;默认严格非增(对齐 libuipc)。**负值抛 `ValueError`**(engine.py:424-425) |
| `energy_rel_tol` | `0.0` | — | 同上相对项;负值抛 `ValueError` |
| `skip_all_collision` | `False` | — | 跳过全部碰撞(碰撞缓冲容量退化为 1,01:595-596) |
| `preconditioner_type` | `1` | — | 1 = MAS(默认;影响 metis_sort 与索引重映射,§3.2/§4.6);0 = 关 |
| `cuda_device` | `0` | — | CUDA 设备号;**进程级**——同进程第二个引擎请求不同设备抛 `LifecycleError`(engine_modules/00_impl_api_surface.inl:373-400) |
| `assets_dir` | `""` | — | 资产目录。空 + wheel 安装模式 + 包内 `data/` 存在 → 自动设 `<package>/data/`(engine.py:433-436);native 侧空则用编译期 `GIPC_ASSETS_DIR` |
| `prismatic_strength_ratio` | `100.0` | — | prismatic 约束刚度系数 |
| `prismatic_driving_strength_ratio` | `100.0` | — | prismatic 驱动刚度系数 |
| `max_revolute_step_per_frame` | `0.1` | rad/帧 | revolute 驱动目标 slew 限速(≈5.7°/帧);精细 FEM softpad 场景建议 0.01(sim_engine.h:703-708);运行时可改(§4.7) |
| `max_prismatic_step_per_frame` | `0.002` | m/帧 | prismatic 驱动目标 slew 限速 |
| `gravity` | `(0.0, -9.8, 0.0)` | m/s² | 重力(**Y-up 约定**) |
| `ground_normal` | `(0.0, 1.0, 0.0)` | — | 地面法向 |
| `ground_offset` | `-1.0` | m | 地面平面沿法向的偏移 |
| `velocity_damping` | `0.0` | — | 每步 `v *= (1−damping)`(sim_engine.h:91)。只见 ABD parms 消费点(02:36);FEM 顶点是否受 damping **待核实** |
| `multienv_mode` | `"merged"` | — | 多环境档位 `"merged"/"isolated"/"strict"`(别名 `0/a`、`1/decoupled/b`、`2/deterministic/c`);**Python 包装层属性,非 native 字段**,`Engine()` 时解析为 STIFF_* 环境变量;`STIFF_MULTIENV_MODE` env 覆盖之。详见多环境分册 |
| `collision_detection_buff_scale` | **`6.0`(Python)** | — | 初始 DCD pair 缓冲容量倍率。**⚠ native 默认是 1.0(sim_engine.h:53),Python 包装层默认 6.0**(engine.py:378-381)——绕过 Python Config 直接构造 `_C.Config()` 的用户拿到的是 1.0,启动期 grow-redo 次数会显著变多。溢出自愈(1.5× 增长并重做检测),该参数只权衡启动开销 vs 显存 |
| `per_env_exit` | `False` | — | 【多环境】per-env 解耦退出:各 env 按自身判据收敛并冻结/掩出;解析为 `STIFF_DECOUPLE_THRESH + STIFF_PERENV_ALPHA + STIFF_PERENV_MASK (+ STIFF_PERENV_TELEM)`(engine.py:516-526);显式 env var 仍赢。多环境生产推荐 `per_env_exit=True, env_newton_iter_cap=100`(CHANGELOG 0.8.5) |
| `**kwargs` | — | — | 透传:native `_cfg` 有该属性名则 `setattr`,**无则静默忽略,无警告**(engine.py:442-444)——拼错参数名不会报错,是常见事故源 |

构造细节:`newton_velocity_tol`/`absolute_dhat`/`max_*_step_per_frame` 用 `hasattr` 守卫写入(兼容旧 native);`gravity`/`ground_normal` 转 float64 np.array 写入。`Config.__repr__` 只显示 dt 与 density。`absolute_epsv`/`friction_anchor` 同样是 `hasattr` 守卫写入(stable@c0339c8 engine.py:388-391)——**这正是它们在旧 native 上静默失效的机制**,见下。

> **v0.8.5.4 摩擦默认值变更(升级 wheel 前必读)**【仅稳定线 v0.8.5.4+;phase-cd 无】
>
> 稳定线在 v0.8.5.3 之后把两个摩擦旋钮**同时翻成默认开**(`dc1a297` 引入、`0894958` 翻默认、`c0339c8` 在 strict 抑制 anchor):`Config.absolute_epsv` 由 `0.0`(场景派生)改为 **`1e-4` m/s**,`Config.friction_anchor` 由关改为 **`True`**。这**不是兼容性修复,而是物理默认值变更:所有含摩擦的场景轨迹都会变**。既有脚本一行不改、只换 wheel,复现出的轨迹就不再和 0.8.5.3 逐位一致(§2.2 的 `fDhat` 公式也随之走 `epsv²` 分支)。
>
> - **动机(实测,非推演)**:flask_cap 双臂 finray 抓取–提升–保持 400 步,受力远在 μ=3.5 摩擦锥内,烧瓶仍在 6 s 保持段滑 3.7 mm、瓶盖锥体转 11–20°。根因两条:(a) legacy epsv 由场景尺度派生(1.9 m 场景 = 19 mm/s,是 IPC 论文默认 `1e-3·l` 的 10×、静精度值 1e-5 m/s 的 1900×);(b) lagged 摩擦的位移锚点每步重置,持握接触必须每步重新滑 `u* = (load/(μ·λ))·eps` 才能重建摩擦力 = 与 epsv 成正比的 stiction creep。
> - **效果**:epsv=1e-5 单独把保持段滑移 3.71 mm → 0.05 mm(step +12%);两者同开(epsv=1e-4 + anchor)滑移 0.00 mm、盖倾角钉在 0.7–1.4°(legacy 11° 且仍在爬),逐步漂移 10 µm/帧 → 5e-12 m/帧(`STIFF_NEWTON_TRACE` 测得)。代价:release 口径 **+9% step**(动态段多解真实 stick-slip 的 Newton/PCG,保持段迭代数不变);`dc1a297` 优化前口径为 +30%。
> - **回到 legacy(三选一,前两者按 CHANGELOG 口径可逐位还原 0.8.5.3)**:
>   ```python
>   Config(absolute_epsv=0.0, friction_anchor=False, ...)   # 代码内
>   ```
>   ```bash
>   STIFF_EPSV=0 STIFF_FRIC_ANCHOR=0 python your_script.py   # 环境变量,进程级
>   ```
>   或直接装回 0.8.5.3 wheel。**两个必须同时关**——只关一个仍是新轨迹。环境变量优先于 Config(stable@c0339c8 GIPC.cu:9499、:9811)。
> - **phase-cd 上没有这两个字段**:`Config(absolute_epsv=1e-4)` 会走 `**kwargs` 透传路径,`hasattr(self._cfg, ...)` 为假 → **静默忽略、无警告**(见表末 `**kwargs` 行),不会报错也不会生效;`STIFF_EPSV` / `STIFF_FRIC_ANCHOR` 在 phase-cd 未注册,会被 knob-registry tripwire 报 `unknown STIFF_* knob`(API_EXECUTION §7.0)。工程线要用真静摩擦,须走未合入的移植分支 `port/friction-anchor-086`(`c735e13` epsv、`57015da` anchors;分支现状与合入前提见 KNOWN_ISSUES §1.5 与 §1.1 修复状态栏)。
> - **行号约定**:本条 `stable@c0339c8 <文件>:<行号>` 指稳定线 **commit `c0339c8` 的 blob**,自 2026-09-08 起与工作树一致(工作树已恢复为 `c0339c8` 内容)。勘探当时工作树曾被误留的 8 文件暂存回退按在 v0.8.5.3、树上 grep `absolute_epsv` 为 0 命中,该状态已清除。
> - **已发布(v0.8.5.4,双 wheel)**:公开仓 Release `v0.8.5.4` 已正式发布(2026-08-11T17:07:37Z,非 draft/prerelease),`cp311`/`cp312` 两个 wheel 均已挂出,README 安装 URL 也已指向它(`gh release view v0.8.5.4 --repo haoxiangNtu/stiff-physics` 亲验;OPEN_POINTS OP-001 已关闭)。**但"可以装"不等于"可以随手升"**:装上 v0.8.5.4 就是上表的新默认值,**所有含摩擦场景的轨迹都会变**——既有回放/金锚/RL 策略要逐位复现,升级时必须同时给上面的两个逃生阀。

### 2.2 dHat / 摩擦 dhat / dTol 派生(GIPC::init)【稳定线+phase-cd】

gipc_modules/09_friction_sets_host_mem.inl:580-622(稳定线同逻辑在单体 GIPC.cu):

```
bboxDiagSize2 = ‖upper − lower‖²            (从 device BVH 根盒直读 48B)
eff = bboxDiagSize2
if absolute_dhat > 0 && relative_dhat > 0:
    eff = absolute_dhat² / relative_dhat²
dHat  = relative_dhat² · eff                 ⇒ absolute_dhat>0 时恰 = absolute_dhat² (m²)
fDhat = 1e-4 · eff                           (摩擦用 dhat)
dTol  = 1e-18 · eff
minKappaCoef = 1e11
```

> **陷阱**:`absolute_dhat` 只有在 `relative_dhat > 0` 时才生效(两处判断均 `abs>0 && rel>0`)。把 `relative_dhat` 设 0 并指望 `absolute_dhat` 接管是错的。
>
> κ 一致性:`suggestKappa`/`upperBoundKappa`/`initKappa` 的 per-env kmax 在 abs>0 且 rel>0 时同样用 `bb = abs²/rel²` 替代合并 bbox(否则 κ 随 env 数稀释;gipc_modules/13_kappa_partition_gradhess.inl:7-38, 221-223)。诊断逃生口 `STIFF_DIAG_KAPPA_MERGEDBB` 强制旧 bbox κ。

κ 播种公式(13:1-51):

```
H_b(d, dHat):  t = d − dHat;  H = −2·ln(d/dHat) − 4t/d + t²/d²     (IPC barrier 二阶导)
suggestKappa:  κ = minKappaCoef · meanMass / (4e-16 · bb · H_b(1e-16·bb, dHat))
upperBound:    κ_max = 100 · 同上;超上限夹紧
```

`meanMass` 在 decouple_thresh + body_groups 声明时改用 **env 0 顶点质量均值**(batch-size 不变式,01:110-137)。

### 2.3 kwargs 可透传的 native-only 字段 【稳定线+phase-cd】

Python 构造器未列出、但 pybind `Config` 有 `def_readwrite` 绑定(bindings/pystiffgipc.cu:98-139),可经 `Config(**kwargs)` 或 `cfg.native.xxx` 设置:

| 字段 | native 默认 | 单位 | 含义(sim_engine.h 行号) |
|---|---|---|---|
| `cloth_thickness` | `1e-3` | m | 布料厚度;参与面积质量与弯曲刚度(:26) |
| `cloth_young_modulus` | `1e6` | Pa | 布料拉伸模量;`stretchStiff = E_cloth/(2(1+ν))`(01:46)(:27) |
| `bend_young_modulus` | `1e5` | Pa | 弯曲模量;`bendStiff = E_bend·h³/(24(1−ν²))`(01:47-48)(:28) |
| `cloth_density` | `2e2` | kg/m³ | 布料**体**密度(非面密度):代码先 `area *= h` 得体积,再 `m += ρ·(area·h)/3`(01:91-101)(:29)。默认 200 kg/m³ 在 h=1mm 下 ≈ 0.2 kg/m² 面密度;**勿直接填真实面密度**(填 0.2 会把质量做轻 1000 倍) |
| `strain_rate` | `100` | — | 应变限制率;`shearStiff = 0.03·stretchStiff·strainRate`(01:49)(:30) |
| `soft_motion_rate` | `1e0` | — | soft-constraint / **stitch spring 刚度**(§4.8)(:31) |
| `linear_system_buff_scale` | `1.0` | — | 线性系统缓冲倍率;碰撞 triplet 下限 `100000·scale`(09:629)(:54) |
| `triplet_internal_margin` | `32.0` | — | 内部 Hessian-triplet 缓冲余量倍率;**只为 FEM-pin 链式法则展开保留**——无 pin 场景引擎自动按 1.0 生效(01:809-810);多环境大场景建议显式降到 ~4(:55-61) |
| `env_newton_iter_cap` | `0` | — | per-env Newton 预算,超限 env 冻结为 TIMEOUT;0=关;仅宿主 per-env 路径(:73-76) |
| `line_search_max_iter` | `64` | — | [T1] 线搜索回溯预算;旧硬编码 8(坏步静默接受);64 对齐 libuipc(:77-80) |

### 2.4 cfg → 引擎内部映射位置速查

`finalize()` 时 `apply_config_to_ipc()` 做逐字段直接赋值(01_config_upload.inl:1-36;稳定线 sim_engine.cu:931-966 逐字段一致)。不走该函数的字段:

- 关节 4 个 ratio、2 个 slew 限速、`velocity_damping`、`dt`、`gravity` → ABD parms(02:26-37);
- `collision_detection_buff_scale` → 碰撞对缓冲容量:`MAX_CCD_PAIRS = scale·(surface·15 + edge·10)·max(dt/0.01, 2.0)`,`MAX_DCD_PAIRS = (surfVert·3 + edge·2)·3·scale`(01:600-608);
- `cuda_device` → `cudaSetDevice`(01:170, 581);
- `triplet_internal_margin` → `ipc.m_triplet_internal_margin`(01:809-810);
- `linear_system_buff_scale` → `ipc.init(...)`(01:857-858)。

---

## 3. 场景构建

### 3.0 标准调用时序 【稳定线+phase-cd】

```
Engine(config)                                   # 构造 + init_cuda
  → load_*()        ← ABD body 必须全部先于 FEM body 加载(违者 std::abort)
  → per-body 材料 / joint / stitch / pin / set_body_groups / set_vertex_boundary   (finalize 前)
  → finalize()                                   # 上传 GPU、建 BVH;一进程仅一个 finalized Engine
  → (finalize 后配置: set_vertex_env_ids / set_env_offsets / set_fixed_joint_strength /
     set_body_apply_gravity / set_body_external_force / set_prismatic_limit_barrier …)
  → step() 循环 + 读数
```

每个 API 的 pre-/post-finalize 时机在附录 13.3 有速查表。

### 3.1 `load_urdf` 【稳定线+phase-cd】

```python
Engine.load_urdf(urdf_path: str, scale: float = 1.0,
                 translation: tuple = (0,0,0), root_fixed: bool = True,
                 revolute_as_motor: bool = False, default_young: float = 1e7,
                 initial_joint_angles: dict[str, float] | None = None) -> None
```

(engine.py:559-591;native `load_urdf(path, Matrix4d, root_fixed, revolute_as_motor, default_young, initial_joint_angles)`,sim_engine.h:166-171)

| 参数 | 说明 |
|---|---|
| `urdf_path` | 相对路径先对 `get_assets_dir()` 解析(拼接结果存在才用) |
| `scale`, `translation` | Python 层组装 4×4:对角 scale + 平移列 |
| `root_fixed` | True 时 importer 把 root link 设为 Fixed(urdf_scene_importer.cpp:405) |
| `revolute_as_motor` | True → revolute 关节以 Motor 边界加载 |
| `default_young` | link mesh 默认杨氏模量 |
| `initial_joint_angles` | 关节名→弧度;给出时按 FK 目标位姿加载而非零位姿,避免加载即穿透。加载后 `get_revolute_current_angles()` 读 0(相对角),绝对角见 §7.4 |

**注意/陷阱**:
- URDF 加载**失败只打印 stderr 并 return,不抛异常**(00_impl_api_surface.inl:455-459)——加载后应检查 `abd_body_count`。
- URDF 引用的 mesh 文件不存在时该 link 被**静默跳过**;须提前用 `eng.native.set_urdf_mesh_override(link_name, msh_path, young)` 覆盖(§10,load_urdf 前调用,每次 load_urdf 消费并清空全部 override,sim_engine.h:593-602)。
- **primitive 碰撞代理(box/sphere/cylinder)**:link 的碰撞元素若一个 mesh 都没有、只有 primitive,importer 生成**保守外接**代理网格并走普通 mesh 路径加载(0.8.4.2 起默认开)——box 按全尺寸精确、无膨胀;sphere = icosphere subdiv-2 整体缩放到各**面平面**距心 ≥ r(外接,径向膨胀 ~2.4%);cylinder(轴 = 局部 +Z)= 24 段外接棱柱,径向顶点取 `r/cos(π/24)`(~0.9%),端盖为中心扇形。同一 link 的多个 primitive 元素**合并成一个 body**,逐元素 origin 烘进顶点(`collision_origin` 归单位阵);产物写 `<urdf 目录>/.stiffgipc_prim_cache/<stem>_<link>_prim.obj`,该目录不可写则退到系统临时目录(urdf_scene_importer.cpp:53-233、897-936)。
- **`STIFF_URDF_PRIM_PROXY=0`(只认精确字面量 `0`,别的拼法一律视为开)恢复 ≤0.8.4 行为:primitive-only link 的碰撞被整体跳过**、打 WARN——即这些 link(轮子、底座等)**根本不参与碰撞**,是行为差异不是性能档位(urdf_scene_importer.cpp:882-896)。反向的迁移症状:依赖"轮子穿地不碰"的旧场景在默认档下会因初始穿透在 `finalize()` 抛错。混合 `[primitive, mesh]` 的 link **两档都只用第一个 mesh 元素**、primitive 元素 WARN 后忽略(≤0.8.4 是首元素为 primitive 时丢掉整条 link 的碰撞,urdf_scene_importer.cpp:937-941)。该旋钮**不在 knob 注册表内**(grep `config/knob_registry.h` 0 命中),由此产生的拼写陷阱见 API_EXECUTION §7.7。
- `eng.native.get_urdf_link_transform(link_name)`:load_urdf 后立即可用(无需 finalize)的 FK 世界变换;**link 名找不到返回单位阵,不抛错**(sim_engine.h:587-591)。
- load record 逐 ABD body 一条,label = URDF link 名(00:487-496)。

### 3.2 `load_mesh` 【稳定线+phase-cd】

```python
Engine.load_mesh(mesh_path: str, dimensions: int = 3,
                 body_type: str|int = "FEM", transform: np.ndarray|None = None,
                 young_modulus: float = 1e7, boundary_type: str|int = "Free",
                 density: float|None = None) -> None
```

(engine.py:597-637;native sim_engine.h:178-183,实现 00:511-553)

| 参数 | 说明 |
|---|---|
| `mesh_path` | `.msh` = 3D 四面体(Gmsh v2.2),`.obj` = 2D 布料/壳;相对路径对 assets_dir 解析 |
| `dimensions` | 2 = 三角壳/布料,3 = 四面体体 |
| `body_type` | `"ABD"`(0,仿射刚体)/ `"FEM"`(1,可变形);映射表 engine.py:593 |
| `transform` | 4×4 齐次变换,逐点 `V' = T·[x,y,z,1]`;缺省单位阵 |
| `young_modulus` | 逐 tet 写入 `vert_youngth_modules`(名字带 vert 但按 tet 索引,load_mesh.cpp:648);**dim=2 布料不接收该参数**——布料刚度来自全局 `cloth_young_modulus` 族 |
| `boundary_type` | `"Free"`(0)/ `"Fixed"`(1)。Python 映射表还含 `"Motor"`(2)/`"Animated"`(3)(engine.py:594-595),但见下方陷阱 |
| `density` | per-body 密度语法糖:load 后对最后一条 load record 调 `set_soft_body_density`(engine.py:634-637);**仅 FEM/布料** |

**body_type × dimensions 组合语义表**(native 分派 simple_scene_importer.cpp:64-97,亲验):

| dimensions | body_type | 实际行为 |
|---|---|---|
| 3 | ABD | `.msh` 四面体 ABD 刚体 ✓ |
| 3 | FEM | `.msh` 四面体 FEM(MAS 开时先 `metis_sort` 重排 + load parts)✓ |
| 2 | FEM | `.obj` 三角布料/壳 FEM ✓ |
| 2 | **ABD** | **⚠ body_type 被忽略,实际加载成 FEM 布料**(`load_triMesh` 内部硬编码 `BodyType::FEM`,load_mesh.cpp:122)。obj 表面刚体的唯一公开入口是 `load_mesh_from_data`/`load_mesh_instanced`(vpf=3 + ABD)→ §3.3 |
| 非 2/3 | 任意 | **静默什么也不加载**,且 `record_load` 仍 push 一条 vertex_count=0 的脏记录(无上层防护) |

**陷阱清单**:
- **文件打不开 = 进程退出**:`load_tetrahedraMesh`/`load_triMesh` 直接 `fprintf(stderr) + exit(-1)`(load_mesh.cpp:527-533, 134-140)——不是异常,无法 try/except。
- **ABD 必须先于 FEM 加载**:FEM 出现后再加载 ABD → `std::abort()`(load_mesh.cpp:1460-1483)。
- **dim=2 时 `boundary_type` 参数被忽略(恒 Free)**:SimpleSceneImporter 路径硬编码第三参 0;钉布料角必须用 `set_vertex_boundary`(§3.8)。
- **Motor/Animated 经公开 loader 不可达**:native loader 只做 `boundary_type==1 ? Fixed : Free`(00:529),Python 映射表里的 `"Motor"`/`"Animated"` 传下去等同 Free。`set_body_animated_target` 的能量门控 `boundary_type==Animated` 因此疑似永不触发(公开 API 下);此为 docstring 与实现的已知矛盾,**待核实**是否存在其它赋值入径。
- MAS(默认 `preconditioner_type=1`)下 FEM 顶点被 metis 重排;用户可见 API(get_vertex_position*、set_vertex_boundary、stitch 索引)一律以**输入序**为准,引擎内部自动换算(逆置换 `vertex_metis_to_input`)。

### 3.3 `load_mesh_from_data` 【稳定线+phase-cd】

```python
Engine.load_mesh_from_data(vertices, faces, verts_per_face: int = 3,
                           dimensions: int = 3, body_type="FEM",
                           transform=None, young_modulus=1e7,
                           boundary_type="Free") -> None
```

(engine.py:639-659;native 实现 engine_modules/03_step_getters_export.inl:2166-2247)

- `vertices` → float64 C 连续 (N,3);`faces` → int32 C 连续 (M,vpf)。
- 实现 = 写临时文件再走既有加载管线,目录固定 **`/tmp/stiffgipc_mesh_data/`**(03:2182,非配置项;多进程并发疑有同名竞态,待核实)。`dim==2 || vpf==3` 写 `.obj`,`vpf==4` 写 Gmsh `.msh`。
- **关键分派**:`body_type==ABD && (dim==2 || vpf==3)` → `load_surfaceMesh_ABD`:**闭合三角网格 ABD 刚体**,质量/质心/惯量经散度定理表面积分(load_mesh.h:237-244)——这是获得 obj 表面刚体的唯一公开入口。其余组合走 §3.2 同款分派。
- record label = `"from_data"`。

> **陷阱**:sim_engine.h:185-186 注释宣称 dim=3 表面数据会 "internally tetrahedralize",但代码中 dim=3 + vpf=3 + FEM 会把 .obj 交给 Gmsh 解析器,**未找到任何四面体化实现,该组合疑似不可用**(待核实)。3D 软体请直接供给四面体(vpf=4)。

### 3.4 `load_mesh_instanced` 【稳定线+phase-cd】

```python
Engine.load_mesh_instanced(vertices, faces, transforms_list: list[np.ndarray],
                           verts_per_face=3, dimensions=3, body_type="ABD",
                           young_modulus=1e7, boundary_type="Free") -> dict
```

(engine.py:661-702;native sim_engine.h:202-212,实现 03:2326-2394)

- 同一网格拓扑 N 实例各自 4×4 变换;拓扑只解析/落盘一次(`tmp_instanced_asset<K>`),逐实例登记 load record(label=`"instanced_asset<K>_i<i>"`,携带 `asset_id`/`instance_id`)。
- 返回 dict:`body_offsets`(list)、`vertex_offsets`(list)、`vertex_counts`(list)、`asset_id`。
- **⚠ 此处 body_type 默认 `"ABD"`**(其余 load 默认 "FEM")。
- transforms 逐个校验 (4,4) float64;绑定内 `mat.transpose()` 做 numpy 行主序 → Eigen 列主序校正(pystiffgipc.cu:272-276)——传 C 连续 numpy 即可,勿自行转置。
- 资产查询:`get_mesh_asset(asset_id)`(越界抛 `std::out_of_range`)、`mesh_asset_count` 属性。`MeshAsset` 字段:`asset_id, num_verts, num_faces, verts_per_face, dimensions, body_type, young_modulus, boundary_type` + `get_rest_vertices() -> (N,3)` / `get_faces() -> (M,vpf)`(pystiffgipc.cu:178-199)。

### 3.5 load records 与三种 body 编号 【稳定线+phase-cd】

`BodyLoadRecord`(sim_engine.h:111-120;绑定字段 pystiffgipc.cu:160-175):

| 字段 | 含义 |
|---|---|
| `body_type` | 0=ABD,1=FEM |
| `body_offset` | **同类型内序号**(ABD 内第几个 / FEM 内第几个),非全局 body id |
| `vertex_offset` | 全局顶点数组(引擎序)首顶点 |
| `vertex_count` | 本次加载顶点数 |
| `asset_id` | 关联 MeshAsset(-1=非实例加载) |
| `instance_id` | 资产第几个实例(默认 0) |
| `label` | 解析后路径 / `"from_data"` / `"instanced_..."` / URDF link 名(**字段名是 label 不是 name**) |

获取:`Engine.get_load_records() -> list`(engine.py:1544-1546;native `get_load_record_count`/`get_load_record(idx)`,越界抛 `std::out_of_range`)。

> **三种编号,务必区分**(混载场景三者不一致):
>
> | 编号 | 定义 | 使用它的 API |
> |---|---|---|
> | 全局 body id | ABD 先占 `[0, n_abd)`,FEM 续接 `[n_abd, total)` | joint 族(仅 ABD 段合法)、`add_collision_exclusion`、`set_abd_body_*`、`set_body_apply_gravity`、`set_body_external_force/wrench` |
> | load-record 下标 | `get_load_records()` 的索引 | `set_body_friction`、`set_soft_body_density`、`set_per_tet_young_for_body` |
> | 同类型内序号 | record.body_offset;BodyView.body_id | `get_abd_body(id)`/`get_fem_body(id)`、`get_fem_body_vertex_range`、ABD transforms 族的 body_offsets |

### 3.6 `BodyView` 【稳定线+phase-cd】

(engine.py:90-175)某 body 在全局顶点/面数组中切片的**只读视图**;自身不持数据,每次调用 `get_vertices()` 等访问器都经 `engine.get_vertices()` 从 GPU 读回一份**新的**宿主数组再切片(engine.py:142-146、1431-1433)。因此 **BodyView 对象**无须每帧重取(原 docstring 语义即此),但**返回的数组是调用时刻的快照**——不会随后续 `step()` 自动更新,取最新状态须重新调用访问器,勿缓存旧数组。经 `get_bodies()` / `get_abd_body(id)` / `get_fem_body(id)` 获取(找不到抛 `IndexError`)。

| 成员 | 说明 |
|---|---|
| `kind -> str` | `'ABD'` 或 `'FEM'` |
| `body_id -> int` | 同类型内序号(= record.body_offset) |
| `label / asset_id / instance_id` | 同 load record |
| `vertex_offset / vertex_count` | 在 `engine.get_vertices()` 中的切片区间 |
| `get_vertices() -> (n,3)` | 当前变形后顶点 |
| `get_vertex_velocities() -> (n,3)` | 当前速度 |
| `get_surface_faces(local_indices=True) -> (m,3)` | 属于该 body 的表面三角形;True 返回 0 基局部索引(可直接索引 get_vertices()),False 保持全局索引。实现为过滤"三顶点全落在本 body 区间"的面 |

辅助:`Engine.get_vertex_body_ids() -> (N,2) int32`(列 0 = body_type,列 1 = 同类型内序号;engine.py:1568-1588)。

### 3.7 碰撞域与多环境划分(API 面)

完整多环境语义(三模式契约、per-env κ/PCG、检疫)见多环境分册;此处给 API 契约。

#### `set_body_groups(groups)` 【稳定线+phase-cd】(finalize 前声明,finalize 时验证生效)

(engine.py:715-726;native sim_engine.h:221-224;校验 01:317-409)

- 一 body 一个 group(env)id,长度必须 == 碰撞体数(ABD 在前、FEM 续后);调用本身只是赋值,**finalize 后调用不生效也不报错**。
- 校验(finalize 时,违者抛 `std::invalid_argument`):非负 id 必须稠密 `[0,N)` 且 **N ≤ 256**(`kGroupSlotCapacity`);`-1` 通配仅 merged 模式允许;任一 isolated/strict 特性开启时必须全 body 分组。
- 生效:跨组 body 对写入 `collision_skip_matrix`(与 `add_collision_exclusion` 合并)→ 无论空间间距,不同 group 永不碰撞;同时建立 per-env 求解基底(`d_body_to_group`/`d_point_to_group`/`d_dof_to_group`)。

#### `set_vertex_env_ids(env_ids)` 【稳定线+phase-cd】(**finalize 后任意时刻**)

(engine.py:728-743;native sim_engine.h:225-228,实现 00:611-660)

- per-**顶点** env id(长度 = 引擎顶点数;ABD 顶点按加载序在前,FEM 粒子随后);broad-phase 跳过两端 env id(均 ≥0)不同的接触对——**无需空间分隔**的跨 env 接触隔离,支持单个 FEM body 的粒子横跨多个 env(body 级无法表达);与块对角求解解耦(仅接触过滤)。
- id < 0 = 与所有 env 碰撞的共享几何;传空序列释放过滤器;每次调用重新上传设备数组(grow-only)。
- 抛错:非空但未 finalize → `LifecycleError`;长度不符 → `std::invalid_argument`;**底层 CUDA symbol 进程级,同一时刻只能一个 Engine 持有该过滤器**,被另一 Engine 占用时抛 `LifecycleError`(00:42-57, 636-639)。

#### 其它碰撞域控制

| API | 时机 | 说明 | 适用线 |
|---|---|---|---|
| `add_collision_exclusion(body_a, body_b)` | finalize 前 | 排除一对**全局 body id** 的碰撞;finalize 时对称写 skip matrix(越界 id 静默忽略) | 【稳定线+phase-cd】 |
| `add_ground_collision_skip(body_id)` | finalize 前 | 该 body 跳过地面碰撞 | 【稳定线+phase-cd】 |
| `eng.native.set_env_offsets(per_group_xyz)` | **finalize 后** | 每 group 世界偏移(flat xyz)。BVH 建在 `_vertexes+offset`(env 空间分离)而窄相位用局部坐标(确定性);up 轴分量保持 0 以共享地面。未 finalize 时 printf 警告并 return(不抛错) | 【稳定线+phase-cd】native-only |
| `eng.native.add_ground(height)` | — | **no-op 占位**:地面恒在,由 `ground_normal/ground_offset` 控制(00:504-509) | 【稳定线+phase-cd】 |
| `eng.native.get_point_groups()` | finalize 后 | (N,) int,输入序每顶点 group id(-1=未分组),供 `verts[groups==g]` 抽取单 env | 【稳定线+phase-cd】native-only |

### 3.8 顶点边界与面朝向

| API | 时机 | 说明 |
|---|---|---|
| `set_vertex_boundary(vertex_index, boundary_type)` 【稳定线+phase-cd】 | finalize 前 | 0=Free,1=Fixed。`vertex_index` 为**输入序**(MAS 下内部经 `vertex_metis_to_input` 换算,O(n) 线性搜);**越界静默 return 不抛错**(00:1048-1069)。任何非 0 的 per-顶点边界值都会让 `_stepForward` 冻结该顶点(08_step_update_topology.inl:16) |
| `set_vertex_boundaries(indices, boundary_type)` 【稳定线+phase-cd】 | finalize 前 | 批量版(Python 循环,engine.py:961-964) |
| `set_abd_body_face_orient(body_id, orient) -> bool` 【稳定线+phase-cd】 | finalize 前 | libuipc 风格逐三角形 {-1,0,+1}:-1 在质量/质心/惯量积分时翻转法向,**不改拓扑**(碰撞/BVH/渲染仍见原始面);空 vector 恢复按 winding。长度不符或 body 无表面网格返回 **False(不抛错)** |
| `get_abd_body_face_orient(body_id)` / `get_abd_surface_body_vertices(body_id)` / `get_abd_surface_body_triangles(body_id)` 【稳定线+phase-cd】 | finalize 前 | 读当前标签 / per-body 局部表面顶点 (N,3) / 三角形 (M,3) |
| `label_face_orient_for_abd_body(body_id, method="flood_fill") -> int` 【稳定线+phase-cd】 | finalize 前 | 纯 Python 增值:边邻接 BFS 传播绕向一致性(多连通分量各自重启;有向体积为负则整体翻转),写入 orient 标签;返回被标记 -1 的面数(engine.py:880-902) |

### 3.9 `add_hybrid_fem_body` 【稳定线+phase-cd】(Python 增值)

```python
Engine.add_hybrid_fem_body(hybrid_data, transform=None,
                           target_abd_body_offset=None) -> int   # 返回 FEM body 的 vertex_offset
```

(engine.py:748-827)加载 ABD-FEM 混合四面体网格(`.npz` 路径或同字段 mapping:`vertices, tets, vertex_abd_body_id, vertex_local_pos, ...`)。**断链提示**:docstring(engine.py:750)说该 npz 由 `tools/build_hybrid_mesh.py` 产出,但**该脚本两树均已不在树内**(`find` 亲验 0 命中);hybrid 资产现状 = 只能用 `Assets/sim_data/` 下的现成 npz,新混合爪资产当前不可再生(详见 §12.2)。实现:`load_mesh_from_data(vpf=4, dim=3, FEM, Free)` → 对 `vertex_abd_body_id ≥ 0` 的刚性顶点调 native `add_fem_pins_with_local_pos`(批量硬 pin,§4.9)。注:npz 内 density/poisson 字段当前未被 FEM loader 使用(engine.py:786-787 注释)。纯 FEM 数据(无刚性顶点)时打印提示并直接返回。

---

## 4. 关节与耦合

### 4.1 通用约定 【稳定线+phase-cd】

- 三个 `add_*_joint` 均须 **finalize() 之前**调用;parent/child 数字顺序任意(引擎内部规范化跨体 Hessian 存储)。
- 校验(00:290-322):parent/child 必须都在 `[0, abd_body_num)` 且不相等,否则抛 `std::invalid_argument`(消息含合法区间与实参)。**关节只能连接 ABD body**(FEM 无 12-DOF q)。
- 返回值 = 约束数组下标。**revolute/fixed 共享 `joint_constraints` 编号空间,prismatic 是独立编号空间**(供 `set_fixed_joint_strength` 等按 idx 寻址)。
- 关节能量均为 mass-scaled 罚(rbs-uipc 公式),**无 dt² 因子**(setup_abd_system_gradient_and_hessian.cu:1388-1391 注释 + 1412 实现;joint_angle_control.h:42 的 dt² 注释是陈旧的,以代码为准)。

### 4.2 `add_fixed_joint` 【稳定线+phase-cd】

```python
Engine.add_fixed_joint(parent_body: int, child_body: int,
                       world_anchor, world_normal, world_bitangent) -> int
```

(engine.py:904-915;native sim_engine.h:384-387,实现 00:914-942)

| 参数 | 说明 |
|---|---|
| `world_anchor` | 单锚点(世界系,t=0 时刻) |
| `world_normal` (n) | 方向轴之一,内部 normalize |
| `world_bitangent` (b) | 方向轴之二,内部 normalize;第三轴 **t = n × b 再归一化**(00:937)。未校验 n ⊥ b,非正交输入的退化行为未验证(待核实) |

**机制**:单锚点 + t/n/b 三轴全刚性(libuipc "Method 2")。GPU 能量(abd_joint_constraint.h:17-30):

```
E = ½K‖J(cp)·qp − J(cq)·qq‖²  +  Σ_{d∈{t,n,b}} ½K‖Ap·d̄p − Aq·d̄q‖²
K = joint_strength_ratio · (m_parent + m_child)        (无 dt²)
```

三方向项给出 rank-9 旋转 Hessian,旋转完全约束(只罚 n+b 会留下绕 t 的软模态)。世界锚点/方向在 finalize 时经 A⁻¹ 转成两 body 材料坐标。

> **陷阱**:默认 `joint_strength_ratio=100` 对"焊接"常太弱(轻 body 时 kappa ~8e-3,每 cm 手部运动滞后 8mm+);hybrid 夹爪焊到 URDF 手腕建议 finalize 后 `eng.native.set_fixed_joint_strength(idx, 1e6)`(§4.5)。

### 4.3 `add_revolute_joint` 【稳定线+phase-cd】

```python
Engine.add_revolute_joint(parent_body, child_body, world_axis, joint_pos,
                          lower_limit: float, upper_limit: float,
                          initial_angle: float = 0.0, name: str = "",
                          passive: bool = False) -> int
```

(engine.py:917-937;native sim_engine.h:393-399,实现 00:944-997)

| 参数 | 说明 |
|---|---|
| `world_axis` | 旋转轴(内部归一化) |
| `joint_pos` | 轴上一点(世界系);两锚点 = `joint_pos ± axis·0.5`(位置约束把两 body 钉在公共轴上) |
| `lower_limit`, `upper_limit` | rad;截断到 `±kSafeAngleLimit = 3.12413936106985`(≈179°,joint_angle_control.h:53) |
| `initial_angle` | rad;非 passive 时的初始伺服目标 |
| `passive` | True → **纯铰链**:位置伺服 strength_ratio 清零,自由摆动;限位罚仍独立生效。False(默认)= 历史行为,位置伺服保持 initial_angle |

限位实现为单侧 mass-scaled 罚(`joint_limit_strength_ratio = 20000.0`,abd_system_parms.h:31,**未暴露到 Config**),0.8.4 起带滞后活动集。

### 4.4 `add_prismatic_joint` 【稳定线+phase-cd】

```python
Engine.add_prismatic_joint(parent_body, child_body, world_center, world_axis,
                           lower_limit: float, upper_limit: float,
                           name: str = "", passive: bool = False) -> int
```

(engine.py:939-955;native sim_engine.h:401-406,实现 00:999-1046)

- 内部构造正交基 n/b;限位(m)**不截断**;`passive=True` → 自由滑轨。
- **⚠ 自动追加 parent-child 碰撞排除**(`collision_exclusion_pairs.push_back`,00:1043)——滑轨副默认互不碰撞;**revolute/fixed 无此行为**,需要时自行 `add_collision_exclusion`。
- 返回下标属于 **prismatic 独立编号空间**。

### 4.5 drive 目标与 strength 族

运行时(finalize 后、每 step 前)可改。有效驱动刚度公式(setup_abd...cu:1393-1413, 1930-2015):

```
K_drive = Config.{revolute|prismatic}_driving_strength_ratio × per-joint strength × (m_parent + m_child)
```

| API | 说明 | 适用线 |
|---|---|---|
| `set_revolute_target(idx, angle_rad)` | 设位置伺服目标角(受 slew 限速逼近) | 【稳定线+phase-cd】 |
| `set_prismatic_target(idx, distance_m)` | 设目标开度 | 【稳定线+phase-cd】 |
| `set_revolute_initial_offset(idx, offset_rad)` | 改初始偏移(角度读数基准,§7.4) | 【稳定线+phase-cd】 |
| `set_revolute_strength(idx, strength)` | per-joint 驱动强度乘子,默认 1.0;调低(如 0.1)= 接触下让步(夹爪压布不压穿,避免 barrier-κ 级联拖慢 Newton);下一 step 生效 | 【稳定线+phase-cd】 |
| `set_prismatic_strength(idx, strength)` | 同上 prismatic 版 | 【稳定线+phase-cd】 |
| `eng.native.set_fixed_joint_strength(idx, kappa)` | **直接覆写**单个 fixed joint 的 kappa(非乘子);finalize 后调用;焊接场景建议 1e6 | 【稳定线+phase-cd】native-only |
| `eng.native.set_revolute_torque(idx, torque)` | [force-control] N·m;**不加梯度项**(驱动梯度内核 NOTE 明言缺 `−τ·∂²θ/∂q²` Hessian 会坏 Newton 收敛,abd_driving_joint.h:217-222)——实现构造对称化轴 wrench `F_k=τ/2·[e_k]ₓ·A_k⁻ᵀ`,经 `joint_wrench → a_ext=M⁻¹F` 折入 q_tilde 作步内常量广义力(cal_q_tilde.cu:70-125,与 `set_prismatic_force` 同一路径,N·m 单位因此天然正确),独立于 PD 项;**纯力矩控制还需 `set_revolute_strength(idx, 0)`** | 【稳定线+phase-cd】native-only |
| `eng.native.set_prismatic_force(idx, force)` | [force-control] N,经 q_tilde 路径 +force 沿 +axis 推 child;纯力控同样需 strength=0 | 【稳定线+phase-cd】native-only |
| `eng.native.set_prismatic_limit_barrier(idx, cl, dir, dhat, kappa, slot=0)` | [force-control] 单边 log-barrier @坐标 cl。**注意:这是钳平的极硬罚簧,不是发散到 +∞ 的真 barrier**——实现对自变量做数值下限 `gc = max(g, 1e-9)`(`abd_driving_joint.h:790-793`),越界侧(g≤0)能量/梯度均为有限值,且关节坐标**无 CCD/α 钳制**;足够大的驱动刚度或外力做功可越过 cl 并被接受(LS 预算耗尽时 WARN 后照常提交,见 §7.3)。常规力控参数下实际很难越过,但**不构成硬保证**。dir=+1 允许侧在 d>cl;dhat=激活带宽(m);kappa≤0 解除;slot 0=closed end、1=open end(两端硬限位就 arm 两个);**finalize 后** | 【稳定线+phase-cd】native-only |
| `get_prismatic_drive_force(idx) -> float` | 当前 `K·(target−d)` 驱动读数——**单位不是 N**:K 为质量标度刚度 `sr·ctrl_sr·(m_p+m_c)`,无 dt² 因子(03:1125-1136;setup_abd_system_gradient_and_hessian.cu:1956),读数处在梯度空间,与物理力平衡时 = `F_物理·dt²`,**换算牛顿需 ÷dt²**(dt=0.01 时直接当 N 用偏差 10⁴ 倍);力限位置控制封顶握力时,须先 ÷dt² 再与 `set_prismatic_force` 的真 N 比较 | 【稳定线+phase-cd】 |
| `get_prismatic_current_distance(idx) -> float` | 当前沿轴开度 d(m) | 【稳定线+phase-cd】 |
| `eng.native.get_revolute_target(idx)` / `get_prismatic_target(idx)` | 读当前目标(setter 有包装,getter 只在 native) | 【稳定线+phase-cd】native-only |

### 4.6 关节信息查询 【稳定线+phase-cd】

- `num_revolute_joints` / `num_prismatic_joints`(property)。
- `get_revolute_joint_info(idx)` / `get_prismatic_joint_info(idx)` / `get_all_joint_infos()` → `JointInfo` 只读字段:`name, lower_limit, upper_limit, target, strength_ratio, is_prismatic`(revolute 限位 rad、prismatic 限位 m;pystiffgipc.cu:146-157)。

### 4.7 每帧目标限速(slew)【稳定线+phase-cd】

驱动目标每帧向 set 目标逼近的最大步长,防止一帧大目标跳变直接把接触打飞:

- 配置:`Config.max_revolute_step_per_frame`(默认 0.1 rad ≈5.7°)、`max_prismatic_step_per_frame`(默认 0.002 m)。
- 运行时:`set_max_revolute_step_per_frame(rad)` / `set_max_prismatic_step_per_frame(m)`(engine.py:1798-1804;下一 step 生效)。精细 FEM softpad-pin 场景建议 0.01 rad(≈0.6°)防自相交(sim_engine.h:703-708)。

### 4.8 `add_stitch_spring` — FEM↔ABD 双边软弹簧 【稳定线+phase-cd】

```python
Engine.add_stitch_spring(fem_vertex_id: int, abd_anchor_vertex_id: int,
                         abd_body_id: int, rest_offset_world=(0,0,0)) -> None
```

(engine.py:829-848;native sim_engine.h:236-251,实现 00:667-686)**必须 finalize 前**。

- 语义:每步把 FEM 顶点拉向 `world_pos(ABD 锚点顶点) + rest_offset_world`。**rest_offset 是世界系偏移,非 body-local**;最佳实践 = finalize 时 FEM 顶点与 ABD 锚点重合并传零偏移,弹簧自然跟踪平移+旋转。
- 能量(energy/14_soft_constraints.inl:39-41):`E = ½ · softMotionRate · rate² · ‖x − target‖²`,engine 路径 rate ≡ 1,故**有效刚度恒 = `Config.soft_motion_rate`**(kwargs 字段,默认 1e0)。Hessian 对角块 `motionRate·I₃`。
- 双边:ABD 侧经 `m_abd_system->m_stitch_*` 接线感受反作用力(13:1161-1168)。
- 索引:`fem_vertex_id` 按**输入序**;MAS 开时 finalize 统一做 input→engine 重映射(01:240-266;不映射则弹簧拉错顶点 → Newton 不收敛,case_40 bug)。
- finalize 末尾 sanity 检查:`softNum·soft_motion_rate / avg(FEM Young) > 1000` 时打印 "stitch system may be too stiff" 大字警告(经验:ratio 130 稳、1.3e4 在 step~131 NaN;02:49-95)。仅警告不抛错。
- 读数:`eng.native.get_stitch_max_stretch(pair_start, pair_count) -> float`(GPU device reduction 求区间 MAX 伸长,单标量,无需全顶点 D2H;stitch 按调用序存储,每根手指是连续区间)与 `get_stitch_max_stretch_batched(starts, counts) -> (n,)`(一次 kernel、每 segment 一个 block,多 env 设计)——均 native-only。

> **已知不一致(待核实)**:无 FEM-pin 的场景中 stitch 的 rest_offset 是纯世界系常量;有 pin 的场景中梯度/Hessian 内核按 `A(q)` 旋转 offset 而**能量 reduction 内核恒用世界系 offset**(14:27-38 vs 82-103;02:122-123 的 DISABLED 注释)——非零 offset + ABD 大旋转 + 有 pin 时能量与梯度不一致。遵循"零偏移+重合"最佳实践即可完全规避。

### 4.9 FEM-pin(硬约束)【稳定线+phase-cd】native-only

| API | 说明 |
|---|---|
| `eng.native.add_fem_pin_to_abd(fem_vertex_id, abd_anchor_vertex_id, abd_body_id, rest_offset_world=(0,0,0))` | 单点硬 pin:FEM 顶点世界位置精确 = ABD 锚点 + 固定偏移;finalize 前(sim_engine.h:312-334) |
| `eng.native.add_fem_pins_with_local_pos(fem_vertex_ids, abd_body_ids, abd_local_positions)` | 批量 pin,显式 ABD **rest 系局部坐标**(形状 (n,),(n,),(n,3),不符抛 `std::invalid_argument`);`fem_vertex_ids` 是**全局**顶点索引(= local_idx + record.vertex_offset);避免 1k+ pin 的逐 pin Python↔C++ 往返;finalize 前(sim_engine.h:336-356)。`add_hybrid_fem_body`(§3.9)内部走这条 |

**机制(M1 substitution)**:被 pin 顶点 `BoundaryType=2`(PCG Δx 跳过),每次 line-search alpha 尝试后内核直接投影 `world = q.t + A(q)·local_pos`(12_host_wrappers_fem.inl:892-914, 997-1005)——IPC 障碍看到的是修正后的 FEM 位形,能量沿 ABD q 方向光滑。M1 无力反馈(ABD 感受不到 FEM 反作用力;M2 计划补 cross-term Hessian)。

**stitch vs pin 对照**:

| | stitch spring | FEM-pin (M1) |
|---|---|---|
| 约束类型 | 软弹簧(能量罚,刚度 `soft_motion_rate`) | 硬约束(直接投影,精确等式) |
| 双边性 | 双边(ABD 感受反力) | 单边(ABD 无反力 → 无条件数恶化) |
| 跟踪误差 | 有(快速运动/LS 回退时滞后) | 零(每步成立) |
| offset 语义 | 世界系 | ABD rest 系局部坐标 |
| 适用 | 缝合/悬挂等弹性联接 | 软指垫与刚性骨架"焊接"等刚性附着 |

---

## 5. per-body 材料

全部须在 **body 加载后、finalize 前**调用(finalize 时展开上 GPU,之后改无效)。`body_offset` 参数一律指 **load-record 下标**(§3.5 编号表)。

### 5.1 `set_body_friction` 【稳定线+phase-cd】

```python
Engine.set_body_friction(body_offset: int, mu: float, ground_mu: float | None = None) -> None
```

(engine.py:1683-1696;native sim_engine.h:275-283,实现 00:735-745)

- per-body 摩擦覆盖(ABD 或 FEM/布均可)。`ground_mu=None`(内部传 −1.0)= 沿用全局 `gd_friction_rate`。
- 抛错:offset 越界或 mu<0 → `std::runtime_error`。
- 配对合成:接触对两侧 μ 取**几何平均** `√(μ_a·μ_b)`(PhysX 风格,energy/02_contact_energy_device.inl:380-396);地面接触用该顶点的 ground μ。
- **逐位一致性承诺**:从不调用此 API 时,per-vertex 摩擦表指针保持 nullptr,全部摩擦内核走 legacy 标量路径,**与旧版本逐位一致**(01:210-212)。
- 示例:GRIP UMI 软手指 μ=3.5 抓 μ=0.4 物体。

### 5.2 密度/质量/惯量族

| API | 对象 | 说明 | 适用线 |
|---|---|---|---|
| `set_soft_body_density(body_offset, density)` | FEM tet 体 / 布料壳 | 归属判定:tet 4 顶点(壳 3 顶点)全落在 record 顶点区间才算;质量装配时覆盖全局 `density`/`cloth_density`。record 名下无软元素(如 ABD)抛 `std::runtime_error` 并提示改用 `set_abd_body_density`;density≤0 抛错(00:747-793) | 【稳定线+phase-cd】 |
| `set_abd_body_density(body_id, density)` | ABD | mass = density × volume(体积经表面积分)(sim_engine.h:566) | 【稳定线+phase-cd】 |
| `set_abd_body_mass(body_id, mass)` | ABD(表面网格) | 直接给总质量 kg,与密度 API 刻意分立(sim_engine.h:570) | 【稳定线+phase-cd】 |
| `set_abd_body_inertia(body_id, mass, com, inertia)` | ABD | mass + COM(3) + 惯量 3×3(row-major,关于 COM,load/world 系);用 URDF authored 惯量替代焊接网格几何推算(sim_engine.h:575-576) | 【稳定线+phase-cd】 |
| `eng.native.set_per_tet_young_for_body(body_offset, per_tet_young)` | FEM tet 体 | 逐 tet 杨氏模量(Pa);数组长度必须 == 该 body 拥有的 tet 数(不符抛 `std::runtime_error`,消息给双方数值);按**引擎内 tet 顺序**对应。⚠ MAS 开时引擎 tet 序 = metis_sort 输出序,与原始 .msh tet 序的对应关系待核实 | 【稳定线+phase-cd】native-only |

注:`load_mesh(density=...)`(§3.2)是 `set_soft_body_density` 的语法糖。ABD 的三个 set(density/mass/inertia)的 `body_id` 是 ABD 内序号(engine.py 包装按此语义传递)。

---

## 6. 步进与生命周期

### 6.1 `Engine.__init__(config=None)` 【稳定线+phase-cd,行为有线差】

(engine.py:467-553)config 缺省新建 `Config()`。流程:解析 multienv 模式(`STIFF_MULTIENV_MODE` env 覆盖 `config.multienv_mode`;未知模式抛 `ValueError`)→ setdefault 模式旗/per-env-exit 旗(显式设置的环境变量总是赢;对同一模式重复创建幂等)→ `_C.SimEngine()` → `set_config` → `init_cuda()`。

**进程级模式锁 【仅 phase-cd】**(engine.py:225-268, 486-553):第一个成功构造(含 init_cuda)的 Engine 提交进程级签名(`multienv_mode` + `per_env_exit` + 全部模式旗当前值);之后:

- 换 `multienv_mode` 构造第二个 Engine → `LifecycleError`("multi-environment mode is process-scoped ... Run different modes in separate subprocesses");
- 换 `per_env_exit` → `LifecycleError`;
- 手改 STIFF_* 模式旗后再 `finalize()`/`step()`/`launch_episode_async()`/`prepare_gpu_rl*()` → `LifecycleError`(`_assert_process_mode_signature` 在这些入口执行);
- 构造失败不毒化后续尝试(锁在成功后才提交)。

理由:STIFF_* 是进程级全局且原生热路径首次使用后缓存;混模式会混用缓存/实时旗并可能破坏 CUDA graph 执行。**跨模式对比必须用子进程**。

稳定线只有"回收本模块设置的旗"逻辑(v0.8.5.1 修复:第一个 Engine 的旗不泄漏给第二个),**允许**同进程切换模式,无 LifecycleError 锁。

### 6.2 `finalize()` 【稳定线+phase-cd】

(engine.py:983-992;native sim_engine.h:423,流程 02_finalize_nandiag.inl:1-47)

计算 FEM 数据、上传 GPU、建 BVH、播种预条件器。前置校验(违者抛 `LifecycleError`,phase-cd;稳定线为等价报错):已 finalize;上次 finalize 失败;CUDA 未初始化。

- **一个进程同时只能有一个 finalized Engine**(`RuntimeOwnerLease`,00:75-105):legacy 求解器缓冲经进程级 CUDA symbol 发布;要 finalize 另一个须先 `reset()`/销毁前一个。phase-cd 在 docstring 与 header 注释明示此约束;稳定线底层同样是全局符号,只是未文档化——**两条线都应遵守**。
- 【仅 phase-cd】finalize 入口做 STIFF_* 旋钮拼写检查(`knob_registry.h`,170 键登记表):未登记的 STIFF_* 变量 stderr WARN,`STIFF_KNOB_STRICT=1` 时升级为 `ConfigurationError`(02:17-19)。稳定线 typo 静默。
- `set_body_groups` 等延迟校验在此爆发(§3.7)。

### 6.3 `step()` 【稳定线+phase-cd】

(engine.py:994-1152;native `void step()`,pybind 带 `gil_scoped_release`)

推进一个 `dt`。默认执行路径两线相同:宿主驱动的帧事务(`IPC_Solver`)。

【仅 phase-cd】扩展语义:

- `prepare_gpu_rl()` 之后 `step()` 变为已录制 RL graph 的**薄异步入队**(Isaac `simulate()` 语义):关节目标发布到设备 action slab,graph launch 立即返回无宿主等待;显式屏障 `synchronize_gpu_rl()`;`end_gpu_rl()` 恢复宿主驱动事务。RL/GPU 驻留 API 全家(`launch_episode_async` 七件套、`prepare_gpu_rl` 全家桶、`gpu_rl_tensors`、`get_gpu_rl_device_abi`)详见 RL/驻留分册;episode 在飞时调 `step()` 抛 `LifecycleError`。
- 整帧 CUDA Graph:`STIFF_FRAME_GRAPH=1`(+`STIFF_FRAME_FULL_GRAPH=1` 走整帧条件图)【实验性,默认关】——大帧场景实测倒贴(默认 OFF 是裁决结论,见 [`../OPTIMIZATION_ROADMAP.md`](../OPTIMIZATION_ROADMAP.md):75-81)。
- 测试旋钮(unset 时零开销):`STIFF_AUTO_PREPARE_AT=k`(第 k 次 step 前自动 prepare_gpu_rl + 健康审计 + OVF 自动恢复 ≤64 次)、`STIFF_MS_DUMP=path`(atexit 保存每步毫秒)、`STIFF_ITER_LOG=1`(每帧打印 Newton 迭代增量)。

【仅 phase-cd】`get_frame_status() -> FrameStatus`(engine.py:1422-1429;§7.6):最新帧边界状态包,legacy/graph-fallback/整帧图三种路径同一布局,`path_flags` 区分实际路径。

### 6.4 `reset()` / `set_log_level()` 【稳定线+phase-cd】

- `reset()`(engine.py:1385-1394):拆除整个世界回到空状态(释放所有 body/约束/GPU 缓冲),**保留当前 Config**;之后同进程重跑 `load_*` + `finalize()` 即可,无须重建 Engine。
- `set_log_level(level)`(engine.py:1375-1383):0 = 静默(压掉 solver banner、Newton 迭代、Kappa、计时细目、一次性 setup 打印;native 侧把 `std::cout` 重定向到 null streambuf);≥1 = verbose(默认)。Python print 不受影响。

### 6.5 checkpoint 【仅 phase-cd(Python 包装);两线格式互不兼容】

```python
Engine.save_checkpoint(path) -> None       # 【仅 phase-cd】
Engine.load_checkpoint(path) -> None       # 【仅 phase-cd】
```

(engine.py:1396-1418)帧边界积分器 checkpoint:save 原子写(版本化 + 校验和,完整落盘后才替换目标);load 仅限**完全相同**场景/材料/模式配置,格式、CRC、拓扑、有限值校验全部通过后才触碰 GPU 活状态。未 finalize 抛 `LifecycleError`;内容/状态非法抛 `CheckpointError`。

| | 稳定线(native-only:`eng.native.save_checkpoint/load_checkpoint`) | phase-cd |
|---|---|---|
| 格式 | magic `0x53544B50`,无版本号、无校验和、非原子写(stable GIPC.cu:16591-16631) | magic `"STIFFCP2"`,kVersion=2,120B 头,endian 标记,CRC-64(poly 0x42F0E1EBA9EA3693),state-flags 组,**本构模型编码进 ABI**(SNK1/SNK2/ARAP 不匹配拒载)(checkpoint/checkpoint_io.cu:33-53) |
| 失败行为 | `printf("[ckpt] MISMATCH")` 后 **return,不抛错** | 抛 `CheckpointError` |
| 承诺 | "full-state checkpoint" | 收窄为 "frame-boundary integrator checkpoint";load 后重建帧入口配对集(实测 5e-17 级一致,门禁 1e-12/逐位) |

**两线 checkpoint 文件互不兼容**(magic 不同)。

### 6.6 GIL 与线程 【仅 phase-cd 的 RL 项;step 两线一致】

pybind 显式释放 GIL 的绑定:`step`、`wait_episode_observation`、`finish_episode`、`prepare_gpu_rl`、`prepare_gpu_rl_episode`、`synchronize_gpu_rl`、`end_gpu_rl`、`launch_episode_async`(校验后手动释放)。其余绑定持 GIL。

---

## 7. 状态读取(post-finalize)

### 7.1 顶点/表面 【稳定线+phase-cd】

| API | 返回 | 说明 |
|---|---|---|
| `get_vertices()` | (N,3) float64 | 当前顶点位置(输入序;GPU 读回) |
| `get_vertex_velocities()` | (N,3) float64 | 当前速度 |
| `get_vertices_device_ptr() -> int` | 设备指针 | [gpu-direct] (N,3) float64 缓冲的原始 CUDA 指针;`warp.array(ptr=..., dtype=wp.vec3d, length=N)` 零拷贝;finalize 后有效。**⚠ 引擎内部顶点序**(非输入序) |
| `eng.native.get_vertex_velocities_device_ptr()` | 设备指针 | 速度缓冲版 【仅 phase-cd】 |
| `get_surface_faces()` | (F,3) uint32 | 表面三角形索引 |
| `get_surface_vertex_indices()` | (S,) uint32 | 表面顶点索引 |
| `vertex_count` / `surface_face_count` | int(property) | GPU 侧计数 |
| `vertex_count_host`(property)/ `get_vertex_position_host(idx)` | — | **pre-finalize 可用**的宿主侧读数(输入序;摆 stitch/joint 锚点用);批量版 `eng.native.get_vertices_host() -> (N,3)` native-only |
| `abd_body_count` / `fem_body_count` | int(property) | body 计数(pre-finalize 亦可) |
| `get_fem_body_vertex_range(fem_body_idx)` | (start, count) | FEM body 顶点区间 |

### 7.2 ABD 位姿与速度 【稳定线+phase-cd】

q = [p, a1, a2, a3](12-DOF 仿射);矩阵约定 `mat4[:3,:3] = Aᵀ`、`mat4[:3,3] = p`(sim_engine.h:526-527)。

| API | 说明 |
|---|---|
| `get_abd_body_transforms(body_offsets) -> (N,4,4)` | offsets 为 ABD 内序号数组(int32) |
| `set_abd_body_transforms(body_offsets, transforms)` | **只写当前 q**;init/episode-reset 请勿用它(陈旧 q_prev 造成幻影速度)——用 `teleport_abd_bodies`(§9.3) |
| `get_abd_body_velocities(body_offsets) -> (N,4,4)` | 4×4 形式的 q 速度 |
| `set_abd_body_velocities(body_offsets, velocities)` | 同上写入 |

### 7.3 物理量/遥测 【稳定线+phase-cd,注明者除外】

| API | 返回 | 说明 |
|---|---|---|
| `get_fem_von_mises_stress()` | (N,) Pa | per-顶点 von Mises:配置的四面体本构 → Cauchy → von Mises → 顶点取 incident tets 的 **MAX**;非 tet 顶点(布/ABD)= 0。稳定线文档写死 Neo-Hookean;phase-cd 措辞随 SNK1/SNK2/ARAP 可选 |
| `get_per_env_newton_iters()` | (256,) int | 各 env 上次求解冻结时的 Newton 迭代号(-1 = 跑满/缺席);需宿主 per-env 路径(`per_env_exit=True` + `env_newton_iter_cap`,或 `STIFF_PERENV_TELEM=1`) |
| `get_per_env_status()` | (256,) int | 0=active/absent, 1=converged, 2=timeout, 3=diverged(NaN 检疫) |
| `get_total_energy_tolerance_accepts()` | int | 能量容差辅助接受的累计次数 |
| `eng.native.get_total_newton_iters()` / `get_total_pcg_iters()` / `get_total_collision_pairs()` / `get_max_collision_pairs()` / `get_total_frames_done()` | — | 进程启动起累计;调用方自行差分。native-only |
| `eng.native.get_ls_exhausted_count()` / `get_ls_nonfinite_count()` | int | [rl-reset] 线搜索预算耗尽/其中能量非有限的累计数;跨 step 差分 = RL 循环丢弃中毒 episode 的信号(merged 模式按契约 WARN 后继续)。**【仅 phase-cd】**native-only |

### 7.4 关节角三件套 【稳定线+phase-cd】

(engine.py:1806-1833)三个都返回 (num_revolute,) float64,rad:

| API | 语义 |
|---|---|
| `get_revolute_current_angles()` | GPU 状态实测角,**相对加载位姿**——`load_urdf(initial_joint_angles=...)` 后立即读为 0 |
| `get_revolute_initial_offsets()` | per-joint URDF 加载时刻角(来自 initial_joint_angles;未传则全 0) |
| `get_revolute_current_angles_abs()` | **绝对 URDF 角 = 相对角 + 初始偏移**;RL 观测/存盘/对比 URDF 限位一律用这个(Python 增值,= 前两者之和) |

### 7.5 GPU 常驻/设备指针汇总

| API | 适用线 | 说明 |
|---|---|---|
| `get_vertices_device_ptr()` | 【稳定线+phase-cd】 | §7.1 |
| `get_contacts_device()` | 【稳定线+phase-cd】 | §8.3 |
| `eng.native.get_vertex_velocities_device_ptr()` | 【仅 phase-cd】 | §7.1 |
| `get_gpu_rl_device_abi()` / `gpu_rl_tensors()` | 【仅 phase-cd】 | RL/驻留分册 |
| `eng.native.get_point_to_group_device_ptr()` | 【仅 phase-cd】 | per-顶点 env id 设备指针(-1=通配) |

### 7.6 `get_frame_status()` 【仅 phase-cd】

返回 `FrameStatus`(39 个只读字段,bindings/pystiffgipc.cu:38-93):`result`(`FrameResult` 枚举:OK/RETRY_REQUIRED/FATAL/RUNTIME_ERROR)、`phase`、`invalid_bits`、`launch_status`、`error_code`、`path_flags`、`err_env/err_primitive/err_newton_iter/err_ls_iter`、工作计数(`substeps/newton_iters/pcg_iters/ls_trials`)、容量高水位(`hw_*` / `required_*`:dcd_pairs/ccd_pairs/triplets/unique_blocks/mas_clusters)、图审计(`root/terminal_graph_nodes`、`*_d2h_nodes`、`graph_launches/host_boundaries`)、数值(`final_alpha/final_energy/max_movement/cfl_alpha/kappa`)、`frame_id/attempt/retry_count/retry_invalid_bits`。`FramePhase/FrameInvalidBits/FramePathFlags/FrameErrorCode` 未注册为 Python enum,只有整型字段值。

---

## 8. 接触力读出

全部须在 `step()` 之后调用。两套单位约定并存,**这是本引擎最常见的用户事故源**:

| 家族 | 单位 | 包含项 |
|---|---|---|
| `get_vertex_contact_forces` / `get_contacts*` | **牛顿**(物理力 = −gradient/dt²,0.8.4.1 修复后符号正确:静置立方体读 +627.2 N == 自重) | normal 可含地面;friction/total 见 §8.1 |
| `get_body_contact_force`(+batched)/ `get_pair_contact_force` | **LEGACY 裸梯度 = −force·dt²,非牛顿** | 仅 body-body barrier 项;**无地面、无摩擦**;为 pre-0.8.4 调用者保留 |

### 8.1 `get_vertex_contact_forces` 【稳定线+phase-cd,摩擦分量行为有重大线差】

```python
Engine.get_vertex_contact_forces(include_ground: bool = True,
                                 components: str = "normal") -> np.ndarray  # (N,3) 牛顿
```

(engine.py:1641-1661;native sim_engine.h:286-296)

| components | 语义 |
|---|---|
| `"normal"`(默认) | body-body barrier 力 + `include_ground` 时的地面接触;按需重建一次接触(BVH+CP) |
| `"friction_lagged"` | 求解器本步**实际使用**的摩擦力(位置当前、法向力 λ 与切向基滞后一步——IPC 半隐式摩擦的诚实标签);对求解器冻结摩擦集只读 |
| `"total"` | normal + friction_lagged(如斜面上静止物体净接触力 ≈ 0) |

非法 components 抛 `KeyError`。

> **⚠ 两线关键差异(权威结论)**:lagged-friction 梯度是步内位移 `(x − o_vertexes)` 的函数;`step()` 末提交后位移恒为 0,post-step 现算恒得 0。
>
> - **稳定线 v0.8.5.3** 含修复:`GIPC::snapshotFrictionForce` 在 updateVelocities 前把摩擦力快照到持久设备缓冲(stable GIPC.cu:16504 定义、16800 调用;GIPC.cuh:332-341),`components="friction_lagged"/"total"` 返回快照 → **读数正确**。
> - **phase-cd 未移植该修复**:`get_vertex_contact_forces` 无快照机制,friction 分量仍是 post-step 现算(engine_modules/03_step_getters_export.inl:1690-1748)——**`friction_lagged`/`total` 分量仍受恒零 bug 影响**。在 phase-cd 上需要摩擦读数的 tactile/传感工作请使用稳定线。
>
> `"normal"` 分量两线行为一致。

### 8.2 legacy 裸梯度族 【稳定线+phase-cd】

| API | 返回 | 说明 |
|---|---|---|
| `get_body_contact_force(vertex_offset, vertex_count)` | (3,) | 顶点区间求和的增量势梯度(**= −force·dt²**);只含 body-body barrier(无地面、无摩擦)。物理牛顿力请改用 `get_vertex_contact_forces` + load-record 区间聚合 |
| `get_body_contact_force_batched(offsets, counts)` | (n_seg,3) | 批量版(per-finger/per-env 区间):一次接触重建 + 一次 D2H |
| `get_pair_contact_force(a_off, a_cnt, b_off, b_cnt)` | (3,) | body A 受 body B 的净 IPC 接触力(A 区间 barrier 梯度和,限于连接 A、B 区间的碰撞对);同 legacy 单位,乘 `−1/dt²` 换算牛顿 |

### 8.3 逐接触导出 【稳定线+phase-cd】

| API | 说明 |
|---|---|
| `get_collision_pairs_clean() -> (N,4) int` | 当前 body-body 碰撞对解码为纯顶点索引 4 元组(PP/PE 以 −1 填充),UIPC 风格 clean export;只读解码视图 |
| `get_contacts_device() -> (count, pair_ptr, force_ptr)` | [Step B] GPU 常驻 per-contact 导出:设备指针(uintptr)——int2 `(bodyA, bodyB)`(bodyB=−1 表地面)与 double3 世界接触力(**牛顿**,作用于 A);可 `warp.array(ptr=..., copy=False)` 包裹;只读,**有效期到下一次调用**。native `compute_contacts(rebuild)` 的 rebuild 参数在 pybind 不可达(恒 false) |
| `get_contacts() -> (pair (N,2), force (N,3))` | 上者的宿主回读(验证用;GPU 直连走 device 版) |
| `eng.native.get_ccd_pairs_clean(motion, alpha=1.0)` | 【仅 phase-cd】【实验性】**验证专用** swept-BVH oracle:重建 full-CCD 候选集;**污染求解器 scratch direction,只能在一次性进程的最后一帧后使用** |

---

## 9. teleport 族与瞬态接触状态

RL episode reset / 手动写状态的正确姿势。核心概念:引擎每步用三份状态推进 —— 当前位置 `_vertexes`、上一步已提交位置 `o_vertexes`、惯性预测子 `xTilta = x + v·dt + g·dt²`。只改其中一份必然出错。

### 9.1 `set_vertex_positions_gpu` / `set_vertex_velocities_gpu` 【稳定线+phase-cd】

- `set_vertex_positions_gpu(positions)`:(N,3) float64 直接写 GPU 当前位置缓冲。
- `set_vertex_velocities_gpu(velocities)`:**⚠ 只写速度缓冲,不重建 xTilta** —— 裸写速度不会让 body 下一步动起来(预测子仍指旧目标)。完整运动学状态交接请用 `teleport_fem_vertices`(engine.py:1451-1462 docstring 原话)。

### 9.2 `teleport_fem_vertices(positions, velocities=None)` 【稳定线+phase-cd】

(engine.py:1464-1481;native sim_engine.h:610-619)

一次写三份:`_vertexes`(当前)、`o_vertexes`(上一步提交)、`xTilta`(预测)——否则下一 step 经 xTilta 回弹到陈旧位姿。`velocities` 给出时同时写速度并令 `xTilta = x + v·dt + g·dt²`(跨移交保惯性);None 时清零速度(与 teleport_abd_bodies 语义对齐)。

### 9.3 `teleport_abd_bodies(body_offsets, transforms)` 【稳定线+phase-cd,内部行为有线差】

(engine.py:1513-1522)设 q/q_prev/q_tilde/q_temp 并清零 q_v/dq;**init/episode-reset 用它而非 `set_abd_body_transforms`**(避免陈旧 q_prev 造成幻影速度)。

两线内部差异(语义近似但不逐位等价):

- **稳定线**(1d05c7a 修复后):写 q 族后立即对 ABD 点区间重导 `x = J·q` 并同步 o_vertexes(否则首个 line search 的 E0 用旧顶点评估 → 每帧 64 次减半预算耗尽)。**stale 接触对问题留给用户手动调 `reset_transient_contact_state()`(§9.5)**。
- **phase-cd**(03_step_getters_export.inl:2484-2585):同样写 q 族 + 重导 x=J·q(对未动 body 逐位中性),并且 (a) 清除被 teleport body 所在 env 的检疫(`reviveEnv`);(b) `invalidateRefitTopology` + `buildBVH()` + `buildCP()` **当场重建帧入口 pair 集**(与 load_checkpoint 同契约)——stale-pair 处理**内建**,因此该线没有也不需要 `reset_transient_contact_state`。

### 9.4 episode reset 推荐序(宿主路径)

```python
# 稳定线
eng.teleport_abd_bodies(offsets, transforms)
eng.teleport_fem_vertices(positions, velocities)
eng.reset_transient_contact_state()     # ← 必须;否则首个 solve 用上一 episode 的 pair 表建摩擦集
# phase-cd:同上两行 teleport 即可(pair 集重建内建);GPU 驻留路径另有
# launch_gpu_rl_reset_async / launch_gpu_rl_reset_masked_async(见 RL 分册)
```

### 9.5 `reset_transient_contact_state()` 【仅稳定线】

```python
Engine.reset_transient_contact_state() -> None    # stable engine.py:941-954
```

[episode reset] 清除跨 teleport 存活的瞬态接触状态:当前/滞后 contact-pair 宿主镜像 + 滞后摩擦快照。背景:原位 episode reset(teleport 族)后,lagged-friction 集来自**上一 episode** 的接触对表,首个新 solve 会施加幻影摩擦(实测 1116 条 stale pair → 24 µm 状态发散 vs 全新构建)。teleport 完成后调用一次;无参数、无返回、无抛错路径。

**刻意不自动调用**:仿真中途传送单个 body 不应清掉其它接触的摩擦状态,因此不内嵌进 teleport API。

**phase-cd 完全没有此 API**(wrapper 与 native grep 均 0 命中)——因其 teleport 内建 pair 集重建(§9.3);但注意 phase-cd 未提供稳定线 1bc13ef 提到的 adaptive-kappa 残差清理的独立入口(0.9 µm 级,待核实其影响)。

---

## 10. native-only API 清册

以下方法 **Python `Engine` 包装层没有**,必须 `eng.native.xxx(...)` 直呼(逐一 grep engine.py 验证)。除注明者外均【稳定线+phase-cd】。

### 10.1 场景构建

| 方法 | 时机 | 一句话 |
|---|---|---|
| `add_ground(height=0.0)` | — | no-op 占位(地面恒在,由 Config 控制) |
| `add_fem_pin_to_abd(fem_vid, abd_anchor_vid, abd_body_id, rest_offset=(0,0,0))` | finalize 前 | 单点硬 pin(§4.9) |
| `add_fem_pins_with_local_pos(fem_vids, abd_body_ids, local_positions)` | finalize 前 | 批量硬 pin,ABD rest 系局部坐标(§4.9) |
| `set_per_tet_young_for_body(body_offset, per_tet_young)` | finalize 前 | 逐 tet 杨氏模量(§5.2) |
| `set_urdf_mesh_override(link_name, msh_path, young=1e7)` | **load_urdf 前** | 覆盖某 link 的 mesh;每次 load_urdf 消费并清空;URDF 引用的 mesh 缺失时必须用(否则该 link 静默跳过) |
| `set_env_offsets(per_group_xyz)` | finalize 后 | 每 group 世界偏移(§3.7) |

### 10.2 URDF / 运动学

| 方法 | 说明 |
|---|---|
| `get_urdf_link_transform(link_name) -> (4,4)` | load_urdf 后即可用(无需 finalize);FK 世界变换;**找不到返回单位阵不抛错** |

### 10.3 力控全族(§4.5 详述)

| 方法 | 时机 |
|---|---|
| `set_revolute_torque(idx, torque)` | 运行时 |
| `set_prismatic_force(idx, force)` | 运行时 |
| `set_prismatic_limit_barrier(idx, cl, dir, dhat, kappa, slot=0)` | finalize 后 |
| `set_fixed_joint_strength(idx, kappa)` | finalize 后 |
| `get_revolute_target(idx)` / `get_prismatic_target(idx)` | 运行时(setter 有包装,getter 没有) |

### 10.4 ABD 运行时

| 方法 | 时机 | 说明 |
|---|---|---|
| `set_body_animated_target(body_id, x, y, z, strength=0.0)` | 运行时 | 软二次罚 `E += ½·strength·(‖q.t−target‖² + ‖A−I‖_F²)`(12 DOF 全罚,留在 PCG 系统内);strength≤0 回退默认 1e6。**⚠ 能量端被 `boundary_type==Animated` 门死,而公开 loader 无法产生 Animated body(§3.2 陷阱)——该 API 经公开加载路径疑似永不生效,待核实** |
| `set_body_apply_gravity(body_id, enabled)` | finalize 后 | 切换某 body 全部顶点重力;用于经 joint 挂在 Fixed 父级上被运动学驱动的 ABD(否则关节罚每步对抗重力 → 漂移 + Newton 不稳);body_id 是**全局** body id |
| `set_body_external_force(body_id, fx, fy, fz)` | finalize 后 | ABD 持久外力(N),以 M⁻¹F 加速度进 q_tilde 预测(同重力);(0,0,0) 清除;镜像 libuipc AffineBodyExternalBodyForce 线性子集 |
| `set_body_external_wrench(body_id, w12)` | finalize 后 | 完整 12-DOF(线性 w[0:3] + 仿射 w[3:12] row-major);长度非 12 抛 `std::runtime_error`;`w[5]=+ω, w[9]=−ω` 即绕 Y 自旋力矩 |

### 10.5 GPU-direct / 读回

| 方法 | 说明 |
|---|---|
| `get_vertex_velocities_device_ptr()` | 【仅 phase-cd】速度缓冲设备指针 |
| `get_vertices_host() -> (N,3)` | pre-finalize 零设备同步批量宿主读(输入序) |
| `get_point_groups() -> (N,)` | 输入序每顶点 group id(§3.7) |
| `get_stitch_max_stretch(pair_start, pair_count)` / `get_stitch_max_stretch_batched(starts, counts)` | stitch 伸长 device reduction(§4.8) |
| `get_ccd_pairs_clean(motion, alpha=1.0)` | 【仅 phase-cd】验证专用 CCD oracle(§8.3) |
| `get_point_to_group_device_ptr()` | 【仅 phase-cd】per-顶点 env id 设备指针(也可经 `get_gpu_rl_device_abi()["point_to_group"]`) |

### 10.6 遥测(§7.3)

`get_total_newton_iters` / `get_total_pcg_iters` / `get_total_collision_pairs` / `get_max_collision_pairs` / `get_total_frames_done` / `get_assets_dir`;【仅 phase-cd】`get_ls_exhausted_count` / `get_ls_nonfinite_count`。

### 10.7 诊断(条件编译)【仅 phase-cd】【实验性,默认关】

| 方法 | 需要的构建开关 |
|---|---|
| `debug_fd_gradient_check(h=1e-6, nprobes=64, seed=12345)` / `debug_fd_hessian_check(h=1e-5, 16, 54321)` / `debug_fd_activity()` | `-DSTIFFGIPC_ENABLE_DIAGNOSTICS=ON`(侵入式有限差分校验) |
| `_print_bvh_coherence_audit()` | `-DSTIFFGIPC_BVH_COHERENCE_AUDIT=ON`(CMakeLists.txt:49,174-175 → 宏 `STIFF_BVH_COHERENCE_AUDIT_BUILD`);绑定被 `#ifdef` 守卫(pystiffgipc.cu:1167-1171),普通构建无此方法 |
| `_reset_bvh_traversal_audit()` / `_get_bvh_traversal_audit()`(4×4)/ `_get_bvh_traversal_body_audit()` | `-DSTIFFGIPC_BVH_TRAVERSAL_AUDIT=ON`(CMakeLists.txt:47,153-154 → 宏 `STIFF_BVH_TRAVERSAL_AUDIT_BUILD`)。注意:这三个绑定本身**无** `#ifdef` 守卫(pystiffgipc.cu:1131-1149),普通构建也可调用,只是未开该开关时计数恒为 0 |

### 10.8 模块级(不经 Engine)

| 函数 | 说明 |
|---|---|
| `pystiffgipc.fem_model()` | 【仅 phase-cd】编译期本构名 "SNK1"/"SNK2"/"ARAP" |
| `pystiffgipc._test_ccd_nan_max_speed_fail_fast` | 内部回归钩子(两线都有,用户可忽略) |

### 10.9 pybind 未单独暴露(只能经组合入口)

`get_episode_slot_first_frame/frame_count`(折叠进 `get_episode_observation` dict)、`get_vertex_positions(_host)`(经 `get_vertices(_host)`)、`get_vertex_contact_force_sum`(经改名 `get_body_contact_force`)、`compute_contacts/contacts_pair_ptr/contacts_force_ptr`(经 `get_contacts(_device)`;rebuild 参数不可达)、全部 `get_gpu_rl_*` 指针/计数 getter(折叠进 `get_gpu_rl_device_abi()` dict,唯一例外 `get_point_to_group_device_ptr` 有独立绑定)。

---

## 11. 便利层:Robot 与 Pipeline 【稳定线+phase-cd】

包级导出的两个纯 Python 便利类(§1.2 导出表),**两树文件逐字节相同**(`stiff_physics/robot.py`、`stiff_physics/pipeline.py`,diff 亲验)。它们不新增引擎能力,只是把 §4.5/§4.6 的关节 API 包成按名寻址 + 单位换算 + 限位夹紧的形态。**索引悬空提示**:API_EXECUTION §9.1 GRIP_MODE=pos 配方里的 `set_prismatic_position(...)`,以及 README §6 duck 示例的 "set_joint_target" 口头简称,指的都是本节 Robot 方法族——`Engine` 上不存在这些名字。

### 11.1 `Robot` 全方法表

`Robot(engine)`(robot.py:32-218)构造时**一次性枚举**引擎当前已注册关节建立名字表与目标缓存(robot.py:49-76);之后再添加的关节不会出现,需重建 Robot。

| 方法/属性 | 单位/说明 |
|---|---|
| `revolute_joints` / `prismatic_joints` / `all_joints`(property) | `list[JointInfo]`(§11.2);`all_joints` = revolute + prismatic 拼接 |
| `get_joint(name_or_index)` | 按名查两类关节;**int 分支只搜 revolute**(robot.py:93-95),prismatic 请按名或直接 `prismatic_joints[i]`;找不到抛 `KeyError` |
| `set_revolute_position(index, value, degree=False)` | 目标角。**默认 rad**;`degree=True` 时 value 为 deg(内部 `math.radians` 换算)。夹紧到 `[lower_limit, upper_limit]`(robot.py:119)后转发 `engine.set_revolute_target(idx, rad)`;slew 限速(§4.7)照常生效 |
| `set_prismatic_position(index, value, millimeters=False)` | 目标开度。**默认 m**;`millimeters=True` 时 value 为 mm(÷1000)。夹紧限位后转发 `engine.set_prismatic_target(idx, m)` |
| `set_joint_position(name_or_index, value, degree=False)` | 按名分派两类(prismatic 忽略 `degree`,恒 m);**int 恒当 revolute 处理**(robot.py:210-211) |
| `get_revolute_target_deg(index)` / `get_prismatic_target_mm(index)` | **deg / mm** 的目标缓存读数(纯宿主,无 GPU 读回)。缓存只跟踪经 Robot 的写入——绕过 Robot 直呼 `eng.set_revolute_target` 的改动在此不可见 |
| `set_revolute_initial_offset(index, offset_rad)` | 恒 rad,直通 §4.5(FK 预摆位后校正 target 语义为绝对 URDF 角) |
| `set_revolute_strength(index, strength)` | 直通 §4.5 per-joint 驱动强度乘子(docstring 给出夹爪让步 0.1 的理由:压布不压穿,避免 barrier-κ 级联) |
| `set_gripper_strength(strength, name_patterns=("finger","knuckle","drive_joint"))` | 批量版:按名字**子串**匹配 revolute 关节逐一 `set_revolute_strength`,返回命中数;xarm7+gripper 典型用法 = 臂保持 1.0、爪降 0.1–0.5 |
| `reset_all()` | 全部 revolute/prismatic 目标归 0(经夹紧——限位不含 0 的关节落到最近限位) |

**单位换算速查(Robot vs Engine)**:

| 量 | Engine 原生 API(§4.5) | Robot 默认 | Robot 可选 |
|---|---|---|---|
| revolute 目标写 | `set_revolute_target(idx, angle_rad)` — rad | rad | `degree=True` → deg |
| prismatic 目标写 | `set_prismatic_target(idx, distance_m)` — m | m | `millimeters=True` → mm |
| 目标读 | `eng.native.get_revolute_target/get_prismatic_target` — rad / m | `get_revolute_target_deg` — **deg**;`get_prismatic_target_mm` — **mm**(均为缓存) | — |

### 11.2 两个 `JointInfo`,别混淆

`from stiff_physics import JointInfo` 得到的是 **robot.py 的 dataclass**(robot.py:12-29),不是 §4.6 引擎查询返回的 pybind `JointInfo`:

| 字段 | robot.py dataclass | pybind(§4.6,`get_*_joint_info` 返回) |
|---|---|---|
| `name` / `lower_limit` / `upper_limit` / `strength_ratio` / `is_prismatic` | ✓(限位 revolute rad / prismatic m,与 pybind 同) | ✓ |
| `index` | ✓(同类型内序号) | ✗ |
| `target` | ✗ | ✓ |
| `lower_limit_deg` / `upper_limit_deg`(property) | ✓ — revolute 换算 deg;**prismatic 原样返回 m,名字带 deg 但不换算**(robot.py:24-29) | ✗ |

### 11.3 `Pipeline` — polyscope 运行循环

(pipeline.py:21-157)GUI 示例的地基:引擎 + Robot + polyscope 渲染 + ImGui 关节滑条的一体化循环。

```python
Pipeline(urdf_path, config=None, global_scale=1.0, translation=(0,0,0),
         root_fixed=True, revolute_as_motor=False, default_young=1e7,
         up_dir="y_up")
```

- 构造顺序固定:**先 `ps.init()` 后建 Engine**(GL 上下文先于 CUDA,规避个别驱动上 GL 破坏 CUDA 上下文,pipeline.py:45-47)→ `load_urdf` → `finalize()` → `Robot(engine)`;成员 `self.engine` / `self.robot` / `self.config` 可直接使用。
- `run()`:注册整机表面网格并进 `ps.show()` 主循环。每帧回调画控制面板(Run/Pause、step 计数、revolute 滑条按 **deg**、prismatic 滑条按 **mm**、Reset All Joints 按钮),运行态下 `engine.step()` + 顶点读回刷新网格。
- `user_gui()`:子类覆写钩子,向同一 ImGui 窗口追加自定义控件(每帧调用,默认空)。
- 模块顶层 `import polyscope`——无 polyscope 环境连 import 都会失败(版本锁 `>=2.4,<2.6`,README §3.1);Robot 无此依赖。
- 适用面:单 URDF 整机场景。多 body/自定义加载序/多环境请自行组合 Engine + Robot + polyscope(参考 pipeline.py 的回调结构)。

---

## 12. 资产与场景工具链

手册主契约面(§1–§10)之外、随仓库分发的辅助模块与脚本。以下清单两树逐字节相同(diff 亲验);**它们随包/仓库提供,但接口未经本手册逐行核对,状态为"提供、未纳入契约面"**——采信其行为前请读源码。

### 12.1 `stiff_physics/` 辅助模块(随包分发)

| 模块 | 内容(取自模块 docstring) | 备注 |
|---|---|---|
| `usd_scene_parser.py` | 解析 Omniverse USD stage 并翻译成引擎 load 调用;单引擎多实例模式(只解析 env_0 模板,再按 per-env 偏移复载);articulation 直接读 UsdPhysics joint schema(Revolute/Prismatic/FixedJoint),无需 URDF | **USD/Isaac 场景迁移入口** |
| `urdf2usd.py` | URDF → USD stage(UsdGeom.Mesh + UsdPhysics API + ArticulationRootAPI),产物可直接喂 `usd_scene_parser` | 与上者构成 USD 工具链 |
| `urdf_loader.py` | 纯 Python URDF 解析器(link/joint 树、全局变换、碰撞网格收集) | 与 native `load_urdf`(§3.1)是**两条并存的 URDF 路径**;宿主侧 FK/网格收集用它 |
| `trajectory.py` | `Assets/trajectories/*.txt` 文本轨迹读取/插值器(每行 `time + N 个 revolute rad + prismatic m`) | 回放示例的数据格式定义处 |
| `mesh_utils.py` | 网格简化与凸分解工具(engine.py / usd_scene_parser / CLI 共用) | — |
| `utils.py` | 路径解析/数学转换 | — |

### 12.2 `tools/` 脚本与 hybrid 资产断链

- `tools/` 两树同装 8 个 Python 脚本(逐字节相同;phase-cd 另多一个 `probe_nested_graph_capture.cu` 图捕获探针)。用户相关:**`fix_obj_winding.py`**——独立 winding 修复工具(`trimesh.fix_normals`,仅限闭流形;引擎视用户碰撞几何为权威,不静默改面索引)。README §7 待核实项 3 与 API_EXECUTION 附录 A#7 的 "examples/ 均不存在" 可关闭:文件**在两树 `tools/fix_obj_winding.py`**,`case_26_perf_tuned.py` docstring 的 `examples/` 前缀陈旧。其余为资产组装(`build_umi_obb_urdf.py`、`build_umi_finray_strategyF.py`)与诊断/剖面脚本。
- **`tools/build_hybrid_mesh.py` 已不在树内**(两树 `find` 亲验 0 命中)。§3.9 `add_hybrid_fem_body` docstring 对它的引用是陈旧断链;hybrid 资产现状 = **只能使用 `Assets/sim_data/` 现成 npz**(`umi_hybrid_sf_v800`/`v1340`/`v1690` 的 `UMI_finray_{L,R}_unified.npz`,另有 `hybrid_d/` 的 `STRATEGY_F` 与 `CASE40_BRIDGE_B`),新混合爪资产当前不可再生。§3.9 已就地标注此断链。

---

## 13. 附录

### 13.1 抛错条件汇总

| 异常 | 触发点 | 适用线 |
|---|---|---|
| `ImportError` | 原生模块找不到;`STIFFGIPC_NATIVE_DIR` 指错;`gpu_rl_tensors()` 无 torch | 两线 / phase-cd |
| `ValueError` | 未知 `multienv_mode`;Config 负能量容差;episode action 形状/缺失(phase-cd) | 两线 |
| `LifecycleError` | 同进程跨 multienv_mode / 跨 per_env_exit / STIFF_* 签名改动;save/load_checkpoint 未 finalize;重复/失败后 finalize;episode 在飞时 step();`set_vertex_env_ids` 进程占用/未 finalize;跨引擎 cuda_device 冲突 | 【仅 phase-cd】(稳定线等价路径多为 RuntimeError/printf) |
| `CheckpointError` | checkpoint 格式/校验和/拓扑/有限值失败 | 【仅 phase-cd】 |
| `ConfigurationError` | `STIFF_KNOB_STRICT=1` 下未登记 STIFF_* 变量;cudaSetDevice 失败 | 【仅 phase-cd】 |
| `IndexError` | `get_abd_body`/`get_fem_body` 找不到 | 两线 |
| `KeyError` | `get_vertex_contact_forces` 非法 components | 两线 |
| `std::invalid_argument`(→ Python RuntimeError 系) | joint 非法/相同 body id;`set_body_groups` 长度/稠密/通配违规(**finalize 时**才爆);`add_fem_pins_with_local_pos` 三数组长度不一;set_config 非有限能量容差 | 两线 |
| `std::runtime_error` | `set_body_friction`/`set_soft_body_density`/`set_per_tet_young_for_body` 参数违规;`set_body_external_wrench` 长度非 12 | 两线 |
| `std::out_of_range` | `get_mesh_asset`/`get_load_record` 越界 | 两线 |
| **进程退出(exit/abort,无法捕获)** | mesh 文件打不开(exit(-1));FEM 之后加载 ABD(abort) | 两线 |

### 13.2 engine.py 环境变量清单(API 层)

完整 170 个 STIFF_* 旋钮登记于 `StiffGIPC/config/knob_registry.h`【仅 phase-cd】,分类与出处见环境变量分册;此处仅列包装层直接消费者:

| 变量 | 作用 | 适用线 |
|---|---|---|
| `STIFFGIPC_NATIVE_DIR` | 覆盖原生模块搜索路径 | 两线 |
| `STIFF_MULTIENV_MODE` | 覆盖 `Config.multienv_mode` | 两线 |
| 模式旗:`STIFF_BVH_ENVDET / STIFF_PERENV_BVH / STIFF_DECOUPLE_THRESH / STIFF_PERGROUP_KAPPA / STIFF_SEGMENTED_PCG / STIFF_PERENV_ALPHA / STIFF_PERENV_PAR`(isolated);`+ STIFF_EE_CANON / STIFF_EE_DETGATE / STIFF_CCD_CANON / STIFF_SPMV_DET`(strict);`STIFF_EE_LB=2`(merged setdefault) | 由 `resolve_multienv_mode()` setdefault;显式设置总是赢 | 两线 |
| per-env-exit 旗:`STIFF_PERENV_MASK / STIFF_PERENV_TELEM`(签名另含 `STIFF_PERENV_MASK_DEV`) | `per_env_exit=True` 解析产物 | 两线 |
| `STIFF_AUTO_PREPARE_AT` / `STIFF_MS_DUMP` / `STIFF_ITER_LOG` | step() 测试旋钮(§6.3) | 【仅 phase-cd】 |
| `STIFF_FRAME_GRAPH / STIFF_FRAME_FULL_GRAPH / STIFF_C4_COLLISION_GRAPH / STIFF_C6_ABD_STEP_GRAPH` | 整帧图族(OVF 恢复协议也临时使用) | 【仅 phase-cd】【实验性,默认关】 |
| `STIFF_PCG_TOL` | 运行时覆盖 pcg_tol | 两线 |
| `STIFF_KNOB_STRICT` | 未登记 STIFF_* 变量从 WARN 升级为 ConfigurationError | 【仅 phase-cd】 |

### 13.3 调用时序速查(finalize 分界)

| 阶段 | 允许的 API |
|---|---|
| **finalize 前** | 全部 `load_*`;`add_*_joint`;`add_stitch_spring`;`add_fem_pin*`;`set_vertex_boundary(-ies)`;face-orient 四件套;per-body 材料全族(§5);`set_body_groups`(声明);`add_collision_exclusion`;`add_ground_collision_skip`;`set_urdf_mesh_override`(load_urdf 前);宿主读数 `vertex_count_host`/`get_vertex_position_host`/`get_vertices_host` |
| **finalize 后** | `step()`;全部 GPU 读数(§7/§8);teleport 族(§9);`set_vertex_env_ids`;`set_env_offsets`;`set_fixed_joint_strength`;`set_body_apply_gravity`;`set_body_external_force/wrench`;`set_prismatic_limit_barrier`;drive 目标/strength/slew 全族;checkpoint(phase-cd);RL/驻留全家(phase-cd) |
| **任意时刻** | `set_log_level`;`reset()`(拆世界保 Config);计数属性 |

---

*本分册基于 phase-cd HEAD `b3ab747` 与稳定线 v0.8.5.3 工作树逐行核对撰写;所有 `文件:行号` 出处可直接在对应仓库检索。文中标注"待核实"的条目未经运行验证或存在代码内部矛盾,采信前请以实验确认。*
