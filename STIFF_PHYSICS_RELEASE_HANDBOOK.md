# stiff-physics 发布与维护手册

> 本文档供 AI 助手（Claude 等）使用，包含发布 stiff-physics 公开仓库所需的全部上下文。

---

## 1. 项目架构总览

```
私有仓库（源码）                          公开仓库（发布）
/home/ps/Downloads/Stiff-GIPC/            /home/ps/stiff-physics/
├── StiffGIPC/          C++ 引擎源码       ├── README.md
│   ├── GIPC.cu         IPC 求解器核心     ├── examples/         22 个示例脚本
│   ├── GIPC.cuh        头文件             ├── assets/           URDF + mesh 资源
│   ├── sim_engine.cu   Python 绑定桥接    │   ├── sim_data/urdf/ 机器人 URDF
│   ├── sim_engine.h                       │   ├── triMesh/       布料网格
│   └── abd_system/     ABD 刚体系统       │   ├── tetMesh/       体素网格
├── bindings/                              │   └── trajectories/  预录关节轨迹
│   └── pystiffgipc.cu  pybind11 绑定      └── .git/
├── stiff_physics/      Python 包                remote: https://github.com/haoxiangNtu/stiff-physics.git
│   ├── engine.py       Engine/Config 封装
│   ├── robot.py        Robot 关节控制
│   ├── _native/        .so 安装位置
│   └── data/           abd_system_config.json
├── examples/           开发用示例脚本
├── Assets/             完整资源目录
├── CMakeLists.txt      CMake 构建配置
├── pyproject.toml      wheel 打包配置
├── build_venv/         Python 3.12 构建目录
├── build_311/          Python 3.11 构建目录
├── build/              Python 3.10 构建目录
└── dist/               已构建的 wheel 文件
```

## 2. GitHub 账号与安全要求

### 公开仓库信息

| 项目 | 值 |
|---|---|
| GitHub 用户名 | `haoxiangNtu` |
| 仓库名 | `stiff-physics` |
| URL | https://github.com/haoxiangNtu/stiff-physics |
| 默认分支 | `master` |
| 可见性 | **Public** |

### Git 作者安全规则（极其重要）

公开仓库**绝对不允许**出现公司账号的 commit。每次 commit 前必须检查：

```bash
# 检查当前配置
git -C /home/ps/stiff-physics config user.name
git -C /home/ps/stiff-physics config user.email

# 必须是以下值：
#   user.name  = haoxiang002
#   user.email = 110380534+haoxiangNtu@users.noreply.github.com

# 如果不对，先修复再 commit：
git -C /home/ps/stiff-physics config user.name "haoxiang002"
git -C /home/ps/stiff-physics config user.email "110380534+haoxiangNtu@users.noreply.github.com"
```

**禁止出现**：`LiHaoxiang-rbs` 或任何包含 `roboscience` 的邮箱。

## 3. Wheel 构建流程

### 前置条件

- CUDA 12.x 工具链已安装
- Python 虚拟环境已配置
- `scikit-build-core` 和 `pybind11` 已安装

### 构建命令

```bash
cd /home/ps/Downloads/Stiff-GIPC

# 激活虚拟环境（根据目标 Python 版本选择）
source .venv/bin/activate         # Python 3.12
# 或
# conda activate env_isaaclab    # Python 3.11

# 构建 wheel（自动编译 C++ + 打包）
pip wheel . --no-build-isolation -w dist/
```

### pyproject.toml 关键配置

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "stiff-physics"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy"]

[tool.scikit-build]
cmake.build-type = "Release"
cmake.args = ["-DBUILD_GL_VIEWER=OFF"]
wheel.packages = ["stiff_physics"]
```

### CMake 关键配置

- `CMAKE_CUDA_ARCHITECTURES` 默认 `"89;120"`（RTX 4090 sm_89 + RTX 5090 sm_120）
- `BUILD_GL_VIEWER=OFF`（wheel 不含 OpenGL viewer，不需要 GLEW/GLUT）
- `BUILD_PYTHON_BINDINGS=ON`

### Wheel 内部结构

```
stiff_physics-0.1.0-cp311-cp311-linux_x86_64.whl
├── stiff_physics/
│   ├── __init__.py
│   ├── engine.py          # Engine, Config
│   ├── robot.py           # Robot
│   ├── urdf_loader.py     # URDF 解析
│   ├── urdf2usd.py        # URDF → USD
│   ├── usd_scene_parser.py
│   ├── _native/
│   │   ├── __init__.py
│   │   ├── pystiffgipc.cpython-311-x86_64-linux-gnu.so  # Python 绑定
│   │   └── libstiffgipc_core.so                          # C++ 核心库
│   └── data/
│       └── scene/
│           └── abd_system_config.json  # ABD 系统默认配置
└── stiff_physics-0.1.0.dist-info/
```

### 当前已构建的 Wheel

| 文件 | 位置 | Python | 大小 |
|---|---|---|---|
| `stiff_physics-0.1.0-cp311-cp311-linux_x86_64.whl` | `/home/ps/Downloads/Stiff-GIPC/dist/` | 3.11 | 28 MB |

### 已发布的 Release

| 版本 | 日期 | 附件 | URL |
|---|---|---|---|
| v0.1.0 | 2026-04-14 | `stiff_physics-0.1.0-cp311-cp311-linux_x86_64.whl` | https://github.com/haoxiangNtu/stiff-physics/releases/tag/v0.1.0 |

### v0.1.0 源码映射（重要历史注记）

v0.1.0 wheel 是从**未 commit 的工作树**构建的（`pip wheel . --no-build-isolation -w dist/`，2026-04-14 20:25 北京），没有对应的 commit hash 或 tag。公开仓 https://github.com/haoxiangNtu/stiff-physics （tag `v0.1.0` 指向公开仓的 `6a90b42`）是**发布快照**，不是构建源码。

**最接近 wheel 内容的私有仓 commit（事后重建）**：

| 分支 | Commit | 含义 |
|---|---|---|
| `lhx/wip-multi-env-instance` | `87f90be` | wheel-equivalent baseline — engine 稳定性修复 + ground config + 打包配置。用 `/tmp/v010_wheel/` 解包的 Python 文件和 `strings` 出的 binary symbol 做对照逆推。 |
| `lhx/wip-multi-env-instance` | `4108d3d` | post-v0.1.0 增量（诊断 API、USD parser 健壮性、多 build 目录 loader、examples/scripts/docs）—— 这些**都不在** v0.1.0 wheel 里。 |

审计 wheel 里具体包含什么 Python / C++ 行为时，以 `87f90be` 为准，**不要**拿当前 HEAD 或 `4108d3d` 来比对。

重建方法（如何验证 wheel 源码映射）：

```bash
# 1. 解包 wheel 对比 Python 文件
mkdir -p /tmp/v010_wheel && cd /tmp/v010_wheel
unzip /home/ps/Downloads/Stiff-GIPC/dist/stiff_physics-0.1.0-cp311-cp311-linux_x86_64.whl
diff stiff_physics/engine.py        <(git show 87f90be:stiff_physics/engine.py)
diff stiff_physics/urdf_loader.py   <(git show 87f90be:stiff_physics/urdf_loader.py)
diff stiff_physics/usd_scene_parser.py <(git show 87f90be:stiff_physics/usd_scene_parser.py)

# 2. 验证 C++ binary 暴露的符号集合
strings stiff_physics/_native/pystiffgipc.cpython-311-x86_64-linux-gnu.so \
  | grep -E '^(ground_normal|ground_offset|get_last_)'
# wheel 里应该有 ground_normal/ground_offset，但不应该有 get_last_*
```

## 4. 公开仓库更新流程

### 4.1 更新示例脚本或资源

```bash
cd /home/ps/stiff-physics

# 从私有仓库复制更新的文件
cp /home/ps/Downloads/Stiff-GIPC/examples/some_example.py examples/

# 如果需要新资源文件，从 Assets/ 复制
cp -r /home/ps/Downloads/Stiff-GIPC/Assets/triMesh/new_mesh.obj assets/triMesh/

# 检查 git 作者（必须！）
git config user.name    # 必须是 haoxiang002
git config user.email   # 必须是 110380534+haoxiangNtu@users.noreply.github.com

# 提交并推送
git add .
git commit -m "Add new_example script"
git push origin master
```

### 4.2 发布新版本 wheel

> **强制流程：commit → tag → 从 tag 构建 wheel**
>
> v0.1.0 是从未 commit 的工作树直接 `pip wheel` 出来的，事后无法定位"wheel 对应哪份源码"，只能通过解包 wheel + 反查 binary symbol 反推（参见第 3 节"v0.1.0 源码映射"）。后续每个版本都必须先把私有仓代码完整 commit、打 tag，再从干净 tag checkout 构建 wheel，避免重蹈覆辙。

#### 分支政策（必读）

发布工作必须在 `release/stable` 分支上进行。这条分支是**唯一**允许打 release tag 的分支，其他分支的 commit 进 release 必须经过 cherry-pick 审计。

| 分支 | 用途 | 能否 cherry-pick 进 `release/stable` |
|---|---|---|
| `release/stable` | release 源码权威分支，每个 release tag 都从这里打 | — (本身就是) |
| `lhx/daily-v2` | **当前 daily-dev** (引擎 API + hybrid mesh 工作主线) | ✅ 单 commit 审过后可以 |
| `lhx/multi-env-instance` | multi-env-instance feature 分支 (多环境 Python API) | ✅ 单 commit 审过后可以 |
| `lhx/hybrid-mesh` | hybrid mesh 研究 + demo (case_29..41) | ⚠️ 引擎部分镜像 daily-v2; 研究 demo 不进 release |
| `lhx/wip-multi-env-instance` | WIP 暂存（含 post-v0.1.0 待审增量） | ⚠️ 单 commit 审过后可以；不要整分支 merge |
| `lhx/case26-engine-opt` | ⚠️ 性能实验（PCG warm start 等） | ❌ **禁止**，未稳定，详见 `docs/internal/PERF_OPT_HANDOVER.md` |
| `lhx/case26-param-tuning` | ⚠️ 参数调优实验 | ❌ **禁止** |
| `perf/m10-equivalence`、`bench/*` | ⚠️ 基准/实验分支 | ❌ **禁止** |

**违反政策的危害**：上述实验分支的代码未经稳定性验证（部分实测降速、部分依赖未提交的环境改动）。如果需要把里面某个具体优化合入 release，必须：(1) 在 `docs/internal/RELEASE_LOG.md` 记录决策、(2) 单独审 commit、(3) 在 release 前的 worktree 跑过完整 case_26 + headless 回归。

```bash
# 1. 私有仓代码必须 100% commit（git status 必须干净）
cd /home/ps/Downloads/Stiff-GIPC
git status --short          # 必须为空（或只有可忽略的 cache/build 产物）
git diff --stat HEAD        # 必须为 0 文件

# 2. 在私有仓打版本 tag（先确认作者，私有仓用工作账号即可）
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 3. 从 tag 干净 checkout 出来构建 wheel（worktree 隔离避免污染主工作树）
git worktree add /tmp/build_v0.2.0 v0.2.0
cd /tmp/build_v0.2.0
source /home/ps/Downloads/Stiff-GIPC/.venv/bin/activate
pip wheel . --no-build-isolation -w /home/ps/Downloads/Stiff-GIPC/dist/
cd /home/ps/Downloads/Stiff-GIPC
git worktree remove /tmp/build_v0.2.0

# 4. 切到公开仓发 release（注意作者必须是 haoxiang002，参见第 2 节）
cd /home/ps/stiff-physics
git config user.name        # 必须 = haoxiang002
git config user.email       # 必须 = 110380534+haoxiangNtu@users.noreply.github.com
gh release create v0.2.0 \
  /home/ps/Downloads/Stiff-GIPC/dist/stiff_physics-0.2.0-cp311-cp311-linux_x86_64.whl \
  --title "v0.2.0" \
  --notes "Release notes here. Built from private repo tag v0.2.0 commit <hash>."

# 5. 更新公开仓 README.md 里 pip install URL 的版本号，commit + push
```

**Release notes 必须记录的字段**：
- 私有仓 tag 名 + commit hash（让审计能 1:1 还原 wheel 源码）
- 主要变更（feat/fix/perf 分类）
- 已知限制 / breaking changes

### 4.3 仅更新开发构建（不发布 wheel）

在私有仓库中对 C++ 引擎做修改后，可以只重新编译 `.so` 而不构建 wheel：

```bash
cd /home/ps/Downloads/Stiff-GIPC

# 编译所有 Python 版本的 .so
cd build_venv && cmake --build . --target pystiffgipc -j$(nproc) && cd ..
cd build_311  && cmake --build . --target pystiffgipc -j$(nproc) && cd ..
cd build      && cmake --build . --target pystiffgipc -j$(nproc) && cd ..
```

引擎加载优先级（`stiff_physics/engine.py` 中 `_import_native()`）：
1. `from stiff_physics._native import pystiffgipc`（wheel 安装模式）
2. `build_venv/` → `build_311/` → `build/`（开发模式，按顺序尝试）

## 5. 测试流程

### 5.1 端到端安装测试（**release 必跑**）

> **强制规则（v0.1.1 + v0.3.0 双教训）：每次发新 release 之前，**`examples/` 下的每一个 `.py` 文件**都必须从这条 fresh wheel-install 流程跑过一遍**——不是抽样、不是只跑 main path、不是只跑改动过的 demo。
>
> v0.1.1 教训：在私有仓 dev worktree 里跑 example 时 `_INSTALLED_MODE=False`，engine 的 `assets_dir` fallback 到编译时 `GIPC_ASSETS_DIR` 宏（巧合指向源码 `Assets/`），脚本即使没显式传 `assets_dir=` 也能跑通；但用户 `pip install` 后 `_INSTALLED_MODE=True`，fallback 切换到 wheel 内 `stiff_physics/data/`（**只含 `scene/abd_system_config.json` 132 字节**，没有 URDF / mesh），脚本立刻 `URDF file does not exist`。`case_26_render_obj_indices.py` 第一次发布就踩了这个坑（dev 测试通过、用户装好后崩）—— 后续每个 example 都必须显式 `Config(assets_dir=ASSETS_DIR)`。
>
> v0.3.0 教训（依赖版本漂移）：`pyproject.toml` 里 `vis = ["polyscope"]` 没有版本约束，**fresh-env install 抓到的是当下 PyPI 上最新版**——dev env (env_isaaclab, frozen 2026-02-06) 一直是 polyscope 2.5.0、demo 在那里测试通过；但 fresh-env 在 4 月底已经升到 2.6.1，**该版本删除了 `psim.SetWindowFontScale`**（imgui upstream 的 deprecation cleanup 引入的 breaking change，但 polyscope 自己没 bump major 版本号）。`case_replay_user_gui.py` 和 `demo_body_view.py` 调用了这个 API，**fresh-env 测试如果真的全跑就该崩、但 v0.3.0 漏测了**。修复：(a) `vis = ["polyscope>=2.4,<2.6"]` 锁定上限（commit c059969），(b) 把 `SetWindowFontScale` 包 `getattr(... , lambda x: None)` fallback。**根因还是 §5.1 流程没真正全跑——以下流程图加严，不能再漏**。

在一个干净的 conda 环境中测试（模拟用户首次安装）：

```bash
# 创建干净环境
conda create -n test_stiff python=3.11 --no-default-packages -y
conda activate test_stiff

# 从 GitHub 安装
cd ~/Downloads
git clone https://github.com/haoxiangNtu/stiff-physics.git test-stiff-physics
cd test-stiff-physics
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v<VERSION>/stiff_physics-<VERSION>-cp311-cp311-linux_x86_64.whl
# IMPORTANT: 使用 pyproject.toml 声明的同一个 polyscope 版本范围。
# 不要写裸 `pip install polyscope`——那会跨时间不可复现，v0.3.0 就因此漏抓 2.6.1 breaking change。
pip install "polyscope>=2.4,<2.6" scipy h5py

# === 必跑：examples/ 下每一个 *.py 都要 sanity-check ===
# 不允许只测主线（case_26_arm_cloth_semi_implicit）然后假定其他都 OK。
# 用脚本扫一遍，确保没有遗漏：
ls examples/*.py | while read f; do
  echo "=== smoke: $f ==="
  timeout 30 python "$f" --help 2>&1 | head -3 || true
  # 对支持 --help 的 demo 看是否打印；不支持的就 try 启动 5 秒后 SIGINT
done

# 然后每个 GUI demo 至少手动启动一次（Polyscope 窗口必须能弹出，仿真能跑 1 step 不崩）：
python examples/case_26_arm_cloth_semi_implicit.py    # 基础 slider GUI
python examples/case_26_perf_tuned.py                 # tuned variant
python examples/case_26_perf_extreme.py               # extreme variant
python examples/case_26_render_obj_indices.py         # per-body 彩色 (v0.1.1)
python examples/case_replay_user_gui.py               # 477 帧 qpos replay (v0.3.0)
python examples/demo_body_view.py                     # body-view demo
# ...其余 examples/*.py 一个不漏

# headless（无 GUI）
python examples/headless_joint_control.py
# 应打印 vertex 数据 + 100 帧仿真完成，无 RuntimeError

# === sanity check 验证清单 ===
# [ ] 每个 examples/*.py 都至少 import + load + step 1 帧成功（用上面 ls + timeout 循环验过）
# [ ] 没有 "URDF file does not exist" 错误（说明 assets_dir 解析正确，v0.1.1 教训）
# [ ] 没有 "cannot create directory: .../sorted_mesh/" 错误（说明 metis 路径已不烤死）
# [ ] 没有 AttributeError 在 polyscope.imgui 或 stiff_physics.engine.* 上（v0.3.0 教训：依赖漂移）
# [ ] Polyscope 窗口正常弹出，仿真可启动
# [ ] headless 100 帧跑完无崩

# 清理
conda deactivate
conda env remove -n test_stiff -y
rm -rf ~/Downloads/test-stiff-physics
```

**新增 example 时的检查表**：
- [ ] 显式 `ASSETS_DIR = str(Path(__file__).resolve().parent.parent / "assets") + "/"`
- [ ] 显式 `Config(..., assets_dir=ASSETS_DIR)` 不要依赖 `engine.native.get_assets_dir()` 默认
- [ ] 在 fresh wheel-install env 里跑过，确认能加载 URDF / mesh
- [ ] 没有用 wheel 没暴露的 API（比如 `get_last_diag()` 是 post-v0.1.x 才有）
- [ ] 用到的第三方 API（`polyscope.imgui` 等）确认在 `pyproject.toml` 锁定的版本范围下都存在；如果用了可能跨版本变化的 API（如 `psim.SetWindowFontScale`），用 `getattr(psim, "...", lambda *a: None)` 包一层 fallback（v0.3.0 教训）

### 5.2 开发模式快速测试

在私有仓库中修改代码后的快速验证：

```bash
cd /home/ps/Downloads/Stiff-GIPC
source .venv/bin/activate

# 基础仿真测试
python examples/case_26_arm_cloth_semi_implicit.py

# headless A/B 性能测试（比较 Y-up vs Z-up）
python examples/debug_arm_driving_ab.py A_baseline_case26 E_full_test_main
```

### 5.3 headless 自动测试

`headless_joint_control.py` 可以用于 CI/CD 测试，无需 GPU 显示：

```bash
python examples/headless_joint_control.py
# 预期输出：3 帧仿真数据，无 GUI
```

## 6. 系统要求

| 要求 | 详情 |
|---|---|
| 操作系统 | Linux x86_64 (Ubuntu 20.04+) |
| GPU | NVIDIA RTX 4090 (sm_89) 或 RTX 5090 (sm_120) |
| 驱动 | NVIDIA driver with CUDA 12.x support |
| Python | 3.11（wheel 当前编译版本） |
| 运行依赖 | `numpy`（wheel 自带）、`polyscope>=2.4,<2.6`（可视化；2.6 删了 SetWindowFontScale，跟着 imgui upstream，未做 demo 迁移前不能解锁）、`scipy`（旋转计算）、`h5py`（部分 replay demo） |
| 系统库 | `liburdfdom-dev`（`sudo apt install liburdfdom-dev`） |

## 7. 已知问题与注意事项

### Z-up 性能差异

引擎原生设计为 Y-up 坐标系。使用 Z-up（`gravity=(0,0,-9.8)`, `ground_normal=(0,0,1)`）时：
- 仿真结果正确，但 Newton 迭代次数约为 Y-up 的 10 倍
- 根本原因是不同坐标系下布料与地面的接触几何不同，不是代码 bug
- 详见 `examples/Z_UP_PERFORMANCE_FINDINGS.md`

### C++ 引擎已修复的 bug

1. **`_calFrictionHessian_gd`**（`GIPC.cu` 1357-1463 行）— 地面摩擦 Hessian 原来硬编码 XZ 平面，已改为从 `_normal` 动态构建切平面基
2. **`_checkGroundCloseVal`**（`GIPC.cu` 6656 行）— 数组索引 `[gidx]` 改为 `[idx]`
3. **`MALLOC_DEVICE_MEM`**（`GIPC.cu` 8665 行）— 地面法线/偏移改为使用配置值

### 自适应 Kappa 机制未启用

`h_close_gpNum` 和 `h_close_cpNum` 从未从 GPU 拷回 CPU，导致 `checkCloseGroundVal()` 和 `checkSelfCloseVal()` 始终返回 false。Kappa 只在 `initKappa` 时设置一次，之后不会自适应翻倍。这是一个已知的设计问题，不影响正常仿真。

### 开发构建多版本共存

私有仓库中有三个构建目录，编译了不同 Python 版本的 `.so`：

| 目录 | Python | .so 文件 |
|---|---|---|
| `build_venv/` | 3.12 | `pystiffgipc.cpython-312-x86_64-linux-gnu.so` |
| `build_311/` | 3.11 | `pystiffgipc.cpython-311-x86_64-linux-gnu.so` |
| `build/` | 3.10 | `pystiffgipc.cpython-310-x86_64-linux-gnu.so` |

修改 C++ 代码后需要**全部重新编译**。

## 8. 关键文件索引

### 私有仓库（`/home/ps/Downloads/Stiff-GIPC/`）

| 文件 | 用途 |
|---|---|
| `StiffGIPC/GIPC.cu` | IPC 求解器核心（11452 行），Newton 循环、碰撞、摩擦 |
| `StiffGIPC/GIPC.cuh` | GIPC 类定义，默认参数 |
| `StiffGIPC/sim_engine.cu` | SimEngine 实现，Python 配置传入 C++ |
| `StiffGIPC/sim_engine.h` | SimEngineConfig 定义 |
| `bindings/pystiffgipc.cu` | pybind11 绑定层 |
| `stiff_physics/engine.py` | Python 封装：Engine, Config |
| `stiff_physics/robot.py` | Python 封装：Robot 关节控制 |
| `CMakeLists.txt` | CMake 构建配置 |
| `pyproject.toml` | scikit-build-core wheel 打包配置 |
| `examples/debug_arm_driving_ab.py` | headless A/B 性能对比测试 |
| `examples/diag_minimal_gravity_ab.py` | 最小化重力方向对比测试 |

### 公开仓库（`/home/ps/stiff-physics/`）

| 文件 | 用途 |
|---|---|
| `README.md` | 安装说明、API 文档、示例列表 |
| `examples/case_26_arm_cloth_semi_implicit.py` | 核心交互示例（XArm7 + 布料） |
| `examples/headless_joint_control.py` | 无 GUI 关节控制示例 |
| `examples/test_main.py` | Z-up 环境测试脚本 |
| `assets/` | URDF、mesh 等资源文件 |

## 9. 常用命令速查

```bash
# === 开发构建（修改 C++ 后） ===
cd /home/ps/Downloads/Stiff-GIPC/build_venv
cmake --build . --target pystiffgipc -j$(nproc)

# === 构建 wheel ===
cd /home/ps/Downloads/Stiff-GIPC
source .venv/bin/activate
pip wheel . --no-build-isolation -w dist/

# === 更新公开仓库 ===
cd /home/ps/stiff-physics
git config user.name   # 确认是 haoxiang002
git add . && git commit -m "message" && git push origin master

# === 发布新版本 ===
gh release create v0.X.0 /path/to/wheel.whl --title "vX" --notes "notes"

# === 端到端测试 ===
conda create -n test python=3.11 -y && conda activate test
pip install <wheel_url> && pip install polyscope scipy
python examples/case_26_arm_cloth_semi_implicit.py
```
