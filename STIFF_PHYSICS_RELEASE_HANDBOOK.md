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

### 5.1 端到端安装测试

在一个干净的 conda 环境中测试（模拟用户首次安装）：

```bash
# 创建干净环境
conda create -n test_stiff python=3.11 --no-default-packages -y
conda activate test_stiff

# 从 GitHub 安装
cd ~/Downloads
git clone https://github.com/haoxiangNtu/stiff-physics.git test-stiff-physics
cd test-stiff-physics
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v0.1.0/stiff_physics-0.1.0-cp311-cp311-linux_x86_64.whl
pip install polyscope scipy

# 运行核心示例
python examples/case_26_arm_cloth_semi_implicit.py
# 应弹出 Polyscope 窗口，点 Run 启动仿真

# headless 测试（无 GUI）
python examples/headless_joint_control.py
# 应打印 vertex 数据，3 帧后退出

# 清理
conda deactivate
conda env remove -n test_stiff -y
rm -rf ~/Downloads/test-stiff-physics
```

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
| 运行依赖 | `numpy`（wheel 自带）、`polyscope`（可视化）、`scipy`（旋转计算） |
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
