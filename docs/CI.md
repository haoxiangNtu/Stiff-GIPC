# 验证策略：只在推送时执行

本仓库不安装轮询器或定时任务。分支推送运行完整门禁；tag 推送视为发版，
在完整门禁之后追加重炮验证。任一命令非零退出都会阻止这次推送。

## 安装与升级

每个 clone 执行一次：

```bash
scripts/install_hooks.sh
```

安装器把版本化的 `scripts/hooks/pre-push` 复制到 Git common directory 下的
`stiffgipc-hooks/pre-push`，再把 `core.hooksPath` 设置为该绝对目录。这样从任意
linked worktree（包括较旧提交）推送时仍会使用同一份已安装钩子。拉取到钩子协议
升级后需要重新运行安装器；当前协议号见 `scripts/GATE_PROTOCOL_VERSION`。

不要直接把 `core.hooksPath` 指向当前 worktree 的 `scripts/hooks`：切换到不含
该文件的旧提交后会静默失去门禁。

## 执行语义

- 分支 ref：对将要推送的准确 commit 运行 `scripts/run_push_gates.sh full`。
- tag ref：对 tag 指向的准确 commit 运行 `scripts/run_push_gates.sh heavy`。
- 一次 push 含多个 ref 时，按 commit 去重；同一 commit 只要有一个 tag ref，
  就升级为 heavy。
- 删除 ref 没有待验证的源树，因此跳过。

钩子不会验证“当前工作目录”或复用它的 `build/`。它先将待推送 SHA 放进临时
detached worktree，再从零配置 Release + Python bindings + diagnostics，强制 Python
扩展从该 worktree 的 build 目录加载。跨 worktree 的 GPU 门禁由 common-directory
文件锁串行化。

完整套件包含真实配置/构建、原生扩展路径校验、冻结区绊线、strict 位级锚、
towel、隔离/隔离失败、MAS、ABD/joint、三模式契约、checkpoint、物理不变量和
foldshirt smoke。heavy 在此基础上追加：

- 三模式 FD（真实 FEM、布料/地面/摩擦、接触/摩擦、软约束活动断言）；
- SNK1/SNK2/ARAP 的 diagnostics-off 生产构建、原生模块身份与物理不变量；
- 19 个 headless demo；
- 三模式正确性与性能矩阵（要求已审定 GPU 基线且 GPU 空闲）；
- `memcheck`、`racecheck`、`initcheck`、`synccheck`。

性能基线按 CUDA compute capability 与 GPU 型号共同索引。录制候选必须使用空闲
GPU、完整 anchor/towel/foldshirt、全部三种模式且至少三次重复；任何正确性失败
都不会写出候选：

```bash
BENCH_REQUIRE_IDLE_GPU=1 BENCH_RECORD_BASELINE=1 BENCH_REPEATS=3 \
  python3 scripts/mode_bench.py
```

审核 `gate-results/mode_bench_baseline_candidate.json` 后，再把确认的数据版本化到
`scripts/baselines/mode_bench.json`；重炮门禁本身只读取已审定基线。

## 日志与手工复现

每次验证的元数据、完整输出与结果文件保留在：

```text
$(git rev-parse --git-common-dir)/gate-logs/
```

其中 `history.log` 是提交级结论，`bypass.log` 是紧急绕过记录。失败日志不会因
临时 validator worktree 清理而消失。

手工运行同一入口：

```bash
scripts/run_push_gates.sh full
scripts/run_push_gates.sh heavy
```

## 逃生口与边界

救火时可显式执行：

```bash
SKIP_GATES=1 git push
```

该操作会把 remote、ref 和 SHA 写入 `bypass.log`。Git 自带的
`git push --no-verify` 会令客户端完全不调用 hook，因而本地 hook 无法记录或阻止；
若发版需要不可绕过的保证，仍应在远端 protected branch/tag 上复用
`scripts/run_push_gates.sh heavy`。

`scripts/ci_watch.sh` 与 `scripts/nightly.sh` 只作为可选旧式 runner 附录保留，
默认不安装、不启用。

## 新增门禁（错误分类学 + 配置收口第一批）

- **G13 geometry**（`scripts/geometry_gate.py`，GPU）：初始可行性类型化拒绝的负向
  测试——FEM 穿地必须在 finalize 抛 `GeometryError`；初始互穿必须在第 0 帧首步
  抛 `GeometryError`（线搜索耗尽+非有限势能签名）；干净场景必须不误报。帧中
  NaN 策略不在此门管辖（isolated 检疫 / merged WARN-继续，均保持原契约）。
- **G14 knob-registry**（`scripts/knob_gate.py`，纯静态无 GPU）：
  `StiffGIPC/config/knob_registry.h` 是全部 STIFF_* 环境旋钮的单一真源；本门断言
  C++/CUDA、python 层、mode resolver 读到的每个旋钮都有注册表行。运行时另有
  finalize 绊线：进程环境里出现注册表外的 STIFF_* 变量 → 大声警告
  （`STIFF_KNOB_STRICT=1` 升级为 `ConfigurationError`）。维护规则：新增
  `getenv("STIFF_...")` 必须同 commit 加注册表行，否则 G14 红。
- **G16 frame-graph**（`scripts/frame_graph_gate.py`，GPU）：覆盖 Phase C 两图
  事务、强制回滚、容量 tier 边界重试/耗尽，以及整帧条件图开关前后的逐位
  指纹；整帧路径必须报告一次 graph launch、一次帧尾 D2H。
- **G17 episode-graph / episode-rl-graph**（`scripts/episode_graph_gate.py`、
  `scripts/episode_graph_rl_gate.py`，GPU）：覆盖 Phase D 两段 observation
  event、纯 FEM exec 跨 episode 复用及逐位一致，并用 fixed+revolute+prismatic
  ABD 场景验证整段动作上传和 RL 数值噪声包络。
- **G17c gpu-rl-graph**（`scripts/gpu_rl_gate.py`，GPU）：GPU-native RL 设备
  ABI——稳态一帧图审计必须 0 host / 0 H2D / 0 D2H 节点；闭环 parity 相位逐帧
  对照公开 setter 基线的噪声包络；零同步相位 back-to-back 发射后核对设备
  frame counter 与终态；step()/episode 锁定、stream 亲和拒绝与 `end_gpu_rl()`
  复活均为负向断言。
- **G17d gpu-native-rl**（`scripts/gpu_native_rl_gate.py`，GPU）：同一 ABI 的
  D2D 动作发布变体——动作轨迹一次上载到临时设备缓冲，每步只入队 D2D 发布+
  仿真图+D2D 观测保留（更贴近动作已驻留显存的真实 GPU policy），末尾一次
  裁决读回，轨迹对照 episode 噪声包络。契约与完成度路线图见
  `docs/GPU_NATIVE_RL_PLAN.md`。
- **G19 isolated-graph**（`scripts/isolated_graph_gate.py`，GPU）：C5——
  **isolated 模式整帧图**（`STIFF_C5_ISOLATED_GRAPH=1`）。四环境落地场景 20 帧
  全图执行零回退（per-env CCD alpha 链、per-env 冻结判定、per-env S3 回溯
  WHILE、统一步长回退 IF、图内 kappa 全在设备）。三重断言：轨迹在双基线包络内；
  **物理隔离成立**（扰动 env3 对邻居的耦合 < 其自身位移的 1e-4）；**图化零新增
  耦合**（图 1.79e-07 vs per-env 树基线 1.80e-07）。契约注意：isolated 承诺
  per-env 公平性+检疫铁律，**不承诺逐位复现**（那是 strict）——所以残余抖动
  （共享 SpMV 原子序被接触动力学放大）是预期内的。strict 仍被资格拒绝：
  容量网格归约会合法重排求和序，准入 strict 等于换锚战役。
- **G18 collision-graph**（`scripts/collision_graph_gate.py`，GPU）：C4-a——
  碰撞（BVH+DCD+CCD 标量链+回溯 LS）整体录进整帧条件图
  （`STIFF_C4_COLLISION_GRAPH=1` 显式 opt-in）。双 cube 落地场景 20 帧全图
  执行零 fallback，接触帧在图内多 Newton 迭代收敛；对照双基线，容量网格归
  约的结合序漂移以 1e-9 相对等价容差裁决（实测 ~1e-13）。契约：kappa/
  close-set 与摩擦集冻结在帧边界值（kappa-quiet 窗口内主张 sync 等价；摩擦
  非零场景被资格检查拒绝，留 C4-c）。

## GPU idle 门的基础设施白名单

远程桌面类进程（如 ToDesk）持有小型 CUDA 上下文，会让 idle 门在远程操作期间
永久 INFRA-BLOCK（含 tag push 的重炮包）。豁免方式：
`BENCH_IDLE_ALLOWLIST=ToDesk`（逗号分隔的进程名子串）。默认空=fail-closed；
每次豁免都会打印进门禁工件日志。真正的计算负载禁止入白名单。
