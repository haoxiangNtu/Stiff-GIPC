# core — 帧编排层（v0.8.6 Phase 2d）

唯一知道"一帧怎么走"的地方。机制（mechanism）分别住在自己的属主文件里，
这里只做**编排（policy/orchestration）**：调用顺序、循环与退出条件、lag 策略。

| 文件 | 内容 |
|---|---|
| `ipc_solver.cu` + `ipc_solver.inl` | `GIPC::IPC_Solver`（帧循环）、`GIPC::solve_subIP`（Newton 循环+隔离冻结）、`GIPC::lineSearch` / `GIPC::postLineSearch`。从 gipc_modules/14 迁入（2d）；**2d step 2 起为独立 TU**（`ipc_solver.cu` 提供 extern kernel/诊断声明外壳，kernel 定义全部留在机制模块） |
| `frame_pipeline.h` | 帧内阶段顺序的**唯一**权威文档（纯注释头）。改 ipc_solver.inl 的阶段顺序必须同 commit 改它 |

规则：
- 新的帧级/迭代级编排逻辑写在这里；新的 kernel、内存管理、不变量**不写在这里**
  （去 contact/pair_buffers、multienv/isolation、device_common/reductions、
  linear_system 等属主文件）。
- 帧号、Newton/PCG/碰撞与阶段耗时均属于 `GIPC` 实例。不得再引入
  `totalNT`/`total_Frames` 一类 translation-unit 全局状态，否则两个 Engine
  会相互污染统计和 checkpoint 时间轴。checkpoint 的格式与 I/O 独立位于
  `checkpoint/checkpoint_io.cu`。
