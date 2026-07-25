# core — 帧编排层（v0.8.6 Phase 2d）

唯一知道"一帧怎么走"的地方。机制（mechanism）分别住在自己的属主文件里，
这里只做**编排（policy/orchestration）**：调用顺序、循环与退出条件、lag 策略。

| 文件 | 内容 |
|---|---|
| `ipc_solver.inl` | `GIPC::IPC_Solver`（帧循环）、`GIPC::solve_subIP`（Newton 循环+隔离冻结）、`GIPC::lineSearch` / `GIPC::postLineSearch`，及其 file-scope 计时/计数状态。从 gipc_modules/14 逐字迁入（2d），作为 GIPC.cu 复合 TU 的**最后一个 include** |
| `frame_pipeline.h` | 帧内阶段顺序的**唯一**权威文档（纯注释头）。改 ipc_solver.inl 的阶段顺序必须同 commit 改它 |

规则：
- 新的帧级/迭代级编排逻辑写在这里；新的 kernel、内存管理、不变量**不写在这里**
  （去 contact/pair_buffers、multienv/isolation、device_common/reductions、
  linear_system 等属主文件）。
- 跨 TU 计数器（totalNT/total_Frames/totalCollisionPairs/…，engine_modules/04
  经 extern 消费）定义在 ipc_solver.inl，保持 external linkage；gipc_modules/14
  的 checkpoint 代码经其文件内前置 extern decl 访问 total_Frames——**别改成 static**。
