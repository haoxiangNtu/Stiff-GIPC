# 整帧 GPU 状态机（Frame-FSM）开发计划

> 分支：`perf/v0842-frame-fsm`（基于 `perf/v0842-gpu-control`，后者保持冻结）
> 立项依据：2026-07-23 Claude × codex 联合可行性评估（有条件立项）。
> 目标环境：A800 / R535（CUDA 12.2 能力）为一等平台；4090 不回退 >3%。

## 冻结的产品契约

1. **正常帧**：CPU 一次 `cudaGraphLaunch(root)` + 引擎专属 stream 一次同步 +
   读取一个 128–256B 的 `FrameStatus`（结果码/阶段/迭代计数/容量水位/关键
   alpha/energy/invalid 位）。禁止 `cudaDeviceSynchronize()`。
2. **溢出帧**：不提交物理状态——GPU 端回滚到帧首事务快照，状态
   `RETRY_REQUIRED(required_*)`，CPU 扩容并重建受影响的分档图后**重试同一帧**；
   物理帧号只在提交图中前进。
3. finalize/warm-up 期允许分配/录制/实例化/上传；帧内零 cudaMalloc、零中途
   D2H、零 host 决策。拓扑/配置变化只在帧间 mark-dirty 重建。
4. 任意 host callback（软约束 functor 等）不属于 fast path——自动回退 legacy
   host solver。legacy 路径保留至全功能 gate 绿。
5. 控制模型：**纯 tail-launch 图族**（12.2 可用，PCG 自环已验证）；conditional
   node 在 A800 实测 probe 通过前视为不可用（tools/probe_conditional_graph.cu）。

## 状态机骨架

```
host launch ROOT
  FRAME_BEGIN --(事务快照/输入拷贝/initKappa/friction)--> NEWTON_DISPATCH
    ASSEMBLY_TIER[S/M/L/XL] --> PCG_BATCH (self-tail) --> POST_PCG_DECIDE
      | converged --> SUBSTEP/FRAME_COMMIT
      | continue  --> CCD_TIER --> LS_TRIAL (self-tail) --> POST_LS --> NEWTON_DISPATCH
  any state -- fatal/overflow --> ROLLBACK_TERMINAL --> FrameStatus
```

要点：容量分档图族（不做单一最大容量——dual-arm 保守上界 102 GiB 反证）；
设备端 transition kernel 依计数选档，每次只 enqueue 一个后继（pending 高水位
≈1，255 上限无关）；每引擎独立 exec family；padding 线程在任何 load/store/
atomic 前退出；`_emit_slot` 的 trash-slot 并发写改为"只计数置位不写 payload"。

## 阶段与验收门（P2 后 go/no-go）

| 阶段 | 内容 | 人日 | 退出门 |
|---|---|---:|---|
| P0 | FrameDeviceState/FrameStatus ABI；专属 stream；merged BVH device bbox+CUB workspace；全分配器/隐式 sync 盘点清单；production 无日志 variant；sm_80 CI；**A800 可消除 gap profile** | 10-15 | 数值不变+全 gate/sanitizer 绿；capture 后节点类型合法；**gap 数字支撑收益，否则停** |
| P1 | PCG 零 D2H；按 tier 缓存 graph exec（停止每 solve 重录制） | 10-15 | host/graph/r2r/tier 三比 BIT-OK；PCG 内 0 D2H；A800 无统计回退 |
| P2 | LS/CCD 自环；DCD/CCD 分档；错误上下文（首败 atomicCAS+完整上下文） | 15-22 | LS 内 0 D2H；0/1/8/64 回溯+invalid+溢出注入全过；retry 位级等价 |
| P3a | triplet/unique/SpMV/默认 MAS 设备化分档（最难；可先交付 diag-only fast path + MAS fallback） | 20-30 | diag/MAS 分别过 strict/full gate；显存预算/水位可解释 |
| P3b | initKappa/friction/per-env/substep/commit-rollback 全接入 | 15-25 | Nsight：正常帧 1 launch+1 status+1 sync；四类 bitwise gate；长程 PPO rollout |
| 加固 | 多引擎并发/reset/OOM/云 soak | 12-18 | A800 精确 R535 镜像 soak + 4090 回归 |

数值门（每阶段）：legacy-vs-graph / run-to-run / 跨 tier 同帧 / overflow-retry
vs 直接足量 四比逐字节；N=1/4/8 cross-batch env0；全 sanitizer。
性能门：A800 精确镜像 ≥3 次交错 A/B（中位数+离散度+GPU idle+API 时间+VRAM 峰值）；
最终 P3 门=A800 生产口径中位数 ≥5% 且回收 ≥50% 的 P0 实测 gap。

## 已知 host-only 残留清单（P0 盘点起点，codex 已考证行号）

merged BVH（calcMaxBV D2H、thrust sort）；碰撞计数 D2H+扩容；triplet converter
动态 reserve/unique D2H；默认 MAS 逐层 levelSize D2H+host launch+Thrust scan；
initKappa（FEM 归约 D2H+ABD 梯度整段回拷）；软约束 host functor；joint 控制每帧
临时 DeviceBuffer；isIntersected 内部 cudaMalloc+D2H（默认关）；帧外层 animation
substep+未消费的 min/max D2H；telemetry event/文件写。

## P3a 实现细目（2026-07-23 调研，converter.cu 逐点）

`h_unique_key_number` 的 D2H（converter.cu:181）有四个下游，逐个去 host 化：
1. `block_values` 清零（:187）：nuniq≤length，改按 length 上界 memsetAsync
   （或合并进 combine kernel 的 guard 写零）；
2. mergebin 容量 grow（:200-207，cudaMalloc 帧内！）：按 triplet 容量上界
   T*9*K 一次性预分配（finalize），显存 288T B（已计入容量核算）；其
   cudaMemset 改 cudaMemsetAsync（同步版在 capture 内非法）；
3. scatter/combine 的 launch 尺寸（:225 apply(nuniq)）：capacity-launch
   （按 length 启动 + `u < d_unique_count` 设备 guard）或并入 tier；
4. SpMV 形状（PCG graph 每 solve 重录的根因）：P1 的 tier 化对象——unique
   count 分档，档内固定 num_items+guard；`d_unique_key_number` 成为设备真值，
   host 镜像仅 telemetry。
另：`_make_unique_indices` 旧路径（cub::Unique，:93-127）确认只有注释引用，
生产走 warp_reduction 路径 ✓ 不需迁移。

## 立即项状态

- [x] sm_80 加入默认构建架构（本分支）
- [x] tools/probe_conditional_graph.cu（A800 探测工具，待百度云执行）
- [ ] A800 gap profile（需 A800 窗口）
- [ ] R535 → 受支持分支的迁移规划（运维项）
