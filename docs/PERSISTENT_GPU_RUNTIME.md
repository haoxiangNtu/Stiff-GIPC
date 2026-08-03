# Persistent GPU Runtime 设计备忘录

## 目的

本文记录一种比 CUDA Graph 更动态的执行模型：启动一个长期驻留的
GPU kernel，由设备侧 runtime 自己推进多个仿真帧、episode、策略动作和
终止状态。它是未来“真正 GPU-native 仿真”的候选路线，不是当前
`codex/full-dynamic-graph` 分支的默认实现。

## 基本模型

CPU 只在初始化时提交一个 persistent kernel：

```text
GPU persistent runtime
  ├─ 读取 device task/action descriptor
  ├─ build BVH
  ├─ 生成 DCD/CCD/contact/triplet
  ├─ 通过 device arena 获取工作区
  ├─ Newton { assemble → PCG → line search }
  ├─ 计算 observation/reward/done
  ├─ 按 reset mask 选择性重置环境
  └─ 发布下一帧或 episode terminal descriptor
```

runtime 不是在设备上调用 `cudaGraphLaunch` 的循环。要真正消除每个子阶段
的 launch 开销，现有细碎 CUDA kernel 必须改写为 persistent kernel 内可调用
的 `__device__` 函数，或改造成设备 work queue 中的任务。若仍使用 CUDA
Dynamic Parallelism 发射子 kernel，设备端仍会支付 launch 开销。

## 与当前动态图路线的关系

两者都使用“设备计数器控制有效工作量”，但层级不同：

| 能力 | 固定拓扑 CUDA Graph | Persistent GPU runtime |
|---|---|---|
| kernel 拓扑 | 捕获后固定 | 设备侧任务/分支可变 |
| live count | 可以 | 可以 |
| 工作区 | 捕获前预分配 | device arena/固定池 |
| episode 内 CPU | 可以为零 | 可以为零 |
| 任意扩容 | 不行，需重录 | 可由自定义 arena 处理，但需上限/策略 |
| Graph 重录 | 边界由 CPU 调 CUDA API | 不需要 Graph 重录 |
| 现有代码复用 | 高 | 低 |
| 实现风险 | 中 | 很高 |

因此，当前路线不是 persistent runtime；它是 persistent runtime 的一个较
保守子集：固定 Graph 拓扑 + 设备 live-count + 设备状态发布。两者共享
设备描述符、容量 tier、overflow 状态和 RL episode 语义，但 persistent
runtime 还需要重写执行器和内存生命周期。

## 主要困难

### 1. Host orchestration

现有 GIPC/Newton/PCG 路径仍包含 host-side CUB API、buffer 分配、stream/
event、异常处理和迭代控制。这些不能直接放入一个 persistent kernel。

### 2. 全局同步

Newton 的每轮由多个阶段组成。把阶段合并进 persistent kernel 后，需要
cooperative groups 的 grid-wide barrier，并保证所有 block 始终驻留；否则
可能死锁或破坏阶段顺序。

### 3. 设备内存管理

需要自定义 device arena、并发分配、回收、generation 和 overflow 策略。
标准 `cudaMalloc`、Graph API 和 host CUB plan 不能在设备代码中直接使用。

### 4. CUB/排序/扫描

需要固定容量、有限 bucket，或完全自定义 device-controlled sort/scan/
reduce。把 device pointer 伪装成 CUB host `num_items` 并不能解决问题。

### 5. 资源与恢复

长驻 kernel 会长期占用 SM。policy、通信、渲染和异常恢复需要显式资源
规划；runtime 出错时通常要终止整个 runtime，而不是只重放一个 Graph node。

## Fallback、overflow 与 CPU 边界

当前固定拓扑 Graph 路线的正确契约是：

```text
帧内：GPU 执行图，CPU 不参与
  ↓
设备发现 overflow / invalid / nonfinite
  ↓
设备写 status、done、committed_frame_count
  ↓
episode 结束或进入 fail-closed 状态
  ↓
CPU 在 episode 边界读取状态、增容、重新 capture Graph
  ↓
下一 episode 再次进入 CPU-zero 稳态
```

因此，当前实现中遇到 fallback 时，答案是：

- 如果 fallback 指“本帧从整帧 Graph 回到普通求解器”，当前实现需要 CPU
  在帧边界做裁决和调用 fallback 路径；这不是 CPU-zero 的稳态帧。
- 如果把 fallback 改成设备只发布错误并结束 episode，则可以保持该 episode
  内 CPU 零参与，但不能在标准 CUDA Graph 内自动扩容和重录。
- 若要求 fallback、扩容、重录也不让 CPU 参与，必须使用 persistent runtime，
  或者预先分配足够大的固定最坏容量，使运行中永不扩容。

## 推荐采用的阶段路线

### 阶段 A：稳妥路线

继续使用固定拓扑 CUDA Graph：

1. 预分配 contact/triplet/PCG/ABD 最大或分档容量；
2. 用 device descriptor 发布 live counts 和 status；
3. episode 内单次 Graph launch，CPU 不逐帧参与；
4. overflow 不在帧内偷偷回退，改为设备 terminal/fail-closed；
5. 只在 episode 边界由 CPU 重录下一张图。

### 阶段 B：混合路线

把最动态的 contact queue、计数、reset、reward/done 先改成 persistent
GPU 子系统，Newton/PCG 主体仍使用 Graph。这样可减少重录需求，同时保留
已有求解器和验证基础。

### 阶段 C：完整 Persistent Runtime

只有在阶段 A/B 证明容量、数值和吞吐都不足时，才把 Newton/PCG/line-search
整体改成设备 work queue + persistent kernel，并重新建立 sanitizer、跨架构
和 FOLD-SHIRT 长时程证明。

## 结论

“固定拓扑 Graph + 设备 live-count”与 persistent runtime 是同一目标下的
两种执行层级，不是同一个实现。前者已经能实现 episode 内 CPU-zero，且最
适合当前工程；后者理论上可以把动态分支、容量管理和 episode 循环全部留在
GPU，但代价是重写执行器、内存管理和同步模型。

在没有完成 persistent runtime 之前，不应把 episode 边界的 CPU 重录描述成
“完全无 CPU 的动态图”。准确说法是：**稳态仿真 GPU-native，容量变化由边界
协议管理**。
