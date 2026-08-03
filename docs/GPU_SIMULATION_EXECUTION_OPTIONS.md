# GPU 仿真执行方案：两条正式路线

本文将后续讨论收敛为两个方案。`GPU-native episode` 不再作为独立的
第三种路线；它只是方案一中“正常稳态帧”的一种工作方式。

## 共同目标

- 仿真主计算在 GPU 上执行；
- 接触、CCD、Newton、PCG、线搜索和 RL 状态尽可能设备驻留；
- 正常稳态帧不发生 H2D/D2H 或宿主同步；
- merged 与 isolated 都能处理接触丰富场景；
- overflow、失败和 reset 具有明确且可验证的语义。

## 方案一：CUDA Graph + 边界 CPU 介入

### 执行模型

```text
正常帧：GPU Graph 完成整帧仿真，CPU 不参与
    ↓
设备发现 overflow / invalid / nonfinite / tier 不足
    ↓
该帧在边界发布 status
    ↓
CPU 只在边界执行必要处理：
  增容、重建 CUB plan、重新 capture Graph、或执行一次 fallback
    ↓
下一帧/下一 episode 继续 GPU Graph 稳态
```

### CPU 可以参与什么

只允许在明确的边界做以下工作：

- 读取设备 status；
- 判断本帧是否已经 commit；
- 扩大 buffer/tier；
- 重建需要 host 参数的 CUB plan；
- 重新 capture/instantiate GraphExec；
- 在无法安全重录时运行一次普通 fallback solver；
- 结束当前 episode 并开始下一 episode。

CPU 不应在正常图帧内部参与：

- 查询接触数；
- 查询 Newton/PCG 收敛；
- 每轮 launch kernel；
- 读回位置或速度；
- 做逐帧 reset/reward/done 判断。

### 优点

- 最大程度复用现有 GIPC/Newton/PCG CUDA kernel；
- CUDA Graph 能显著降低大量细碎 kernel 的 launch 开销；
- 失败和扩容语义容易审计；
- 改造风险和验证成本可控；
- merged/isolated 可以逐步加入资格检查。

### 缺点

- overflow 帧或重录边界不是 CPU-zero；
- CUB 的 host-baked `num_items` 仍需固定容量、bucket 或边界重录；
- 无法在标准 CUDA Graph 内任意增加节点、分配显存或重建 GraphExec；
- 最坏情况下可能频繁 fallback，影响吞吐和 RL episode 连续性。

### 正确的对外承诺

> 正常稳态帧 GPU-only；容量变化、重录和必要 fallback 在帧边界由 CPU
> 处理；边界完成后恢复 GPU-only 稳态。

这不是“任何情况下 CPU 都不参与”，但它是当前工程最稳妥的 CUDA Graph
路线。

## 方案二：Persistent GPU Runtime

### 执行模型

CPU 初始化时只启动一个长期驻留 kernel：

```text
GPU persistent runtime
  ├─ 读取 device task/action
  ├─ build BVH
  ├─ 生成 contact / CCD / triplet
  ├─ 从 device arena 获取工作区
  ├─ Newton { assemble → PCG → line search }
  ├─ 设备侧 reward / done / reset
  └─ 继续下一帧或下一 episode
```

设备侧 runtime 自己推进帧循环、处理分支、发布状态，不依赖 CPU 每帧
launch Graph，也不需要 CPU 为每个 overflow 重录 Graph。

### 必须重写的部分

- 将 host orchestration 改成 device work queue；
- 自定义 device arena allocator 和 generation 管理；
- 自定义 device scan/sort/reduce，或设计有限 bucket；
- 将 Newton/PCG 阶段改写成 persistent kernel 可调用的 device 函数；
- 使用 cooperative groups 实现全局阶段同步；
- 设计 device error、overflow、done 和 recovery 协议；
- 解决长驻 kernel 对 policy、通信和其他 CUDA 工作的资源占用。

### 重要限制

Persistent runtime 不是“在 GPU 上继续调用 CUDA Graph API”。如果内部仍
通过 CUDA Dynamic Parallelism 发射大量子 kernel，launch 开销仍然存在。
要真正减少 launch，必须把细碎 kernel 合并为 persistent kernel 内的
device function 或 work-queue task。

### 优点

- 可以把帧循环、动态分支和 episode 循环留在 GPU；
- 不需要标准 CUDA Graph 的逐 episode 重录；
- 更适合动态接触数、动态 Newton 迭代、设备 policy/reward/reset；
- 理论上最接近真正的 GPU-native 仿真。

### 缺点

- 需要重构大量现有求解器代码；
- device 全局同步和 allocator 复杂；
- CUB、排序、扫描和临时内存不能直接复用现有 host API；
- 长驻 kernel 的资源、异常恢复和调试风险高；
- 需要重新建立数值等价、sanitizer、跨架构和长时程证据。

### 正确的对外承诺

> 初始化阶段由 CPU 启动 runtime；启动后，帧推进、动态工作区、overflow、
> reset、reward 和 done 都由 GPU runtime 管理。

## 两个方案的直接比较

| 项目 | CUDA Graph + 边界 CPU | Persistent GPU Runtime |
|---|---|---|
| 正常帧 CPU | 无 | 无 |
| overflow 处理 | 边界 CPU | GPU runtime |
| 扩容/重录 | 边界 CPU | 不需要 Graph 重录，但需 device arena |
| 现有代码复用 | 高 | 低 |
| kernel launch 优化 | CUDA Graph | 单次长驻 kernel/内部任务 |
| 动态性 | 中等 | 高 |
| 实现风险 | 中 | 很高 |
| 近期可交付性 | 高 | 低 |
| 适合当前分支 | 是 | 作为后续研究路线 |

## 建议的讨论顺序

### 近期工程路线

优先推进方案一：

1. 完整接触链进入整帧 Graph；
2. 设备计数器和容量 tier 完整化；
3. 正常帧零 H2D/D2H/同步；
4. overflow 帧明确在边界由 CPU 处理；
5. 证明 fallback 后下一帧/下一 episode 的语义正确；
6. 完整 FOLD-SHIRT merged/isolated 1550 帧验证。

### 长期研究路线

若方案一的 fallback、重录或容量浪费成为主要瓶颈，再切入方案二，优先
把最动态的 contact queue、设备 reset、reward/done 和容量管理改造成
persistent GPU 子系统，Newton/PCG 主体可以暂时保留 Graph。

## 最终判断标准

方案一只有在以下条件同时满足时才算完成：

- 稳态帧零宿主同步；
- overflow 不污染已提交状态；
- fallback 后数值与普通路径等价；
- 重录后 Graph handle、容量和 episode 计数不串扰；
- merged/isolated 完整长轨迹通过。

方案二只有在以下条件同时满足时才算可用：

- persistent kernel 能连续推进长 episode；
- device arena 不越界且可复用；
- 动态接触和 Newton/PCG 阶段无死锁；
- GPU 侧 done/reset/reward 正确；
- 长时间运行不需要 CPU 重启或重录；
- 性能确实优于方案一，而不仅仅是理论上更“纯 GPU”。

