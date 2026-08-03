# GPU 仿真执行方案：两条正式路线（复审核订版）

更新时间：2026-08-03

审核对象：`codex/full-dynamic-graph`（当前 HEAD `ec1f1a3`）

Persistent 实验分支：`codex/persistent-gpu-runtime`（当前仅分叉，尚无 runtime 实现）

本文只保留两条正式路线：

1. 固定拓扑 CUDA Graph，CPU 负责必要的帧边界控制、overflow、扩容和重录；
2. Persistent GPU Runtime，由长期驻留的设备执行器自主推进仿真。

本文区分“目标设计”和“当前代码事实”。未通过对应门禁前，不使用“完全
GPU-only”或“GPU 自主无限扩容”等表述。

## 共同目标

- buildBVH、接触/CCD、Newton、装配、PCG 和线搜索的主要计算留在 GPU；
- merged 与 isolated 都能运行接触丰富场景；
- 正常路径减少细碎 kernel 的 host launch 成本；
- overflow、rollback、fallback、扩容和状态 commit 有明确事务语义；
- 最终以 FOLD-SHIRT 长轨迹、nsys、数值等价和 v0.8.5 性能对比裁决。

## 方案一：固定拓扑 CUDA Graph + CPU 边界控制

### 1. 当前代码实际做了什么

当前普通 `step()` 的整帧图已经把帧内部的条件循环录入 Graph，但 CPU 仍在
**每一帧**参与控制面：

```text
CPU 准备 FrameBegin/FrameTerminal input
  ↓ H2D
CPU cudaGraphLaunch(full_exec)
  ↓
GPU 执行 BVH → contact/CCD → Newton/PCG/LS → status
  ↓ D2H（恰好一个 FrameStatus）
CPU cudaStreamSynchronize
  ↓
CPU 裁决 status、更新 kappa/telemetry/接触计数镜像
```

代码证据：

- `frame_transaction.cu:1211`：Graph 内有 FrameBegin H2D；
- `frame_transaction.cu:1293`：Graph 内有 FrameTerminal H2D；
- `frame_transaction.cu:1304`：Graph 内有 FrameStatus D2H；
- `frame_transaction.cu:1332`：普通 full graph 明确要求恰好一个 D2H；
- `frame_transaction.cu:2491`：CPU 发射 `cudaGraphLaunch`；
- `frame_transaction.cu:2505`：CPU 每帧 `cudaStreamSynchronize`；
- `frame_transaction.cu:4178`：接触帧在边界回读设备 pair snapshot。

因此当前准确表述是：

> **帧内部物理计算 Graph 化；CPU 每帧仍做一次 launch、等待、状态回读和
> 边界裁决。**

当前代码尚不符合“正常帧 CPU 完全不参与，只有 overflow 才唤醒 CPU”。

### 2. overflow 的当前语义

设备在图内检测 DCD/CCD/contact-triplet/unique/MAS 容量越界并写入 status。
CPU 在当前物理帧的边界：

1. 读取失败 status；
2. 恢复帧首 FEM/ABD/kappa/contact 快照；
3. 增长对应 capacity tier 并推进 buffer generation；
4. 默认不在新图中重试该帧，而是用 release solver 完成同一个物理帧；
5. 下一帧按新 generation 重新 capture/instantiate GraphExec。

`STIFF_GRAPH_INGRAPH_RETRY=1` 只是实验开关；默认是边界 fallback，因为图内
重试改变 grid/reduction 顺序，曾放大接触场景的非确定性。

该机制的方向正确，但 overflow 帧不是 GPU-only，而且 fallback 不是一项
轻量状态处理——CPU 会重新组织并运行普通 solver 路径。

### 3. 方案一有两个可选契约

#### 契约 1A：同步 host-stepped Graph（当前现实路线）

- CPU 每帧发射一张 Graph；
- CPU 每帧等待并读取一个 status；
- 帧内部不做 Newton/PCG/LS 的逐迭代 host 往返；
- overflow 时在同一边界恢复、扩容、fallback，下一帧重录。

这是当前代码最接近且最容易完成的产品契约。对外应称为“整帧 GPU Graph”，
不能称为“正常帧 CPU-zero”。

#### 契约 1B：正常帧不触碰 CPU，仅 overflow 才返回边界

这要求 GPU 自主连续推进多个帧，否则 CPU 无法在不检查每帧 status 的前提下
知道是否 overflow。必须增加：

- 设备 action/task ring；
- 多帧设备循环或 tail-launched Graph；
- 设备 status/done ring；
- overflow 时停止继续 commit；
- CPU 通过 event/轮询在异常或批次完成时介入。

它仍属于固定拓扑 Graph 路线，但执行机制本质上是多帧驻留。若动作必须由
CPU 每帧在线产生，就无法同时满足“CPU 只在 overflow 时介入”。

### 4. 当前验证事实

已经成立：

- merged FOLD-SHIRT（7187 顶点、42 ABD）完成 1551 帧；
- 审计结果 `full=1545, fallback=6, overflow=0`；
- frame graph normal、rollback、unique-tier overflow retry 门禁通过；
- 多帧 device-native Graph API 的 4 帧单次 launch 原型通过，Graph 审计
  `h2d=0, d2h=0`。

仍未成立：

- isolated 四环境完整 1551 帧；
- FOLD-SHIRT graph-vs-release 完整最终状态/状态轨迹等价；
- C4 squeeze 的 kappa 门禁曾超过已有噪声包络；
- 所有 CUB `num_items` 和 contact/triplet 段的设备长度覆盖；
- A800 接触丰富长窗口 nsys 证明；
- Graph-on 对大场景的性能优势。已有文档测量中 Graph-on 常慢于 Graph-off。

### 5. CUB 和“动态图”的边界

现有 CUB sort/scan/reduce 的 `num_items` 多数仍由 host 在 capture 时烤入。
固定容量 + device live mask 可以保证正确性，但不等于所有算子都按实时长度
执行。可选工程策略只有：

- 固定容量执行并保证 padding 中性；
- 预录有限 capacity bucket，在边界选择/重录；
- 为高收益段实现 device-controlled primitive；
- 接受容量宽度空转，若实测 launch width 不是瓶颈则不改。

标准 CUDA Graph 不能在图内 `cudaMalloc`、增删节点或重新 instantiate
GraphExec。

### 6. 方案一准确承诺

短期可承诺：

> 正常帧的物理内部由一张条件 CUDA Graph 完成；CPU 每帧只负责图发射和
> 边界状态裁决；overflow 帧在边界恢复、扩容和 fallback，随后重录。

只有契约 1B 的多帧驻留门禁通过后，才可以承诺：

> 正常帧 CPU-zero，仅批次完成或 overflow 时 CPU 介入。

## 方案二：Persistent GPU Runtime

### 1. 目标执行模型

CPU 在初始化时发射长期驻留的 GPU 执行器：

```text
Persistent scheduler
  ├─ 读取 device action/task queue
  ├─ build BVH
  ├─ 生成 contact / CCD / triplet
  ├─ 从 device arena 分配临时工作区
  ├─ Newton { assemble → PCG → line search }
  ├─ observation / reward / done / reset
  └─ 推进下一帧或停止并发布 terminal status
```

它不是“在 GPU 上重新 capture CUDA Graph”。它需要设备 work queue、设备
同步和设备内存生命周期，并重构当前 host orchestration。

### 2. 必须修正的扩容表述

Persistent runtime 不能无限自主扩大物理显存。device allocator 只能从 CPU
初始化时已经保留/提交的 arena 中切分空间。

arena 耗尽时只有三种诚实策略：

1. 设备 fail-closed，等待 CPU 分配更大的 arena 并重启 runtime；
2. 初始化时预留最坏容量，运行中只做设备侧子分配；
3. 将任务分块/降级并明确报告，不允许静默丢失接触。

因此方案二可以减少 Graph 重录，但不能承诺“显存耗尽后仍无 CPU 无限扩容”。

### 3. kernel launch 开销

Persistent runtime 只有在细碎 kernel 被融合为 `__device__` 阶段或设备任务时
才能真正消除 launch 成本。若内部继续用 CUDA Dynamic Parallelism 发射原有
子 kernel，launch 开销和调度成本仍然存在，且可能比 CUDA Graph 更慢。

Newton/PCG/排序还需要跨 block 全局同步。Cooperative Groups 要求整个
cooperative grid 满足驻留条件；问题规模过大时不能简单把所有原 kernel 塞入
一个巨型 persistent kernel。

### 4. 主要重构和风险

- host orchestration → device scheduler/work queue；
- 固定上限的 device arena、generation、rollback；
- CUB host API → 自定义 primitive、固定 bucket 或混合子图；
- Newton/PCG/LS 阶段的 grid-wide barrier；
- persistent kernel 与 policy、通信、渲染的 SM 资源共存；
- CUDA 异常后的 runtime 终止和恢复；
- 浮点归约顺序改变后的重新定锚；
- sanitizer、跨架构、长时程和吞吐证据全部重建。

### 5. 推荐的原型顺序

在迁移完整 GIPC 前只做三个有独立裁决价值的原型：

1. device task ring + persistent scheduler，连续推进简化帧状态；
2. device arena allocate/reset/rollback/overflow，验证无越界和 generation；
3. 一个简化 PCG 或 contact pipeline，与普通 launch 和 CUDA Graph 对比。

只有原型证明吞吐或动态性收益，才逐段迁移 BVH/contact/Newton。可以考虑
“persistent control plane + 预实例化 CUDA 子图”的混合设计，但子图拓扑和
物理显存上限仍然固定。

### 6. 方案二准确承诺

可追求：

> 在预分配 arena 的容量范围内，GPU runtime 自主推进帧、处理动态任务、
> reward/done/reset 和可恢复 overflow，不需要 CPU 逐帧发射或 Graph 重录。

不可承诺：

> arena/物理显存耗尽后仍能无 CPU 无限扩容并继续运行。

## 两个方案的修订比较

| 项目 | 方案一：固定 Graph | 方案二：Persistent Runtime |
|---|---|---|
| 当前实现状态 | 整帧图已存在，仍有每帧 CPU 边界 | 只有独立分支和设计文档 |
| 正常帧 CPU | 当前有一次 launch、sync、status 裁决 | 目标为无逐帧 CPU |
| 物理内部 host 往返 | Newton/PCG/LS 已可消除 | 目标为全部设备调度 |
| overflow | CPU 边界恢复、扩容、fallback | arena 内设备处理；硬耗尽仍需停机/CPU |
| 动态显存 | CPU 边界增长 | 仅能在预分配 arena 内子分配 |
| 现有 kernel 复用 | 高 | 低；混合子图可提高复用 |
| launch 优化 | CUDA Graph | 融合设备阶段/长期驻留调度器 |
| 实现风险 | 中 | 很高 |
| 近期可交付性 | 高 | 先做原型裁决 |
| 性能结论 | 未证明普遍快于 Graph-off/v0.8.5 | 完全未知，必须实测 |

## 推荐决策

1. 以方案一契约 1A 收口正确性和长轨迹，避免把“整帧 Graph”误称为
   “CPU-zero”；
2. 若产品要求 CPU 只在 overflow 时出现，在方案一内实现并验证契约 1B；
3. Persistent 分支先完成 scheduler/arena/简化 solver 三个原型；
4. 用 nsys 和端到端吞吐比较 1A、1B、Persistent 原型与 v0.8.5，再决定是否
   重写完整执行器。

## 交给下一轮审核者的问题

请重点复核：

1. 普通 full graph 是否还有本文遗漏的每帧 host work、H2D/D2H 或同步；
2. overflow fallback 是否在所有 FEM/ABD/contact/friction 状态上真正回到帧首；
3. `full=1545/fallback=6` 中六个 fallback 的原因和数值影响；
4. isolated 四环境 1551 帧为何尚未完成，是否存在结构性资格限制；
5. CUB 固定容量路径的 padding 是否对所有消费者都数值中性；
6. Persistent arena 硬耗尽时是否存在可行的纯设备恢复策略；
7. Persistent + device-launched/pre-instantiated Graph 混合路线是否比巨型
   cooperative kernel 更适合当前代码；
8. 哪条路线在 A800 的真实接触丰富 RL 工作负载上能超过 v0.8.5。
