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
4. SpMV 形状（PCG graph 每 solve 重录的根因）：P1 已 tier 化 ✓。
   ⚠️ 2026-07-23 实证：`d_unique_key_number` 挂在共享块
   `d_contact_start_block+4`，装配管线的中间 kernel 会覆盖它——solve 前的
   host 重发布**不是冗余**（删除后 strict 哈希漂移 f7fb→ce42，三比内部仍
   一致=错得一致的静默错误，靠跨版本哈希对照抓获）。P3a 完全体必须给
   unique count 一个**独立、不复用的设备槽**，converter 直写该槽后才可删
   host 重发布与 mirror D2H。
另：`_make_unique_indices` 旧路径（cub::Unique，:93-127）确认只有注释引用，
生产走 warp_reduction 路径 ✓ 不需迁移。

## 立即项状态

- [x] sm_80 加入默认构建架构（本分支）
- [x] tools/probe_conditional_graph.cu（A800 探测工具，待百度云执行）
- [ ] A800 gap profile（需 A800 窗口）
- [ ] R535 → 受支持分支的迁移规划（运维项）

## P3b-1 六边界消除状态（2026-07-23）

| 边界 | 状态 | 提交 | 机制 |
|---|---|---|---|
| Newton 收敛决策 | ✓ codex | d8035e6 | phase 链 + FrameDeviceState |
| CCD α / LS 试步 | ✓ codex | d8035e6 | PHASE_CCD→LINE_SEARCH→POST 链 |
| MAS 层级 extent | ✓ | 9bee717 | 分配背书上界 m_allocClusterTotal + 消费 kernel 读 d_levelSize[levelnum] 裁切（可空 extent 指针，legacy 位级不变） |
| converter unique | ◐ 全局 convert 已消；ABD 链保留 | 41aff31 + 8229166 | bound 布局仅限 start==0 的最终全局 convert（消费端 *d_unique 守卫 + pad 中性化）。ABD slice/中段 convert 是 展开×16/归并收缩 循环的收缩步，bound 会每帧 ×16 复利（part3_fem 1.38M→22M→353M），保留 exact 回读，待 tier-dispatch 收编 |
| per-env S3 freeze | codex 进行中 | p3b1/s3-freeze 分支 | S4 mask 先例扩展 |
| contact counts / 装配图化 | 未开工（P3b-2） | — | 见下节 |

锚 hash f7fb5a786c2d7935 全程保持（devlevels 0/1 × convert-count 0/1 × graph 0/1）。

## P3b-2 剩余工作（1+1+1 终态）

现状（4090, strict quad, FG=1 稳态, 33f33b7 后）：
**6.2 graphLaunch + 4.8 streamSync + ~0 deviceSync + 35.5 blocking memcpy
+ 1.2 memcpyToSymbol / 帧**（起点 6.2/43.6/7.7/29.7/35.5）。
已消：muda 便捷方法隐藏 wait（fill/D2D copy/resize/clear —— vendored muda
去 wait，host 内存传输保留）、per-env BVH 池 event fork-join（racecheck 0）、
接触四件套 guard 设备化（d_pairSnapCur/Last 专用快照槽）、ToSymbol
alloc-once。全量门 37 PASS + 锚 16/16 + sanitizer 0 + 双引擎（dewait_gates）。

剩余 35.5 blocking memcpy 构成（DtoH: 6.7×4B 计数 + 2.4×8B 标量 +
2.2×20B cp5 + 2.1×12B + 1.3×16B + 1.2×24B cp_gp6 + 0.5×32B）= 检测计数
回读 + ABD 链 exact 收缩 + 分段起点 —— 全部归 tier-dispatch 子块：
ABD 链 tier 化要点：host 反馈值 = tier(nuniq)（稳态常量，无 ×16 复利，
区别于第一版反馈 length 的事故），kernel 按 *d_uniq 精确展开 + pad 中性化
到 tier 边界，OVF 比较 kernel 超档走 required_unique_blocks RETRY（通道已
存在 frame_transaction.cu:1236）。

1. contact counts D2H（h_cpNum/h_gpNum/partition 四段起点）：
   - [x] 接触 triplet tier 布局（406e4a5）：staging 即最终布局、无压实
     D2D；converter scratch 自保容量；bound 公式动态项 ×2；diag 零块守卫。
     途中修出潜伏 bug 966afa1（friction rank 计数被同帧 barrier 污染 →
     exact 时代摩擦 hessian 块静默丢失）。
   - [ ] 5-int/4-int 检测计数 D2H 本身的图内消除——定案设计
     「tier 稳态重放 + OVF-RETRY 跨档重建」（无需 conditional graph，
     A800/R535 兼容）：
     * 帧内 launch 全部按当前 tier 向量（host 静态），kernel guard 读
       设备计数（*_cpNum / d_levelSize / d_unique_key_number）；
     * 检测 kernel 后接 tiny 比较 kernel：设备计数 > 当前 tier 上限 →
       fsm_record_error(OVF_DCD_PAIRS 等) → 终端 FrameStatus 回读时
       host grow + 重建受影响 tier 图 + 同帧重试（既有 RETRY 契约）；
     * partition 四段起点 = class tier 前缀（tier 稳态下 host 静态），
       跨档同走 RETRY；
     * muda TempBuffer 容量单调（resize ≤ capacity 零开销）→ CUB temp
       预热后 capture-safe；growth 帧走图外路径并重建 exec。
     配套机械项：launch guard 设备化（number 参数→设备计数）、
     setup_abd 两处单流冗余 cudaDeviceSynchronize 删除、
     MAS BuildCollisionConnection 的 cpNum tier 化。
2. 装配段图化：per-frame cudaMemcpyToSymbol 全部迁到 alloc/grow 一次绑定
   （g_matbin 在 PrepareHessian_bcoo:2578 每调一次 — 迁到
   initPreconditioner_Matrix；setup_abd g_abd_sysbin/hessbin :50/:72、
   cal_q_tilde g_abd_wrenchbin :58、pcg set_seg_* 同类）。muda/CUB 封装
   capture-safe 化后 ASSEMBLY phase 捕获为 tier 图族。
3. phase 链闭合：ASSEMBLY→PCG→NEWTON_DECIDE→CCD→LS→POST_LS→next NEWTON
   全部 device cudaGraphLaunch 尾链；终端图唯一 FrameStatus D2H；
   step() 单 cudaStreamSynchronize。
4. 归因工具：STIFF_API_AUDIT=1 + 两步差分（tools 提交 d0e81bf）。

⚠️ 判定基准不变：strict 锚 f7fb5a786c2d7935；nsys 1+1+1 结构验证；
compute-sanitizer memcheck/racecheck 零报告。
