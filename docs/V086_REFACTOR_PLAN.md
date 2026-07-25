# v0.8.6 模块化重构蓝图（Modularization Blueprint）

日期：2026-07-25。前置：v0.8.5.1 修复链（b6c1f09..05c3f75）+ 五视角全面审核
（`comprehensive_audit_2026-07-25.md`）。本文是重构的**工程设计**：为什么、
分成什么、按什么顺序、每一步用什么门禁证明没改坏。

---

## 0. 问题定义（为什么"总是这么修不是办法"）

本轮审核修掉的每一类 bug，根源都不是"某行写错"，而是**结构性缺陷**让错误可以发生：

| 修过的 bug | 背后的结构缺陷 |
|---|---|
| towel-strict discard-grow 销毁活矩阵 | 缓冲区**所有权/生命周期不成文**——"此刻 buffer 是死是活"靠注释断言，没有类型/接口强制 |
| DCD>CCD 镜像越界（bdd0776 类） | 不变量**分散在 5 个调用点**人肉维护，没有单一收口 |
| MAS 双容量口径混淆 | 分配尺寸是**局部变量**，用后即弃；断言验的是另一组 buffer |
| I1==0 预留槽垃圾 / 摩擦 rank 错相位 | "发射端预留、装配端必写满"的**槽位契约**只存在于口口相传 |
| barrier 分歧/shfl UB（两处） | 归约模板**复制粘贴 26+ 份**，修一处漏一处 |
| 隔离被 throw 绕过 | 失效处置策略（冻结 vs 杀进程）**散落在每个检测点**，不是一个模块的职责 |
| h_close_* 死镜像 | 宿主镜像**没有统一的刷新纪律**，断了链无人知晓 |

单文件现状：`GIPC.cu` **16,849 行**（kernel、宿主编排、内存管理、多环境机制、
诊断全部混居），`sim_engine.cu` 4,475 行，`mlbvh.cu` 3,171 行，
`MASPreconditioner.cu` 3,126 行。任何修改的爆炸半径都是整个文件。

## 1. 约束（这个项目的重构与普通项目不同的地方）

1. **位级契约是最高约束**。strict 锚 `f7fb5a786c2d7935` 是全部历史验证的锚点。
   nvcc 默认 `-fmad=true`：函数**跨编译单元移动**可能改变 FMA 收缩与内联决策 →
   浮点位级可能漂移。因此"**先 #include 复合拆分（同一 TU，预处理后逐字节等于
   原文件，位级不可能变），后逐个物理分离 TU（每次一个模块+全门禁）**"是唯一
   稳妥顺序。
2. **上游同步压力**。`mlbvh.cu`/装配 kernel 与 KemengHuang 上游同源演化，结构大改
   会让上游 patch 无法移植。涉及上游语义的部分（smooth/mollifier 死分支、close-set
   死链）**明确不动**（用户指令）。
3. **验证成本高**（全门禁 ~40 分钟）。阶段必须**粗粒度、可独立回滚**——每阶段一个
   commit，门禁全绿才落。

## 2. 目标模块架构

```
StiffGIPC/
├── core/                    # 编排层（唯一知道"一帧怎么走"的地方）
│   ├── ipc_solver.cu        #   IPC_Solver / solve_subIP / lineSearch / postLineSearch
│   └── frame_pipeline.h     #   帧内阶段顺序的**唯一**文档（装配→求解→CCD→线搜索）
├── contact/
│   ├── broadphase/          #   mlbvh（LBVH 构建、遍历、发射）〔上游同源，最小扰动〕
│   ├── pair_buffers.{h,cu}  #   ★ PairBufferManager：DCD/CCD 容量、grow-redo、
│   │                        #     trash-slot、set_emit_caps——不变量的唯一属主
│   ├── barrier_assembly.cu  #   barrier/friction G/H kernel（编码解码契约文档随身）
│   └── encoding.h           #   ★ int4 编码契约的单一真源（活编码族+死分支标注）
├── linear_system/           # （现有目录，已相对干净）
│   ├── triplet_matrix.*     #   ★ GIPCTripletMatrix：增长只经 ensure_capacity_{preserve,discard}
│   ├── converter/ spmv/ pcg/ preconditioner/
├── multienv/
│   ├── isolation.{h,cu}     #   ★ EnvIsolation：quarantine 全家（探针/归因/冻结/方向清零/
│   │                        #     skip 表/遥测）——铁律的唯一属主
│   └── env_state.h          #   per-env 状态生命周期（alpha/status/kappa_group 的 reset 纪律）
├── abd_system/              # （现有目录）
├── device_common/
│   ├── reductions.cuh       #   ★ 归约模板的唯一实现（block/warp 规约、中性元素参与、
│   │                        #     掩码纪律）——替换 26+ 份复制粘贴
│   └── mirrors.h            #   ★ HostMirror<T>：宿主镜像的"写者-刷新点-消费者"显式化
└── engine/                  #   sim_engine 拆分（API 校验 / finalize / 资产 / 导出）
```

★ = 直接对应本轮审核 bug 类的"结构性疫苗"。

## 3. 分阶段计划（每阶段 = 一个 commit + 全门禁）

**Phase 0 —— 安全网（本轮已就位）**
`scripts/verify_gates.sh`：G1 位级锚 + G2 towel-strict + G3/G4 双隔离门 +
G5 MAS oracle + G6 kick + G7 revolute + G8 foldshirt 冒烟，一键 PASS/FAIL 表。
以后**每个阶段的合入条件就是这条命令返回 0**。

**Phase 1 —— GIPC.cu 的 #include 复合拆分（本轮执行，零数值风险）**
16,849 行按语义切成 15 个 `gipc_modules/NN_*.inl`，`GIPC.cu` 变成按序 #include
的复合文件。**同一编译单元**：预处理产物与原文件逐字节相同 → 位级不可能变
（拆分脚本断言"各分片按序拼接 == 原文件字节级相等"）。收益：导航/评审/
所有权立即模块化；每个 .inl 的头部即未来 TU 分离的依赖清单草稿。
mlbvh.cu / MASPreconditioner.cu / sim_engine.cu 的同型拆分 = Phase 1b/1c（后续）。

**Phase 2 —— 叶子模块物理分离（每次一个 TU，独立 commit）**
分离顺序按耦合从低到高：
2a. `device_common/reductions` —— 能量/alpha 归约 kernel 家族先**统一到模板**再
    分离（同时消灭 26 份复制粘贴；模板实例化在各 TU 内，FMA 语境逐 kernel 验证）。
2b. `contact/pair_buffers` —— MALLOC/FREE、grow-redo 循环、set_emit_caps 收口成
    PairBufferManager（DCD≤CCD 不变量从"5 个调用点"变成"1 个类的私有事实"）。
2c. `multienv/isolation` —— quarantine 全家 + per-env kernel 迁出。
2d. `core/ipc_solver` —— solve_subIP/IPC_Solver/lineSearch 最后动（最高耦合）。
每步的位级预期：**理论上可能因 FMA/内联漂移**——若 G1 锚变，用 STIFF_KSUM/
CPU oracle 判定是"合法的等价重排"还是真回归；锚合法漂移时按既有流程换金值
（三平台复跑 + 记录），**不允许静默换锚**。

**Phase 3 —— 接口硬化（结构性疫苗落地）**
3a. `encoding.h`：int4 编码契约单一真源（含 smooth 死分支标注——**不激活，只文档化**）。
3b. `HostMirror<T>`：h_cpNum/h_gpNum/h_unique_key_number 等换成带"最后刷新点"
    断言的薄包装（debug 构建下抓"消费陈旧镜像"）。
3c. 槽位契约断言：装配 kernel 出口的 debug-only "预留槽必写满"检查
    （STIFF_SLOT_AUDIT=1 时启用）。
3d. RAII 化 device 缓冲所有权（cudaFree 后置空的系统性解法）。

**Phase 4 —— sim_engine 拆分 + Python 层对齐**（API 校验/finalize/资产/导出四块）

**Phase 5 —— 死代码清理**（单独 commit：幽灵 buffer、零调用函数、注释掉的调用——
**不含** smooth/mollifier/close-set，那三者按用户指令冻结）。

## 4. 风险登记

| 风险 | 缓解 |
|---|---|
| TU 分离引发 FMA/内联位级漂移 | Phase 1 用同 TU 复合拆分归零此风险；Phase 2 逐 TU 分离 + 锚门禁 + 漂移三态判定流程 |
| 上游 patch 无法移植 | broadphase/装配 kernel 保持函数级原样，只动文件归属；encoding.h 只文档化不改码 |
| 归约模板统一改变数值 | 模板参数化"中性元素+归约算子"，逐 kernel 与原实现 diff 指令级验证（nvdisasm 抽查）+ 锚 |
| 重构期间修 bug 冲突 | 阶段粗粒度（天级不是周级），每阶段独立可回滚 |
| 验证时长膨胀 | verify_gates.sh quick 模式做内环，全量只在阶段合入点 |

## 5. 明确不做的事

- 不改物理模型（mollifier/smooth 冻结、close-set κ-加倍死链冻结——用户指令）。
- 不在重构 commit 里夹带行为修复（修复永远单独 commit，先修后搬或先搬后修）。
- 不追求"一次到位的漂亮"——每一步的价值必须独立成立（可导航性/爆炸半径/不变量收口）。

## 6. Phase 1 执行记录

见 `StiffGIPC/gipc_modules/README.md`（模块清单+切分校验）。拆分脚本断言：
`sha256(cat 00..14) == sha256(原 GIPC.cu)`；门禁：`scripts/verify_gates.sh` 全绿
（G1 锚逐位 = f7fb5a786c2d7935）。
