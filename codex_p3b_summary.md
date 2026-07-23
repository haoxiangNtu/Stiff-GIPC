# P3b implementation notes

## P3b-1 子块 1：Newton / LS / CCD 相位边界 host 决策消除

基线为 `4c1b623`，实现提交为 `abe81c3` 与 `d8035e6`。本节改动仅在
`STIFF_FRAME_GRAPH=1` 且正常 global LS graph 可用时生效；
`STIFF_FRAME_GRAPH=0`、legacy LS、诊断/验证开关和不满足捕获条件的路径仍走原有
host packet 逻辑。per-env S3 仍保持原状，留给子块 5 的统一 fallback。

### 相位连接

| 边界 | 改造前 | frame-graph 改造后 |
| --- | --- | --- |
| Newton convergence | host 读取 4 B `gradVanish` 并分支 | `_newton_convergence_decide` 发布 `newton_converged`；`_newton_decide_transition` 在 device 选择 `COMMIT` 或 `CCD` |
| CCD → LS | host 读取 8-double/64 B packet，校验并取 alpha | CCD combine 在 device 校验，调用 `fsm_record_error`，并把 `alpha/cfl_alpha` 与下一 phase 写入共享 `FrameDeviceState` |
| LS → POST_LS | host 读取 384 B `LsDeviceControl`，再调用 `postLineSearch` | LS terminal tail-launch `POST_LS` graph；接受态 close-list 重建和 `ASSEMBLY/ROLLBACK` transition 均在 device |

当前 host 只读取 `FrameDeviceState.phase`（4 B）推进相位。接受接触对数量由一个
mapped pinned mirror 发布，并借同一次 phase D2H 的 stream 同步语义更新 host launch
metadata；host 不读取这些计数来做分支。`FrameStatus.host_boundaries` 现在记录实际
phase 穿针次数，不再用 `newton_iters` 代填。

`postLineSearch` 的历史 Kappa close-check 依赖 `h_close_cpNum/h_close_gpNum`，而这两个
host mirror 在现有实现中从未由 device 回填，因此有效行为是重建下一轮 close list。
POST_LS graph 精确保留该有效 kernel 序列；若 Kappa 未初始化、semi-implicit 开启或
历史 close mirror 非零，则自动退回 legacy 路径。

### host boundary 与 D2H 审计

transaction 固定场景包含一次接受 Newton step 和一次终止 convergence probe：
`newton_iters=1`、`ls_trials=1`、剩余 `host_boundaries=2`，即每次 probe 仅一次
phase-only 穿针。原 `4c1b623` 把 `host_boundaries` 直接写成 `newton_iters`（该场景显示
为 1），没有计入真实的 grad/CCD/LS host 决策，不能作为诚实的前值；按调用点审计，
每个接受 Newton step 已消除 3 个 host 决策 payload（4 B grad、64 B CCD、384 B LS）。

相同命令、相同 merged/N=3/3-frame 场景的 Nsight Systems CUDA memcpy 统计：

| D2H packet | `4c1b623` | `d8035e6` |
| --- | ---: | ---: |
| 4 B | 25 | 28 |
| 64 B CCD | 4 | 0 |
| 384 B LS | 4 | 0 |
| D2H 总次数 | 49 | 44 |
| D2H 总字节 | 4680 | 2900 |

目标 packet 共减少 8 次；phase bridge 净增 3 次 4 B。最终 D2H 调用数下降
10.2%，字节数下降 38.0%。报告分别为
`/tmp/p3b1_before_merged_4c1b623.nsys-rep` 与
`/tmp/p3b1_after_d8035e6.nsys-rep`。

### 验收结果

- 编译：`cmake --build build -j2 --target pystiffgipc` 通过。
- strict 位级：frame-graph 0/1、run-to-run、PCG device-loop 0/1 均为
  `f7fb5a786c2d7935`。
- strict quad-gate：frame-graph 0/1 均 5/5 PASS。
- MAS oracle：5 个场景全部 PASS；jaw：双侧 OPENED；dualarm：4 项 True。
- LS large-grow：CCD tier 连续增长、friction 二次增长、post-grow graph rebind PASS。
- frame transaction：legacy/graph A/graph B/三类 retry 均同 checkpoint hash，NaN
  rollback PASS；root D2H node 0、terminal D2H node 1。
- CCD NaN max-speed：device tail fail-fast PASS。
- compute-sanitizer（frame-graph on，strict N=3）：memcheck 0 errors；racecheck
  0 hazards、0 errors、0 warnings。
