# mas_modules — MASPreconditioner.cu 复合编译单元的语义模块（v0.8.6 Phase 1b）

按序 #include 进同一 TU；拼接与拆分前文件字节级相等（sha256 见
ORIGINAL_SHA256.txt）。include 顺序禁止重排。

| 模块 | 内容 |
|---|---|
| 00_binned_accum | binned 确定性累加 device 全局 + comb kernel |
| 01_aggregation_kernels | L0/L1/Lx 聚合、prefix-sum、cluster 生长（72da264 修复族） |
| 02_inverse_restrict_collect | __inverse6_P96x96（lens-B inRange 修复在此）、multiLevelR、collectZ |
| 03_schwarz_apply | Schwarz apply（Sym3/6/9 + fused8 opt-in） |
| 04_collision_connection | 碰撞连通 kernel |
| 05_envseg_host_pipeline | per-env 分段 kernel + 宿主管线（Reorder/PrepareHessian/预条件应用/init/Free、
|                        | ensureOutputClusterCapacity（lens-A 双容量修复）在此） |
