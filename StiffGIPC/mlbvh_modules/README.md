# mlbvh_modules — mlbvh.cu 复合编译单元的语义模块（v0.8.6 Phase 1b）

按序 #include 进同一 TU；拼接与拆分前文件字节级相等（sha256 见
ORIGINAL_SHA256.txt）。include 顺序禁止重排；新代码写进语义所属模块。
⚠️ 本文件族与 KemengHuang 上游同源——移植上游 patch 时先 `cat 00..06 > mlbvh.cu`
还原单文件、应用 patch、再重跑拆分脚本（字节等价性质保证往返无损）。
⚠️ v0.8.6 P5 起还原产物 ≠ Phase-1 快照：03 已删死码 `_checkPTintersection_fullCCD`、
05 已删其 2 处注释调用行。上游 patch 若命中这些位置附近的 hunk 需手动对齐。

| 模块 | 内容 |
|---|---|
| 00_gates_globals | 确定性/诊断 device 全局（g_ee_canon/detgate/nomollify/vloc…）+ setter |
| 01_caps_aabb_morton | _emit_slot/set_emit_caps（DCD≤CCD 检查点）、AABB merge/overlap、morton、排除表 |
| 02_distances_dtypes | _d_PP/PE/PT/EE、eps_x、dtype 分类 |
| 03_pair_emission | _checkPTintersection/_checkEEintersection（int4 编码单一产地；smooth 死分支冻结） |
| 04_lbvh_build | _reduct_max_box（lens-B 修复在此）、叶盒、内部节点、MC hash |
| 05_traversal_queries | 六个遍历栈 query kernel（vf/ee × dcd/ccd/lb 变体） |
| 06_host_wrappers | calcMaxBV、Construct/SelfCollitionDetect 等宿主管线 |
