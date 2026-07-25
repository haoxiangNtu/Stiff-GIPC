# gipc_modules — GIPC.cu 复合编译单元的语义模块（v0.8.6 Phase 1）

这些 `.inl` 由 `GIPC.cu` **按序 #include** 进同一个编译单元。拆分脚本断言
"00..14 按序拼接 == 拆分前 GIPC.cu 字节级相等"（原文件 sha256 见
`ORIGINAL_SHA256.txt`），因此编译产物与 strict 位级锚不可能改变。

规则（Phase 2 物理分离前）：
- **include 顺序承载语义，禁止重排**；跨模块的 file-scope 依赖（`g_*` device
  全局、static、模板）暂时合法——Phase 2 逐模块解开。
- 新代码写进语义所属的模块文件，不写回 GIPC.cu。
- 文件内容当前为原样切片（注释块可能跨切点归属前一片）；重命名/整理随
  Phase 2 逐模块进行。

| 模块 | 内容 | 未来归属（见 docs/V086_REFACTOR_PLAN.md） |
|---|---|---|
| 00_prelude_common | includes、日志级、makePD、write_triplet/zero、hash、triplet 分区/重排、拓扑排序 kernel、通用 max/min 归约 | device_common |
| 01_contact_energy_device | barrier/friction 能量 device 函数、_selfConstraintVal、injective step、_pair_mu | contact |
| 02_friction_assembly | _calFrictionHessian(_gd)（含 9a06408 rank 语境） | contact/barrier_assembly |
| 03_barrier_assembly_split | legacy split-GH `_calBarrierHessian`（SymGH 下已强制回退）、moveMemory | contact（候选死码清理对象） |
| 04_binned_grad_fused_assembly | binned 梯度机制（g_gbin/_gfxAdd）、融合 `_calBarrierGradientAndHessian`（I1==0 零填充在此） | contact/barrier_assembly |
| 05_close_gradients | close-set kernel（死镜像语境，冻结）、_reduct_MSelfDist、friction/barrier gradient、_ec_emit | contact |
| 06_kinetic_soft_ground | kinetic、软约束、**铁律 kernel 群**（scan/zero/probe/mark）、地面检测/梯度/close、ground trial | multienv/isolation + contact/ground |
| 07_energy_alpha_reductions | 全部 _get*Energy_Reduction + alpha/cfl/injective 归约 + __add_reduction（c3087a7 模板家族） | device_common/reductions |
| 08_step_update_topology | _stepForward 家族（0×NaN 守卫）、速度/边界/xTilta、排序更新 kernel、相交查询 | core + engine |
| 09_friction_sets_host_mem | 摩擦 lastH kernel、FREE/MALLOC_DEVICE_MEM、init、buildFrictionSets、地面/自 CCD 宿主包装 | contact/pair_buffers |
| 10_ccd_buildcp_quarantine | CCD alpha 组合 kernel、InjectiveStepSize、**buildCP（grow-redo）**、快照、**quarantine 全家**、throw 族、ccd-nan 自测 | contact/pair_buffers + multienv/isolation |
| 11_perenv_machinery | per-env kernel 全家（kappa/alpha/mask/ground/self min）、per-env BVH 构建、pool | multienv |
| 12_host_wrappers_fem | per-env build、装配宿主包装、FEM 单元 G/H 计算、step_forward 宿主、排序/更新宿主、minMovement | core + abd/fem |
| 13_kappa_partition_gradhess | suggest/upperBound/initKappa、partitionContactHessian、**computeGradientAndHessian**（帧首扩容在此） | core/frame_pipeline |
| 14_energy_linesearch_solver | 能量宿主聚合（含 DeviceOut/perenv）、_global_ls_decide（NaN 守卫）、close 临时缓冲/摩擦缓冲 malloc、updateVelocities/xTilta、checkpoint。**2d 起编排已迁出**：lineSearch/postLineSearch/solve_subIP/IPC_Solver → `core/ipc_solver.inl`（复合 TU 最后一个 include） | core/ipc_solver（已完成） |
