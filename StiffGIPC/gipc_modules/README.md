# gipc_modules — GIPC.cu 复合编译单元的语义模块

这些 `.inl` 由 `GIPC.cu` **按序 #include** 进同一个编译单元。最初 Phase 1
切片有字节级拼接证明（原文件 sha256 见 `ORIGINAL_SHA256.txt`）；此后已经发生
经构建、strict 锚和门禁验证的职责迁移，因此不能再把当前树表述为原文件的
字节级拼接。

规则：
- **include 顺序承载语义，禁止重排**；跨模块的 file-scope 依赖（`g_*` device
  全局、static、模板）在尚未物理分离的模块中暂时合法。
- 新代码写进语义所属的模块文件，不写回 GIPC.cu。
- `00_prelude_common.inl` 只保留公共头、TU 全局和有序 include；不要再把 kernel
  塞回 prelude。

| 模块 | 内容 | 未来归属（见 docs/V086_REFACTOR_PLAN.md） |
|---|---|---|
| 00_prelude_common | 公共 includes、日志级、下列 prelude 子模块的有序装配 | composite prelude |
| 01_contact_energy_device | barrier/friction 能量 device 函数、_selfConstraintVal、injective step、_pair_mu | contact |
| 02_friction_assembly | _calFrictionHessian(_gd)（含 9a06408 rank 语境） | contact/barrier_assembly |
| 03_barrier_assembly_split | legacy split-GH `_calBarrierHessian`（SymGH 下已强制回退）、moveMemory | contact（候选死码清理对象） |
| energy/03_barrier_fused_assembly | binned 梯度机制（g_gbin/_gfxAdd）、融合 `_calBarrierGradientAndHessian`（I1==0 零填充在此） | contact/barrier_assembly |
| 05_close_gradients | close-set kernel（死镜像语境，冻结）、_reduct_MSelfDist、friction/barrier gradient、_ec_emit | contact |
| 06_kinetic_soft_ground | kinetic、软约束、**铁律 kernel 群**（scan/zero/probe/mark）、地面检测/梯度/close、ground trial | multienv/isolation + contact/ground |
| 07_energy_alpha_reductions | 全部 _get*Energy_Reduction + alpha/cfl/injective 归约 + __add_reduction（c3087a7 模板家族） | device_common/reductions |
| 08_step_update_topology | _stepForward 家族（0×NaN 守卫）、速度/边界/xTilta、排序更新 kernel、相交查询 | core + engine |
| 09_friction_sets_host_mem | 摩擦 lastH kernel、FREE/MALLOC_DEVICE_MEM、init、buildFrictionSets、地面/自 CCD 宿主包装 | contact/pair_buffers |
| 10_ccd_buildcp_quarantine | CCD alpha 组合 kernel、InjectiveStepSize、**buildCP（grow-redo）**、快照、**quarantine 全家**、throw 族、ccd-nan 自测 | contact/pair_buffers + multienv/isolation |
| 11_perenv_machinery | per-env kernel 全家（kappa/alpha/mask/ground/self min）、per-env BVH 构建、pool | multienv |
| 12_host_wrappers_fem | per-env build、装配宿主包装、FEM 单元 G/H 计算、step_forward 宿主、排序/更新宿主、minMovement | core + abd/fem |
| 13_kappa_partition_gradhess | suggest/upperBound/initKappa、partitionContactHessian、**computeGradientAndHessian**（帧首扩容在此） | core/frame_pipeline |
| 14_energy_linesearch_solver | 能量宿主聚合（含 DeviceOut/perenv）、_global_ls_decide（NaN 守卫）、close 临时缓冲/摩擦缓冲 malloc、updateVelocities/xTilta、checkpoint。**2d 起编排已迁出**：lineSearch/postLineSearch/solve_subIP/IPC_Solver → `core/ipc_solver.cu`（**独立 TU**，2d step 2） | core/ipc_solver（已完成） |

`00` 的子模块仍按原声明顺序进入同一 TU：

| 文件 | 职责 |
|---|---|
| `device_common/spatial_hash.inl` | Morton/spatial hash helper |
| `linear_system/triplet_partition_kernels.inl` | collision triplet 分区与重排 |
| `device_common/topology_sort_kernels.inl` | topology 映射与 Morton sort kernel |
| `device_common/reduction_kernels.inl` | 通用 max/min reduction kernel |

未被任何调用点引用的旧通用 `makePD` 模板已移除；实际本构使用各自明确的
`makePDSNK`/`makePDGeneral` 实现。融合 barrier kernel 的物理 TU 抽离则已做过
同参数实验并被性能指标否决，详见 `energy/03_barrier_fused_assembly.inl`。
