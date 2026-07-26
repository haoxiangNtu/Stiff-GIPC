# 能量/本构分离蓝图（v0.8.6 深水区 TU 分离的目标形态）

**目标（用户指令 2026-07-26）**：各种本构（constitutive models）能轻易分离开；
未来添加/修改能量与本构像"加一个文件"，不动其他模块。构建工程学收益（增量
编译）是副产品，不是目标。

## 现状：能量项散在四处

能量项的真实清单（type 分派器 @ gipc_modules/14，2026-07-26 逐行核对）：

| type | 项 | 尺寸源 | 归约 kernel（07） | G/H 所在 |
|---|---|---|---|---|
| 0 | kinetic | fem_point | _getKineticEnergy_Reduction_3D | 06 |
| 1 | FEM 四面体弹性 | tet_count | _getFEMEnergy_Reduction_3D | 12+femEnergy.cuh |
| 2 | IPC barrier | h_cpNum[0] | _getBarrierEnergy_Reduction_3D | 01/04/05 |
| 3 | delta（线搜索方向项） | fem_point | _getDeltaEnergy_Reduction | — |
| 4 | 地面 barrier | h_gpNum | _computeGroundEnergy_Reduction | 06 |
| 5 | 摩擦（lagged） | h_cpNum_last[0] | _getFrictionEnergy_Reduction_3D | 02 |
| 6 | 地面摩擦 | h_gpNum_last | _getFrictionEnergy_gd_Reduction_3D | 02 |
| 7 | RestStableNHK | tet_count | _getRestStableNHKEnergy_Reduction_3D | femEnergy.cuh |
| 8 | 三角膜（布料面内） | triangleNum | _get_triangleFEMEnergy_Reduction_3D | 12 |
| 9 | 软约束 | softNum | _computeSoftConstraintEnergy_Reduction | 06 |
| 10 | 弯曲（Quad/角度两 kernel） | tri_edge_num | _getQuadBending/_getBendingEnergy | 12 |
| 11 | **残留槽**（sizing 有、switch 永不分派） | triangleNum | — | — |

本构库：femEnergy.cuh 同时住着 ARAP 与 SNK（USE_SNK1）两套；弯曲有
USE_QUADRATIC_BENDING 开关。ABD 仿射能在 abd_system/（已独立）。close-set
与 smooth/mollifier 相关分支冻结（绊线 G0.5 在钉）。

## 三阶段

**E1 — 语义切分（同 TU、零位级风险）**：把 01/02/04..07/12/14 中的能量/本构
代码按"项"重组为 `StiffGIPC/energy/NN_<term>.inl`（Phase-1 式字节等价拼接
断言），复合 TU 按序 include。同步写 `energy/energy_terms.h`（项表单一真源：
type↔槽位↔kernel↔属主）。冻结分支逐字随行、绊线锚同 commit 更新。

**E2 — 接口硬化**：分派器改为读项表驱动（槽位顺序逐字保持=位级中性，锚裁决）；
"加一个本构"的工作流成文：新文件实现 energy/gradient/hessian 三件套 + 项表
加一行。ARAP/SNK 等多本构从 #ifdef 择一走向注册表择一（此步先文档化，不改行为）。

**E3 — 物理 TU 分离 ✅ 完成（2026-07-27，六梯级）**：九项全部独立 .cu
（kinetic→ground+soft+delta→bending+triangle→fem_elastic→friction→barrier），
**锚六级全程逐位不动**（FMA 漂移理论未兑现——项核的融合决策核内自足）。
成文例外：**融合装配 energy/03 留复合 TU**（encode/decode 依赖树深，
物理迁移是独立未来战役，需单独性能/锚裁决）。
路上清账：FrictionUtils.cuh 34 函数 + GIPC_PDerivative.cuh 41 函数 inline 化
（头文件定义无 inline 的历史欠账，双 TU 时代由 nvlink/ld 逼出）；RANK 单源化
contact/barrier_rank.h；makePDGeneral/write_triplet/_ec_emit/_penv_energy_accum
/_gfxAdd 提升共享头。

## 明确不做

- merged fail-isolate（拥有者决定 2026-07-26：隔离不是 merged 的职责）。
- 激活任何冻结分支；type 11 残留槽在 E2 清理（单独小 commit，P5 式论证）。
