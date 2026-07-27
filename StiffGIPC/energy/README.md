# energy — 能量/本构层（v0.8.6 E1 完成形态）

目标（拥有者指令 2026-07-26）：**加一个本构 = 加一个文件 + 项表一行**。
蓝图：docs/ENERGY_SEPARATION_PLAN.md；项表单一真源：energy_terms.h。

| 文件 | 内容 | 槽位 |
|---|---|---|
| energy_terms.h | 12 型项表（type↔尺寸源↔kernel↔G/H 属主） | 文档 |
| 02_contact_energy_device.inl | barrier/friction 能量 device 函数 + _pair_mu（冻结 smooth 分支逐字在内） | 老 01 位 |
| 03_barrier_fused_assembly.inl | binned 梯度机制 + 融合 _calBarrierGradientAndHessian；受控物理抽离因 stack +41%、SASS +9.5% 被否决 | 老 04 位 |
| 10_kinetic.inl | E 归约 + G/E kernel | 老 07 位 |
| 11_fem_elastic.inl | FEM+RestNHK E 归约（元素 G/H 在上游 femEnergy.cuh，最小扰动） | 老 07 位 |
| 12_triangle_membrane.inl | 膜 E 归约 | 老 07 位 |
| 13_bending.inl | 弯曲 E 归约（quad #ifdef + angle） | 老 07 位 |
| 14_soft_constraints.inl | E 归约 + G/H | 老 07 位 |
| 15_barrier.inl | E 归约 + _calBarrierGradient | 老 07 位 |
| 16_friction.inl | E 归约 + Hessian（老 02）+ 梯度（老 05） | 老 07 位 |
| 17_ground.inl | E 归约 + G/H | 老 07 位 |
| 18_delta.inl | 线搜索方向项 E 归约 | 老 07 位 |
| 01_energy_host_dispatch.inl | type-switch 宿主分派 + 能量合并 + computeEnergy 族 | 14 前 |

规则：
- 改 01 的 switch 必须同 commit 改 energy_terms.h。
- 槽位（include 位置）承载依赖顺序：02 先于 05 残留的消费者、10..18 先于宿主
  分派与 13 的 G/H 启动点——**别搬 include 位置**。
- 冻结区（smooth/mollifier 分支、close 链）逐字随行仍冻结，G0.5 绊线在钉。
- E2 注册表 ✅、E3 物理 TU ✅（九项独立 .cu，锚六级不动；10..18 *.inl 现为各
  自 .cu 的正文，复合仅剩 01 宿主分派 + 02 共享能量头 + 03 融合装配例外）。
  新本构工作流：写 <term>.cu+.inl（核+两注册成员）→ GIPC_ENERGY_TERMS 加行。
- `03` 继续留在复合 TU 是有量化证据的性能决策，不是遗漏；指标与复验条件见
  `docs/ENERGY_SEPARATION_PLAN.md`。
