# 物理模型、导数与 checkpoint 验证

## 本构模型

四面体 FEM 的模型在构建时集中选择：

```bash
cmake -S . -B build -DSTIFFGIPC_FEM_MODEL=SNK1
```

合法值为 `SNK1`（默认）、`SNK2`、`ARAP`；非法值在配置阶段失败。一个 native
module 只包含一种四面体本构，这不是逐 body 的运行时材料注册表。Python 可查询：

```python
import stiff_physics
print(stiff_physics.fem_model())
```

`scripts/model_validation.py` 在 merged 与 strict 子进程中检查静止态、刚体客观性、
匀速守恒、重力与密度解耦、一阶时间步收敛，以及逐 body 地面摩擦响应。它验证
关键不变量，不替代针对具体材料参数的实验标定。

## 导数诊断

侵入式 FD API 在生产构建中默认不存在：

```bash
cmake -S . -B build -DSTIFFGIPC_ENABLE_DIAGNOSTICS=OFF
```

门禁构建显式开启 diagnostics。`scripts/fd_gate.py` 对多组步长做 E→G 检查，并在
纯 FEM 场景做 G→H 对角检查；场景同时断言 FEM、布料、弯曲、地面、接触、摩擦和
软约束确实有活动项，避免“零贡献也通过”的假阳性。不同 multi-env mode 必须运行
在不同子进程，因为 native 热路径的模式是进程级配置。

**FD 覆盖按模型口径**：门禁构建的默认模型是 SNK1，因此常规 suite 的 FD 检查只
覆盖 SNK1 的解析导数；SNK2/ARAP 在生产矩阵里是 diagnostics-OFF（刻意断言 FD 符
号不在场）。重炮层（tag push）以 `FD_MATRIX=1` 运行 `model_build_matrix.sh`，为
三个模型各做一个 diagnostics-ON 构建并完整跑 `fd_gate.py`——三模型的解析梯度都
经 FD 对照，而不只是应力不变量。

## Checkpoint v2

`Engine.save_checkpoint(path)` 在同目录写临时文件、同步后原子 rename；
`Engine.load_checkpoint(path)` 只接受同一拓扑、材料签名、本构模型、执行模式和
环境布局。格式包含显式 magic/version/endian/scalar ABI、payload 长度与 CRC64。
校验、解码和有限值检查在修改 live GPU 状态之前完成，因此损坏、截断、旧格式或
场景不匹配不会留下半加载状态。

保存内容覆盖 FEM 的当前/前一帧位置、速度和 predictor，soft target，ABD 的
`q/q_prev/q_v/q_tilde` 与持久 external force，标量/逐组 Kappa、隔离
active/quarantine/NaN 状态、recheck 计数
与 quarantine 形成的逐 body ground-skip 表、每 Engine 帧号。调用点应位于
`step()` 之间的帧边界；外部控制器自己的状态和未来控制输入仍由应用保存。
应用须在 load 前重建当前控制配置，并从恢复帧继续提供相同的控制输入；checkpoint
不替外部控制器保存其目标轨迹或内部状态。
`scripts/checkpoint_gate.py` 在 merged/strict 下检查
载入瞬间位级恢复、下一步重启一致（strict 逐位；merged 因无序原子归约按严容差）、
原子写、损坏/截断/旧格式的事务式拒绝，以及同计数但不同场景的签名拒绝。此外有一
个**活跃接触场景**（堆叠立方体在持续摩擦接触中保存，保存帧断言碰撞对数非零）——
无接触场景的往返一致证明不了接触/摩擦相关状态的恢复，这个场景专门堵这个洞。摩擦
锚与 Kappa 不入档是设计而非遗漏：两者每帧从帧首构型确定性重建（`buildFrictionSets`
/`initKappa`），帧边界 checkpoint 恢复构型即恢复它们。

**第三类不入档状态：stepping 期间构建的顺序载体（未定名，已定界）**。接触场景
实测：merged 恢复续跑有**可复现** ~6e-6 位置 / ~6e-4 速度系统差；三路判别实验
（全新引擎整轨重放 ~6.6e-9 ≈ 散布地板；老引擎自重载 ~2.1e-9 = load 无副作用；
全新引擎恢复到边界 5.76e-6 = 1000× 地板）把它钉在"只在逐帧 stepping 中构建、
不在档案里"的顺序状态上；strict 免疫（EE/CCD 规范化排序 + binned 序无关沉积让
发射顺序不影响位），merged/isolated 按发射顺序原子求和所以可见。**不是 PCG 温
启动**——`STIFF_PCG_WARM` 默认关（显式 opt-in，两侧引擎都零启动）；若显式开启，
温启动向量会成为另一个不入档载体，属已知取舍。精确定位载体（BVH/配对缓冲布局/
预条件器结构之一）为跟进项。叠加原子噪声偶发翻转离散分支（配对集、线搜索减半，
~1e-4..1e-2 漂移），merged/isolated 的接触续跑因此只按硬顶断言（pos≤2e-3、
vel≤2e-2，仍低于"缺摩擦态"签名 μ·g·dt≈5e-2 一个量级）；完备性由 strict 接触
场景的**逐位**断言承担——它若失败即顺序载体（或未来开启的温启动）侵入 strict
路径的信号。
