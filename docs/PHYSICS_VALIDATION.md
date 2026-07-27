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
原子写、损坏/截断/旧格式的事务式拒绝，以及同计数但不同场景的签名拒绝。
