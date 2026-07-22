# stiff-physics v0.8.5 功能使用指南（增量）

v0.8.4 指南（`FEATURE_GUIDE_v0.8.4_zh.md`）全部内容仍然有效；本篇只写 v0.8.5 新增。

---

## 1. URDF primitive 碰撞体：box / sphere / cylinder 直接可用

v0.8.4 及以前：URDF 里 `<collision>` 用几何 primitive 的 link 会被**整个跳过**
（只有 WARNING），必须事先把 primitive 转成网格。v0.8.5 起自动转换：

```python
eng.load_urdf("robot_with_primitives.urdf", translation=(0, 0.5, 0))
# [UrdfSceneImporter] link 'base': generated conservative primitive proxy
#     (box+sphere, 170 verts, 332 tris) -> .../.stiffgipc_prim_cache/...obj
```

规则与保证：

- **保守外包**（proxy ⊇ 精确体，接触只会提早、不会漏穿）：
  box 精确 8 顶点；sphere 用 icosphere 细分 2 级、逐面外推（半径膨胀 ~2.4%）；
  cylinder 24 段外切棱柱（膨胀 ~0.9%）。
- 同一 link 的**多个** collision 元素（含各自 `<origin>`）合并成一个 ABD body。
- mesh 与 primitive 混用的 link：优先取第一个 mesh 元素（并对被忽略的
  primitive 打 WARNING）——旧版本在首元素是 primitive 时会把整个 link 的
  碰撞丢掉，v0.8.5 顺带修复。
- 生成的 .obj 缓存在 URDF 旁 `.stiffgipc_prim_cache/`（目录只读时落系统
  temp），可提交进资产库也可 gitignore。
- 质量仍按体积 × 密度计算（与 mesh 路径一致；URDF `<inertial>` 不参与）。

回归：`examples/test_urdf_primitives.py`。

## 2. 接触力分量导出：法向 / 滞后摩擦 / 合力

```python
F_n   = eng.get_vertex_contact_forces(components="normal")           # 默认
F_f   = eng.get_vertex_contact_forces(components="friction_lagged")  # 冻结 lastH 摩擦
F_tot = eng.get_vertex_contact_forces(components="total")            # 两者之和
```

- 单位牛顿（物理力 = −∇E/dt²，v0.8.4.1 已修符号：地面支持力 y 分量为正）。
- `friction_lagged` 读的是本帧求解使用的**滞后摩擦锚点**（lastH λ/切向基），
  只读、不重建摩擦对——与求解器看到的力严格同源。
- 静止物体摩擦分量为 0；`normal + friction_lagged == total` 逐顶点成立
  （回归断言 additivity_err = 0）。

## 3. von Mises 应力与求解器本构一致

`get_fem_von_mises()` 现在与求解器共用同一 P(F)（Stable-NHK1：
P = μF + r(J−1−μ/r)·cof F），σ = P·Fᵀ/J 取偏应力。退化单元（J≈0）返回
**NaN** 而非 0——做统计时用 `np.nanmax`/`np.nanmean`。

## 4. 关节强度推荐配置

prismatic 与 revolute 同 ratio 时 prismatic 明显偏软（约束平移 DOF 没有杠杆
臂）；`joint_strength_ratio=100` 在重负载下会缓慢下垂。三套已验证 profile
（软抓取 / 刚性压持 / 快速拖拽）与调参顺序见 `docs/JOINT_TUNING_v0.8.5_zh.md`。

## 5. strict 模式确定性：五门自检入库

```bash
python examples/test_strict_quadgate.py
# G1 run-to-run N=2 / G2 run-to-run N=8 / G3 cross-env / G4 batch N=2-vs-N=8
# / G5 MAS 漂移监控
```

已知限制（文档化于测试内）：`preconditioner_type=1`（MAS）的聚合结构随总
env 数变化，batch 不变性有 ~1e-9 漂移（run-to-run 与 cross-env 不受影响）；
需要跨 batch-size bit 级一致时用 `preconditioner_type=0`。
另注意 `set_env_offsets()` 必须在 `finalize()` **之后**调用。

## 6. 求解器健壮性（自动生效，无需配置）

- 线搜索/边界移动的相交回退循环有界（预算 = `line_search_max_iter`，默认
  64）；起点已穿插时抛出带诊断的异常（fail-fast），不再无限自旋。
- semi-implicit β 衰减改用上一迭代**已接受**的步长（`semi_implicit_enabled`
  默认仍为 False，行为不变；开启时时序正确）。

## 附：v0.8.5 变更速查

| 类别 | 内容 |
|---|---|
| 新能力 | URDF primitive proxy；接触力 components 参数 |
| 正确性 | vM 本构一致（退化→NaN）；β 时序；混合 collision link 修复 |
| 健壮性 | isIntersected 预算回退 ×3 处 |
| 测试 | quad-gate 五门、四杆闭环、gd_friction 直接断言、M0 哨兵 ×4 |
| 文档 | JOINT_TUNING（A3/B3 推荐配置） |
