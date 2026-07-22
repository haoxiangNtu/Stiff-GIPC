# 关节强度调参指南（A3/B3 推荐配置） — v0.8.4.2

针对审计条目 **A3**（prismatic/revolute 强度不对称）与 **B3**（`joint_strength_ratio=100` 偏软）。
两者都**不是缺陷修复**：默认值保持历史兼容，本文给出经过回放/哨兵验证的推荐组合。

## 参数一览

| Config 参数 | 默认 | 作用对象 |
|---|---|---|
| `joint_strength_ratio` | 100 | 固定/球铰 4 点约束的保持刚度 |
| `revolute_driving_strength_ratio` | 100 | revolute 驱动（位置伺服）刚度 |
| `prismatic_strength_ratio` | 100 | prismatic 约束轴向保持刚度 |
| `prismatic_driving_strength_ratio` | 100 | prismatic 驱动刚度 |
| `max_prismatic_step_per_frame` | ∞ | 每帧驱动目标步长限幅（m） |

所有 ratio 均为相对质量尺度的倍率：同一数值在不同密度/尺寸场景下行为一致。

## A3：为什么 prismatic 需要比 revolute 大得多

revolute 的 4 点约束天然有杠杆臂（力偶），而 prismatic 约束的是平移
DOF，接触反力直接全额顶在约束上。**同为 100 时 prismatic 明显偏软**
（表现为：夹爪压持时手指被顶开、驱动目标滞后 5–20 mm）。
经验换算：**prismatic 需要 revolute 的 20–40 倍** ratio 才有相近的跟踪硬度。

## B3：`joint_strength_ratio=100` 的症状与升档

100 的历史默认在重负载（被抓物 > 关节子链质量的 ~1/10）时可见：

- 关节处缓慢下垂（每帧 μm 级、累计 mm 级）；
- 高速轨迹回放时子链滞后于父链。

升到 500–2000 可消除。**不要盲目加大**：ratio 直接进 Hessian 对角，
过大（>1e4）会推高条件数 → PCG 迭代上升 → 帧时间变差
（与 log-barrier 病态同理，见 v0.8.4 限位设计讨论）。

## 推荐 profile（均为已验证组合）

### 1. 软体/布料交互抓取（foldshirt、毛巾类）
```python
Config(joint_strength_ratio=100.0,
       revolute_driving_strength_ratio=100.0,
       prismatic_strength_ratio=2000.0)
```
验证：`examples/replay_foldshirt_multienv.py`（1551 帧全轨迹，三模式）。

### 2. 刚性压持/精确直线驱动（按压、插入）
```python
Config(prismatic_strength_ratio=2000.0,
       prismatic_driving_strength_ratio=100.0,
       max_prismatic_step_per_frame=0.002)   # 2 mm/帧，防止一帧穿透式目标跳变
```
验证：`examples/test_sentinel_d2_press.py`（30 cm 压入软块无塌陷）。

### 3. 快速拖拽/大行程指尖（高速抓放）
```python
Config(prismatic_strength_ratio=4000.0,
       prismatic_driving_strength_ratio=200.0,
       max_prismatic_step_per_frame=0.004)
```
验证：`examples/test_sentinel_d7_ghostdrag.py`（0.35 m/s 拖拽 + 快速离手）。

### 调参顺序

1. 先用 profile 起步，只在观察到**具体症状**时改对应参数；
2. 跟踪滞后 → 加 `*_driving_strength_ratio`（×2 步进）；
3. 约束被顶开 → 加 `*_strength_ratio`（×2 步进）；
4. 迭代数上升（`STIFF_ITER_LOG=1` 观测）→ 说明加过头了，退一档；
5. 目标跳变引起穿透/爆迭代 → 收紧 `max_prismatic_step_per_frame`
   （推荐 ≤ 接触厚度 `absolute_dhat` 的 2–4 倍）。
