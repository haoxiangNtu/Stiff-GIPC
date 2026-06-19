# UMI finray 软爪三模式 —— 改动说明 + 如何判断对错

这份说明对应 `umi_finray_lib.py` + 9 个入口文件。目的:让你能**逐项判断每个改动是否必要**,以及**怎么在 GUI 里看效果**。

---

## 三种夹爪模式(GRIP_MODE)

| 模式 | 一句话 | 机制 |
|---|---|---|
| `pos` | 纯位置控制 | 把 grip 指令映射成开度,关节直接驱到那个位置 |
| `stitch` | 形变门控位控 | 边合拢边看缝线形变,夹住(形变够大)就停;松了就重夹 |
| `force` | 力驱 → 锁位 | 恒力合拢,夹住后锁成位置(刚体);两端硬 barrier 防过冲/飞出 |

布料场景 `pos` 最干净;刚体抓取 `force` 最稳;`stitch` 是中间档。

---

## 这次所有改动,逐项「必要 / 冗余」

### 通用
| 改动 | 在干嘛 | 必要? |
|---|---|---|
| **连续 grip 映射** | grip∈[-1,1] 线性映射到开度(原来是 ≥0 全开 / <0 全闭的二值突跳) | **必要**。foldshirt/beaker 轨迹是连续值,二值会突跳。 |
| **cupshirt 机械臂底座 y=-3.0** | qpos_case39 场景手臂要这么放才够得到杯子 | **必要**。否则夹爪在杯子上方 3m,根本够不到。 |
| **friction 默认 1.0** | 夹爪↔物体摩擦系数(原 0.4/0.8) | **必要**(防滑);可调。 |
| **地面可视化 `none`** | polyscope 不画错位地面 | 纯显示,无关物理。 |

### stitch 模式
| 旋钮 | 在干嘛 | 必要? |
|---|---|---|
| `GRIP_STITCH_THRESH=2e-5` | 「夹住」的形变阈值。软布形变低于它→闭到底;刚体高于它→可锁 | **必要** |
| `GRIP_STITCH_MIN_S=0.3` | 必须先闭到 70% 才允许锁(否则刚体在半开就误锁、滑落) | **必要**(真正的修复) |
| `GRIP_STITCH_RESUME_FRAC=0.5` | 夹住后形变骤降(物体掉了)→ 解锁重新夹(v0.6.4 滞回) | **必要** |
| `GRIP_STITCH_DEBOUNCE=1` | 连续 N 帧超阈值才锁 | **冗余**(默认=1=关)。我找到 min_s 之前瞎试加的,A/B 实测有 min_s 后它没用。 |
| `GRIP_CLOSE_DS=0.03` | 合拢速度(原 0.02 慢) | **冗余**。同上,减速只是更柔,跟正确性无关,已恢复 0.03。 |

> **stitch 真正起作用的只有 3 件:`THRESH + MIN_S + RESUME`。** debounce / march 减速是我在定位到 min_s 之前的试错残留,现已默认关闭(留旋钮以防万一)。

### force 模式
| 改动 | 在干嘛 | 必要? |
|---|---|---|
| 闭合端 barrier (slot0) | 力驱时不让越过全闭位(防过冲) | **必要** |
| **张开端 barrier (slot1, 引擎改)** | 力驱时不让被布顶出全开位(防飞出) | **必要**(否则软布上自由指飞出 hand) |
| **force-lock(夹住即锁位)** | 夹住后从力驱切位置锁,移动时不打滑 | **必要**(你提的「跟不上机械臂」) |
| `GRIP_FORCE_LOCK_MAX_S=0.7` | 必须闭过 70% 才允许锁(防力 ramp 时锁在全开) | **必要**(防误锁) |
| 场景自适应锁(刚体锁/布纯力驱) | 布没有干净卡住点,锁会冻在半开 | **必要** |
| `force_strength` 软弹簧 | 旧版防飞出用的软位置 home | **已废弃**(默认 0)。被「张开端硬 barrier」取代,留作可选。 |

> **force 真正必要的:两端硬 barrier + 夹住锁位(场景自适应)。** 软弹簧 `force_strength` 是中间步骤,已被硬 barrier 取代(默认 0)。

---

## 怎么在 GUI 里看 + 判断对错

前缀:`cd /home/ps/Downloads/Stiff-GIPC-06me && conda activate test_stiff_v061`

### 1. 看开合是否跟着指令(pos,连续映射)
```bash
PYTHONPATH=. GRIP_MODE=pos python examples/replay_foldshirt_finray.py
```
**看**:点 Run,夹爪应随轨迹**平滑**开合(不是突然全开/全闭)。

### 2. 看力驱「夹住锁位、随臂移动不掉」(你的核心问题)
```bash
PYTHONPATH=. GRIP_MODE=force python examples/replay_cupshirt_finray.py
```
**看**:杯子掉到地面 → 手臂下去 → 夹爪夹住 → 抬起,**杯子被稳稳夹住跟着升起不滑落**。

### 3. 看「张开端 barrier」真的在防飞出(对照实验)
```bash
# 有 barrier(默认):夹爪指头不飞出 hand
PYTHONPATH=. GRIP_MODE=force python examples/replay_foldshirt_finray.py
# 关掉 barrier:应能看到夹爪一根指头被布顶飞出 hand 之外
PYTHONPATH=. GRIP_MODE=force GRIP_BARRIER_KAPPA=0 GRIP_FORCE_LOCK=0 python examples/replay_foldshirt_finray.py
```
**看**:第二条里夹爪指头会"飞"出张开极限(穿出手掌范围);第一条不会。这就证明 barrier 在起作用。

### 4. headless 数字诊断(辅助,看不见时用)
```bash
PYTHONPATH=. SC=cupshirt GRIP_MODE=force CASE39ME_NUM_ENVS=2 python examples/diag_finray_grip.py
```
**字段含义**:
- `dmin` = 左爪最终开度(越接近 0 越闭合);
- `obj_rose` = 物体相对最低点升高了多少(刚体 >0.02 = 夹起来了);
- `GRIP+LIFT / CLOSED / slip-no-lift` = 判定;
- `*** FLY-OUT ***` = 指头越过关节极限(出问题)。

---

## 如果你想要最小改动集

debounce / march / force_strength 这三个是冗余/已废弃的(都默认关)。要的话我可以**直接删掉这些代码**,只留必要的(连续映射 + 底座修复 + 摩擦 + stitch 的 THRESH/MIN_S/RESUME + force 的两端 barrier + 场景自适应锁位)。说一声即可。
