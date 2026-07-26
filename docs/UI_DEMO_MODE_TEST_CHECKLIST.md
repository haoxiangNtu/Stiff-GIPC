# UI Demo 三模式手测清单（refactor/v086-energy-modular 分支）

拥有者手测流程（2026-07-27）：非 headless 的 demo 直接跑、一个个测。
模式经 `STIFF_MULTIENV_MODE` 注入（python resolver 已验证：env 覆盖 config，
strict 经 resolver 路径与显式 flag 位级同值——G9/A800 双证）。

## 通用跑法

```bash
cd Stiff-GIPC-p0fix   # 分支 refactor/v086-energy-modular，build/ 已就绪
STIFF_MULTIENV_MODE=merged   STIFF_LOG_LEVEL=1 python3 examples/<demo>.py
STIFF_MULTIENV_MODE=isolated STIFF_LOG_LEVEL=1 python3 examples/<demo>.py
STIFF_MULTIENV_MODE=strict   STIFF_LOG_LEVEL=1 python3 examples/<demo>.py
```

- `STIFF_LOG_LEVEL=1`：引擎逐帧/一次性求解打印；退出时 GlobalTimer
  **TIMING BREAKDOWN**（总耗时口径，任何 demo 免费自带）。
- 可疑时加 `STIFF_MIRROR_AUDIT=1 STIFF_SLOT_AUDIT=1`（武装诊断，陈旧镜像/
  未写满槽当场炸并报名）。
- strict 复现性手验：同一 demo 同参数跑两次，观察轨迹应完全一致。

## 软爪家族（优先）

| demo | 场景 | 注意 |
|---|---|---|
| case_27_mobile_s1_softgripper_cup.py | 移动底盘+软爪抓杯 | 软爪本体 |
| case_umi_finray_ui.py | UMI finray 软指 | |
| case_umi_finray_ui_obb.py | finray + OBB 变体 | |
| case_umi_finray_force_ui.py | finray 力控 | |
| case_27_mobile_s1_hybrid.py / _clean | 混合刚软方案 | STRATEGY_F 资产 |
| case_27_mobile_s1_obb_except_gripper.py / _clean | OBB 例外爪 | |

## 布料/整机家族

| demo | 场景 |
|---|---|
| case_26_arm_cloth_semi_implicit.py（+perf_tuned/extreme） | 机械臂+布 |
| case_27_ridgeback_panda_cloth.py | 整机+布 |
| case_39_full_scale.py（/_semi） | foldshirt 全尺度 UI 版 |
| case_40_unified.py（/_semi） | 统一场景 |

## 单 env demo 的模式语义提示

多数 UI demo 是单 env（groups=1）：isolated 的 per-env 隔离机制自动不激活
（`perEnvIsolationLive` 要求 groups>1），此时 isolated≈merged+若干 env-flag；
**strict 仍完全有意义**（canonical 顺序 ⇒ 单 env 逐位可复现）。多 env 语义
验证以 headless 矩阵（G9 + mode_bench + A800）为准。

## 已自动化覆盖（无需手测）

anchor/towel/foldshirt × 三模式（G9 门 + mode_bench，4090+A800 双平台，
判定/总耗时/FPS/峰值 Newton 全记录）；盘子三模式走 A800 驻场 episode。
