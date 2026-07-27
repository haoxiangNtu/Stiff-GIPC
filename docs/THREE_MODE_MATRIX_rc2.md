# v0.8.6-rc2 三模式全矩阵（headless，2026-07-28）

指令：全部 demo 与轨迹重放（含 ModelScope towel 与夹盘子）× merged/isolated/strict。

## 本地 19-demo × 3 模式（RTX 4090 D, sm_89）——56/56 全绿

| demo | merged | isolated | strict |
|---|---|---|---|
| anchor | 0.8s | 0.5s | 0.5s |
| towel（ModelScope 布巾重摆配方） | 6.8s | 5.9s | 8.1s |
| foldshirt | 40.6s | 47.8s | 61.6s |
| finray_beaker / _1env | 9.4 / 8.0s | 12.5 / 8.8s | 20.0 / 13.6s |
| finray_cupshirt / _1env | 10.7 / 7.1s | 13.3 / 7.2s | 18.0 / 12.3s |
| finray_foldshirt / _1env | 26.6 / 15.0s | 31.9 / 17.9s | 42.6 / 28.5s |
| midrun_quarantine | （契约限制） | 0.5s | 0.6s |
| abd_badmesh | 0.4s | 0.4s | 0.4s |
| bench_case26 | 4.1s | 3.8s | 6.5s |
| case39 / _multienv | 6.4 / 7.9s | 6.0 / 10.6s | 10.3 / 19.3s |
| umi_beaker | 10.0s | 10.5s | 14.5s |
| umi_cupshirt_fg | 7.6s | 7.3s | 12.8s |
| umi_sf / _obb | 14.1 / 17.6s | 41.6 / 18.6s | 56.0 / 33.1s |
| diag_finray_grip | 15.5s | 23.2s | 42.1s |

全部 PASS：rc=0、无 NaN/异常特征行、有标记者标记在场。midrun_quarantine 的
isolated 列为该门禁首次真实执行（此前默认双模式只跑 strict 列）。

## A800 盘子夹取（ModelScope，sm_80，228 帧驻场 episode）——3/3 全绿

| 模式 | 牛顿总数 | 中位 | 峰值 | 总耗时 |
|---|---|---|---|---|
| merged | 715 | 3.0 | 14@fr164 | 26.1s |
| isolated | 669 | 2.0 | 22@fr178 | 33.4s |
| strict | 550 | 2.0 | 15@fr202 | 45.0s |

三跑 EXIT=0、energy_tolerance_accepts=0。strict 峰值 15 / 45.0s 与上一战役
逐字复现；rc2 wheel 于 A800 就地构建（cp311，urdfdom target 修复后一次通过）。

跨架构口径：本地 sm_89 + A800 sm_80，同一 rc2 源（6b0e02e）。
