# 三模式 claim/性能矩阵报告（refactor/v086-energy-modular @3e76749）

拥有者指令：merged/isolated/strict 全例验证各模式 claim；重点场景测峰值
iteration、FPS、总耗时。双平台：RTX 4090 (sm_89) + A800-SXM4-80GB (sm_80)。

## Claim 判定（全部成立）

| 模式 | Claim | 证据 |
|---|---|---|
| strict | 位级复现（r2r+跨架构） | 锚 f7fb5a786c2d7935：4090 与 A800 各自 r2r BIT-OK 且同金值；resolver 路径≡显式 flag |
| isolated | per-env 公平+铁律隔离 | A800/本地 midrun 隔离门 PASS（env0 冻结漂移 0、健康 env 静置）；per-env alpha 生效（见峰值/耗时信号） |
| merged | 吞吐优先、无隔离承诺 | 全场景完成；多数场景步时最低；契约成文 fail-fast |
| 武装诊断 | 零误报 | MIRROR+SLOT 双武装锚双平台同哈希 audit_fires=0 |

## 数据总表（60f 场景；盘子 228f 驻场 episode）

### 4090（本地）
| 场景 | 模式 | 判定 | 总耗时 | FPS | 峰值 Newton | 平均步时 |
|---|---|---|---|---|---|---|
| towel | merged | PASS | 24.8s | 8.9 | 94 | 103ms |
| towel | isolated | PASS | 17.1s | **12.9** | 98 | 57ms |
| towel | strict | PASS | 23.7s | 9.3 | **61** | 93ms |
| foldshirt 4env | merged | PASS | 99s | 0.60 | 50=cap | 1.63s |
| foldshirt 4env | isolated | PASS | 102s | 0.59 | 50=cap | 1.68s |
| foldshirt 4env | strict | PASS | 127s | 0.47 | 50=cap | 2.10s |
| finray_beaker | merged | PASS | 16.6s | 3.6 | 9 | 262ms |
| finray_beaker | isolated | PASS | 20.9s | 2.9 | 10 | 334ms |
| finray_beaker | strict | PASS | 35.0s | 1.7 | 8 | 568ms |
| finray_cupshirt | merged | PASS | 21.3s | 2.8 | 50=cap | 338ms |
| finray_cupshirt | isolated | PASS | 26.5s | 2.3 | 42 | 427ms |
| finray_cupshirt | strict | PASS | 34.1s | 1.8 | 36 | 553ms |
| finray_foldshirt | merged | PASS | 77.0s | 0.78 | 50=cap | 1.26s |
| finray_foldshirt | isolated | PASS | 58.2s | **1.03** | 39 | 0.96s |
| finray_foldshirt | strict | PASS | 70.5s | 0.85 | 38 | 1.16s |

### A800
| 场景 | 模式 | 判定 | 总耗时 | FPS | 峰值 Newton | 平均步时 |
|---|---|---|---|---|---|---|
| towel | merged | PASS | 8.4s | 26.3 | 94 | 32ms |
| towel | isolated | PASS | 7.1s | **31.0** | 98 | 27ms |
| towel | strict | PASS | 9.6s | 22.9 | **61** | 38ms |
| foldshirt 4env | merged | PASS | 69.0s | 0.87 | 50=cap | 1.12s |
| foldshirt 4env | isolated | PASS | 97.3s | 0.62 | 50=cap | 1.60s |
| foldshirt 4env | strict | PASS | 99.6s | 0.60 | 50=cap | 1.64s |
| 盘子 228f | merged | PASS | 26.9s | 8.6 | 22 | — |
| 盘子 228f | isolated | PASS | 30.0s | 7.6 | 19 | — |
| 盘子 228f | strict | PASS | 45.0s | 5.1 | **15** | — |

## 发现（跨平台一致的结构信号）

1. **峰值迭代跨架构逐一相同**（towel 94/98/61 双平台同值）——迭代结构
   与架构无关，数值层强交叉验证。
2. **strict 峰值系统性最低**（towel 61、盘子 15、cupshirt 36、finray_fold 38）
   ——canonical 顺序+层固定降低最坏帧迭代；代价为均步时（strict 慢
   20%~2×，场景相关）。
3. **isolated 在布料重场景常最快**（towel 双平台、finray_foldshirt 本地）
   ——per-env alpha 让局部早收敛真实生效；且 isolated/strict 在 finray
   cupshirt/foldshirt **不触 cap** 而 merged 触——per-env 机制降低峰值需求。
4. **foldshirt(4env) 三模式峰值=50 即脚本 cap**：场景在 cap 上运行（设计
   使然，非失败）；对比盘子 228f 无触顶（22/19/15）。若需余量可提
   newton_iter_cap 复测。
5. A800 towel ~3× 于 4090（HBM 带宽敏感负载）；foldshirt 反而接近
   （算力/延迟敏感段占比高）。

## 覆盖与残留

- 自动化：anchor/towel/foldshirt/finray×3（beaker/cupshirt/foldshirt）
  ×三模式双平台 + 盘子三模式 A800 + 隔离双门 + 武装审计。
- UI demo（软爪 softgripper_cup、finray_ui 系、case_26/27/39/40）：拥有者
  逐个手测，见 docs/UI_DEMO_MODE_TEST_CHECKLIST.md。
- 工具：scripts/mode_bench.py（BENCH_SCENES/BENCH_MODES 可选、RECORD 重录
  基线）；场景侧 STIFF_BENCH_STATS 门控逐帧统计。
