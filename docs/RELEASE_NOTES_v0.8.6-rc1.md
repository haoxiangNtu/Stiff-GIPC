# Stiff-GIPC v0.8.6-rc1（内部测试版）

**性质**：内部测试用 release candidate，非对外发布。分支 `internal/v0.8.6-rc1`，
tag `v0.8.6-rc1-internal`，仅推私有仓（对外镜像不动）。

## 主线：v0.8.6 模块化重构（27.6k 行单体 → 语义模块 + 物理 TU）

- **Phase 1/1b**：四单体 #include 复合拆分（字节等价断言），gipc/mlbvh/mas/engine
  全部语义模块化
- **Phase 2a–2d**：归约模板统一（22/25 kernel）、pair-buffer 增长机制单属主
  （顺手抓获 per-env CCD 镜像缩水潜伏 OOB）、隔离铁律机制单属主+契约首次成文、
  帧编排单属主 + `core/frame_pipeline.h` 帧序唯一权威
- **物理 TU 分离**：core/ipc_solver、contact/pair_buffers、multienv/isolation +
  **能量层 E1–E3 全竟**：九项能量各自独立 .cu（kinetic/fem_elastic/membrane/
  bending/soft/barrier/friction/ground/delta），"加本构 = 加文件 + 注册表一行"
  （`GIPC_ENERGY_TERMS` X-macro 单源）。融合装配（energy/03）留复合 TU=成文例外
- **锚不动纪录**：全部语义/物理阶段 strict 锚 `f7fb5a786c2d7935` 逐位不变，
  含跨架构（4090 sm_89 ≡ A800 sm_80）

## 结构性疫苗（P3 四针 + 扩面）

- `contact/encoding.h` int4 编码契约真源；`mode_contract.h` 三模式承诺表；
  `frame_pipeline.h`；隔离契约
- `HostMirror`（镜像保鲜审计）+ `DeviceBuffer`（RAII）+ 槽位契约审计 +
  discard 合法性单发窗口——**全套武装常驻门禁**
- 门禁体系 12 段：G0 真实构建 + G0.5 冻结绊线 + 八门 + G9 三模式门 +
  G10 坏网格 ABD 门

## 关键修复

- **towel-strict 崩溃根因**（pre-solve discard 毁活跃矩阵）→ preserve/discard
  契约 + 帧首合法窗口
- **case-26 布不掉/能量 NaN**（本 rc 新修）：坏网格 ABD 体经 PSD 钳制后质量
  矩阵奇异 → `M⁻¹` 产 NaN q̃；修复=特征值正 ε 底板。根子早于重构、由 NaN
  拒绝修复（fa0cd63）暴露。永久防线：per-body 哨兵 + 15 槽能量 dump + G10
- 审计批修复（NaN 线搜索判定、ExclusiveSum 时序等六项）+ 五镜头七缺陷批

## 三模式验证（本 rc 的验收依据）

双平台（4090+A800）33 格矩阵全绿：strict 位级（r2r+跨架构+双配置路径互证）、
isolated 铁律隔离（漂移 0）、merged 吞吐（契约成文 fail-fast）。结构发现：
峰值迭代跨架构逐一相同；strict 峰值系统性最低；isolated 布料场景常最快且不触
cap。详表：`docs/MODE_MATRIX_REPORT_2026-07-27.md`。
自动化：`scripts/demo_verify.py` **19 个 headless demo**（含 replay 轨迹家族九项：case39 双形态、UMI 四变体、finray 单/多 env、diag）全绿；UI demo 拥有者 13 连逐个目验通过（软爪六件套、整机/全尺度/统一场景、finray UI 三连；含 case-26 修复复验）。

## 已知事项 / 冻结区

- smooth/mollifier 分支与 close-set 链按指令冻结（G0.5 绊线在钉）
- 融合 barrier 装配仍复合 TU（后续战役）
- foldshirt 类场景峰值=newton_iter_cap(50) 系脚本设定（盘子 228f 无触顶）
- UI 软爪 demo 的 headless 移植未做（由 finray replay 家族承担自动化）

## 部署

wheel：cp311（sm_80/89/120；包声明 requires-python>=3.11，cp310 本地开发直接用 build/ 树不经 wheel）。A800 bundle 配方与坑
（npz 双件、盘子 episode 走 /data + STIFF_REPLAY_PATH、cwd=/data/stiff-physics）
见记忆/报告。

---

# 附录：rc1 后加固合并（6c562bc + 补充，未随 rc1 tag 发布）

外部审核分支 audit/codex-fable-review-20260727 经三路对抗审计后合入。要点与
**调用方可见的行为变更**：

## 行为变更（升级注意）
- 性能计数 getter（total_newton_iters/pcg_iters/collision_pairs/max_collision_pairs/
  frames_done）从进程级累计改为 **per-Engine 实例**；依赖跨 reset 累计值的调用方
  读数会变。
- `get_fem_von_mises_stress` 按编译期本构（SNK1/SNK2/ARAP）计算，不再硬编码
  Neo-Hookean；两个接触力/导出 getter 改为**输入序**返回（与
  `get_vertex_positions()` 对齐）。
- `STIFF_*` 旗标解析统一为**值感知**（`=0`/空串=关）；标准 resolver 只写 "1"
  或删除变量，不受影响；手写 `=0` 当"开"用的环境会翻转。
- 单进程单 Engine 强制（RuntimeOwnerLease）：先前静默共享可变缓冲的多 Engine
  用法现在显式抛 LifecycleError；reset() 后可重建。
- 侵入式 FD API 移出生产 ABI（diagnostics 构建才有）。checkpoint v1（STKP）移除，
  由带 CRC64/事务校验的 v2（STIFFCP2）替代。

## 修复定性更正（相对该分支自述）
- g_vloc：基线树中为 **泄漏 + 陈旧全局设备指针**（仅 STIFF_EE_CANON 路径可触发，
  非默认公开 API 路径）；字面 use-after-free 是该提交自身补加的 `release(m_d_vloc)`
  才会引入的，修复（buildCP 顶部无条件重发布）与该 free 正确同船。合入后闭环。
- “三模型 FD 验证”原实为三模型不变量验证 + SNK1 FD；本合并补 `FD_MATRIX=1`
  重炮腿后，三模型解析梯度均经 FD 对照（见 PHYSICS_VALIDATION.md）。

## 工程裁决存档
- 融合 barrier 核 TU 抽取被性能带否决（双盲复现：+9.5% SASS、栈帧 +8KB、
  460B/线程溢出、--maxrregcount=200 恶化至 1072B）；裁决注释入
  energy/03_barrier_fused_assembly.inl 头部。
- 性能基准协议：median-of-3，首跑受时钟爬坡/冷缓存污染约 10%，禁止单跑定带；
  GPU 非空闲时 mode_bench rc=2 fail-closed（tag 门禁 INFRA-BLOCK）。

---

# v0.8.6-rc2-internal（内测二版）

rc1 后共 11 提交（6c562bc..HEAD），全部带 15 段门禁绿 + 锚 f7fb5a786c2d7935
不动实据（gate-artifacts/）。

## 修复
- **checkpoint 续跑 5e-6 偏差破案**：帧入口碰撞对集合（跨帧携带不入档）；
  load 后重建 → 实测 5e-17；门禁全例收紧（strict 逐位/其余 1e-12）。
- **ABD teleport 顶点失同步**：只写 q 不映射 x（该 API 此前零使用者）；现
  teleport 内 cal_x_from_q 全块同步（对未动 body 位级中性）。
- **跨引擎串态三族清除**（描述符 phase-0 全竟）：模式闩锁值追踪（strict↔
  merged 串模式风险）、静态设备 scratch 全部实例化（21 处：s_ebin、
  pe_all/g_sink、18 个求解器小 scratch + 审计 out）。

## 新能力（RL episode-reset 契约）
- 两个 teleport 尾部重建帧入口配对集（与 checkpoint 同契约）。
- **检疫复活**：teleport 触及检疫 env 自动清标志（自纠正——坏 reset 一帧内
  被重检疫）；矢量化训练不再永久丢 env。
- **step 健康 getter**：get_ls_exhausted_count / get_ls_nonfinite_count——
  merged WARN-继续契约下，训练循环的弃 episode 信号。
- merged 穿地 reset 在 teleport 调用点即抛 GeometryError。

## 门禁（12 段 → 15 段）
- G13 geometry：初始不可行性类型化拒绝负向测试（穿地/互穿/无误报）。
- G14 knob-registry：84+17+11 个 STIFF_* 旋钮单源一致性（静态）+ finalize
  未知旋钮绊线（STIFF_KNOB_STRICT=1 升抛错）。
- G15 rl-reset：传送连续性/坏重置健康信号/穿地类型化拒绝/检疫→复活全弧。

## 工程决策存档
- dlto 实验闭环：TU 边界中性=否（复合驻留裁决维持）；复合+dlto 运行时基准
  六格全快零回归（towel -10~12%、foldshirt merged -15.2%、isolated -3%、
  strict -0.7%），strict 确定性完好（新不动点 0544461bd82123ae 两跑逐位一
  致）。采纳=全引擎换锚（A800 重验+基线重录），建议 v0.9 周期执行，owner
  拍板。基准工件在 gate-artifacts/。
- **dlto 已采纳**（rc2 后首个变更，45d74f0）：设备链接时优化默认开启——TU 边
  界不再是优化屏障（怪兽核 254reg/23.8KB 栈→178reg/2KB、跨 TU ABI 调用
  320→76、运行时最高 -15%）。strict 锚移至新不动点 **0544461bd82123ae**
  （run-to-run 逐位稳定，15 段门禁全绿），三处金值与九格性能基线已随切换重
  录。构建代价：全量 +76%、增量每次 ~60s 串行 dlink。**A800 复验：sm_80 dlto
  锚=同值 0544461bd82123ae 两跑逐位稳定——跨架构位级一致在 dlto 下幸存，单一
  金值保住**（深因：strict 的规范化排序+binned 序无关沉积使结果与各架构的指
  令调度解耦）。
- GPU idle 门新增 BENCH_IDLE_ALLOWLIST（远程桌面类基础设施进程显式豁免，
  默认空 fail-closed，每次豁免入日志）——远程操作期间 tag push 不再死锁。
- 描述符 phase-1（三族 __constant__ 迁移）按蓝图留 v0.9：串行多引擎正确性
  已由 phase-0 闭合，迁移是架构整合，不在发布前触碰热核访存路径。
