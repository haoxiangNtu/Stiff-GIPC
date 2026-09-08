# 审计台账(AUDIT_LEDGER)——手册对抗校验中被抓出并已修正的事实性缺陷

## 0. 这份台账是什么

本手册(`docs/manual/` 八个分册)在成稿过程中走过一轮**对抗式全量校验**:
21 个校验代理拿着代码树逐句复核手册断言、只报"能被代码或 git 证伪的事实性错误",
共提出 **137 条 issue**;随后 8 个修订代理**逐条回到代码里复核**,确认成立的才改,
不成立的写明理由跳过。本台账固化其中**"校验代理发现 + 修订代理复核确认"**的那部分。

它**不是**待办清单——`OPEN_POINTS.md` 才是待核实项登记表(37 条 open/closed 条目)。
本台账记的是**已经改掉的错**:一份可供人类逐条复算的问责记录。

### 怎么用

每条给出七列:**编号 / 错误陈述(原文短引) / 正确事实 / 证据(file:line) / 影响面 / 已修入哪份文档哪节 / 验证方法**。

逐条复核的方法就是**照"验证方法"一列自己跑一遍**。绝大多数是一行 shell:
命令输出与"正确事实"列一致,该条即确认;不一致,说明台账或修订本身有问题,请就地推翻。

两棵树的路径约定(下文命令中直接展开):

| 记法 | 路径 | 状态 |
|---|---|---|
| **工程线** | `/home/ps/Downloads/Stiff-GIPC-c1-ls-graph` | `codex/phase-cd`,台账编制时 HEAD `b3ab747`,现为 `75b3f47`(手册与 v0.8.5.4 delta 已入库) |
| **稳定线** | `/home/ps/Downloads/Stiff-GIPC-stable-08` | 分支 HEAD = `c0339c8` = tag `v0.8.5.4`;工作树自 2026-09-08 起 == HEAD(v0.8.5.4);台账与手册的稳定线行号按 tag `v0.8.5.3`(`b8e27a1`)blob,复核用 `git show v0.8.5.3:<文件>`(编制时工作树曾被误留的 8 文件暂存回退按在该内容上,owner 确认非本意,已 reset) |

`file:line` 一律沿用校验记录里的原始锚点(它们已被校验代理与修订代理两次复核)。
行号会随后续编辑漂移——若命令对不上,先 `grep` 符号名再核行号,不要直接判台账为错。

### 诚实声明

1. **被判定"不成立"而跳过的条目也一并列出**(§4.1),连同修订代理给的跳过理由,供人类复议。
   跳过 ≠ 无事:其中 9 条是"校验清单基于旧修订、当前文本已是正确形态",1 条是"前提已失效",
   1 条是"校验代理自己查漏了出处"。
2. **另有 16 条独立缺陷,校验代理提了、但在任何修订代理的处理清单里都找不到对应记录**(§4.2)。
   本文逐条给出它们**在当前文档中的实际存活状态**(grep 亲验),全部标"待人类复议"。
   这是本轮审计最大的一处系统性缺口,不掩盖。
3. 凡原始素材不足以定论的,写"**需人工复核**",不补写、不推断。
4. 本台账**未**重跑全部 137 条的代码证据。编制时抽样亲验了 8 项(5 项计数 + 3 项稳定仓版本状态,
   见 §5.4),并 grep 核对了 §4.2 全部 16 条的存活状态;其余沿用两级代理的复核结论。
   **这是台账的可信度上界,请据此使用。**

---

## 1. 主表

分组:**A** 数值/计数错误 · **B** 单位与量纲陷阱 · **C** 版本与归属错误 · **D** 机制描述错误 · **E** 引用与链接错误。
编号后的 `×N` = 该缺陷在原始素材中被**独立提出 N 次**(不同校验代理各自抓到,已合并为一条)。

### A 组:数值与计数错误

| 编号 | 错误陈述(原文短引) | 正确事实 | 证据(file:line) | 影响面 | 已修入 | 验证方法 |
|---|---|---|---|---|---|---|
| **D-001** ×4 | "`FrameStatus`(40 只读字段)" | Python 绑定实为 **39** 个 `def_readonly`;struct 的 `contact_class_count[4]` 未绑定到 Python | `bindings/pystiffgipc.cu:38-93` | 按 40 写字段遍历/断言会对不上;手册字段清单原本还漏列 `launch_status` | API_CORE §7.6(并补回 `launch_status`)、README 分册对照表第 10 行 | `awk '/py::class_<frame_fsm::FrameStatus>/,/py::class_<SimEngineConfig>/' 工程线/bindings/pystiffgipc.cu \| grep -c def_readonly` → `39` |
| **D-002** ×3 | "`knob_registry.h` 唯一登记表(**~150** 个)" | X-macro 实登记 **170** 条 `X(STIFF_*)`,另加元旋钮 `STIFF_KNOB_STRICT`;少记 20 条(以正确值 170 为基准偏低约 12%) | `StiffGIPC/config/knob_registry.h:23` 起 | 低估治理面;读者以为登记表还没覆盖大半旋钮 | API_CORE §6.2 与 §11.2(两处)、README 治理表第 12 项(`README.md:139`) | `grep -cE '^\s*X\(STIFF_' 工程线/StiffGIPC/config/knob_registry.h` → `170` |
| **D-003** ×2 | "G14 时点 **84+17+11** 个" | G14 落地提交 `06a6710` 时登记表恰为 **84 条总数**(diag 42 + solver 15 + perf 11 + mode_isolated 7 + mode_strict 4 + python 3 + audit 2),提交正文说的是 "all 84";不存在 "再加 17+11" 的口径;所引 `CI.md:97-102` 与 `knob_registry.h:198-239` 都不含任何计数 | `git show 06a6710:StiffGIPC/config/knob_registry.h` | 编造的分项口径无处可核;读者按它去查 CI.md 会一无所获 | **未见修订记录 → 见 §4.2 / U-14(当前仍存活于 `CHANGELOG_TIMELINE.md:495`)** | `git -C 工程线 show 06a6710:StiffGIPC/config/knob_registry.h \| grep -c 'X(STIFF'` → `84` |
| **D-004** ×2 | "`test_*.py` 共 **29** 个" | phase-cd `examples/` 实为 **30** 个;29 是稳定线的数字;两树唯一差异 = `test_abd_badmesh_kinetic.py` | `ls 工程线/examples/` | §6 明言索引对象是 phase-cd 树,却用了稳定线计数 | README §6、API_EXECUTION §9.6 | `ls 工程线/examples \| grep -c '^test_.*\.py$'` → `30`;稳定线同命令 → `29` |
| **D-005** ×4 | "`gipc_modules/`(**15**)" | 当前树只有 **14** 个 `.inl`(编号 00–14 缺 04);`04_binned_grad_fused_assembly.inl` 在 E1d(`7983be1`)已移入 `energy/`;`GIPC.cu` 的 include 也是 14 条。"15" 只在 phase1(`8d649d9`)时点成立 | `ls StiffGIPC/gipc_modules/*.inl`;`StiffGIPC/GIPC.cu:16-41` | 既单列 `energy/` 又按 15 计 gipc_modules = 重复计数;现状描述错 | CHANGELOG_TIMELINE §1.1(§3.1 的历史表述保留 15,正确);README 已先行改正 | `ls 工程线/StiffGIPC/gipc_modules/*.inl \| wc -l` → `14` |
| **D-006** ×3 | "`GIPC.cu` **−16,874 行**" | 16,874 是该文件的 **churn 合计**(增+删);实际 `+27 / −16,847`。16,849 行的文件不可能净减 16,874 行 | `git show 8d649d9 --numstat -- StiffGIPC/GIPC.cu` | 把 `--stat` 的总变更行数误读成删除量,拆分规模被夸大 | CHANGELOG_TIMELINE §3.1 | `git -C 工程线 show 8d649d9 --numstat -- StiffGIPC/GIPC.cu` → `27  16847` |
| **D-007** ×2 | "`d611138` +432/−41" | `+432/−41`(15 文件)是 C2 系列**首条** `389b530` 的 stat;`d611138` 实为 6 文件 `+985/−8` | `git show 389b530/d611138 --shortstat` | diffstat 张冠李戴,工作量归错提交 | CHANGELOG_TIMELINE §3.6 | `git -C 工程线 show d611138 --shortstat` → `6 files … 985 insertions(+), 8 deletions(-)` |
| **D-008** | "stable `GIPC.cu` 中 `STIFF_DECOUPLE_THRESH` 有 **20** 处直接读点" | 实为 **17** 处 `getenv`(字符串总出现 23 次,含注释);无任何口径得出 20 | 稳定线 `StiffGIPC/GIPC.cu` | 用于论证"旋钮散读"的量化论据不实 | API_EXECUTION §1.4 | `grep -o 'getenv("STIFF_DECOUPLE_THRESH")' 稳定线/StiffGIPC/GIPC.cu \| wc -l` → `17` |
| **D-009** ×2 | "dt=0.02 吞吐 **1.2-1.4×**,见 [stable] `README.md:213-224` U 形扫描" | 被引 U 形扫描峰值是 dt=0.020 → **1.21×**(全表最高);**1.4× 不在该表内**。1.43× 是 `README.md:176-209` 的 case_26 **累积调参配方**(β_tol+newton_tol+dt 三项叠加)总吞吐 | 稳定线 `README.md:213-224`(0.005→0.68×、0.020→1.21× peak、0.030→1.17×) | 单项收益被夸大 ~16%,且归错出处 | README §参数表(1.43× 重新归属到 case_26 配方) | `sed -n '213,224p' 稳定线/README.md` — 峰值 `1.21×` |
| **D-010** ×4 | "整帧图开 = 墙钟 **+10~15%**(`../OPTIMIZATION_ROADMAP.md:78-81`)" | 被引处原文是"大帧回放:图开 = 墙钟 **+3%~75%(场景相关)**";+10~15% 出自 `SIMULATOR_EXECUTION_DESIGN.md` 附录 D(4090 锚场景口径);A800 实测最坏 **35–43%**;手册自己的 §8.2 也写着 +43%/+35% | `docs/OPTIMIZATION_ROADMAP.md:78-80`;`docs/SIMULATOR_EXECUTION_DESIGN.md:240`;`docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:366` | **headline 结论行**:读者据此做通道决策,会把最坏情形低估 3–5 倍。且违反手册 §0"数字一律引自证据文档"的自订约定 | KNOWN_ISSUES(改为权威源 +3%~75%,附录 D 的 +10~15% 钉到锚场景口径)、API_EXECUTION §8.1(并把 +10% 正名为 **Newton 计数盈余**而非墙钟);README 已先行改正 | `sed -n '78,80p' 工程线/docs/OPTIMIZATION_ROADMAP.md` |
| **D-011** | "15 例 13 过、graph-on **1.09–1.94×**" | 表中 forcegrip **2.75×**、UMI_beaker 2.07× 均计为 pass,过例真实上界是 **2.75×**;来源文档散文段自身写错,手册未对表复核就照抄 | `docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:16,18` vs 散文 `:30-31` | 低估最差过例代价约 40% | **未见修订记录 → §4.2 / U-12(仍存活于 `CHANGELOG_TIMELINE.md:339`)** | `sed -n '14,20p;28,32p' 工程线/docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md` |
| **D-012** | "全驻留理论上限 ≈3–4×(**后由 Phase D 兑现为 A800 5.0×**)" | 3–4× 上限是对 **A800 盘子 228 帧大帧回放**算的(29.2s→~8-9s);5.0× 是**另一负载**(D4 RL 微步 300 步)。原负载上图开实测反而 +35~43% 倒贴,该上限**从未被兑现** | `docs/PHASE_A_GPU_RESIDENCY_PROFILE.md:19`;`docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:352-360` | 跨 regime 混接因果,与手册自己的结论"决定胜负的是 regime 不是特性"直接抵触 | **未见修订记录 → §4.2 / U-11(仍存活于 `CHANGELOG_TIMELINE.md:291`)** | `sed -n '19p' 工程线/docs/PHASE_A_GPU_RESIDENCY_PROFILE.md` |
| **D-013** ×3 | "forcegrip 图开 13.34→9.78 s(−27%),step 图比 **1.96→1.46×**" | 13.34→9.78 s 那轮(对照 graph-off 6.81 s,`c74071a`)的比值是 **1.44×**;**1.46×** 出自另一轮干净卡出货测量(off 7.10 / on 10.37,`b842d2f`)。秒数配了另一口径的倍率 | `docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:278-279` vs `:334` | 违反手册开篇"引用倍率必须连同测量口径"的自律;两轮数据被拼成一句 | CHANGELOG_TIMELINE §3.8(两轮拆开分别标注) | `git -C 工程线 log -1 c74071a` 与 `git -C 工程线 log -1 b842d2f`,比对两组秒数 |
| **D-014** | "RL 微步:residency 通道 **2.8–5×** 胜" | 区间由两个不同比较基准拼成:5× 只在 A800 且是 vs v0.8.5 口径;2.8× 是 HEAD 内部 episode-vs-宿主 口径;同节表格的 4090 行是 4.00→3.18 ms/步 ≈ **1.27×**。诚实区间是 **~1.3×(4090) 到 5.0×(A800)** | 该文档 §7 表自身;`docs/SIMULATOR_EXECUTION_DESIGN.md:74` | 4090 用户按 vs-v0.8.5 理解会高估收益 2 倍以上 | PRINCIPLES_EXECUTION §7 | 打开 `PRINCIPLES_EXECUTION.md` §7 表,核 D4 4090 行 `4.00 → 3.18` |
| **D-015** | "A800 D4 关节接触 **150 步**:19.3 → 9.7 → 3.85 ms/步" | 19.3/9.7/3.85 这条链出自 **300 步**(dt=0.01,contact priming 后)的 C6-w 横测轮;"150 步"是 2026-08-04 SIM_EXEC 轮(以 4090 为主)的实验描述。数字本身正确,**出处描述错** | `docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:356-364`;`docs/SIMULATOR_EXECUTION_DESIGN.md:131` | 手册 §8.2 自己要求"分轮引用",§8.1 却把 300 步轮的数字挂在 150 步标签下 | API_EXECUTION §8.1(A800 链标 300 步/C6-w 轮,4090 链保留 150 步轮) | `sed -n '355,364p' 工程线/docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md` — 节内写明 "300 steps after contact priming" |
| **D-016** | 同一表格前言写"**4090 干净卡**",RL 行却标 **A800** | 两份源文档互斥:`A800_ALLEXAMPLES` 把该表放在 "(4090, clean GPU)" 标题下;`SIMULATOR_EXECUTION_DESIGN.md:132-137` 把同组数标为 A800 列(4090 列是 4.25/3.72/3.18)。**平台口径本身存疑** | `A800_ALLEXAMPLES_TIMING_2026-08-01.md:342,355-363`;`SIMULATOR_EXECUTION_DESIGN.md:132-137` | 违反本文导言"每个数字都带口径";平台标错会让复现实验选错卡 | CHANGELOG_TIMELINE §3.10 与 §4(逐行标口径,取 SIM_EXEC 的 A800 归属)并**新增 §8 待核实项 #8** | 对比 `sed -n '342p' …A800_ALLEX…md` 与 `sed -n '132,137p' …SIMULATOR_EXECUTION_DESIGN.md` |
| **D-017** ×2 | "宿主仅 **40 次** cudaGraphLaunch + **40 次** 24B D2D action 发布" | D2D 发布实为 **80 笔**(ABI 每步发布两笔小 D2D);4090 与 A800 节点级 Nsight 复核均为 80。"40 次"抄自 `0735481` 提交正文,2026-08-06 干净复测已给出 80;手册内部 `:351` 写 80、`:305` 写 40 自相矛盾 | `docs/GPU_NATIVE_RL_PLAN.md:109,46,120` | 证据链定格数字自相矛盾,复现者对不上计数 | CHANGELOG_TIMELINE §3.7(改 80 并注明 40-vs-80 源冲突) | `grep -n 'D2D' 工程线/docs/GPU_NATIVE_RL_PLAN.md \| head` |
| **D-018** | "256 粒度的 2 的幂阶梯,**最多 2× 放大**(实测 971162 → 2097152 = **2.16×**)" | 按同段给出的公式:`blocks=ceil(971162/256)=3794` → 上取 2 的幂 `4096` → tier `1,048,576`,仅 **1.08×**;阶梯放大上界数学上 **< 2×**,不可能产出 2.16×。2.16 必含公式之外的因子(headroom×2 时代) | `StiffGIPC/linear_system/utils/capacity_tier.h:15-16,45-49`(矛盾源头是 `:18-19` 被照抄的代码注释) | 同句内"最多 2×"与"=2.16×"直接冲突,读者无法据此估容量 | PRINCIPLES_EXECUTION §1.6(拆开归因,注明注释数字含阶梯外因子) | `sed -n '10,20p;42,50p' 工程线/StiffGIPC/linear_system/utils/capacity_tier.h`,自算 971162 → 1048576 |
| **D-019** | "同一轴 8 帧内再次跨档 → **每连击多左移一档**" | `escalation_peek` 返回 `min(streak+1, 2)`,即**额外左移最多 2 档**;代码注释 "earn 2x per streak step, **capped at 4x**" | `StiffGIPC/frame_fsm/frame_transaction.cu:4165-4176` | 字面读作无界升级(连击 5 次→左移 5 档),与 ≤4× 封顶不符 | PRINCIPLES_EXECUTION §5.4 | `sed -n '4165,4176p' 工程线/StiffGIPC/frame_fsm/frame_transaction.cu` |
| **D-020** ×2 | "strict 的峰值 Newton 系统性**最低**(盘子 **15 vs 22/19**),(THREE_MODE_MATRIX_rc2)" | 22/19/15 出自更早的 `MODE_MATRIX_REPORT_2026-07-27`;被点名的 rc2 同场景是 merged **14**@fr164 / isolated 22@fr178 / strict 15@fr202——**merged 更低**,直接反驳"系统性最低"在盘子场景成立 | `docs/THREE_MODE_MATRIX_rc2.md:31-33`;`docs/MODE_MATRIX_REPORT_2026-07-27.md:44-47` | 两次战役的数在同一句里混引同一出处,且被所引新数据推翻 | PRINCIPLES_EXECUTION §4.3(改为"多数场景最低(07-27 批),非普适",要求引用注明批次) | `sed -n '31,33p' 工程线/docs/THREE_MODE_MATRIX_rc2.md` |
| **D-021** ×2 | "移植量很小(清 **5+5** 个宿主计数镜像 + 摩擦快照旗标 + `Kappa=0`)" | 稳定线 `reset_transient_contact_state` 除 `h_cpNum[0..4]`/`h_cpNum_last[0..4]` 外,**还清 `h_gpNum` 与 `h_gpNum_last` 两个地面配对镜像**(共 12 个计数镜像) | 稳定线 `StiffGIPC/sim_engine.cu:3620-3636`(:3628-3629 为两个 gp 镜像) | 照 "5+5+旗标+Kappa" 移植,会把幽灵摩擦缺陷原样换到**地面通道**复现——同节的根因行自己点名了 gp 镜像,修复状态行却漏了 | KNOWN_ISSUES §1.2(已补 gp 镜像与后果说明) | `sed -n '3620,3636p' 稳定线/StiffGIPC/sim_engine.cu` |
| **D-022** | "μ 饱和实测表 + 14 臂抓取矩阵(0.018/0.044/0.086/0.153、62–808 N、提升 50–64 mm、≤2 mm、~15 mm)" | **全部数字在两个代码树、稳定线 CHANGELOG、以及手册指定的实测权威源中全部检索不到**,唯一出处是手册自身。§9.6 声称"有 CHANGELOG 背书",但 CHANGELOG 只有剪切台 0.600±0.008 一条;与 v0.8.5.3 记录"beaker 摩擦实验待开"也相抵触 | 全仓 grep 仅命中 `PRINCIPLES_CONTACT.md:448-453` 自身;稳定线 `CHANGELOG.md:7-41` 无 μ 扫描 | 引用时无仓库内出处可核 | PRINCIPLES_CONTACT §5.2(先撤表);**Fable 5.1 复验更正(2026-09-08)**:这些数字**不是无中生有**——它们是 2026-08-11 A800 beaker 战役的真实测量(会话记录),只是原始日志未归档进仓库,两树 grep 自然零命中。已重建为 `docs/BEAKER_FRICTION_CAMPAIGN_2026-08-11.md`(带出处声明)并回填 §5.2;归档/重跑登记 OP-038 | `grep -rn '0.153\|62–808' 工程线 稳定线 --include=*.md` — 除本手册外零命中 |
| **D-023** | "beaker 抓握场景修复后摩擦读数 **5.96 N**(法向 20 N)" | 该组数字在被引的稳定线 CHANGELOG(v0.8.5.3 节)与两仓任何文档/脚本中都不存在;CHANGELOG 记载的发布验证只有剪切台 0.600±0.008 与 266–270 滞后配对 | 稳定线 `CHANGELOG.md:15-27` | ~~伪造的"发布验证点"~~ → **判定更正(Fable 5.1,2026-09-08)**:5.96 N / 20 N 是 2026-08-11 稳定线 wheel 验证时的真实读数(会话记录),错在"CHANGELOG 背书"这个出处归属——CHANGELOG 里确实没有;数字本身真实但未归档 | PRINCIPLES_CONTACT(整句删除;现由 `docs/BEAKER_FRICTION_CAMPAIGN_2026-08-11.md` §2 承载,OP-038) | `grep -rn '5\.96' 稳定线/CHANGELOG.md` → 零命中 |
| **D-024** | "全部 v0.8.5 后工程化工作(**209 条提交**)只存在于本地仓库" | 远端分支 `origin/internal/v0.8.6-rc1` 已含其中 **62 条**;给出的证据(无 `origin/codex/phase-cd`)只支持"该分支未推",支持不了"全部只在本地" | `git branch -a` 含 `remotes/origin/internal/v0.8.6-rc1`(tip `148a894`);`comm -12` 两个 rev-list = 62 | 对备份/发布决策是实质性差别(误判"一台机器丢了就全没了") | **未见修订记录 → §4.2 / U-15(仍存活于 `CHANGELOG_TIMELINE.md:520`)** | `git -C 工程线 rev-list v0.8.5..origin/internal/v0.8.6-rc1 \| sort > /tmp/a; git -C 工程线 rev-list v0.8.5..HEAD \| sort > /tmp/b; comm -12 /tmp/a /tmp/b \| wc -l` |
| **D-025** ×2 | "`git describe` = `v0.8.6-rc2-internal-147-gb3ab747`" | 裸 `git describe` 实际输出 **`v0.8.6-rc1-internal-164-gb3ab747`**:`v0.8.6-rc2-internal` 是**轻量 tag**,describe 默认只认 annotated tag。需 `git describe --tags` 才得文中值。"rc2 后 147 条"计数本身正确 | `git cat-file -t v0.8.6-rc2-internal` → `commit`;`v0.8.6-rc1-internal` → `tag` | 标称"亲验"的命令输出实为拼造,读者照抄命令得不到同样结果 | CHANGELOG_TIMELINE §2.7(改 `--tags` 并附轻量 tag 说明) | `git -C 工程线 describe` vs `git -C 工程线 describe --tags` |
| **D-026** ×2 | "六场景 **1.62×~10.8× 全慢**" | 该证据表只有 **5 个倍率**;第六个 `beaker_finray` 图开**根本没跑完**(FAIL — OVF_CCD retry budget,无倍率),`towel` 除慢外还是**物理 FAIL**。文档自述 "two of six scenes still fail" | `docs/GRAPH_DEFAULT_ON_EVIDENCE.md:4,32-39` | 把"一个跑不完 + 一个物理错"弱化成纯性能问题 | KNOWN_ISSUES §3.2(改为 5 倍率 + 1 FAIL) | `sed -n '4p;32,39p' 工程线/docs/GRAPH_DEFAULT_ON_EVIDENCE.md` |
| **D-027** | "C6-p:forcegrip 4090 **2.08×→1.39×**" 与 "C6-w:step 图比 **1.96→1.46×**" | 同一场景同一卡的 graph-vs-host 比值,8/1(C6-p 后)记 1.39×、8/3(C6-w 前)却是 1.96×,**中间的回退全文无一句解释**;C6-p 行也未带帧数/轮数口径 | `docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:75` vs `:278,:334` | 内部不自洽,违反本文"每个数字必须连同测量口径"的自律 | **未见修订记录 → §4.2 / U-13(C6-p 行仍存活于 `CHANGELOG_TIMELINE.md:341`)** | `sed -n '75p;278p;334p' 工程线/docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md` |

### B 组:单位与量纲陷阱

> 本组是本轮审计最集中的一类高危缺陷:引擎的梯度空间约定是 `dE/dx = −F·dt²`,
> **`−1/dt²` 换算只在两条路径上做过**:`get_vertex_contact_forces`
> (`03_step_getters_export.inl:1758` 的 `neg_inv_dt2`)与 `get_contacts_device`
> (经 `GIPC::exportContacts`,`12_host_wrappers_fem.inl:445` 的 `inv_dt2`)。
> 其余 getter 全是裸梯度。dt=0.01 时数值差 **1e4 倍**且符号反向。

| 编号 | 错误陈述(原文短引) | 正确事实 | 证据(file:line) | 影响面 | 已修入 | 验证方法 |
|---|---|---|---|---|---|---|
| **D-028** | "`GRIP_TARGET=0.03`(**N**);0.03 在默认步长帽下给 ~17-18mm 捏合" | 比较对象是 `get_body_contact_force(_batched)` 的**范数之和**,该 getter 返回原始增量势垒梯度 `dE/dx = −force·dt²`(LEGACY UNITS,**无** `−1/dt²` 换算)。按 foldshirt 家族默认 dt=0.02,阈值 0.03 对应约 **75 N** 物理力;换 dt 后同一数值的物理含义随 dt² 漂移。手册照抄了示例代码里同样错误的 "(N)" 注释 | `stiff_physics/engine.py:1591-1601,1783-1793`(docstring 明写 "NOT Newtons");`StiffGIPC/engine_modules/03_step_getters_export.inl:1141-1145`;`examples/umi_finray_lib.py:517-521` | **高危**:抓取力阈值差 2500 倍(0.03 vs 75 N);换 dt 后行为静默漂移 | API_EXECUTION §9.1(明标"单位不是牛顿",给出 ≈75 N 换算与 dt² 漂移警告) | `grep -n 'NOT Newtons' 工程线/stiff_physics/engine.py` |
| **D-029** ×2 | "`get_prismatic_drive_force(idx)` → 当前 `K·(target−d)` 驱动力(**N**)" | `drv.stiffness = ratio·per-joint·(m_p+m_c)` 是**质量量纲、刻意无 dt² 因子**;返回的是裸增量势梯度(kg·m ≡ N·s²),换算牛顿需 **÷dt²**,dt=0.01 时差 **1e4 倍**。同表 `set_prismatic_force` 经 `M⁻¹F·dt²` 进 q_tilde 才是真牛顿——两者被当同量纲比较 | `03_step_getters_export.inl:1136`(无 ÷dt²);`abd_system/…/setup_abd_system_gradient_and_hessian.cu:1956`;单位约定 `03_step_getters_export.inl:1753-1759`;真牛顿对照 `cal_q_tilde.cu:196-215` | **高危**:力限位置控制"封顶握力"会直接比错单位。根源是 pybind docstring 自带 "(N)",手册未加甄别地继承 | API_CORE §4.5(改为梯度空间 `F_phys·dt²`,明写"与 `set_prismatic_force` 比较前先 ÷dt²") | `sed -n '1136p' 工程线/StiffGIPC/engine_modules/03_step_getters_export.inl` — 无任何 `/dt` |
| **D-030** | "`get_pair_contact_force(A,B)` \| raw 梯度(**Python 侧乘 −1/dt²**)" | Python 包装**原样透传**;docstring 明说返回 "raw IP scaling",要求**调用者自己**套 `−1/dt²`;C++ 注释同 | `stiff_physics/engine.py:1607-1615`;`03_step_getters_export.inl:1357` 附近 | 信了括号里的说法就把 `−F·dt²` 当牛顿用;还与同格主标签"raw 梯度"自相矛盾 | **未见修订记录 → §4.2 / U-01(仍存活于 `PRINCIPLES_CONTACT.md:610`)** | `sed -n '1607,1615p' 工程线/stiff_physics/engine.py` — return 处无任何缩放 |
| **D-031** ×2 | "`cloth_density` \| `2e2` \| **—** \| 布料**面**密度;`m += ρ·area·h/3`" | 质量装配先 `area *= clothThickness`(面积×厚度=体积)再 `m += ρ·area/3`(此处 `area` 已是体积,勿再乘 h),ρ 乘的是**体积** → `cloth_density` 是**体密度 kg/m³**。默认 200 kg/m³ 在 h=1mm 下 ≈ 0.2 kg/m² 面密度。手册自己写出的含 h 公式已排除面密度解读,单位栏还留空 "—" | `StiffGIPC/engine_modules/01_config_upload.inl:91,99-101`;对照 FEM 体密度 `:67-70` | **用户按织物常用面密度(~0.2 kg/m²)填值,质量会错 ~1000 倍**(厚度被乘两次) | API_CORE §2.x(标 kg/m³ + 1000 倍陷阱警告);PRINCIPLES_DYNAMICS §参数表 | `sed -n '88,102p' 工程线/StiffGIPC/engine_modules/01_config_upload.inl` |
| **D-032** | "`energy_abs_tol`/`energy_rel_tol` \| 0.0/0.0 \| **J** / 相对" | 线搜索比较的是**增量势(IP)标量**:动能项 `½m‖x−x̃‖²` 未除 dt²(kg·m² = J·s²),弹性项乘 dt² 后同为 J·s²,接触/摩擦项由 κ/λ 吸收同一量纲 → 单位是 **J·s²**,不是 J | `energy/01_energy_host_dispatch.inl:285-298`(slot0 裸加、slot1..3 乘 dt²);`energy/10_kinetic.inl:7-52` | 按物理焦耳估容差会差 **dt²≈1e-4 倍**;这是该分册唯一一处落进 dt² 陷阱的标注 | PRINCIPLES_DYNAMICS §参数表(改 J·s²,附 dt² 说明) | `sed -n '285,300p' 工程线/StiffGIPC/energy/01_energy_host_dispatch.inl` |
| **D-033** | "`fDhat` = 1e-4·eff \| **m²**;派生 `eps` = √fDhat·dt \| **m**" | 二者不可能同时成立:若 fDhat 是 m²,则 `eps=√fDhat·dt` 是 m·s、`fricDHat=fDhat·dt²` 是 m²·s²,却在代码里与切向位移平方 `‖u‖²`[m²] 直接比较。要成立,fDhat 必须读作**平方速度 (m/s)² = epsv²**。§4.3 正文自己也写 "epsv ≈ 1e-2·eff_scene_diag [m/s]";v0.8.5.4 提交信息坐实("absolute_epsv = 1e-4 m/s") | `energy/16_friction.inl:830`;`energy/02_contact_energy_device.inl:363-365`;稳定线提交 `0894958` | 两张表内部互斥,按 m² 读会把静滑阈值调错量级 | PRINCIPLES_CONTACT §1.3 与 §8.1(两表统一为 (m/s)²=epsv²) | `sed -n '830p' 工程线/StiffGIPC/energy/16_friction.inl` — 调用侧 `fDhat*dt*dt, sqrt(fDhat)*dt` |
| **D-034** | "λ = −κ·2√d·∂b/∂d(**法向力幅**)……约 **418 N** 法向接触力下 stitch 伸长 ~10.5 µm" | 全篇(含专讲 `get_vertex_contact_forces` 的 §5.8)**没有一处声明接触力读数的换算约定**:导出缓冲存的是 IP 梯度 `dE/dx = −force·dt²`,物理牛顿 = `−gradient/dt²`(代码注释还记载首个发布曾错乘 `+1/dt²` 把所有力矢量翻向)。λ 实为 `dt²×物理法向力`,单位非 N | `03_step_getters_export.inl:1753-1767`;λ 定义 `gipc_modules/09_friction_sets_host_mem.inl:152-158,29-30` | 读者自行从梯度/λ 换算力时极易漏掉 `/dt²` 与负号 | PRINCIPLES_DYNAMICS §5.8(补换算约定说明 + λ 的 IP 量纲括注) | `sed -n '1753,1767p' 工程线/StiffGIPC/engine_modules/03_step_getters_export.inl` |

### C 组:版本与归属错误

| 编号 | 错误陈述(原文短引) | 正确事实 | 证据(file:line) | 影响面 | 已修入 | 验证方法 |
|---|---|---|---|---|---|---|
| **D-035** ×3 | 硬钉(USE_HARD_PIN)替换法弹性链式法则缺失标 **【仅 phase-cd】**("稳定线未核实存在该路径") | 稳定线单体 `GIPC.cu` 里有**逐字相同**的 KNOWN LIMITATION 注释块与活代码路径(inertia-only 链式法则、`m5_drive_joint2_test` k=1000 撞 cap、TODO M3.5),连 `[M3.5]` WARN 截断警告也在;Python 绑定暴露 `add_fem_pin_to_abd`,`n_fem_pins>0` 即走该路径。**一条 grep 即可证伪** | 稳定线 `StiffGIPC/GIPC.cu:13894-13906`、`:13787`、`:13906`;绑定 `bindings/pystiffgipc.cu:250-292` | **高危**:这是 Newton 不收敛级的正确性缺陷,稳定线用户(硬钉+动态关节)被错误告知与己无关 | KNOWN_ISSUES §1.4(改标【稳定线+phase-cd】) | `grep -n 'KNOWN LIMITATION' 稳定线/StiffGIPC/GIPC.cu` → `13894` 附近 |
| **D-036** | "确定性:strict run-to-run 逐位 **+ 跨架构(sm_89 ≡ sm_80)**金锚 …【**稳定线+phase-cd**】" | 跨架构锚等值只在 **phase-cd 的 dlto 构建**上实测过;稳定线的锚 `f7fb5a786c2d7935` 在其 CHANGELOG 中明确是 **cross-ENV(跨环境)**锚,全库无任何 sm_89≡sm_80 的稳定线证据;稳定线以含 PTX 的 `"80;89;120"` 构建(JIT 生成的 SASS 随驱动/架构而变) | 稳定线 `CHANGELOG.md:58`;稳定线 `CMakeLists.txt:15`;跨架构证据仅 `docs/RELEASE_NOTES_v0.8.6-rc1.md:132-137` | **高危**:依赖稳定线 wheel 做跨 GPU 逐位复现实验的用户会被误导;确定性过度承诺 | README(该行已在当前修订中为正确形态,见 §4.1 skip #4) | `grep -n 'cross-env determinism anchor' 稳定线/CHANGELOG.md` |
| **D-037** ×3 | "磁盘上的稳定线工作树 checkout 在 **v0.8.5.4 之后一个提交**(`c0339c8`)" | `c0339c8` **就是 tag `v0.8.5.4` 指向的提交本身**(`git describe --tags --long` = `v0.8.5.4-0-gc0339c8`);带 "release(v0.8.5.4)" 消息的 `0894958` 是其**父**提交。**更关键**:工作树文件内容经 8 个文件的未提交回退,与 `v0.8.5.3`(`b8e27a1`)**逐字节相同**(`git diff v0.8.5.3` 为空),故稳定线行号偏移**恰为零**,"可能有个位数偏移"的前提不成立 | 稳定线 `git describe --tags --long`;`git log --oneline -3`;`git status --short` = 8 文件 staged | 双向误导:一边把 c0339c8 的行为排除在 v0.8.5.4 之外,一边让读者以为行号有偏移。**且工作树状态极脆弱**——`git checkout/stash/reset` 任一操作会让"稳定线"静默变成行为不同的 v0.8.5.4 | API_CORE 版本行(加脆弱性警告)、PRINCIPLES_DYNAMICS §0 行号约定 + 文末快照行 + STIFF_EPSV 附录行 | `git -C 稳定线 describe --tags --long` → `v0.8.5.4-0-gc0339c8`;**2026-09-08 后**:`git -C 稳定线 status --short` → 空、`git -C 稳定线 diff v0.8.5.3 --stat` → 8 files +511/−8(工作树已恢复 HEAD 内容;原暂存回退系误留) |
| **D-038** ×2 | rc2 增量表把 "**dlto 默认开**(`40c9f11`/`45d74f0`)"、"金锚迁移 → `0544461bd82123ae`"、"三模式全矩阵(`19b257a`)56/56 绿" 列为 **v0.8.6-rc2-internal 的增量** | 三者都**晚于** rc2 tag `6b0e02e`(07-28 01:59)与 release 本体 `7e39591`(01:43):`19b257a` 03:12、`40c9f11`/`45d74f0` 08:10。**checkout rc2 tag 得到的仍是旧锚 `f7fb5a786c2d7935`、dlto 未开**。`RELEASE_NOTES_v0.8.6-rc1.md:130` 自注 "rc2 后首个变更",手册抄录时丢掉了"rc2 后"限定词 | `git merge-base --is-ancestor` 三者对 rc2 tag 均非祖先;`docs/RELEASE_NOTES_v0.8.6-rc1.md:97-98,129-130` | **高危**:锚值与默认集的版本归属全错,按版本条目复现会拿到不同的确定性锚 | CHANGELOG_TIMELINE §2.6 / §4 rc2 行 / §5 金锚表(拆分为"tag 内增量"与"tag 后采纳") | `git -C 工程线 merge-base --is-ancestor 40c9f11 v0.8.6-rc2-internal; echo $?` → 非 0 |
| **D-039** | "### 3.3 rc1 工程化(7/27)" 表,末行以 `6c9d730`(rc1 tag)收尾 | 该表 **12 行中有 11 行的提交在 rc1 tag(`6c9d730`,07-27 09:58)之后**(6c562bc 19:51、c86a1ea 23:08 … 06a6710 22:42);只有 `d0c0071`(02:00)在 rc1 内。把 rc1→rc2 窗口的工作编排为"通往 rc1 的战役"并以 release 行收尾,还与 §2.6 rc2 增量表**双重记账** | `git merge-base --is-ancestor <hash> v0.8.6-rc1-internal` 逐条亲验 | 战役叙事与版本边界错位,同一工作被记两遍 | CHANGELOG_TIMELINE §3.3(标题/前言/TOC 均改为 "rc1→rc2 窗口") | `git -C 工程线 merge-base --is-ancestor 06a6710 v0.8.6-rc1-internal; echo $?` → 非 0 |
| **D-040** ×2 | "后续配套(同窗口,**工程线** `03e70f6`,7/24):常规增长搬到帧首 discard-grow" | `03e70f6`(2026-07-24)**早于分叉点**(v0.8.5.2 / `05c3f75`,07-25),是两线**共同历史**;稳定线同样携带该 [P0-mem] 机制 | `git merge-base --is-ancestor 03e70f6 v0.8.5.2` 为真(两仓);稳定线 `StiffGIPC/GIPC.cu:13221,13255`("2.7 = 2 x 1.35") | 稳定线用户误以为自己缺此内存优化(仍双驻留 24GB 缓冲) | CHANGELOG_TIMELINE §2.2(改标"分叉前/两线共有") | `git -C 稳定线 merge-base --is-ancestor 03e70f6 HEAD; echo $?` → `0` |
| **D-041** ×2 | "相对稳定线的公开 API 增量(**名字级 diff 亲验**):… `save_checkpoint`/`load_checkpoint`(v2 格式)" | 稳定线 pybind **同样暴露这两个名字**(v1 格式,magic `0x53544B50`),名字级并非增量;真正的增量是 **v2 格式**与 Python `Engine` 包装层(带 LifecycleError/CheckpointError 语义) | 稳定线 `bindings/pystiffgipc.cu:496-497`;实现 `GIPC.cu:16594`,magic `:16612` | 照此用 `hasattr` 探测两线会得到**假阳性**;且与 §1.4"同名 API 行为差异"自相矛盾 | CHANGELOG_TIMELINE §2.7(移出名字级清单 + hasattr 警告) | `grep -n 'save_checkpoint' 稳定线/bindings/pystiffgipc.cu` |
| **D-042** | "稳定线**有 checkpoint v2**,但帧入口配对集重建修复是否回传未验证" | 稳定线**没有** v2:其 save/load 是调试级格式(magic `STKP`),只存 FEM 4 数组 + ABD q 族 + Kappa + total_Frames;**无 14 旗标模式位图、无格式版本号、无"换模式恢复硬拒"**;更危险的是 load 端 magic/计数不匹配时只 `printf "[ckpt] MISMATCH"` 后**静默 return**(不抛错,继续用未还原状态) | 稳定线 `GIPC.cu:16591-16631`、`GIPC.cuh:602-608`;对照 phase-cd `checkpoint/checkpoint_io.cu:292-309,970-972`;`STIFFCP2` 在稳定线全树 0 命中 | 读者误以为稳定线 checkpoint 具备同等安全语义,实际会**静默载入未还原状态** | PRINCIPLES_EXECUTION §4.5(改标【仅 phase-cd】+ 写明稳定线格式的静默失败风险) | `grep -rn 'STIFFCP2' 稳定线/` → 零命中;`sed -n '16628,16632p' 稳定线/StiffGIPC/GIPC.cu` |
| **D-043** | "# 6. 步健康遥测 **【仅 phase-cd】**(稳定线无这些计数器)" | §6.1 表中 **8 个 getter 有 6 个稳定线也有并已绑 pybind**(`get_total_newton_iters`/`get_total_pcg_iters`/`get_total_collision_pairs`/`get_max_collision_pairs`/`get_total_frames_done`/`get_total_energy_tolerance_accepts`);真正仅 phase-cd 的只有 `get_ls_exhausted_count`/`get_ls_nonfinite_count` 与 `get_frame_status`。**真实差异是**:稳定线上这些是 TU 全局(有跨引擎串台风险),phase-cd 已成员化 | 稳定线 `StiffGIPC/sim_engine.h:642-647`;`bindings/pystiffgipc.cu:435-447`;`sim_engine.cu:4515`(返回 TU 全局 `totalNT`) | 稳定线用户误以为没有任何 Newton/PCG 遥测可用 | API_EXECUTION §6(改标【稳定线+phase-cd】,点明 TU 全局 vs 成员化的真实差异) | `sed -n '642,647p' 稳定线/StiffGIPC/sim_engine.h` |
| **D-044** | "# 5. checkpoint 与 get_frame_status **【仅 phase-cd】**"(节标题 + 目录) | 节级标签过宽:`save_checkpoint`/`load_checkpoint` 在稳定线存在且可用(legacy 无版本格式);仅 **v2 格式、~5e-17 续跑精度、`get_frame_status`** 是 phase-cd 专属 | 稳定线 `GIPC.cu:16593-16655` | 只看目录/标题的稳定线用户会误以为完全没有 checkpoint 能力 | API_EXECUTION §5(标题与 TOC 收窄) | `grep -n 'GIPC::save_checkpoint' 稳定线/StiffGIPC/GIPC.cu` |
| **D-045** | "[P1-dyn] 非 hybrid 场景按实际接触数动态增长 **【仅 phase-cd】**" | 稳定线**同款存在**(同注释同逻辑:"Non-hybrid scenes … instead of the worst-case 2*(surf+edge)*29",`m_dynamic_triplet`、帧首 provable-upper-bound grow 都在),共 3 处 | 稳定线 `StiffGIPC/GIPC.cu:9503-9509`、`12932-12937`、`13227-13233` | 稳定线用户误以为自己走最坏值预分配 | PRINCIPLES_CONTACT §容量策略(改标【稳定线+phase-cd】,附三处行号) | `grep -n 'P1-dyn' 稳定线/StiffGIPC/GIPC.cu` → 3 处 |
| **D-046** | "S4-dev 设备派生 mask **为 phase-cd 工作**" | 该机制(`_mask_from_env_alpha` + `STIFF_PERENV_MASK_DEV` 全套)在稳定线 v0.8.5.3(`b8e27a1`)中**已完整存在** | 稳定线 `GIPC.cu:10904`(注释)、`:10916`(kernel)、`:15467-15470`(门)、`:15524`、`:16301-16305`(调用点) | 读者以为稳定线只有 host 侧 mask | PRINCIPLES_EXECUTION §3.5(改标【稳定线+phase-cd】并注明同源) | `git -C 稳定线 show b8e27a1:StiffGIPC/GIPC.cu \| grep -n 'S4-dev'` |
| **D-047** ×2 | "四步协议 **【稳定线+phase-cd】** … 4. **复活**(`reviveEnv`)(稳定线 teleport 无 reviveEnv 步)" | 稳定线**全树没有 `reviveEnv`**——不是"teleport 没接上这一步",而是整个函数/复活路径都不存在。稳定线 env 一旦检疫**无复活途径**,其对应物是 `reset_transient_contact_state` | 稳定线 grep `reviveEnv` 唯一命中 `GIPC.cu:11828`(无关注释);phase-cd `multienv/isolation.cu:192` | 括号写法暗示稳定线也有该机制、可另行调用 | PRINCIPLES_CONTACT §6.4(第 4 步整体改标【仅 phase-cd】+ 协议头拆分 + 指向 §7.4) | `grep -rn 'reviveEnv' 稳定线/StiffGIPC/` |
| **D-048** | "`reset_transient_contact_state()` … phase-cd【**无**(其 teleport 自带 pair 集重建)】" | "teleport 自带重建"只覆盖稳定线该 API 的**一半语义**:稳定线除清 pair 镜像外还清 lagged 摩擦快照并**将 Kappa 归零**(实现注释:仅清 pair/friction 镜像后仍有实测 **0.9 µm** 状态发散)。phase-cd 的 `teleport_abd_bodies` 只重建当前 pair 集,**从不触碰 `*_last` 镜像与 Kappa** | 稳定线 `sim_engine.cu:3620-3636`;phase-cd `03_step_getters_export.inl:2484-2585`(无一行清镜像/Kappa) | phase-cd 上就地 episode 重置**无法达到"如新进程"语义**,且无替代 API;写成"等效"是过度承诺 | README(该行已在当前修订中为正确形态,见 §4.1 skip #8);KNOWN_ISSUES §1.2 已补 Kappa | `sed -n '2484,2585p' 工程线/StiffGIPC/engine_modules/03_step_getters_export.inl \| grep -c 'Kappa'` → `0` |
| **D-049** | "tag `v0.8.5.4` … **CHANGELOG 无条目**" | tag 所指树的 `CHANGELOG.md` **有完整 `## [0.8.5.4] — 2026-08-12` 条目**(且正是 §1.5 所引 0.00 mm / +9% 数字的出处);只有被回退的磁盘工作区里没有 | `git -C 稳定线 show v0.8.5.4:CHANGELOG.md` 行 7 / 38 / 40 | 作为"发布状态待核实"的证据链,会误导读者以为 0.8.5.4 从未写过 changelog | **已关闭(2026-09-08)**:OP-001 关闭那一轮顺带改正,两处现文均已写明"条目存在于 tag 树,只是被回退的磁盘副本没有"(grep "CHANGELOG 无条目" 两文件零命中) | `git -C 稳定线 show v0.8.5.4:CHANGELOG.md \| sed -n '1,12p'` |
| **D-050** | "经 `git diff --stat b8e27a1..HEAD` 亲验,其后差异**只触及摩擦锚/静摩擦相关文件**" | 实际触及 **8 个文件**,还包括 `CHANGELOG.md`、`bindings/pystiffgipc.cu`、`pyproject.toml`;且其中含**正式发布提交** `0894958`("release(v0.8.5.4): default-on true static friction"),不只是零散摩擦修复 | `git -C 稳定线 diff --stat b8e27a1..HEAD` = 8 files(GIPC.cu +403、GIPC.cuh +35、…) | 把 v0.8.5.4 发布轻描为"数个提交、只触摩擦文件",低估工作树与文档锚点的行为差距 | PRINCIPLES_EXECUTION §0 亲验段(改为完整 8 文件清单 + 点名 v0.8.5.4 发布) | `git -C 稳定线 diff --stat b8e27a1..HEAD` |
| **D-051** | "**行为承诺**:无 solver-dynamics 改动;不调用新 API 时轨迹与 0.8.5.2 **完全一致**" | `1d05c7a` 改变的是**既有 API** `teleport_abd_bodies` 的行为(v0.8.5.2 已存在,非新 API;修复前 teleport 后 40/40 帧 LS 预算耗尽,修复后必然不同轨迹)。调用 teleport 的 0.8.5.2 脚本在 0.8.5.3 上轨迹**不一致** | 稳定线 `git grep -c teleport_abd_bodies v0.8.5.2 -- StiffGIPC/sim_engine.h` = 2(可溯至 `9ab50f4`/2026-06-25);`CHANGELOG.md:10-12` 与 `:27-30` 并存 | **危险的过度承诺**,且与同节表格第 3 行自相矛盾;上游 CHANGELOG 本已过宽,手册照抄并强化为"完全一致" | **未见修订记录 → §4.2 / U-09(仍存活于 `CHANGELOG_TIMELINE.md:184`)** | `git -C 稳定线 grep -c teleport_abd_bodies v0.8.5.2 -- StiffGIPC/sim_engine.h` → `2` |
| **D-052** | "phase-cd **未移植**;移植量很小 …【5 步移植建议】" | 对 HEAD `b3ab747` 而言"未移植"属实,但**同仓已存在完整的移植成品**:分支 `port/friction-anchor-086`,首提交 `3ec2734` 已按模块化布局落位(GIPC.cuh 成员、09_friction 快照、ipc_solver 提交前钩子、engine_modules/03 accessor、bindings、engine.py),后续还移植了 v0.8.5.4 的 absolute_epsv/friction anchors | `git -C 工程线 log port/friction-anchor-086` | 5 步"移植建议"在重新发明已存在的工作,且未提示该分支存在 | KNOWN_ISSUES §1.2(指向该分支)。**修订代理同时纠正了校验代理的证据**:该分支基点是 `6b0e02e` 而非 `b3ab747`,落后 HEAD 147 条,须前移并重新过门禁;且 `3ec2734` 未触 `frame_fsm/`,图路径快照坑仍开 | `git -C 工程线 log --oneline -1 3ec2734^` → `6b0e02e` |
| **D-092**(补录) | "§2.5 例外(**仅 phase-cd**)… ③ LS 后仍相交的二级回溯耗尽**抛** `std::runtime_error(\"mesh intersection persists after energy line-search backtracking…\")`" | 该 intersection-persists 抛错在稳定线**同样存在且消息文本逐字相同**(另有 type-0 变体)。该表三项中只有 ①(帧 0 `gipc::GeometryError` 类型化异常)与 ②(图内 `INV_LS_BUDGET`/starved-LS 重试)是 phase-cd 独有(稳定线 grep `GeometryError` 0 命中) | 稳定线 `StiffGIPC/GIPC.cu:15235-15242`、`:15101-15111`;phase-cd `core/ipc_solver.inl:855-862` | 稳定线用户以为此情形下只会 WARN 继续,**实际两线都会中止** | **未见修订记录 → §4.2 / U-05(仍存活于 `KNOWN_ISSUES.md:213` ③)** | `grep -n 'mesh intersection persists' 稳定线/StiffGIPC/GIPC.cu` |
| **D-053** | "**逐位(BITWISE):run-to-run 且跨架构**(已证 sm_80 ≡ sm_89,锚 `f7fb5a786c2d7935`)" | 引用了 **dlto 采纳前的陈旧锚**:当前发布面的 strict 不动点是 `0544461bd82123ae`(同文档 §2.1 自己写明,跨架构复验也是在新锚上做的)。且表格未带 §2.1 声明为必要条件的"同一 wheel/同一编译配置"前提。根因是 `mode_contract.h` 文本本身过时,手册照抄未校正 | `StiffGIPC/multienv/mode_contract.h:35`(陈旧锚);`docs/RELEASE_NOTES_v0.8.6-rc1.md:132-137` | 同一文档两处给出两个"已证锚";跨架构逐位承诺在表格里显得无条件 | API_EXECUTION §1.1(改用当前锚 + 同 wheel/同编译前提,并注明 `mode_contract.h:35` 是 pre-dlto) | `sed -n '35p' 工程线/StiffGIPC/multienv/mode_contract.h` |

### D 组:机制描述错误

| 编号 | 错误陈述(原文短引) | 正确事实 | 证据(file:line) | 影响面 | 已修入 | 验证方法 |
|---|---|---|---|---|---|---|
| **D-054** ×3 | "NaN line-search 缺口(非有限试探能量**被读作『非下降』而接受**——修复为继续回溯)" | **因果写反**:在回溯线搜索里"被读作非下降"意味着继续减半、不会被接受。真实缺陷是 NaN 与任何数比较皆为假,落到 `status 0 = 『接受为下降(accepted descent)』`——被读作**下降**而接受;改判 status 1 继续回溯才是**修复**行为 | `gipc_modules/14_energy_linesearch_solver.inl:52-57`(注释原文);`core/ipc_solver.inl:520`;`git show fa0cd63` | 按字面读,bug 行为与 fix 行为相同,**读者无法重建缺陷** | CHANGELOG_TIMELINE §2.3 | `sed -n '52,57p' 工程线/StiffGIPC/gipc_modules/14_energy_linesearch_solver.inl` |
| **D-055** ×2 | "`set_revolute_torque(idx, torque)` \| N·m,**加 `−τ·∂θ/∂q` 到驱动梯度**(无 Hessian)" | 与实现**相反**:代码注释明写 ext_torque 恰恰**不**作为梯度项施加(那条路径需要 `−τ·∂²θ/∂q²` Hessian 才能收敛);实际是把力矩折算成对称化轴 wrench 并入 **q_tilde 惯性预测**(经 `M⁻¹·wrench·dt²`,solve 期间恒定)。与同表 `set_prismatic_force` 是**同一机制**,却被写成两种 | `abd_system/abd_driving_joint.h:217-223`(NOT applied here as a gradient term);`cal_q_tilde.cu:78-117,196-215`;全树 grep `ext_torque` 无梯度装配端消费 | 因果后果不同:力矩在整个 solve 中**冻结于步首位形**、经动能项生效,而非每次梯度装配现算。手册描述的正是被代码**点名放弃**的方案 | API_CORE §4.5(改写为 q_tilde 机制) | `sed -n '217,223p' 工程线/StiffGIPC/abd_system/abd_driving_joint.h` |
| **D-056** ×2 | "自身不持数据,每次访问重新切片(numpy view),因此**拿到的数组自动反映最新 `step()`**,无须每帧重取" | 每次调 `BodyView.get_vertices()` 都触发一次**全新的 GPU→host 拷贝并分配新 numpy 数组**——返回值是**快照**;持有旧数组跨 step 读取只会看到陈旧数据(**静默错值**)。原 docstring 说的是 "you don't need to fetch a fresh **BodyView** each frame"(指 view 对象,不是数组) | `bindings/pystiffgipc.cu:777-782`;`stiff_physics/engine.py:143-147`、`:93-96` | **主语被换掉的过度承诺**:诱导用户缓存数组,读到陈旧数据且无任何报错。"numpy view"的说法同样误导——切片是对本次拷贝的 view,与 GPU 状态无关 | API_CORE §BodyView(改为"调用时快照;重新调用 accessor,勿缓存数组") | `sed -n '142,147p' 工程线/stiff_physics/engine.py` |
| **D-057** | "`set_prismatic_limit_barrier(...)` \| 求解器**保证开度 d 永不越过 cl**(力控下的硬不过冲)" | 实现对 barrier 自变量做了数值下限钳制 `gc = max(g, 1e-9)`,**越界侧(g≤0)能量/梯度均为有限值**——是一根钳平的极硬罚簧,不是发散到 +∞ 的真 barrier;关节坐标也没有 CCD/alpha 钳制。足够大的驱动刚度或外力做功可越过 cl 并被接受 | `abd_system/abd_driving_joint.h:790-793`;`sim_engine.h:77-80`(LS 预算耗尽仅 report non-descent) | **过度承诺**,且与手册 §7.3 自承"merged 模式 LS 耗尽时 WARN 后继续提交步"内部矛盾 | **未见修订记录 → §4.2 / U-16(仍存活于 `API_CORE.md:520`)** | `sed -n '788,795p' 工程线/StiffGIPC/abd_system/abd_driving_joint.h` |
| **D-058** ×2 | "**IRON LAW**:病态 env 中途被检疫为完全惰性(INERT),健康 env 继续跑"(模式表中**无任何前置条件**) | 检疫机制有可用性门 `perEnvIsolationLive()`:除 groups>1 与 `STIFF_PERENV_ALPHA` 外,**还要求 host 遥测路径开**(`env_newton_iter_cap>0` 或 `STIFF_PERENV_TELEM`)——而 isolated/strict 的旗标 bundle **都不含** `STIFF_PERENV_TELEM`,`env_newton_iter_cap` 默认 0。纯 `multienv_mode="isolated"` 默认走纯设备快路径(代码自述"**蓄意没有 NaN 防御**"),`quarantineEnv` 直接 return false,调用者保留 throw——**病态 env 一样杀死整个 batch,与 merged 无异** | `multienv/isolation.cu:78-85,112-115`、`isolation.cuh:8-13`;`sim_engine.h:76`(cap=0);`engine.py:195-206,387`(bundle 无 TELEM、`per_env_exit=False`) | **高危 + 前后矛盾**:手册自己的 §1.9/§3.3 写明了这个门,承诺表却无条件呈现铁律——**读者按表配置拿不到承诺** | PRINCIPLES_EXECUTION §3.1、API_EXECUTION §1.1(铁律行补前置条件 + "默认配置拿不到中途检疫"警示) | `sed -n '78,85p' 工程线/StiffGIPC/multienv/isolation.cu` |
| **D-059** | "整帧 CUDA Graph \| **一帧 = 一次 `cudaGraphLaunch`**(Newton/PCG/LS 全在图内条件节点)" | 只设 `STIFF_FRAME_GRAPH=1` 走的是**审计两图事务**:root 图发射 → 宿主边界 → terminal 图发射,一帧默认 **2 次 launch + 1 个宿主边界**(FrameStatus 默认值即 `graph_launches=2`/`host_boundaries=1`);单发射 full graph 需**手册未提的额外 knob** `STIFF_FRAME_FULL_GRAPH`,且 capture 不可用时自动回落两图事务;回退重试帧还会追加发射 | `frame_fsm/frame_transaction.cu:64-65,3991,4055,4583-4585,2581,2601`;`config/knob_registry.h:77` | 机制描述对应的是未出现在手册里的子模式;按此预期 launch 计数会对不上 | README(该行已在当前修订中为正确形态,见 §4.1 skip #7) | `sed -n '64,65p' 工程线/StiffGIPC/frame_fsm/frame_transaction.cu` → 默认 2/1 |
| **D-060** ×2 | "lagged 摩擦梯度是步内位移的函数,提交后位移为零 → `friction_lagged`/`total` **恒零**" | `total` 不恒零:`components=2` 时 `want_normal` 为真,getter 仍会重建 BVH/CP 并计算 barrier 梯度,**total 退化为 normal-only**(接触存在时非零)。恒零的只有 `friction_lagged`(components=1);稳定线 CHANGELOG 的原描述也是 "always returned zero **friction**" | `03_step_getters_export.inl:1697-1701,1712,1739-1746`;零因果链 `energy/16_friction.inl:595,643` + `gipc_modules/08_step_update_topology.inl:89,94` | 用 `total==0` 作为 bug 特征去验证的用户会**误判 phase-cd 已修复**;且与同文档 §2.1"仍受恒零 bug 影响"(受影响≠恒零)前后不一致 | README(当前为正确形态,见 §4.1 skip #6);PRINCIPLES_DYNAMICS §5.8 与 KNOWN_ISSUES 均已用"受影响"措辞。**CHANGELOG_TIMELINE §7 的转述待查 → §4.2 / U-10** | `sed -n '1739,1746p' 工程线/StiffGIPC/engine_modules/03_step_getters_export.inl` |
| **D-061** ×3 | "双关节剪切台(gel μ=1.0 / bar μ=0.6,per-body μ **几何平均后接触对有效 μ 由 bar 侧主导设定为 0.6**)" | 几何平均下 `μ_pair = √(1.0×0.6) ≈ 0.775 ≠ 0.6`,且**几何平均没有"由某侧主导"的性质**(那是 min 律)。实测 0.600 若为真,只能说明被测滑动界面的有效 μ 本身就是 0.6(两侧同为 0.6,或走地面摩擦路径——地面 μ 按 per-vertex `vert_mu_gd` 直取、不做混合)。原始出处只报了测量值,**这句 gloss 是手册自己发明的** | `energy/02_contact_energy_device.inl:384-396`(`return sqrt(vmu[a]*vmu[b])`);`energy/16_friction.inl:112-115`;稳定线 `CHANGELOG.md:20-24` | 照此机制做 per-body μ 标定,下游会**预测错 29%** | PRINCIPLES_CONTACT §5.1(删除"主导"说法,重写为组合律警示;疑点入 §9.6) | `sed -n '384,396p' 工程线/StiffGIPC/energy/02_contact_energy_device.inl` |
| **D-062** ×2 | "**CCD 认证步长**:每次 Newton 迭代只允许走 `α ≤ ACCD 认证值 × slackness` 的一段";"`STIFF_CCD_SLACK_A` \| ground **ACCD** slackness(η = 1−slack)" | 两种 slackness 语义被混为一谈:**ground 路径根本不走 ACCD、也不存在 η**——它是解析式 `α = slackness·(dist/coef)`,slack_a 是**直接乘子**;**self/swept** 路径才把 `η = 1−slackness` 作为 `CCDDistRatio` 喂进 ACCD **内部推进**,而 ACCD 返回的 toc **被原样使用、不再乘 slackness** | `gipc_modules/07_energy_alpha_reductions.inl:60-61`(ground,无 η)、`:130`(η 进 ACCD)、`:156-178`(toc 未再乘);`ACCD.cu:491-521` | 两条路径的安全边距语义相反,按错的那条估边距会失准 | PRINCIPLES_CONTACT §3.5 第 2 条(改写)。**§3.3 表述待查 → §4.2 / U-03** | `sed -n '58,62p;128,132p' 工程线/StiffGIPC/gipc_modules/07_energy_alpha_reductions.inl` |
| **D-063** | "§3.5 保证:线性化轨迹上**不发生穿越**"(未列 floor 例外) | `_ccd_final_alpha_combine` 在 refined 咨询分支执行 `alpha = max(min(temp_alpha, refined), cfl_floor)`——当 swept 精细 ACCD 认证值小于 `cfl_floor` 时,最终 α 被启发式的 `sqrt(dHat)` 尺度 floor **顶起、超过认证值**。代码注释原话:"As a FLOOR it **OVERRIDES** a tiny certified refined value … risks tunneling through thin geometry";而 swept 通道是"从 DCD 邻域之外快速逼近"的**唯一**认证者 | `gipc_modules/10_ccd_buildcp_quarantine.inl:49-53,65-72,61-64` | 对 swept-only 快速逼近对,存在**文档化的未认证位移窗口**;§3.3 如实写了穿薄风险,§3.5 的保证条款却未把它列入边界 | PRINCIPLES_CONTACT §3.5(边界列表新增"未认证位移窗口"条目) | `sed -n '49,72p' 工程线/StiffGIPC/gipc_modules/10_ccd_buildcp_quarantine.inl` |
| **D-064** | "`d ≤ 0` 属于几何不可行,由 fail-fast 机制处理"(**无限定,读作对所有接触对类型**) | 类型化的 d≤0 检测**只存在于地面路径**:`_gdCollapse` 仅由地面检测/地面 CCD 写入;自接触的 `_d_PP/_d_PE/_d_PT/_d_EE` 返回**平方**距离、构造上恒 ≥0,不存在对应机制——自接触的几何不可行表现为网格穿插(平方距离仍为正),而唯一能抓它的整网格相交复查 `isIntersected` **默认直接 return false** | `gipc_modules/06_kinetic_soft_ground.inl:33-39`;`07_energy_alpha_reductions.inl:52-56`;`mlbvh_modules/02_distances_dtypes.inl:1-29`;`14_energy_linesearch_solver.inl:547-560` | 读者以为 body-body 穿透也有类型化 fail-fast 兜底,**实际防线只是 CCD α + 势垒发散** | PRINCIPLES_CONTACT §1.1(限定为地面路径) | `sed -n '33,39p' 工程线/StiffGIPC/gipc_modules/06_kinetic_soft_ground.inl` |
| **D-065** | "buildCP 主流程:… 6. `snapshotDcdCcdPairs()` → `throwIfGroundDistanceInvalid()`"(写作**无条件**步骤) | 代码是 `if(!m_ls_defer_counts) throwIfGroundDistanceInvalid();`——在**默认 merged 设备 line-search 的 trial-defer 模式**下,buildCP 尾部**并不抛**;地面塌陷标志改由 line search 决策读回捎带、在消费端 `handleGroundCollapse` 处理(同一 trial、幂等) | `gipc_modules/10_ccd_buildcp_quarantine.inl:2122-2126`;`core/ipc_solver.inl:481-485` | 语义上防线仍在,但把**有条件**的 fail-fast 写成无条件步骤,会让读者在 defer 路径上错误预期同步抛错的时序 | PRINCIPLES_CONTACT §2.3 第 6 步(补门控说明) | `sed -n '2120,2128p' 工程线/StiffGIPC/gipc_modules/10_ccd_buildcp_quarantine.inl` |
| **D-066** | "地面轴恒按最坏 `surf_vertexNum` 烘焙(**无 OVF 位,截断不可检测**)" | 地面轴容量被烘焙在其**真实上界**(每个表面顶点至多一个地面配对,cap = `surf_vertexNum`),因此**截断在结构上不可能发生**——"无 OVF 位"的原因是**不需要**,而非存在一个检测不到的截断风险 | `gipc_modules/10_ccd_buildcp_quarantine.inl:2273,2548`;`frame_fsm/frame_status.cuh:46-50` | 现文读起来像一条潜伏危险,**方向反了** | **未见修订记录 → §4.2 / U-08(仍存活于 `KNOWN_ISSUES.md:363`、`PRINCIPLES_EXECUTION.md:997`)** | `sed -n '2273p;2548p' 工程线/StiffGIPC/gipc_modules/10_ccd_buildcp_quarantine.inl` |
| **D-067** | "`reset_transient_contact_state`「清零 current/lagged contact-pair host 镜像 + 摩擦快照」…「24 µm → 调用后 **6.7e-9 m**」" | **漏掉第三个被清零的状态:自适应 Kappa**。实现在清镜像与 `m_have_fric_snap` 之后还有 `g.Kappa = 0.0`(下个 solve 走 fresh-process 的 `suggestKappa`);`1bc13ef` 明确分层测得:只清 pair/摩擦镜像后**仍残留 0.9 µm**(kappa 携带上一 episode 接触历史),**清掉 kappa 才到 6.7e-9 m** | 稳定线 `sim_engine.cu:3620-3636`(注释给出 0.9 µm 实测依据);commit `1bc13ef` 正文 | **按手册配方复刻只能到 ~0.9 µm**,却把 6.7e-9 归因于错误的机制集;也削弱了 §1.4 的迁移对照(phase-cd teleport 同样不复位 Kappa,"语义近似"的差距被低估) | CHANGELOG_TIMELINE §1.4 与 §2.4(补 Kappa 归零 + 1bc13ef 分层实测 + phase-cd teleport 不复位的注) | `sed -n '3620,3636p' 稳定线/StiffGIPC/sim_engine.cu \| grep -n 'Kappa'` |
| **D-068** | "对不属于所选模式的旗标**写显式 `\"0\"` 覆盖**;解析器会回收自己设的键" | **解析器从不写 `"0"`**:`resolve_multienv_mode` 只对 bundle 旗 `setdefault "1"`,对不再属于所选模式的旗走 `_retract_our_flags` **直接 `del`**(用户改过的保留)。该说法照抄了 `mode_config.h:36-39` 的**陈旧注释**,而同仓 `mode_contract.h:47` 已明文纠正为 "the resolver retracts only its own flags"。`env_on` 值感知的真实理由是**用户手工 =0 覆盖** | `stiff_physics/engine.py:277-284,287-320`;`multienv/mode_contract.h:46-47` | 读者据此推断环境变量最终状态会全错 | API_EXECUTION §1.2 与 §1.4(改写 + 注明陈旧注释来源) | `grep -n '"0"' 工程线/stiff_physics/engine.py` — 仅命中别名表与注释 |
| **D-069** | "每次 **step/teleport/checkpoint** 入口再次断言(`_assert_process_mode_signature`)" | 实际调用点是 **finalize / step / launch_episode_async / prepare_gpu_rl / prepare_gpu_rl_episode** 五处;**teleport 与 checkpoint 入口都不调用它**。宣称的三个入口里只有 step 成立 | `stiff_physics/engine.py:989,1006,1169,1222,1234`;`save/load_checkpoint`(`:1396-1418`)只查 `self._finalized` | 读者以为跨模式误用会在 teleport/checkpoint 处被拦住,实际不会 | API_EXECUTION §1.3 | `grep -n '_assert_process_mode_signature()' 工程线/stiff_physics/engine.py` |
| **D-070** ×3 | "`STIFF_LOG_LEVEL>=1` 时打印 …" / 表行 "`STIFF_LOG_LEVEL=1` 【稳定线+phase-cd】" | **引擎中不存在名为 `STIFF_LOG_LEVEL` 的环境变量**(两树 `StiffGIPC/`、`stiff_physics/`、`bindings/` 全 grep 零命中,仅个别示例脚本 setdefault 后无人消费)。真实的门是全局 `g_gipc_log_level`,**默认值 1(即默认就打印)**,唯一写点是 `SimEngine::set_log_level` / `Engine.set_log_level`。语义与"需开环境变量才有"**相反**;更糟的是它不在 `knob_registry.h` 内,`STIFF_KNOB_STRICT=1` 下设置它会直接抛 `ConfigurationError` | `gipc_modules/00_prelude_common.inl:151`(默认 1);`engine_modules/00_impl_api_surface.inl:325-327`(唯一写点);`core/ipc_solver.inl:2477-2478`;`knob_registry.h:208-231` | 监控建议引用了一个**不存在的旋钮名**,且在手册自己推荐的 strict 跑法下会抛错 | KNOWN_ISSUES §2.4、PRINCIPLES_EXECUTION §8 表(改写为"非旋钮"+ 死旋钮 + strict 下抛错)、PRINCIPLES_CONTACT §8.2 | `grep -rn 'STIFF_LOG_LEVEL' 工程线/StiffGIPC 工程线/stiff_physics 工程线/bindings` → 零命中 |
| **D-071** | "**小场景 GUI 与抓取场景普遍加 `STIFF_SKIP_CCD_SANITY=1`**" | **死旋钮当标准做法推荐**:两线的 line-search 尾部 CCD sanity 复检**早已默认跳过**,代码中不再读取该名(唯一 getenv 是反向的 `GIPC_FORCE_CCD_SANITY`;示例 docstring 里的写法是历史残留)。设了纯属无效;且它不在 phase-cd 登记表内,finalize 时会触发手册 §2.2 自己描述的 unknown-knob WARN,`STIFF_KNOB_STRICT=1` 下直接 `ConfigurationError` | `gipc_modules/14_energy_linesearch_solver.inl:556-558,568-574`;稳定线 `GIPC.cu:14894-14899`;`knob_registry.h` grep 无该行 | §6 的建议与 §2.2 的旋钮治理叙述**自相矛盾**(顺带:同段推荐的 `STIFF_BENCH_STATS` 有效但同样未登记,也会 WARN) | README §6(改写为历史 no-op + 未登记后果;`STIFF_BENCH_STATS` 标"有效但未登记") | `grep -rn 'STIFF_SKIP_CCD_SANITY' 工程线/StiffGIPC` → 零命中;`grep -n 'GIPC_FORCE_CCD_SANITY' 工程线/StiffGIPC/gipc_modules/14_energy_linesearch_solver.inl` |
| **D-072** | "towel **82× 异常 = 重录风暴**" | **过度归因,且后一行已推翻**:源文档的 C6-n 解剖只把 468 s 中的 **219 s** 归给 19 个重录风暴帧(其余为容量宽度固定成本);C6-o 的 nsys 归因明确"**推翻重录假说**"——two-graph towel 全程**零** capture/instantiate 调用,真因是 OVF 已标记而宿主帧末才看,导致空转烧 150k PCG 迭代 | `docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:35-37` 与 `:50-53` | 用 "=" 把 82× 全部归给重录风暴,并在 C6-o 行删去"假说被推翻"的关键转折,读者得到**错误的根因模型** | CHANGELOG_TIMELINE §3.8(给出三层分解 + C6-o 行恢复 nsys 推翻的转折) | `sed -n '35,37p;50,53p' 工程线/docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md` |
| **D-073** | "地面摩擦 `E = λ‖v‖²/(2ε)`(**C0 头**,与自碰的 C1 f0 不同——代码事实)" | 二次头在 `‖v‖=ε` 处与滑动支的**值**(λε/2 对 λ(ε−ε/2))和**一阶导**(λ 对 λ)都连续 → 同属 **C1**(只是不 C2)。两条摩擦路径的真实差异是 **f0 的函数形式**(自碰是三次复合式,地面是纯二次头),而非连续性阶数 | `energy/02_contact_energy_device.inl:285-293`;自碰 `FrictionUtils.cuh:435-438` | "C0 头"会误导读者以为地面摩擦能量在静滑转换处有**导数跳变** | PRINCIPLES_DYNAMICS §摩擦(改为 C1 + 点明真实差异) | `sed -n '285,293p' 工程线/StiffGIPC/energy/02_contact_energy_device.inl`,在 ‖v‖=ε 处手算两支值与斜率 |
| **D-074** | "⚠ 调试标签错位(仅 `STIFF_ENERGY_VALIDATE` 的 `kSlotNames` 打印)" | 同一套互换标签还**独立复制**在手册力荐的 `STIFF_LSX_DIAG` 逐槽剖析里:`core/ipc_solver.inl` 的 `kN[15]` 同样把索引 5 写成 "ground"、6 写成 "barrier",而真实写入是 type 2(自碰 barrier)→slot 5、type 4(ground)→slot 6。手册**没有任何指针**提示 lsx-diag 输出也需按 §1.3 纠正 | `core/ipc_solver.inl:750-752,757-760`;真实映射 `energy/01_energy_host_dispatch.inl:317-318,296-298` | 按 §4.6 用 lsx-diag 追线搜索耗尽的读者会把**自碰 barrier 泄漏误读成地面项**(或反之),恰好毁掉该诊断流程的结论 | PRINCIPLES_DYNAMICS §1.3、§4.6、附录 C 第 4 条(三处均加警告)。**Fable 5.1 复验(2026-09-08)**:已确认为**纯标签缺陷、物理正确**——自碰核 `02_contact_energy_device.inl:35-39` 内部已乘 Kappa 后写 slot 5、combine 裸加;地面核 `energy/17_ground.inl:36` 能量式不含 Kappa 写 slot 6、combine 乘一次 Kappa。两处打印器(`ipc_solver.inl:750-754` kN、`01_energy_host_dispatch.inl:367` kSlotNames)都把 5/6 标反,属**代码级诊断标签 bug**,建议一并修 | `sed -n '750,754p' 工程线/StiffGIPC/core/ipc_solver.inl` |
| **D-075** | "变了 → `runtime_error(\"graph buffers changed; **prepare the episode again**\")`(`frame_transaction.cu:4433` 附近)" | `:4433` 附近是 `frame_graph_device_state()`/tier 复位代码,**与代际检查无关**;§3.2 语境(gpu_rl 单帧 launch)的代际检查在 **`:3235-3237`**,报错文案是 "prepare the **graph** again";引用的 "episode" 文案属于 episode 发射路径(`:3185-3187`、`:3302-3304`) | `frame_fsm/frame_transaction.cu:3235-3237,3185-3187,3302-3304,4425-4440` | 行号与文案双错,按图索骥落在无关代码上 | API_EXECUTION §3.2 | `sed -n '3235,3237p' 工程线/StiffGIPC/frame_fsm/frame_transaction.cu` |

### E 组:引用与链接错误

| 编号 | 错误陈述(原文短引) | 正确事实 | 证据(file:line) | 影响面 | 已修入 | 验证方法 |
|---|---|---|---|---|---|---|
| **D-076** ×2 | "`../docs/A800_ALLEXAMPLES_TIMING_2026-08-01.md:358-364`" | 手册位于 `docs/manual/`,`../docs/…` 解析为**不存在的** `docs/docs/…`;正确相对路径是 `../A800_ALLEXAMPLES_TIMING_2026-08-01.md`(与同文件里其它 `../OPTIMIZATION_ROADMAP.md` 引用风格一致)。行号与数字本身正确 | `docs/` 下无 `docs/` 子目录 | 死链 | README(当前为正确形态,见 §4.1 skip #5) | `ls 工程线/docs/docs` → 不存在 |
| **D-077** | "见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)"(以及 §5 分册导读表列出的 7 个分册) | **提出时** `docs/manual/` 只有 `README.md`,7 个分册全部尚不存在——包括用来背书 phase-cd 摩擦恒零警告的 KNOWN_ISSUES.md | 提出时 `ls docs/manual/` → 仅 README.md | 关键警告的背书指向空文件 | **该条已失效并被跳过**:分册现已全部就位,`KNOWN_ISSUES.md` 427 行且覆盖该议题(见 §4.1 skip #10) | `ls 工程线/docs/manual/` → 9 个 .md |
| **D-078** | "姊妹分册:[多环境三模式](MULTIENV_MODES.md) · [整帧图与 GPU 驻留 RL](FRAME_GRAPH_RL.md) · [接触/CCD/摩擦](CONTACT_CCD.md)";"§4.1 机制链详见 FRAME_GRAPH_RL.md" | 三个链接目标在 `docs/manual/` 中**均不存在**(实际分册名不同);§4.1 把关键机制说明委托给不存在的文件。另 §5.5 引用的 `STIFF_PHYSICS_RELEASE_HANDBOOK.md` 在**仓库根目录**,裸相对名从 `docs/manual/` 解析不到 | `ls docs/manual/`;`STIFF_PHYSICS_RELEASE_HANDBOOK.md` 在仓库根 | 死链 + 机制说明落空 | **未见修订记录,但已自然消失**(当前 grep 零命中)→ §4.2 / U-06,标"已不存活,无需处理,仅备查" | `grep -rn 'MULTIENV_MODES.md\|FRAME_GRAPH_RL.md\|CONTACT_CCD.md' 工程线/docs/manual/` → 零命中 |
| **D-079** ×2 | "铰接一帧图 **551 节点**(`GPU_NATIVE_RL_PLAN.md:53-128`)" | `551` 在 `GPU_NATIVE_RL_PLAN.md` 全文 **grep 零命中**;真实出处是 `PHASE_C_FRAME_GRAPH_PLAN.md:248,312`。同行其余数字(1057/5655/2.461ms/11.571851ms 等)在所引文件内核实无误 | `docs/PHASE_C_FRAME_GRAPH_PLAN.md:248,312` | 引用锚错位,读者去查会一无所获 | CHANGELOG_TIMELINE §3.9 | `grep -n '551' 工程线/docs/GPU_NATIVE_RL_PLAN.md` → 零命中 |
| **D-080** | "验收面:12 段武装门禁全绿;**demo_verify 19-run**;**owner 13-demo UI 走查**(`RELEASE_NOTES_v0.8.6-rc1.md:7-27`)" | `:7-27` **不含**后两项——两者在 `RN:44`;`:7-27` 只覆盖重构阶段/结构疫苗/12 段门禁 | `docs/RELEASE_NOTES_v0.8.6-rc1.md:44` | 引用行号范围与所列内容不符 | CHANGELOG_TIMELINE §2.5(拆分为 `:7-27` + `:44`) | `sed -n '44p' 工程线/docs/RELEASE_NOTES_v0.8.6-rc1.md` |
| **D-081** | "重接触场景推荐 `M=0.9, CFL=1.0`(Newton **−7.9%**;**OPTIMIZATION_ROADMAP §α**)" | 该组配置与数字**不在** `OPTIMIZATION_ROADMAP.md`,而在 `SIMULATOR_EXECUTION_DESIGN.md:228-229`(及 244-245)。ROADMAP §2 的"α 配方 research 档"是**另一档**(未上线,"实测 −16% Newton"),全文不含 −7.9% / M=0.9 / CFL=1.0 | `docs/SIMULATOR_EXECUTION_DESIGN.md:228-229,244-245`;`docs/OPTIMIZATION_ROADMAP.md:34` | 读者按引用去 ROADMAP 会见到 −16%,**误判本表数字有错** | PRINCIPLES_DYNAMICS §旋钮表(改引用 + 加"勿与 research 档混"注) | `grep -n '7\.9%' 工程线/docs/*.md` → 仅 `SIMULATOR_EXECUTION_DESIGN.md` |
| **D-082** | "特征值 clamp:phase-cd `1e-6·λmax` / 稳定线 `cwiseMax(0.0)`(**setup 977-1017**;`abd_system.cu:998-1001`)" | 第一个锚点**文件名写串**:`setup_abd_system_gradient_and_hessian.cu:977-1017` 是 **stitch 交叉 Hessian 的 triplet 写块**,与 PSD 强制毫无关系。两处 clamp 都在 `abd_system.cu`:phase-cd `:998-1001`(第二个锚点正确)、稳定线 `:981`。"977-1017"恰与稳定线 `abd_system.cu` 该区段吻合 | phase-cd `abd_system/abd_system.cu:998-1001`;稳定线 `abd_system/abd_system.cu:981` | 按锚跳行落在无关代码上;**内容论断本身为真** | PRINCIPLES_DYNAMICS §ABD(锚点更正) | `sed -n '998,1001p' 工程线/StiffGIPC/abd_system/abd_system.cu` |
| **D-083** ×2 | "单线程组合核 `_global_energy_combine`(稳定线同名核在 **GIPC.cu:14314ff**)" | 稳定线该核定义于 **`:14283`**(发射于 `:14330`);**`:14314` 是 `GIPC::computeEnergy_DeviceOut` 的定义**,不是同名核。偏差 31 行,超出手册自称的"个位数偏移"容差,且落在**不同符号**上 | 稳定线 `StiffGIPC/GIPC.cu:14283,14330,14314` | 按符号名检索会发现引用对象不符 | PRINCIPLES_DYNAMICS(锚点更正为 `:14283` 定义 / `:14330` 发射) | `grep -n '_global_energy_combine' 稳定线/StiffGIPC/GIPC.cu` → `14283` |
| **D-084** | "merged 契约明文『无可复现承诺』(`multienv/mode_contract.h:**23**`)" | 该行在**第 22 行**;第 23 行是 "pinned by: kick + foldshirt-smoke behavioral gates + G9 envelope" | `StiffGIPC/multienv/mode_contract.h:22` | 偏一行 | KNOWN_ISSUES §2.6 | `grep -n 'NO reproducibility promise' 工程线/StiffGIPC/multienv/mode_contract.h` |
| **D-085** | "生产推荐(稳定线 `CHANGELOG.md:**94-98**`):`per_env_exit=True, env_newton_iter_cap=100`" | 该推荐位于 **`:141-145`**(0.8.5 节的 "Recommended stability configuration");`:90-98` 是 0.8.5.1 节的 `_calBarrierGradientAndHessian` hardening,与此无关 | 稳定线 `CHANGELOG.md:141-142` | 行号错,跳过去看到不相干内容 | API_EXECUTION §1.7 | `sed -n '141,145p' 稳定线/CHANGELOG.md` |
| **D-086** | "PE(点-边)形编码 … 出处 `03:167`" | `mlbvh_modules/03_pair_emission.inl:167` 是 `if(g_ee_detgate){ … }`(**确定性发射门**的距离计算),不是 PE 发射;PE 形编码的实际发射行是 `03:276/283`(dtype2/3)与 `03:390/397`(dtype4/5),判别式 `w<0` 分支在 `energy/02_contact_energy_device.inl:167`——疑为把 `02:167` 误写成 `03:167` | `mlbvh_modules/03_pair_emission.inl:167,276,283,390,397`;`energy/02_contact_energy_device.inl:167` | 编码格式的权威锚指向无关代码 | PRINCIPLES_CONTACT §配对编码表 | `sed -n '167p;276p;283p' 工程线/StiffGIPC/mlbvh_modules/03_pair_emission.inl` — `:167` 为 detgate 行 |
| **D-087** | "(实现 `stable:sim_engine.cu:3620-3636`;pybind `bindings/pystiffgipc.cu:422`;Python `engine.py:941-952`)" | 后两个引用**未加 `stable:` 前缀**,而手册自订约定是"未注明的 `文件:行号` 均指 phase-cd 树"——但该 API **仅稳定线**有:phase-cd 全树无 `reset_transient_contact_state`,其 `engine.py:941-952` 处是 `add_prismatic_joint` 的 docstring | phase-cd `stiff_physics/engine.py:938-955` = `add_prismatic_joint`;稳定线 `engine.py:941-952` 确为该 API | 按手册自己的规则解析,两个锚点落在**错误的树、错误的代码**上 | **未见修订记录 → §4.2 / U-02(仍存活于 `PRINCIPLES_CONTACT.md:594`)** | `sed -n '938,955p' 工程线/stiff_physics/engine.py` |
| **D-088** | "`CASE39ME_TRAJ_GLOB` / `_TRAJ_JITTER` / `_PIN_ENV0`(§9.2 表,标题『正主 `examples/replay_foldshirt_multienv.py`』)" | 这三个旋钮**不被** `replay_foldshirt_multienv.py` 消费(grep 0 命中),只在 finray 族的 `umi_finray_lib.py` 里读 | `examples/umi_finray_lib.py:769,784,787` | 用户在该脚本上设了**静默无效** | API_EXECUTION §9.2(标注为 finray 族专用) | `grep -n 'CASE39ME_TRAJ_GLOB' 工程线/examples/replay_foldshirt_multienv.py` → 零命中 |
| **D-089** | "`_print_bvh_coherence_audit()` \| **`STIFF_BVH_COHERENCE_AUDIT`** 构建" | CMake 选项实为 **`STIFFGIPC_BVH_COHERENCE_AUDIT`**(宏为 `STIFF_BVH_COHERENCE_AUDIT_BUILD`),按手册写 `-DSTIFF_BVH_COHERENCE_AUDIT=ON` **无效**;同表首行给的却是精确 CMake 旗。另:该节标题称"条件编译",但 traversal-audit 三件套的 pybind 绑定**无任何 `#ifdef` 守卫**(普通构建即可调用,只是计数为空),真正被守卫的只有 `_print_bvh_coherence_audit` 一个 | `CMakeLists.txt:49,174-175,47,153-154`;`bindings/pystiffgipc.cu:1131-1149`(无 ifdef)vs `:1167-1171`(有) | 照抄的构建命令是**无效开关** | API_CORE §10.7(开关名与宏名更正 + ifdef 守卫范围说明) | `grep -n 'BVH_COHERENCE_AUDIT' 工程线/CMakeLists.txt` |
| **D-090** | 版本自检片段 `importlib.metadata.version("stiff-physics")` | 在**手册自己文档化的 phase-cd 工作流**下会抛 `PackageNotFoundError`:phase-cd 被定为"仅源码构建"(§2.2),标准跑法是 `PYTHONPATH=. python …`,此时没有安装任何 stiff-physics 发行版。版本字符串本身(0.8.5.3 / 0.8.6rc2)属实;紧随其后的 `hasattr` 探测才是两线都可靠的方法 | `pyproject.toml:7`(仅 pip 安装后产生 dist-info);手册 §3.2 构建流程只有 cmake | 读者照抄自检片段会拿到异常而非版本号 | README(补前提标注 + 指向 hasattr 探测) | `cd 工程线 && PYTHONPATH=. python -c 'import importlib.metadata as m; print(m.version("stiff-physics"))'` — 未 pip 安装时抛 `PackageNotFoundError` |
| **D-091** | "`04_binned_grad_fused_assembly.inl`…"(隐含仍在 `gipc_modules/`) | 该文件在 E1d(`7983be1`)已移入 `energy/`,`GIPC.cu` 的对应位置由 `energy/03_barrier_fused_assembly.inl` 顶替 | `git log --diff-filter=D -- 'StiffGIPC/gipc_modules/04*'` → `7983be1`;`GIPC.cu:21` | 与 D-005 同源(重复计数的根因) | CHANGELOG_TIMELINE §1.1(与 D-005 同一处修订) | `git -C 工程线 log --oneline --diff-filter=D -- 'StiffGIPC/gipc_modules/04*'` |

---

## 2. 最有价值的五条

以下五条不是"最严重",而是**最容易被复制传播、且错了之后最难自查**的五条。

### 2.1 D-001 —— `FrameStatus` 到底有几个字段(40 vs 39)

**为什么容易错**:`frame_status.cuh` 的 C++ struct 与 pybind 绑定块**不是一一对应**的。struct 里有 `contact_class_count[4]` 这样的数组成员,pybind 没有绑定它(数组要额外包装)。数手册字段清单、数 struct 成员、数绑定块——三条路会给出三个数。手册当时数的是 struct 侧,写了 40;Python 侧实际只暴露 39。更巧的是,手册 §7.6 罗列的字段名**恰好也是 39 个**(它漏列了 `launch_status`),两个 39 撞在一起,反而掩盖了"清单本身也有漏"这件事。

**错了会怎样**:写遥测采集器时按 40 个字段做定长断言或 `zip` 配对,要么静默错位,要么在 CI 里报一个查不出来源的形状不匹配。更隐蔽的是:`contact_class_count` 是**真实存在但拿不到**的诊断量——手册写 40 会让人以为它可以从 Python 读,于是花时间去找一个不存在的属性。

**怎么自查**:唯一可信的口径是**绑定块本身**,不是 struct、不是文档清单:

```bash
awk '/py::class_<frame_fsm::FrameStatus>/,/py::class_<SimEngineConfig>/' \
  /home/ps/Downloads/Stiff-GIPC-c1-ls-graph/bindings/pystiffgipc.cu | grep -c def_readonly
# → 39
```

在 Python 侧的等价自查(更贴近实际用法):`len([a for a in dir(st) if not a.startswith('_')])`,再与上面的数对齐。若哪天有人补绑了 `contact_class_count`,两个数会同步变——这正是应该用绑定块而不是文档做真源的理由。

### 2.2 D-002 —— 旋钮登记表是 170 条,不是 ~150

**为什么容易错**:`~150` 是个**约数**,写的时候没人会去精确 grep;而 X-macro 注册表(`STIFF_KNOB_REGISTRY(X)`)是**持续增长**的——每个新旋钮加一行,数字每周都在变。手册里同一个 "~150" 出现在 §6.2 和 §11.2 两处,一次估错就复制了两遍。更糟的是历史包袱:CHANGELOG 里还留着 G14 时点的 "84+17+11"(见 D-003,那本身也是错的),于是文档里同时存在三个互不相干的数量口径。

**错了会怎样**:低估 13% 本身危害有限,真正的问题是**它削弱了"登记表是唯一真源"这个论断的可信度**。旋钮治理机制的全部价值就在于"没登记的会被抓出来";读者如果发现文档说 ~150 而实际 170,自然会怀疑"那是不是还有一批旋钮压根没进表"。而事实恰恰相反——170 是**完整**的,`STIFF_KNOB_STRICT=1` 会把任何未登记的 `STIFF_*` 升级成 `ConfigurationError`。一个约数动摇了一个强保证。

**怎么自查**:

```bash
grep -cE '^\s*X\(STIFF_' /home/ps/Downloads/Stiff-GIPC-c1-ls-graph/StiffGIPC/config/knob_registry.h
# → 170  (另有元旋钮 STIFF_KNOB_STRICT 不在表内,单独处理)
```

分类计数(audit 2 / diag 80 / mode_isolated 7 / mode_strict 4 / perf 55 / python 3 / solver 19)可用同一 grep 加类别字段统计。**引用这个数时请连同 commit 一起写**,否则下周就又错了。

### 2.3 D-031 —— `cloth_density` 是体密度,填面密度会错 1000 倍

**为什么容易错**:名字里没有 "volumetric",默认值 `2e2` 看着也不像 kg/m³(布料给人的直觉是 kg/m²);手册原文甚至**同时**写了"面密度"和含厚度 h 的公式 `m += ρ·area·h/3`——公式里有 h,就说明 ρ 乘的是体积,两句话自己打自己。而单位栏还留着 "—",等于放弃了唯一能纠正直觉的地方。真正的陷阱在装配顺序:

```
01_config_upload.inl:91   area *= ipc.clothThickness   // 面积 → 体积
01_config_upload.inl:99   masses += tri_rho * area / 3 // ρ × 体积
```

`area` 这个变量名在第 91 行之后**已经是体积了**,读代码的人若只看第 99 行,会以为 ρ 乘的是面积。

**错了会怎样**:织物的常用面密度是 ~0.2 kg/m²。用户按手册标的"面密度"填 `cloth_density=0.2`,而引擎把它当 kg/m³ 用,再乘一次厚度 h=1e-3 —— 质量**轻 1000 倍**。后果不是报错,是**静默的错物理**:布料飘得像烟,或者在接触里被推得飞出去;调试者会去怀疑刚度、阻尼、dt、接触参数,唯独不会怀疑密度,因为它"填的是标准值"。默认值 `2e2 kg/m³` 在 h=1mm 下对应 0.2 kg/m² —— 默认是对的,**只有主动改参数的人会掉进去**,而这些人恰恰是最相信文档的人。

**怎么自查**:两步。① 读装配顺序,确认 `area` 在乘 ρ 之前已被乘过厚度:

```bash
sed -n '88,102p' /home/ps/Downloads/Stiff-GIPC-c1-ls-graph/StiffGIPC/engine_modules/01_config_upload.inl
```

② 更硬的验证:建一块已知面积 A、厚度 h 的布,`finalize()` 后读回总质量,核对 `m == ρ·A·h`。若你填的是面密度,读回的质量会比预期小 1/h 倍(h=1e-3 时即 1000 倍)。**任何时候改 `cloth_density`,都值得跑一次这个质量回读。**

### 2.4 D-028 —— `GRIP_TARGET` 的 `−f·dt²` 遗留单位

**为什么容易错**:这是全套单位陷阱里**伪装得最好**的一条,因为**四层都在骗你**:①示例代码 `umi_finray_lib.py` 的注释写着 "(N)";②手册照抄了这个注释;③默认值 `0.03` 看起来完全像一个合理的小握力(30 mN);④比较的那一侧 `get_body_contact_force` 名字里带 "force"。但引擎的梯度空间约定是 `dE/dx = −F·dt²`,而 **`−1/dt²` 换算只在 `get_vertex_contact_forces` 一条路径上做过**(`03_step_getters_export.inl:1758` 的 `neg_inv_dt2`)。`get_body_contact_force(_batched)` 是 pre-0.8.4 的 legacy getter,返回**裸梯度**,docstring 里白纸黑字写着 "NOT Newtons"——但示例代码的注释盖过了 docstring。

**错了会怎样**:foldshirt 家族默认 `dt=0.02`,`dt² = 4e-4`。阈值 `0.03` 在梯度空间对应的物理力约 **75 N** —— 不是 0.03 N,差 **2500 倍**。这意味着:①任何"按牛顿设握力上限"的尝试都会离谱地失效(设 5 N 实际是 12500 N,或反过来永远触发不了);②**更阴险的是 dt 依赖**——同一个 `GRIP_TARGET=0.03`,dt 从 0.02 改到 0.01,物理阈值会变成 1/4,抓取行为**静默改变**,而调参者以为自己只动了时间步。这条把"单位错"和"参数随 dt 漂移"两个坑叠在了一起。

**怎么自查**:

```bash
grep -n 'NOT Newtons' /home/ps/Downloads/Stiff-GIPC-c1-ls-graph/stiff_physics/engine.py
# engine.py:1591-1601 / 1783-1793 —— docstring 是权威,示例注释不是
```

判定规则一句话:**只有 `get_vertex_contact_forces` 与 `get_contacts_device` 返回牛顿,其余接触力 getter 一律是 `−F·dt²`**(前者在 `03_step_getters_export.inl:1758` 乘 `neg_inv_dt2`,后者在 `GIPC::exportContacts`/`12_host_wrappers_fem.inl:445` 乘 `inv_dt2`;`get_contacts_device` 只覆盖 barrier 接触对,不含摩擦)。要换算就自己乘 `−1/dt²`。要验证某个阈值到底是多少牛顿:同一帧里同时读 `get_body_contact_force`(裸梯度)与 `get_vertex_contact_forces`(牛顿)在同一批顶点上的和,比值应恒为 `−dt²`;比值不是 `−dt²`,说明你读错了 getter。

### 2.5 D-037 —— 稳定线 HEAD 实为 v0.8.5.4,工作树内容却是 v0.8.5.3

**为什么容易错**:这是**整轮审计里最反直觉的一条**,三个校验代理各自撞上,给出了**三种互不相同的错误描述**——足见其迷惑性。真实状态是一个罕见的三重错位:

| 问什么 | 答案 |
|---|---|
| 分支 HEAD 在哪? | `c0339c8` |
| `c0339c8` 是什么? | **就是 tag `v0.8.5.4` 本身**(`git describe --tags --long` = `v0.8.5.4-0-gc0339c8`) |
| 那 "release(v0.8.5.4)" 的提交呢? | `0894958`,是 `c0339c8` 的**父**提交 |
| 磁盘上的文件内容是哪个版本? | **编制时 v0.8.5.3**——8 个文件有未提交的回退改动,`git diff v0.8.5.3` 为空;**2026-09-08 起 v0.8.5.4**(owner 确认该回退非本意,已 `reset --hard`,补丁备份留存) |

于是:tag 名比 release 提交**晚一个**,工作树内容比 HEAD **早一个版本**,而 `git log` 和 `ls` 会告诉你两件矛盾的事。校验代理们分别错在:"c0339c8 是 v0.8.5.4 之后一提交"(错,它就是 tag)、"工作树是 v0.8.5.4 内容、行号有偏移"(错,内容是 v0.8.5.3、偏移为零)。

**错了会怎样**:两条独立的伤害。①**行号全体失准或全体准确,取决于你信哪个描述**——手册所有 `stable:GIPC.cu:NNNN` 锚点是按 `b8e27a1`(v0.8.5.3)写的,而工作树内容恰好就是它,所以**偏移实为零**;手册却自我声明"可能有个位数偏移",让读者对准确的锚点产生不必要的怀疑。②**远比行号严重的是脆弱性**(历史状态,2026-09-08 已消除):v0.8.5.3 的内容只靠 8 个**未提交**的回退改动维持。任何人执行 `git checkout .`、`git stash`、`git reset --hard`,甚至一次不小心的 IDE "discard changes",这棵"稳定线"就**静默变成 v0.8.5.4**——一个默认开启真静摩擦(`absolute_epsv` + 持久摩擦锚)、摩擦行为不同的版本。此后所有基于"稳定线"的对照实验都在测另一个东西,而且**没有任何报错**。

**怎么自查**(2026-09-08 起的正确状态):每次动稳定仓之前跑这三行:

```bash
ST=/home/ps/Downloads/Stiff-GIPC-stable-08
git -C $ST describe --tags --long      # → v0.8.5.4-0-gc0339c8   (HEAD 就是 tag)
git -C $ST status --short              # → 空                   (工作树 == HEAD,无残留暂存)
git -C $ST diff v0.8.5.3 --stat        # → 8 files, +511/-8     (工作树是 v0.8.5.4 内容)
```

第二行有输出,说明又有人往工作树塞了未提交改动。**复核手册的 v0.8.5.3 行号一律用 `git show v0.8.5.3:<文件> | sed -n 'N p'`,永远不要按磁盘行号**——这样引用锚就钉在不可变的 tag 上,与工作树状态无关。

---

## 3.(编号索引)

按被修订的文档归档,便于逐册复核:

| 文档 | 本台账条目 |
|---|---|
| `README.md` | D-001, D-002, D-004, D-009, D-071, D-090(+ skip: D-010, D-005, D-036, D-048, D-059, D-060, D-076, D-077) |
| `API_CORE.md` | D-001, D-002, D-029, D-031, D-037, D-055, D-056, D-089 |
| `API_EXECUTION.md` | D-004, D-008, D-015, D-028, D-043, D-044, D-053, D-058, D-068, D-069, D-075, D-085, D-088, D-010 |
| `PRINCIPLES_CONTACT.md` | D-022, D-023, D-033, D-045, D-047, D-061, D-062, D-063, D-064, D-065, D-070, D-086 |
| `PRINCIPLES_DYNAMICS.md` | D-032, D-034, D-037, D-073, D-074, D-081, D-082, D-083 |
| `PRINCIPLES_EXECUTION.md` | D-014, D-018, D-019, D-020, D-042, D-046, D-050, D-058, D-070 |
| `KNOWN_ISSUES.md` | D-010, D-021, D-026, D-035, D-052, D-070, D-084(+ 未修:D-049, D-066, D-092) |
| `CHANGELOG_TIMELINE.md` | D-005, D-006, D-007, D-013, D-016, D-017, D-025, D-038, D-039, D-040, D-041, D-054, D-067, D-072, D-079, D-080, D-091 |

---

## 4. 被判定不成立或未获修订确认的条目

### 4.1 修订代理明确跳过的 11 条(附跳过理由,**待人类复议**)

全部 8 个修订代理中,只有 2 个报告了跳过;其余 6 个(API_CORE / PRINCIPLES_DYNAMICS / KNOWN_ISSUES / PRINCIPLES_EXECUTION / API_EXECUTION / PRINCIPLES_CONTACT)**逐条复核后全部确认成立,零跳过**。

**README.md 修订代理跳过 10 条**,其中 9 条同一理由 + 1 条独立理由:

| # | 被跳过的 issue | 跳过理由(修订代理原述) | 复议要点 |
|---|---|---|---|
| 1 | 图开 +10~15% 引用与出处不符 | "已在当前 README 修订中为正确形态(README:70/74/93/104-106/148);校验清单显然是针对更早的修订起草的,逐条核对当前段落均已符合 issue 要求的修正" | 该缺陷在**别的分册确实成立并已修**(见 D-010)。README 侧需人工确认"当前形态"是否真的正确,而非代理误判 |
| 2 | 图开 +10~15%(第二位校验者独立提出) | 同上 | 同上 |
| 3 | `gipc_modules` 15 → 14 | 同上 | 同 D-005:该缺陷在 CHANGELOG_TIMELINE 确实成立并已修 |
| 4 | `gipc_modules` 15 → 14(第二位) | 同上 | 同上 |
| 5 | `../docs/A800_…` 路径解析为 `docs/docs/` | 同上 | 同 D-076 |
| 6 | **跨架构逐位确定性归属(唯一 high 级跳过条目)** | 同上 | **优先复议**:这是本轮唯一被跳过的 high;若 README 当前形态未真正收窄,稳定线用户仍会被误导(见 D-036) |
| 7 | "total 恒零"措辞 | 同上 | 同 D-060 |
| 8 | 两图事务 vs 整帧图机制 | 同上 | 同 D-059 |
| 9 | `reset_transient_contact_state` 非等效 | 同上 | 同 D-048 |
| 10 | `KNOWN_ISSUES.md` 链接指向不存在的文件 | "**该条已不再成立**:`docs/manual/` 现已包含全部 7 个分册,`KNOWN_ISSUES.md` 有 427 行且覆盖了 phase-cd 摩擦恒零议题" | 已亲验:`ls docs/manual/` 现有 9 个 .md。**该跳过成立** |

**CHANGELOG_TIMELINE.md 修订代理跳过 1 条**:

| # | 被跳过的 issue | 跳过理由(修订代理原述) | 复议要点 |
|---|---|---|---|
| 11 | "beaker 破案:差 = 纯启动段 **~0.3–0.6 s**"中的下限 0.3 无出处、区间系拼造 | "**该指控本身是错的**:`SIMULATOR_EXECUTION_DESIGN.md:167`(附录 A beaker 行)写着『+4%(混沌带内,启动 +0.3s)』;校验代理只看了 `:123` 和 A800 文档。作为附带修正(不计入 fix),脚注现已改引 `:167`,使区间自洽" | **这是校验代理自己查漏出处的一例**。复议只需 `sed -n '167p' docs/SIMULATOR_EXECUTION_DESIGN.md` 确认该行存在 |

### 4.2 提出后未见任何修订记录的 16 条(**全部待人类复议**)

这是本轮审计的**系统性缺口**:8 个修订代理共收到 100 条 issue,而校验代理提出了 137 条。差额中除跨代理重复项外,以下 16 条独立缺陷在任何修订代理的处理清单中**都找不到对应记录**。

已用 grep 逐条核对它们**在当前文档中的实际存活状态**:

| 编号 | 缺陷 | 当前状态(2026-09-08 grep 亲验) | 优先级判断 |
|---|---|---|---|
| **U-01** | `get_pair_contact_force` 标"Python 侧乘 −1/dt²",实际不乘(D-030) | **仍存活**:`PRINCIPLES_CONTACT.md:610` | **高**——`−f·dt²` 陷阱同族,与 D-028/D-029 同级;且同分册 `API_CORE.md:781` 已是正确措辞("乘 −1/dt² 换算牛顿"=要求调用者自己乘),两处口径不一致 |
| **U-02** | `reset_transient_contact_state` 引用缺 `stable:` 前缀(D-087) | **仍存活**:`PRINCIPLES_CONTACT.md:594` 中 `bindings/pystiffgipc.cu:422` 与 `engine.py:941-952` 仍无前缀 | 中——按手册自订约定会落到 phase-cd 的 `add_prismatic_joint` |
| **U-03** | `STIFF_CCD_SLACK_A` 标 "ground ACCD slackness(η=1−slack)"(D-062 的 §3.3 半边) | **仍存活**:`PRINCIPLES_CONTACT.md:271`。§3.5 已修,§3.3 表未同步 | 中——同一分册内两处口径相反 |
| **U-04** | PRINCIPLES_CONTACT 版本表把稳定线分支状态标为 tag v0.8.5.3(D-037 的该分册半边) | **仍存活**:`PRINCIPLES_CONTACT.md:11`。API_CORE 与 PRINCIPLES_DYNAMICS 已修,本分册未同步 | 中——见 §2.5 的脆弱性论证 |
| **U-05** | intersection-persists 抛错列入"例外(仅 phase-cd)",稳定线同文本同抛(D-092) | **仍存活**:`KNOWN_ISSUES.md:213` ③ | 中——稳定线用户会以为此情形只 WARN 不中止 |
| **U-06** | 姊妹分册死链 `MULTIENV_MODES.md` / `FRAME_GRAPH_RL.md` / `CONTACT_CCD.md`(D-078) | **已不存活**(grep 零命中) | 无需处理,仅备查 |
| **U-07** | "v0.8.5.4 CHANGELOG 无条目"——tag 树实有完整条目(D-049) | ~~仍存活~~ → **已关闭(2026-09-08)**:随 OP-001 关闭一并改正,两文件 grep "CHANGELOG 无条目" 零命中 | ~~中~~ 已消除 |
| **U-08** | 地面轴"无 OVF 位,截断不可检测"方向反了(D-066) | **仍存活**:`KNOWN_ISSUES.md:363`、`PRINCIPLES_EXECUTION.md:997` | 中——把一条"不需要"写成"潜伏危险" |
| **U-09** | "不调用新 API 时轨迹与 0.8.5.2 完全一致"(teleport 是反例)(D-051) | **仍存活**:`CHANGELOG_TIMELINE.md:184`(:73 亦有同款转述) | **高**——危险的过度承诺,且与同节表格第 3 行自相矛盾 |
| **U-10** | CHANGELOG_TIMELINE §7 转述 friction "total 恒零"(D-060 的该分册半边) | **需人工复核**:README/PRINCIPLES_DYNAMICS/KNOWN_ISSUES 已用"受影响"措辞,CHANGELOG_TIMELINE §7 的原句未定位到 | 低——其余分册已正确 |
| **U-11** | "全驻留 3–4×(后由 Phase D 兑现为 A800 5.0×)"跨 regime 混接(D-012) | **仍存活**:`CHANGELOG_TIMELINE.md:291` | **高**——与手册自己的核心结论("决定胜负的是 regime 不是特性")直接抵触 |
| **U-12** | "graph-on 1.09–1.94×",过例真实上界 2.75×(D-011) | **仍存活**:`CHANGELOG_TIMELINE.md:339` | 中——低估最差过例代价约 40% |
| **U-13** | C6-p 1.39× 与 C6-w 1.96× 之间的回退无解释、C6-p 行无口径(D-027) | **仍存活**:`CHANGELOG_TIMELINE.md:341`、`:467`;C6-w 行已修 | 中——修了一半,断层反而更显眼 |
| **U-14** | 旋钮 "G14 时点 84+17+11 个"(D-003) | **仍存活**:`CHANGELOG_TIMELINE.md:495` | 中——与已修正的 170 并存于同一手册 |
| **U-15** | "209 条提交只存在于本地仓库",实际 62 条已推 `origin/internal/v0.8.6-rc1`(D-024) | **仍存活**:`CHANGELOG_TIMELINE.md:520` | **高**——影响备份/发布决策 |
| **U-16** | `set_prismatic_limit_barrier`"保证开度 d 永不越过 cl"(D-057) | **仍存活**:`API_CORE.md:520` | **高**——过度承诺,且与手册 §7.3 自承的 LS 耗尽后继续提交内部矛盾 |

**建议的复议顺序**:U-01 → U-16 → U-09 → U-11 → U-15(五条高优先),再扫中优先九条,U-06/U-10 最后。

### 4.3 修订代理反向纠正校验代理的 4 处

复核不是单向的。以下 4 处,修订代理在动手前发现**校验代理给的证据本身有误**,并在文档里写下了更准确的版本——这些是台账可信度的正面证据:

| 涉及条目 | 校验代理的说法 | 修订代理的纠正 |
|---|---|---|
| D-052 | "分支 `port/friction-anchor-086` 的 merge-base = `b3ab747`(即被审的 HEAD)" | **假**:该分支从 `6b0e02e`(2026-07-28)分出,落后 phase-cd HEAD **147 条**(`3ec2734` 的父提交是 `6b0e02e`)。文档因此改为"须前移到 HEAD + 重新过门禁";另发现 `3ec2734` **未触任何 `frame_fsm/` 文件**,图路径快照坑在该分支上仍开 |
| §4.1 #11 | "beaker `~0.3–0.6 s` 的下限 0.3 是拼造的" | **假**:`SIMULATOR_EXECUTION_DESIGN.md:167` 有"启动 +0.3s";校验代理只查了 `:123` |
| D-037 | 编排方给的事实表写 "稳定线 HEAD `b8e27a1` = v0.8.5.3" | **与仓库不符**:`b8e27a1` 是 v0.8.5.3 的 tag 提交,但分支 HEAD 是 `c0339c8`/v0.8.5.4。修订代理按"自行核实优先"的授权采信了仓库 |
| D-070 | "`STIFF_LOG_LEVEL` 两树 grep 零命中" | **更精确**:库代码(`StiffGIPC/`、`stiff_physics/`、`bindings/`)零读取属实,但**个别示例脚本确实 setdefault 了它**——只是无人消费。文档采用了更精确的表述 |

---

## 5. 审计方法与统计

### 5.1 方法

两级对抗结构,**校验与修订由不同代理承担**,且都要求"回到代码里"而不是"读文档判断":

```
起草(8 分册)
   │
   ├─► 第一级:21 个对抗校验代理,分派到各分册(每册 2–4 个独立代理)
   │      规则:只报能被代码或 git 证伪的事实性错误;每条必须给 quote + problem
   │            + evidence(file:line)+ severity;不报风格、不报"建议补充"
   │      产出:137 条 issue
   │
   ├─► 去重与分派(编排层合并跨代理重复项)
   │      产出:交给修订代理的 100 条清单
   │
   └─► 第二级:8 个修订代理,每册一个
          规则:动手前逐条回代码复核;确认成立才改;不成立写明理由跳过
          产出:89 条确认修复 + 11 条跳过 + 4 处反向纠正校验代理的证据
```

关键设计:**多个校验代理独立审同一分册**。重复抓到的条目(本台账中的 `×N`)是最可信的——`×4` 意味着四个互不通信的代理各自从代码里得出了同一结论。

### 5.2 统计

| 指标 | 数 |
|---|---|
| 代理返回值总数 | **29**(21 校验 + 8 修订) |
| 校验代理提出的 issue 条目 | **137** |
| 修订代理收到的(去重后)清单 | **100** |
| ├ 复核确认并修复 | **89** |
| └ 复核后判定不成立/已失效而跳过 | **11**(见 §4.1) |
| 合并重复后的**独立缺陷** | **92**(D-001 … D-091,外加补录的 D-092) |
| ├ 有修订代理确认的修复记录 | **75**(其中 3 条只修了跨分册的一半:D-037 / D-060 / D-062) |
| ├ 仅被跳过、无独立修复记录 | **4**(D-036, D-059, D-076, D-077;前三条理由为"当前文本已正确",第四条前提已失效) |
| └ 提出后未见任何修订记录 | **13**(见 §4.2) |
| §4.2 待复议条目合计 | **16**(13 条无记录 + 3 条仅部分修订);**其中 15 条经 grep 确认当前仍存活** |
| 修订代理反向纠正校验代理证据 | **4 处**(见 §4.3) |

> **编号说明**:D-001…D-091 按 A→E 分组连号;**D-092 是补录**——编制本台账时发现 §4.2/U-05
> 在主表中漏号,补在 C 组末尾而非重排全表(重排会打断 §2/§3/§4 的交叉引用)。

**severity 分布**(校验代理自评,按 137 条原始条目):high 与 medium 集中在 C 组(版本归属)与 D 组(机制描述),low 集中在 A 组(计数)与 E 组(引用)。**本轮唯一被跳过的 high 是跨架构确定性归属**(§4.1 #6),优先复议。

### 5.3 缺陷在文档间的分布

| 文档 | 收到的 issue | 确认修复 | 跳过 | 该册**当前仍存活**的未修条目 |
|---|---|---|---|---|
| `CHANGELOG_TIMELINE.md` | 18 | 17 | 1 | **6**(U-07/09/11/12/13/14/15 中的 6 条) |
| `API_EXECUTION.md` | 14 | 14 | 0 | 0 |
| `PRINCIPLES_CONTACT.md` | 12 | 12 | 0 | **4**(U-01/02/03/04) |
| `PRINCIPLES_DYNAMICS.md` | 10 | 10 | 0 | 0 |
| `PRINCIPLES_EXECUTION.md` | 10 | 10 | 0 | 1(U-08 的一半) |
| `KNOWN_ISSUES.md` | 9 | 9 | 0 | **3**(U-05/07/08 = D-092/049/066) |
| `API_CORE.md` | 8 | 8 | 0 | 1(U-16) |
| `README.md` | 19 | 9 | 10 | 0 |

读法:**修复率与残留量不相关**。`CHANGELOG_TIMELINE.md` 修了 17 条(最多),但也留下最多未处理条目——因为它收到的校验最密集(4 个代理、36 条原始 issue),去重分派时漏掉的也最多。`README.md` 的 10 条跳过全部是"清单基于旧修订",不代表它质量更好,只代表它被审了两轮。

### 5.4 抽样亲验记录(本台账自己的可信度证据)

编制本台账时**重跑**了以下命令,结果与两级代理的结论一致:

| 项 | 命令输出 | 与台账一致 |
|---|---|---|
| FrameStatus `def_readonly` 计数 | `39` | ✓ D-001 |
| `knob_registry.h` 的 `X(STIFF_*)` | `170` | ✓ D-002 |
| `gipc_modules/*.inl` | `14` | ✓ D-005 |
| phase-cd `examples/test_*.py` | `30` | ✓ D-004 |
| 稳定线 `examples/test_*.py` | `29` | ✓ D-004 |
| 稳定仓 `git describe --tags --long` | `v0.8.5.4-0-gc0339c8` | ✓ D-037 |
| 稳定仓 `git diff v0.8.5.3 --stat`(编制时) | 空;**2026-09-08 起 8 files +511**(工作树已恢复 HEAD) | ✓ D-037 |
| 稳定仓 `git status --short`(编制时) | 8 个 `M`;**2026-09-08 起为空** | ✓ D-037 |

另于 §4.2 逐条 grep 了 16 条未修条目在当前文档中的存活状态(结果见该节表格)。

**其余 84 条独立缺陷的代码证据未在编制阶段重跑**,沿用校验代理 + 修订代理的两级复核结论。这是本台账可信度的上界:**逐条复核仍需人类照"验证方法"一列自己跑。**

---

*台账编制:2026-09-08。原始素材:29 份代理返回值转储(137 条 issue + 8 份修订 notes)。*
*本台账只记"已抓出并处理"的事实性缺陷;待核实项见 [`OPEN_POINTS.md`](OPEN_POINTS.md)。*
