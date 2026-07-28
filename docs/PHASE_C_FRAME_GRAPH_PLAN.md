# Phase C：整帧 CUDA Graph 蓝图（2026-07-28 起草）

目标形态（用户原话）：整帧图 + 条件节点（CUDA 12.4+，本机 driver=13010 ✓），
增长只许发生在帧边界（discard-grow 铁律已保证大半），异常改为帧尾裁决，
BVH sort n 固定。Phase D 在其上叠 episode 级驻留（动作整段预上传、帧图循环
消费、观测异步双缓冲流回）。

## 现存逐迭代宿主同步清单（B2' 后的全部残余，towel/foldshirt 实测）

| 读点 | 大小 | 宿主消费 | 图化策略 |
|---|---|---|---|
| ccd 标量链 | 72B×0.81/iter | alpha/temp_alpha 起点值 + validate + diag | alpha 已在 slots[5] 设备驻留；LS 起点改读设备（见下）；validate→帧尾旗标；diag→帧尾批量 |
| LS decision | 12B×1/trial | 回溯 while 环控制 | **WHILE 条件节点**：`_global_ls_decide` 已写 status[0]，改写条件句柄；trial 体=stepForward+buildCP+energy+decide 全设备 |
| LS 出口计数刷新 | 24B+4B×1/iter | 下迭代 GH 装配的 offset 算术（global_triplet_offset 等） | 最难点：装配 offset 是宿主算术编入 launch 参数。对策=装配核改设备 offset 读（`d_cpNum` 前缀和已在设备）或每迭代一次 32B 打包读（保留，图外） |
| GH start-ids | 16B×0.97/iter | ABD/FEM contact 段偏移宿主算术 | 同上打包读或设备 offset 化 |
| PCG 状态 | 16B×1/iter | k/h_break（device-loop 后的收尾读） | Newton WHILE 条件节点内消化 |
| Newton 收敛 | （含于上） | distToOpt_PN < threshold break | **WHILE 条件节点**：残差已设备归约，写条件句柄 |

## 分层实施（每层独立可验，沿用 建→锚→套件→指纹→推 环）

**C-1 LS 试探环图化**（最高价值/最低风险）：
- cudaGraphConditionalHandle + WHILE node 包 trial 体；`_global_ls_decide`
  尾部 `cudaGraphSetConditional(handle, status[0]==1 && trial<budget)`。
- 溢出恢复/gdCollapse：decide 已写 status[1]/[2]——条件节点内不可宿主恢复
  → 溢出时条件节点退出 + 帧尾（迭代尾）宿主检查 status[1] 走 legacy 重评
  （罕见路径，语义=今日恢复的迭代尾版）。
- 出口计数刷新保留（图外 1 次），decision 读消失（-1/trial）。
- 前提：trial 体内 buildCP 的 grow 不可发生——B3 已保证（trial defer 模式
  发射 trash+计数器，无 grow）✓；BVH build 的 thrust sort 需换 cub 固定
  scratch（thrust 在 capture 下会 malloc——检查：_mc_sort_active 已是 cub
  预分配（per-env 路径注释），merged 路径 thrust::sort_by_key 需换）。

**C-2 Newton 环图化**：外层 WHILE（收敛/迭代帽双条件），体=GH 装配+solve+
ccd_alpha+LS 子图。前提=C-1 + 装配 offset 设备化（或按迭代 exec-update 注
参）。异常（GeometryError 族）→设备旗标，WHILE 退出后帧尾统一 throw（用户
指定语义）。

**C-3 帧图**：Newton WHILE + 帧首 buildBVH/buildCP + 帧尾 checkpoint 配对
集（帧入口配对集=carrier，检查点门禁 1e-12/逐位已锁）。增长事件=图失效→
帧边界重录（B2' 代数计数器直接复用）。

**C-4 Phase D**：episode 驻留。动作序列 H2D 一次；帧图 WHILE-loop 消费；
观测 async D2H 双缓冲 + event 围栏；宿主整 episode 零阻塞。

## 风险与既得资产
- 条件节点要求 12.4+ runtime/driver：本机 ✓；A800 nsys 2026.3.1 环境待查
  runtime 版本（venv wheel sm_80 需带 12.4+ toolkit 构建——dlto 锚已跨架构）。
- strict 模式：图化后 thrust→cub 替换等必须保锚逐位（cub 定序 ✓）；strict
  可先走非图路径（同 B2' 的 !SPMV_DET 门控先例），锚免疫。
- 已得资产：PCG device-loop（尾launch 自循环图先例）、B2' 容量网格/设备计
  数模式全套、单一代数计数器、15 段门禁+指纹配方。

## C-1 完成后的修订（2026-07-28 实战回填）

C-1 已落地（STIFF_LS_GRAPH，默认 0；七连环教训见 PHASE_A 战报）。两条对
C-2/C-3 的**硬性铁律**（C-1 用 sanitizer/诊断读回逐一实证）：
1. 任何按值进核的逐迭代/逐帧标量（c1m、κ、offset、count、bound）都必须
   设备槽化（seed 核按值下发 + `*_dev` 尾参），否则缓存图重放用陈值——
   症状=物理劣化（E 失配/回溯烧尽）而非崩溃，极难定位；
2. 任何 free+malloc（resize_discard/ensure_*/grow）都必须
   `++pcg_buffer_generation()`——迄今补齐：pair/reduce/sort/MAS 输出/
   snapshot/摩擦 lastH 六族；症状=重放悬垂（illegal access）。

C-2 侦察数据：GH 装配 offset 网 = 13 号文件 28 处 global_triplet_offset
+ 9 处镜像计数读，全部烤进各能量项装配核的 launch 参数；设备侧已有
d_*_contact_start_id[5] 块与 d_unique_key_number（B2'-a）。C-2 路线：
- 装配 offset 一次性打包 seed（把 h_cpNum_last 族/偏移按值下发 12-slot
  设备数组），装配核 +offset_dev 尾参（③/C-1 同款）；
- Newton WHILE 尾launch 图包 {GH→convert→MAS→PCG(device-loop)→ccd→LS 子图}；
  收敛判定=残差设备归约写 status；κ 更新（postLineSearch close 检查）
  帧内设备化或留图外（先留图外=每迭代一次小同步，C-3 再收）。
- 验证环不变：build→anchor→套件→towel/foldshirt knob 双态→指纹→推送。

## C-2 核心简化发现（2026-07-28 深读 13 号回填）

offset 累加结构：接触项（barrier 684/700 += f(h_cpNum[2/3/4])、ground 751
+= h_gpNum）之后，FEM 各项偏移 = 接触段总量 + **场景常量步长**
（fem_tet×10 / tri_edge×10 / tri×6 / softNum）。⇒ 逐迭代可变性坍缩为
**一个设备标量 `d_contact_triplet_total`**（tiny 核从 _cpNum 槽直算
c2·M6+c3·M9+c4·M12+gp——无需宿主镜像）：
- FEM 装配核 +`const int* d_contact_total` 尾参 + 各自常量部分和（烤死安全）；
- 接触装配核本就按 MatIndex 设备散射 ✓；
- converter：length 参数（=offset）需容量界+d_live（B2'-a 的
  d_unique_key_number 已铺路；cub sort/RLE 容量化）；宿主 offset 镜像累加
  保留（图外消费者：ensure_capacity_preserve/诊断）。
执行序：①d_contact_total 核+FEM 装配尾参（机械 ~8 核）→ ②converter 容量化
→ ③Newton WHILE 尾launch 图（LS 子图复用、PCG device-loop 内嵌、κ 更新暂留
图外）→ 验证环同 C-1。
