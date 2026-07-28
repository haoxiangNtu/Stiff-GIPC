# Phase A：全 GPU 驻留战役的定量依据（A800 nsys 剖析，2026-07-28）

场景：ModelScope 盘子 228 帧驻场 episode，dlto rc2 wheel，A800（sm_80）。
工件：/root/workspace/plate_nsys_{merged,isolated}.nsys-rep + stats 日志。

## 主账本（merged）

| 量 | 值 | 折合每帧 |
|---|---|---|
| 仿真墙钟（profiler 下） | 29.2s | 128ms |
| **GPU 内核忙碌总计** | **≈7.0s** | ≈31ms |
| **GPU 等主机（差额）** | **≈22s ≈ 75%** | ≈97ms |
| 阻塞 cudaMemcpy | 24.3s API / 31,475 次 | **138 次/帧** |
| cudaMemcpyToSymbol | 1.2s / 30,029 次 | 132 次/帧 |
| cudaStreamSynchronize | 1.2s / 34,412 次 | 151 次/帧 |
| cudaLaunchKernel | 1.0s / **341,867 次** | **1,500 次/帧** |

阻塞 memcpy 是事实同步点（其耗时含隐式排空等待）：主机每帧把 GPU 排空 ~138 次。
**全驻留（Phase B/C）在此机的理论上限 ≈ 3-4×**（29.2s → ~8-9s，若 GPU 不再排空）。

## 内核侧结构（次要发现）

- 广相查询家族占内核时间 38%+（_selfQuery_ee_lb2 16.8% 领跑、ee_ccd 11.0%、
  vf 10.4%、vf_ccd 6.0%）——EE_LB=2 已在 merged 生效。
- cub RadixSort 47,896 实例（≈210 次/帧，BVH 每帧全重建的排序）——图化时随帧图
  一起驻留，launch 开销同灭。

## 结论 → Phase B/C 靶点排序

1. 138 次/帧阻塞拷贝：计数/α/能量/收敛读回 → 设备驻留谓词（设备线搜索、PCG 自
   尾图、设备 CCD 链三个既有模式连环）。
2. 132 次/帧 ToSymbol：EE 门族每 buildCP 重发布 → 描述符战役 phase-1 的
   `__constant__` 合并单写正好同解（两战役在此汇合）。
3. 1,500 次/帧发射：整帧 CUDA Graph（CUDA 12.4+ 条件节点）一次发射吃掉全部。

isolated 对照：wall 22.9s、newton 545——结构同病，比例相近。
