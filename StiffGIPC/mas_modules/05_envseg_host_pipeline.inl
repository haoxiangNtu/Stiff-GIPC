#include "linear_system/utils/pcg_capacity_mode.h"  // [B2'-b]
#include <cub/device/device_scan.cuh>
__global__ void _mas_env_base(const unsigned int* prefixSum, const unsigned int* prefix,
                              int wpe, int n_env, int* envBase, int* envStart, int* padTot)
{
    if(blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int base = 0;
    for(int e = 0; e < n_env; e++)
    {
        int          sW = e * wpe, eW = (e + 1) * wpe;
        envStart[e]     = (int)prefixSum[sW];
        unsigned int Ce = prefixSum[eW - 1] + prefix[eW - 1] - prefixSum[sW];   // clusters in env e
        envBase[e]      = base;
        base += ((int)Ce + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
    *padTot = base;
}
// apply the per-env aligned offset to every warp's prefix (envStart precomputed ⇒ no RAW hazard)
__global__ void _mas_env_apply(unsigned int* prefixSum, const int* envBase, const int* envStart,
                               int warpNum, int wpe)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if(w >= warpNum)
        return;
    int e       = w / wpe;
    prefixSum[w] = (unsigned int)(envBase[e] + ((int)prefixSum[w] - envStart[e]));
}
// override d_levelSize[L].x with the per-env-padded cluster total (device — no host copy)
__global__ void _mas_env_setx(int2* levelSizeSlot, const int* padTot)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        levelSizeSlot->x = *padTot;
}
// [B3 s8] device-resident env segmentation for the level loop: warpNum and the
// divisibility guard move in-kernel, so no per-level count readback. A level
// whose warp count stops dividing evenly falls back to segN=1 exactly like the
// host check did. d_segwpe = {segN, wpe} feeds the apply/setx guards.
__global__ void _mas_env_base_dev(const unsigned int* prefixSum, const unsigned int* prefix,
                                  const int2* levelSize, int level, int n_env_static,
                                  int* envBase, int* envStart, int* padTot, int2* segwpe)
{
    if(blockIdx.x != 0 || threadIdx.x != 0)
        return;
    const int number  = levelSize[level].x;
    const int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    const int n_env   = n_env_static;
    if(n_env <= 1 || warpNum <= 0 || warpNum % n_env != 0)
    {
        *segwpe = make_int2(1, 0);
        return;
    }
    const int wpe = warpNum / n_env;
    *segwpe       = make_int2(n_env, wpe);
    int base = 0;
    for(int e = 0; e < n_env; e++)
    {
        int          sW = e * wpe, eW = (e + 1) * wpe;
        envStart[e]     = (int)prefixSum[sW];
        unsigned int Ce = prefixSum[eW - 1] + prefix[eW - 1] - prefixSum[sW];
        envBase[e]      = base;
        base += ((int)Ce + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
    *padTot = base;
}
__global__ void _mas_env_apply_dev(unsigned int* prefixSum, const int* envBase, const int* envStart,
                                   const int2* levelSize, int level, const int2* segwpe)
{
    const int segN = segwpe->x;
    if(segN <= 1)
        return;
    const int wpe     = segwpe->y;
    const int number  = levelSize[level].x;
    const int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    int       w       = blockIdx.x * blockDim.x + threadIdx.x;
    if(w >= warpNum)
        return;
    int e        = w / wpe;
    prefixSum[w] = (unsigned int)(envBase[e] + ((int)prefixSum[w] - envStart[e]));
}
__global__ void _mas_env_setx_dev(int2* levelSizeSlot, const int* padTot, const int2* segwpe)
{
    if(blockIdx.x == 0 && threadIdx.x == 0 && segwpe->x > 1)
        levelSizeSlot->x = *padTot;
}
// [B3 s8] the host-static part of _mas_envSegN (env var + verified env count +
// scratch cap); the warpNum-dependent checks now live in _mas_env_base_dev.
static int _mas_envSegN_static(int numEnvs)
{
    const char* force = getenv("STIFF_MAS_SEG");
    if(force && atoi(force) == 0)
        return 1;
    return (numEnvs >= 1 && numEnvs <= 4096) ? numEnvs : 1;
}
// Host-side decision only (getenv + integer arithmetic, no device access → no sync): returns the
// effective env count to segment by, or <=1 when disabled.
//   default (STIFF_MAS_SEG unset): ON for multi-env (numEnvs>1) in EVERY mode.
//       Rationale: the envs never interact (the system is block-diagonal per env), so a per-env MAS
//       hierarchy is the CORRECT preconditioner; a global MAS would aggregate non-interacting envs
//       into shared coarse banks — meaningless AND slower. For strict it additionally gives the
//       cross-env/batch bit-identity contract. Single-env (numEnvs==1) → nothing to segment → OFF.
//   explicit override (any mode, e.g. for A/B perf tests):
//       STIFF_MAS_SEG=0 → force OFF  (measure the old global MAS)
//       STIFF_MAS_SEG=1 → force ON, use body-group count
//       STIFF_MAS_SEG=N → force ON, N envs
static int _mas_envSegN(int warpNum, int numEnvs)
{
    const char* force = getenv("STIFF_MAS_SEG");
    int n_env;
    if(force)
    {
        n_env = atoi(force);            // 0=off, 1=auto(body-group count)
        if(n_env == 0)
            return 1;
        if(n_env != 1 && n_env != numEnvs)
        {
            // A forced N different from the host-verified env count would
            // bypass the bank-range homogeneity guard in sim_engine.cu and
            // segment through env boundaries. Honor only the verified count.
            static int warned = 0;
            if(!warned)
            {
                warned = 1;
                printf("[per-env MAS] STIFF_MAS_SEG=%d ignored (verified env "
                       "count is %d); using the verified count\n",
                       n_env, numEnvs);
            }
        }
        n_env = numEnvs;
    }
    else
    {
        n_env = numEnvs;                // default: ON for multi-env, every mode (see rationale above)
    }
    // 4096 = d_envBase/d_envStart scratch capacity (also enforced host-side).
    if(n_env <= 1 || n_env > 4096 || warpNum <= 0 || warpNum % n_env != 0)
        return 1;
    return n_env;
}

void MASPreconditioner::BuildConnectMaskL0()
{

    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                             d_neighborNum,
                                             d_neighborList,
                                             d_fineConnectMask,
                                             d_partId_map_real,
                                             d_real_map_partId,
                                             number);
#else
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0<<<numBlocks, blockSize>>>(
        d_neighborStart, d_neighborNum, d_neighborList, d_fineConnectMask, number);
#endif
}

void MASPreconditioner::PreparePrefixSumL0()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0_new<<<numBlocks, blockSize>>>(
        d_prefixOriginal, d_fineConnectMask, d_partId_map_real, number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0<<<numBlocks, blockSize>>>(d_prefixOriginal, d_fineConnectMask, number);
#endif
}

void MASPreconditioner::BuildLevel1()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    {
        size_t workspace_bytes = m_scanTempBytes;
        CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
            d_scanTemp,
            workspace_bytes,
            d_prefixOriginal,
            d_prefixSumOriginal,
            warpNum,
            cudaStreamPerThread));
    }
    // [per-env MAS] pad each env's level-1 clusters to a BANKSIZE-aligned block (no bank sharing).
    int _segN = _mas_envSegN(warpNum, m_numEnvs);
    if(_segN > 1)
    {
        int wpe = warpNum / _segN;
        _mas_env_base<<<1, 1>>>((unsigned int*)d_prefixSumOriginal, (unsigned int*)d_prefixOriginal,
                                wpe, _segN, d_envBase, d_envStart, d_padTot);
        _mas_env_apply<<<(warpNum + 255) / 256, 256>>>((unsigned int*)d_prefixSumOriginal, d_envBase,
                                                       d_envStart, warpNum, wpe);
    }
    _buildLevel1_new<<<numBlocks, blockSize>>>(d_levelSize,
                                               d_coarseSpaceTables,
                                               d_goingNext,
                                               d_fineConnectMask,
                                               d_prefixSumOriginal,
                                               d_prefixOriginal,
                                               d_partId_map_real,
                                               number);
    if(_segN > 1)
        _mas_env_setx<<<1, 1>>>(d_levelSize + 1, d_padTot);   // padded level-1 cluster count (device)
#else
    int number    = totalNodes;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    {
        size_t workspace_bytes = m_scanTempBytes;
        CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
            d_scanTemp,
            workspace_bytes,
            d_prefixOriginal,
            d_prefixSumOriginal,
            warpNum,
            cudaStreamPerThread));
    }
    _buildLevel1<<<numBlocks, blockSize>>>(d_levelSize,
                                           d_coarseSpaceTables,
                                           d_goingNext,
                                           d_fineConnectMask,
                                           d_prefixSumOriginal,
                                           d_prefixOriginal,
                                           number);
#endif
}

void MASPreconditioner::BuildConnectMaskLx(int level)
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                                      d_neighborNum,
                                                      d_neighborList,
                                                      d_coarseSpaceTables,
                                                      d_nextConnectMask,
                                                      d_fineConnectMask,
                                                      level,
                                                      d_partId_map_real,
                                                      totalNodes,
                                                      number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx<<<numBlocks, blockSize>>>(d_neighborStart,
                                                  d_neighborNum,
                                                  d_neighborList,
                                                  d_coarseSpaceTables,
                                                  d_nextConnectMask,
                                                  d_fineConnectMask,
                                                  level,
                                                  number);
#endif
}

void MASPreconditioner::NextLevelCluster(int level)
{
    // [B3 mas-8] single-env: capacity grid + device-read count (no host
    // readback fed us `number`); multi-env keeps the legacy host-count path
    // (its env-padding arithmetic below PrefixSumLx needs host warp counts).
    if(m_numEnvs <= 1)
    {
        int blockSize = DEFAULT_BLOCKSIZE;
        int numBlocks = (m_clusterCap + blockSize - 1) / blockSize;
        _nextLevelCluster_dev<<<numBlocks, blockSize>>>(
            d_nextConnectMask, d_nextPrefix, d_levelSize, level);
        return;
    }
    // [B3 s8] multi-env: same device-count kernel, capacity grid.
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (m_clusterCap + blockSize - 1) / blockSize;
    _nextLevelCluster_dev<<<numBlocks, blockSize>>>(
        d_nextConnectMask, d_nextPrefix, d_levelSize, level);
}

void MASPreconditioner::ComputeNextLevel(int level)
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _computeNextLevel<<<numBlocks, blockSize>>>(
        d_coarseSpaceTables, d_nextConnectMask, level, number);
}

void MASPreconditioner::PrefixSumLx(int level)
{
    // [B3 mas-8] single-env: fixed capacity-sized scan (prefix buffer is
    // zeroed to capacity in ReorderRealtime, so the padded tail scans as a
    // constant and consumers only read the live prefix range) + device-read
    // count/levelBegin in the kernel. Bitwise-identical live results: the
    // scan prefix over [0, warpNum) is unchanged by trailing zeros, and the
    // kernel does exactly the legacy work for idx < number.
    if(m_numEnvs <= 1)
    {
        int blockSize = BANKSIZE * BANKSIZE;
        int numBlocks = (m_clusterCap + blockSize - 1) / blockSize;
        int warpCap   = (m_clusterCap + BANKSIZE - 1) / BANKSIZE;
        size_t workspace_bytes = m_scanTempBytes;
        CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
            d_scanTemp,
            workspace_bytes,
            d_nextPrefix,
            d_nextPrefixSum,
            warpCap,
            cudaStreamPerThread));
        _prefixSumLx_dev<<<numBlocks, blockSize>>>(d_levelSize,
                                                   d_nextPrefix,
                                                   d_nextPrefixSum,
                                                   d_nextConnectMask,
                                                   d_goingNext,
                                                   level);
        return;
    }
    // [B3 s8] multi-env: capacity scan over the zero-padded prefix buffer
    // (live prefix range unchanged by trailing zeros), then the env padding
    // chain reads its geometry from d_levelSize/d_segwpe on device — the
    // per-level h_clevelSize readbacks are gone.
    {
        int blockSize = BANKSIZE * BANKSIZE;
        int numBlocks = (m_clusterCap + blockSize - 1) / blockSize;
        int warpCap   = (m_clusterCap + BANKSIZE - 1) / BANKSIZE;
        size_t workspace_bytes = m_scanTempBytes;
        CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
            d_scanTemp,
            workspace_bytes,
            d_nextPrefix,
            d_nextPrefixSum,
            warpCap,
            cudaStreamPerThread));
        _mas_env_base_dev<<<1, 1>>>(d_nextPrefixSum, d_nextPrefix, d_levelSize, level,
                                    _mas_envSegN_static(m_numEnvs),
                                    d_envBase, d_envStart, d_padTot, d_segwpe);
        _mas_env_apply_dev<<<(warpCap + 255) / 256, 256>>>(
            d_nextPrefixSum, d_envBase, d_envStart, d_levelSize, level, d_segwpe);
        _prefixSumLx_dev<<<numBlocks, blockSize>>>(d_levelSize, d_nextPrefix, d_nextPrefixSum,
                                                   d_nextConnectMask, d_goingNext, level);
        _mas_env_setx_dev<<<1, 1>>>(d_levelSize + level + 1, d_padTot, d_segwpe);
    }
}

void MASPreconditioner::AggregationKernel()
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _aggregationKernel<<<numBlocks, blockSize>>>(
        d_denseLevel, d_coarseTable, d_goingNext, levelnum, number);
}


void MASPreconditioner::computeNumLevels(int vertNum)
{
    int totalSz = 0;
    int nLevel  = 1;
    int levelSz = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    totalSz += levelSz;

    while(levelSz > BANKSIZE)
    {
        levelSz /= BANKSIZE;

        nLevel++;
        levelSz = (levelSz + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
        totalSz += levelSz;
    }
    nLevel   = nLevel + 1;
    levelnum = nLevel > 6 ? 6 : nLevel;
    printf("level num:  %d\n", levelnum);
    //totalSize = totalSz * SizeRatio;
}

void MASPreconditioner::BuildCollisionConnection(unsigned int* connectionMsk,
                                                 int*          coarseTableSpace,
                                                 int           level,
                                                 int           cpNum)
{
    int number    = cpNum;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    _buildCollisionConnection_new<<<numBlocks, blockSize>>>(connectionMsk,
                                                            coarseTableSpace,
                                                            _collisonPairs,
                                                            d_real_map_partId,
                                                            level,
                                                            collision_node_Offset,
                                                            totalNodes,
                                                            number);

#else
    _buildCollisionConnection<<<numBlocks, blockSize>>>(
        connectionMsk, coarseTableSpace, _collisonPairs, level, collision_node_Offset, totalNodes, number);

#endif
}
#include <fstream>
bool MASPreconditioner::deviceExtentActive() const
{
    return m_allocClusterTotal > 0
           && GIPCTripletMatrix::device_count_mode();
}

int MASPreconditioner::exactClusterCountBlocking() const
{
    if(!deviceExtentActive())
        return totalNumberClusters;
    int2 extent{};
    CUDA_SAFE_CALL(cudaMemcpy(&extent,
                              d_levelSize + levelnum,
                              sizeof(int2),
                              cudaMemcpyDeviceToHost));
    return extent.y;
}

int MASPreconditioner::ReorderRealtime(int cpNum)
{
    CUDA_SAFE_CALL(cudaMemsetAsync(
        d_levelSize,
        0,
        levelnum * sizeof(int2),
        cudaStreamPerThread));


    BuildConnectMaskL0();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(cpNum)
        BuildCollisionConnection(d_fineConnectMask, nullptr, -1, cpNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    PreparePrefixSumL0();

    BuildLevel1();
    for(int level = 1; level < levelnum; level++)
    {
        // clear the FULL cluster-space capacity: padded slots beyond the real
        // cluster count are read by _nextLevelCluster/_prefixSumLx and must be
        // deterministic zeros, not stale/uninitialized memory.
        CUDA_SAFE_CALL(cudaMemsetAsync(
            d_nextConnectMask,
            0,
            m_clusterCap * sizeof(int),
            cudaStreamPerThread));

        BuildConnectMaskLx(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        if(cpNum)
            BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum);

        // [B3 mas-8] single-env: the per-level count stays device-resident
        // (written by the previous _prefixSumLx, read by the _dev kernels) —
        // the host mirror readback was the round-trip king (8/Newton-iter).
        // The prefix buffer is zeroed to capacity so the fixed-size scan in
        // PrefixSumLx sees deterministic zeros in the padded tail.
        // [B3 s8] both paths: capacity-zeroed prefix buffer, no per-level
        // count readback (the level loop is now fully device-resident).
        CUDA_SAFE_CALL(cudaMemsetAsync(
            d_nextPrefix, 0,
            ((m_clusterCap + BANKSIZE - 1) / BANKSIZE) * sizeof(unsigned int),
            cudaStreamPerThread));

        NextLevelCluster(level);



        PrefixSumLx(level);

        ComputeNextLevel(level);

    }

    if(deviceExtentActive())
    {
        totalNumberClusters = m_allocClusterTotal;
    }
    else
    {
        CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize,
                                  d_levelSize + levelnum,
                                  sizeof(int2),
                                  cudaMemcpyDeviceToHost));
        totalNumberClusters = h_clevelSize.y;
    }

    AggregationKernel();

    return totalNumberClusters;
}


void MASPreconditioner::PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                                            int*             row_ids,
                                            int*             col_ids,
                                            uint32_t*        indices,
                                            int              offset,
                                            int              triplet_number,
                                            const int*       d_triplet_number)
{
    //cudaEvent_t start, end0, end1, end2;
    //cudaEventCreate(&start);
    //cudaEventCreate(&end0);
    //cudaEventCreate(&end1);

    //cudaEventRecord(start);



    using namespace muda;
    int tripletNum = triplet_number;
    // [4.3] zero the binned coarse-aggregation accumulator + bind the device-symbol pointer.
    // Only COARSE cluster-blocks accumulate (fine blocks are set directly at level 0).
    {
        int startC = totalMapNodes / BANKSIZE;
        int endC   = totalNumberClusters / BANKSIZE;
        if(endC > startC)
            CUDA_SAFE_CALL(cudaMemsetAsync(
                d_matbin + (size_t)startC * MAS_NB * 9 * BINNED_K,
                0,
                (size_t)(endC - startC) * MAS_NB * 9 * BINNED_K
                    * sizeof(double),
                cudaStreamPerThread));
    }
    if(true)
    {
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                tripletNum,
                [offset           = offset,
                 levelNum         = levelnum,
                 _goingNext       = d_goingNext,
                 _invMatrix       = d_inverseMatMas,
                 _real_map_partId = d_real_map_partId,
                 indices,
                 d_triplet_number,
                 triplet_values, row_ids, col_ids] __device__(int I) mutable
                {
                    if(d_triplet_number && I >= *d_triplet_number)
                        return;
                    int index                              = indices[I];
                    auto vertRid_real                      = row_ids[index];
                    auto vertCid_real                       = col_ids[index];
                    auto H = triplet_values[index];
                    //auto&& [vertRid_real, vertCid_real, H] = hessian(index);
                    vertRid_real -= offset;
                    vertCid_real -= offset;
                    int vertCid = _real_map_partId[vertCid_real];
                    int vertRid = _real_map_partId[vertRid_real];
                    int cPid    = vertCid / BANKSIZE;


                    if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                    {
                        if(vertCid >= vertRid)
                        {
                            int bvRid = vertRid % BANKSIZE;
                            int bvCid = vertCid % BANKSIZE;
                            int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;

                            _invMatrix[cPid].M[index] = H;
                        }
                    }
                    else
                    {
                        int level = 0;
                        while(level < levelNum - 1)
                        {
                            level++;
                            if(level == 1)
                            {
                                vertCid = _goingNext[vertCid_real];
                                vertRid = _goingNext[vertRid_real];
                            }
                            else
                            {
                                vertCid = _goingNext[vertCid];
                                vertRid = _goingNext[vertRid];
                            }
                            cPid = vertCid / BANKSIZE;
                            if(vertCid / BANKSIZE == vertRid / BANKSIZE)
                            {

                                if(vertCid >= vertRid)
                                {
                                    int bvRid = vertRid % BANKSIZE;
                                    int bvCid = vertCid % BANKSIZE;
                                    int index = BANKSIZE * bvRid
                                                - bvRid * (bvRid + 1) / 2 + bvCid;
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            binned_deposit(
                                                g_matbin
                                                    + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j)
                                                          * BINNED_K,
                                                H(i, j));
                                            if(vertCid == vertRid)
                                            {
                                                binned_deposit(
                                                    g_matbin
                                                        + (((size_t)cPid * MAS_NB + index) * 9 + i * 3
                                                           + j)
                                                              * BINNED_K,
                                                    H(j, i));
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    int bvRid = vertRid % BANKSIZE;
                                    int bvCid = vertCid % BANKSIZE;
                                    int index = BANKSIZE * bvCid
                                                - bvCid * (bvCid + 1) / 2 + bvRid;
                                    for(int i = 0; i < 3; i++)
                                    {
                                        for(int j = 0; j < 3; j++)
                                        {
                                            binned_deposit(
                                                g_matbin
                                                    + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j)
                                                          * BINNED_K,
                                                H(j, i));
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

        tripletNum    = totalMapNodes * BANKSIZE;
        int threadNum = BANKSIZE * BANKSIZE;
        int blockNum  = (tripletNum + threadNum - 1) / threadNum;

        ParallelFor(blockNum, threadNum)
            .file_line(__FILE__, __LINE__)
            .apply(
                tripletNum,
                [levelNum         = levelnum,
                 _goingNext       = d_goingNext,
                 _invMatrix       = d_inverseMatMas,
                 _partId_map_real = d_partId_map_real,
                 _fineConnectMsk  = d_fineConnectMask,
                 _prefix0 = d_prefixOriginal] __device__(int idx) mutable
                {
                    int HSIZE = (BANKSIZE * BANKSIZE);
                    int Hid   = idx / HSIZE;
                    int LMRid = (idx % HSIZE) / BANKSIZE;
                    int LMCid = (idx % HSIZE) % BANKSIZE;

                    int MRid = Hid * BANKSIZE + LMRid;
                    int MCid = Hid * BANKSIZE + LMCid;

                    int            rdx = _partId_map_real[MRid];
                    int            cdx = _partId_map_real[MCid];
                    __shared__ int prefix;

                    if(threadIdx.x == 0)
                    {
                        prefix = _prefix0[Hid];
                    }
                    __syncthreads();
                    Eigen::Matrix3d mat3;
                    if(LMCid >= LMRid)
                    {
                        int index = BANKSIZE * LMRid - LMRid * (LMRid + 1) / 2 + LMCid;
                        mat3 = _invMatrix[Hid].M[index];
                    }
                    else
                    {
                        int index = BANKSIZE * LMCid - LMCid * (LMCid + 1) / 2 + LMRid;
                        mat3 = _invMatrix[Hid].M[index].transpose();
                    }

                    // prefix==1: every fine pair of this bank maps to ONE coarse
                    // diagonal block. Two aggregation paths:
                    //
                    //  * fast (non-strict): warp-tree sum + one deposit per warp.
                    //    The OLD tree ran its collectives INSIDE the validPair
                    //    branch with mask=0xffffffff, so padding lanes never
                    //    executed the collective they were named in -> CUDA UB on
                    //    partially filled banks (the real historical defect; a
                    //    32-lane warp covers two ROWS of the SAME bank - it never
                    //    spans two banks, blockDim 256 == one bank). Corrected:
                    //    ALL lanes participate unconditionally, padding pairs
                    //    contribute exact zeros.
                    //
                    //  * strict (g_det_reduce): per-pair binned deposits - exact,
                    //    order-independent, hence layout- and batch-size-invariant
                    //    (the v0.8.4.2 batch-determinism contract). No plain-double
                    //    partial sums survive on this path.
                    const bool validPair = (rdx >= 0) && (cdx >= 0);
                    if(prefix == 1 && !g_det_reduce)
                    {
                        Eigen::Matrix3d mat3_c;
                        if(validPair)
                            mat3_c = mat3;
                        else
                            mat3_c.setZero();
                        for(int iter = 1; iter < 32; iter <<= 1)
                        {
                            for(int i = 0; i < 3; i++)
                                for(int j = 0; j < 3; j++)
                                {
                                    double t = __shfl_down_sync(0xffffffffu,
                                                                mat3_c(i, j), iter);
                                    mat3_c(i, j) += t;
                                }
                        }
                        unsigned validMsk = __ballot_sync(0xffffffffu, validPair);
                        int      leadLane = validMsk ? (__ffs(validMsk) - 1) : 0;
                        int      leadRdx  = __shfl_sync(0xffffffffu, rdx, leadLane);
                        if((threadIdx.x & 0x1f) == 0 && validMsk)
                        {
                            int level  = 0;
                            int nextId = leadRdx;
                            while(level < levelNum - 1)
                            {
                                level++;
                                nextId    = _goingNext[nextId];
                                int cPid  = nextId / BANKSIZE;
                                int bvRid = nextId % BANKSIZE;
                                int index = BANKSIZE * bvRid
                                            - bvRid * (bvRid + 1) / 2 + bvRid;
                                for(int i = 0; i < 3; i++)
                                    for(int j = 0; j < 3; j++)
                                        binned_deposit(
                                            g_matbin
                                                + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j)
                                                      * BINNED_K,
                                            mat3_c(i, j));
                            }
                        }
                    }
                    else if(validPair)
                    {
                        if(prefix == 1)
                        {
                            // strict: deposit this pair's own block. rdx and cdx
                            // share the goingNext chain (single cluster), so rdx's
                            // chain diagonal slot is the correct target.
                            int level  = 0;
                            int nextId = rdx;
                            while(level < levelNum - 1)
                            {
                                level++;
                                nextId    = _goingNext[nextId];
                                int cPid  = nextId / BANKSIZE;
                                int bvRid = nextId % BANKSIZE;
                                int index = BANKSIZE * bvRid
                                            - bvRid * (bvRid + 1) / 2 + bvRid;
                                for(int i = 0; i < 3; i++)
                                {
                                    for(int j = 0; j < 3; j++)
                                    {
                                        binned_deposit(
                                            g_matbin
                                                + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j)
                                                      * BINNED_K,
                                            mat3(i, j));
                                    }
                                }
                            }
                        }
                        else
                        {
                            int level = 0;
                            while(level < levelNum - 1)
                            {
                                level++;
                                rdx      = _goingNext[rdx];
                                cdx      = _goingNext[cdx];
                                int cPid = cdx / BANKSIZE;
                                if(rdx / BANKSIZE == cdx / BANKSIZE)
                                {

                                    if(cdx >= rdx)
                                    {

                                        int bvRid = rdx % BANKSIZE;
                                        int bvCid = cdx % BANKSIZE;
                                        int index = BANKSIZE * bvRid
                                                    - bvRid * (bvRid + 1) / 2 + bvCid;


                                        for(int i = 0; i < 3; i++)
                                        {
                                            for(int j = 0; j < 3; j++)
                                            {
                                                binned_deposit(
                                                    g_matbin
                                                        + (((size_t)cPid * MAS_NB + index) * 9 + i * 3
                                                           + j)
                                                              * BINNED_K,
                                                    mat3(i, j));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
    }
    


    //cudaEventRecord(end0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int blockSize2 = 32 * 3;
    //int number2    = totalNumberClusters / BANKSIZE;
    int number2    = totalNumberClusters * 3;
    if(number2 < 1)
        return;
    int numBlocks2 = (number2 + blockSize2 - 1) / blockSize2;
    const int2* extent =
        deviceExtentActive() ? d_levelSize + levelnum : nullptr;

    {  // [4.3] combine binned coarse aggregation back into d_inverseMatMas (before inversion)
        int startC = totalMapNodes / BANKSIZE;
        int endC   = totalNumberClusters / BANKSIZE;
        int nblk   = (endC - startC) * MAS_NB;
        if(nblk > 0)
            _mas_comb_mat<<<(nblk + 255) / 256, 256>>>(
                d_inverseMatMas, d_matbin, startC, endC, extent);
    }

    __inverse6_P96x96<<<numBlocks2, blockSize2>>>(
        d_precondMatMas, d_inverseMatMas, number2, extent);

    //cudaEventRecord(end1);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    ////cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\ntime0 = %f,  time1 = %f\n\n", time0, time1);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}


void MASPreconditioner::BuildMultiLevelR(const double3* R)
{


#ifdef GROUP
    int number = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized_new<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_prefixOriginal, d_fineConnectMask, d_partId_map_real, levelnum, number);

#else
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_fineConnectMask, levelnum, number);
#endif
}

// [mas-apply-resize] The Schwarz apply launches are the in-graph PCG's
// dominant per-iteration cost: under deviceExtentActive the cluster extent
// is the ALLOCATED capacity (ReorderRealtime pins totalNumberClusters =
// m_allocClusterTotal), so every replayed iteration does real work on every
// padded cluster -- measured 63-65% of the in-graph PCG body, 3.3x the spmv.
// Unlike the spmv (whose pad lanes exit on the masked count), these kernels
// have no per-thread live guard, so the only fix is to narrow the recorded
// grid itself: arm the device resizer with the device-resident cluster count
// (d_levelSize[levelnum].y, the same value exactClusterCountBlocking reads).
// Outside capture arm() self-disables and returns -1.
static bool mas_apply_resize_enabled()
{
    static int v = -1;
    if(v < 0)
    {
        const char* e = getenv("STIFF_MAS_APPLY_RESIZE");
        v             = e && e[0] ? (atoi(e) != 0 ? 1 : 0) : 0;
    }
    return v != 0;
}

const int* MASPreconditioner::device_cluster_count() const
{
    // int2{x, y}: the total cluster count is the y component of the entry
    // one past the last level.
    return reinterpret_cast<const int*>(d_levelSize + levelnum) + 1;
}

void MASPreconditioner::SchwarzLocalXSym()
{
    //int matNum    = totalNumberClusters / BANKSIZE;
    int number    = (pcg_grid_capacity_mode() ? m_outputClusterCap : totalNumberClusters)
                    * BANKSIZE * 3;   // [B2'-b] capacity grid under cached capture
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    const int _rs_slot =
        mas_apply_resize_enabled()
            ? gipc::graph_resize::arm(device_cluster_count(), BANKSIZE * 3, blockSize, numBlocks)
            : -1;
    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym3<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number, d_levelSize, levelnum);
    if(_rs_slot >= 0)
        gipc::graph_resize::bind_last(_rs_slot);
}

void MASPreconditioner::SchwarzLocalXSym_block3()
{
    //int matNum    = totalNumberClusters / BANKSIZE;
    int number = (pcg_grid_capacity_mode() ? m_outputClusterCap : totalNumberClusters)
                 * BANKSIZE;   // [B2'-b] capacity grid under cached capture
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    const int _rs_slot =
        mas_apply_resize_enabled()
            ? gipc::graph_resize::arm(device_cluster_count(), BANKSIZE, blockSize, numBlocks)
            : -1;
    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym6<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number, d_levelSize, levelnum);
    if(_rs_slot >= 0)
        gipc::graph_resize::bind_last(_rs_slot);
}

void MASPreconditioner::SchwarzLocalXSym_sym()
{
    int matNum    = (pcg_grid_capacity_mode() ? m_outputClusterCap : totalNumberClusters)
                    / BANKSIZE;   // [B2'-b] capacity grid under cached capture
    int number = matNum * (1 + BANKSIZE) * BANKSIZE / 2;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    // (1+BANKSIZE)/2 is not integral: round the multiplier UP so the resized
    // grid is never narrower than the work.
    const int _rs_slot =
        mas_apply_resize_enabled()
            ? gipc::graph_resize::arm(device_cluster_count(), (2 + BANKSIZE) / 2, blockSize, numBlocks)
            : -1;
    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym9<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number, d_levelSize, levelnum);
    if(_rs_slot >= 0)
        gipc::graph_resize::bind_last(_rs_slot);
}

void MASPreconditioner::CollectFinalZ(double3* Z)
{
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    __collectFinalZ_new<<<numBlocks, blockSize>>>(
        Z, d_multiLevelZ, d_coarseTable, d_real_map_partId, levelnum, number);
#else
    __collectFinalZ<<<numBlocks, blockSize>>>(Z, d_multiLevelZ, d_coarseTable, levelnum, number);
#endif

}



void MASPreconditioner::setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                               int*             row_ids,
                                               int*             col_ids,
                                               uint32_t*        indices,
                                               int              offset,
                                               int              triplet_num,
                                               const int*       d_triplet_num,
                                               int              cpNum)
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        d_neighborList,
        d_neighborListInit,
        neighborListSize * sizeof(unsigned int),
        cudaMemcpyDeviceToDevice,
        cudaStreamPerThread));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        d_neighborNum,
        d_neighborNumInit,
        totalNodes * sizeof(unsigned int),
        cudaMemcpyDeviceToDevice,
        cudaStreamPerThread));


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ReorderRealtime(cpNum);

    // [audit lens-A fix] the per-frame cluster count (real contact
    // connectivity) can exceed the init-time zero-contact allocation of the
    // output-layer buffers — grow them BEFORE the memsets/deposits below.
    ensureOutputClusterCapacity(totalNumberClusters);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef SYME

    CUDA_SAFE_CALL(cudaMemsetAsync(
        d_inverseMatMas,
        0,
        totalNumberClusters / BANKSIZE
            * sizeof(__GEIGEN__::MasMatrixSymT),
        cudaStreamPerThread));
#else
    CUDA_SAFE_CALL(cudaMemsetAsync(
        d_MatMas,
        0,
        totalNumberClusters / BANKSIZE
            * sizeof(__GEIGEN__::MasMatrixT),
        cudaStreamPerThread));
#endif
    PrepareHessian_bcoo(triplet_values,
                        row_ids,
                        col_ids,
                        indices,
                        offset,
                        triplet_num,
                        d_triplet_num);

    // [debug] STIFF_MAS_DUMP=<dir>: dump the FIRST assembly's raw bcoo triplet
    // input + per-bank cluster prefix, enabling a FULL external CPU oracle
    // that independently rebuilds fine writes, the kernel-1 cross-bank ladder
    // and the kernel-2 intra-bank aggregation (see examples/test_mas_oracle.py).
    {
        static int  _tdumped = 0;
        const char* _dd      = getenv("STIFF_MAS_DUMP");
        if(_dd && !_tdumped)
        {
            _tdumped = 1;
            cudaDeviceSynchronize();
            auto wr = [&](const char* name, const void* dev, size_t bytes)
            {
                std::vector<char> h(bytes);
                cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost);
                char p[768];
                snprintf(p, sizeof(p), "%s/%s.bin", _dd, name);
                FILE* f = fopen(p, "wb");
                if(f) { fwrite(h.data(), 1, bytes, f); fclose(f); }
            };
            wr("mas_trip_rows", row_ids, (size_t)(offset + triplet_num) * sizeof(int));
            wr("mas_trip_cols", col_ids, (size_t)(offset + triplet_num) * sizeof(int));
            wr("mas_trip_vals", triplet_values,
               (size_t)(offset + triplet_num) * sizeof(Eigen::Matrix3d));
            wr("mas_trip_idx", indices, (size_t)triplet_num * sizeof(uint32_t));
            wr("mas_prefix0", d_prefixOriginal,
               (size_t)(totalMapNodes / BANKSIZE) * sizeof(int));
            printf("[mas-dump] wrote triplets (offset=%d num=%d banks=%d)\n",
                   offset, triplet_num, totalMapNodes / BANKSIZE);
        }
    }

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


static void _mas_ksum(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t acc = 1469598103934665603ULL;
    for(uint64_t v : h) { acc ^= v; acc *= 1099511628211ULL; }
    printf("[masksum] %-14s %016llx\n", name, (unsigned long long)acc);
}

void MASPreconditioner::preconditioning(const double3* R, double3* Z)
{
    if(totalNodes < 1)
        return;
    if(getenv("STIFF_KSUM")) { cudaDeviceSynchronize();
        _mas_ksum("precondMat", d_precondMatMas,
                  (size_t)(totalNumberClusters / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixSymf)); }

    // Fusion is deliberately opt-in: only the exact value "1" enables it.
    const char* fuseEnv = getenv("STIFF_MAS_FUSE");
    const bool  fuse = fuseEnv && fuseEnv[0] == '1' && fuseEnv[1] == '\0';
    const size_t clusterCapacity = static_cast<size_t>(m_clusterCap)
                                   * static_cast<size_t>(levelnum);
    if(totalNumberClusters < 0
       || static_cast<size_t>(totalNumberClusters) > clusterCapacity
       || totalNumberClusters % BANKSIZE != 0)
        throw std::runtime_error(
            "[MAS] hierarchy exceeds m_clusterCap or is not bank aligned");

    static bool fuseValidated = false;
    static int  dumped        = 0;
    const char* dumpDir       = getenv("STIFF_MAS_DUMP");
    const bool  dumpThisCall  = dumpDir && !dumped;
    const bool  validateFuse  = fuse && !fuseValidated
                               && getenv("STIFF_MAS_FUSE_VALIDATE");

    // [MAS graph-capture] all Async on the PTDS stream: the sync cudaMemset /
    // cudaMemcpyToSymbol variants broke PCG-graph capture (symbols now bound
    // once at alloc in the MAS malloc routine).
    if(!fuse)
    {
        // [B2'-b] capacity-sized (m_outputClusterCap): identical for the live
        // range, and the byte counts stop varying per solve so a captured
        // apply graph stays valid across Newton iterations.
        CUDA_SAFE_CALL(cudaMemsetAsync(
            d_multiLevelR + totalMapNodes,
            0,
            (size_t)(m_outputClusterCap - totalMapNodes) * sizeof(Eigen::Vector3f),
            0));
        CUDA_SAFE_CALL(cudaMemsetAsync(
            d_multiLevelZ,
            0,
            (size_t)m_outputClusterCap * sizeof(Precision_T3),
            0));
    }

    // [4.3] zero the binned accumulators. mR: only the COARSE
    // slots accumulate (fine [0,totalMapNodes) is set directly in __buildMultiLevelR); mZ: all.
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mRbin + (size_t)totalMapNodes * 3 * BINNED_K, 0,
                              (size_t)(m_outputClusterCap - totalMapNodes) * 3 * BINNED_K
                                  * sizeof(double), 0));   // [B2'-b] capacity-sized
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mZbin, 0,
                              (size_t)m_outputClusterCap * 3 * BINNED_K * sizeof(double), 0));   // [B2'-b]

    BuildMultiLevelR(R);
    const int2* extent =
        deviceExtentActive() ? d_levelSize + levelnum : nullptr;

    auto runLegacyPath = [&]()
    {
        int n = totalNumberClusters - totalMapNodes;
        if(n > 0)
            _mas_comb_mR<<<(n + 255) / 256, 256>>>(d_multiLevelR, d_mRbin, totalMapNodes,
                                                   totalNumberClusters, extent);
        SchwarzLocalXSym_block3();
        n = totalNumberClusters;
        if(n > 0)
            _mas_comb_mZ<<<(n + 255) / 256, 256>>>(
                d_multiLevelZ, d_mZbin, n, extent);
        CollectFinalZ(Z);
    };

    if(fuse)
    {
        const int matrixCount = totalNumberClusters / BANKSIZE;
        if(matrixCount > 0)
            _schwarzLocalXSym6_fused8<<<matrixCount * 2, 128>>>(
                d_precondMatMas,
                d_multiLevelR,
                d_mRbin,
                d_mZbin,
                totalMapNodes,
                totalNumberClusters,
                clusterCapacity,
                extent);

        const int blocks = (totalNodes + DEFAULT_BLOCKSIZE - 1) / DEFAULT_BLOCKSIZE;
        __collectFinalZ_binned_new<<<blocks, DEFAULT_BLOCKSIZE>>>(
            Z,
            d_mZbin,
            d_coarseTable,
            d_real_map_partId,
            levelnum,
            totalNodes,
            totalNumberClusters,
            clusterCapacity,
            extent);

        std::vector<double3> fusedOut;
        if(validateFuse)
        {
            fusedOut.resize(totalNodes);
            CUDA_SAFE_CALL(cudaMemcpy(fusedOut.data(),
                                      Z,
                                      totalNodes * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
        }

        // The fused path intentionally does not materialize mlR/mlZ. Replay
        // the legacy chain for the one-shot equivalence oracle, and also for
        // the first MAS dump so that its documented diagnostic buffers remain
        // complete. Both modes are excluded from PCG graph capture.
        if(validateFuse || dumpThisCall)
        {
            CUDA_SAFE_CALL(cudaMemsetAsync(
                d_multiLevelZ,
                0,
                totalNumberClusters * sizeof(Precision_T3),
                0));
            CUDA_SAFE_CALL(cudaMemsetAsync(
                d_mZbin,
                0,
                (size_t)totalNumberClusters * 3 * BINNED_K * sizeof(double),
                0));
            runLegacyPath();
        }

        if(validateFuse)
        {
            std::vector<double3> legacyOut(totalNodes);
            CUDA_SAFE_CALL(cudaMemcpy(legacyOut.data(),
                                      Z,
                                      totalNodes * sizeof(double3),
                                      cudaMemcpyDeviceToHost));

            size_t mismatches = 0;
            double maxAbs     = 0.0;
            for(int i = 0; i < totalNodes; ++i)
            {
                if(std::memcmp(&fusedOut[i], &legacyOut[i], sizeof(double3)) != 0)
                    ++mismatches;
                maxAbs = std::max(maxAbs, std::fabs(fusedOut[i].x - legacyOut[i].x));
                maxAbs = std::max(maxAbs, std::fabs(fusedOut[i].y - legacyOut[i].y));
                maxAbs = std::max(maxAbs, std::fabs(fusedOut[i].z - legacyOut[i].z));
            }
            printf("[mas-fuse-validate] nodes=%d mismatches=%zu max_abs=%.3e\n",
                   totalNodes,
                   mismatches,
                   maxAbs);
            if(mismatches != 0)
                throw std::runtime_error(
                    "[MAS] fused Schwarz/collect differs from legacy path");
            fuseValidated = true;
        }
    }
    else
    {
        runLegacyPath();
    }

    // [debug] STIFF_MAS_DUMP=<dir>: dump the FIRST preconditioning call's input R,
    // multilevel R/Z buffers and output Z as raw binaries (batch-drift bisection).
    {
        if(dumpThisCall)
        {
            dumped = 1;
            cudaDeviceSynchronize();
            auto wr = [&](const char* name, const void* dev, size_t bytes)
            {
                std::vector<char> h(bytes);
                cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost);
                char p[768];
                snprintf(p, sizeof(p), "%s/%s.bin", dumpDir, name);
                FILE* f = fopen(p, "wb");
                if(f) { fwrite(h.data(), 1, bytes, f); fclose(f); }
            };
            wr("mas_R",   R,             (size_t)totalNodes * sizeof(double3));
            wr("mas_mlR", d_multiLevelR, (size_t)totalNumberClusters * sizeof(Eigen::Vector3f));
            wr("mas_mlZ", d_multiLevelZ, (size_t)totalNumberClusters * sizeof(Precision_T3));
            wr("mas_Z",   Z,             (size_t)totalNodes * sizeof(double3));
            wr("mas_pmat", d_precondMatMas,
               (size_t)(totalNumberClusters / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixSymf));
#ifdef SYME
            wr("mas_imat", d_inverseMatMas,
               (size_t)(totalNumberClusters / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixSymT));
#endif
            // topology for external (CPU) multi-level oracles
            wr("mas_goingNext", d_goingNext,
               (size_t)totalNumberClusters * sizeof(unsigned int));
            wr("mas_map", d_partId_map_real, (size_t)totalMapNodes * sizeof(int));
            wr("mas_levelSize", d_levelSize, (size_t)(levelnum + 1) * sizeof(int2));
            printf("[mas-dump] wrote R/mlR/mlZ/Z to %s (nodes=%d clusters=%d mapNodes=%d levels=%d)\n",
                   dumpDir, totalNodes, totalNumberClusters, totalMapNodes, levelnum);
        }
    }
    //cudaEventRecord(end2);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    //cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\npreconditioning  time0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}

void MASPreconditioner::initPreconditioner_Neighbor(int vertNum,
                                                    int mCollision_node_offset,
                                                    int totalNeighborNum,
                                                    int4* m_collisonPairs,
                                                    int   partMapSize)
{
    //bankSize = 32;
    if(vertNum < 1)
    {
        totalNodes = 0;
        return;
    }
    int maxNodes = partMapSize > vertNum ? partMapSize : vertNum;
    // [per-env MAS] Hierarchy DEPTH from the per-env node count whenever env
    // segmentation will be active (same predicate as the aggregation,
    // _mas_envSegN). A depth computed from the GLOBAL count grows with batch
    // size N (e.g. 16 verts/env: N=2 -> 2 levels, N=8 -> 3 levels), so the
    // same physical env gets a different preconditioner in different batch
    // sizes -> PCG converges along a different path -> ~1e-9 batch drift.
    // m_numEnvs must therefore be set BEFORE this call (see sim_engine.cu).
    {
        const int warpNumL0 = (maxNodes + BANKSIZE - 1) / BANKSIZE;
        const int segN      = _mas_envSegN(warpNumL0, m_numEnvs);
        computeNumLevels(segN > 1 ? maxNodes / segN : maxNodes);
    }
    totalMapNodes         = partMapSize;
    collision_node_Offset = mCollision_node_offset;
    _collisonPairs        = m_collisonPairs;
    totalNodes            = vertNum;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_denseLevel, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_real_map_partId, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseTable, vertNum * sizeof(__GEIGEN__::itable)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseSpaceTables,
                              vertNum * levelnum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_levelSize, (levelnum + 1) * sizeof(int2)));
    // [per-env MAS] cluster-space arrays must hold PADDED per-level cluster
    // counts, which can exceed vertNum on small scenes (see m_clusterCap doc).
    m_clusterCap = maxNodes + (std::max(1, m_numEnvs) + 1) * BANKSIZE;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_goingNext,
                              (size_t)m_clusterCap * levelnum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefix, m_clusterCap * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefixSum, m_clusterCap * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixSumOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fineConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextConnectMask, m_clusterCap * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborList, totalNeighborNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStartTemp, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNum, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborListInit, totalNeighborNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNumInit, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_partId_map_real, partMapSize * sizeof(int)));
    // [per-env MAS] small device scratch for the env-segmented prefix (fixed max #envs = 4096).
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_envBase,  4096 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_envStart, 4096 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_padTot,   sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_segwpe,   sizeof(int2)));  // [B3 s8]

    // Query both scan value types at the maximum fixed launch extent.  CUB
    // then receives this persistent workspace on every frame, avoiding
    // Thrust's capture-illegal temporary cudaMalloc.
    const int max_scan_items =
        (std::max(vertNum, m_clusterCap) + BANKSIZE - 1) / BANKSIZE;
    size_t int_scan_bytes  = 0;
    size_t uint_scan_bytes = 0;
    CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr,
        int_scan_bytes,
        d_prefixOriginal,
        d_prefixSumOriginal,
        max_scan_items,
        cudaStreamPerThread));
    CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
        nullptr,
        uint_scan_bytes,
        d_nextPrefix,
        d_nextPrefixSum,
        max_scan_items,
        cudaStreamPerThread));
    m_scanTempBytes = std::max(int_scan_bytes, uint_scan_bytes);
    if(m_scanTempBytes)
        CUDA_SAFE_CALL(cudaMalloc(&d_scanTemp, m_scanTempBytes));
}

void MASPreconditioner::initPreconditioner_Matrix()
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

    m_allocClusterTotal = 0;
    const int exactCluster = ReorderRealtime(0);
    int totalCluster =
        std::max(exactCluster, static_cast<int>(exactCluster * 1.05));
#ifdef SYME
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_inverseMatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_MatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixT)));
#endif

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_precondMatMas,
                              totalCluster / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymf)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelR, totalCluster * sizeof(Eigen::Vector3f)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelZ, totalCluster * sizeof(Precision_T3)));
    // [4.3] binned reproducible-FP accumulators (double, K bins per scalar) for MAS determinism
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mRbin, (size_t)3 * totalCluster * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mZbin, (size_t)3 * totalCluster * BINNED_K * sizeof(double)));
    // [MAS graph-capture] bind the binned-accumulator device symbols ONCE here —
    // the per-call cudaMemcpyToSymbol in preconditioning() was a synchronous API
    // call inside the PCG-graph capture region (cudaErrorStreamCaptureUnsupported).
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_mRbin, &d_mRbin, sizeof(double*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_mZbin, &d_mZbin, sizeof(double*)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_matbin,
                              (size_t)(totalCluster / BANKSIZE) * MAS_NB * 9 * BINNED_K
                                  * sizeof(double)));
    CUDA_SAFE_CALL(
        cudaMemcpyToSymbol(g_matbin, &d_matbin, sizeof(double*)));
    m_outputClusterCap = totalCluster;   // [audit lens-A fix] remember the TRUE size
    m_allocClusterTotal =
        totalCluster / BANKSIZE * BANKSIZE;
}

// [audit lens-A fix] Grow the OUTPUT-layer buffer group when the per-frame
// cluster count exceeds the init-time allocation. The init sizing used the
// ZERO-contact hierarchy (*1.05) and was never stored; per-frame counts from
// real contact connectivity can exceed it, and the only assertion checked
// m_clusterCap*levelnum — the capacity of a DIFFERENT scratch group — so the
// overflow was silent OOB. All these buffers are fully rewritten every
// setPreconditioner/PrepareHessian, so a DISCARDING grow is legal; the binned
// device symbols must be re-bound after the realloc (they cache raw pointers).
void MASPreconditioner::ensureOutputClusterCapacity(int need)
{
    if(need <= m_outputClusterCap)
        return;
    ++pcg_buffer_generation();   // [B2'-b] apply-path pointers move: cached PCG graph is stale
    int newCap = need + need / 16 + BANKSIZE;             // ~6% headroom
    newCap     = ((newCap + BANKSIZE - 1) / BANKSIZE) * BANKSIZE;
    printf("[MAS][grow] output cluster buffers: %d -> %d (frame cluster count %d "
           "exceeded the init-time allocation)\n",
           m_outputClusterCap, newCap, need);
#ifdef SYME
    CUDA_SAFE_CALL(cudaFree(d_inverseMatMas));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_inverseMatMas,
                              (size_t)(newCap / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaFree(d_MatMas));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_MatMas,
                              (size_t)(newCap / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixT)));
#endif
    CUDA_SAFE_CALL(cudaFree(d_precondMatMas));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_precondMatMas,
                              (size_t)(newCap / BANKSIZE) * sizeof(__GEIGEN__::MasMatrixSymf)));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelR));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelR, (size_t)newCap * sizeof(Eigen::Vector3f)));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelZ));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelZ, (size_t)newCap * sizeof(Precision_T3)));
    CUDA_SAFE_CALL(cudaFree(d_mRbin));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mRbin,
                              (size_t)3 * newCap * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaFree(d_mZbin));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mZbin,
                              (size_t)3 * newCap * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_mRbin, &d_mRbin, sizeof(double*)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_mZbin, &d_mZbin, sizeof(double*)));
    CUDA_SAFE_CALL(cudaFree(d_matbin));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_matbin,
                              (size_t)(newCap / BANKSIZE) * MAS_NB * 9 * BINNED_K
                                  * sizeof(double)));
    CUDA_SAFE_CALL(
        cudaMemcpyToSymbol(g_matbin, &d_matbin, sizeof(double*)));
    m_outputClusterCap = newCap;
    if(m_allocClusterTotal > 0)
        m_allocClusterTotal =
            newCap / BANKSIZE * BANKSIZE;
}

void MASPreconditioner::FreeMAS()
{
    if(totalNodes < 1)
        return;
    auto release = [](auto*& pointer)
    {
        if(pointer)
        {
            CUDA_SAFE_CALL(cudaFree(pointer));
            pointer = nullptr;
        }
    };
    // [per-env MAS] env-segmentation scratch (allocated unconditionally in init)
    release(d_envBase);
    release(d_envStart);
    release(d_padTot);
    release(d_denseLevel);
    release(d_coarseSpaceTables);
    release(d_coarseTable);
    release(d_levelSize);
    release(d_goingNext);
    release(d_prefixOriginal);
    release(d_nextPrefix);
    release(d_nextPrefixSum);
    release(d_scanTemp);
    m_scanTempBytes = 0;
    release(d_prefixSumOriginal);
    release(d_fineConnectMask);
    release(d_nextConnectMask);
    release(d_neighborList);
    release(d_neighborListInit);
    release(d_neighborStart);
    release(d_neighborStartTemp);
    release(d_neighborNum);
    release(d_neighborNumInit);
    release(d_partId_map_real);
    release(d_real_map_partId);
#ifdef SYME
    release(d_inverseMatMas);
#else
    release(d_MatMas);
#endif

    release(d_precondMatMas);
    release(d_multiLevelR);
    release(d_multiLevelZ);
    release(d_mRbin);
    release(d_mZbin);
    release(d_matbin);
    totalNodes = 0;
    totalMapNodes = 0;
    totalNumberClusters = 0;
    m_clusterCap = 0;
    m_outputClusterCap = 0;
    m_allocClusterTotal = 0;
}
