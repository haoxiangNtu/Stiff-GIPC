__global__ void __inverse6_P96x96(__GEIGEN__::MasMatrixSymf* _preMatrix,
                                  __GEIGEN__::MasMatrixSymT* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // [audit lens-B fix] NO early return: the Gauss-Jordan loop below runs 7
    // __syncthreads() per pivot. With 96 threads/block handling 2 matrices, a
    // half-full last block (numbers % 96 == 48, i.e. an odd bank count — a
    // natural coarse-level terminal state) would have lanes 48-95 exit before
    // the barriers while lanes 0-47 keep hitting them: barrier-divergence UB
    // (documented hang). Out-of-range threads stay resident on an identity
    // phantom matrix in their own shared slab and skip only the global reads
    // and the final writeback (the inRange idiom of the _new kernels).
    const bool inRange = idx < numbers;

    int matId       = idx / (BANKSIZE * 3);
    int i           = idx % (BANKSIZE * 3);
    int block_matId = threadIdx.x / (BANKSIZE * 3);

    __shared__ double sPMas[32 / BANKSIZE][BANKSIZE * 3][BANKSIZE * 3];
    __shared__ double colm[32 / BANKSIZE][BANKSIZE * 3];

    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(!inRange)
        {   // phantom slab: zeros (diagonal fixed to 1 below) — Gauss-Jordan
            // on the identity, every pivot rt = 1, no NaN, no global access.
            sPMas[block_matId][j][i] = 0.0;
        }
        else if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](j % 3, i % 3);
        }
        else
        {
            index = BANKSIZE * colId - colId * (colId + 1) / 2 + rowId;
            sPMas[block_matId][j][i] = _invMatrix[matId].M[index](i % 3, j % 3);
        }
        if(i == j)
        {
            if(sPMas[block_matId][j][i] == 0)
            {
                sPMas[block_matId][j][i] = 1;
            }
        }
    }

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();

        rt = sPMas[block_matId][j][j];

        colm[block_matId][i] = sPMas[block_matId][i][j];

        __syncthreads();
        if(i == j)
        {

            sPMas[block_matId][i][j] = 1;
        }
        else
        {
            sPMas[block_matId][i][j] = 0;
        }
        __syncthreads();
        sPMas[block_matId][j][i] /= rt;

        __syncthreads();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                sPMas[block_matId][k][i] += rate * sPMas[block_matId][j][i];
            }
        }

        j++;
    }
    __syncthreads();
    if(i % 3 < 2)
        sPMas[block_matId][i + 1][i] = sPMas[block_matId][i][i + 1];
    else
        sPMas[block_matId][i][i - 2] = sPMas[block_matId][i - 2][i];
    __syncthreads();
    //__threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(inRange && colId >= rowId)   // [audit lens-B fix] phantom threads never write back
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            _preMatrix[matId].M[index](j % 3, i % 3) = sPMas[block_matId][j][i];
        }
    }
}


__global__ void __buildMultiLevelR_optimized_new(const double3* _R,
                                                 Eigen::Vector3f*  _multiLR,
                                                 int*           _goingNext,
                                                 int*           _prefixOrigin,
                                                 unsigned int*  _fineConnectMsk,
                                                 int* _partId_map_real,
                                                 int  levelNum,
                                                 int  numbers)
{
    int pdx = blockIdx.x * blockDim.x + threadIdx.x;
    // NO early return: the fast path below runs full-warp collectives, so every
    // lane (including out-of-range tail lanes, idx=-1, zero contribution) must
    // reach them. numbers is a multiple of BANKSIZE, so out-of-range coverage
    // is whole banks, never a partial bank.
    const bool inRange = pdx < numbers;

    Eigen::Vector3f r;
    int             idx = inRange ? _partId_map_real[pdx] : -1;
    if(idx >= 0)
    {

        r[0] = _R[idx].x;
        r[1] = _R[idx].y;
        r[2] = _R[idx].z;
    }
    else
    {
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
    }

    int laneId      = threadIdx.x % BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int gwarpId     = pdx / BANKSIZE;
    int level       = 0;
    //int rdx         = _real_map_partId[idx];
    if(inRange)
        _multiLR[pdx] = r;

    __shared__ FloatP c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    __shared__ int prefixSum[DEFAULT_WARPNUM];

    // [MAS determinism] mask of the real (idx>=0) lanes in this warp, captured while ALL 32 lanes
    // are still converged. Used below to drive a deterministic per-cluster residual reduction that
    // replaces the order-dependent float atomicAdd (which broke strict cross-env / run-to-run
    // bit-identity). Padding lanes (idx<0) are excluded and never participate in the warp collective.
    unsigned int _activeMsk = __ballot_sync(0xffffffffu, idx >= 0);

    if(laneId == 0)
    {
        // out-of-range banks: mark tree-compatible (their lanes carry zeros)
        prefixSum[localWarpId] = inRange ? _prefixOrigin[gwarpId] : 1;
    }
    __syncwarp();   // publish prefixSum to the whole warp (no early return above)

    // [fast path, corrected] non-strict only: when every bank of this warp is a
    // single cluster, an interval shuffle-tree (segments delimited by bank
    // starts) + one deposit per bank is the cheap aggregation. The OLD tree ran
    // with mask=__activemask() while padding lanes had already diverged out —
    // __shfl_down_sync then sourced lanes OUTSIDE the mask: CUDA-undefined
    // register garbage that varied with launch shape (the batch-drift bug).
    // Corrected: no early return, ALL 32 lanes participate under a full mask,
    // padding/out-of-range lanes contribute exact zeros. Strict mode
    // (g_det_reduce) keeps the deterministic ascending-lane reduction below —
    // its per-cluster sums feed exact order-independent binned deposits.
    const bool warpTree =
        (!g_det_reduce)
        && __all_sync(0xffffffffu, prefixSum[localWarpId] == 1);
    if(warpTree)
    {
        // Banks are BANKSIZE-aligned within the physical warp, so the segment
        // geometry is static: no ballot/brev/clz needed (the old
        // `mark << (warpId+1)` computed a 32-bit shift by 32 on physical
        // lane 31 — C++ UB). interval = distance to my bank's end.
        const bool     bBoundary = (laneId == 0);
        const unsigned interval  = (BANKSIZE - 1) - laneId;

        for(int iter = 1; iter < BANKSIZE; iter <<= 1)
        {
            float tmpx = __shfl_down_sync(0xffffffffu, r[0], iter);
            float tmpy = __shfl_down_sync(0xffffffffu, r[1], iter);
            float tmpz = __shfl_down_sync(0xffffffffu, r[2], iter);
            if(interval >= (unsigned)iter)
            {
                r[0] += tmpx;
                r[1] += tmpy;
                r[2] += tmpz;
            }
        }
        // bank-start lanes hold their bank's sum; the map fills real slots
        // from lane 0, so an in-range bank start always has idx >= 0.
        if(bBoundary && idx >= 0)
        {
            while(level < levelNum - 1)
            {
                level++;
                idx = _goingNext[idx];
                binned_deposit(g_mRbin + ((size_t)idx * 3 + 0) * BINNED_K, (double)r[0]);
                binned_deposit(g_mRbin + ((size_t)idx * 3 + 1) * BINNED_K, (double)r[1]);
                binned_deposit(g_mRbin + ((size_t)idx * 3 + 2) * BINNED_K, (double)r[2]);
            }
        }
        return;
    }

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        // Strict mode, or a warp whose banks are not all single-cluster:
        // deterministic ascending-lane reduction (handles prefix==1 too, since
        // connectMsk is then the bank's full real-slot mask).
        {
            // [MAS determinism] Deterministic replacement for the old order-dependent
            //   atomicAdd(c_sumResidual + warp*BANKSIZE + elected_lane, r)
            // which summed each cluster's lane residuals into its elected lane in
            // hardware-scheduling order → different FP rounding per env / per run → broke
            // strict cross-env & run-to-run bit-identity (P=0 diagonal was bit-exact, P=1 MAS
            // diverged from the 2nd Newton solve). Now: every real lane writes its own residual
            // into its own slot (no contention), then each cluster's elected lane sums its
            // members in ASCENDING lane order (reproducible). Matches the binned-determinism
            // design already used on the if-branch and the Sym6 apply.
            c_sumResidual[threadIdx.x]                         = r[0];
            c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = r[1];
            c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = r[2];
            __syncwarp(_activeMsk);

            unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
            if(electedPrefix == 0)
            {
                FloatP sx = 0, sy = 0, sz = 0;
                int    base = localWarpId * BANKSIZE;
                for(unsigned int m = connectMsk; m; m &= (m - 1))
                {
                    int l = __ffs(m) - 1;
                    sx += c_sumResidual[base + l];
                    sy += c_sumResidual[base + l + DEFAULT_BLOCKSIZE];
                    sz += c_sumResidual[base + l + 2 * DEFAULT_BLOCKSIZE];
                }
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    binned_deposit(g_mRbin + ((size_t)idx * 3 + 0) * BINNED_K, (double)sx);
                    binned_deposit(g_mRbin + ((size_t)idx * 3 + 1) * BINNED_K, (double)sy);
                    binned_deposit(g_mRbin + ((size_t)idx * 3 + 2) * BINNED_K, (double)sz);
                }
            }
        }
    }
}


__global__ void __collectFinalZ_new(double3*                  _Z,
                                    const Precision_T3*       d_multiLevelZ,
                                    const __GEIGEN__::itable* _coarseTable,
                                    int*                      _real_map_partId,
                                    int                       levelnum,
                                    int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Precision_T3 cz;  // = d_multiLevelZ[idx];
    int          rdx            = _real_map_partId[idx];
    cz.x                        = d_multiLevelZ[rdx].x;
    cz.y                        = d_multiLevelZ[rdx].y;
    cz.z                        = d_multiLevelZ[rdx].z;
    __GEIGEN__::itable table    = _coarseTable[idx];
    int*               tablePtr = table.index;
    for(int i = 1; i < levelnum; i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}

// Fused mZ-bin combine + hierarchy gather. The explicit float cast preserves
// _mas_comb_mZ's double->float materialization before CollectFinalZ performs
// its fixed-order level sum.
__device__ __forceinline__ float _mas_comb_component(const double* bin,
                                                      int           node,
                                                      int           component)
{
    return static_cast<float>(binned_combine(
        bin + ((size_t)node * 3 + component) * BINNED_K));
}

__global__ void __collectFinalZ_binned_new(double3*                  Z,
                                           const double*             mZbin,
                                           const __GEIGEN__::itable* coarseTable,
                                           const int*                realMapPartId,
                                           int                       levelNum,
                                           int                       number,
                                           int                       clusterCount,
                                           size_t                    clusterCapacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    // v0.8.4.2 can pad each environment independently, so table entries are
    // cumulative hierarchy IDs rather than values bounded by totalNodes.
    // Check both the current logical range and m_clusterCap's physical range
    // before dereferencing a binned cluster slot.
    int  rdx   = realMapPartId[idx];
    bool valid = rdx >= 0 && rdx < clusterCount
                 && static_cast<size_t>(rdx) < clusterCapacity;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    if(valid)
    {
        cx = _mas_comb_component(mZbin, rdx, 0);
        cy = _mas_comb_component(mZbin, rdx, 1);
        cz = _mas_comb_component(mZbin, rdx, 2);
    }

    const int* table = coarseTable[idx].index;
    for(int level = 1; level < levelNum && valid; ++level)
    {
        int node = table[level - 1];
        valid    = node >= 0 && node < clusterCount
                   && static_cast<size_t>(node) < clusterCapacity;
        if(valid)
        {
            cx += _mas_comb_component(mZbin, node, 0);
            cy += _mas_comb_component(mZbin, node, 1);
            cz += _mas_comb_component(mZbin, node, 2);
        }
    }

    // An invalid internally-generated hierarchy must never turn into an OOB
    // read or a plausible zero correction. Preserve the solver's fail-fast
    // behavior by surfacing a NaN if an individual table entry is corrupt.
    if(!valid)
    {
        Z[idx] = make_double3(CUDART_NAN, CUDART_NAN, CUDART_NAN);
        return;
    }
    Z[idx] = make_double3(static_cast<double>(cx),
                          static_cast<double>(cy),
                          static_cast<double>(cz));
}



