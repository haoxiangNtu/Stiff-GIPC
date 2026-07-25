__global__ void _buildCML0(const unsigned int* _neighborStart,
                           unsigned int*       _neighborNum,
                           unsigned int*       _neighborList,
                           unsigned int*       _fineConnectedMsk,
                           int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    int          numNeighbor = _neighborNum[idx];
    unsigned int connectMsk  = (1U << laneId);
    int          nk          = 0;
    int          startId     = _neighborStart[idx];
    for(int i = 0; i < numNeighbor; i++)
    {
        int vIdConnected     = _neighborList[startId + i];
        int warpIdxConnected = vIdConnected / BANKSIZE;
        if(warpId == warpIdxConnected)
        {
            unsigned int laneIdxConnected = vIdConnected % BANKSIZE;
            connectMsk |= (1U << laneIdxConnected);
        }
        else
        {
            _neighborList[startId + nk] = vIdConnected;
            nk++;
        }
    }
    _neighborNum[idx]      = nk;
    _fineConnectedMsk[idx] = connectMsk;
}

__global__ void _buildCML0_new(const unsigned int* _neighborStart,
                               unsigned int*       _neighborNum,
                               unsigned int*       _neighborList,
                               unsigned int*       _fineConnectedMsk,
                               int*                _partId_map_real,
                               int*                _real_map_partId,
                               int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int warpId = tdx / BANKSIZE;
    int laneId = tdx % BANKSIZE;
    int idx    = _partId_map_real[tdx];
    if(idx >= 0)
    {

        int          numNeighbor = _neighborNum[idx];
        unsigned int connectMsk  = (1U << laneId);
        int          nk          = 0;
        int          startId     = _neighborStart[idx];
        for(int i = 0; i < numNeighbor; i++)
        {
            int vIdConnected = _neighborList[startId + i];
            //vIdConnected         = _real_map_partId[vIdConnected];
            int warpIdxConnected = _real_map_partId[vIdConnected] / BANKSIZE;
            if(warpId == warpIdxConnected)
            {
                unsigned int laneIdxConnected = _real_map_partId[vIdConnected] % BANKSIZE;
                connectMsk |= (1U << laneIdxConnected);
            }
            else
            {
                _neighborList[startId + nk] = vIdConnected;
                nk++;
            }
        }
        _neighborNum[idx]      = nk;
        _fineConnectedMsk[idx] = connectMsk;
    }
}


__device__ unsigned int _LanemaskLt(int laneIdx)
{
    return (1U << laneIdx) - 1;
}

__global__ void _preparePrefixSumL0(int* _prefixOriginal, unsigned int* _fineConnectedMsk, int vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          localWarpId = threadIdx.x / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    unsigned int connectMsk  = _fineConnectedMsk[idx];
    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    unsigned int visited   = (1U << laneId);
    while(connectMsk != -1)
    {
        unsigned int todo = visited ^ connectMsk;

        if(!todo)
            break;

        unsigned int nextVist = __ffs(todo) - 1;
        visited |= (1U << nextVist);
        connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
    }

    _fineConnectedMsk[idx] = connectMsk;

    unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        //prefixSum[warpId]++;
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
    {
        _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}

__global__ void _preparePrefixSumL0_new(int*          _prefixOriginal,
                                        unsigned int* _fineConnectedMsk,
                                        int*          _partId_map_real,
                                        //int*          _real_map_partId,
                                        int vertNum)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    // [racecheck] no early return: the intra-warp shared-memory handshakes
    // below need __syncwarp() with a mask every named lane actually reaches
    // (independent-thread scheduling on sm_70+ gives no implicit lockstep).
    // Out-of-range lanes carry idx=-1 and never touch global memory.
    const bool inRange     = tdx < vertNum;
    int        warpId      = tdx / BANKSIZE;
    int        localWarpId = threadIdx.x / BANKSIZE;
    int        laneId      = tdx % BANKSIZE;

    int idx = inRange ? _partId_map_real[tdx] : -1;


    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];

    unsigned int connectMsk = (idx >= 0) ? _fineConnectedMsk[idx] : 0u;
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    __syncwarp();   // publish cacheMask + zeroed prefixSum to the whole warp

    if(idx >= 0)
    {
        unsigned int visited = (1U << laneId);
        while(connectMsk != -1)
        {
            unsigned int todo = visited ^ connectMsk;

            if(!todo)
                break;

            unsigned int nextVist = __ffs(todo) - 1;
            visited |= (1U << nextVist);
            connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
        }

        _fineConnectedMsk[idx] = connectMsk;

        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            //prefixSum[warpId]++;
            atomicAdd(prefixSum + localWarpId, 1);
        }
    }
    __syncwarp();   // all cluster-count atomics land before lane 0 reads
    if(idx >= 0 && laneId == 0)
    {
        _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}


__global__ void _buildLevel1(int2*               _levelSize,
                             int*                _coarseSpaceTable,
                             int*                _goingNext,
                             const unsigned int* _fineConnectedMsk,
                             const int*          _prefixSumOriginal,
                             const int*          _prefixOriginal,
                             int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(idx == vertNum - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _fineConnectedMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    //lanePrefix2 += _prefixSumOriginal[warpId];

    //unsigned int elected_lane = __ffs(connMsk) - 1;
    //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


    _coarseSpaceTable[idx + 0 * vertNum] = theLanePrefix;
    _goingNext[idx] = theLanePrefix + (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}


__global__ void _buildLevel1_new(int2*               _levelSize,
                                 int*                _coarseSpaceTable,
                                 int*                _goingNext,
                                 const unsigned int* _fineConnectedMsk,
                                 const int*          _prefixSumOriginal,
                                 const int*          _prefixOriginal,
                                 int*                _partId_map_real,
                                 int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    // [racecheck] no early return + explicit __syncwarp() between the three
    // shared-memory phases (zero electedMask -> atomicOr votes -> lanePrefix
    // publish -> cross-lane read). Out-of-range lanes carry idx=-1.
    const bool inRange     = tdx < number;
    int        warpId      = tdx / BANKSIZE;
    int        localWarpId = threadIdx.x / BANKSIZE;
    int        laneId      = tdx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(tdx == number - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
    int          idx     = inRange ? _partId_map_real[tdx] : -1;
    unsigned int connMsk = (idx >= 0) ? _fineConnectedMsk[idx] : 0u;
    __syncwarp();   // electedMask zeros visible warp-wide

    if(idx >= 0)
    {
        unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));
        if(electedPrefix == 0)
        {
            atomicOr(electedMask + localWarpId, (1U << laneId));
        }
    }
    __syncwarp();   // all election votes landed

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId))
                              + (inRange ? _prefixSumOriginal[warpId] : 0);
    __syncwarp();   // lanePrefix published before cross-lane reads

    if(idx >= 0)
    {
        unsigned int elected_lane = __ffs(connMsk) - 1;
        unsigned int theLanePrefix =
            lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


        _coarseSpaceTable[idx] = theLanePrefix;
        _goingNext[idx] = theLanePrefix + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
}


__global__ void _buildConnectMaskLx(const unsigned int* _neighborStart,
                                    unsigned int*       _neighborNum,
                                    unsigned int*       _neighborList,
                                    int*                _coarseSpaceTable,
                                    unsigned int*       _nextConnectedMsk,
                                    const unsigned int* _fineConnectedMsk,
                                    int                 level,
                                    int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    unsigned int prefixMsk = _fineConnectedMsk[idx];
    unsigned int connMsk   = 0;
    unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
    int          kn        = _neighborNum[idx];
    int          nk        = 0;
    int          startId   = _neighborStart[idx];
    for(int i = 0; i < kn; i++)
    {
        unsigned int connect = _neighborList[startId + i];
        unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

        if(coarseIdx / BANKSIZE == coarseConnect / BANKSIZE)
        {
            unsigned int off = coarseConnect % BANKSIZE;
            connMsk |= (1U << off);
        }
        else
        {
            _neighborList[startId + nk] = connect;
            nk++;
        }
    }

    _neighborNum[idx] = nk;

    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    cacheMsk[threadIdx.x] = 0;

    if(__popc(prefixMsk) == BANKSIZE)
    {
        atomicOr(cacheMsk + localWarpId * BANKSIZE, connMsk);
        connMsk = cacheMsk[localWarpId * BANKSIZE];
        //if (laneId == 0) {
        //	cacheMsk[localWarpId] = 0;
        //}
    }
    else
    {
        unsigned int electedLane = __ffs(prefixMsk) - 1;
        if(connMsk)
        {
            atomicOr(cacheMsk + localWarpId * BANKSIZE + electedLane, connMsk);
        }
        connMsk = cacheMsk[localWarpId * BANKSIZE + electedLane];
    }

    unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

    if(connMsk && electedPrefix == 0)
    {
        atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
    }
}

__global__ void _buildConnectMaskLx_new(const unsigned int* _neighborStart,
                                        unsigned int*       _neighborNum,
                                        unsigned int*       _neighborList,
                                        int*                _coarseSpaceTable,
                                        unsigned int*       _nextConnectedMsk,
                                        const unsigned int* _fineConnectedMsk,
                                        int                 level,
                                        int*                _partId_map_real,
                                        //int*                _real_map_partId,
                                        int vertNum,
                                        int number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    // [racecheck] no early return; the shared cacheMsk handshake is split into
    // zero -> vote -> read phases separated by full-warp __syncwarp().
    const bool     inRange     = tdx < number;
    int            warpId      = tdx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = tdx % BANKSIZE;
    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    int            idx = inRange ? _partId_map_real[tdx] : -1;

    cacheMsk[threadIdx.x] = 0;
    __syncwarp();   // zeros visible before any vote

    unsigned int prefixMsk = 0;
    unsigned int connMsk   = 0;
    unsigned int coarseIdx = 0;
    if(idx >= 0)
    {
        prefixMsk = _fineConnectedMsk[idx];
        coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
        int kn      = _neighborNum[idx];
        int nk      = 0;
        int startId = _neighborStart[idx];
        for(int i = 0; i < kn; i++)
        {
            unsigned int connect = _neighborList[startId + i];
            unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

            if(coarseIdx / BANKSIZE == coarseConnect / BANKSIZE)
            {
                unsigned int off = coarseConnect % BANKSIZE;
                connMsk |= (1U << off);
            }
            else
            {
                _neighborList[startId + nk] = connect;
                nk++;
            }
        }

        _neighborNum[idx] = nk;

        if(__popc(prefixMsk) == BANKSIZE)
        {
            atomicOr(cacheMsk + localWarpId * BANKSIZE, connMsk);
        }
        else
        {
            unsigned int electedLane = __ffs(prefixMsk) - 1;
            if(connMsk)
            {
                atomicOr(cacheMsk + localWarpId * BANKSIZE + electedLane, connMsk);
            }
        }
    }
    __syncwarp();   // all votes landed before the cross-lane reads

    if(idx >= 0)
    {
        if(__popc(prefixMsk) == BANKSIZE)
        {
            connMsk = cacheMsk[localWarpId * BANKSIZE];
        }
        else
        {
            unsigned int electedLane = __ffs(prefixMsk) - 1;
            connMsk = cacheMsk[localWarpId * BANKSIZE + electedLane];
        }

        unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

        if(connMsk && electedPrefix == 0)
        {
            atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
        }
    }
}


__global__ void _nextLevelCluster(unsigned int* _nextConnectedMsk, unsigned int* _nextPrefix, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // [racecheck] no early return + __syncwarp() around the cachedMsk /
    // prefixSum handshakes. Out-of-range lanes stay self-connected zeros and
    // never touch global memory.
    const bool     inRange     = idx < number;
    int            warpId      = idx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = idx % BANKSIZE;
    __shared__ int prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    unsigned int connMsk = (1U << laneId);

    if(inRange)
        connMsk |= _nextConnectedMsk[idx];

    //unsigned int cachedMsk = connMsk;

    __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
    cachedMsk[threadIdx.x] = connMsk;
    __syncwarp();   // cachedMsk + zeroed prefixSum published warp-wide

    unsigned int visited = (1U << laneId);

    while(true)
    {
        unsigned int todo = visited ^ connMsk;

        if(!todo)
            break;

        unsigned int nextVisit = __ffs(todo) - 1;

        visited |= (1U << nextVisit);

        connMsk |= cachedMsk[nextVisit + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cachedMsk, nextVisit);
    }

    if(inRange)
        _nextConnectedMsk[idx] = connMsk;

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(inRange && electedPrefix == 0)
    {
        atomicAdd(prefixSum + localWarpId, 1);
    }
    __syncwarp();   // cluster-count votes landed before lane 0 reads

    if(inRange && laneId == 0)
        _nextPrefix[warpId] = prefixSum[localWarpId];
}

__global__ void _prefixSumLx(int2*         _levelSize,
                             unsigned int* _nextPrefix,
                             unsigned int* _nextPrefixSum,
                             unsigned int* _nextConnectMsk,
                             int*          _goingNext,
                             int           level,
                             int           levelBegin,
                             int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // [racecheck] no early return + __syncwarp() between the zero / vote /
    // publish / read phases of the electedMask & lanePrefix handshakes.
    const bool inRange     = idx < number;
    int        warpId      = idx / BANKSIZE;
    int        localWarpId = threadIdx.x / BANKSIZE;
    int        laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }

    if(idx == number - 1)
    {
        _levelSize[level + 1].x = _nextPrefixSum[warpId] + _nextPrefix[warpId];
        _levelSize[level + 1].y = levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = inRange ? _nextConnectMsk[idx] : (1U << laneId);
    __syncwarp();   // electedMask zeros visible

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(inRange && electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }
    __syncwarp();   // votes landed

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId))
                              + (inRange ? _nextPrefixSum[warpId] : 0);
    __syncwarp();   // lanePrefix published before cross-lane reads

    if(inRange)
    {
        unsigned int elected_lane = __ffs(connMsk) - 1;
        unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);

        _nextConnectMsk[idx] = theLanePrefix;
        _goingNext[idx + levelBegin] =
            theLanePrefix + levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }
}

__global__ void _computeNextLevel(int*          _coarseSpaceTable,
                                  unsigned int* _nextConnectMsk,
                                  int           level,
                                  int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int next = _coarseSpaceTable[(level - 1) * number + idx];
    _coarseSpaceTable[(level)*number + idx] = _nextConnectMsk[next];
}

__global__ void _aggregationKernel(int*                _denseLevel,
                                   __GEIGEN__::itable* _coarseTable,
                                   int*                _goingNext,
                                   int                 levelNum,
                                   int                 number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int currentId = idx;
    //int aggLevel  = levelNum - 1;
    //__shared__ int4 ctable[DEFAULT_BLOCKSIZE];
    __GEIGEN__::itable ctable;
    for(int l = 0; l < levelNum - 1; l++)
    {
        int next = _goingNext[currentId];

        //int next0 = __shfl_sync(0xffffffff, next, 0);
        //printf("%d   %d   %d    %d\n", next, next0, l,  idx);
        //if (next == next0) {
        //	aggLevel = std::min(l, aggLevel);
        //}

        currentId           = next;
        *(ctable.index + l) = next;
    }

    //_denseLevel[idx] = aggLevel;

    //printf("%d   %d\n", aggLevel, idx);

    _coarseTable[idx] = ctable;
}




