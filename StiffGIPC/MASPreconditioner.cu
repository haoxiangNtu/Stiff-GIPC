//
// MASPreconditioner.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "MASPreconditioner.cuh"
#include <linear_system/utils/binned_reduce.cuh>  // [4.3] MAS determinism: binned reproducible FP
#include "cuda_tools/cuda_tools.h"
#include "device_launch_parameters.h"
#include <muda/launch/launch.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cooperative_groups.h"
using namespace cooperative_groups;
//#include "Eigen/Eigen"
using namespace std;
#define SYME
#define GROUP

// [4.3] MAS determinism: binned reproducible-FP accumulation for the three non-deterministic
// float/double atomicAdd targets — d_multiLevelR (restrict), d_multiLevelZ (Schwarz apply),
// d_inverseMatMas (coarse Hessian aggregation). Order-independent ⇒ bit-identical. The bins are
// set via cudaMemcpyToSymbol before the depositing kernels; combined back by the kernels below.
#define MAS_NB (BANKSIZE * (BANKSIZE + 1) / 2)
__device__ double* g_mRbin  = nullptr;
__device__ double* g_mZbin  = nullptr;
__device__ double* g_matbin = nullptr;

__global__ void _mas_comb_mZ(Precision_T3* mZ, const double* bin, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    mZ[i].x = binned_combine(bin + ((size_t)i * 3 + 0) * BINNED_K);
    mZ[i].y = binned_combine(bin + ((size_t)i * 3 + 1) * BINNED_K);
    mZ[i].z = binned_combine(bin + ((size_t)i * 3 + 2) * BINNED_K);
}
__global__ void _mas_comb_mR(Eigen::Vector3f* mR, const double* bin, int start, int end)
{
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= end) return;
    mR[i][0] = (float)binned_combine(bin + ((size_t)i * 3 + 0) * BINNED_K);
    mR[i][1] = (float)binned_combine(bin + ((size_t)i * 3 + 1) * BINNED_K);
    mR[i][2] = (float)binned_combine(bin + ((size_t)i * 3 + 2) * BINNED_K);
}
__global__ void _mas_comb_mat(__GEIGEN__::MasMatrixSymT* mat, const double* bin, int startC, int endC)
{
    int t    = blockIdx.x * blockDim.x + threadIdx.x;
    int nblk = (endC - startC) * MAS_NB;
    if(t >= nblk) return;
    int cPid  = startC + t / MAS_NB;
    int index = t % MAS_NB;
#pragma unroll
    for(int i = 0; i < 3; i++)
#pragma unroll
        for(int j = 0; j < 3; j++)
            mat[cPid].M[index](i, j) =
                binned_combine(bin + (((size_t)cPid * MAS_NB + index) * 9 + i * 3 + j) * BINNED_K);
}

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
    if(tdx >= vertNum)
        return;
    int warpId      = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = tdx % BANKSIZE;

    int idx = _partId_map_real[tdx];


    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectedMsk[idx];
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
    if(tdx >= number)
        return;
    int warpId      = tdx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = tdx % BANKSIZE;

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
    int idx = _partId_map_real[tdx];
    if(idx >= 0)
    {

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
    if(tdx >= number)
        return;
    int            warpId      = tdx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = tdx % BANKSIZE;
    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    int            idx = _partId_map_real[tdx];
    if(idx >= 0)
    {

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
}


__global__ void _nextLevelCluster(unsigned int* _nextConnectedMsk, unsigned int* _nextPrefix, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int            warpId      = idx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = idx % BANKSIZE;
    __shared__ int prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    unsigned int connMsk = (1U << laneId);

    connMsk |= _nextConnectedMsk[idx];

    //unsigned int cachedMsk = connMsk;

    __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
    cachedMsk[threadIdx.x] = connMsk;
    unsigned int visited   = (1U << laneId);

    while(true)
    {
        unsigned int todo = visited ^ connMsk;

        if(!todo)
            break;

        unsigned int nextVisit = __ffs(todo) - 1;

        visited |= (1U << nextVisit);

        connMsk |= cachedMsk[nextVisit + localWarpId * BANKSIZE];  //__shfl_sync(0xffffffff, cachedMsk, nextVisit);
    }

    _nextConnectedMsk[idx] = connMsk;

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
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
    if(idx >= number)
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

    if(idx == number - 1)
    {
        _levelSize[level + 1].x = _nextPrefixSum[warpId] + _nextPrefix[warpId];
        _levelSize[level + 1].y = levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _nextConnectMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _nextPrefixSum[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);

    _nextConnectMsk[idx] = theLanePrefix;
    _goingNext[idx + levelBegin] =
        theLanePrefix + levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
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




__global__ void __inverse6_P96x96(__GEIGEN__::MasMatrixSymf* _preMatrix,
                                  __GEIGEN__::MasMatrixSymT* _invMatrix,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

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
        if(colId >= rowId)
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
        if(colId >= rowId)
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
    if(pdx >= numbers)
        return;

    Eigen::Vector3f r;
    int             idx = _partId_map_real[pdx];
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
        prefixSum[localWarpId] = _prefixOrigin[gwarpId];
    }

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        if(prefixSum[localWarpId] == 1)
        {
            auto mask_val  = __activemask();
            int  warpId    = threadIdx.x & 0x1f;
            bool bBoundary = (laneId == 0) || (warpId == 0);

            unsigned int mark     = __ballot_sync(mask_val, bBoundary);
            mark                  = __brev(mark);
            int          clzlen   = __clz(mark << (warpId + 1));
            unsigned int interval = std::min(clzlen, 31 - warpId);


            for(int iter = 1; iter < BANKSIZE; iter <<= 1)
            {
                float tmpx = __shfl_down_sync(mask_val, r[0], iter);
                float tmpy = __shfl_down_sync(mask_val, r[1], iter);
                float tmpz = __shfl_down_sync(mask_val, r[2], iter);
                if(interval >= iter)
                {
                    r[0] += tmpx;
                    r[1] += tmpy;
                    r[2] += tmpz;
                }
            }
            //int level = 0;

            if(bBoundary)
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
        else
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



__global__ void _schwarzLocalXSym3(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    int r3id = MRid % 3;

    int    lvrid = vrid % BANKSIZE;
    int    lvcid = vcid % BANKSIZE;
    FloatP rdata = 0;

    __shared__ Eigen::Vector3f smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(lvcid >= lvrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index](r3id, 0) * smR[lvcid][0]
                + Pred[Hid].M[index](r3id, 1) * smR[lvcid][1]
                + Pred[Hid].M[index](r3id, 2) * smR[lvcid][2];
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index](0, r3id) * smR[lvcid][0]
                + Pred[Hid].M[index](1, r3id) * smR[lvcid][1]
                + Pred[Hid].M[index](2, r3id) * smR[lvcid][2];
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}


__global__ void _schwarzLocalXSym6(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * BANKSIZE);

    int Hid   = idx / hessianSize;
    int lvrid = (idx % hessianSize) / (BANKSIZE);
    int lvcid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + lvrid;
    int vcid = Hid * BANKSIZE + lvcid;

    Eigen::Vector3f rdata;
    //rdata.setZero();

    __shared__ Eigen::Vector3f smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(vcid >= vrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index] * smR[lvcid];
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].transpose() * smR[lvcid];
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata[0], iter);
        FloatP tmpy = __shfl_down_sync(0xffffffff, rdata[1], iter);
        FloatP tmpz = __shfl_down_sync(0xffffffff, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        // [4.3] binned deposit instead of float atomicAdd → order-independent ⇒ deterministic
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 0) * BINNED_K, (double)rdata[0]);
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 1) * BINNED_K, (double)rdata[1]);
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 2) * BINNED_K, (double)rdata[2]);
    }
}


__device__ void get_index(int& row, int& col, const int& hash, const int& size)
{
    //row = 0;
    for(row = 0; row < size; row++)
    {
        col = hash - size * row + row * (row + 1) / 2;
        if(col >= 0 && col < size)
        {
            if(size * row - row * (row + 1) / 2 + col == hash)
                return;
        }
    }
}


__global__ void _schwarzLocalXSym9(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * (1 + BANKSIZE)) / 2;

    int Hid   = idx / hessianSize;
    int index = (idx % hessianSize);
    int lvrid, lvcid;
    get_index(lvrid, lvcid, index, BANKSIZE);

    int vrid = Hid * BANKSIZE + lvrid;
    int vcid = Hid * BANKSIZE + lvcid;

    __shared__ int row_ids[BANKSIZE * BANKSIZE];
    row_ids[threadIdx.x] = vrid;

    __syncthreads();
    int prev_i = -1;
    if(threadIdx.x > 0)
    {
        prev_i = row_ids[threadIdx.x - 1];
    }

    auto block_value = Pred[Hid].M[index];
    Eigen::Vector3f rdata = block_value * mR[vcid];

    if(vrid != vcid)  // process lower triangle
    {
        Eigen::Vector3f vec_ =
            block_value.transpose() * mR[vrid];

        atomicAdd((&(mZ[vcid].x)), vec_[0]);
        atomicAdd((&(mZ[vcid].y)), vec_[1]);
        atomicAdd((&(mZ[vcid].z)), vec_[2]);
    }


    int warpId = threadIdx.x & 0x1f;
    //int lane_id = threadIdx.x % BANKSIZE;

    bool bBoundary = (warpId == 0) || (prev_i != vrid);
    auto mask_val  = __activemask();

    unsigned int mark     = __ballot_sync(mask_val, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    mark = interval;
    for(int iter = 1; iter & 0x1f; iter <<= 1)
    {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        if(tmp > mark)
            mark = tmp;
    }
    int maxSize = __shfl_sync(mask_val, mark, 0);
    //__syncthreads();

    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        float tmpx = __shfl_down_sync(mask_val, rdata[0], iter);
        float tmpy = __shfl_down_sync(mask_val, rdata[1], iter);
        float tmpz = __shfl_down_sync(mask_val, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x)), rdata[0]);
        atomicAdd((&(mZ[vrid].y)), rdata[1]);
        atomicAdd((&(mZ[vrid].z)), rdata[2]);
    }
}


__global__ void _buildCollisionConnection_new(unsigned int* _pConnect,
                                              const int*    _pCoarseSpaceTable,
                                              const const int4* _collisionPair,
                                              const int* _real_map_partId,
                                              int        level,
                                              int        node_offset,
                                              int        vertNum,
                                              int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collitionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }

        for(int i = 0; i < 4; i++)
            collitionPairStartId[i] -= node_offset;

        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] =
                        _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                else
                    cpVid[i] = -1;
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                else
                    cpVid[i] = -1;
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId || myId < 0 || otId < 0)
                {
                    continue;
                }
                if(myId / BANKSIZE == otId / BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(cpVid[i] >= 0)
                {
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
                }
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                {
                    atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                }
        }
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];

                for(int i = 0; i < 2; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;
                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                for(int i = 0; i < 3; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            for(int i = 0; i < 4; i++)
                collitionPairStartId[i] -= node_offset;
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                    else
                        cpVid[i] = -1;
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                    else
                        cpVid[i] = -1;
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId || myId < 0 || otId < 0)
                    {
                        continue;
                    }
                    if(myId / BANKSIZE == otId / BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % BANKSIZE));
                        connMsk[j] |= (1U << (myId % BANKSIZE));
                    }
                }
            }

            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(cpVid[i] >= 0)
                    {
                        atomicOr(_pConnect + cpVid[i], connMsk[i]);
                    }
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                    {
                        atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                    }
            }
        }
    }
}


// [per-env MAS] Env-segment an already-exclusive-scanned per-warp prefix so each env's coarse
// clusters occupy a BANKSIZE-aligned block → envs never share a bank at any level → the
// block-diagonal Schwarz smoother stays intra-env → strict cross-env / batch bit-identity.
// Homogeneous multi-env: env e owns warps [e*wpe,(e+1)*wpe). ALL device-side (integer, exact →
// zero determinism impact; no host round-trip, no sync — the earlier host cudaMemcpy version was
// only an implementation shortcut, not a determinism requirement).
//
// _mas_env_base: n_env is small → single thread. Computes each env's aligned base offset + its
// pre-mutation scan start (envStart, so _apply has no read-after-write hazard) + the padded total.
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
        n_env = atoi(force);            // 0=off, 1=auto(body-group count), N=force N
        if(n_env == 1)
            n_env = numEnvs;
    }
    else
    {
        n_env = numEnvs;                // default: ON for multi-env, every mode (see rationale above)
    }
    if(n_env <= 1 || warpNum <= 0 || warpNum % n_env != 0)
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
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
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
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
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
    int number    = h_clevelSize.x;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _nextLevelCluster<<<numBlocks, blockSize>>>(d_nextConnectMask, d_nextPrefix, number);
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
    int number     = h_clevelSize.x;
    if(number < 1)
        return;
    int levelBegin = h_clevelSize.y;
    int blockSize  = BANKSIZE * BANKSIZE;
    int numBlocks  = (number + blockSize - 1) / blockSize;

    int warpNum = (number + BANKSIZE - 1) / BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(d_nextPrefix),
                           thrust::device_ptr<unsigned int>(d_nextPrefix) + warpNum,
                           thrust::device_ptr<unsigned int>(d_nextPrefixSum));
    // [per-env MAS] pad each env's level-(level+1) clusters to a BANKSIZE-aligned block.
    int _segN = _mas_envSegN(warpNum, m_numEnvs);
    if(_segN > 1)
    {
        int wpe = warpNum / _segN;
        _mas_env_base<<<1, 1>>>(d_nextPrefixSum, d_nextPrefix, wpe, _segN, d_envBase, d_envStart, d_padTot);
        _mas_env_apply<<<(warpNum + 255) / 256, 256>>>(d_nextPrefixSum, d_envBase, d_envStart, warpNum, wpe);
    }

    _prefixSumLx<<<numBlocks, blockSize>>>(
        d_levelSize, d_nextPrefix, d_nextPrefixSum, d_nextConnectMask, d_goingNext, level, levelBegin, number);
    if(_segN > 1)
        _mas_env_setx<<<1, 1>>>(d_levelSize + level + 1, d_padTot);   // padded cluster count (device)
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
int MASPreconditioner::ReorderRealtime(int cpNum)
{
    CUDA_SAFE_CALL(cudaMemset(d_levelSize, 0, levelnum * sizeof(int2)));


    BuildConnectMaskL0();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(cpNum)
        BuildCollisionConnection(d_fineConnectMask, nullptr, -1, cpNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    PreparePrefixSumL0();

    BuildLevel1();
    for(int level = 1; level < levelnum; level++)
    {
        CUDA_SAFE_CALL(cudaMemset(d_nextConnectMask, 0, totalNodes * sizeof(int)));

        BuildConnectMaskLx(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        if(cpNum)
            BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum);

        CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + level, sizeof(int2), cudaMemcpyDeviceToHost));

        NextLevelCluster(level);



        PrefixSumLx(level);

        ComputeNextLevel(level);

    }

    CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + levelnum, sizeof(int2), cudaMemcpyDeviceToHost));

    totalNumberClusters = h_clevelSize.y;

    AggregationKernel();

    return totalNumberClusters;
}


void MASPreconditioner::PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                                            int*             row_ids,
                                            int*             col_ids,
                                            uint32_t*        indices,
                                            int              offset,
                                            int              triplet_number)
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
            CUDA_SAFE_CALL(cudaMemset(
                d_matbin + (size_t)startC * MAS_NB * 9 * BINNED_K, 0,
                (size_t)(endC - startC) * MAS_NB * 9 * BINNED_K * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_matbin, &d_matbin, sizeof(double*)));
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
                 triplet_values, row_ids, col_ids] __device__(int I) mutable
                {
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

                    if((rdx >= 0) && (cdx >= 0))
                    {
                        if(prefix == 1)
                        {
                            int warpId = threadIdx.x & 0x1f;
                            bool bBoundary = (warpId == 0) || (rdx < 0) || (cdx < 0);
                            unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
                            mark = __brev(mark);
                            int clzlen = __clz(mark << (warpId + 1));
                            unsigned int interval = std::min(clzlen, 31 - warpId);
                            for(int iter = 1; iter < 32; iter <<= 1)
                            {
                                Eigen::Matrix3d matTemp;
                                for(int i = 0; i < 3; i++)
                                {
                                    for(int j = 0; j < 3; j++)
                                    {
                                        matTemp(i, j) =
                                            __shfl_down_sync(0xffffffff, mat3(i, j), iter);
                                    }
                                }
                                if(interval >= iter)
                                {
                                    mat3 = mat3 + matTemp;
                                }
                            }
                            int level = 0;
                            if(bBoundary)
                            {
                                int nextId = _goingNext[rdx];
                                while(level < levelNum - 1)
                                {
                                    level++;
                                    int cPid  = nextId / BANKSIZE;
                                    int bvRid = nextId % BANKSIZE;
                                    int bvCid = nextId % BANKSIZE;
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
                                                mat3(i, j));
                                        }
                                    }
                                    nextId = _goingNext[nextId];
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

    {  // [4.3] combine binned coarse aggregation back into d_inverseMatMas (before inversion)
        int startC = totalMapNodes / BANKSIZE;
        int endC   = totalNumberClusters / BANKSIZE;
        int nblk   = (endC - startC) * MAS_NB;
        if(nblk > 0)
            _mas_comb_mat<<<(nblk + 255) / 256, 256>>>(d_inverseMatMas, d_matbin, startC, endC);
    }

    __inverse6_P96x96<<<numBlocks2, blockSize2>>>(d_precondMatMas, d_inverseMatMas, number2);

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

void MASPreconditioner::SchwarzLocalXSym()
{
    //int matNum    = totalNumberClusters / BANKSIZE;
    int number    = totalNumberClusters * BANKSIZE * 3;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym3<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void MASPreconditioner::SchwarzLocalXSym_block3()
{
    //int matNum    = totalNumberClusters / BANKSIZE;
    int number = totalNumberClusters * BANKSIZE;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym6<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void MASPreconditioner::SchwarzLocalXSym_sym()
{
    int matNum    = totalNumberClusters / BANKSIZE;
    int number = matNum * (1 + BANKSIZE) * BANKSIZE / 2;
    if(number < 1)
        return;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym9<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
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
                                               int              cpNum)
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


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ReorderRealtime(cpNum);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef SYME

    CUDA_SAFE_CALL(cudaMemset(
        d_inverseMatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMemset(
        d_MatMas, 0, totalNumberClusters / BANKSIZE * sizeof(__GEIGEN__::MasMatrixT)));
#endif
    PrepareHessian_bcoo(triplet_values, row_ids, col_ids, indices, offset, triplet_num);

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
    // [MAS graph-capture] all Async on the PTDS stream: the sync cudaMemset /
    // cudaMemcpyToSymbol variants broke PCG-graph capture (symbols now bound
    // once at alloc in the MAS malloc routine).
    CUDA_SAFE_CALL(cudaMemsetAsync(d_multiLevelR + totalMapNodes,
                              0,
                              (totalNumberClusters - totalMapNodes) * sizeof(Eigen::Vector3f), 0));

    CUDA_SAFE_CALL(cudaMemsetAsync(d_multiLevelZ, 0, (totalNumberClusters) * sizeof(Precision_T3), 0));

    // [4.3] zero the binned accumulators. mR: only the COARSE
    // slots accumulate (fine [0,totalMapNodes) is set directly in __buildMultiLevelR); mZ: all.
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mRbin + (size_t)totalMapNodes * 3 * BINNED_K, 0,
                              (size_t)(totalNumberClusters - totalMapNodes) * 3 * BINNED_K
                                  * sizeof(double), 0));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_mZbin, 0,
                              (size_t)totalNumberClusters * 3 * BINNED_K * sizeof(double), 0));

    BuildMultiLevelR(R);
    {  // [4.3] combine binned coarse mR back into d_multiLevelR
        int n = totalNumberClusters - totalMapNodes;
        if(n > 0)
            _mas_comb_mR<<<(n + 255) / 256, 256>>>(d_multiLevelR, d_mRbin, totalMapNodes,
                                                   totalNumberClusters);
    }

    SchwarzLocalXSym_block3();
    {  // [4.3] combine binned mZ back into d_multiLevelZ
        int n = totalNumberClusters;
        if(n > 0)
            _mas_comb_mZ<<<(n + 255) / 256, 256>>>(d_multiLevelZ, d_mZbin, n);
    }

    CollectFinalZ(Z);
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
    computeNumLevels(maxNodes);
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
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_goingNext,
                              vertNum * levelnum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefix, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefixSum, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixSumOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fineConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextConnectMask, vertNum * sizeof(unsigned int)));
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

    int totalCluster = ReorderRealtime(0) * 1.05;
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
}

void MASPreconditioner::FreeMAS()
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaFree(d_denseLevel));
    CUDA_SAFE_CALL(cudaFree(d_coarseSpaceTables));
    CUDA_SAFE_CALL(cudaFree(d_levelSize));
    CUDA_SAFE_CALL(cudaFree(d_goingNext));
    CUDA_SAFE_CALL(cudaFree(d_prefixOriginal));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefix));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefixSum));
    CUDA_SAFE_CALL(cudaFree(d_prefixSumOriginal));
    CUDA_SAFE_CALL(cudaFree(d_fineConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_nextConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_neighborList));
    CUDA_SAFE_CALL(cudaFree(d_neighborListInit));
    CUDA_SAFE_CALL(cudaFree(d_neighborStart));
    CUDA_SAFE_CALL(cudaFree(d_neighborStartTemp));
    CUDA_SAFE_CALL(cudaFree(d_neighborNum));
    CUDA_SAFE_CALL(cudaFree(d_neighborNumInit));
    CUDA_SAFE_CALL(cudaFree(d_partId_map_real));
    CUDA_SAFE_CALL(cudaFree(d_real_map_partId));
#ifdef SYME
    CUDA_SAFE_CALL(cudaFree(d_inverseMatMas));
#else
    CUDA_SAFE_CALL(cudaFree(d_MatMas));
#endif

    CUDA_SAFE_CALL(cudaFree(d_precondMatMas));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelR));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelZ));
    if(d_mRbin) CUDA_SAFE_CALL(cudaFree(d_mRbin));
    if(d_mZbin) CUDA_SAFE_CALL(cudaFree(d_mZbin));
    if(d_matbin) CUDA_SAFE_CALL(cudaFree(d_matbin));
}
