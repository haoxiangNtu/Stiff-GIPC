// ============================================================================
// gipc_modules/05 — post-E1d residue: the close-val family (FROZEN dead-mirror
// context, untouched), _reduct_MSelfDist, and the _ec_emit contact-force
// export helper (position load-bearing: defined before the barrier gradient
// that uses it, which now lives in energy/15_barrier.inl at the old module-07
// slot). Friction/barrier gradient kernels moved to energy/16 / energy/15.
// ============================================================================
__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number,
                                 const uint32_t* d_live = nullptr)
{
    // [C4-b] capacity-grid launch inside the frame graph: the live DCD pair
    // count is read on device (host mirrors stay frozen).
    if(d_live)
        number = static_cast<int>(*d_live);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < dTol)
    {
        int tidx                   = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx]  = dist2;
    }
}

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number,
                                   int*           _isChange_grp = nullptr,
                                   const int*     p2g           = nullptr,
                                   const uint32_t* d_live       = nullptr)
{
    // [C4-b] the previous iteration's close-set count, read on device.
    if(d_live)
        number = static_cast<int>(*d_live);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _close_collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < _close_collisionVal[idx])
    {
        *_isChange = 1;
        // [multi-env per-group κ] flag only THIS pair's env (intra-env after P1).
        if(_isChange_grp && p2g)
        { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) _isChange_grp[_gg] = 1; }  /* [-1 guard] */
    }
}


__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    double2 temp = make_double2(0.0, 0.0);
    if(idx < number)
    {
        int4   MMCVIDI = _collisionPairs[idx];
        double tempv   = _selfConstraintVal(_vertexes, MMCVIDI);
        temp = make_double2(1.0 / tempv, tempv);
    }
    int     warpTid = threadIdx.x % 32;
    int     warpId  = (threadIdx.x >> 5);
    double  nextTp;
    int     warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
        double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
        temp.x         = std::max(temp.x, tempMin);
        temp.y         = std::max(temp.y, tempMax);
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = (warpTid < warpNum) ? sdata[warpTid] : make_double2(0.0, 0.0);

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
            double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
            temp.x         = std::max(temp.x, tempMin);
            temp.y         = std::max(temp.y, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _queue[blockIdx.x] = temp;
    }
}

#include "energy/term_common.cuh"  // [E3.6] _ec_emit hoisted (shared with term TUs)

