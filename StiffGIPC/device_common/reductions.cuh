#pragma once
// ============================================================================
// device_common/reductions.cuh — the ONE block-reduction implementation
// (v0.8.6 Phase 2a; docs/V086_REFACTOR_PLAN.md).
//
// Replaces the reduction tail that was copy-pasted across the *_Reduction /
// _reduct_* kernel family in GIPC.cu — the copy-paste is WHY the c3087a7 UB
// sweep missed twins in other files (_reduct_max_box). One implementation,
// one place to audit.
//
// The instruction sequence is kept IDENTICAL to the historical tail so the
// conversion is bitwise-neutral (verified by the strict anchor gate):
//   - full-mask warp shuffle ladder i = 1,2,4,8,16 (fixed associativity);
//   - lane 0 of each warp deposits to shared tep[warpId]; __syncthreads;
//   - warp 0 combines the warpNum partials (padding lanes read the neutral);
//   - thread 0 writes the block result.
//
// CALLER CONTRACT (the c3087a7 neutral-participation rules):
//   - every thread of the block must be RESIDENT at the call (no early
//     returns before it) — out-of-range threads carry the neutral value;
//   - `tep` is shared memory with >= blockDim.x/32 slots;
//   - the call must be the LAST statement of the kernel (warps other than
//     warp 0 leave the function at the internal barrier-free point).
// ============================================================================

__device__ __forceinline__ void gipc_block_sum_to(double  temp,
                                                  double* tep,
                                                  int     number,
                                                  int     idof,
                                                  double* out_slot)
{
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        temp = (warpTid < warpNum) ? tep[warpTid] : 0.0;
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        *out_slot = temp;
    }
}

// ---------------------------------------------------------------------------
// MIN/MAX variants (Phase 2a-2). Two historical shapes exist:
//  - *_full_to: the second stage runs INSIDE `if(warpId == 0)` with a full
//    32-lane ladder over neutral-guarded shared loads (min/max are idempotent
//    over the neutral, so the fixed 32 ladder is exact);
//  - gipc_block_max_tail_to: the sum-family shape (early warp retire +
//    warpNum-bounded ladder, neutral 0.0) used by the cfl/injective kernels.
// Operand order std::min(temp, other) / std::max(temp, other) is preserved
// EXACTLY (NaN propagation semantics depend on it).
// Same caller contract as gipc_block_sum_to.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void gipc_block_min_full_to(double  temp,
                                                       double* tep,
                                                       int     number,
                                                       int     idof,
                                                       double  neutral,
                                                       double* out_slot)
{
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double other = __shfl_down_sync(0xffffffff, temp, i);
        temp         = std::min(temp, other);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(warpId == 0)
    {
        temp = (warpTid < warpNum) ? tep[warpTid] : neutral;
        for(int i = 1; i < 32; i = (i << 1))
        {
            double other = __shfl_down_sync(0xffffffff, temp, i);
            temp         = std::min(temp, other);
        }
        if(warpTid == 0)
            *out_slot = temp;
    }
}

__device__ __forceinline__ void gipc_block_max_full_to(double  temp,
                                                       double* tep,
                                                       int     number,
                                                       int     idof,
                                                       double  neutral,
                                                       double* out_slot)
{
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double other = __shfl_down_sync(0xffffffff, temp, i);
        temp         = std::max(temp, other);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(warpId == 0)
    {
        temp = (warpTid < warpNum) ? tep[warpTid] : neutral;
        for(int i = 1; i < 32; i = (i << 1))
        {
            double other = __shfl_down_sync(0xffffffff, temp, i);
            temp         = std::max(temp, other);
        }
        if(warpTid == 0)
            *out_slot = temp;
    }
}

__device__ __forceinline__ void gipc_block_max_tail_to(double  temp,
                                                       double* tep,
                                                       int     number,
                                                       int     idof,
                                                       double* out_slot)
{
    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double other = __shfl_down_sync(0xffffffff, temp, i);
        temp         = std::max(temp, other);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        temp = (warpTid < warpNum) ? tep[warpTid] : 0.0;
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double other = __shfl_down_sync(0xffffffff, temp, i);
            temp         = std::max(temp, other);
        }
    }
    if(threadIdx.x == 0)
    {
        *out_slot = temp;
    }
}
