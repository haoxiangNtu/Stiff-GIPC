// ============================================================================
// contact/pair_buffers.cu — pair-buffer growth MECHANICS bodies (own TU).
// Contract, division of responsibility, and the latent-OOB history live in
// contact/pair_buffers.cuh. Host-only code: zero device code moves here.
// ============================================================================
#include <cuda_runtime.h>
#include "cuda_tools/cuda_tools.h"
#include "contact/pair_buffers.cuh"

// Initial allocation from pre-set caps (caps must already hold the sizing).
void pair_buffers_alloc(PairBuffers b)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.mat, ((size_t)b.dcd_cap + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.dcd, ((size_t)b.dcd_cap + 1) * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.ccd, ((size_t)b.ccd_cap + 1) * sizeof(int4)));
    set_emit_caps(b.dcd_cap, b.ccd_cap);
}

// Grow the DCD pair set (pairs + assembly rank) to new_dcd. The CCD mirror is
// grown IN LOCKSTEP only when new_dcd exceeds its cap — and NEVER shrunk:
// allocation size and published cap stay equal by construction.
void pair_buffers_grow_dcd(PairBuffers b, int new_dcd)
{
    CUDA_SAFE_CALL(cudaFree(b.dcd));
    CUDA_SAFE_CALL(cudaFree(b.mat));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.dcd, ((size_t)new_dcd + 1) * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.mat, ((size_t)new_dcd + 1) * sizeof(int)));
    b.dcd_cap = new_dcd;
    if(b.dcd_cap > b.ccd_cap)
    {
        CUDA_SAFE_CALL(cudaFree(b.ccd));
        CUDA_SAFE_CALL(cudaMalloc((void**)&b.ccd, ((size_t)new_dcd + 1) * sizeof(int4)));
        b.ccd_cap = new_dcd;
    }
    set_emit_caps(b.dcd_cap, b.ccd_cap);
}

// Grow the CCD/swept pair set to new_ccd (monotone: callers only request
// growth; a request below the current cap is a no-op republish).
void pair_buffers_grow_ccd(PairBuffers b, int new_ccd)
{
    if(new_ccd > b.ccd_cap)
    {
        CUDA_SAFE_CALL(cudaFree(b.ccd));
        CUDA_SAFE_CALL(cudaMalloc((void**)&b.ccd, ((size_t)new_ccd + 1) * sizeof(int4)));
        b.ccd_cap = new_ccd;
    }
    set_emit_caps(b.dcd_cap, b.ccd_cap);
}

void pair_buffers_free(PairBuffers b)
{
    if(b.mat) { CUDA_SAFE_CALL(cudaFree(b.mat)); b.mat = nullptr; }
    if(b.dcd) { CUDA_SAFE_CALL(cudaFree(b.dcd)); b.dcd = nullptr; }
    if(b.ccd) { CUDA_SAFE_CALL(cudaFree(b.ccd)); b.ccd = nullptr; }
}
