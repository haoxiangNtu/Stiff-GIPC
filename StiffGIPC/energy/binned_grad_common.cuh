// ============================================================================
// energy/binned_grad_common.cuh — binned-gradient device interface for term
// TUs (E3.2). The __device__ globals are DEFINED in
// energy/03_barrier_fused_assembly.inl (composite TU), resolved cross-TU by
// RDC; _gfxAdd/_binDepBase are verbatim hoists (inline, ODR-safe). BINNED_*
// macros come from the GIPC.cuh include chain.
// ============================================================================
#pragma once
#include "linear_system/utils/binned_reduce.cuh"  // BINNED_K/W/E0

extern __device__ double* g_gbin;
extern __device__ int     g_binned_on;
extern __device__ int     g_ground_hess_legacy;
extern __device__ int     g_bar_trace;
extern __device__ int     g_tgt0;
extern __device__ int     g_tgt1;

__device__ inline void _gfxAdd(int v, int comp, double val)
{
    if(g_bar_trace && (v == g_tgt0 || v == g_tgt1))
        printf("DEP %d %d %.17e\n", v, comp, val);
    double x    = val;
    size_t base = ((size_t)v * 3 + comp) * BINNED_K;
    if(!g_binned_on)
    {
        atomicAdd(&g_gbin[base], val);   // fast path: raw plain-atomic into bin 0 (correct, non-det order)
        return;
    }
#pragma unroll
    for(int k = 0; k < BINNED_K; ++k)
    {
        double M  = ldexp(1.5, BINNED_E0 - k * BINNED_W);   // constant args → folded
        double q  = __dadd_rn(M, x);
        double hi = __dsub_rn(q, M);
        atomicAdd(&g_gbin[base + k], hi);                   // same-grid exact ⇒ order-free
        x         = __dsub_rn(x, hi);
    }
}
// [4.3] base-pointer binned deposit (for the ABD FEM-pin coupling into g_abd_sysbin).
extern __device__ double* g_abd_sysbin;
__device__ inline void _binDepBase(double* bins, double val)
{
    double x = val;
#pragma unroll
    for(int k = 0; k < BINNED_K; ++k)
    { double M = ldexp(1.5, BINNED_E0 - k * BINNED_W); double q = __dadd_rn(M, x);
      double hi = __dsub_rn(q, M); atomicAdd(&bins[k], hi); x = __dsub_rn(x, hi); }
}
