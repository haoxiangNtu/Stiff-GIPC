#pragma once
// [multi-env determinism 4.3] Binned (reproducible) floating-point accumulation — see
// docs/internal/MULTIENV_DETERMINISM_HANDOVER.md. K exponent bins, each an EXACT fixed-point
// slice (anchor-aligned); atomic deposits into a bin are exact ⇒ order-independent ⇒
// bit-identical regardless of thread scheduling, with full dynamic range. Constants MUST match
// the gradient binning in GIPC.cu (BINNED_K/W/E0). Verified: tools/binned_reduction_test.cu.
// __dadd_rn/__dsub_rn block compiler reassociation (survive --use_fast_math).
#include <math.h>

#ifndef BINNED_K
#define BINNED_K 4
#define BINNED_W 30
#define BINNED_E0 60
#endif

// deposit `val` into the K contiguous bins at `bins` (order-independent, exact).
__device__ __forceinline__ void binned_deposit(double* bins, double val)
{
    double x = val;
#pragma unroll
    for(int k = 0; k < BINNED_K; ++k)
    {
        double M  = ldexp(1.5, BINNED_E0 - k * BINNED_W);   // constant args → folded
        double q  = __dadd_rn(M, x);
        double hi = __dsub_rn(q, M);
        atomicAdd(&bins[k], hi);
        x         = __dsub_rn(x, hi);
    }
}

// combine the K bins (finest first, fixed order) into one double.
__device__ __forceinline__ double binned_combine(const double* bins)
{
    double s = 0.0;
#pragma unroll
    for(int k = BINNED_K - 1; k >= 0; --k)
        s += bins[k];
    return s;
}

// [4.3 ABD] deterministic替代 muda atomic_add on a 12-vector at scalar offset `off` (off=body*12).
template <class V>
__device__ __forceinline__ void bin_add12_off(double* base, int off, const V& v)
{
#pragma unroll
    for(int c = 0; c < 12; ++c)
        binned_deposit(base + ((size_t)off + c) * BINNED_K, v(c));
}
// same, body-indexed (off = body*12).
template <class V>
__device__ __forceinline__ void bin_add12(double* base, int body, const V& v)
{
    bin_add12_off(base, body * 12, v);
}
// deterministic替代 muda atomic_add on a 12x12 matrix for `body` (144 scalars row-major).
template <class M>
__device__ __forceinline__ void bin_add144(double* base, int body, const M& H)
{
#pragma unroll
    for(int i = 0; i < 12; ++i)
#pragma unroll
        for(int j = 0; j < 12; ++j)
            binned_deposit(base + ((size_t)body * 144 + i * 12 + j) * BINNED_K, H(i, j));
}
