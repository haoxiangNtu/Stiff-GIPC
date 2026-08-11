// ============================================================================
// energy/term_common.cuh — shared device helpers for energy-term TUs (E3).
// inline __device__: each including TU gets its own copy (ODR-safe). Hoisted
// from gipc_modules/06 so physically-separated term TUs can use them.
// ============================================================================
#pragma once

#include "../linear_system/utils/binned_reduce.cuh"

// [strict-LS N-invariance] per-env energy deposit. Plain atomicAdd(double) sums in
// thread-scheduling order, which depends on the TOTAL batch shape → env_0's per-env
// energy carries batch-dependent last-ulp jitter → near the S3 descent threshold this
// FLIPS a backtrack-halve decision → alpha diverges across N. Same root as the ABD fix
// in cal_abd_energy_perenv — this closes the FEM/contact side. binned_deposit is
// exponent-binned (exact, order-independent) under the central g_det_reduce gate and a
// single plain atomic into bin 0 otherwise. Callers pass a BINNED_K-wide bins block
// (ng*BINNED_K), combined into the plain per-env slice by _penv_bins_combine at the end
// of Energy_Add_Reduction_Algorithm_DeviceOut.
__device__ inline void _penv_energy_accum(double* penv, const int* p2g, int vid, int ng, double e)
{
    if(!penv || !p2g) return;
    int g = p2g[vid];
    if(g >= 0 && g < ng) binned_deposit(penv + (size_t)g * BINNED_K, e);
}

// ── _ec_emit contact-force export helper (E3.6 hoist from gipc_modules/05) ──
__device__ inline void _ec_emit(int idx, int2* op, double3* of, const int* pbid,
                                double inv_dt2, int va, int vb, int vc, int vd,
                                int nv, const double* g)
{
    if(!op)
        return;
    int    vs[4] = {va, vb, vc, vd};
    int    bA = pbid[va], bB = pbid[va];
    double fx = 0, fy = 0, fz = 0;
    for(int k = 0; k < nv; k++)
    {
        int b = pbid[vs[k]];
        if(b == bA)
        {
            fx += g[3 * k]; fy += g[3 * k + 1]; fz += g[3 * k + 2];
        }
        else
            bB = b;
    }
    op[idx] = make_int2(bA, bB);
    of[idx] = make_double3(-fx * inv_dt2, -fy * inv_dt2, -fz * inv_dt2);
}
