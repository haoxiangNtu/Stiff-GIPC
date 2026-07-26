// ============================================================================
// energy/term_common.cuh — shared device helpers for energy-term TUs (E3).
// inline __device__: each including TU gets its own copy (ODR-safe). Hoisted
// from gipc_modules/06 so physically-separated term TUs can use them.
// ============================================================================
#pragma once

__device__ inline void _penv_energy_accum(double* penv, const int* p2g, int vid, int ng, double e)
{
    if(!penv || !p2g) return;
    int g = p2g[vid];
    if(g >= 0 && g < ng) atomicAdd(&penv[g], e);
}
