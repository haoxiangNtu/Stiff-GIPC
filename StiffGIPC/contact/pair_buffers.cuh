#pragma once
// ============================================================================
// contact/pair_buffers.cuh — the single owner of contact pair-buffer GROWTH
// MECHANICS (v0.8.6 Phase 2b; docs/V086_REFACTOR_PLAN.md).
//
// Before this file, the alloc/grow/cap-publication mechanics were hand-copied
// across FIVE sites (init + four grow-redo loops), each responsible for the
// full checklist: +1 trash slot, DCD<=CCD mirror lockstep, bvh pointer rebind,
// set_emit_caps publication. The audit predicted a missed-checklist accident;
// converting the sites surfaced one already latent: the per-env DCD grow
// reallocated the CCD mirror at new_dcd+1 UNCONDITIONALLY while updating
// ccd_cap only when new_dcd exceeded it — when new_dcd < ccd_cap (guaranteed
// for the FIRST per-env overflow, since initial sizing keeps CCD >> DCD) the
// allocation SHRANK BELOW the published cap, and the next swept-CCD detection
// (clamped by the larger stale cap) could write past the allocation.
//
// Division of responsibility:
//   - POLICY (when to grow, the new capacity, the redo choreography, bvh
//     pointer rebinding) stays at the call sites;
//   - MECHANICS (free+malloc with the +1 trash slot, never shrinking the CCD
//     mirror below its cap, keeping caps == allocations, publishing through
//     set_emit_caps) live HERE and nowhere else.
//
// All buffers hold cap+1 entries: slot [cap] is the overflow trash slot that
// _emit_slot clamps to (see mlbvh_modules/01); a write there always coincides
// with a host-side grow+redo, so it is never consumed.
// ============================================================================

// published to the device-side emit clamps (defined in mlbvh_modules/00)
void set_emit_caps(int dcd_cap, int ccd_cap);

struct PairBuffers
{
    int4*& dcd;       // GIPC::_collisonPairs      (DCD encoded pairs)
    int*&  mat;       // GIPC::_MatIndex           (per-pair assembly rank)
    int4*& ccd;       // GIPC::_ccd_collisonPairs  (CCD mirror / swept pairs)
    int&   dcd_cap;   // GIPC::MAX_COLLITION_PAIRS_NUM
    int&   ccd_cap;   // GIPC::MAX_CCD_COLLITION_PAIRS_NUM
};

// Initial allocation from pre-set caps (caps must already hold the sizing).
inline void pair_buffers_alloc(PairBuffers b)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.mat, ((size_t)b.dcd_cap + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.dcd, ((size_t)b.dcd_cap + 1) * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&b.ccd, ((size_t)b.ccd_cap + 1) * sizeof(int4)));
    set_emit_caps(b.dcd_cap, b.ccd_cap);
}

// Grow the DCD pair set (pairs + assembly rank) to new_dcd. The CCD mirror is
// grown IN LOCKSTEP only when new_dcd exceeds its cap — and NEVER shrunk:
// allocation size and published cap stay equal by construction.
inline void pair_buffers_grow_dcd(PairBuffers b, int new_dcd)
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
inline void pair_buffers_grow_ccd(PairBuffers b, int new_ccd)
{
    if(new_ccd > b.ccd_cap)
    {
        CUDA_SAFE_CALL(cudaFree(b.ccd));
        CUDA_SAFE_CALL(cudaMalloc((void**)&b.ccd, ((size_t)new_ccd + 1) * sizeof(int4)));
        b.ccd_cap = new_ccd;
    }
    set_emit_caps(b.dcd_cap, b.ccd_cap);
}

inline void pair_buffers_free(PairBuffers b)
{
    if(b.mat) { CUDA_SAFE_CALL(cudaFree(b.mat)); b.mat = nullptr; }
    if(b.dcd) { CUDA_SAFE_CALL(cudaFree(b.dcd)); b.dcd = nullptr; }
    if(b.ccd) { CUDA_SAFE_CALL(cudaFree(b.ccd)); b.ccd = nullptr; }
}
