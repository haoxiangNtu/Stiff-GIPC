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
// [TU split] mechanics bodies live in contact/pair_buffers.cu since the
// v0.8.6 physical-separation pass; this header carries the contract + decls.

struct PairBuffers
{
    int4*& dcd;       // GIPC::_collisonPairs      (DCD encoded pairs)
    int*&  mat;       // GIPC::_MatIndex           (per-pair assembly rank)
    int4*& ccd;       // GIPC::_ccd_collisonPairs  (CCD mirror / swept pairs)
    int&   dcd_cap;   // GIPC::MAX_COLLITION_PAIRS_NUM
    int&   ccd_cap;   // GIPC::MAX_CCD_COLLITION_PAIRS_NUM
};


// Initial allocation from pre-set caps (caps must already hold the sizing).
void pair_buffers_alloc(PairBuffers b);
// Grow the DCD pair set to new_dcd; CCD mirror grows in lockstep, never shrinks.
void pair_buffers_grow_dcd(PairBuffers b, int new_dcd);
// Grow the CCD/swept pair set to new_ccd (monotone; below-cap = republish).
void pair_buffers_grow_ccd(PairBuffers b, int new_ccd);
void pair_buffers_free(PairBuffers b);
