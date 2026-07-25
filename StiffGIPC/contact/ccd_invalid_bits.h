// ============================================================================
// contact/ccd_invalid_bits.h — CCD alpha invalid-mask bit contract.
// Single source of truth, shared by the GIPC.cu composite TU (fail-fast
// throw sites, gipc_modules/10) and the core/ipc_solver.cu orchestration TU
// (mask assembly around the CCD alpha chain). Hoisted verbatim from
// gipc_modules/00 in v0.8.6 Phase 2d step 2.
// ============================================================================
#pragma once

enum CcdAlphaInvalidBits : int
{
    kCcdInvalidGlobalGround  = 1 << 0,
    kCcdInvalidGlobalNarrow  = 1 << 1,
    kCcdInvalidGlobalRefined = 1 << 2,
    kCcdInvalidPerEnvGround  = 1 << 3,
    kCcdInvalidPerEnvNarrow  = 1 << 4,
    kCcdInvalidPerEnvRefined = 1 << 5,
};
static constexpr int kCcdInvalidEffectiveMask = (1 << 6) - 1;
static constexpr int kCcdRawInvalid           = 1;
