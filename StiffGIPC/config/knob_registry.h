#pragma once
// ============================================================================
// STIFF_* knob registry — the single source of truth for every environment
// variable the engine (or the stiff_physics python layer) reads.
//
// MAINTENANCE RULE: adding a getenv("STIFF_...")/env_on("STIFF_...") anywhere
// requires a row here — scripts/knob_gate.py (G14) fails the suite otherwise.
// The finalize-time tripwire warns on any STIFF_* variable present in the
// process environment that has no row (typo = silent no-op, historically);
// STIFF_KNOB_STRICT=1 upgrades the warning to ConfigurationError.
//
// Categories: mode_isolated / mode_strict (resolver bundles), perf, solver,
// audit, diag (probes+dumps), python (read by the python layer only).
// ============================================================================
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "errors.h"

extern "C" { extern char** environ; }  // POSIX, global scope

#define STIFF_KNOB_REGISTRY(X) \
    X(STIFF_A0_DUMP, "diag") \
    X(STIFF_ABD_DBG, "diag") \
    X(STIFF_ABD_DUMP, "diag") \
    X(STIFF_ABD_PRECOND_LEGACY, "solver") \
    X(STIFF_ALPHA_DBG, "diag") \
    X(STIFF_BAR_TGT0, "diag") \
    X(STIFF_BAR_TGT1, "diag") \
    X(STIFF_BAR_TRACE, "diag") \
    X(STIFF_BVH_ENVDET, "mode_isolated") \
    X(STIFF_BVH_ENVPART, "diag") \
    X(STIFF_CCD_CANON, "mode_strict") \
    X(STIFF_CCD_VALIDATE, "diag") \
    X(STIFF_CONTACT_DBG, "diag") \
    X(STIFF_DECOUPLE_THRESH, "mode_isolated") \
    X(STIFF_DEVICE_LINESEARCH, "perf") \
    X(STIFF_DEVICE_LINESEARCH_VALIDATE, "diag") \
    X(STIFF_DIAG_BINNED_GRAD, "diag") \
    X(STIFF_DIAG_KAPPA_MERGEDBB, "diag") \
    X(STIFF_DRIVE_SUBSTEP, "perf") \
    X(STIFF_DUMP_FRAME, "diag") \
    X(STIFF_EE_CANON, "mode_strict") \
    X(STIFF_EE_DETGATE, "mode_strict") \
    X(STIFF_EE_LB, "perf") \
    X(STIFF_EE_NODEDUP, "solver") \
    X(STIFF_EE_NOMOLLIFY, "solver") \
    X(STIFF_EE_TRACE, "diag") \
    X(STIFF_ENERGY_VALIDATE, "diag") \
    X(STIFF_GRAD_PRE, "diag") \
    X(STIFF_GRAD_PROBE, "diag") \
    X(STIFF_GROUND_HESS_LEGACY, "solver") \
    X(STIFF_HESS_ENV0, "diag") \
    X(STIFF_H_DUMP, "diag") \
    X(STIFF_ITER_LOG, "python") \
    X(STIFF_KSUM, "diag") \
    X(STIFF_MAS_DUMP, "diag") \
    X(STIFF_MAS_FUSE, "perf") \
    X(STIFF_MAS_FUSE_VALIDATE, "diag") \
    X(STIFF_MAS_SEG, "perf") \
    X(STIFF_MCDUMP, "diag") \
    X(STIFF_MERGED_DIAG_FRAME, "diag") \
    X(STIFF_MIRROR_AUDIT, "audit") \
    X(STIFF_NVTX, "diag") \
    X(STIFF_MULTIENV_MODE, "python") \
    X(STIFF_NO_REFINE, "perf") \
    X(STIFF_PCG_CHECK_K, "diag") \
    X(STIFF_PCG_DEVICE_LOOP, "perf") \
    X(STIFF_PCG_EW, "solver") \
    X(STIFF_PCG_EW_ETAMAX, "solver") \
    X(STIFF_PCG_EW_GAMMA, "solver") \
    X(STIFF_PCG_GRAPH, "perf") \
    X(STIFF_PCG_GRAPH_DIAG, "diag") \
    X(STIFF_PCG_TOL, "solver") \
    X(STIFF_PCG_WARM, "perf") \
    X(STIFF_PENV_STATS, "diag") \
    X(STIFF_PERENV_ALPHA, "mode_isolated") \
    X(STIFF_PERENV_BVH, "mode_isolated") \
    X(STIFF_PERENV_K, "diag") \
    X(STIFF_PERENV_MASK, "solver") \
    X(STIFF_PERENV_MASK_DEV, "solver") \
    X(STIFF_PERENV_PAR, "mode_isolated") \
    X(STIFF_PERENV_TELEM, "python") \
    X(STIFF_PERGROUP_KAPPA, "mode_isolated") \
    X(STIFF_PHASE_TIME, "diag") \
    X(STIFF_PROBE_K, "diag") \
    X(STIFF_S1_DEBUG, "diag") \
    X(STIFF_S3_VALIDATE, "diag") \
    X(STIFF_SEED_DIAG, "diag") \
    X(STIFF_SEGMENTED_PCG, "mode_isolated") \
    X(STIFF_SEG_BINNED, "perf") \
    X(STIFF_SEG_DIAG, "diag") \
    X(STIFF_SEG_WARP, "perf") \
    X(STIFF_SHAPE_STAGE, "diag") \
    X(STIFF_SKIP_E, "solver") \
    X(STIFF_SKIP_F, "solver") \
    X(STIFF_SKIP_FRIC, "solver") \
    X(STIFF_SKIP_GRND, "solver") \
    X(STIFF_SLOT_AUDIT, "audit") \
    X(STIFF_SPLIT_GH, "solver") \
    X(STIFF_SPMV_DET, "mode_strict") \
    X(STIFF_STACK_DIAG, "diag") \
    X(STIFF_XENV, "diag") \
    X(STIFF_XENV_DUMP, "diag") \
    X(STIFF_XENV_ID, "diag") \
    X(STIFF_XSKIP_DBG, "diag") \
    X(STIFF_Z_DUMP, "diag") \
    /* end */

namespace gipc {
inline bool stiff_knob_known(const char* name)
{
#define KNOB_MATCH(sym, cat) if(std::strcmp(name, #sym) == 0) return true;
    STIFF_KNOB_REGISTRY(KNOB_MATCH)
#undef KNOB_MATCH
    // registry meta-knob itself
    return std::strcmp(name, "STIFF_KNOB_STRICT") == 0;
}

// Scan the environment for STIFF_* variables missing from the registry.
// Called once at finalize. Warn by default; STIFF_KNOB_STRICT=1 throws.
inline void stiff_check_unknown_knobs()
{
    const char* strict_env = std::getenv("STIFF_KNOB_STRICT");
    const bool  strict = strict_env && strict_env[0] == '1';
    for(char** e = ::environ; *e; ++e)
    {
        if(std::strncmp(*e, "STIFF_", 6) != 0)
            continue;
        const char* eq = std::strchr(*e, '=');
        if(!eq)
            continue;
        char name[128];
        const size_t len = (size_t)(eq - *e);
        if(len >= sizeof(name))
            continue;
        std::memcpy(name, *e, len);
        name[len] = '\0';
        if(stiff_knob_known(name))
            continue;
        if(strict)
            throw ConfigurationError(
                std::string("unknown STIFF_* knob in environment: ") + name
                + " (typo? see StiffGIPC/config/knob_registry.h)");
        std::fprintf(stderr,
                     "[knob-registry][WARN] unknown STIFF_* knob '%s' — the "
                     "engine reads no such variable (typo = silent no-op). "
                     "Registry: StiffGIPC/config/knob_registry.h\n",
                     name);
    }
}
}  // namespace gipc
