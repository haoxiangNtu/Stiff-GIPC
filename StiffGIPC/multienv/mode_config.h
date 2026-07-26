// ============================================================================
// multienv/mode_config.h — ModeConfig: the finalize-time snapshot of the
// multi-env mode flags (v0.8.6 mode-design hardening, tier C1).
//
// The mode -> flag-bundle RESOLUTION lives in stiff_physics/engine.py; the
// engine process then carries a soup of STIFF_* env vars. This struct is the
// C++-side SINGLE RUNTIME TRUTH: captured once at finalize (after the python
// resolver has run), printed at log level >= 1, and coherence-checked so a
// half-configured bundle warns loudly instead of running silently. Scattered
// getenv() readers migrate to GIPC::mode_config() incrementally; new code
// must read the snapshot, not the environment.
//
// Behavior contract: capture reads the SAME env vars the legacy readers read,
// at a point strictly after the resolver set them — converting a reader to
// the snapshot is behavior-neutral unless someone mutates the environment
// mid-run (which nothing supports).
// ============================================================================
#pragma once
#include <cstdio>
#include <cstdlib>

struct ModeConfig
{
    enum Mode : int { Merged = 0, Isolated = 1, Strict = 2 };
    Mode mode = Merged;

    // isolated bundle
    bool bvh_envdet = false, perenv_bvh = false, decouple_thresh = false,
         pergroup_kappa = false, segmented_pcg = false, perenv_alpha = false,
         perenv_par = false;
    // strict extras
    bool ee_canon = false, ee_detgate = false, ccd_canon = false, spmv_det = false;
    // isolation availability inputs (see isolation.cuh contract)
    bool perenv_telem = false;

    static bool env_on(const char* k) { return std::getenv(k) != nullptr; }

    static ModeConfig capture_from_env()
    {
        ModeConfig c;
        c.bvh_envdet      = env_on("STIFF_BVH_ENVDET");
        c.perenv_bvh      = env_on("STIFF_PERENV_BVH");
        c.decouple_thresh = env_on("STIFF_DECOUPLE_THRESH");
        c.pergroup_kappa  = env_on("STIFF_PERGROUP_KAPPA");
        c.segmented_pcg   = env_on("STIFF_SEGMENTED_PCG");
        c.perenv_alpha    = env_on("STIFF_PERENV_ALPHA");
        c.perenv_par      = env_on("STIFF_PERENV_PAR");
        c.ee_canon        = env_on("STIFF_EE_CANON");
        c.ee_detgate      = env_on("STIFF_EE_DETGATE");
        c.ccd_canon       = env_on("STIFF_CCD_CANON");
        c.spmv_det        = env_on("STIFF_SPMV_DET");
        c.perenv_telem    = env_on("STIFF_PERENV_TELEM");
        const int strict_extras = c.ee_canon + c.ee_detgate + c.ccd_canon + c.spmv_det;
        c.mode = c.perenv_alpha ? (strict_extras == 4 ? Strict : Isolated) : Merged;
        return c;
    }

    const char* mode_name() const
    {
        return mode == Strict ? "strict" : mode == Isolated ? "isolated" : "merged";
    }

    // Half-config detection: combinations the resolver never produces.
    // Returns the number of warnings printed (0 = coherent).
    int warn_if_incoherent() const
    {
        int w = 0;
        const int strict_extras = ee_canon + ee_detgate + ccd_canon + spmv_det;
        if(strict_extras != 0 && strict_extras != 4)
        {
            std::fprintf(stderr,
                         "[mode-config] WARNING: partial strict bundle (%d/4 of "
                         "EE_CANON/EE_DETGATE/CCD_CANON/SPMV_DET) — determinism is "
                         "NOT promised in this state\n", strict_extras);
            ++w;
        }
        if(strict_extras > 0 && !perenv_alpha)
        {
            std::fprintf(stderr,
                         "[mode-config] WARNING: strict extras without "
                         "STIFF_PERENV_ALPHA — this is no known mode\n");
            ++w;
        }
        const int iso_bundle = bvh_envdet + perenv_bvh + decouple_thresh
                             + pergroup_kappa + segmented_pcg + perenv_alpha + perenv_par;
        if(iso_bundle != 0 && iso_bundle != 7)
        {
            std::fprintf(stderr,
                         "[mode-config] WARNING: partial isolated bundle (%d/7) — "
                         "per-env fairness/isolation promises may not hold "
                         "(dev overrides are fine; half-configured modes are not)\n",
                         iso_bundle);
            ++w;
        }
        return w;
    }

    void print(int log_level) const
    {
        if(log_level < 1)
            return;
        std::printf("[mode-config] resolved=%s  iso[envdet=%d perenv_bvh=%d "
                    "decouple=%d pergroup_kappa=%d seg_pcg=%d perenv_alpha=%d "
                    "par=%d]  strict[ee_canon=%d detgate=%d ccd_canon=%d "
                    "spmv_det=%d]  telem=%d\n",
                    mode_name(), (int)bvh_envdet, (int)perenv_bvh,
                    (int)decouple_thresh, (int)pergroup_kappa, (int)segmented_pcg,
                    (int)perenv_alpha, (int)perenv_par, (int)ee_canon,
                    (int)ee_detgate, (int)ccd_canon, (int)spmv_det, (int)perenv_telem);
    }
};
