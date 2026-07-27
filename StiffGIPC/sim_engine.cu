// sim_engine.cu — composite TU (v0.8.6 Phase 1b). Modules under engine_modules/
// are included IN ORDER; ordered concatenation is byte-identical to the
// pre-split file (asserted; sha256 in engine_modules/ORIGINAL_SHA256.txt).
// Include order is load-bearing — never reorder. See docs/V086_REFACTOR_PLAN.md.
#include "config/knob_registry.h"  // [knob-registry] STIFF_* single source + tripwire
#include "engine_modules/00_impl_api_surface.inl"
#include "engine_modules/01_config_upload.inl"
#include "engine_modules/02_finalize_nandiag.inl"
#include "engine_modules/03_step_getters_export.inl"
#include "engine_modules/04_teleport_checkpoint.inl"
