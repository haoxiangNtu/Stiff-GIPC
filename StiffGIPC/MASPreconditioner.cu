// MASPreconditioner.cu — composite TU (v0.8.6 Phase 1b). Modules under mas_modules/
// are included IN ORDER; ordered concatenation is byte-identical to the
// pre-split file (asserted; sha256 in mas_modules/ORIGINAL_SHA256.txt).
// Include order is load-bearing — never reorder. See docs/V086_REFACTOR_PLAN.md.
#include "linear_system/utils/graph_node_resize.h"
#include "mas_modules/00_binned_accum.inl"
#include "mas_modules/01_aggregation_kernels.inl"
#include "mas_modules/02_inverse_restrict_collect.inl"
#include "mas_modules/03_schwarz_apply.inl"
#include "mas_modules/04_collision_connection.inl"
#include "mas_modules/05_envseg_host_pipeline.inl"
