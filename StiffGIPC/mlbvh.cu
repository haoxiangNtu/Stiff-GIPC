// mlbvh.cu — composite TU (v0.8.6 Phase 1b). Modules under mlbvh_modules/
// are included IN ORDER; ordered concatenation is byte-identical to the
// pre-split file (asserted; sha256 in mlbvh_modules/ORIGINAL_SHA256.txt).
// Include order is load-bearing — never reorder. See docs/V086_REFACTOR_PLAN.md.
#include "mlbvh_modules/00_gates_globals.inl"
#include "mlbvh_modules/01_caps_aabb_morton.inl"
#include "mlbvh_modules/02_distances_dtypes.inl"
#include "mlbvh_modules/03_pair_emission.inl"
#include "mlbvh_modules/04_lbvh_build.inl"
#include "mlbvh_modules/05_traversal_queries.inl"
#include "mlbvh_modules/06_host_wrappers.inl"
