// ============================================================================
// GIPC.cu — composite translation unit (v0.8.6 Phase-1 modularization).
//
// The former 16.8k-line monolith is decomposed into semantic modules under
// gipc_modules/, INCLUDED IN ORDER into this single TU. Preprocessed output
// is byte-identical to the pre-split file (asserted by the split script), so
// compiled code — and the strict bitwise anchor — cannot change.
//
// RULES until Phase 2 (physical TU separation, see docs/V086_REFACTOR_PLAN.md):
//  - The include ORDER below is load-bearing. Never reorder.
//  - New code goes into the semantically-owning module file, not here.
//  - Cross-module file-scope dependencies (device globals g_*, statics,
//    templates) are ALLOWED for now — they are what Phase 2 will untangle.
// ============================================================================
#include "gipc_modules/00_prelude_common.inl"
#include "gipc_modules/01_contact_energy_device.inl"
#include "gipc_modules/02_friction_assembly.inl"
#include "gipc_modules/03_barrier_assembly_split.inl"
#include "gipc_modules/04_binned_grad_fused_assembly.inl"
#include "gipc_modules/05_close_gradients.inl"
#include "gipc_modules/06_kinetic_soft_ground.inl"
#include "gipc_modules/07_energy_alpha_reductions.inl"
#include "gipc_modules/08_step_update_topology.inl"
#include "gipc_modules/09_friction_sets_host_mem.inl"
#include "gipc_modules/10_ccd_buildcp_quarantine.inl"
#include "gipc_modules/11_perenv_machinery.inl"
#include "gipc_modules/12_host_wrappers_fem.inl"
#include "gipc_modules/13_kappa_partition_gradhess.inl"
#include "gipc_modules/14_energy_linesearch_solver.inl"
// core/ipc_solver.cu is its OWN TU since Phase 2d step 2 (orchestration split out).
