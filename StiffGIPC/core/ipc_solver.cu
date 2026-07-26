// ============================================================================
// core/ipc_solver.cu — frame orchestration as its OWN translation unit
// (v0.8.6 Phase 2d step 2: first physical TU separation of the refactor).
//
// Host-only orchestration: every __global__ kernel it launches is DEFINED in
// the GIPC.cu composite TU (or multienv/isolation.cuh compiled therein) and
// reached here through the extern declarations below under relocatable device
// code (CUDA_SEPARABLE_COMPILATION ON). Device codegen therefore stays in the
// composite TU — separating this file moves no device code at all.
//
// If you add a kernel launch to the orchestration, add its extern declaration
// here and KEEP the definition in the mechanism's owning module.
// ============================================================================
#include "GIPC.cuh"
#include "eigen_data.h"                 // Vector12 (checkpoint/ABD paths)
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <gipc/gipc.h>
#include "cuda_tools/cuda_tools.h"      // CUDA_SAFE_CALL
#include <gipc/statistics.h>            // stats / gipc::Json
#include <gipc_path.h>                  // gipc::output_dir
#include <gipc/utils/timer.h>           // gipc::GlobalTimer

#include "contact/ccd_invalid_bits.h"  // CCD invalid-mask contract
#include "solver_stats.h"  // [phase4] decl/def type-check for the cross-TU counters

// ---- file-scope globals owned by the composite TU ----
#include "device_common/debug_probes.h"  // [A4] typed decls for the debug globals

// ---- host mechanism wrappers owned by the composite TU ----
extern void calcMinMovement_DeviceOut(const double3* _moveDir, double* _queue, const int& number);
extern void stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir,
                        int* bType, double alpha, bool moveBoundary, int numbers);
extern void throwForInvalidCcdMask(int invalid, const char* context);
extern void validateFinalCcdStateOrThrow(const double* state, const char* context);

// ---- kernels owned by their mechanism modules (composite TU) ----
extern __global__ void _ccd_final_alpha_combine(double* slots, int have_ccd_pairs, double d_hat, double ccd_size, int* invalid, const int* refined_invalid);
extern __global__ void _ccd_initial_alpha_combine(double* slots, int have_ground, int have_self, int* invalid);
extern __global__ void _fill_double(double* values, double value, int count);
extern __global__ void _gather_abd_body_alpha(const int* body_to_group, const double* env_alpha, double* abd_body_alpha, int abd_body_num, int ng);
extern __global__ void _global_ls_decide(const double* energy0, const double* energy1, double c1m, double alpha, double energy_abs_tol, double energy_rel_tol, int* status);
extern __global__ void _mask_fill(int* m, int v, int n);
extern __global__ void _mask_from_env_alpha(int* env_active, const double* env_alpha, int ng);
extern __global__ void _newton_convergence_decide(const double* max_movement, double threshold, int* converged);
extern __global__ void _per_env_alpha_compute(double* env_alpha, const double* scratch, int ng, double sq, double ccd_size, int have_ccd, double temp_alpha, double alpha_CFL, int decouple, int no_refine, double thr_cv, const double* env_bbox2, double ntol_dt, double vtol_dt, const int* refined_invalid, const int* ccd_alpha_invalid, int* cnt);
extern __global__ void _per_env_groundAlpha_min(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, const double3* moveDir, const int* p2g, double* per_env_alpha, double slackness, int number, const int* _point_body_id, const int* _ground_skip_body, int _ground_body_count, int ng, int* ccd_alpha_invalid);
extern __global__ void _per_env_max_cfl(const int* p2g, const double3* moveDir, const uint32_t* mSVI, double* per_env_max, int n_surf, int ng);
extern __global__ void _per_env_max_move(const int* p2g, const double3* moveDir, double* per_env_max, int n, int ng);
extern __global__ void _per_env_selfAlpha_min(const double3* vertexes, const int4* pairs, const double3* moveDir, const int* p2g, double* per_env_alpha, double slackness, int number, int ng, const int* vloc, int* ccd_alpha_invalid, int invalid_bit, int* refined_invalid);
extern __global__ void _per_env_sqnorm_accum(const int* p2g, const double3* vec, double* per_env_sq, int* per_env_cnt, int n, int ng);
extern __global__ void _per_group_kappa_double(double* kappa_group, const int* close_grp, const double* env_alpha, int ng, double kappaMax, double* maxK_out);
extern __global__ void _promote_per_env_refined_invalid(const double* scratch, int ng, int have_ccd, double sq, double temp_alpha, double alpha_CFL, int decouple, int no_refine, const int* refined_invalid, int* ccd_alpha_invalid);
extern __global__ void _s3_decide(const double* Eg0, const double* Eg1, double* env_alpha, int* decision_counts, int ng, double energy_abs_tol, double energy_rel_tol);
extern __global__ void _s3_halve_all(double* env_alpha, int ng);
extern __global__ void _scan_dir_nonfinite(const double3* dir, const int* p2g, int* flags, int n);
extern __global__ void _zero_dir_quarantined(double3* dir, const int* p2g, const int* quar, int n);

#include "ipc_solver.inl"
