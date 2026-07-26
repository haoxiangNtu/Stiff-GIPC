// ============================================================================
// energy/kinetic.cu — kinetic term as its OWN TU (E3 rung 1; the ladder's
// lowest-FP-density term goes first per docs/ENERGY_SEPARATION_PLAN.md).
// Kernels + the two registry members compile here; the composite reaches the
// launcher via class linkage and _calKineticGradient via composite_externs.
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "energy/term_common.cuh"

#include "10_kinetic.inl"
