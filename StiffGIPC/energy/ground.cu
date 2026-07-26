// ============================================================================
// energy/ground.cu — ground term as its OWN TU (E3 rung 2).
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "energy/term_common.cuh"
#include "device_common/triplet_write.cuh"
#include "energy/binned_grad_common.cuh"

#include "17_ground.inl"
