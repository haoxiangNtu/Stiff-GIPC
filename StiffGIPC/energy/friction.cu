// ============================================================================
// energy/friction.cu — friction term (self + ground: E reductions, gradients,
// Hessians, registry members) as its OWN TU (E3 rung 5).
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "device_common/triplet_write.cuh"
#include "device_common/matrix_pd.cuh"
#include "energy/term_common.cuh"
#include "energy/binned_grad_common.cuh"
#include "energy/02_contact_energy_device.inl"
using namespace Eigen;  // matches the composite prelude context

#include "16_friction.inl"
