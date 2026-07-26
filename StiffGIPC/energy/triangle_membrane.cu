// ============================================================================
// energy/triangle_membrane.cu — triangle_membrane term as its OWN TU (E3 rung 3).
// Element G/H kernels stay in femEnergy.cuh (upstream constitutive library),
// launched by the composite's module-12 wrappers — only the reduction and the
// registry members live here.
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "energy/term_common.cuh"
#include "femEnergy.cuh"  // __cal_*_energy device decls (defs in femEnergy.cu, RDC)

#include "12_triangle_membrane.inl"
