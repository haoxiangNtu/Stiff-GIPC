// ============================================================================
// energy/fem_elastic.cu — FEM elastic (tet SNK/ARAP + rest-stable NHK) as its
// OWN TU (E3 rung 4). Constitutive device functions come from the upstream
// femEnergy library (decls femEnergy.cuh, defs femEnergy.cu, RDC cross-TU —
// the pattern proven bitwise-neutral in rung 3). Element G/H kernels stay
// upstream, launched by module-12 wrappers.
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "energy/term_common.cuh"
#include "femEnergy.cuh"

#include "11_fem_elastic.inl"
