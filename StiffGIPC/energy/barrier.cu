// ============================================================================
// energy/barrier.cu — IPC barrier term (E reduction + _calBarrierGradient +
// registry members) as its OWN TU (E3 rung 6, the ladder's top). The FUSED
// barrier G/H assembly (energy/03) deliberately STAYS in the composite — its
// encode/decode dependency tree is the documented exception; moving it is a
// separate future campaign with its own perf/anchor adjudication.
// Frozen smooth branches inside the moved bodies travel VERBATIM, still dead.
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_common/reductions.cuh"
#include "device_common/triplet_write.cuh"
#include "device_common/matrix_pd.cuh"
#include "energy/term_common.cuh"
#include "energy/binned_grad_common.cuh"
#include "GIPC_PDerivative.cuh"
#include "femEnergy.cuh"  // __computePFDsPX3D_* decls (defs femEnergy.cu, RDC)
#include "energy/02_contact_energy_device.inl"
using namespace Eigen;  // matches the composite prelude context

#include "15_barrier.inl"
