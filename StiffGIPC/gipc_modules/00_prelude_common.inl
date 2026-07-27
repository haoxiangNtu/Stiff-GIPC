// GIPC composite translation-unit prelude.
//
// Keep this file declarative: common headers, TU globals, and ordered module
// includes only. Device helpers and kernels belong to their owning modules so
// future physical TU extraction has explicit seams.

#include "GIPC.cuh"
#include "eigen_data.h"  // Vector12 for stitch local-frame fix
#include <stdexcept>     // [d-floor fail-fast] ground-distance collapse -> throw
#include <string>
#include <gipc/gipc.h>
#include "cuda_tools/cuda_tools.h"
#include "GIPC_PDerivative.cuh"
#include "fem_parameters.h"
#include "errors.h"  // typed taxonomy for composite-TU throws
#include "device_common/reductions.cuh"  // unified block reductions
#include "device_common/debug_probes.h"
#include "contact/pair_buffers.cuh"
#include "multienv/isolation.cuh"
#include "ACCD.cuh"
#include "femEnergy.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "FrictionUtils.cuh"
#include <cfloat>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <limits>
#include "Eigen/Eigen"
#include <gipc/statistics.h>
#include <gipc_path.h>
#include <gipc/utils/timer.h>

#include <muda/cub/device/device_radix_sort.h>
#include <cub/device/device_radix_sort.cuh>
using namespace Eigen;

// Global log verbosity for per-frame and one-time solver prints.
int g_gipc_log_level = 1;

#include "contact/barrier_rank.h"
#define NEWF

#include "contact/ccd_invalid_bits.h"
#include "contact/encoding.h"
#include "energy/energy_terms.h"
#include "device_common/matrix_pd.cuh"
#include "device_common/triplet_write.cuh"

// The include order below intentionally preserves the former monolithic
// prelude's declaration order. This is a source-ownership split only: all
// modules still compile into the same composite CUDA translation unit.
#include "device_common/spatial_hash.inl"
#include "linear_system/triplet_partition_kernels.inl"
#include "device_common/topology_sort_kernels.inl"
#include "device_common/reduction_kernels.inl"
