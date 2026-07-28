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
#include <linear_system/utils/capacity_tier.h>

#include <muda/cub/device/device_radix_sort.h>
#include <cub/device/device_radix_sort.cuh>
using namespace Eigen;

// Contact assembly shares the final-converter count-residency switch.  The
// release path remains byte-for-byte exact-count; frame-graph preparation
// opts into capacity-shaped launches and device-side count guards.
static bool contact_tier_layout_mode()
{
    return GIPCTripletMatrix::device_count_mode();
}

struct ContactTripletTierLayout
{
    int n4 = 0;
    int n3 = 0;
    int n2 = 0;
    int c4 = 0;
    int c3 = 0;
    int c2 = 0;
    int exact_triplets = 0;
    int tier_triplets  = 0;
};

static ContactTripletTierLayout make_contact_triplet_tier(
    const HostMirrorArray<uint32_t, 5>& counts)
{
    ContactTripletTierLayout layout;
    layout.n4 = static_cast<int>(counts[4]);
    layout.n3 = static_cast<int>(counts[3]);
    layout.n2 = static_cast<int>(counts[2]);
    layout.c4 =
        layout.n4 ? gipc::assembly_capacity_tier(layout.n4) : 0;
    layout.c3 =
        layout.n3 ? gipc::assembly_capacity_tier(layout.n3) : 0;
    layout.c2 =
        layout.n2 ? gipc::assembly_capacity_tier(layout.n2) : 0;

    const long long exact =
        static_cast<long long>(layout.n4) * M12_Off
        + static_cast<long long>(layout.n3) * M9_Off
        + static_cast<long long>(layout.n2) * M6_Off;
    const long long tier =
        static_cast<long long>(layout.c4) * M12_Off
        + static_cast<long long>(layout.c3) * M9_Off
        + static_cast<long long>(layout.c2) * M6_Off;
    if(exact > std::numeric_limits<int>::max()
       || tier > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "contact triplet capacity tier exceeds 32-bit offsets");

    layout.exact_triplets = static_cast<int>(exact);
    layout.tier_triplets  = static_cast<int>(tier);
    return layout;
}

static int contact_triplet_extent(
    const ContactTripletTierLayout& layout)
{
    return contact_tier_layout_mode() ? layout.tier_triplets
                                      : layout.exact_triplets;
}

static int contact_scalar_extent(int count)
{
    return contact_tier_layout_mode() && count > 0
               ? gipc::assembly_capacity_tier(count)
               : count;
}

static int contact_pair_launch_extent(int count)
{
    return contact_tier_layout_mode() && count > 0
               ? gipc::assembly_capacity_tier(count)
               : count;
}

static void clear_contact_triplet_span(GIPCTripletMatrix& triplets,
                                       int                start,
                                       int                count)
{
    if(count <= 0)
        return;
    if(start < 0
       || static_cast<size_t>(start)
                  > std::numeric_limits<size_t>::max()
                        - static_cast<size_t>(count))
        throw std::overflow_error("contact triplet span overflow");

    const size_t need =
        static_cast<size_t>(start) + static_cast<size_t>(count);
    if(triplets.triplet_capacity() < need)
    {
        // The prefix [0,start) is already assembled. Preserve it while the
        // legacy path remains a correctness backstop outside graph capture.
        // Phase-C frame preparation makes this branch a no-op before capture.
        triplets.ensure_capacity_preserve(static_cast<size_t>(start), need);
    }
    CUDA_SAFE_CALL(cudaMemsetAsync(triplets.block_row_indices(start),
                                   0,
                                   static_cast<size_t>(count) * sizeof(int),
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(triplets.block_col_indices(start),
                                   0,
                                   static_cast<size_t>(count) * sizeof(int),
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(
        triplets.block_values(start),
        0,
        static_cast<size_t>(count) * sizeof(Eigen::Matrix3d),
        cudaStreamPerThread));
}

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
