#include <linear_system/utils/converter.h>
#include <linear_system/utils/capacity_tier.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <linear_system/utils/binned_reduce.cuh>
#include <frame_fsm/frame_status.cuh>

#include <algorithm>

namespace gipc
{

template <typename T>
__global__ inline void moveMemory_2(T* data,
                                    int output_start,
                                    int input_start,
                                    int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

// The device slot is the exact count truth. In frame-graph mode the host
// mirror is an upper-bound layout count and this value is consumed directly.
__global__ void _finalize_unique_count(int* dst,
                                       const uint32_t* last_partition)
{
    *dst = static_cast<int>(*last_partition) + 1;
}

__global__ void _publish_unique_frame_state(
    const int* d_unique,
    int armed_tier,
    frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x || !frame)
        return;

    const int exact = *d_unique;
    frame->unique_count = exact;
    atomicMax(&frame->hw_unique_blocks, exact);
    if(armed_tier <= 0 || exact <= armed_tier)
        return;

    atomicMax(&frame->required_unique_blocks, exact);
    frame_fsm::fsm_record_error(frame,
                                frame_fsm::ERR_CAPACITY,
                                frame_fsm::OVF_UNIQUE_BLOCKS,
                                -1,
                                -1);
    atomicCAS(&frame->result,
              frame_fsm::FRAME_OK,
              frame_fsm::FRAME_RETRY_REQUIRED);
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&          start,
                        const int&          length,
                        const int&          out_start_id,
                        ConvertLayout       layout)
{
    const int capacity =
        GIPCTripletMatrix::device_count_mode()
                && layout == ConvertLayout::FinalGlobal
            ? assembly_capacity_tier(length)
            : length;
    convert(
        global_triplets, start, length, capacity, out_start_id, layout);
}

void Converter::ensure_capacity(int capacity)
{
    if(capacity <= 0)
        return;

    const size_t need = static_cast<size_t>(capacity) * 9 * BINNED_K;
    if(need <= m_mergebin_cap)
        return;

    if(m_mergebin)
        CUDA_SAFE_CALL(cudaFree(m_mergebin));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_mergebin, need * sizeof(double)));
    m_mergebin_cap = need;
}

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&          start,
                        const int&          length,
                        const int&          capacity,
                        const int&          out_start_id,
                        ConvertLayout       layout)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    MUDA_ASSERT(capacity >= length,
                "converter capacity must cover exact triplet count");

    // Capture-safe envelopes. These branches are no-ops after frame-boundary
    // preparation; they remain safety nets for the legacy path.
    if(global_triplets.m_block_index.capacity()
       < static_cast<size_t>(capacity))
    {
        global_triplets.resize_collision_hash_size(
            static_cast<size_t>(capacity));
        global_triplets.global_external_max_capcity =
            std::max(global_triplets.global_external_max_capcity, capacity);
    }
    const size_t payload_need = static_cast<size_t>(out_start_id)
                              + static_cast<size_t>(capacity);
    if(global_triplets.triplet_capacity() < payload_need)
        global_triplets.reserve_triplets(payload_need + payload_need / 16);

    global_triplets.h_unique_key_number.invalidate();
    _radix_sort_indices_and_blocks(
        global_triplets, start, length, capacity, out_start_id);
    _make_unique_block_warp_reduction(
        global_triplets, start, length, capacity, out_start_id, layout);
}

void Converter::_radix_sort_indices_and_blocks(
    GIPCTripletMatrix& global_triplets,
    const int&         start,
    const int&         length,
    const int&         capacity,
    const int&         out_start_id)
{
    using namespace muda;

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input     = global_triplets.block_index();
    auto ij_hash_input   = global_triplets.block_hash_value();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(capacity,
               [row_indices = src_row_indices,
                col_indices = src_col_indices,
                ij_hash_input,
                index_input,
                length] __device__(int i) mutable
               {
                   index_input[i] = i;
                   ij_hash_input[i] =
                       i < length
                           ? (static_cast<uint64_t>(
                                  static_cast<uint32_t>(row_indices[i]))
                              << 32)
                                 + static_cast<uint64_t>(
                                     static_cast<uint32_t>(
                                         col_indices[i]))
                           : ~uint64_t{0};
               });

    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                capacity);

    auto dst_val = global_triplets.block_values(out_start_id);
    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(capacity,
               [sort_index = global_triplets.block_sort_index(),
                src_blocks,
                dst_val,
                length] __device__(int i) mutable
               {
                   if(i < length)
                       dst_val[i] = src_blocks[sort_index[i]];
                   else
                       dst_val[i].setZero();
               });
}

void Converter::_make_unique_block_warp_reduction(
    GIPCTripletMatrix& global_triplets,
    const int&         start,
    const int&         length,
    const int&         capacity,
    const int&         out_start_id,
    ConvertLayout      layout)
{
    using namespace muda;

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(capacity,
               [sorted_partition_input,
                length,
                ij_hash = global_triplets.block_sort_hash_value()] __device__(
                   int i) mutable
               {
                   sorted_partition_input[i] =
                       i + 1 < length && ij_hash[i] != ij_hash[i + 1] ? 1
                                                                     : 0;
               });

    auto sorted_partition_output = global_triplets.block_index();
    DeviceScan().ExclusiveSum(
        sorted_partition_input, sorted_partition_output, capacity);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(capacity,
               [row_indices,
                col_indices,
                ij_hash = global_triplets.block_sort_hash_value(),
                sorted_partition_output,
                length] __device__(int i) mutable
               {
                   if(i >= length)
                       return;
                   const int index = sorted_partition_output[i];
                   if(i == 0 || index != sorted_partition_output[i - 1])
                   {
                       const auto key = ij_hash[i];
                       row_indices[index] = key >> 32;
                       col_indices[index] = key & 0xffffffff;
                   }
               });

    _finalize_unique_count<<<1, 1>>>(
        global_triplets.d_unique_key_number,
        sorted_partition_output + length - 1);

    // A raw upper-bound layout is valid at the final global convert. ABD
    // slice converts instead use the stable power-of-two tier armed during
    // warm-up. The exact device count still guards the merge and an
    // undershoot poisons the surrounding frame transaction.
    bool bound_layout =
        GIPCTripletMatrix::device_count_mode()
        && layout == ConvertLayout::FinalGlobal;
    const int tier_index =
        layout == ConvertLayout::AbdFinal ? 1 : 0;
    int armed_abd_tier  = 0;
    int host_merge_count = length;
    if(bound_layout)
    {
        global_triplets.h_unique_key_number = length;
    }
    else if(GIPCTripletMatrix::device_count_mode()
            && global_triplets.m_abd_tier_txn_ok
            && global_triplets.m_abd_unique_tier[tier_index] > 0
            && global_triplets.m_abd_unique_tier[tier_index] <= length)
    {
        armed_abd_tier =
            global_triplets.m_abd_unique_tier[tier_index];
        global_triplets.h_unique_key_number = armed_abd_tier;
        bound_layout = true;
    }
    else
    {
        CUDA_SAFE_CALL(
            cudaMemcpy(global_triplets.h_unique_key_number.refresh_dst(),
                       global_triplets.d_unique_key_number,
                       sizeof(int),
                       cudaMemcpyDeviceToHost));
        host_merge_count = global_triplets.h_unique_key_number;
        if(GIPCTripletMatrix::device_count_mode()
           && layout != ConvertLayout::FinalGlobal)
            global_triplets.m_abd_unique_tier[tier_index] =
                assembly_capacity_tier(host_merge_count);
    }

    _publish_unique_frame_state<<<1, 1, 0, cudaStreamPerThread>>>(
        global_triplets.d_unique_key_number,
        global_triplets.m_abd_unique_test_tier > 0
            ? global_triplets.m_abd_unique_test_tier
            : armed_abd_tier,
        global_triplets.m_frame_device_state);

    const int clear_count = bound_layout ? length : host_merge_count;
    CUDA_SAFE_CALL(cudaMemsetAsync(
        global_triplets.block_values(start),
        0,
        static_cast<size_t>(clear_count) * sizeof(Eigen::Matrix3d),
        cudaStreamPerThread));

    // Deterministic, order-independent duplicate-block merge.
    const int merge_capacity =
        bound_layout ? capacity : host_merge_count;
    ensure_capacity(merge_capacity);
    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_mergebin,
        0,
        static_cast<size_t>(merge_capacity) * 9 * BINNED_K
            * sizeof(double),
        cudaStreamPerThread));

    auto* src_blocks = global_triplets.block_values(out_start_id);
    auto* dst_blocks = global_triplets.block_values(start);
    double* mbin     = m_mergebin;

    ParallelFor(256)
        .kernel_name("binned_block_merge_scatter")
        .apply(bound_layout ? capacity : length,
               [src_blocks,
                mbin,
                sorted_partition_output,
                length] __device__(int i) mutable
               {
                   if(i >= length)
                       return;
                   const int out = sorted_partition_output[i];
                   const double* sd =
                       reinterpret_cast<const double*>(src_blocks + i);
#pragma unroll
                   for(int c = 0; c < 9; ++c)
                       binned_deposit(
                           mbin
                               + (static_cast<size_t>(out) * 9 + c)
                                     * BINNED_K,
                           sd[c]);
               });

    ParallelFor(256)
        .kernel_name("binned_block_merge_combine")
        .apply(merge_capacity,
               [dst_blocks,
                mbin,
                d_uniq = global_triplets.d_unique_key_number,
                bound_layout] __device__(int u) mutable
               {
                   if(bound_layout && u >= *d_uniq)
                       return;
                   double* dd =
                       reinterpret_cast<double*>(dst_blocks + u);
#pragma unroll
                   for(int c = 0; c < 9; ++c)
                       dd[c] = binned_combine(
                           mbin
                           + (static_cast<size_t>(u) * 9 + c)
                                 * BINNED_K);
               });

    if(bound_layout)
    {
        ParallelFor(256)
            .kernel_name("neutralize_pad_unique_ids")
            .apply(capacity,
                   [rows = global_triplets.block_row_indices(start),
                    cols = global_triplets.block_col_indices(start),
                    d_uniq = global_triplets.d_unique_key_number,
                    length] __device__(int u) mutable
                   {
                       if(u < *d_uniq || u >= length)
                           return;
                       rows[u] = 0;
                       cols[u] = 0;
                   });
    }
}

Converter::~Converter()
{
    if(m_mergebin)
        cudaFree(m_mergebin);
}

}  // namespace gipc
