#include <linear_system/utils/converter.h>
#include <linear_system/utils/capacity_tier.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <linear_system/utils/binned_reduce.cuh>

namespace gipc
{

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&          start,
                        const int&          length,
                        const int&          out_start_id)
{
    convert(global_triplets,
            start,
            length,
            assembly_capacity_tier(length),
            out_start_id);
}

template <typename T>
__global__ inline void moveMemory_2(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&                          start,
                        const int&                          length,
                        const int&                          capacity,
                        const int&                          out_start_id)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    MUDA_ASSERT(capacity >= length, "converter tier must cover exact triplet count");
    // [P3b-2/contact-tier] own the hash/sort scratch envelope: the hash-build
    // writes index/hash over the FULL capacity. Callers historically sized
    // these buffers transitively (pre-assembly bound*1.1); the tier layout
    // widens slice capacities past that implicit contract, so grow here by
    // the ACTUAL buffer capacity, not the external bookkeeping mirror.
    if(global_triplets.m_block_index.capacity() < static_cast<size_t>(capacity))
    {
        global_triplets.resize_collision_hash_size(static_cast<size_t>(capacity));
        if(global_triplets.global_external_max_capcity < capacity)
            global_triplets.global_external_max_capcity = capacity;
    }
    _radix_sort_indices_and_blocks(global_triplets, start, length, capacity, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    //_make_unique_indices(global_triplets, start, length, out_start_id);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    _make_unique_block_warp_reduction(
        global_triplets, start, length, capacity, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



void Converter::_radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                               const int& start,
                                               const int& length,
                                               const int& capacity,
                                               const int& out_start_id)
{
    using namespace muda;

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input   = global_triplets.block_index();
    auto ij_hash_input = global_triplets.block_hash_value();

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
                   if(i < length)
                       ij_hash_input[i] =
                           (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   else
                       ij_hash_input[i] = ~uint64_t{0};
               });

    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                capacity);

    auto dst_val = global_triplets.block_values() + out_start_id;
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


void Converter::_make_unique_indices(GIPCTripletMatrix& global_triplets,
                                     const int&         start,
                                     const int&         length,
                                     const int&         out_start_id)
{
    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);

    auto unique_key = global_triplets.block_hash_value();
    auto sort_key   = global_triplets.block_sort_hash_value();

    muda::DeviceRunLengthEncode().Encode(sort_key,
                                         unique_key,
                                         global_triplets.block_temp_buffer(),
                                         global_triplets.d_unique_key_number,
                                         length);

    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(global_triplets.h_unique_key_number,

               [row_indices, col_indices, unique_key] __device__(int i) mutable
               {
                   row_indices[i] = unique_key[i] >> 32;
                   col_indices[i] = unique_key[i] & 0xffffffff;
               });
}





void Converter::_make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                                  const int& start, const int& length,
                                                  const int& capacity, const int& out_start_id)
{
    using namespace muda;

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(capacity,
               [sorted_partition_input,
                ij_hash = global_triplets.block_sort_hash_value(),
                length] __device__(int i) mutable
               {
                   sorted_partition_input[i] =
                       i + 1 < length && ij_hash[i] != ij_hash[i + 1] ? 1 : 0;
               });
    auto sorted_partition_output = global_triplets.block_index();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input, sorted_partition_output, capacity);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);


    muda::ParallelFor(256)
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
                   int index = sorted_partition_output[i];
                   if(i == 0)
                   {

                       auto key           = ij_hash[i];
                       row_indices[index] = key >> 32;
                       col_indices[index] = key & 0xffffffff;
                   }
                   else
                   {
                       if(index != sorted_partition_output[i - 1])
                       {
                           auto key           = ij_hash[i];
                           row_indices[index] = key >> 32;
                           col_indices[index] = key & 0xffffffff;
                       }
                   }
               });


    // [frame-fsm P3a-prep] publish the exact unique count on the DEVICE first
    // (scan tail + 1) so downstream launches can guard against it without a
    // round-trip; the host mirror read below now only feeds the SpMV shape
    // publication (removed entirely in full P3a).
    {
        auto* d_cnt = global_triplets.d_unique_key_number;
        auto* d_src = sorted_partition_output + length - 1;
        muda::ParallelFor(1)
            .kernel_name("publish_unique_count")
            .apply(1,
                   [d_cnt, d_src] __device__(int) mutable
                   { *d_cnt = *d_src + 1; });
    }
    if(GIPCTripletMatrix::device_count_mode())
    {
        // [P3b-1/convert-count] no assembly-boundary readback: the host
        // mirror keeps the pre-merge LENGTH as a layout upper bound and the
        // true unique count lives only in d_unique_key_number. Pad slots
        // [nuniq, length) keep zero VALUES (cleared below) and get their ids
        // neutralized to (0,0) after the merge, so downstream consumers
        // either guard by the device count (spmv/diag/MAS input) or deposit
        // bitwise-invisible +0.0 into the always-present (0,0) key during
        // the next binned merge.
        global_triplets.h_unique_key_number = length;
    }
    else
    {
        CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                                  global_triplets.d_unique_key_number,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost));
    }

    // Upper-bound (nuniq <= length) async clear: legal inside capture and
    // independent of the host mirror.  Keep this at the exact sub-range
    // length: ABD converts one contact-kind slice in place, so clearing the
    // whole tier here would erase the adjacent contact-kind slice.
    CUDA_SAFE_CALL(cudaMemsetAsync(global_triplets.block_values(start),
                                   0,
                                   (size_t)length * sizeof(Eigen::Matrix3d),
                                   cudaStreamPerThread));

    // [multi-env determinism 4.3 #4] DETERMINISTIC merge of duplicate (i,j) blocks. The old
    // FastSegmentalReduce summed each segment in sorted-array (= stable-sort = emission) order,
    // which is non-deterministic for contact triplets ⇒ non-det matrix values. Instead scatter
    // each sorted block's 9 doubles into a binned accumulator keyed by the unique output index
    // (order-independent ⇒ bit-identical), then combine.
    {
        // [frame-fsm P3a-prep] size the merge bins by the LENGTH upper bound
        // (nuniq <= length always) on power-of-two steps: growth happens on
        // capacity tiers only (O(log) mallocs over a run, none once the
        // high-water tier is reached), and the clear is stream-ordered.
        size_t need = (size_t)capacity * 9 * BINNED_K;
        if(need > m_mergebin_cap)
        {
            if(m_mergebin)
                cudaFree(m_mergebin);
            cudaMalloc((void**)&m_mergebin, need * sizeof(double));
            m_mergebin_cap = need;
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(m_mergebin, 0,
                                       (size_t)capacity * 9 * BINNED_K * sizeof(double),
                                       cudaStreamPerThread));

        auto* src_blocks = global_triplets.block_values(out_start_id);  // sorted src (length)
        auto* dst_blocks = global_triplets.block_values(start);         // unique out (nuniq)
        double* mbin     = m_mergebin;

        ParallelFor(256)
            .kernel_name("binned_block_merge_scatter")
            .apply(capacity,
                   [src_blocks, mbin, sorted_partition_output, length] __device__(int i) mutable
                   {
                       if(i >= length)
                           return;
                       int           out = sorted_partition_output[i];
                       const double* sd  = reinterpret_cast<const double*>(src_blocks + i);
#pragma unroll
                       for(int c = 0; c < 9; ++c)
                           binned_deposit(mbin + ((size_t)out * 9 + c) * BINNED_K, sd[c]);
                   });

        auto* d_uniq = global_triplets.d_unique_key_number;
        ParallelFor(256)
            .kernel_name("binned_block_merge_combine")
            .apply(capacity,
                   [dst_blocks, mbin, d_uniq] __device__(int u) mutable
                   {
                       if(u >= *d_uniq)
                           return;
                       double* dd = reinterpret_cast<double*>(dst_blocks + u);
#pragma unroll
                       for(int c = 0; c < 9; ++c)
                           dd[c] = binned_combine(mbin + ((size_t)u * 9 + c) * BINNED_K);
                   });
    }

    if(GIPCTripletMatrix::device_count_mode())
    {
        // [P3b-1/convert-count] neutralize pad ids in [nuniq, length): the
        // scatter above writes ids only at real segment starts, so these
        // slots would otherwise hold stale indices. Values are already zero
        // from the range clear above.
        auto* d_uniq_pad = global_triplets.d_unique_key_number;
        auto  pad_rows   = global_triplets.block_row_indices(start);
        auto  pad_cols   = global_triplets.block_col_indices(start);
        ParallelFor(256)
            .kernel_name("neutralize_pad_unique_ids")
            .apply(capacity,
                   [pad_rows, pad_cols, d_uniq_pad, length] __device__(int u) mutable
                   {
                       if(u < *d_uniq_pad || u >= length)
                           return;
                       pad_rows[u] = 0;
                       pad_cols[u] = 0;
                   });
    }
}

Converter::~Converter()
{
    if(m_mergebin)
        cudaFree(m_mergebin);
}

void Converter::ge2sym(GIPCTripletMatrix& global_triplets)
{
    using namespace muda;

    auto counts  = global_triplets.block_index();
    auto offsets = global_triplets.block_sort_index();
    auto block_temp = global_triplets.block_values(global_triplets.h_unique_key_number);
    auto blocks      = global_triplets.block_values();
    auto ij_hash     = global_triplets.block_hash_value();
    auto row_indices = global_triplets.block_row_indices();
    auto col_indices = global_triplets.block_col_indices();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [row_indices, col_indices, ij_hash, blocks, block_temp, counts] __device__(int i) mutable
               {
                   counts[i] = row_indices[i] <= col_indices[i] ? 1 : 0;
                   ij_hash[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   block_temp[i] = blocks[i];
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts, offsets, global_triplets.h_unique_key_number);

    // set the values
    auto dst_blocks = global_triplets.block_values();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [dst_blocks,
                block_temp,
                ij_hash,
                row_indices,
                col_indices,
                counts,
                offsets,
                total_count = global_triplets.d_unique_key_number,
                number = global_triplets.h_unique_key_number] __device__(int i) mutable
               {
                   auto count  = counts[i];
                   auto offset = offsets[i];

                   if(count != 0)
                   {
                       dst_blocks[offset]  = block_temp[i];
                       auto ij             = ij_hash[i];
                       row_indices[offset] = ij >> 32;
                       col_indices[offset] = ij & 0xffffffff;
                   }

                   if(i == number - 1)
                   {
                       *total_count = offsets[i] + counts[i];
                   }
               });


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
}

}  // namespace gipc
