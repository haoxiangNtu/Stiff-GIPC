#include <linear_system/utils/pcg_capacity_mode.h>   // [C-3 prep]
#include <linear_system/utils/converter.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <linear_system/utils/binned_reduce.cuh>

namespace gipc
{

template <typename T>
__global__ inline void moveMemory_2(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

// [C-3 prep] capacity-mode helpers: the assembled length lives on device so
// the recorded convert graph never bakes a per-iteration count. Sentinel-key
// padding pushes dead slots to the sort tail; the partition-flag formula is
// uniform across the live/sentinel boundary, so only the writers need masks.
__global__ void _seed_convert_len(int* dst, int len)
{
    *dst = len;
}
__global__ void _finalize_unique_count_dev(int* dst, const uint32_t* partition, const int* d_len)
{
    *dst = (int)partition[*d_len - 1] + 1;
}

// [B2'-a] the device slot becomes the count truth (+1 applied on device);
// the host mirror is copied FROM it and in-PCG-loop consumers (spmv) read it
// live, so the captured graph no longer bakes the count into kernel params.
__global__ void _finalize_unique_count(int* dst, const uint32_t* last_partition)
{
    *dst = (int)(*last_partition) + 1;
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&                          start,
                        const int&                          length,
                        const int&                          out_start_id)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    global_triplets.h_unique_key_number.invalidate();  // [B1] device truth changes below
    // [C-3 prep] capacity mode: all grids are capacity-sized, kernels mask by
    // the device length. Transitionally the length is seeded here by value
    // (host truth at call time); the Newton graph replaces this seed with the
    // in-graph device accumulation.
    if(m_capacity_mode)
        _seed_convert_len<<<1, 1>>>(global_triplets.d_convert_len, length);
    _radix_sort_indices_and_blocks(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());



    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    _make_unique_block_warp_reduction(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



void Converter::_radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                               const int& start,
                                               const int& length,
                                               const int& out_start_id)
{
    using namespace muda;

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input   = global_triplets.block_index();
    auto ij_hash_input = global_triplets.block_hash_value();

    // [C-3 prep] capacity mode: fixed grid + sentinel padding (all-ones keys
    // sort to the tail and never produce writes downstream).
    const int  N  = m_capacity_mode ? (int)global_triplets.triplet_capacity() : length;
    const int* dl = m_capacity_mode ? global_triplets.d_convert_len : nullptr;

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(N,
               [row_indices = src_row_indices,
                col_indices = src_col_indices,
                ij_hash_input,
                index_input,
                dl] __device__(int i) mutable
               {
                   if(dl && i >= *dl)
                   {
                       ij_hash_input[i] = ~uint64_t{0};   // sentinel: sorts last
                       index_input[i]   = i;
                       return;
                   }
                   ij_hash_input[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   index_input[i] = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                N);

    auto dst_val = global_triplets.block_values() + out_start_id;
    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(N,
               [sort_index = global_triplets.block_sort_index(),
                src_blocks,
                dst_val,
                dl] __device__(int i) mutable
               {
                   if(dl && i >= *dl)
                       return;
                   dst_val[i] = src_blocks[sort_index[i]];

               });
}







void Converter::_make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                                  const int& start, const int& length, const int& out_start_id)
{
    using namespace muda;

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    // [C-3 prep] capacity mode geometry (see _radix_sort_indices_and_blocks).
    const int  N  = m_capacity_mode ? (int)global_triplets.triplet_capacity() : length;
    const int* dl = m_capacity_mode ? global_triplets.d_convert_len : nullptr;
    // [audit v0.8.5.1] cover element [length-1] too: it was never written but
    // the ExclusiveSum below reads all `length` inputs. The last input feeds no
    // output (exclusive scan), so results were always correct — this only
    // silences the uninitialized-read (compute-sanitizer initcheck noise).
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(N,
               [sorted_partition_input,
                n       = N,
                ij_hash = global_triplets.block_sort_hash_value()] __device__(int i) mutable
               {
                   // Uniform across the live/sentinel boundary: the last live key
                   // differs from the first sentinel (flag 1 = closes the last
                   // real unique); sentinel-interior pairs are equal (flag 0).
                   sorted_partition_input[i] =
                       (i + 1 < n) ? (ij_hash[i] != ij_hash[i + 1] ? 1 : 0) : 0;
               });
    auto sorted_partition_output = global_triplets.block_index();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input, sorted_partition_output, N);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);


    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(N,
               [row_indices,
                col_indices,
                ij_hash = global_triplets.block_sort_hash_value(),
                sorted_partition_output,
                dl] __device__(int i) mutable
               {
                   if(dl && i >= *dl)   // sentinel region writes nothing
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


    if(m_capacity_mode)
    {
        _finalize_unique_count_dev<<<1, 1>>>(global_triplets.d_unique_key_number,
                                             sorted_partition_output,
                                             global_triplets.d_convert_len);
        // Mirror refresh + zeroing become graph-friendly: the refresh is the
        // caller's ONE post-graph read; the zeroing covers capacity (combine
        // fully rewrites the live prefix; dead slots masked by d_unique).
        CUDA_SAFE_CALL(cudaMemcpy(global_triplets.h_unique_key_number.refresh_dst(),
                                  global_triplets.d_unique_key_number,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemsetAsync(global_triplets.block_values(start),
                                       0,
                                       global_triplets.triplet_capacity()
                                           * sizeof(Eigen::Matrix3d)));
    }
    else
    {
    _finalize_unique_count<<<1, 1>>>(global_triplets.d_unique_key_number,
                                     sorted_partition_output + length - 1);
    CUDA_SAFE_CALL(cudaMemcpy(global_triplets.h_unique_key_number.refresh_dst(),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemset(global_triplets.block_values(start),
                              0,
                              global_triplets.h_unique_key_number * sizeof(Eigen::Matrix3d)));
    }

    // [multi-env determinism 4.3 #4] DETERMINISTIC merge of duplicate (i,j) blocks. The old
    // FastSegmentalReduce summed each segment in sorted-array (= stable-sort = emission) order,
    // which is non-deterministic for contact triplets ⇒ non-det matrix values. Instead scatter
    // each sorted block's 9 doubles into a binned accumulator keyed by the unique output index
    // (order-independent ⇒ bit-identical), then combine.
    {
        int    nuniq = global_triplets.h_unique_key_number;
        size_t need  = (size_t)nuniq * 9 * BINNED_K;
        if(need > m_mergebin_cap)
        {   // [audit lens-A fix] CUDA_SAFE_CALL: a swallowed cudaMalloc failure
            // left m_mergebin dangling while m_mergebin_cap claimed the new
            // size — every later call would then memset/scatter through it.
            ++pcg_buffer_generation();   // [C-3 prep] mergebin ptr baked in graphs
            if(m_mergebin)
                CUDA_SAFE_CALL(cudaFree(m_mergebin));
            CUDA_SAFE_CALL(cudaMalloc((void**)&m_mergebin, need * sizeof(double)));
            m_mergebin_cap = need;
        }
        // [C-3 prep] capacity mode zeroes the full record-time capacity (byte
        // count stable across iterations); grows bump the generation above.
        if(m_capacity_mode)
            CUDA_SAFE_CALL(cudaMemsetAsync(m_mergebin, 0, m_mergebin_cap * sizeof(double)));
        else
            CUDA_SAFE_CALL(cudaMemset(m_mergebin, 0, need * sizeof(double)));

        auto* src_blocks = global_triplets.block_values(out_start_id);  // sorted src (length)
        auto* dst_blocks = global_triplets.block_values(start);         // unique out (nuniq)
        double* mbin     = m_mergebin;

        ParallelFor(256)
            .kernel_name("binned_block_merge_scatter")
            .apply(N,
                   [src_blocks, mbin, sorted_partition_output, dl] __device__(int i) mutable
                   {
                       if(dl && i >= *dl)
                           return;
                       int           out = sorted_partition_output[i];
                       const double* sd  = reinterpret_cast<const double*>(src_blocks + i);
#pragma unroll
                       for(int c = 0; c < 9; ++c)
                           binned_deposit(mbin + ((size_t)out * 9 + c) * BINNED_K, sd[c]);
                   });

        ParallelFor(256)
            .kernel_name("binned_block_merge_combine")
            .apply(m_capacity_mode ? (int)(m_mergebin_cap / (9 * BINNED_K)) : nuniq,
                   [dst_blocks, mbin,
                    du = m_capacity_mode ? (const int*)global_triplets.d_unique_key_number
                                         : (const int*)nullptr] __device__(int u) mutable
                   {
                       if(du && u >= *du)
                           return;
                       double* dd = reinterpret_cast<double*>(dst_blocks + u);
#pragma unroll
                       for(int c = 0; c < 9; ++c)
                           dd[c] = binned_combine(mbin + ((size_t)u * 9 + c) * BINNED_K);
                   });
    }
}

Converter::~Converter()
{
    if(m_mergebin)
        cudaFree(m_mergebin);
}


}  // namespace gipc