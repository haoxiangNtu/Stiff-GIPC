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
    if(getenv("STIFF_TIER_DIAG"))
        printf("[tier-diag] convert start=%d len=%d cap=%d out=%d bv_cap=%zu idx_cap=%zu\n",
               start, length, capacity, out_start_id,
               global_triplets.m_block_values.capacity(),
               global_triplets.m_block_index.capacity());
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
    // Same self-guarantee for the triplet payload: stage 2 writes the FULL
    // [out_start_id, out_start_id + capacity) range. Under the bound layout
    // the ABD expansion multiplies a bound count by the per-pair block count,
    // which can outgrow every upstream reservation formula; grow here where
    // the true requirement is exact. muda reserve preserves contents.
    {
        const size_t need_bv =
            static_cast<size_t>(out_start_id) + static_cast<size_t>(capacity);
        if(global_triplets.m_block_values.capacity() < need_bv
           || global_triplets.m_block_row_indices.capacity() < need_bv
           || global_triplets.m_block_col_indices.capacity() < need_bv)
            global_triplets.reserve_triplets(need_bv + need_bv / 16);
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
    // [P3b-1/convert-count] The bound layout (host mirror = pre-merge LENGTH,
    // true count only in d_unique_key_number) is valid ONLY for the FINAL
    // global convert (start == 0): its consumers all guard by the device
    // count and nothing lays out storage from its result. The ABD chain's
    // slice/mid converts (start > 0) feed the ×16 hessian expansion — there
    // the unique merge is the CONTRACTION step of a per-frame expand/merge
    // cycle, and a bound would compound ×16 every frame (1.4M → 22M → 353M).
    // Those boundaries keep the exact readback until tier-dispatch lands.
    bool bound_layout =
        GIPCTripletMatrix::device_count_mode() && start == 0;
    if(bound_layout)
    {
        global_triplets.h_unique_key_number = length;
    }
    else if(GIPCTripletMatrix::device_count_mode()
            && global_triplets.m_abd_tier_txn_ok
            && global_triplets.m_abd_uniq_tier > 0
            && global_triplets.m_abd_uniq_tier <= length)
    {
        // [P3b-2/abd-tier] steady state: layout by the armed tier constant,
        // true count stays on device (consumers guard by *d_unique). The
        // expansion feedback becomes tier -> tier (fixed point), so the ×16
        // compounding of the naive bound layout cannot occur. A verify
        // kernel re-arms via exact readback when nuniq outgrows the tier
        // (checked below after the merge publishes d_unique_key_number).
        global_triplets.h_unique_key_number = global_triplets.m_abd_uniq_tier;
        bound_layout                        = true;  // pad-neutralize [nuniq, tier)
    }
    else
    {
        CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                                  global_triplets.d_unique_key_number,
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost));
        if(GIPCTripletMatrix::device_count_mode() && start > 0)
        {
            // Arm (or re-arm) the steady-state tier from the exact count.
            const int t = gipc::assembly_capacity_tier(
                global_triplets.h_unique_key_number);
            global_triplets.m_abd_uniq_tier = t;
        }
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

    if(start > 0 && GIPCTripletMatrix::device_count_mode()
       && global_triplets.m_abd_uniq_tier > 0)
    {
        // [P3b-2/abd-tier] OVF check: if the true nuniq outgrew the armed
        // tier, this frame's tier layout under-covers — latch the flag; the
        // frame boundary disarms the tier and retries transactionally.
        auto* d_uniq_chk = global_triplets.d_unique_key_number;
        auto* d_ovf      = global_triplets.d_abd_tier_ovf;
        const int tier_now = global_triplets.m_abd_uniq_tier;
        ParallelFor(1).kernel_name("abd_tier_ovf_check").apply(1,
            [d_uniq_chk, d_ovf, tier_now] __device__(int) mutable
            {
                if(*d_uniq_chk > tier_now)
                    *d_ovf = *d_uniq_chk;
            });
    }

    if(bound_layout)
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
