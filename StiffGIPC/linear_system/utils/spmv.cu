#include <linear_system/utils/spmv.h>
#include <linear_system/utils/binned_reduce.cuh>
#include <muda/launch/launch.h>
#include <cub/warp/warp_reduce.cuh>

namespace gipc
{

Spmv::~Spmv()
{
    if(m_ybin)
        cudaFree(m_ybin);
}

void Spmv::warp_reduce_sym_spmv(Float                         a,
                                Eigen::Matrix3d*              triplet_values,
                                int*                          row_ids,
                                int*                          col_ids,
                                int                           triplet_count,
                                muda::CDenseVectorView<Float> x,
                                Float                         b,
                                muda::DenseVectorView<Float>  y,
                                const int*                    s4_active,
                                const int*                    s4_dof_to_group,
                                int                           s4_ng)

{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    // [perf/fast-spmv] merged/isolated (STIFF_FAST_GRAD) don't need the binned order-free y:
    // accumulate straight into y with atomicAdd — skips the ybin memset (~8MB/call: THE dominant
    // memset load) and the combine pass. strict (no FAST_GRAD / SPMV_DET set) keeps the binned path.
    static int s_fast = -1;
    if(s_fast < 0)
        s_fast = getenv("STIFF_SPMV_DET") ? 0 : 1;   // [det-gating] ybin(order-free y) = strict-only
    const bool fast = (s_fast == 1);

    // [multi-env determinism 4.3] grow + zero the binned y accumulator. The matvec deposits
    // the a*A*x contributions here (order-independent) and we combine into y afterwards, so
    // the result is bit-identical regardless of atomic/thread order.
    if(!fast)
    {
        size_t need = (size_t)y.size() * BINNED_K;
        if(need > m_ybin_cap)
        {
            if(m_ybin)
                cudaFree(m_ybin);
            cudaMalloc((void**)&m_ybin, need * sizeof(double));
            m_ybin_cap = need;
            cudaMemset(m_ybin, 0, need * sizeof(double));   // once at (re)alloc
        }
        // [strict-perf] NO per-call memset: the combine kernel re-zeroes each bin after reading it
        // (read-then-zero == memset-before-next-deposit; ybin is spmv-private) — removes the ~8MB
        // memset EVERY spmv (the dominant memset load). Bit-identical by construction.
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 256;
    int block_count = (triplet_count + block_dim - 1) / block_dim;

    // [env-det] when set, deposit each row contribution per-entry (fully order-free) instead of
    // warp-segmented-reducing first — the warp reduce sums in lane order, which differs cross-env
    // for mirror rows at different global triplet positions (the last 1-ULP cross-env seed).
    bool det = (getenv("STIFF_SPMV_DET") != nullptr);

    // [seg-fused dot] one-shot accumulator pointer (consumed per call; seg_pcg re-sets each iter).
    double* seg_dot = m_seg_dot_partials;
    int     seg_ng  = m_seg_dot_ng;
    m_seg_dot_partials = nullptr;

    muda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a     = a,
             Mats3 = triplet_values,
             rows  = row_ids,
             cols  = col_ids,
             triplet_count,
             x = x.viewer().name("x"),
             b = b,
             det = det,
             fast = fast,
             seg_dot, seg_ng,
             s4_active, s4_dof_to_group, s4_ng,
             ybin = m_ybin,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
                if(global_thread_id >= triplet_count)
                    return;
                // [multi-env S4] skip triplets whose row env is masked. y[masked]
                // stays 0 (pre-zeroed when b==0) and masked p is 0 -> result
                // identical, compute saved. Row/col same env after P1.
                if(s4_active)
                {
                    int rg = s4_dof_to_group[rows[global_thread_id]];
                    if(rg >= 0 && rg < s4_ng && s4_active[rg] == 0)
                        return;
                }
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];

                int     prev_i = -1;
                int     i      = -1;
                char    flags;
                Vector3 vec;
                double  dotc = 0.0;   // [seg-fused dot] this triplet's contribution to x·(aAx)

                // set the previous row index
                if(global_thread_id > 0)
                {
                    //auto prev_triplet = A(global_thread_id - 1);
                    prev_i = rows[global_thread_id - 1];
                }


                {
                    //auto Triplet = Mats3[];
                    i                = rows[global_thread_id];
                    auto j           = cols[global_thread_id];
                    auto block_value = Mats3[global_thread_id];
                    vec = block_value * x.segment<N>(j * N).as_eigen();

                    //flags.is_valid = 1;

                    if(i != j)  // process lower triangle
                    {
                        Vector3 vec_ = a * block_value.transpose()
                                       * x.segment<N>(i * N).as_eigen();

                        if(fast)
                        {
                            atomicAdd(&y(j * N + 0), vec_(0));
                            atomicAdd(&y(j * N + 1), vec_(1));
                            atomicAdd(&y(j * N + 2), vec_(2));
                        }
                        else
                        {
                        // [4.3] deterministic deposit instead of y.atomic_add(vec_)
                        binned_deposit(ybin + (size_t)(j * N + 0) * BINNED_K, vec_(0));
                        binned_deposit(ybin + (size_t)(j * N + 1) * BINNED_K, vec_(1));
                        binned_deposit(ybin + (size_t)(j * N + 2) * BINNED_K, vec_(2));
                        }
                        // [seg-fused dot] x_row·(a·block·x_col) + x_col·(a·blockᵀ·x_row).
                        // NOTE: segments re-materialized INLINE (never store the muda segment
                        // proxy in an auto/eval temporary — dangling Map ⇒ 0x0 writes).
                        if(seg_dot)
                            dotc = a * x.segment<N>(i * N).as_eigen().dot(vec)
                                   + x.segment<N>(j * N).as_eigen().dot(vec_);
                    }
                    else if(seg_dot)   // i==j: x_row == x_col
                        dotc = a * x.segment<N>(j * N).as_eigen().dot(vec);
                }

                // [seg-fused dot] block-hashed strided partials keep the per-env atomics cool
                // (ng*PSTRIDE slots); the caller's combine kernel sums + re-zeroes them.
                if(seg_dot && dotc != 0.0)
                {
                    int g = s4_dof_to_group ? s4_dof_to_group[i] : 0;
                    if(g >= 0 && g < seg_ng)
                        atomicAdd(&seg_dot[(size_t)g * 256 + (blockIdx.x & 255)], dotc);
                }

                if(det)
                {
                    // [env-det] per-entry deposit (no warp reduce) ⇒ fully order-free ⇒ the row
                    // contribution is bit-identical regardless of warp/triplet layout ⇒ cross-env mirror.
                    Vector3 result = a * vec;
                    binned_deposit(ybin + (size_t)(i * N + 0) * BINNED_K, result(0));
                    binned_deposit(ybin + (size_t)(i * N + 1) * BINNED_K, result(1));
                    binned_deposit(ybin + (size_t)(i * N + 2) * BINNED_K, result(2));
                }
                else
                {
                if((lane_id == 0) || (prev_i != i))
                {
                    flags = 1;
                }
                else
                {
                    flags = 0;
                }

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });
                // ----------------------------------- warp reduce -----------------------------------------------


                if(flags)
                {
                    Vector3 result = a * vec;
                    if(fast)
                    {   // [perf/fast-spmv] straight atomic accumulate (no binned round-trip)
                        atomicAdd(&y(i * N + 0), result(0));
                        atomicAdd(&y(i * N + 1), result(1));
                        atomicAdd(&y(i * N + 2), result(2));
                    }
                    else
                    {
                    // [4.3] deterministic deposit instead of seg_y.atomic_add(a*vec)
                    binned_deposit(ybin + (size_t)(i * N + 0) * BINNED_K, result(0));
                    binned_deposit(ybin + (size_t)(i * N + 1) * BINNED_K, result(1));
                    binned_deposit(ybin + (size_t)(i * N + 2) * BINNED_K, result(2));
                    }
                }
                }
            });

    if(fast)
        return;   // [perf/fast-spmv] y already fully accumulated — no binned combine needed

    // [multi-env determinism 4.3] combine the binned a*A*x contributions into y (which already
    // holds b*y or 0). Order-independent ⇒ the matvec is now bit-identical run-to-run.
    muda::ParallelFor()
        .kernel_name("spmv_binned_combine")
        .apply(y.size(),
               [ybin = m_ybin, y = y.viewer().name("y")] __device__(int d) mutable
               {
                   double* b = ybin + (size_t)d * BINNED_K;
                   y(d) += binned_combine(b);
#pragma unroll
                   for(int kk = 0; kk < BINNED_K; ++kk) b[kk] = 0.0;   // re-zero for the next spmv
               });
}
}  // namespace gipc
