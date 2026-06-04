#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <cuda_tools/cuda_device_buffer.h>
#include <Eigen/Eigen>
#include <cstdio>
#include <cstdlib>

// [cp-stats] global-scope (not gipc::) trackers defined in GIPC.cu
extern uint32_t g_peak_triplet_used;
extern uint32_t g_triplet_reserved;

namespace gipc
{

// ============================================================================
// ① PoC: Sparsity-pattern stability detector
// ----------------------------------------------------------------------------
// After each Newton iter's global matrix build, compare the sorted unique
// (row,col) hash array to the previous Newton iter's. Equal hash arrays =>
// pattern unchanged => future ① sparsity caching can skip the radix sort and
// dedup, reusing the cached permutation. Pure INSTRUMENTATION here: doesn't
// change correctness or speed (slight overhead from the 1-int D2H per call).
//
// Enable logging by setting env STIFF_PATTERN_LOG=1.
// ============================================================================
namespace
{
__global__ void __pattern_compare_kernel(const uint64_t* cur,
                                         const uint64_t* prev,
                                         int             N,
                                         int*            diff_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    if(cur[idx] != prev[idx]) atomicOr(diff_flag, 1);
}

struct PatternDetectorState
{
    cudatool::CudaDeviceBuffer<uint64_t> prev_buf;
    int  prev_count    = -1;
    int* d_diff_flag   = nullptr;
    int  total_calls   = 0;
    int  matched_calls = 0;
    bool log_enabled   = false;
    bool log_checked   = false;
};
PatternDetectorState g_pat;

void check_pattern_log_env()
{
    if(g_pat.log_checked) return;
    const char* e = std::getenv("STIFF_PATTERN_LOG");
    g_pat.log_enabled = (e && *e && *e != '0');
    g_pat.log_checked = true;
}

void detect_pattern_change(GIPCTripletMatrix* gt)
{
    check_pattern_log_env();
    int             N       = gt->h_unique_key_number;
    const uint64_t* current = gt->block_hash_value();

    if(g_pat.d_diff_flag == nullptr)
        CUDA_SAFE_CALL(cudaMalloc(&g_pat.d_diff_flag, sizeof(int)));

    g_pat.total_calls++;
    int matched = 0;

    if(g_pat.prev_count == N && N > 0)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(g_pat.d_diff_flag, 0, sizeof(int)));
        int blocks = (N + 255) / 256;
        __pattern_compare_kernel<<<blocks, 256>>>(
            current, g_pat.prev_buf.data(), N, g_pat.d_diff_flag);
        int h_diff = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&h_diff, g_pat.d_diff_flag,
                                  sizeof(int), cudaMemcpyDeviceToHost));
        if(h_diff == 0) { g_pat.matched_calls++; matched = 1; }
    }

    // Save current as prev for next compare
    g_pat.prev_buf.resize(N);
    if(N > 0)
        CUDA_SAFE_CALL(cudaMemcpyAsync(g_pat.prev_buf.data(), current,
                                       N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    g_pat.prev_count = N;

    if(g_pat.log_enabled)
        fprintf(stderr, "[PATTERN] call=%d N=%d matched=%d cum=%d/%d (%.1f%%)\n",
                g_pat.total_calls, N, matched,
                g_pat.matched_calls, g_pat.total_calls,
                100.0 * g_pat.matched_calls / g_pat.total_calls);
}
}  // namespace

// Exposed so the headless harness can print final stats.
void gipc_pattern_get_stats(int* total, int* matched)
{
    if(total)   *total   = g_pat.total_calls;
    if(matched) *matched = g_pat.matched_calls;
}

bool GlobalLinearSystem::build_linear_system()
{
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);
    // PCG's initial guess must start finite; ::resize() leaves new elements
    // uninitialized and the IterativeSolver reads x on the very first spmv.
    CUDA_SAFE_CALL(cudaMemset(m_x.view().data(), 0,
                              total_rhs_count * sizeof(Float)));

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();

    // ① PoC: detect whether the sparsity PATTERN (sorted unique (row,col)
    // hashes) is unchanged from the previous Newton iter. Statistic only.
    detect_pattern_change(gipc_global_triplet);

    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);

    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    muda::wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    MUDA_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
        return 0;
    MUDA_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    auto iter = m_solver->solve(m_x, m_b);
    distribute_solution();
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                              muda::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



// ============================================================================
// ① Sparsity-cache fingerprint kernel + helpers
// ============================================================================
namespace
{
__global__ void __compute_input_fingerprint(const int* rows,
                                            const int* cols,
                                            int        n,
                                            unsigned long long* out)
{
    // Hierarchical XOR reduction (warp-shuffle → block-shared → one atomic
    // per block).  Previous version did `atomicXor(out, key)` per thread →
    // tens of thousands of atomics serialized on one L2 line (was 6% of
    // case39 GPU time / 376 µs avg).  This pattern keeps the same XOR-fold
    // semantics (XOR is associative + commutative) at fraction of the cost.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long key = 0ULL;
    if(idx < n)
    {
        key = (static_cast<unsigned long long>(static_cast<unsigned int>(rows[idx])) << 32)
            | static_cast<unsigned long long>(static_cast<unsigned int>(cols[idx]));
        // Splittable-mix so trivial (row,col) swaps don't collide.
        key ^= key >> 33; key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33; key *= 0xc4ceb9fe1a85ec53ULL;
        key ^= key >> 33;
    }
    // Warp-level XOR reduction (32 → 1 lane via butterfly shuffle).
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
        key ^= __shfl_xor_sync(0xFFFFFFFFu, key, offset);
    // Each warp's lane 0 now holds the warp-XOR.  Combine warps in shared mem.
    constexpr int MAX_WARPS = 32;  // block size capped at 1024 = 32 warps
    __shared__ unsigned long long s_warp[MAX_WARPS];
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if(laneId == 0)
        s_warp[warpId] = key;
    __syncthreads();
    // Final reduction by warp 0 (max blockDim/32 warps).
    if(warpId == 0)
    {
        int    nWarps = (blockDim.x + 31) >> 5;
        unsigned long long acc = (laneId < nWarps) ? s_warp[laneId] : 0ULL;
        #pragma unroll
        for(int offset = 16; offset > 0; offset >>= 1)
            acc ^= __shfl_xor_sync(0xFFFFFFFFu, acc, offset);
        if(laneId == 0)
            atomicXor(out, acc);   // one atomic per block, not per thread
    }
}

// Skip-path gather: dst_val[i] = src_val[cached_perm[i]]
// In-place safe — same pattern the original _radix_sort_indices_and_blocks uses.
__global__ void __gather_values_by_perm(Eigen::Matrix3d*       dst_val,
                                        const Eigen::Matrix3d* src_val,
                                        const uint32_t*        perm,
                                        int                    n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    dst_val[idx] = src_val[perm[idx]];
}

// ───────────────────────────────────────────────────────────────────────────
// ③ Assembly CUDA Graph (skip-path):
// The convert_new() skip path is 6 sync-free ops (3 D2D + gather kernel +
// memset + FastSegmentalReduce). Capture them once into a CUDA graph and
// replay across Newton iters when the cached pointers/sizes match. Invalidate
// when length, N_unique, or any buffer pointer changes (resize realloc).
// ───────────────────────────────────────────────────────────────────────────
struct SkipGraphState
{
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int             length     = -1;
    int             N_unique   = -1;
    const void*     src_perm   = nullptr;
    const void*     src_part   = nullptr;
    const void*     src_row    = nullptr;
    const void*     src_col    = nullptr;
    const void*     src_hash   = nullptr;
    const void*     dst_row    = nullptr;
    const void*     dst_col    = nullptr;
    const void*     dst_hash   = nullptr;
    const void*     values     = nullptr;
    int             replays    = 0;
    int             captures   = 0;
};
SkipGraphState g_skip_g;

bool skip_graph_can_replay(GIPCTripletMatrix* gt, int length, int N_unique)
{
    return g_skip_g.graph_exec != nullptr
           && g_skip_g.length   == length
           && g_skip_g.N_unique == N_unique
           && g_skip_g.src_perm == gt->m_cache_sort_index.data()
           && g_skip_g.src_part == gt->m_cache_partition_output.data()
           && g_skip_g.src_row  == gt->m_cache_unique_row.data()
           && g_skip_g.src_col  == gt->m_cache_unique_col.data()
           && g_skip_g.src_hash == gt->m_cache_unique_hash.data()
           && g_skip_g.dst_row  == gt->block_row_indices()
           && g_skip_g.dst_col  == gt->block_col_indices()
           && g_skip_g.dst_hash == gt->block_hash_value()
           && g_skip_g.values   == gt->block_values();
}

void skip_graph_destroy()
{
    if(g_skip_g.graph_exec) { cudaGraphExecDestroy(g_skip_g.graph_exec); g_skip_g.graph_exec = nullptr; }
    if(g_skip_g.graph)      { cudaGraphDestroy(g_skip_g.graph);          g_skip_g.graph      = nullptr; }
}
}  // namespace

// Expose for harness stats.
void gipc_skip_graph_get_stats(int* replays, int* captures)
{
    if(replays)  *replays  = g_skip_g.replays;
    if(captures) *captures = g_skip_g.captures;
}

void GlobalLinearSystem::convert_new()
{
    auto*    gt        = gipc_global_triplet;
    const int length   = gt->global_triplet_offset;
    if(length < 1)
        return;

    // [cp-stats] realized global-Hessian triplet count this build vs reserved
    // capacity (the buffer is sized total_internal*32 + collision_max; see the
    // ×32 chain-rule margin). Tracked for STIFF_CP_STATS buffer-utilization report.
    {
        if((uint32_t)length > ::g_peak_triplet_used) ::g_peak_triplet_used = (uint32_t)length;
        ::g_triplet_reserved = (uint32_t)gt->triplet_count();
    }

    // ───────────────────────────────────────────────────────────────────
    // Compute input fingerprint (XOR-fold of mixed (row,col) keys).
    // One D2H of 8 bytes at end. Cheap vs. the radix sort it might skip.
    // ───────────────────────────────────────────────────────────────────
    static unsigned long long* d_fp = nullptr;
    if(d_fp == nullptr) CUDA_SAFE_CALL(cudaMalloc(&d_fp, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_fp, 0, sizeof(unsigned long long)));
    {
        int blocks = (length + 255) / 256;
        __compute_input_fingerprint<<<blocks, 256>>>(
            gt->block_row_indices(), gt->block_col_indices(), length, d_fp);
    }
    unsigned long long h_fp = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_fp, d_fp, sizeof(h_fp), cudaMemcpyDeviceToHost));

    const bool use_skip = gt->m_cache_valid
                          && gt->m_cache_length == length
                          && gt->m_cache_input_fingerprint == h_fp;

    if(use_skip)
    {
        // ─── SKIP PATH ─── pattern matches cache. Skip radix sort + RLE +
        // scatter; only gather new values and segmental-reduce them.
        // Memory layout matches the full path's (out_start_id=length): input
        // at [0:length), scratch at [length:2*length), output at [0:N_unique).
        // ALL six ops below are async / launch-only (no host sync), so they
        // capture into a single CUDA graph that we replay across Newton iters
        // with the same cached pointers — see SkipGraphState above.
        const int N_unique = gt->m_cache_unique_count;

        if(!skip_graph_can_replay(gt, length, N_unique))
        {
            skip_graph_destroy();
            cudaStream_t s = cudaStreamPerThread;
            CUDA_SAFE_CALL(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));

            CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_row_indices(),
                                           gt->m_cache_unique_row.data(),
                                           N_unique * sizeof(int),
                                           cudaMemcpyDeviceToDevice, s));
            CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_col_indices(),
                                           gt->m_cache_unique_col.data(),
                                           N_unique * sizeof(int),
                                           cudaMemcpyDeviceToDevice, s));
            CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_hash_value(),
                                           gt->m_cache_unique_hash.data(),
                                           N_unique * sizeof(uint64_t),
                                           cudaMemcpyDeviceToDevice, s));
            {
                int blocks = (length + 255) / 256;
                __gather_values_by_perm<<<blocks, 256, 0, s>>>(
                    gt->block_values() + length,
                    gt->block_values(),
                    gt->m_cache_sort_index.data(), length);
            }
            CUDA_SAFE_CALL(cudaMemsetAsync(gt->block_values(), 0,
                                           N_unique * sizeof(Eigen::Matrix3d), s));
            muda::FastSegmentalReduce(s)
                .kernel_name("convert_new_skip_graph")
                .reduce(length,
                        gt->m_cache_partition_output.data(),
                        gt->block_values() + length,
                        gt->block_values());

            CUDA_SAFE_CALL(cudaStreamEndCapture(s, &g_skip_g.graph));
            CUDA_SAFE_CALL(cudaGraphInstantiate(&g_skip_g.graph_exec, g_skip_g.graph,
                                                nullptr, nullptr, 0));

            g_skip_g.length   = length;
            g_skip_g.N_unique = N_unique;
            g_skip_g.src_perm = gt->m_cache_sort_index.data();
            g_skip_g.src_part = gt->m_cache_partition_output.data();
            g_skip_g.src_row  = gt->m_cache_unique_row.data();
            g_skip_g.src_col  = gt->m_cache_unique_col.data();
            g_skip_g.src_hash = gt->m_cache_unique_hash.data();
            g_skip_g.dst_row  = gt->block_row_indices();
            g_skip_g.dst_col  = gt->block_col_indices();
            g_skip_g.dst_hash = gt->block_hash_value();
            g_skip_g.values   = gt->block_values();
            g_skip_g.captures++;
        }

        CUDA_SAFE_CALL(cudaGraphLaunch(g_skip_g.graph_exec, cudaStreamPerThread));
        g_skip_g.replays++;

        gt->h_unique_key_number = N_unique;
        return;
    }

    // Full-path = pattern changed → cached buffers may shift → invalidate graph.
    skip_graph_destroy();

    // ─── FULL PATH ─── pattern changed (or first call). Run normal convert
    // (original signature: out_start_id = length so sort writes to
    // [length:2*length) scratch), then save the result into the cache.
    m_converter.convert(*gt, 0, length, length);

    // Save cache (post-convert): permutation, partition, unique pattern arrays.
    const int N_unique = gt->h_unique_key_number;
    gt->m_cache_sort_index.resize(length);
    gt->m_cache_partition_output.resize(length);
    gt->m_cache_unique_row.resize(N_unique);
    gt->m_cache_unique_col.resize(N_unique);
    gt->m_cache_unique_hash.resize(N_unique);

    CUDA_SAFE_CALL(cudaMemcpyAsync(gt->m_cache_sort_index.data(),
                                   gt->block_sort_index(),
                                   length * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gt->m_cache_partition_output.data(),
                                   gt->block_index(),
                                   length * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gt->m_cache_unique_row.data(),
                                   gt->block_row_indices(),
                                   N_unique * sizeof(int),
                                   cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gt->m_cache_unique_col.data(),
                                   gt->block_col_indices(),
                                   N_unique * sizeof(int),
                                   cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(gt->m_cache_unique_hash.data(),
                                   gt->block_hash_value(),
                                   N_unique * sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice));

    gt->m_cache_unique_count      = N_unique;
    gt->m_cache_length            = length;
    gt->m_cache_input_fingerprint = h_fp;
    gt->m_cache_valid             = true;
}



void GlobalLinearSystem::spmv(Float                         a,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
{

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y);
}
}  // namespace gipc
