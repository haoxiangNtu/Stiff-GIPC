#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <cuda_tools/cuda_device_buffer.h>
#include <Eigen/Eigen>
#include <cstdio>
#include <cstdlib>

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    unsigned long long key =
        (static_cast<unsigned long long>(static_cast<unsigned int>(rows[idx])) << 32)
        | static_cast<unsigned long long>(static_cast<unsigned int>(cols[idx]));
    // Mix to spread bits (so trivial swaps don't collide). Splittable-mix.
    key ^= key >> 33; key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33; key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    atomicXor(out, key);
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
}  // namespace

void GlobalLinearSystem::convert_new()
{
    auto*    gt        = gipc_global_triplet;
    const int length   = gt->global_triplet_offset;
    if(length < 1)
        return;

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
        // Memory layout matches the full path's: input values live at
        // block_values[0:length); gather writes sorted-permuted values to
        // block_values[length:2*length); segmental-reduce reads from there
        // and writes aggregated unique values back to block_values[0:N_unique).
        // (Buffer capacity covers this — total_max_global_triplet_num*32 in
        // GIPC::build_gipc_system.)
        const int N_unique = gt->m_cache_unique_count;

        // Restore pattern arrays into their canonical positions (downstream
        // consumers read block_row_indices/col_indices/hash_value[0:N_unique]).
        CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_row_indices(),
                                       gt->m_cache_unique_row.data(),
                                       N_unique * sizeof(int),
                                       cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_col_indices(),
                                       gt->m_cache_unique_col.data(),
                                       N_unique * sizeof(int),
                                       cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpyAsync(gt->block_hash_value(),
                                       gt->m_cache_unique_hash.data(),
                                       N_unique * sizeof(uint64_t),
                                       cudaMemcpyDeviceToDevice));

        // Gather new values through the cached permutation. Write to scratch
        // region [length:2*length) (NOT in-place — avoids the same race the
        // original sort path also avoids by using out_start_id=length).
        {
            int blocks = (length + 255) / 256;
            __gather_values_by_perm<<<blocks, 256>>>(
                gt->block_values() + length,  // dst (sorted values scratch)
                gt->block_values(),           // src (new input values)
                gt->m_cache_sort_index.data(), length);
        }

        // Zero the aggregation range [0:N_unique), then segmental-reduce
        // from the scratch sorted values into it.
        CUDA_SAFE_CALL(cudaMemsetAsync(gt->block_values(),
                                       0,
                                       N_unique * sizeof(Eigen::Matrix3d)));
        muda::FastSegmentalReduce()
            .kernel_name("convert_new_skip")
            .reduce(length,
                    gt->m_cache_partition_output.data(),
                    gt->block_values() + length,  // INPUT (sorted scratch)
                    gt->block_values());          // OUTPUT (aggregated)

        gt->h_unique_key_number = N_unique;
        return;
    }

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
