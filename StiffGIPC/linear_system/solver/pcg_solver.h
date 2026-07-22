#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <cstdint>
#include <vector>

namespace gipc
{
// Stable device ABI for the P1 PCG continuation graph.  The graph owns
// `iteration` while it is running; a future P2 Newton-decision executable can
// be supplied through `successor_exec` without changing the cached PCG graph.
struct alignas(16) PCGDeviceState
{
    unsigned long long iteration      = 1;
    unsigned long long max_iteration  = 0;
    unsigned long long successor_exec = 0;
    unsigned long long solve_serial   = 0;
    int iteration_active = 0;
    int converged        = 0;
    int terminal         = 0;
    int segmented        = 0;
};

struct alignas(16) PCGDeviceCounters
{
    unsigned long long total_iterations        = 0;
    unsigned long long pending_iterations      = 0;
    unsigned long long solve_count             = 0;
    unsigned long long device_graph_solve_count = 0;
    unsigned long long fallback_solve_count    = 0;
};

class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float global_tol_rate = 1e-4;
    bool  use_bsr         = true;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver();

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

    IterativeSolverStats collect_stats(
        bool reset_pending = true,
        cudaStream_t stream = cudaStreamPerThread) override;

    // P2 hook. The successor must be an uploaded device-launchable executable
    // owned by the same engine; nullptr keeps the P1 host phase boundary.
    void set_device_continuation(cudaGraphExec_t successor)
    {
        m_successor_exec = static_cast<unsigned long long>(
            reinterpret_cast<std::uintptr_t>(successor));
    }

  private:

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p

    // Device-side scalars to keep alpha/beta/rz/dot_res off the host hot path.
    // Each PCG iter previously synced to host twice (one cudaMemcpy per dot
    // product); with these on device, host only syncs every K iters to read
    // the convergence flag.
    Float*     d_rz       = nullptr;
    Float*     d_rz0      = nullptr;
    Float*     d_rz_new   = nullptr;
    Float*     d_dot_res  = nullptr;
    Float*     d_alpha    = nullptr;
    Float*     d_beta     = nullptr;
    int*       d_break    = nullptr;
    // [device-loop graph] state and counters remain device-resident across
    // solve() calls. Fast solves never read either allocation back.
    PCGDeviceState*    d_graph_state = nullptr;
    PCGDeviceCounters* d_counters    = nullptr;
    struct GraphCacheKey
    {
        int device       = -1;
        int mode         = 0;
        int dof_count    = 0;
        int dof_tier     = 0;
        int unique_tier  = 0;
        int group_count  = 0;
        int check_k      = 0;
        std::uint64_t preconditioner = 0;
        std::uint64_t bindings       = 0;
        std::uint64_t features       = 0;
        std::uint64_t tolerance_bits = 0;

        bool operator==(const GraphCacheKey& rhs) const;
    };

    struct GraphCacheEntry
    {
        GraphCacheKey key;
        cudaGraphExec_t exec = nullptr;
    };

    // Per-engine executable family. Entries are immutable after their first
    // instantiate+upload; a tier/key hit goes straight to cudaGraphLaunch.
    std::vector<GraphCacheEntry> m_graph_cache;
    unsigned long long m_graph_cache_hits     = 0;
    unsigned long long m_graph_cache_misses   = 0;
    unsigned long long m_graph_capture_errors = 0;
    unsigned long long m_successor_exec       = 0;
    int        h_break    = 0;
    bool       d_scalars_alloced = false;

    // Step E: cub::DeviceReduce temp storage (alloc'd lazily on first dot).
    void*      cub_temp_ptr   = nullptr;
    size_t     cub_temp_bytes = 0;

    // [multi-env P3] SEGMENTED (block-diagonal) PCG state. The matrix is block-diagonal after P1
    // (no cross-env contacts) and the preconditioner is intra-env, so the ONLY cross-env coupling
    // in the solve is the GLOBAL dot products (one scalar α/β/convergence over all envs). Per-env
    // dots (binned → exact, order-independent ⇒ per-env deterministic AND cross-env symmetric)
    // + per-env α/β/convergence make each env a mathematically INDEPENDENT solve → cross-env
    // bit-identical for identical envs. Gated (STIFF_SEGMENTED_PCG + dof_to_group present).
    // A single active group deliberately stays on this path: batch count must not change the
    // numerical algorithm, and ng=1 launches only one real slot rather than the slot capacity.
    int        m_seg_ng       = 0;          // allocated active-group capacity; 0 ⇒ unallocated
    Float*     d_rz_g         = nullptr;    // [ng] per-env r·z
    Float*     d_rz0_g        = nullptr;    // [ng] per-env initial r·z
    Float*     d_rzn_g        = nullptr;    // [ng] per-env new r·z
    Float*     d_dot_g        = nullptr;    // [ng] per-env p·Ap
    double*    d_segbin        = nullptr;   // [ng*BINNED_K] segmented binned-dot accumulator
    int*       d_break_g      = nullptr;    // [ng] per-env converged flag
    double*    d_dot_partials = nullptr;    // [ng*256] spmv-fused per-env dot partials (fast path)
    // [warm-start (1)] per-env safeguarded warm start (STIFF_PCG_WARM, default off):
    Float*     d_rr_g         = nullptr;    // [ng] ||b - A*x0||^2 per env
    Float*     d_bb_g         = nullptr;    // [ng] ||b||^2 per env
    int*       d_use_warm     = nullptr;    // [ng] per-env decision: 1 = keep x0, 0 = reset to zero
    // [E-W (2)] per-env adaptive forcing terms (STIFF_PCG_EW, default off):
    Float*     d_ew_prev      = nullptr;    // [ng] previous solve's rz0_g (gradient-norm^2 proxy)
    Float*     d_tol2_g       = nullptr;    // [ng] this solve's per-env tolerance (eta^2)
    bool       m_seg_alloced  = false;

    PCGSolverConfig   m_config;

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
    // [multi-env P3] segmented block-diagonal PCG (per-env α/β/convergence). dof_to_group is
    // block-indexed (DOF i → block i/3 → group dof_to_group[i/3]).
    SizeT seg_pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter,
                  const int* dof_to_group, int ng);
    // segmented binned dot: out_g[g] = Σ_{i: dof_to_group[i/3]==g} a[i]*b[i] (exact, per-env).
    void seg_dot(const Float* a, const Float* b, const int* d2g, int ng, int n, Float* out_g);

    GraphCacheKey make_graph_cache_key(int mode,
                                       muda::DenseVectorView<Float> x,
                                       int group_count,
                                       SizeT check_k,
                                       std::uint64_t features,
                                       double tolerance) const;
    GraphCacheEntry* find_graph_cache(const GraphCacheKey& key);
    void clear_graph_cache();
};
}  // namespace gipc
