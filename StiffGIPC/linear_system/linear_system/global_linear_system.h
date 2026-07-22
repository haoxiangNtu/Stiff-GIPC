#pragma once
#include <cstdint>
#include <list>
#include <linear_system/utils/spmv.h>
#include <linear_system/utils/converter.h>
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <muda/ext/linear_system.h>
#include <gipc/utils/json.h>


namespace gipc
{
class GlobalLinearSystem
{
    template <typename T>
    using U = std::unique_ptr<T>;
    friend class IterativeSolver;
    friend class IPreconditioner;
    friend class ILinearSubsystem;
    friend class LocalPreconditioner;

  public:
    GlobalLinearSystem() {}

    ~GlobalLinearSystem();

    static constexpr int BlockSize = 3;

    template <typename T, typename... Args>
    T& create(Args&&... args)
    {
        if constexpr(std::is_base_of_v<ILinearSubsystem, T>)
        {
            return static_cast<T&>(
                _create_subsystem(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IterativeSolver, T>)
        {
            return static_cast<T&>(
                _create_solver(std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else if constexpr(std::is_base_of_v<IPreconditioner, T>)
        {
            return static_cast<T&>(_create_preconditioner(
                std::make_unique<T>(std::forward<Args>(args)...)));
        }
        else
        {
            MUDA_ASSERT(false, "Unknown type");
        }
    }

    /**
     * \brief solve the global linear system, using the specified solver
     * 
     * \details `solve_linear_system()` will:
     * - build the global linear system from the subsystems
     * - distribute the assembly assignments to the subsystems
     * - solve the linear system using the specified solver
     * - distribute the solution to the subsystems
     */
    gipc::SizeT solve_linear_system();

    // Explicit frame/user-boundary telemetry read. The fast solve path only
    // updates device counters; callers batch the single D2H here after their
    // frame stream has reached its normal synchronization boundary.
    IterativeSolverStats collect_solver_stats(
        bool reset_pending = true,
        cudaStream_t stream = cudaStreamPerThread)
    {
        return m_solver ? m_solver->collect_stats(reset_pending, stream)
                        : IterativeSolverStats{};
    }

    Json               as_json() const;
    GIPCTripletMatrix* gipc_global_triplet = nullptr;

    // [multi-env S4] per-env mask injected by GIPC each Newton iter (null = off).
    // solve_linear_system zeros m_b for masked-env DOFs (RHS-zero -> those envs
    // produce 0, no pollution of the shared PCG alpha/beta); spmv skips masked
    // envs' triplets (the saving). active[ng]: 1=solve,0=masked. dof_to_group
    // is block-indexed (size = block count); a DOF i belongs to block i/3.
    void set_env_mask(const int* active, const int* dof_to_group, int ng)
    { m_s4_active = active; m_s4_dof_to_group = dof_to_group; m_s4_ng = ng; }
    // [seg-fused dot] arm the NEXT spmv to also accumulate the per-env x·(aAx) into `partials`
    // (one-shot; see Spmv::set_seg_dot_accum). Used by seg_pcg's fast path.
    void set_seg_dot_accum(double* partials, int ng) { m_spmv.set_seg_dot_accum(partials, ng); }
    // [seg-fused dot] arm the NEXT apply_preconditioner to accumulate the per-env r·z (of the
    // FINAL z) into `partials`. Returns false (and arms nothing) unless the global AND all local
    // preconditioners are seg_dot_capable — caller falls back to the standalone dot.
    bool arm_precond_seg_dot(double* partials, const int* d2g, int ng)
    {
        if(!m_global_preconditioner || !m_global_preconditioner->seg_dot_capable())
            return false;
        for(auto& p : m_local_preconditioners)
            if(!p->seg_dot_capable())
                return false;
        m_global_preconditioner->arm_seg_dot(partials, d2g, ng);
        for(auto& p : m_local_preconditioners)
            p->arm_seg_dot(partials, d2g, ng);
        return true;
    }
    // [pcg-graph] true iff the global AND all local preconditioners declare their
    // apply() graph-capturable. Gates the PCG CUDA-graph capture (MAS -> false).
    bool precond_graph_capturable() const
    {
        // No global preconditioner (e.g. P_type=1: MAS+ABD are LOCALS) is fine —
        // only registered preconditioners must be capture-safe.
        if(m_global_preconditioner && !m_global_preconditioner->graph_capturable())
            return false;
        for(auto& p : m_local_preconditioners)
            if(!p->graph_capturable())
                return false;
        return true;
    }

    // P1 cache-key inputs. Counts map to fixed launch-capacity tiers; pointer
    // and preconditioner signatures prevent an executable from surviving a
    // topology/reallocation boundary with stale captured kernel arguments.
    int           pcg_dof_tier() const;
    int           pcg_unique_tier() const;
    std::uint64_t pcg_preconditioner_signature() const;
    std::uint64_t pcg_binding_signature() const;
    bool          pcg_spmv_workspace_ready() const
    {
        return m_spmv.graph_workspace_ready(m_b.size());
    }
    const int* m_s4_active       = nullptr;
    const int* m_s4_dof_to_group = nullptr;
    int        m_s4_ng           = 0;

  private:
    std::vector<U<ILinearSubsystem>> m_subsystems;
    std::vector<DiagonalSubsystem*>  m_inner_subsystems;

    std::vector<U<LocalPreconditioner>> m_local_preconditioners;
    U<GlobalPreconditioner>             m_global_preconditioner;
    U<IterativeSolver>                  m_solver;

    muda::LinearSystemContext      m_context;
    muda::DeviceDenseVector<Float> m_x;
    muda::DeviceDenseVector<Float> m_b;

    std::vector<SizeT> m_rhs_count_per_subsystem;
    std::vector<SizeT> m_rhs_offset_per_subsystem;
    std::vector<Float> m_accuracy_statisfied_per_subsystem;

    size_t                         reserved_triplet_count = 0;
    Spmv                           m_spmv;
    Converter                      m_converter;
    muda::DeviceDenseVector<Float> fake_y;


    bool build_linear_system();
    void distribute_solution();
    void apply_preconditioner(muda::DenseVectorView<Float>  z,
                              muda::CDenseVectorView<Float> r);

    void convert_new();

    void spmv(Float a, muda::CDenseVectorView<Float> x, Float b, muda::DenseVectorView<Float> y);

    DiagonalSubsystem& _create_subsystem(U<DiagonalSubsystem>&& subsystem);

    IterativeSolver& _create_solver(U<IterativeSolver>&& solver);
    LocalPreconditioner& _create_preconditioner(U<LocalPreconditioner>&& preconditioner);
    GlobalPreconditioner& _create_preconditioner(U<GlobalPreconditioner>&& preconditioner);
};
}  // namespace gipc
