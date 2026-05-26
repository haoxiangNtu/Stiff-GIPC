#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>

namespace gipc
{
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
    int        h_break    = 0;
    bool       d_scalars_alloced = false;

    // Step E: cub::DeviceReduce temp storage (alloc'd lazily on first dot).
    void*      cub_temp_ptr   = nullptr;
    size_t     cub_temp_bytes = 0;

    // ③ CUDA Graph PoC: capture one PCG iteration body and replay it for
    // iter 2..max_iter (iter 1 runs normally so cub_temp gets allocated first
    // and the captured topology is stable). Rebuilt whenever dof changes
    // across solves (matrix sparsity changes per Newton iteration).
    cudaGraph_t      m_pcg_graph      = nullptr;
    cudaGraphExec_t  m_pcg_graph_exec = nullptr;
    size_t           m_graph_dof      = 0;
    // BLOCKER: capture fails because muda has an internal cudaStreamSynchronize
    // somewhere in the PCG body path (spmv / preconditioner / CUB) that throws
    // cudaErrorStreamCaptureUnsupported even in Relaxed mode. Until that sync
    // is located and eliminated (muda surgery), keep the graph dormant.
    // Flip to true to retry once the muda block is fixed.
    bool             m_graph_enabled  = false;

    PCGSolverConfig   m_config;

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
