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

    // Step C: custom stream so we can capture/replay the PCG inner loop
    // as a CUDA graph (eliminates per-launch host overhead). Lazily
    // created on first solve(); destroyed in dtor.
    cudaStream_t pcg_stream = nullptr;

    PCGSolverConfig   m_config;

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
