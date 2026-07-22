#pragma once
#include <linear_system/linear_system/i_preconditioner.h>
class BHessian;
class MASPreconditioner;

namespace gipc
{
class FEMLinearSubsystem;

class MAS_Preconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    MASPreconditioner& MAS_Prec;
    // [MAS graph-capture] apply() is capture-safe since the symbol binds moved to
    // alloc time and the zeroing went cudaMemsetAsync (see MASPreconditioner.cu).
    // Synchronous diagnostics are gated by PCGSolver before capture. The
    // production apply path is capture-safe (symbol binds are hoisted and
    // zeroing is async).
    bool graph_capturable() const override { return true; }
    double*            masses;
    uint32_t*          cpNum;

  public:
    MAS_Preconditioner(FEMLinearSubsystem& subsystem, MASPreconditioner& mMAS, double* mMasses, uint32_t* mCpNum);
    virtual void assemble() override;
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;
    //const int preconditioner_id = 1;
};
}  // namespace gipc
