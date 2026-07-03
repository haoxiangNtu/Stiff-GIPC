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
    // NOTE: STIFF_KSUM diagnostic still syncs — capture failure falls back to the
    // plain PCG loop gracefully, so the diag flag stays usable.
    // LEFT false for v0.8.0: apply() itself is capture-safe now (symbol binds
    // hoisted, memsets Async) but engaging the merged graph on a MAS scene still
    // trips a muda wait inside the captured region (dangling capture -> teardown
    // cudaFree storm). MAS runs the plain PCG loop (healthy, case_26 4.3s);
    // the ~3% graph gain moves to the MAS-B follow-up.
    bool graph_capturable() const override { return false; }
    double*            masses;
    uint32_t*          cpNum;

  public:
    MAS_Preconditioner(FEMLinearSubsystem& subsystem, MASPreconditioner& mMAS, double* mMasses, uint32_t* mCpNum);
    virtual void assemble() override;
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;
    //const int preconditioner_id = 1;
};
}  // namespace gipc
