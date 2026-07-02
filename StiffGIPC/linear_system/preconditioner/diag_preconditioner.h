#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class DiagPreconditioner : public GlobalPreconditioner
{
  public:
    DiagPreconditioner() = default;


  public:

    virtual void assemble(GIPCTripletMatrix& global_triplets) override;

    virtual void apply(muda::CDenseVectorView<gipc::Float> r,
                       muda::DenseVectorView<gipc::Float>  z) override;

    // [seg-fused dot] accumulate per-env r·z (all blocks, diag z) alongside the z-write.
    bool seg_dot_capable() const override { return true; }
    void arm_seg_dot(double* partials, const int* d2g, int ng) override
    { m_sd_partials = partials; m_sd_d2g = d2g; m_sd_ng = ng; }

  private:
    muda::DeviceBuffer<gipc::Matrix3x3> m_diag3x3;
    double*    m_sd_partials = nullptr;   // [seg-fused dot] one-shot
    const int* m_sd_d2g      = nullptr;
    int        m_sd_ng       = 0;
};
}  // namespace gipc
