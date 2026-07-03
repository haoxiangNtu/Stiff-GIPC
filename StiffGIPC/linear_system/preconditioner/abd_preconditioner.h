#pragma once
#include <linear_system/linear_system/i_preconditioner.h>

namespace gipc
{
class ABDSystem;
class ABDSimData;
class ABDLinearSubsystem;
class ABDPreconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    ABDSystem&  m_abd;
    ABDSimData& m_sim_data;

  public:
    ABDPreconditioner(ABDLinearSubsystem& subsystem, ABDSystem& abd, ABDSimData& sim_data);
    virtual void assemble() override;
    virtual void apply(muda::CDenseVectorView<Float> r, muda::DenseVectorView<Float> z) override;
    //const int preconditioner_id = 0;

    // [seg-fused dot] correction accumulation: r·(z_new − z_old) per body (z_old = the global
    // diag preconditioner's z, read before overwrite) → Σ equals r·z of the FINAL z.
    bool seg_dot_capable() const override { return true; }
    bool graph_capturable() const override { return true; }
    void arm_seg_dot(double* partials, const int* d2g, int ng) override
    { m_sd_partials = partials; m_sd_d2g = d2g; m_sd_ng = ng; }

  private:
    double*    m_sd_partials = nullptr;   // one-shot
    const int* m_sd_d2g      = nullptr;
    int        m_sd_ng       = 0;
};
}  // namespace gipc
