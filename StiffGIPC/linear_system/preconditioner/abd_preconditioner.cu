#include <linear_system/preconditioner/abd_preconditioner.h>
#include <abd_system/abd_system.h>
#include <linear_system/subsystem/abd_linear_subsystem.h>

namespace gipc
{
ABDPreconditioner::ABDPreconditioner(ABDLinearSubsystem& subsystem, ABDSystem& abd, ABDSimData& sim_data)
    : Base(subsystem)
    , m_abd(abd)
    , m_sim_data(sim_data)
{
    preconditioner_id = 0;
}

void ABDPreconditioner::assemble()
{
    m_abd._cal_abd_system_preconditioner(m_sim_data);
}

void ABDPreconditioner::apply(muda::CDenseVectorView<Float> r,
                              muda::DenseVectorView<Float>  z)
{
    using namespace muda;

    auto abd_body_count = m_sim_data.abd_fem_count_info().abd_body_num;
    auto abd_inv_diag   = m_abd.abd_system_diag_preconditioner.view();

    // [seg-fused dot] one-shot: also accumulate the per-env correction r·(z_new − z_old)
    // (z_old = the global diag preconditioner's z, read before overwrite) so the running
    // per-env Σ equals r·z of the FINAL z. Body i's first 3x3-block index = i*4 (ABD at DOF 0).
    double*    sd_p   = m_sd_partials;
    const int* sd_d2g = m_sd_d2g;
    int        sd_ng  = m_sd_ng;
    m_sd_partials     = nullptr;

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_body_count,
               [r = r.viewer().name("r"),
                z = z.viewer().name("z"),
                inv = abd_inv_diag.viewer().name("inv"),
                sd_p, sd_d2g, sd_ng] __device__(int i) mutable
               {
                   using Vec12 = Eigen::Matrix<double, 12, 1>;
                   if(sd_p)
                   {
                       Vec12 zold = z.segment<12>(i * 12).as_eigen();
                       Vec12 znew = inv(i) * r.segment<12>(i * 12).as_eigen();
                       z.segment<12>(i * 12).as_eigen() = znew;
                       double c = (znew - zold).dot(r.segment<12>(i * 12).as_eigen());
                       int    g = sd_d2g ? sd_d2g[i * 4] : -1;
                       if(c != 0.0 && g >= 0 && g < sd_ng)
                           atomicAdd(&sd_p[(size_t)g * 256 + (i & 255)], c);
                       return;
                   }
                   z.segment<12>(i * 12).as_eigen() =
                       inv(i) * r.segment<12>(i * 12).as_eigen();
               });
}
}  // namespace gipc
