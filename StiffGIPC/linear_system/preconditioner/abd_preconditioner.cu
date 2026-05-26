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

    // ncu: with default block, N=574 bodies -> grid=2, only 2/114 SMs busy.
    // 32 threads/block (1 warp) -> grid=18 -> 18 SMs.
    ParallelFor(32)
        .kernel_name(__FUNCTION__)
        .apply(abd_body_count,
               [r = r.viewer().name("r"),
                z = z.viewer().name("z"),
                inv = abd_inv_diag.viewer().name("inv")] __device__(int i) mutable
               {
                   z.segment<12>(i * 12).as_eigen() =
                       inv(i) * r.segment<12>(i * 12).as_eigen();
               });
}
}  // namespace gipc
