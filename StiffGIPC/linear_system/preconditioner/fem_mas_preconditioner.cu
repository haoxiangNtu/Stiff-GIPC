#include <linear_system/preconditioner/fem_mas_preconditioner.h>
#include <linear_system/subsystem/fem_linear_subsystem.h>
#include <gipc/utils/timer.h>
namespace gipc
{
MAS_Preconditioner::MAS_Preconditioner(FEMLinearSubsystem& subsystem,
                                       MASPreconditioner&  mMAS,
                                       double*             mMasses,
                                       uint32_t*           mCpNum)
    : Base(subsystem)
    , MAS_Prec(mMAS)
    , masses(mMasses)
    , cpNum(mCpNum)
{
    preconditioner_id = 1;
}

void MAS_Preconditioner::assemble()
{
    int collision_num = static_cast<int>(*cpNum);
    gipc::Timer timer{"precomputing mas Preconditioner"};
    const bool  device_levels  = MAS_Prec.deviceLevelsEnabled();
    int         triplet_number = 0;
    uint32_t*   indices =
        calculate_subsystem_bcoo_indices(triplet_number, device_levels);
    MAS_Prec.setPreconditioner_bcoo(system_bcoo_matrix(),
                                    system_bcoo_rows(),
                                    system_bcoo_cols(),
                                    indices,
                                    get_offset(),
                                    triplet_number,
                                    device_levels ? subsystem_bcoo_count_device() : nullptr,
                                    collision_num);
}

void MAS_Preconditioner::apply(muda::CDenseVectorView<Float> r,
                              muda::DenseVectorView<Float>  z)
{
    MAS_Prec.preconditioning((double3*)r.data(), (double3*)z.data());
}
}  // namespace gipc
