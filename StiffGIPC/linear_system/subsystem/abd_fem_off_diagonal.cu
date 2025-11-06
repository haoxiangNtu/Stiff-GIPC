#include <linear_system/subsystem/abd_fem_off_diagonal.h>
#include <abd_system/abd_system.h>

namespace gipc
{
ABDFEMOffDiagonal::ABDFEMOffDiagonal(GIPC&               gipc,
                                     device_TetraData&   tetra_data,
                                     ABDLinearSubsystem& abd,
                                     FEMLinearSubsystem& fem)
    : Base(abd, fem)
    , m_gipc(gipc)
    , m_abd_sim_data(abd.m_abd_sim_data)
    , m_tetra_data(tetra_data)
{
}

muda::CBufferView<int> ABDFEMOffDiagonal::boundary_type() const
{
    auto fem_offset = m_gipc.abd_fem_count_info.fem_point_offset;
    auto fem_count  = m_gipc.abd_fem_count_info.fem_point_num;
    return muda::CBufferView<int>(m_tetra_data.BoundaryType, fem_offset, fem_count);
}

void ABDFEMOffDiagonal::report_subsystem_info()
{
    //tex: $\mathbf{J} \mathbf{H}$
    auto upper_hessian_count = m_gipc.gipc_global_triplet.abd_fem_contact_num;  //m_contact_system.abd_fem_contact_hessians().size() * 4;
    auto lower_hessian_count = m_gipc.gipc_global_triplet.fem_abd_contact_num;  //m_contact_system.fem_abd_contact_hessians().size() * 4;

    hessian_block_count(upper_hessian_count, lower_hessian_count);
}

void ABDFEMOffDiagonal::assemble(TripletMatrixView upper, TripletMatrixView lower)
{

    auto count = m_gipc.gipc_global_triplet.fem_abd_contact_num;
    if(count < 1)
        return;
    using namespace muda;

    auto abd_body_num_offset = m_gipc.abd_fem_count_info.abd_body_num * 4;


    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(m_gipc.gipc_global_triplet.abd_fem_contact_num,
               [triplet_b = m_gipc.gipc_global_triplet.block_values(
                    m_gipc.gipc_global_triplet.h_abd_fem_contact_start_id),
                rows = m_gipc.gipc_global_triplet.block_row_indices(
                    m_gipc.gipc_global_triplet.h_abd_fem_contact_start_id),
                cols = m_gipc.gipc_global_triplet.block_col_indices(
                    m_gipc.gipc_global_triplet.h_abd_fem_contact_start_id),
                upper = upper.viewer().name("upper"),
                abd_body_num_offset] __device__(int i) mutable
               {
                   upper(i).write(rows[i], cols[i] - abd_body_num_offset, triplet_b[i]);
               });

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(m_gipc.gipc_global_triplet.fem_abd_contact_num,
               [triplet_b = m_gipc.gipc_global_triplet.block_values(
                    m_gipc.gipc_global_triplet.h_fem_abd_contact_start_id),
                rows = m_gipc.gipc_global_triplet.block_row_indices(
                    m_gipc.gipc_global_triplet.h_fem_abd_contact_start_id),
                cols = m_gipc.gipc_global_triplet.block_col_indices(
                    m_gipc.gipc_global_triplet.h_fem_abd_contact_start_id),
                lower = lower.viewer().name("lower"),
                abd_body_num_offset] __device__(int i) mutable
               {
                   lower(i).write(rows[i] - abd_body_num_offset, cols[i], triplet_b[i]);
               });

}
}  // namespace gipc