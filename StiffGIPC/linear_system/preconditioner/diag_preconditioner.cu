#include <linear_system/preconditioner/diag_preconditioner.h>
#include <muda/ext/eigen/inverse.h>
#include <gipc/utils/timer.h>
namespace gipc
{
namespace details
{

    void diag_assemble(muda::BufferView<gipc::Matrix<3, 3>>  diag_inv,
                       GIPCTripletMatrix&                   global_triplets)
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(global_triplets.h_unique_key_number,
                   [diag = diag_inv.viewer().name("diag"),
                    hessian = global_triplets.block_values(),
                    rows = global_triplets.block_row_indices(),
                    cols = global_triplets.block_col_indices()] __device__(int I) mutable
                   {
                       auto i           = rows[I];
                       auto j           = cols[I];
                       auto H           = hessian[I];
                       if(i != j)
                           return;

                       diag(i) = eigen::inverse(H);
                   });
    }

    void apply_diag(muda::CDenseVectorView<gipc::Float>  r,
                    muda::DenseVectorView<gipc::Float>   z,
                    muda::BufferView<gipc::Matrix<3, 3>> diag_inv)
    {
        using namespace muda;

        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = r.viewer().name("r"),
                    z = z.viewer().name("z"),
                    diag_inv = diag_inv.viewer().name("diag_inv")] __device__(int I) mutable
                   {
                       auto& D = diag_inv(I);
                       z.segment<3>(I * 3).as_eigen() =
                           D * r.segment<3>(I * 3).as_eigen();
                   });
    }

    // [seg-fused dot] z = D⁻¹r AND per-env accumulate r·z (this block) into strided partials.
    // NOTE: segment proxies consumed inline / named PlainObject copies only (muda proxy trap).
    void apply_diag_seg_dot(muda::CDenseVectorView<gipc::Float>  r,
                            muda::DenseVectorView<gipc::Float>   z,
                            muda::BufferView<gipc::Matrix<3, 3>> diag_inv,
                            double* partials, const int* d2g, int ng)
    {
        using namespace muda;

        ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(diag_inv.size(),
                   [r = r.viewer().name("r"),
                    z = z.viewer().name("z"),
                    diag_inv = diag_inv.viewer().name("diag_inv"),
                    partials, d2g, ng] __device__(int I) mutable
                   {
                       auto&           D  = diag_inv(I);
                       Eigen::Vector3d zv = D * r.segment<3>(I * 3).as_eigen();
                       z.segment<3>(I * 3).as_eigen() = zv;
                       double c = zv.dot(r.segment<3>(I * 3).as_eigen());
                       int    g = d2g[I];
                       if(c != 0.0 && g >= 0 && g < ng)
                           atomicAdd(&partials[(size_t)g * 256 + (I & 255)], c);
                   });
    }
}  // namespace details


void DiagPreconditioner::assemble(GIPCTripletMatrix& global_triplets)
{
    gipc::Timer timer{"precomputing Preconditioner"};
    auto        cols = global_triplets.block_cols();
    m_diag3x3.resize(cols);
    details::diag_assemble(m_diag3x3.view(), global_triplets);
}

void DiagPreconditioner::apply(muda::CDenseVectorView<gipc::Float> r,
                                  muda::DenseVectorView<gipc::Float>  z)
{
    //z.buffer_view().copy_from(r.buffer_view());
    if(m_sd_partials)   // [seg-fused dot] one-shot armed apply
    {
        details::apply_diag_seg_dot(r, z, m_diag3x3, m_sd_partials, m_sd_d2g, m_sd_ng);
        m_sd_partials = nullptr;
        return;
    }
    details::apply_diag(r, z, m_diag3x3);
}
}  // namespace OLD_GIPC
