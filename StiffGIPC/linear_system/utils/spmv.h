#pragma once
#include <gipc/type_define.h>

#include <muda/ext/linear_system/dense_vector_view.h>

namespace gipc
{
class Spmv
{
  public:

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y,
                              // [multi-env S4] skip triplets whose row env is
                              // masked (null = no masking). row block -> group via
                              // s4_dof_to_group[row]; skip if !s4_active[group].
                              const int*                    s4_active = nullptr,
                              const int*                    s4_dof_to_group = nullptr,
                              int                           s4_ng = 0);
};
}  // namespace gipc
