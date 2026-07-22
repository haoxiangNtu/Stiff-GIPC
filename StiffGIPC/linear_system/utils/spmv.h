#pragma once
#include <cstdint>
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
                              // P1 graph tier: launch a fixed capacity while
                              // guarding every load with the device-resident
                              // exact count.
                              int                           launch_triplet_count,
                              const int*                    d_triplet_count,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y,
                              // [multi-env S4] skip triplets whose row env is
                              // masked (null = no masking). row block -> group via
                              // s4_dof_to_group[row]; skip if !s4_active[group].
                              const int*                    s4_active = nullptr,
                              const int*                    s4_dof_to_group = nullptr,
                              int                           s4_ng = 0);
    ~Spmv();

    // [seg-fused dot] when set (by seg_pcg's fast path), the spmv ALSO accumulates the per-env
    // dot x·(aAx) into `partials` (ng*SEG_DOT_PSTRIDE strided slots, block-hashed to spread
    // atomics; combined+rezeroed by the caller's combine kernel). Cleared after use.
    static constexpr int SEG_DOT_PSTRIDE = 256;
    void set_seg_dot_accum(double* partials, int ng)
    { m_seg_dot_partials = partials; m_seg_dot_ng = ng; }

    std::uintptr_t graph_binding_token() const
    {
        return reinterpret_cast<std::uintptr_t>(m_ybin);
    }

    bool graph_workspace_ready(std::size_t scalar_dof_count) const;

  private:
    // [multi-env determinism 4.3] binned accumulator for y (BINNED_K bins per scalar DOF):
    // makes the matvec's row accumulation order-independent ⇒ deterministic. Grown lazily.
    double* m_ybin     = nullptr;
    size_t  m_ybin_cap = 0;
    double* m_seg_dot_partials = nullptr;   // [seg-fused dot] null = off
    int     m_seg_dot_ng       = 0;
};
}  // namespace gipc
