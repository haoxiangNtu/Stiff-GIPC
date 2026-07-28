#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
namespace gipc
{
void ABDSystem::copy_q_to_q_temp(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd = sim_data.device;
    // DeviceBuffer's copy-assignment finishes with LaunchCore::wait(), which
    // injects a host synchronization even when both buffers are already the
    // right size (and even for an empty FEM-only scene).  Keep capacity growth
    // at the boundary, then record only the stream-ordered D2D copy.
    if(abd.body_id_to_q_temp.size() != abd.body_id_to_q.size())
        abd.body_id_to_q_temp.resize(abd.body_id_to_q.size());
    if(abd.body_id_to_q.size())
        abd.body_id_to_q_temp.view().copy_from(
            abd.body_id_to_q.view());
}

void ABDSystem::step_forward(ABDSimData&                sim_data,
                             muda::BufferView<double3>  vertexes,
                             double                     alpha,
                             const double*              per_body_alpha,
                             const double*              alpha_dev)
{
    using namespace muda;
    auto& abd                       = sim_data.device;
    auto  abd_body_count            = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count        = sim_data.abd_fem_count_info().abd_point_num;
    auto  unique_poit_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  boundary_type       = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(abd_body_count,
               [boundary_type = boundary_type.cviewer().name("is_fixed"),
                q_temps       = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs            = abd.body_id_to_q.viewer().name("qs"),
                dqs           = abd.body_id_to_dq.cviewer().name("dqs"),
                per_body_alpha,
                alpha_dev,
                alpha] __device__(int i) mutable
               {
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                       return;
                   // [C-1 ls-graph] device scalar trial alpha when armed.
                   if(alpha_dev)
                       alpha = *alpha_dev;
                   // [multi-env S2] per-body alpha (env g's bodies step uniformly
                   // with env g's FEM verts); fall back to scalar alpha if <0.
                   double a = alpha;
                   if(per_body_alpha && per_body_alpha[i] >= 0.0) a = per_body_alpha[i];
                   // [audit lens-D fix] frozen body (a==0) keeps its last
                   // accepted q: with NaN/Inf dq, q_temp - dq*0 = NaN — the
                   // freeze failed exactly when needed. Bit-neutral for a != 0.
                   if(a == 0.0)
                       qs(i) = q_temps(i);
                   else
                       qs(i) = q_temps(i) - a * dqs(i);
               });

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(unique_point_count,
               [vertices     = vertexes.viewer().name("vertices"),
                unique_point_id_to_body_id =
                    unique_poit_id_to_body_id.cviewer().name("unique_point_id_to_body_id"),
                Js       = abd.unique_point_id_to_J.cviewer().name("Js"),
                q_temps  = abd.body_id_to_q_temp.cviewer().name("q_temps"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                dqs      = abd.body_id_to_dq.cviewer().name("dqs"),
                alpha] __device__(int i) mutable
               {
                   auto        body_id         = unique_point_id_to_body_id(i);
                   const auto& q               = qs(body_id);
                   const auto& J               = Js(i);
                   auto&       vert            = vertices(i);
                   auto        v               = J * q;
                   vert.x                      = v.x();
                   vert.y                      = v.y();
                   vert.z                      = v.z();
               });
}

}  // namespace gipc
