#include <abd_system/abd_system.h>
#include <abd_system/abd_driving_joint.h>   // [force-control] extract_A, RevoluteDrivingGPUData
#include <muda/cub/device/device_reduce.h>
#include <gipc/utils/timer.h>
#include <linear_system/utils/binned_reduce.cuh>
extern __device__ double* g_abd_wrenchbin;   // [4.3] defined in setup_abd_system_gradient_and_hessian.cu
namespace gipc
{
__global__ void _abd_sysbin_combine_k(double* g, const double* bin, int n);  // [4.3] fwd (defined in setup_...cu)
// [force-control] Device-safe inverse-transpose of a 3x3 (cofactor / det), to
// match libuipc's exact A^-T in the joint-torque wrench (handles affine
// shear/scale, not just the rigid A^-T = A limit).  Eigen's built-in .inverse()
// is not device-safe in this build, so this is written out explicitly.
// Falls back to A (rigid limit) if A is near-singular.
MUDA_GENERIC inline Matrix3x3 inv_transpose_3x3(const Matrix3x3& A)
{
    Float a = A(0,0), b = A(0,1), c = A(0,2);
    Float d = A(1,0), e = A(1,1), f = A(1,2);
    Float g = A(2,0), h = A(2,1), i = A(2,2);
    // cofactors
    Float C00 = e*i - f*h, C01 = -(d*i - f*g), C02 = d*h - e*g;
    Float C10 = -(b*i - c*h), C11 = a*i - c*g, C12 = -(a*h - b*g);
    Float C20 = b*f - c*e, C21 = -(a*f - c*d), C22 = a*e - b*d;
    Float det = a*C00 + b*C01 + c*C02;
    Matrix3x3 R;
    if(det > Float(-1e-12) && det < Float(1e-12)) { R = A; return R; }
    Float inv = Float(1) / det;
    // A^-1 = adj/det = (cofactor^T)/det ;  A^-T = (A^-1)^T = cofactor/det
    R(0,0) = C00*inv; R(0,1) = C01*inv; R(0,2) = C02*inv;
    R(1,0) = C10*inv; R(1,1) = C11*inv; R(1,2) = C12*inv;
    R(2,0) = C20*inv; R(2,1) = C21*inv; R(2,2) = C22*inv;
    return R;
}

void ABDSystem::cal_q_tilde(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd            = sim_data.device;
    auto  abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  kappa          = parms.kappa;
    auto  dt             = parms.dt;
    auto  boundary_type  = sim_data.body_id_to_boundary_type();

    // [force-control] (B / libuipc q_tilde path) Convert revolute-joint torques
    // into per-body generalized forces ONCE per step (constant within the Newton
    // solve, so the kinetic Hessian M handles them — no joint-torque Hessian
    // needed). Each torque joint adds F_k = [0; vec(+/-tau/2 [e]_x A_k^-T)] to
    // its parent (-) and child (+), accumulated into body_id_to_abd_joint_wrench.
    abd.body_id_to_abd_joint_wrench.fill(Vector12::Zero());
    // [multi-env determinism 4.3] wrench is assembled by atomic_add (joint/prismatic) → bin it
    // deterministically: alloc+zero the binned buffer, bind the global; combine back before use.
    if(abd_body_count > 0)
    {
        size_t wn = (size_t)abd_body_count * 12 * BINNED_K;
        if(wn > m_abd_wrenchbin_cap)
        { if(m_abd_wrenchbin) cudaFree(m_abd_wrenchbin); cudaMalloc((void**)&m_abd_wrenchbin, wn * sizeof(double)); m_abd_wrenchbin_cap = wn;
          cudaMemcpyToSymbol(g_abd_wrenchbin, &m_abd_wrenchbin, sizeof(double*)); }
        cudaMemsetAsync(m_abd_wrenchbin, 0, wn * sizeof(double), 0);
    }
    if(m_num_revolute_driving > 0)
    {
        ParallelFor()
            .kernel_name("cal_joint_torque_wrench")
            .apply(m_num_revolute_driving,
                   [drvs    = m_revolute_driving_data.cviewer().name("drv"),
                    q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                    wrench  = abd.body_id_to_abd_joint_wrench.viewer().name("wrench")]
                   __device__(int i) mutable
                   {
                       const auto& drv = drvs(i);
                       if(drv.ext_torque == Float(0)) return;
                       int pid = drv.parent_body_id;
                       int cid = drv.child_body_id;

                       Matrix3x3 Ap = extract_A(q_prev(pid));
                       Matrix3x3 Ac = extract_A(q_prev(cid));

                       // Per-body raw world axis from each body's own
                       // perpendicular frame (libuipc Axis Direction): p_bar,
                       // pN_bar (parent) and q_bar, qN_bar (child) are unit and
                       // perpendicular to the axis, so axis_material = p_bar x
                       // pN_bar; e_world = A * axis_material.
                       Vector3 ei = (Ap * drv.p_bar.cross(drv.pN_bar));
                       Vector3 ej = (Ac * drv.q_bar.cross(drv.qN_bar));
                       Float ni = ei.norm(), nj = ej.norm();
                       if(ni < Float(1e-12) || nj < Float(1e-12)) return;
                       ei /= ni; ej /= nj;
                       // Symmetrize exactly as public libuipc
                       // (affine_body_revolute_joint.cu): e_j = 1/2 (e_j + e_i),
                       // e_i = -e_j. The two raw axes point the same way (the
                       // shared joint axis is invariant under rotation about
                       // itself), so the sum stays ~unit; e_i = -e_j gives
                       // bit-exact Newton's 3rd law. (No re-normalize / no
                       // dot-align — matching libuipc byte-for-byte.)
                       ej = Float(0.5) * (ej + ei);
                       ei = -ej;

                       Matrix3x3 exi = Matrix3x3::Zero();
                       exi(0,1) = -ei.z(); exi(0,2) =  ei.y();
                       exi(1,0) =  ei.z(); exi(1,2) = -ei.x();
                       exi(2,0) = -ei.y(); exi(2,1) =  ei.x();
                       Matrix3x3 exj = Matrix3x3::Zero();
                       exj(0,1) = -ej.z(); exj(0,2) =  ej.y();
                       exj(1,0) =  ej.z(); exj(1,2) = -ej.x();
                       exj(2,0) = -ej.y(); exj(2,1) =  ej.x();

                       // F_k = tau/2 [e_k]_x A_k^-T (libuipc, child-positive).
                       // e_i = -e_j carries the opposite-sign reaction to parent.
                       // Full A^-T via device-safe cofactor inverse (shear/scale).
                       Float half = Float(0.5) * drv.ext_torque;
                       Matrix3x3 FpA = half * exi * inv_transpose_3x3(Ap);
                       Matrix3x3 FcA = half * exj * inv_transpose_3x3(Ac);

                       Vector12 Fp = Vector12::Zero();
                       Vector12 Fc = Vector12::Zero();
                       Fp.segment<3>(3) = FpA.row(0).transpose(); Fp.segment<3>(6) = FpA.row(1).transpose(); Fp.segment<3>(9) = FpA.row(2).transpose();
                       Fc.segment<3>(3) = FcA.row(0).transpose(); Fc.segment<3>(6) = FcA.row(1).transpose(); Fc.segment<3>(9) = FcA.row(2).transpose();

                       bin_add12(g_abd_wrenchbin, pid, Fp);
                       bin_add12(g_abd_wrenchbin, cid, Fc);
                   });
    }

    // [force-control] (B / libuipc q_tilde path) Prismatic-joint external force.
    // libuipc AffineBodyPrismaticJointExternalForce: a scalar force f along the
    // joint axis becomes a LINEAR force +/- f*t on each body's translation DOF
    // (no skew, no A^-T — translational, not rotational). t is each body's own
    // world tangent (A * t_bar), symmetrized like the revolute axis.
    if(m_num_prismatic_driving > 0)
    {
        ParallelFor()
            .kernel_name("cal_joint_prismatic_force_wrench")
            .apply(m_num_prismatic_driving,
                   [drvs    = m_prismatic_driving_data.cviewer().name("pdrv"),
                    q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                    wrench  = abd.body_id_to_abd_joint_wrench.viewer().name("wrench")]
                   __device__(int i) mutable
                   {
                       const auto& drv = drvs(i);
                       if(drv.ext_force == Float(0)) return;
                       int pid = drv.parent_body_id;
                       int cid = drv.child_body_id;

                       Matrix3x3 Ap = extract_A(q_prev(pid));
                       Matrix3x3 Ac = extract_A(q_prev(cid));

                       // Per-body world tangent t = A * t_bar (libuipc vec_x).
                       Vector3 ti = Ap * drv.tp_bar;
                       Vector3 tj = Ac * drv.tq_bar;
                       // Symmetrize exactly as public libuipc
                       // (affine_body_prismatic_joint.cu): t_j = 1/2 (t_i+t_j),
                       // t_i = -t_j. (No re-normalize — byte-for-byte.)
                       tj = Float(0.5) * (ti + tj);
                       ti = -tj;

                       Float f = drv.ext_force;
                       Vector12 Fp = Vector12::Zero();
                       Vector12 Fc = Vector12::Zero();
                       Fp.segment<3>(0) = f * ti;   // parent gets -f*t
                       Fc.segment<3>(0) = f * tj;   // child  gets +f*t

                       bin_add12(g_abd_wrenchbin, pid, Fp);
                       bin_add12(g_abd_wrenchbin, cid, Fc);
                   });
    }

    // [multi-env determinism 4.3] combine the binned wrench back into body_id_to_abd_joint_wrench
    // (deterministic) before it is consumed into q_tilde below.
    if(abd_body_count > 0)
    {
        int n = abd_body_count * 12, bs = 256, gs = (n + bs - 1) / bs;
        if(gs > 0)  // [zero-ABD guard] gridDim=0 launch = cudaErrorInvalidConfiguration
        _abd_sysbin_combine_k<<<gs, bs>>>(
            (double*)abd.body_id_to_abd_joint_wrench.data(), m_abd_wrenchbin, n);
    }

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_body_count,
               [boundary_type = boundary_type.cviewer().name("btype"),
                q_prevs  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                q_vs     = abd.body_id_to_q_v.viewer().name("q_velocities"),
                q_tildes = abd.body_id_to_q_tilde.viewer().name("q_tilde"),
                affine_gravity = abd.body_id_to_abd_gravity.cviewer().name("affine_gravity"),
                ext_forces = abd.body_id_to_abd_ext_force.cviewer().name("ext_force"),
                joint_wrench = abd.body_id_to_abd_joint_wrench.cviewer().name("joint_wrench"),
                mass_invs  = abd.body_id_to_abd_mass_inv.cviewer().name("mass_inv"),
                dt = dt,
                vel_damp = parms.velocity_damping] __device__(int i) mutable
               {
                   auto& q_prev = q_prevs(i);
                   auto& q_v    = q_vs(i);
                   auto& g      = affine_gravity(i);
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       q_tildes(i) = q_prev;
                   }
                   else
                   {
                       if(vel_damp > 0.0)
                           q_v *= (1.0 - vel_damp);
                       // [force-control] external generalized force (user per-body
                       // ext_force + revolute-joint torque wrench) -> acceleration
                       // a_ext = M^{-1} F, fed into q_tilde exactly like gravity
                       // (libuipc external-force/torque q_tilde path; constant within
                       // the solve, kinetic term E=1/2 (q-q_tilde)^T M (q-q_tilde) handles it).
                       Vector12 a_ext = mass_invs(i) * (ext_forces(i) + joint_wrench(i));
                       q_tildes(i) = q_prev + q_v * dt + (g + a_ext) * (dt * dt);
                   }
               });

    //m_local_tolerance.resize(abd.body_id_to_q_tilde.size());

    //ParallelFor()
    //    .file_line(__FILE__, __LINE__)
    //    .apply(abd_body_count,
    //           [local_tolerance = m_local_tolerance.viewer().name("local_tolerance"),
    //            q_tildes = abd.body_id_to_q_tilde.cviewer().name("q_tilde"),
    //            qs = abd.body_id_to_q.cviewer().name("q")] __device__(int i) mutable
    //           {
    //               auto& q_tilde      = q_tildes(i);
    //               auto& q            = qs(i);
    //               local_tolerance(i) = (q_tilde - q).norm();
    //           });

    //muda::DeviceReduce().Max(m_local_tolerance.data(),
    //                         m_local_tolerance_max.data(),
    //                         m_local_tolerance.size());

    //m_suggest_max_tolerance = m_local_tolerance_max;
}
}  // namespace gipc