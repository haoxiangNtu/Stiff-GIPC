#include <abd_system/abd_system.h>
#include <muda/cub/device/device_reduce.h>
#include <abd_system/abd_energy.h>
#include <abd_system/abd_joint_constraint.h>
#include <linear_system/utils/binned_reduce.cuh>  // [decouple] order-free per-env energy
namespace gipc
{
Float ABDSystem::cal_abd_kinetic_energy(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd       = sim_data.device;
    auto  abd_count = sim_data.abd_fem_count_info().abd_body_num;
    m_kinetic_energy_per_affine_body.resize(abd_count);
    auto& abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  boundry_type   = sim_data.body_id_to_boundary_type();
    if(!abd_count)
        return 0;
    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_count,
               [kinetic_energies = m_kinetic_energy_per_affine_body.viewer().name("kinetic_energies"),
                qs       = abd.body_id_to_q.viewer().name("qs"),
                q_prev   = abd.body_id_to_q_prev.viewer().name("q_prev"),
                q_tildes = abd.body_id_to_q_tilde.viewer().name("q_tildes"),
                Ms       = abd.body_id_to_abd_mass.viewer().name("Ms"),
                boundary_type    = boundry_type.cviewer().name("btype"),
                body_motor_data  = sim_data.body_motor_params(),
                dt             = parms.dt,
                motor_speed    = parms.motor_speed,
                motor_strength = parms.motor_strength] __device__(int i) mutable
               {
                   auto& K       = kinetic_energies(i);
                   auto& q       = qs(i);
                   auto& q_tilde = q_tildes(i);
                   auto& M       = Ms(i);


                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       K = 0.0;
                   }
                   else
                   {
                       if(boundary_type(i) == BodyBoundaryType::Free)
                       {
                           Vector12 dq = q - q_tilde;
                           K           = 0.5 * dq.dot(M * dq);
                       }

                       if(boundary_type(i) == BodyBoundaryType::Animated)
                       {
                           // Kinetic energy
                           Vector12 dq_kin = q - q_tilde;
                           K = 0.5 * dq_kin.dot(M * dq_kin);

                           // Animated penalty energy (absolute target)
                           // body_motor_params = [target_x, target_y, target_z, strength, 0]
                           // Constrain translation toward target, affine A toward identity.
                           Vector3 aim_pos = q_tilde.segment<3>(0);
                           double  anim_strength = 1e6;
                           if(body_motor_data)
                           {
                               aim_pos(0) = body_motor_data[i * 5 + 0];
                               aim_pos(1) = body_motor_data[i * 5 + 1];
                               aim_pos(2) = body_motor_data[i * 5 + 2];
                               double st = body_motor_data[i * 5 + 3];
                               if(st > 0.0) anim_strength = st;
                           }
                           // q_aim: target position + identity affine matrix
                           Vector12 q_aim;
                           q_aim.segment<3>(0) = aim_pos;
                           q_aim(3) = 1.0; q_aim(4) = 0.0; q_aim(5) = 0.0;
                           q_aim(6) = 0.0; q_aim(7) = 1.0; q_aim(8) = 0.0;
                           q_aim(9) = 0.0; q_aim(10) = 0.0; q_aim(11) = 1.0;
                           Vector12 dq_anim = q - q_aim;
                           // Penalize all 12 DOFs: translation + affine
                           Matrix12x12 PowMass = anim_strength * Matrix12x12::Identity();
                           K += 0.5 * dq_anim.dot(PowMass * dq_anim);
                       }

                       if(boundary_type(i) == BodyBoundaryType::Motor)
                       {
                           {
                               Vector12 dq = q - q_tilde;
                               K           = 0.5 * dq.dot(M * dq);
                           }

                           // Read per-body motor params: [axis_x, axis_y, axis_z, speed, strength]
                           Vector3 rot_axis = Vector3::UnitX();
                           double  body_speed    = motor_speed;
                           double  body_strength = motor_strength;
                           if(body_motor_data)
                           {
                               double ax = body_motor_data[i * 5 + 0];
                               double ay = body_motor_data[i * 5 + 1];
                               double az = body_motor_data[i * 5 + 2];
                               double sp = body_motor_data[i * 5 + 3];
                               double st = body_motor_data[i * 5 + 4];
                               double len = sqrt(ax * ax + ay * ay + az * az);
                               if(len > 1e-10)
                                   rot_axis = Vector3{ax, ay, az} / len;
                               if(sp > 0.0)
                                   body_speed = sp;
                               if(st > 0.0)
                                   body_strength = st;
                           }

                           Vector3 bar_x0 = Vector3::Zero();
                           Vector3 bar_x1 = Vector3::UnitX();
                           Vector3 bar_x2 = Vector3::UnitY();
                           Vector3 bar_x3 = Vector3::UnitZ();

                           auto mat0 = ABDJacobi{bar_x0}.to_mat();
                           auto mat1 = ABDJacobi{bar_x1}.to_mat();
                           auto mat2 = ABDJacobi{bar_x2}.to_mat();
                           auto mat3 = ABDJacobi{bar_x3}.to_mat();

                           Matrix12x12 J;
                           J.block<3, 12>(0, 0) = mat0;
                           J.block<3, 12>(3, 0) = mat1;
                           J.block<3, 12>(6, 0) = mat2;
                           J.block<3, 12>(9, 0) = mat3;

                           Matrix12x12 inv_J = eigen::inverse(J);

                           auto theta = body_speed * dt;
                           // rotate around per-body axis
                           auto R = Eigen::AngleAxisd(theta, rot_axis);

                           Vector3 x1_P = R * bar_x1;
                           Vector3 x2_P = R * bar_x2;
                           Vector3 x3_P = R * bar_x3;

                           auto mat0_delta = ABDJacobi{Vector3::Zero()}.to_mat();
                           auto mat1_delta = ABDJacobi{x1_P - bar_x1}.to_mat();
                           auto mat2_delta = ABDJacobi{x2_P - bar_x2}.to_mat();
                           auto mat3_delta = ABDJacobi{x3_P - bar_x3}.to_mat();

                           Matrix12x12 J_delta;
                           J_delta.block<3, 12>(0, 0) = mat0_delta;
                           J_delta.block<3, 12>(3, 0) = mat1_delta;
                           J_delta.block<3, 12>(6, 0) = mat2_delta;
                           J_delta.block<3, 12>(9, 0) = mat3_delta;

                           //Vector12 q_p = inv_J * J_delta * q_prev(i) + q_prev(i);
                           Vector12 q_p = inv_J * J_delta * q_tilde + q_tilde;
                           q_p.segment<3>(3).normalize();
                           q_p.segment<3>(6).normalize();
                           q_p.segment<3>(9).normalize();
                           Vector12 dq      = q - q_p;
                           dq.segment<3>(0) = Vector3::Zero();

                           Matrix12x12 PowMass = Matrix12x12::Zero();
                           PowMass.block<9, 9>(3, 3) =
                           body_strength * Ms(i).to_mat().block<9, 9>(3, 3);

                           K += 0.5 * dq.dot(PowMass * dq);
                       }
                   }
               });
    muda::DeviceReduce().Sum(m_kinetic_energy_per_affine_body.data(),
                             m_kinetic_energy.data(),
                             abd_count);
    return m_kinetic_energy;
}
Float ABDSystem::cal_abd_shape_energy(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd       = sim_data.device;
    auto  abd_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  kappa     = parms.kappa;
    auto  dt        = parms.dt;
    if(!abd_count)
        return 0;
    m_shape_energy_per_affine_body.resize(abd_count);

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(abd_count,
               [shape_energies = m_shape_energy_per_affine_body.viewer().name("abd_shape_energy"),
                qs      = abd.body_id_to_q.viewer().name("q"),
                kappa   = kappa,
                volumes = abd.body_id_to_volume.cviewer().name("volumes"),
                dt      = dt] __device__(int i) mutable
               {
                   auto& V      = shape_energies(i);
                   auto& q      = qs(i);
                   auto& volume = volumes(i);

                   V = kappa * volume * dt * dt * shape_energy(q);
               });

    muda::DeviceReduce().Sum(
        m_shape_energy_per_affine_body.data(), m_shape_energy.data(), abd_count);

    return m_shape_energy;
}

Float ABDSystem::cal_abd_joint_energy(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd       = sim_data.device;
    auto  num_joints = m_num_joints;

    if(!num_joints)
        return 0;

    auto kappa_fallback = parms.joint_strength_ratio;  // fallback; per-joint kappa takes priority

    m_joint_energy_per_joint.resize(num_joints);

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(num_joints,
               [energies       = m_joint_energy_per_joint.viewer().name("joint_energies"),
                joints         = m_joint_data.cviewer().name("joint_data"),
                qs             = abd.body_id_to_q.cviewer().name("qs"),
                kappa_fallback] __device__(int j) mutable
               {
                   auto& joint     = joints(j);
                   auto& q_parent  = qs(joint.parent_body_id);
                   auto& q_child   = qs(joint.child_body_id);
                   energies(j) = joint_constraint_energy(joint, q_parent, q_child, kappa_fallback);
               });

    muda::DeviceReduce().Sum(
        m_joint_energy_per_joint.data(), m_joint_energy.data(), num_joints);

    return m_joint_energy;
}

// [multi-env S3] per-env ABD energy: segment-sum each per-element energy array by
// env. body-keyed terms (kinetic, shape) use body i; constraint terms use
// parent_body_id. atomicAdd into env_out (raw device double*, size ng).
double ABDSystem::cal_abd_energy_perenv(ABDSimData& sim_data, const int* body_to_group,
                                        int ng, double* env_out)
{
    using namespace muda;
    double total = 0.0;
    total += cal_abd_kinetic_energy(sim_data);
    total += cal_abd_shape_energy(sim_data);
    total += cal_abd_joint_energy(sim_data);
    total += cal_abd_revolute_driving_energy(sim_data);
    total += cal_abd_prismatic_energy(sim_data);
    total += cal_abd_prismatic_driving_energy(sim_data);

    auto abd_count = sim_data.abd_fem_count_info().abd_body_num;
    if(!env_out || !body_to_group) return total;

    // [decouple] order-free per-env energy accumulation. The previous atomicAdd(&env_out[g], E)
    // sums doubles in thread-scheduling order, which depends on the TOTAL work (batch) → env_0's
    // per-env ABD energy gets a ~1e-16 batch-dependent jitter → near the per-env line-search descent
    // threshold this FLIPS env_0's backtrack-halve decision → env_0's ABD step alpha differs → its
    // gripper/arm pose drifts across batches (confirmed root: ABD dq identical, q drifts after the
    // last step). Binned deposit (Demmel-Nguyen, exact per-bin) is order-independent → bit-identical.
    static double* s_ebin = nullptr; static int s_ebin_ng = 0;
    if(s_ebin_ng < ng)
    {
        if(s_ebin) cudaFree(s_ebin);
        cudaMalloc((void**)&s_ebin, (size_t)ng * BINNED_K * sizeof(double));
        s_ebin_ng = ng;
    }
    cudaMemset(s_ebin, 0, (size_t)ng * BINNED_K * sizeof(double));
    double* ebin = s_ebin;

    // per-body terms: kinetic, shape (element i -> body i)
    if(abd_count > 0)
    {
        ParallelFor(256).apply(abd_count,
            [E = m_kinetic_energy_per_affine_body.cviewer().name("K"),
             body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[i]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
        ParallelFor(256).apply(abd_count,
            [E = m_shape_energy_per_affine_body.cviewer().name("V"),
             body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[i]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
    }
    // constraint terms: keyed by parent_body_id (parent/child same env after P1)
    if(m_num_joints > 0)
        ParallelFor(256).apply(m_num_joints,
            [E = m_joint_energy_per_joint.cviewer().name("J"),
             d = m_joint_data.cviewer().name("jd"), body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[d(i).parent_body_id]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
    if(m_num_revolute_driving > 0)
        ParallelFor(256).apply(m_num_revolute_driving,
            [E = m_revolute_driving_energy_per.cviewer().name("R"),
             d = m_revolute_driving_data.cviewer().name("rd"), body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[d(i).parent_body_id]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
    if(m_num_prismatic > 0)
        ParallelFor(256).apply(m_num_prismatic,
            [E = m_prismatic_energy_per.cviewer().name("P"),
             d = m_prismatic_data.cviewer().name("pd"), body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[d(i).parent_body_id]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
    if(m_num_prismatic_driving > 0)
        ParallelFor(256).apply(m_num_prismatic_driving,
            [E = m_prismatic_driving_energy_per.cviewer().name("PD"),
             d = m_prismatic_driving_data.cviewer().name("pdd"), body_to_group, ng, ebin] __device__(int i) mutable
            { int g = body_to_group[d(i).parent_body_id]; if(g >= 0 && g < ng) binned_deposit(ebin + (size_t)g * BINNED_K, E(i)); });
    // combine bins -> env_out (fixed order, exact)
    ParallelFor(256).apply(ng,
        [ebin, env_out] __device__(int g) mutable
        { env_out[g] += binned_combine(ebin + (size_t)g * BINNED_K); });
    return total;
}

}  // namespace gipc