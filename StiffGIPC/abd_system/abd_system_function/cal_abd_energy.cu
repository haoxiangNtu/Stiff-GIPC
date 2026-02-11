#include <abd_system/abd_system.h>
#include <muda/cub/device/device_reduce.h>
#include <abd_system/abd_energy.h>
#include <abd_system/abd_joint_constraint.h>
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

    auto kdt2 = parms.joint_stiffness * parms.dt * parms.dt;

    m_joint_energy_per_joint.resize(num_joints);

    ParallelFor()
        .kernel_name(__FUNCTION__)
        .apply(num_joints,
               [energies   = m_joint_energy_per_joint.viewer().name("joint_energies"),
                joints     = m_joint_data.cviewer().name("joint_data"),
                qs         = abd.body_id_to_q.cviewer().name("qs"),
                kdt2] __device__(int j) mutable
               {
                   auto& joint     = joints(j);
                   auto& q_parent  = qs(joint.parent_body_id);
                   auto& q_child   = qs(joint.child_body_id);
                   energies(j) = joint_constraint_energy(joint, q_parent, q_child, kdt2);
               });

    muda::DeviceReduce().Sum(
        m_joint_energy_per_joint.data(), m_joint_energy.data(), num_joints);

    return m_joint_energy;
}

}  // namespace gipc