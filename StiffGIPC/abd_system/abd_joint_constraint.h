#pragma once
#include <gipc/type_define.h>
#include <abd_system/abd_jacobi_matrix.h>
#include <muda/muda_def.h>

namespace gipc
{

/// Maximum number of constraint point pairs per joint.
/// Revolute uses 2 (two axis endpoints), Fixed uses 4 (non-coplanar).
constexpr int kMaxJointConstraintPoints = 4;

/// GPU-side joint constraint data for a single joint.
/// Stored in SoA layout via separate DeviceBuffers in ABDSystem.
struct JointConstraintGPUData
{
    int     parent_body_id;
    int     child_body_id;
    int     num_points;  // 2 for revolute, 4 for fixed
    Vector3 parent_xbar[kMaxJointConstraintPoints];  // material coords in parent body
    Vector3 child_xbar[kMaxJointConstraintPoints];   // material coords in child body
};


/// Compute the energy contribution of a single joint constraint.
///
/// E = stiffness * dt^2 * 0.5 * sum_k ||J(parent_xbar_k) * q_parent - J(child_xbar_k) * q_child||^2
///
MUDA_GENERIC inline Float joint_constraint_energy(const JointConstraintGPUData& joint,
                                                   const Vector12&               q_parent,
                                                   const Vector12&               q_child,
                                                   Float                         kdt2)
{
    Float energy = 0.0;
    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);
        Vector3   error = J_parent * q_parent - J_child * q_child;
        energy += 0.5 * kdt2 * error.squaredNorm();
    }
    return energy;
}

/// Compute gradient for both parent and child bodies.
///
/// grad_parent = + K*dt^2 * sum_k J_parent_k^T * (J_parent_k * q_parent - J_child_k * q_child)
/// grad_child  = - K*dt^2 * sum_k J_child_k^T  * (J_parent_k * q_parent - J_child_k * q_child)
///
MUDA_GENERIC inline void joint_constraint_gradient(const JointConstraintGPUData& joint,
                                                    const Vector12&               q_parent,
                                                    const Vector12&               q_child,
                                                    Float                         kdt2,
                                                    Vector12& grad_parent_out,
                                                    Vector12& grad_child_out)
{
    grad_parent_out = Vector12::Zero();
    grad_child_out  = Vector12::Zero();

    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);
        Vector3   error = J_parent * q_parent - J_child * q_child;

        grad_parent_out += kdt2 * (J_parent.T() * error);
        grad_child_out  -= kdt2 * (J_child.T() * error);
    }
}

/// Compute Hessian blocks for the joint constraint.
///
/// H_pp = + K*dt^2 * sum_k J_parent_k^T * I_3 * J_parent_k   (12x12, SPD)
/// H_cc = + K*dt^2 * sum_k J_child_k^T  * I_3 * J_child_k    (12x12, SPD)
/// H_pc = - K*dt^2 * sum_k J_parent_k^T * I_3 * J_child_k    (12x12, cross-body)
///
/// Note: The total 24x24 Hessian [H_pp H_pc; H_cp H_cc] is SPD by construction
/// (it equals K*dt^2 * [J_p; -J_c]^T * [J_p; -J_c]), so no need for make_pd().
///
MUDA_GENERIC inline void joint_constraint_hessian(const JointConstraintGPUData& joint,
                                                   Float                         kdt2,
                                                   Matrix12x12& H_pp_out,
                                                   Matrix12x12& H_cc_out,
                                                   Matrix12x12& H_pc_out)
{
    H_pp_out = Matrix12x12::Zero();
    H_cc_out = Matrix12x12::Zero();
    H_pc_out = Matrix12x12::Zero();

    Matrix3x3 K_I3 = kdt2 * Matrix3x3::Identity();
    Matrix3x3 neg_K_I3 = -kdt2 * Matrix3x3::Identity();

    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);

        H_pp_out += ABDJacobi::JT_H_J(J_parent.T(), K_I3, J_parent);
        H_cc_out += ABDJacobi::JT_H_J(J_child.T(), K_I3, J_child);
        H_pc_out += ABDJacobi::JT_H_J(J_parent.T(), neg_K_I3, J_child);
    }
}

}  // namespace gipc
