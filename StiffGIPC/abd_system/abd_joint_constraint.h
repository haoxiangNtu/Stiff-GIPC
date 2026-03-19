#pragma once
#include <gipc/type_define.h>
#include <abd_system/abd_jacobi_matrix.h>
#include <muda/muda_def.h>

namespace gipc
{

/// Maximum number of constraint point pairs per joint.
/// Revolute uses 2 (axis endpoints). Fixed uses 1 (joint center, Method 2).
constexpr int kMaxJointConstraintPoints = 4;

/// GPU-side joint constraint data for a single joint.
/// Stiffness is mass-based (rbs-uipc style): kappa = sr * (m_parent + m_child).
/// No dt^2 factor -- joint energies act as stiff penalty terms in the IP.
///
/// For fixed joints (rbs-uipc Method 2):
///   num_points = 1 (joint center for position constraint)
///   has_direction_constraint = true
///   n_bar / b_bar store normal and bitangent in each body's material frame
///
/// Energy = E_pos + E_n + E_b
///   E_pos = 0.5*K * ||J(cp)*qp - J(cq)*qq||^2
///   E_n   = 0.5*K * ||Ap*np_bar - Aq*nq_bar||^2
///   E_b   = 0.5*K * ||Ap*bp_bar - Aq*bq_bar||^2
struct JointConstraintGPUData
{
    int     parent_body_id;
    int     child_body_id;
    int     num_points;
    Vector3 parent_xbar[kMaxJointConstraintPoints];
    Vector3 child_xbar[kMaxJointConstraintPoints];
    Float   point_weight[kMaxJointConstraintPoints];
    Float   kappa;

    int     has_direction_constraint;  // 1 for fixed joints, 0 otherwise
    Vector3 parent_n_bar;   // normal in parent material frame
    Vector3 child_n_bar;    // normal in child material frame
    Vector3 parent_b_bar;   // bitangent in parent material frame
    Vector3 child_b_bar;    // bitangent in child material frame
};


// ============================================================================
// Direction-only Jacobi helpers
// ============================================================================
// DirJ(d_bar) maps q -> A * d_bar  (rotation part only, no translation).
//   DirJ is a 3x12 matrix: [0_3x3 | diag3(d_bar^T)]
//   where diag3 means the same 1x3 row [d0,d1,d2] repeated on the diagonal.

/// Apply DirJ(d_bar) * q = A * d_bar
MUDA_GENERIC inline Vector3 dir_jacobi_mul(const Vector3& d_bar, const Vector12& q)
{
    return Vector3(q(3)*d_bar(0) + q(4)*d_bar(1) + q(5)*d_bar(2),
                   q(6)*d_bar(0) + q(7)*d_bar(1) + q(8)*d_bar(2),
                   q(9)*d_bar(0) + q(10)*d_bar(1) + q(11)*d_bar(2));
}

/// Apply DirJ(d_bar)^T * v  (returns 12-vector, first 3 components are zero)
MUDA_GENERIC inline Vector12 dir_jacobi_T_mul(const Vector3& d_bar, const Vector3& v)
{
    Vector12 r = Vector12::Zero();
    r(3)  = d_bar(0) * v(0);
    r(4)  = d_bar(1) * v(0);
    r(5)  = d_bar(2) * v(0);
    r(6)  = d_bar(0) * v(1);
    r(7)  = d_bar(1) * v(1);
    r(8)  = d_bar(2) * v(1);
    r(9)  = d_bar(0) * v(2);
    r(10) = d_bar(1) * v(2);
    r(11) = d_bar(2) * v(2);
    return r;
}

/// Compute DirJ(a)^T * (K * I_3) * DirJ(b) -> 12x12 matrix.
/// Result has non-zero entries only in the 9x9 rotation block [3:12, 3:12].
/// The 3x3 diagonal sub-blocks are K * a * b^T.
MUDA_GENERIC inline void dir_JT_K_J(const Vector3& a_bar,
                                     Float          K,
                                     const Vector3& b_bar,
                                     Matrix12x12&   out)
{
    out = Matrix12x12::Zero();
    Matrix3x3 ab = K * (a_bar * b_bar.transpose());
    out.block<3, 3>(3, 3)  = ab;
    out.block<3, 3>(6, 6)  = ab;
    out.block<3, 3>(9, 9)  = ab;
}


// ============================================================================
// Joint constraint energy / gradient / Hessian
// ============================================================================

MUDA_GENERIC inline Float joint_constraint_energy(const JointConstraintGPUData& joint,
                                                   const Vector12&               q_parent,
                                                   const Vector12&               q_child,
                                                   Float                         kappa_fallback)
{
    Float energy = 0.0;
    Float K = (joint.kappa > 0.0) ? joint.kappa : kappa_fallback;

    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);
        Vector3   error = J_parent * q_parent - J_child * q_child;
        energy += 0.5 * K * joint.point_weight[k] * error.squaredNorm();
    }

    if(joint.has_direction_constraint)
    {
        Vector3 n_err = dir_jacobi_mul(joint.parent_n_bar, q_parent)
                      - dir_jacobi_mul(joint.child_n_bar, q_child);
        Vector3 b_err = dir_jacobi_mul(joint.parent_b_bar, q_parent)
                      - dir_jacobi_mul(joint.child_b_bar, q_child);
        energy += 0.5 * K * n_err.squaredNorm();
        energy += 0.5 * K * b_err.squaredNorm();
    }

    return energy;
}

MUDA_GENERIC inline void joint_constraint_gradient(const JointConstraintGPUData& joint,
                                                    const Vector12&               q_parent,
                                                    const Vector12&               q_child,
                                                    Float                         kappa_fallback,
                                                    Vector12& grad_parent_out,
                                                    Vector12& grad_child_out)
{
    grad_parent_out = Vector12::Zero();
    grad_child_out  = Vector12::Zero();
    Float K = (joint.kappa > 0.0) ? joint.kappa : kappa_fallback;

    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);
        Vector3   error = J_parent * q_parent - J_child * q_child;
        Float     w     = joint.point_weight[k];

        grad_parent_out += K * w * (J_parent.T() * error);
        grad_child_out  -= K * w * (J_child.T() * error);
    }

    if(joint.has_direction_constraint)
    {
        Vector3 n_err = dir_jacobi_mul(joint.parent_n_bar, q_parent)
                      - dir_jacobi_mul(joint.child_n_bar, q_child);
        Vector3 b_err = dir_jacobi_mul(joint.parent_b_bar, q_parent)
                      - dir_jacobi_mul(joint.child_b_bar, q_child);

        grad_parent_out += K * dir_jacobi_T_mul(joint.parent_n_bar, n_err);
        grad_child_out  -= K * dir_jacobi_T_mul(joint.child_n_bar, n_err);
        grad_parent_out += K * dir_jacobi_T_mul(joint.parent_b_bar, b_err);
        grad_child_out  -= K * dir_jacobi_T_mul(joint.child_b_bar, b_err);
    }
}

MUDA_GENERIC inline void joint_constraint_hessian(const JointConstraintGPUData& joint,
                                                   Float                         kappa_fallback,
                                                   Matrix12x12& H_pp_out,
                                                   Matrix12x12& H_cc_out,
                                                   Matrix12x12& H_pc_out)
{
    H_pp_out = Matrix12x12::Zero();
    H_cc_out = Matrix12x12::Zero();
    H_pc_out = Matrix12x12::Zero();
    Float K = (joint.kappa > 0.0) ? joint.kappa : kappa_fallback;

    for(int k = 0; k < joint.num_points; k++)
    {
        ABDJacobi J_parent(joint.parent_xbar[k]);
        ABDJacobi J_child(joint.child_xbar[k]);
        Float     w = joint.point_weight[k];

        Matrix3x3 K_I3     =  K * w * Matrix3x3::Identity();
        Matrix3x3 neg_K_I3 = -K * w * Matrix3x3::Identity();

        H_pp_out += ABDJacobi::JT_H_J(J_parent.T(), K_I3, J_parent);
        H_cc_out += ABDJacobi::JT_H_J(J_child.T(), K_I3, J_child);
        H_pc_out += ABDJacobi::JT_H_J(J_parent.T(), neg_K_I3, J_child);
    }

    if(joint.has_direction_constraint)
    {
        Matrix12x12 tmp;

        // Normal: DirJ(np)^T * K*I * DirJ(np), etc.
        dir_JT_K_J(joint.parent_n_bar,  K, joint.parent_n_bar, tmp);  H_pp_out += tmp;
        dir_JT_K_J(joint.child_n_bar,   K, joint.child_n_bar,  tmp);  H_cc_out += tmp;
        dir_JT_K_J(joint.parent_n_bar, -K, joint.child_n_bar,  tmp);  H_pc_out += tmp;

        // Bitangent
        dir_JT_K_J(joint.parent_b_bar,  K, joint.parent_b_bar, tmp);  H_pp_out += tmp;
        dir_JT_K_J(joint.child_b_bar,   K, joint.child_b_bar,  tmp);  H_cc_out += tmp;
        dir_JT_K_J(joint.parent_b_bar, -K, joint.child_b_bar,  tmp);  H_pc_out += tmp;
    }
}

}  // namespace gipc
