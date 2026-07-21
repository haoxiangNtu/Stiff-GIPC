#pragma once
#include <gipc/type_define.h>
#include <abd_system/abd_jacobi_matrix.h>
#include <muda/muda_def.h>

namespace gipc
{

// ============================================================================
// Revolute Driving Joint  (sin-based angle control)
// ============================================================================
//
// Energy:  E = 0.5 * K * sin(theta - theta_target)^2
//            = 0.5 * K * (sin(theta)*cos(theta_target) - cos(theta)*sin(theta_target))^2
//
// where theta is the current angle between the two bodies around the joint axis.
//
// We define 4 direction vectors in spatial space:
//   p  = J(p_bar)  * q1   -- direction vector on body1 (perpendicular to axis)
//   pN = J(pN_bar) * q1   -- direction vector on body1 rotated 90 deg from p around axis
//   q  = J(q_bar)  * q2   -- direction vector on body2 (perpendicular to axis)
//   qN = J(qN_bar) * q2   -- direction vector on body2 rotated 90 deg from q around axis
//
// NOTE: p_bar, pN_bar, q_bar, qN_bar are *direction* vectors (not points).
//       For ABD, J(x_bar)*q = p + A*x_bar. When x_bar is a direction (offset from
//       body center), J gives the world-space direction = A*x_bar.
//       But since these are pure directions, we need J applied without translation.
//       We store them as if they were offset positions at distance 1 from body center.
//       Then the spatial direction is: d = J(x_bar)*q - p_body = A*x_bar.
//       So we actually store direction_xbar in material frame and compute:
//         p_world = A_parent * p_bar  (no translation needed for directions)
//
// Since ABDJacobi * q = p + A * x_bar, if we want just A * x_bar (pure direction),
// we need to subtract the translation p. But we can compute this more efficiently.
//
// For symmetric sin/cos:
//   cos(theta) = (p·q + pN·qN) / 2
//   sin(theta) = (q·pN - qN·p) / 2
//
// Let s = sin(theta)*cos(theta_tgt) - cos(theta)*sin(theta_tgt)
// Then E = 0.5 * K * s^2
//
// Gradient: dE/d[q1,q2] = K * s * ds/d[q1,q2]
// Hessian:  H = K * (ds/d[q1,q2])^T * (ds/d[q1,q2])   (Gauss-Newton, guaranteed SPD)
//             + K * s * d2s/d[q1,q2]^2                  (need make_pd)
//
// For optimization efficiency, we use the Gauss-Newton approximation:
//   H ≈ K * (ds/d[q1,q2])^T * (ds/d[q1,q2])   (SPD, no make_pd needed)
//
// ============================================================================

struct RevoluteDrivingGPUData
{
    int parent_body_id;   // body1  (q1)
    int child_body_id;    // body2  (q2)

    // Material-space direction vectors (unit length, ⊥ to joint axis)
    // These are directions: world_dir = A * dir_xbar
    Vector3 p_bar;    // body1 perpendicular direction
    Vector3 pN_bar;   // body1 perpendicular direction (90 deg rotated from p around axis)
    Vector3 q_bar;    // body2 perpendicular direction  (same as p_bar at rest)
    Vector3 qN_bar;   // body2 perpendicular direction  (same as pN_bar at rest)

    Float   stiffness;              // K = sr * ctrl_sr * (m_parent + m_child) * dt²
    Float   target_angle;           // theta_tgt in radians (relative to loaded config)
    Float   initial_angle_offset;   // offset from FK-loaded pose (absolute URDF target = relative target + offset)

    // [force-control] external joint torque (N·m) about the joint axis.
    // Adds the generalized force -tau * dtheta/dq to the gradient (no Hessian,
    // treated as a constant force like libuipc's external torque). Independent
    // of the PD term: stiffness>0 = position control, ext_torque!=0 = torque
    // control; set stiffness=0 for pure torque control. Both can be 0.
    Float   ext_torque = Float(0);

    // [joint limit] One-sided position-limit PENALTY (NOT a barrier), expressed in the
    // RELATIVE angle frame (already offset by initial_angle_offset, like target_angle).
    // Active for ANY control mode because it is a function of the CURRENT angle theta,
    // not the drive target — mirrors MuJoCo's mode-independent joint limit. Implemented
    // by reusing the driving spring with target = the violated bound and stiffness =
    // limit_stiffness, so it inherits the same gradient + SPD Gauss-Newton Hessian.
    Float   lower_limit     = -Float(1e30);  // relative-frame lower bound (rad); huge = none
    Float   upper_limit     =  Float(1e30);  // relative-frame upper bound (rad)
    Float   limit_stiffness =  Float(0);     // K_lim = limit_strength_ratio*(m_p+m_c); 0 = off
    // [limit lagged active-set] Frozen ONCE PER FRAME by
    // update_revolute_driving_targets from theta_prev: -1 = lower bound active,
    // +1 = upper, 0 = free. The energy/gradient/Hessian use this frozen flag
    // instead of re-testing theta every Newton iteration. Rationale: the
    // per-iteration test made the limit term appear/disappear between iterates;
    // on the inactive side the Hessian carries NO limit curvature, so the Newton
    // direction is a huge free-fall step, line search cuts alpha to ~1e-7, the
    // next iterate lands active, gets pushed back out, and the active set
    // chatters until newton_iter_cap (measured: limited hinge hit the 1000 cap
    // from the first bound contact, single frames ~19 s). With the set frozen
    // the in-frame energy is a fixed smooth quadratic -> Newton converges in a
    // few iterations. Release rule = multiplier sign (see the update kernel).
    int     limit_active    = 0;
};


/// Extract the 3x3 rotation matrix A from the ABD state vector q = [p, a1, a2, a3]
MUDA_GENERIC inline Matrix3x3 extract_A(const Vector12& q)
{
    Matrix3x3 A;
    A.row(0) = q.segment<3>(3).transpose();
    A.row(1) = q.segment<3>(6).transpose();
    A.row(2) = q.segment<3>(9).transpose();
    return A;
}


/// Compute the revolute driving joint energy.
///
/// E = 0.5 * K * ( sin(delta)^2 + beta * (1 - cos(delta))^2 )
///
/// where delta = theta - theta_tgt.
/// The sin^2 term drives toward delta=0. The small (1-cos)^2 term
/// breaks the pi-periodic degeneracy of sin^2 (which has minima at
/// both delta=0 and delta=pi).  beta=0.02 penalises the pi branch
/// without perturbing normal tracking.
///
MUDA_GENERIC inline Float revolute_driving_energy(
    const RevoluteDrivingGPUData& drv,
    const Vector12& q1,   // parent body state
    const Vector12& q2)   // child body state
{
    constexpr Float beta = Float(0.02);

    Matrix3x3 A1 = extract_A(q1);
    Matrix3x3 A2 = extract_A(q2);

    Vector3 p  = A1 * drv.p_bar;
    Vector3 pN = A1 * drv.pN_bar;
    Vector3 q  = A2 * drv.q_bar;
    Vector3 qN = A2 * drv.qN_bar;

    Float cos_theta = 0.5 * (p.dot(q) + pN.dot(qN));
    Float sin_theta = 0.5 * (q.dot(pN) - qN.dot(p));

    Float cos_tgt = cos(drv.target_angle);
    Float sin_tgt = sin(drv.target_angle);

    Float s     = sin_theta * cos_tgt - cos_theta * sin_tgt;   // sin(delta)
    Float c     = cos_theta * cos_tgt + sin_theta * sin_tgt;   // cos(delta)
    Float c_err = Float(1) - c;                                // 1 - cos(delta)

    return Float(0.5) * drv.stiffness * (s * s + beta * c_err * c_err);
}


/// Compute gradient of revolute driving energy w.r.t. q1 and q2.
///
/// E = 0.5 * K * ( s^2 + beta * c_err^2 )
/// where  s     = sin(delta),   c_err = 1 - cos(delta)
///
/// dE/dq = K * ( s * ds/dq  -  beta * c_err * dc/dq )
///
/// dc/d{dir} is related to ds/d{dir} by a 90-degree rotation:
///   dc/dp  =  ds/dpN,   dc/dpN = -ds/dp
///   dc/dq  = -ds/dqN,   dc/dqN =  ds/dq
///
MUDA_GENERIC inline void revolute_driving_gradient(
    const RevoluteDrivingGPUData& drv,
    const Vector12& q1,
    const Vector12& q2,
    Vector12& grad_q1_out,
    Vector12& grad_q2_out)
{
    constexpr Float beta = Float(0.02);

    Matrix3x3 A1 = extract_A(q1);
    Matrix3x3 A2 = extract_A(q2);

    Vector3 p  = A1 * drv.p_bar;
    Vector3 pN = A1 * drv.pN_bar;
    Vector3 q  = A2 * drv.q_bar;
    Vector3 qN = A2 * drv.qN_bar;

    Float cos_theta = 0.5 * (p.dot(q) + pN.dot(qN));
    Float sin_theta = 0.5 * (q.dot(pN) - qN.dot(p));

    Float cos_tgt = cos(drv.target_angle);
    Float sin_tgt = sin(drv.target_angle);

    Float s     = sin_theta * cos_tgt - cos_theta * sin_tgt;   // sin(delta)
    Float c     = cos_theta * cos_tgt + sin_theta * sin_tgt;   // cos(delta)
    Float c_err = Float(1) - c;                                // 1 - cos(delta)

    // ds/d{direction}
    Vector3 ds_dp  = -0.5 * (cos_tgt * qN + sin_tgt * q);
    Vector3 ds_dpN =  0.5 * (cos_tgt * q  - sin_tgt * qN);
    Vector3 ds_dq  =  0.5 * (cos_tgt * pN - sin_tgt * p);
    Vector3 ds_dqN = -0.5 * (cos_tgt * p  + sin_tgt * pN);

    // dc/d{direction}  (90-degree relation to ds)
    Vector3 dc_dp  =  ds_dpN;           //  0.5*(cos_tgt*q  - sin_tgt*qN)
    Vector3 dc_dpN = -ds_dp;            //  0.5*(cos_tgt*qN + sin_tgt*q)
    Vector3 dc_dq  = -ds_dqN;           //  0.5*(cos_tgt*p  + sin_tgt*pN)
    Vector3 dc_dqN =  ds_dq;            //  0.5*(cos_tgt*pN - sin_tgt*p)

    auto JdirT_times = [](const Vector3& x_bar, const Vector3& g) -> Vector12
    {
        Vector12 result;
        result.segment<3>(0) = Vector3::Zero();
        result.segment<3>(3) = x_bar * g(0);
        result.segment<3>(6) = x_bar * g(1);
        result.segment<3>(9) = x_bar * g(2);
        return result;
    };

    Vector12 ds_dq1 = JdirT_times(drv.p_bar, ds_dp)  + JdirT_times(drv.pN_bar, ds_dpN);
    Vector12 ds_dq2 = JdirT_times(drv.q_bar, ds_dq)   + JdirT_times(drv.qN_bar, ds_dqN);
    Vector12 dc_dq1 = JdirT_times(drv.p_bar, dc_dp)  + JdirT_times(drv.pN_bar, dc_dpN);
    Vector12 dc_dq2 = JdirT_times(drv.q_bar, dc_dq)   + JdirT_times(drv.qN_bar, dc_dqN);

    Float K = drv.stiffness;
    grad_q1_out = K * (s * ds_dq1 - beta * c_err * dc_dq1);
    grad_q2_out = K * (s * ds_dq2 - beta * c_err * dc_dq2);
    // NOTE: external joint torque (drv.ext_torque) is NOT applied here as a
    // gradient term — that path needs the -tau*d2theta/dq2 Hessian for Newton
    // to converge.  Instead it is folded into q_tilde as a constant generalized
    // force each step (cal_q_tilde.cu, libuipc q_tilde path) so it stays
    // constant within the solve and the kinetic Hessian M handles it.
}


/// Compute Hessian of revolute driving energy using Gauss-Newton approximation.
///
/// E = 0.5 * K * ( s^2 + beta * c_err^2 )
///
/// Gauss-Newton treats this as sum of squared residuals:
///   r1 = sqrt(K) * s,     r2 = sqrt(K*beta) * c_err
///   H_GN = J1^T J1 + J2^T J2
///        = K * ds*ds^T + K*beta * dc*dc^T
///
/// This is SPD by construction (sum of rank-1 outer products).
///
MUDA_GENERIC inline void revolute_driving_hessian(
    const RevoluteDrivingGPUData& drv,
    const Vector12& q1,
    const Vector12& q2,
    Matrix12x12& H_11_out,
    Matrix12x12& H_22_out,
    Matrix12x12& H_12_out)
{
    constexpr Float beta = Float(0.02);

    Matrix3x3 A1 = extract_A(q1);
    Matrix3x3 A2 = extract_A(q2);

    Vector3 p  = A1 * drv.p_bar;
    Vector3 pN = A1 * drv.pN_bar;
    Vector3 q  = A2 * drv.q_bar;
    Vector3 qN = A2 * drv.qN_bar;

    Float cos_tgt = cos(drv.target_angle);
    Float sin_tgt = sin(drv.target_angle);

    // ds/d{direction}
    Vector3 ds_dp  = -0.5 * (cos_tgt * qN + sin_tgt * q);
    Vector3 ds_dpN =  0.5 * (cos_tgt * q  - sin_tgt * qN);
    Vector3 ds_dq  =  0.5 * (cos_tgt * pN - sin_tgt * p);
    Vector3 ds_dqN = -0.5 * (cos_tgt * p  + sin_tgt * pN);

    // dc/d{direction}  (90-degree relation)
    Vector3 dc_dp  =  ds_dpN;
    Vector3 dc_dpN = -ds_dp;
    Vector3 dc_dq  = -ds_dqN;
    Vector3 dc_dqN =  ds_dq;

    auto JdirT_times = [](const Vector3& x_bar, const Vector3& g) -> Vector12
    {
        Vector12 result;
        result.segment<3>(0) = Vector3::Zero();
        result.segment<3>(3) = x_bar * g(0);
        result.segment<3>(6) = x_bar * g(1);
        result.segment<3>(9) = x_bar * g(2);
        return result;
    };

    Vector12 ds_dq1 = JdirT_times(drv.p_bar, ds_dp)  + JdirT_times(drv.pN_bar, ds_dpN);
    Vector12 ds_dq2 = JdirT_times(drv.q_bar, ds_dq)   + JdirT_times(drv.qN_bar, ds_dqN);
    Vector12 dc_dq1 = JdirT_times(drv.p_bar, dc_dp)  + JdirT_times(drv.pN_bar, dc_dpN);
    Vector12 dc_dq2 = JdirT_times(drv.q_bar, dc_dq)   + JdirT_times(drv.qN_bar, dc_dqN);

    Float K = drv.stiffness;
    H_11_out = K * (ds_dq1 * ds_dq1.transpose() + beta * dc_dq1 * dc_dq1.transpose());
    H_22_out = K * (ds_dq2 * ds_dq2.transpose() + beta * dc_dq2 * dc_dq2.transpose());
    H_12_out = K * (ds_dq1 * ds_dq2.transpose() + beta * dc_dq1 * dc_dq2.transpose());
}


/// Current RELATIVE joint angle theta = atan2(sin_theta, cos_theta) (same frame as
/// target_angle: absolute = theta + initial_angle_offset).
MUDA_GENERIC inline Float revolute_current_relative_angle(
    const RevoluteDrivingGPUData& drv, const Vector12& q1, const Vector12& q2)
{
    Matrix3x3 A1 = extract_A(q1);
    Matrix3x3 A2 = extract_A(q2);
    Vector3 p  = A1 * drv.p_bar;
    Vector3 pN = A1 * drv.pN_bar;
    Vector3 q  = A2 * drv.q_bar;
    Vector3 qN = A2 * drv.qN_bar;
    Float cos_theta = Float(0.5) * (p.dot(q) + pN.dot(qN));
    Float sin_theta = Float(0.5) * (q.dot(pN) - qN.dot(p));
    return atan2(sin_theta, cos_theta);
}

/// One-sided joint-LIMIT penalty energy. Reuses the driving spring with target = the
/// violated bound and stiffness = limit_stiffness. Returns 0 when within [lower, upper].
MUDA_GENERIC inline Float revolute_limit_energy(
    const RevoluteDrivingGPUData& drv, const Vector12& q1, const Vector12& q2)
{
    if(drv.limit_stiffness <= Float(0)) return Float(0);
    // [limit lagged active-set] use the per-FRAME frozen flag, not a
    // per-iteration theta test (that chattered; see limit_active in the struct).
    // While active the spring is TWO-SIDED toward the bound for this frame —
    // equivalent to treating the inequality as an equality constraint for one
    // frame; the update kernel releases it when the multiplier sign says the
    // spring is pulling instead of pushing.
    if(drv.limit_active == 0) return Float(0);
    Float bound = (drv.limit_active < 0) ? drv.lower_limit : drv.upper_limit;
    RevoluteDrivingGPUData d = drv;
    d.target_angle = bound;
    d.stiffness    = drv.limit_stiffness;
    d.ext_torque   = Float(0);
    return revolute_driving_energy(d, q1, q2);
}

/// One-sided joint-LIMIT penalty gradient + Gauss-Newton Hessian (zero when within range).
MUDA_GENERIC inline void revolute_limit_gradient_hessian(
    const RevoluteDrivingGPUData& drv, const Vector12& q1, const Vector12& q2,
    Vector12& g1, Vector12& g2, Matrix12x12& H11, Matrix12x12& H22, Matrix12x12& H12)
{
    g1.setZero(); g2.setZero(); H11.setZero(); H22.setZero(); H12.setZero();
    if(drv.limit_stiffness <= Float(0)) return;
    // [limit lagged active-set] frozen flag; must MATCH revolute_limit_energy
    // exactly or line search sees an energy inconsistent with the gradient.
    if(drv.limit_active == 0) return;
    Float bound = (drv.limit_active < 0) ? drv.lower_limit : drv.upper_limit;
    RevoluteDrivingGPUData d = drv;
    d.target_angle = bound;
    d.stiffness    = drv.limit_stiffness;
    d.ext_torque   = Float(0);
    revolute_driving_gradient(d, q1, q2, g1, g2);
    revolute_driving_hessian(d, q1, q2, H11, H22, H12);
}


// ============================================================================
// Prismatic Joint Constraint  (cross-product + direction matching)
// ============================================================================
//
// Constrains body A to only translate along axis t relative to body B.
// No relative rotation allowed.
//
// Constraint points in material space:
//   C_p_bar, C_q_bar  -- joint center on each body
//   t_p_bar, t_q_bar  -- axis direction on each body
//   n_p_bar, n_q_bar  -- normal direction on each body
//   b_p_bar, b_q_bar  -- binormal direction on each body
//
// Energy terms:
//   E0 = 0.5*K * ||(C_q - C_p) × t_p||^2
//   E1 = 0.5*K * ||(C_p - C_q) × t_q||^2
//   E2 = 0.5*K * ||n_p - n_q||^2
//   E3 = 0.5*K * ||b_p - b_q||^2
//   E = E0 + E1 + E2 + E3
//

struct PrismaticJointGPUData
{
    int parent_body_id;  // body p
    int child_body_id;   // body q

    // Material-space data
    Vector3 Cp_bar;   // joint center in parent
    Vector3 Cq_bar;   // joint center in child
    Vector3 tp_bar;   // axis direction in parent (unit, direction only)
    Vector3 tq_bar;   // axis direction in child (unit, direction only)
    Vector3 np_bar;   // normal direction in parent (unit, direction only)
    Vector3 nq_bar;   // normal direction in child (unit, direction only)
    Vector3 bp_bar;   // binormal direction in parent (unit, direction only)
    Vector3 bq_bar;   // binormal direction in child (unit, direction only)
};


/// Compute the prismatic joint constraint energy.
MUDA_GENERIC inline Float prismatic_constraint_energy(
    const PrismaticJointGPUData& pj,
    const Vector12& qp,  // parent body state
    const Vector12& qq,  // child body state
    Float kappa)
{
    // Compute spatial quantities
    ABDJacobi J_Cp(pj.Cp_bar);
    ABDJacobi J_Cq(pj.Cq_bar);
    Vector3 Cp = J_Cp * qp;
    Vector3 Cq = J_Cq * qq;

    Matrix3x3 Ap = extract_A(qp);
    Matrix3x3 Aq = extract_A(qq);

    Vector3 tp = Ap * pj.tp_bar;
    Vector3 tq = Aq * pj.tq_bar;
    Vector3 np = Ap * pj.np_bar;
    Vector3 nq = Aq * pj.nq_bar;
    Vector3 bp = Ap * pj.bp_bar;
    Vector3 bq = Aq * pj.bq_bar;

    Vector3 diff = Cq - Cp;
    Vector3 cross0 = diff.cross(tp);   // (C_q - C_p) × t_p
    Vector3 cross1 = (-diff).cross(tq);  // (C_p - C_q) × t_q
    Vector3 n_err = np - nq;
    Vector3 b_err = bp - bq;

    Float E0 = 0.5 * kappa * cross0.squaredNorm();
    Float E1 = 0.5 * kappa * cross1.squaredNorm();
    Float E2 = 0.5 * kappa * n_err.squaredNorm();
    Float E3 = 0.5 * kappa * b_err.squaredNorm();

    return E0 + E1 + E2 + E3;
}


/// Compute gradient and Hessian of prismatic joint constraint.
///
/// We use the chain rule approach from the reference:
///   F = [C_p - C_q; t_p; t_q]  (9x1)
///   E01 = 0.5*K*||(C_q-C_p)×t_p||^2 + 0.5*K*||(C_p-C_q)×t_q||^2
///   F = J_01 * [qp; qq]
///
/// For E2, E3 we use the same positional constraint approach as fixed joints.
///
MUDA_GENERIC inline void prismatic_constraint_gradient_hessian(
    const PrismaticJointGPUData& pj,
    const Vector12& qp,
    const Vector12& qq,
    Float kappa,
    Vector12& grad_p_out,
    Vector12& grad_q_out,
    Matrix12x12& H_pp_out,
    Matrix12x12& H_qq_out,
    Matrix12x12& H_pq_out)
{
    // === E2 and E3: direction matching (same as positional constraint) ===
    // E2 = 0.5*K*||n_p - n_q||^2 where n_p = A_p * n_bar, n_q = A_q * n_bar
    // E3 = 0.5*K*||b_p - b_q||^2
    //
    // These are exactly positional constraints on direction vectors.
    // Using ABDJacobi formulation but for directions (no translation term):
    //   n_p = A_p * n_p_bar  (direction, no +p)
    //   We handle this by using JointConstraintGPUData logic with direction-only Jacobi.
    //
    // For implementation simplicity, we use the fact that for direction vectors:
    //   d(A*x_bar)/dq is the "direction Jacobi" (translation part = 0)
    //   The gradient/hessian pattern is:
    //     grad_p += K * J_dir(n_p_bar)^T * (A_p*n_p_bar - A_q*n_q_bar)
    //     grad_q -= K * J_dir(n_q_bar)^T * (A_p*n_p_bar - A_q*n_q_bar)
    //   etc.

    Matrix3x3 Ap = extract_A(qp);
    Matrix3x3 Aq = extract_A(qq);

    Vector3 Cp = ABDJacobi(pj.Cp_bar) * qp;
    Vector3 Cq = ABDJacobi(pj.Cq_bar) * qq;
    Vector3 tp = Ap * pj.tp_bar;
    Vector3 tq = Aq * pj.tq_bar;
    Vector3 np = Ap * pj.np_bar;
    Vector3 nq = Aq * pj.nq_bar;
    Vector3 bp = Ap * pj.bp_bar;
    Vector3 bq = Aq * pj.bq_bar;

    grad_p_out = Vector12::Zero();
    grad_q_out = Vector12::Zero();
    H_pp_out   = Matrix12x12::Zero();
    H_qq_out   = Matrix12x12::Zero();
    H_pq_out   = Matrix12x12::Zero();

    // Helper: compute J_dir^T * g for direction-only Jacobi
    auto JdirT = [](const Vector3& x_bar, const Vector3& g) -> Vector12
    {
        Vector12 r;
        r.segment<3>(0) = Vector3::Zero();
        r.segment<3>(3) = x_bar * g(0);
        r.segment<3>(6) = x_bar * g(1);
        r.segment<3>(9) = x_bar * g(2);
        return r;
    };

    // Helper: compute J_dir^T * M * J_dir  (12x12) for a single direction
    auto JdirT_M_Jdir = [](const Vector3& x, const Matrix3x3& M, const Vector3& y) -> Matrix12x12
    {
        // Same structure as ABDJacobi::JT_H_J but with zero translation block
        Matrix12x12 ret = Matrix12x12::Zero();
        Matrix3x3 xy = x * y.transpose();
        // The 3x3 block at (3+3i, 3+3j) = M(i,j) * x * y^T
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                ret.block<3,3>(3+3*i, 3+3*j) = xy * M(i,j);
        return ret;
    };

    // ---- E2: ||n_p - n_q||^2 ----
    {
        Vector3 err = np - nq;
        // grad w.r.t. qp: K * J_dir(n_p_bar)^T * err
        grad_p_out += kappa * JdirT(pj.np_bar, err);
        // grad w.r.t. qq: -K * J_dir(n_q_bar)^T * err
        grad_q_out -= kappa * JdirT(pj.nq_bar, err);
        // Hessian (Gauss-Newton, SPD):
        Matrix3x3 I3 = Matrix3x3::Identity();
        H_pp_out += kappa * JdirT_M_Jdir(pj.np_bar, I3, pj.np_bar);
        H_qq_out += kappa * JdirT_M_Jdir(pj.nq_bar, I3, pj.nq_bar);
        H_pq_out -= kappa * JdirT_M_Jdir(pj.np_bar, I3, pj.nq_bar);
    }

    // ---- E3: ||b_p - b_q||^2 ----
    {
        Vector3 err = bp - bq;
        grad_p_out += kappa * JdirT(pj.bp_bar, err);
        grad_q_out -= kappa * JdirT(pj.bq_bar, err);
        Matrix3x3 I3 = Matrix3x3::Identity();
        H_pp_out += kappa * JdirT_M_Jdir(pj.bp_bar, I3, pj.bp_bar);
        H_qq_out += kappa * JdirT_M_Jdir(pj.bq_bar, I3, pj.bq_bar);
        H_pq_out -= kappa * JdirT_M_Jdir(pj.bp_bar, I3, pj.bq_bar);
    }

    // ---- E0: 0.5*K*||(C_q - C_p) × t_p||^2 ----
    // Let d = C_q - C_p, cross0 = d × t_p
    // E0 = 0.5*K*||cross0||^2
    //
    // Using F01 = [C_p - C_q; t_p; t_q] and chain rule:
    //   d(cross0)/d(d) = -[t_p]_x  (skew-symmetric matrix of t_p)
    //   d(cross0)/d(t_p) = [d]_x    (skew-symmetric matrix of d)
    //
    // The full approach: compute gradient/hessian of ||a×b||^2 directly.
    // d(||a×b||^2)/da = 2*(a×b)×b = 2*([b]_x^T * [b]_x * a - (a^T*b)*b ... )
    // Simpler: use d(||a×b||^2)/da = 2*[b]_x^T * (a × b)
    //          d(||a×b||^2)/db = -2*[a]_x^T * (a × b)
    {
        Vector3 d = Cq - Cp;  // d depends on qp (via Cp) and qq (via Cq)
        Vector3 cross0 = d.cross(tp);

        // Skew-symmetric matrices
        Matrix3x3 tp_x;
        tp_x << 0, -tp(2), tp(1),
                 tp(2), 0, -tp(0),
                -tp(1), tp(0), 0;

        Matrix3x3 d_x;
        d_x << 0, -d(2), d(1),
               d(2), 0, -d(0),
              -d(1), d(0), 0;

        // d(cross0)/d(d) = -tp_x  (since cross0 = d × tp = -tp × d = -[tp]_x * d)
        // Actually: a × b = [a]_x * b, so d × tp = [d]_x * tp
        // d(d×tp)/d(d) = skew(tp) ... no.
        // d×tp = [d]_x * tp.  d([d]_x * tp)/dd: this is the derivative of cross product.
        // ∂(d×tp)_i/∂d_j = ε_{ijk} tp_k = -[tp]_x_{ij}
        // So d(d×tp)/dd = -[tp]_x^T  ... let's be more careful.
        // (d×tp)_0 = d_1*tp_2 - d_2*tp_1
        // ∂/∂d_0 = 0, ∂/∂d_1 = tp_2, ∂/∂d_2 = -tp_1
        // This gives row 0 of the Jacobian = [0, tp_2, -tp_1] = -tp_x^T row 0?
        // tp_x = [0, -tp_2, tp_1; tp_2, 0, -tp_0; -tp_1, tp_0, 0]
        // -tp_x^T = [0, -tp_2, tp_1; tp_2, 0, -tp_0; -tp_1, tp_0, 0] = tp_x (skew-sym!)
        // No: -tp_x^T = -[-tp_x] = tp_x. But skew is anti-symmetric: tp_x^T = -tp_x.
        // So -tp_x^T = tp_x.
        // Hmm, let me just use: d(d×tp)/dd = [tp]_x^T  (anti-symmetric, so = -[tp]_x)
        // Actually from the formula: d(a×b)/da = -[b]_x
        // d(d×tp)/dd = -[tp]_x
        //
        // d(d×tp)/dtp: similarly, d(a×b)/db = [a]_x
        // d(d×tp)/dtp = [d]_x

        // Gradient of E0 w.r.t. the 9 intermediate variables F = [d; tp]
        // But d = Cq-Cp depends on translations, tp = Ap*tp_bar depends on rotations.
        //
        // Let's compute directly:
        // dE0/d(d) = K * (-[tp]_x)^T * cross0 = K * [tp]_x * cross0 = K * tp × cross0
        // Wait: [tp]_x * cross0 = tp × cross0.  And (-[tp]_x)^T = [tp]_x.
        // dE0/d(d) = kappa * [tp]_x * cross0 = kappa * (tp × cross0)
        // Hmm that doesn't seem right dimensionally. Let me redo.
        //
        // E0 = 0.5*K*cross0^T*cross0, where cross0 = d × tp
        // dE0/dcross0 = K * cross0
        // dcross0/dd = -[tp]_x  (see above)
        // dE0/dd = dcross0/dd^T * dE0/dcross0 = (-[tp]_x)^T * K * cross0 = [tp]_x * K * cross0
        //        = K * tp × cross0
        // But tp × (d × tp) = d*(tp·tp) - tp*(tp·d) = d - tp*(tp·d)  (if |tp|=1)
        // This makes sense: it projects d onto the plane ⊥ tp.

        Vector3 dE0_dd  = kappa * tp.cross(cross0);   // = kappa * (tp × (d × tp))
        // dE0/dtp: dcross0/dtp = [d]_x
        Vector3 dE0_dtp = kappa * (-d).cross(cross0);  // = kappa * [d]_x^T * cross0 = kappa*cross0×d... 
        // Actually: dcross0/dtp = [d]_x, so dE0/dtp = [d]_x^T * K*cross0 = -[d]_x * K*cross0 = K*(cross0 × d)
        // Hmm: [d]_x^T = -[d]_x.
        // dE0/dtp = (-[d]_x) * kappa * cross0 = kappa * (cross0 × d)  ... wait that's also not right.
        // [d]_x * v = d × v, so [d]_x^T * v = -(d × v) = v × d
        // dE0/dtp = [d]_x^T * kappa*cross0 = kappa * (cross0 × d)
        // But cross0 = d×tp, so cross0 × d = (d×tp)×d = -d×(d×tp) = -(d(d·tp) - tp*|d|²)
        // Hmm, let me just compute numerically-friendly form.

        // Let's just use: dE0/dtp = kappa * d_x^T * cross0 where d_x^T = -d_x
        // = -kappa * [d]_x * cross0 = -kappa * (d × cross0)
        dE0_dtp = -kappa * d.cross(cross0);

        // Now map to ABD DOFs:
        // d = Cq - Cp, so dd/dqp = -J(Cp_bar), dd/dqq = +J(Cq_bar)
        // tp = Ap*tp_bar, so dtp/dqp = J_dir(tp_bar), dtp/dqq = 0
        ABDJacobi J_Cp(pj.Cp_bar);
        ABDJacobi J_Cq(pj.Cq_bar);

        grad_p_out += -(J_Cp.T() * dE0_dd) + JdirT(pj.tp_bar, dE0_dtp);
        grad_q_out +=  (J_Cq.T() * dE0_dd);

        // Hessian of E0 (Gauss-Newton approximation):
        // H = K * J_chain^T * J_chain where J_chain maps [qp,qq] -> cross0
        //
        // dcross0/dqp = (-[tp]_x)*(-J_Cp) + [d]_x * J_dir(tp_bar)
        //             = [tp]_x * J_Cp + [d]_x * J_dir(tp_bar)
        // dcross0/dqq = (-[tp]_x) * J_Cq
        //
        // For Gauss-Newton we need the 3x12 Jacobian dcross0/dqp and dcross0/dqq.
        // This is complex but we can build the 3x24 Jacobian and compute H = K * J^T J.
        //
        // However, for implementation simplicity, we use the product form:
        // H_pp = K * (dcross0/dqp)^T * (dcross0/dqp)
        // etc.
        //
        // dcross0/dqp (3x12) = (-tp_x) * (-J_Cp.to_mat()) + d_x * J_dir(tp_bar).to_mat()
        //                    = tp_x * J_Cp_mat + d_x * J_dir_tp_mat

        // Build J_Cp_mat (3x12)
        Matrix3x12 J_Cp_mat = J_Cp.to_mat();
        Matrix3x12 J_Cq_mat = J_Cq.to_mat();

        // Build J_dir_tp_mat (3x12): direction-only Jacobi for tp_bar
        Matrix3x12 J_dir_tp;
        J_dir_tp.setZero();
        J_dir_tp.block<1,3>(0, 3) = pj.tp_bar.transpose();
        J_dir_tp.block<1,3>(1, 6) = pj.tp_bar.transpose();
        J_dir_tp.block<1,3>(2, 9) = pj.tp_bar.transpose();

        // dcross0/dqp = tp_x * J_Cp_mat + d_x * J_dir_tp   ... wait sign.
        // cross0 = d × tp, d = Cq - Cp.
        // dcross0/dqp = d(d×tp)/dd * dd/dqp + d(d×tp)/dtp * dtp/dqp
        //             = (-tp_x) * (-J_Cp_mat) + d_x * J_dir_tp
        //             = tp_x * J_Cp_mat + d_x * J_dir_tp
        // Hmm: d(d×tp)/dd = -[tp]_x  (as derived above)
        //   Wait, I derived d(d×tp)/dd and got -[tp]_x.
        //   Then d(d×tp)/dd * dd/dqp = -[tp]_x * (-J_Cp) = [tp]_x * J_Cp. OK.
        // And d(d×tp)/dtp = [d]_x.

        Matrix3x12 dcross0_dqp = tp_x * J_Cp_mat + d_x * J_dir_tp;
        Matrix3x12 dcross0_dqq = (-tp_x) * J_Cq_mat;  // only d depends on qq, not tp

        H_pp_out += kappa * (dcross0_dqp.transpose() * dcross0_dqp);
        H_qq_out += kappa * (dcross0_dqq.transpose() * dcross0_dqq);
        H_pq_out += kappa * (dcross0_dqp.transpose() * dcross0_dqq);
    }

    // ---- E1: 0.5*K*||(C_p - C_q) × t_q||^2 ----
    // Same structure as E0 but with swapped roles:
    //   d1 = Cp - Cq, t = tq (on body q)
    //   cross1 = d1 × tq
    {
        Vector3 d1 = Cp - Cq;
        Vector3 cross1 = d1.cross(tq);

        Matrix3x3 tq_x;
        tq_x << 0, -tq(2), tq(1),
                 tq(2), 0, -tq(0),
                -tq(1), tq(0), 0;

        Matrix3x3 d1_x;
        d1_x << 0, -d1(2), d1(1),
                d1(2), 0, -d1(0),
               -d1(1), d1(0), 0;

        Vector3 dE1_dd1  = kappa * tq.cross(cross1);
        Vector3 dE1_dtq  = -kappa * d1.cross(cross1);

        ABDJacobi J_Cp(pj.Cp_bar);
        ABDJacobi J_Cq(pj.Cq_bar);

        // dd1/dqp = +J_Cp, dd1/dqq = -J_Cq
        // dtq/dqq = J_dir(tq_bar), dtq/dqp = 0
        grad_p_out +=  (J_Cp.T() * dE1_dd1);
        grad_q_out += -(J_Cq.T() * dE1_dd1) + JdirT(pj.tq_bar, dE1_dtq);

        Matrix3x12 J_Cp_mat = J_Cp.to_mat();
        Matrix3x12 J_Cq_mat = J_Cq.to_mat();

        Matrix3x12 J_dir_tq;
        J_dir_tq.setZero();
        J_dir_tq.block<1,3>(0, 3) = pj.tq_bar.transpose();
        J_dir_tq.block<1,3>(1, 6) = pj.tq_bar.transpose();
        J_dir_tq.block<1,3>(2, 9) = pj.tq_bar.transpose();

        // dcross1/dqp = (-[tq]_x) * J_Cp  (only d1 depends on qp)
        // dcross1/dqq = (-[tq]_x) * (-J_Cq) + [d1]_x * J_dir_tq
        //             = [tq]_x * J_Cq + [d1]_x * J_dir_tq
        // Wait: d(d1×tq)/dd1 = -[tq]_x (same derivation as before).
        //   dd1/dqp = +J_Cp, so contribution = -tq_x * J_Cp
        //   dd1/dqq = -J_Cq, so contribution = -tq_x * (-J_Cq) = tq_x * J_Cq
        // d(d1×tq)/dtq = [d1]_x, dtq/dqq = J_dir_tq

        Matrix3x12 dcross1_dqp = (-tq_x) * J_Cp_mat;
        Matrix3x12 dcross1_dqq = tq_x * J_Cq_mat + d1_x * J_dir_tq;

        H_pp_out += kappa * (dcross1_dqp.transpose() * dcross1_dqp);
        H_qq_out += kappa * (dcross1_dqq.transpose() * dcross1_dqq);
        H_pq_out += kappa * (dcross1_dqp.transpose() * dcross1_dqq);
    }
}


// ============================================================================
// Prismatic Driving Joint  (linear displacement control)
// ============================================================================
//
// Energy: E = 0.5 * K * (d - d_target)^2
// where d = (C_q - C_p) · t_q
//
// Chain rule:
//   dE/d(qp,qq) = K * (d - d_tgt) * dd/d(qp,qq)
//   H = K * (dd/d(qp,qq))^T * (dd/d(qp,qq))  (Gauss-Newton, SPD)
//

struct PrismaticDrivingGPUData
{
    int parent_body_id;
    int child_body_id;

    Vector3 Cp_bar;    // joint center in parent material
    Vector3 Cq_bar;    // joint center in child material
    Vector3 tq_bar;    // axis direction in child material (unit)
    // [force-control] axis direction in PARENT material (unit). At rest both
    // bodies see the same world axis; libuipc's prismatic external force keeps
    // a per-body tangent (Vector6 t_bar) so each rotates with its own body and
    // symmetrizes. We store the parent copy here for the q_tilde wrench path.
    Vector3 tp_bar = Vector3::Zero();

    Float stiffness;       // K
    Float target_distance; // d_tgt
    // [force-control] external force (N) along the prismatic axis, fed through
    // the q_tilde path (no Hessian; constant within the Newton solve). Mirrors
    // libuipc AffineBodyPrismaticJointExternalForce: +f*t to child, -f*t to
    // parent (translation DOF). Independent of the PD term: stiffness>0 =
    // position control, ext_force!=0 = force control; set stiffness=0 for pure.
    Float ext_force = Float(0);

    // [force-control / limit barrier] One-sided IPC log-barrier on the prismatic
    // coordinate d at the CLOSED end, so d can never cross limit_cl no matter how
    // large the (force) drive — guaranteeing the gripper never overshoots past
    // fully-closed while staying force-controlled. gap g = (d - limit_cl)*limit_dir
    // is >0 while open and ->0 at the limit; barrier active for 0<g<limit_dhat.
    // limit_kappa<=0 disables it (default — no effect on existing scenes).
    Float limit_cl    = Float(0);
    Float limit_dir   = Float(1);   // +1 if open end is at d>cl, else -1
    Float limit_dhat  = Float(0);   // activation band (m)
    Float limit_kappa = Float(0);   // barrier stiffness (<=0 = off)
    // [force-control] SECOND one-sided barrier (slot 1), used for the OPEN end so a
    // pure-force grip cannot overshoot past either limit. Same semantics as the
    // first set; gap g2 = (d - limit_cl2)*limit_dir2. limit_kappa2<=0 = off.
    Float limit_cl2    = Float(0);
    Float limit_dir2   = Float(1);
    Float limit_dhat2  = Float(0);
    Float limit_kappa2 = Float(0);

    // [joint limit] One-sided position-limit PENALTY (same scheme as the revolute
    // limit; distinct from the IPC barrier fields above). Active for any control mode
    // (function of the current displacement d, not the drive target). lo/hi are in the
    // engine's prismatic-coordinate frame (= the d computed by prismatic_driving_*).
    Float lower_limit     = -Float(1e30);
    Float upper_limit     =  Float(1e30);
    Float pen_limit_stiffness = Float(0);  // K_lim = limit_strength_ratio*(m_p+m_c); 0 = off
};


/// One-sided IPC log-barrier on the prismatic coordinate at the closed limit.
/// Returns barrier energy E and its 1st/2nd derivatives w.r.t. d. b(g) =
/// -kappa*(g-dhat)^2*ln(g/dhat) for 0<g<dhat (g = (d-cl)*dir). Off if kappa<=0.
// Core: ONE one-sided barrier given explicit (cl, dir, dhat, kappa).
MUDA_GENERIC inline void _prismatic_one_barrier(
    Float d, Float cl, Float dir, Float dh, Float k,
    Float& E, Float& dEdd, Float& d2Edd2)
{
    E = Float(0); dEdd = Float(0); d2Edd2 = Float(0);
    if(k <= Float(0) || dh <= Float(0)) return;
    Float g   = (d - cl) * dir;                  // >0 on the allowed side
    if(g >= dh) return;                          // outside barrier support
    Float gc = g < Float(1e-9) ? Float(1e-9) : g;  // numerical floor (avoid log<=0)
    Float t  = gc - dh;                          // <0 in support
    Float ln = log(gc / dh);
    E      = -k * t * t * ln;
    Float bp  = -k * (Float(2) * t * ln + t * t / gc);             // b'(g)
    Float bpp = -k * (Float(2) * ln + Float(4) * t / gc - (t * t) / (gc * gc)); // b''(g)
    dEdd   = bp * dir;                           // chain dg/dd = dir
    d2Edd2 = bpp;                                // dir^2 = 1
    if(d2Edd2 < Float(0)) d2Edd2 = Float(0);     // SPD safety
}

// Sum of BOTH one-sided barriers (slot 0 = closed end, slot 1 = open end). Either
// is off when its kappa<=0, so single-barrier and no-barrier scenes are unchanged.
MUDA_GENERIC inline void prismatic_limit_barrier(
    Float d, const PrismaticDrivingGPUData& drv,
    Float& E, Float& dEdd, Float& d2Edd2)
{
    Float E1, g1, h1, E2, g2, h2;
    _prismatic_one_barrier(d, drv.limit_cl,  drv.limit_dir,  drv.limit_dhat,  drv.limit_kappa,  E1, g1, h1);
    _prismatic_one_barrier(d, drv.limit_cl2, drv.limit_dir2, drv.limit_dhat2, drv.limit_kappa2, E2, g2, h2);
    E = E1 + E2; dEdd = g1 + g2; d2Edd2 = h1 + h2;
}


/// Compute prismatic driving energy.
MUDA_GENERIC inline Float prismatic_driving_energy(
    const PrismaticDrivingGPUData& drv,
    const Vector12& qp,
    const Vector12& qq)
{
    Vector3 Cp = ABDJacobi(drv.Cp_bar) * qp;
    Vector3 Cq = ABDJacobi(drv.Cq_bar) * qq;
    Vector3 tq = extract_A(qq) * drv.tq_bar;

    Float d = (Cq - Cp).dot(tq);
    Float err = d - drv.target_distance;
    Float Eb, dummy1, dummy2;
    prismatic_limit_barrier(d, drv, Eb, dummy1, dummy2);
    return 0.5 * drv.stiffness * err * err + Eb;
}


/// Compute gradient and Hessian of prismatic driving energy.
MUDA_GENERIC inline void prismatic_driving_gradient_hessian(
    const PrismaticDrivingGPUData& drv,
    const Vector12& qp,
    const Vector12& qq,
    Vector12& grad_p_out,
    Vector12& grad_q_out,
    Matrix12x12& H_pp_out,
    Matrix12x12& H_qq_out,
    Matrix12x12& H_pq_out)
{
    ABDJacobi J_Cp(drv.Cp_bar);
    ABDJacobi J_Cq(drv.Cq_bar);

    Vector3 Cp = J_Cp * qp;
    Vector3 Cq = J_Cq * qq;
    Matrix3x3 Aq = extract_A(qq);
    Vector3 tq = Aq * drv.tq_bar;

    Vector3 diff = Cq - Cp;
    Float d = diff.dot(tq);
    Float err = d - drv.target_distance;
    Float K = drv.stiffness;

    // d = (Cq - Cp) · tq
    // dd/dqp = -tq^T * J_Cp  (1x12)
    // dd/dqq = tq^T * J_Cq + (Cq-Cp)^T * J_dir(tq_bar)  (1x12)
    //        = tq^T * J_Cq + diff^T * J_dir_tq

    auto JdirT = [](const Vector3& x_bar, const Vector3& g) -> Vector12
    {
        Vector12 r;
        r.segment<3>(0) = Vector3::Zero();
        r.segment<3>(3) = x_bar * g(0);
        r.segment<3>(6) = x_bar * g(1);
        r.segment<3>(9) = x_bar * g(2);
        return r;
    };

    // dd/dqp (12x1 via transposition)
    Vector12 dd_dqp = -(J_Cp.T() * tq);
    // dd/dqq: J_Cq^T * tq + J_dir(tq_bar)^T * diff
    Vector12 dd_dqq = (J_Cq.T() * tq) + JdirT(drv.tq_bar, diff);

    // PD term (dE/dd = K*err, d2E/dd2 = K) + one-sided limit barrier (same chain,
    // both depend on d only -> share dd/dq). Gauss-Newton: drop d2d/dq2.
    Float Eb, b_dEdd, b_d2Edd2;
    prismatic_limit_barrier(d, drv, Eb, b_dEdd, b_d2Edd2);
    Float gscale = K * err + b_dEdd;   // total dE/dd
    Float hscale = K + b_d2Edd2;       // total d2E/dd2 (>=0, SPD)

    // Gradient: (dE/dd) * dd/dq
    grad_p_out = gscale * dd_dqp;
    grad_q_out = gscale * dd_dqq;

    // Gauss-Newton Hessian: (d2E/dd2) * dd/dq * dd/dq^T
    H_pp_out = hscale * dd_dqp * dd_dqp.transpose();
    H_qq_out = hscale * dd_dqq * dd_dqq.transpose();
    H_pq_out = hscale * dd_dqp * dd_dqq.transpose();
}


/// Current prismatic displacement d = (Cq - Cp)·tq (engine joint-coordinate frame).
MUDA_GENERIC inline Float prismatic_current_displacement(
    const PrismaticDrivingGPUData& drv, const Vector12& qp, const Vector12& qq)
{
    Vector3 Cp = ABDJacobi(drv.Cp_bar) * qp;
    Vector3 Cq = ABDJacobi(drv.Cq_bar) * qq;
    Vector3 tq = extract_A(qq) * drv.tq_bar;
    return (Cq - Cp).dot(tq);
}

/// One-sided prismatic position-limit PENALTY (same scheme as revolute_limit_*). Reuses
/// the driving spring toward the violated bound with the barrier disabled. Zero in range.
MUDA_GENERIC inline Float prismatic_limit_energy(
    const PrismaticDrivingGPUData& drv, const Vector12& qp, const Vector12& qq)
{
    if(drv.pen_limit_stiffness <= Float(0)) return Float(0);
    Float d = prismatic_current_displacement(drv, qp, qq);
    Float bound;
    if(d < drv.lower_limit)      bound = drv.lower_limit;
    else if(d > drv.upper_limit) bound = drv.upper_limit;
    else return Float(0);
    PrismaticDrivingGPUData c = drv;
    c.target_distance = bound; c.stiffness = drv.pen_limit_stiffness; c.ext_force = Float(0);
    c.limit_kappa = Float(0); c.limit_kappa2 = Float(0);  // disable barrier in the reused call
    return prismatic_driving_energy(c, qp, qq);
}

MUDA_GENERIC inline void prismatic_limit_gradient_hessian(
    const PrismaticDrivingGPUData& drv, const Vector12& qp, const Vector12& qq,
    Vector12& gp, Vector12& gq, Matrix12x12& Hpp, Matrix12x12& Hqq, Matrix12x12& Hpq)
{
    gp.setZero(); gq.setZero(); Hpp.setZero(); Hqq.setZero(); Hpq.setZero();
    if(drv.pen_limit_stiffness <= Float(0)) return;
    Float d = prismatic_current_displacement(drv, qp, qq);
    Float bound;
    if(d < drv.lower_limit)      bound = drv.lower_limit;
    else if(d > drv.upper_limit) bound = drv.upper_limit;
    else return;
    PrismaticDrivingGPUData c = drv;
    c.target_distance = bound; c.stiffness = drv.pen_limit_stiffness; c.ext_force = Float(0);
    c.limit_kappa = Float(0); c.limit_kappa2 = Float(0);
    prismatic_driving_gradient_hessian(c, qp, qq, gp, gq, Hpp, Hqq, Hpq);
}


}  // namespace gipc
