#pragma once
#include <gipc/type_define.h>
namespace gipc
{
class ABDSystemParms
{
  public:
    Vector12 init_q_v     = Vector12::Zero();
    Vector3  gravity      = Vector3{0, -9.8, 0};  // m/s^2
    Float    dt           = 0.01;                 // s
    Float    mass_density = 1e3;                  // kg/m^3
    Float    kappa        = 1e8;
    Float    motor_speed  = 31.4;  // rad/s
    Float    motor_strength = 10; // how strong the motor is, related to the body mass

    // Joint positional constraint: kappa = strength_ratio * (m_parent + m_child).
    // Matches rbs-uipc formulation where stiffness scales with body mass for
    // scale-independent conditioning.  Default 100 matches rbs-uipc default.
    Float    joint_strength_ratio = 100.0;

    // Revolute driving energy: K = strength_ratio * (m_parent + m_child).
    // Same mass-based formulation as rbs-uipc (default 100).
    Float    revolute_driving_strength_ratio = 100.0;

    // [joint limit] One-sided position-limit PENALTY stiffness: K_lim =
    // joint_limit_strength_ratio * (m_parent + m_child). A mass-scaled spring toward the
    // violated bound (NOT a barrier). Because it is an implicit energy term (gradient + SPD
    // Gauss-Newton Hessian) it stays stable even when stiff. 0 = disable limits.
    // 20000 (vs the 1000 used elsewhere): a stiff drive (target_ke~1e3) needs a stiff
    // limit spring to actually hold the bound (1000 -> ~2.5x overshoot; 20000 -> ~5%).
    Float    joint_limit_strength_ratio = 20000.0;

    // Prismatic joint constraint: kappa = strength_ratio * (m_parent + m_child).
    Float    prismatic_strength_ratio = 100.0;

    // Prismatic driving energy: K = strength_ratio * (m_parent + m_child).
    Float    prismatic_driving_strength_ratio = 100.0;

    // Per-frame rate limits for driving target convergence.
    // Set large values temporarily to allow fast initial settling.
    Float    max_revolute_step_per_frame  = 0.1;    // rad
    Float    max_prismatic_step_per_frame = 0.002;  // m

    Float    velocity_damping = 0.0;  // per-step: v *= (1 - damping)
};
}  // namespace gipc
