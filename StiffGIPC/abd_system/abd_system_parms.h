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

    // Prismatic joint constraint: kappa = strength_ratio * (m_parent + m_child).
    Float    prismatic_strength_ratio = 100.0;

    // Prismatic driving energy: K = strength_ratio * (m_parent + m_child).
    Float    prismatic_driving_strength_ratio = 100.0;

    // Per-frame rate limits for driving target convergence.
    // Set large values temporarily to allow fast initial settling.
    Float    max_revolute_step_per_frame  = 0.1;    // rad
    Float    max_prismatic_step_per_frame = 0.002;  // m
};
}  // namespace gipc
