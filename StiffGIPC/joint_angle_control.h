#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

/// Per-joint data needed for runtime angle control via the UI.
///
/// For each revolute joint:
///   - The positional constraint uses 2 axis endpoint points only.
///   - Angle control is via a SEPARATE sin-based driving energy:
///       E = 0.5 * K * sin(theta - theta_target)^2
///     which uses direction vectors (not additional constraint points).
///
/// This struct stores the axis/perpendicular directions needed to
/// initialise the RevoluteDrivingGPUData on the GPU, plus runtime
/// target_angle and UI display info.
///
struct JointAngleControlInfo
{
    int         constraint_index = -1;   ///< Index into tetMesh.joint_constraints[]
    int         control_point_idx = 2;   ///< (legacy, unused with new driving approach)

    /// Rest-configuration parent material-frame quantities
    Eigen::Vector3d origin_xbar;         ///< (legacy, unused with new driving approach)
    Eigen::Vector3d axis_dir;            ///< rotation axis direction (unit) in world frame
    Eigen::Vector3d n_dir;               ///< direction ⊥ axis (unit) in world frame
    double          offset_r = 0.05;     ///< (legacy, unused with new driving approach)

    /// Current target angle in radians (set by UI slider).
    /// When initial_angle_offset != 0 this is still an absolute URDF angle;
    /// the offset is subtracted internally so the driving energy sees
    /// effective_target = target_angle - initial_angle_offset.
    double target_angle = 0.0;

    /// Angle offset for FK-loaded poses. When the arm is loaded via
    /// set_initial_joint_angles(), this stores the initial angle so that
    /// the driving energy uses the correct reference.
    double initial_angle_offset = 0.0;

    /// Per-joint stiffness multiplier for the revolute driving energy.
    /// Actual K = parms.revolute_driving_strength_ratio * strength_ratio * (m_i+m_j) * dt².
    double strength_ratio = 1.0;

    /// Joint limits from URDF (in radians).
    /// Kept away from ±180° to avoid unstable branch switching near wrap boundaries.
    static constexpr double kSafeAngleLimit = 3.12413936106985;  // 179 deg
    double lower_limit = -kSafeAngleLimit;
    double upper_limit =  kSafeAngleLimit;

    /// Joint name for UI display
    std::string joint_name;

    /// (Legacy) Compute the updated parent_xbar for the angle-control point.
    Eigen::Vector3d compute_parent_control_xbar() const
    {
        Eigen::Vector3d m = axis_dir.cross(n_dir);
        return origin_xbar
               + offset_r * (std::cos(target_angle) * n_dir
                            + std::sin(target_angle) * m);
    }
};


/// Per-prismatic-joint data for runtime linear displacement control via UI.
struct PrismaticDrivingControlInfo
{
    int prismatic_constraint_index = -1;  ///< Index into tetMesh.prismatic_constraints[]

    Eigen::Vector3d axis_dir;             ///< translation axis (unit, world frame)

    double target_distance = 0.0;         ///< desired displacement along axis (meters)
    double strength_ratio  = 1.0;         ///< per-joint stiffness multiplier

    double lower_limit = 0.0;             ///< min displacement (from URDF)
    double upper_limit = 0.04;            ///< max displacement (from URDF)

    std::string joint_name;               ///< for UI display
};
