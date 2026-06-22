#pragma once

#include <Eigen/Dense>

// Lightweight struct for joint constraint data on the host side.
// Separated from load_mesh.h to avoid heavy header dependencies in CUDA code.
struct JointConstraintHostInfo
{
    int parent_body_id;
    int child_body_id;
    enum class Type { Fixed, Revolute } type;
    int             num_points;  // 2 for revolute (axis endpoints), 1 for fixed (Method 2)
    Eigen::Vector3d world_anchor[4];  // world-space positions at t=0
    double          point_weight[4] = {1.0, 1.0, 1.0, 1.0};

    // Direction constraints for fixed joints (libuipc affine_body_fixed_joint:
    // penalize all 3 affine basis axes t,n,b so rotation is fully constrained).
    // World space at t=0; converted to material coords at init.
    bool            has_direction_constraint = false;
    Eigen::Vector3d world_tangent;    // world-space tangent direction (3rd axis = normal x bitangent)
    Eigen::Vector3d world_normal;     // world-space normal direction
    Eigen::Vector3d world_bitangent;  // world-space bitangent direction
};


// Host-side info for prismatic joint constraints.
// Stores world-space quantities needed to initialize PrismaticJointGPUData.
struct PrismaticJointHostInfo
{
    int parent_body_id;
    int child_body_id;

    Eigen::Vector3d world_center;     // joint center position (world space)
    Eigen::Vector3d world_axis;       // translation axis direction (unit, world space)
    Eigen::Vector3d world_normal;     // normal perpendicular to axis (unit, world space)
    Eigen::Vector3d world_bitangent;  // bitangent = axis x normal (unit, world space)
};
