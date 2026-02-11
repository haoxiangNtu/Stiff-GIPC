#pragma once

#include <Eigen/Dense>

// Lightweight struct for joint constraint data on the host side.
// Separated from load_mesh.h to avoid heavy header dependencies in CUDA code.
struct JointConstraintHostInfo
{
    int parent_body_id;
    int child_body_id;
    enum class Type { Fixed, Revolute } type;
    int             num_points;  // 2 for revolute, 4 for fixed
    Eigen::Vector3d world_anchor[4];  // world-space positions at t=0
};
