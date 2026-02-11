#pragma once
#include <string>
#include <vector>
#include <Eigen/Core>

namespace gipc
{

/// A simple per-convex-component mesh structure
struct ObjConvexComponent
{
    std::vector<Eigen::Vector3d> vertices;           // local vertex positions
    std::vector<Eigen::Vector3i> faces;              // triangle faces (0-indexed into this->vertices)
};

/// Result of loading an .obj file
struct ObjMeshData
{
    std::vector<Eigen::Vector3d> vertices;           // all vertices
    std::vector<Eigen::Vector3i> faces;              // all triangle faces (0-indexed)
    std::vector<ObjConvexComponent> convex_components; // per-component data (if obj has 'o' groups)
};

/// Load a triangle mesh from a .obj file.
/// Returns true on success.
/// Supports multiple convex components (VHACD output with 'o convex_N' groups).
/// Only reads 'v' (vertex) and 'f' (face) lines. Face indices are converted to 0-based.
bool load_obj_mesh(const std::string& filepath, ObjMeshData& out);

}  // namespace gipc
