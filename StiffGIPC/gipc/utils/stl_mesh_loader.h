#pragma once
#include <string>
#include <vector>
#include <Eigen/Core>
#include <gipc/utils/obj_mesh_loader.h>  // reuse ObjMeshData

namespace gipc
{

/// Load a triangle mesh from a binary STL file.
/// Populates the same ObjMeshData structure (vertices + faces, single component).
/// Returns true on success.
bool load_stl_mesh(const std::string& filepath, ObjMeshData& out);

/// Load a triangle mesh from either .obj or .stl file (auto-detected by extension).
/// Returns true on success.
bool load_surface_mesh(const std::string& filepath, ObjMeshData& out);

}  // namespace gipc
