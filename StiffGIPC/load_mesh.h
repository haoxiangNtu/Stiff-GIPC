//
// load_mesh.h
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef FEM_MESH_H
#define FEM_MESH_H
//#include "Eigen/Eigen"
//#include "mIPC.h"
#include <vector>
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include "eigen_data.h"
#include <gipc/abd_fem_count_info.h>
#include <gipc/body_type.h>
#include <Eigen/Core>
#include <body_boundary_type.h>
#include <joint_constraint_host_info.h>
#include <joint_angle_control.h>

using namespace std;

class mesh_obj
{
  public:
    vector<double3> vertexes;
    vector<double3> normals;
    vector<uint3>   facenormals;
    vector<uint3>   faces;
    vector<uint2>   edges;
    int             vertexNum;
    int             faceNum;
    int             edgeNum;
    //void InitMesh(int type, double scale);
    bool load_mesh(const std::string& filename, double scale, double3 transform);
};

class tetrahedra_obj
{
  public:
    double         maxVolum = 0.0f;
    vector<bool>   isNBC;
    vector<bool>   isCollide;
    vector<double> volum;
    vector<double> area;
    vector<double> masses;
    vector<int>                    apply_gravity;
    double                         meanMass  = 0.0f;
    double                         meanVolum = 0.0f;
    vector<double3>                vertexes;
    vector<uint32_t>               partId;
    vector<vector<uint32_t>>       part_block;
    vector<int>                    partId_map_real;
    vector<int>                    real_map_partId;
    uint32_t                       part_offset = 0;
    vector<int>                    boundaryTypies;
    vector<uint4>                  tetrahedras;
    vector<uint3>                  triangles;
    vector<uint32_t>               targetIndex;
    vector<double3>                forces;
    vector<double3>                velocities;
    vector<double3>                d_velocities;
    vector<__GEIGEN__::Matrix3x3d> DM_inverse;
    vector<__GEIGEN__::Matrix2x2d> tri_DM_inverse;
    vector<__GEIGEN__::Matrix3x3d> constraints;

    vector<double3>    targetPos;
    vector<double3>    tetra_fiberDir;
    vector<double>     vert_youngth_modules;
    vector<double>     lengthRate;
    vector<double>     volumeRate;
    std::vector<uint2> tri_edges_adj_points;
    std::vector<uint2> tri_edges;


    vector<uint32_t> surfId2TetId;
    vector<uint3>    surface;

    vector<uint32_t> surfVerts;
    vector<uint2>    surfEdges;

    vector<double3> xTilta;
    vector<double3> dx_Elastic;
    vector<double3> acceleration;
    vector<double3> rest_V;
    vector<double3> V_prev;


    vector<vector<unsigned int>> vertNeighbors;
    vector<unsigned int>         neighborList;
    vector<unsigned int>         neighborStart;
    vector<unsigned int>         neighborNum;

    int D12x12Num        = 0;
    int D9x9Num          = 0;
    int D6x6Num          = 0;
    int D3x3Num          = 0;
    int vertexNum        = 0;
    int tetrahedraNum    = 0;
    int triangleNum      = 0;
    int softNum          = 0;
    int vertexOffset     = 0;
    int abd_vertexOffset = 0;
    int abd_tetOffset    = 0;


    double3 minTConer = make_double3(0, 0, 0);
    double3 maxTConer = make_double3(0, 0, 0);

    double3 minConer = make_double3(0, 0, 0);
    double3 maxConer = make_double3(0, 0, 0);

    gipc::ABDFEMCountInfo         abd_fem_count_info{};
    std::vector<BodyBoundaryType> body_id_to_is_fixed;
    std::vector<int>              point_id_to_body_id;
    std::vector<int>              tet_id_to_body_id;

    // Collision exclusion pairs: pairs of body IDs that should NOT collide.
    // Typically populated from URDF joint adjacency (parent-child link pairs).
    std::vector<std::pair<int, int>> collision_exclusion_pairs;

    // Per-body ground collision skip flags: body IDs listed here skip ground collision.
    std::vector<int> ground_collision_skip_body_ids;

    // [MAS-perm] Global vertex permutation: vertex_metis_to_input[engine_idx]
    // returns the input-mesh vertex index for the i-th engine-internal vertex.
    // Identity (== engine_idx) for ABD bodies and FEM bodies loaded with
    // preconditioner_type == 0 (no MAS reorder). Populated only when
    // simple_scene_importer.cpp captures sort_index from metis_sort.
    // Size grows with each load_mesh / load_geometry call.
    //
    // Used by SimEngine::get_vertex_positions / get_surface_faces /
    // set_vertex_positions_gpu to expose user-facing API in the original
    // input-mesh order (transparent unscramble), independent of whether
    // MAS preconditioner shuffles internal storage for PCG-banking.
    std::vector<int> vertex_metis_to_input;

    // Joint constraints between ABD bodies (populated by URDF importer).
    // World-space anchor positions are stored; material coords computed on GPU after init.
    std::vector<JointConstraintHostInfo> joint_constraints;

    // Per-revolute-joint angle control info (populated by URDF importer).
    // Stores axis/perpendicular directions needed to update the "angle control"
    // constraint point at runtime based on UI slider target angles.
    std::vector<JointAngleControlInfo> joint_angle_controls;

    // Prismatic joint constraints between ABD bodies (populated by URDF importer).
    std::vector<PrismaticJointHostInfo> prismatic_constraints;

    // Per-prismatic-joint driving control info (populated by URDF importer).
    std::vector<PrismaticDrivingControlInfo> prismatic_drive_controls;

    // Per-body motor parameters (indexed by body_id, size = abd_body_num)
    // motor_axis: the rotation axis in the body's local frame (normalized)
    // motor_speed: angular velocity in rad/s (0 means use global default)
    // motor_strength: penalty strength (0 means use global default)
    struct BodyMotorInfo
    {
        double axis_x = 1.0, axis_y = 0.0, axis_z = 0.0;  // default UnitX
        double speed    = 0.0;  // 0 = use global default
        double strength = 0.0;  // 0 = use global default
    };
    std::vector<BodyMotorInfo> body_motor_infos;

    tetrahedra_obj();
    int getVertNeighbors();
    //void InitMesh(int type, double scale);
    bool load_tetrahedraMesh(const std::string& filename,
                             double             scale,
                             //   double             youngth_module,
                             double3        position_offset,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);

    bool load_parts(const std::string& filename);


    bool load_tetrahedraMesh(const std::string&     filename,
                             const Eigen::Matrix4d& transform,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);

    bool load_tetrahedraMesh(const std::string&     filename,
                             const Eigen::Matrix4d& transform,
                             double                 youngth_module,
                             gipc::BodyType body_type = gipc::BodyType::FEM,
                             BodyBoundaryType body_boundary_type = BodyBoundaryType::Free);


    bool load_triMesh(const std::string& filename, double scale, double3 transform, int boundaryType);
    bool load_triMesh(const std::string& filename, const Eigen::Matrix4d& transform, int boundaryType);

    /// Load a surface mesh (.obj) as an ABD body using native surface integrals.
    /// Mass, volume, and gravity are computed directly from the closed triangle
    /// mesh via Divergence theorem — no tetrahedralization or fan-tet hack.
    /// The surface vertices are still added for collision detection.
    bool load_surfaceMesh_ABD(const std::string&     obj_filename,
                              const Eigen::Matrix4d& transform,
                              double                 youngth_module   = 1e7,
                              BodyBoundaryType       boundary_type    = BodyBoundaryType::Free);

    /// Per-body surface mesh data for bodies loaded from triangle meshes.
    /// Used by ABDSystem to compute mass/volume/gravity via surface integrals
    /// instead of the tet-based path.
    struct SurfaceMeshBodyInfo
    {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<Eigen::Vector3i> triangles;
        /// Per-face orientation override (libuipc-style). Same semantics as
        /// `ABDSurfaceMeshBody::orient` — empty = use winding, non-empty = one
        /// {-1,0,+1} per face used to flip integration-time normal sign.
        /// Populated via `SimEngine::set_abd_body_face_orient` before finalize.
        std::vector<int>             orient;
        int body_id          = -1;
        int vert_global_offset = 0;  // global index of this body's first vertex
    };
    std::vector<SurfaceMeshBodyInfo> surface_mesh_bodies;


    bool load_animation(const std::string& filename, double scale, double3 transform);
    bool load_tetrahedraMesh_IPC_TetMesh(const std::string& filename,
                                         double             scale,
                                         double3            position_offset,
                                         bool               isfixed);
    //void load_test(double scale, int num = 1);
    void getSurface();
    bool output_tetrahedraMesh(const std::string& filename);

    bool output_tetrahedraMesh_reorder(const std::string&      filename,
                                       const vector<uint32_t>& new_order,
                                       const vector<uint32_t>& order_map);

  private:
    bool abd_load_phase         = true;
    int  current_body_id        = 0;
    int  current_body_point_num = 0;
    void begin_load_body(const std::string& filename,
                         gipc::BodyType     body_type,
                         BodyBoundaryType   body_is_fixed);
    void set_body_tet_num(gipc::BodyType body_type, int tet_num);
    void set_body_tri_num(gipc::BodyType body_type, int tri_num);
    void set_body_point_num(gipc::BodyType body_type, int point_num);
};

#endif  // !FEM_MESH.H