//
// device_fem_data.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#ifndef __DEVICE_FEM_MESHES_CUH__
#define __DEVICE_FEM_MESHES_CUH__

//#include <cuda_runtime.h>
#include "gpu_eigen_libs.cuh"
#include <cstdint>
#include <body_boundary_type.h>
#include "Eigen/Eigen"
class device_TetraData
{
  public:
    double3* vertexes        = nullptr;
    double3* o_vertexes      = nullptr;
    double3* rest_vertexes   = nullptr;
    double3* targetVert      = nullptr;
    double3* temp_double3Mem = nullptr;
    double3* velocities      = nullptr;
    double3* xTilta          = nullptr;
    double3* fb              = nullptr;
    double3* totalForce      = nullptr;
    uint4*   tetrahedras     = nullptr;
    uint3*   triangles       = nullptr;

    uint2* tri_edges           = nullptr;
    uint2* tri_edge_adj_vertex = nullptr;

#ifdef USE_QUADRATIC_BENDING
    Eigen::Matrix4d* quad_bending_Q = nullptr;  // Precomputed Q matrices for quadratic bending
#endif

    uint32_t* targetIndex     = nullptr;
    uint4*    tempTetrahedras = nullptr;
    double*   volum           = nullptr;
    double*   area            = nullptr;

    double* lengthRate = nullptr;
    double* volumeRate = nullptr;
    double*   masses           = nullptr;
    int*    apply_gravity = nullptr;
    double*   tempDouble       = nullptr;

    __GEIGEN__::Matrix3x3d* DmInverses       = nullptr;
    __GEIGEN__::Matrix2x2d* triDmInverses    = nullptr;
    int*                    BoundaryType     = nullptr;

    double3*          shape_grads              = nullptr;
    BodyBoundaryType* body_id_to_boundary_type = nullptr;
    int*              point_id_to_body_id      = nullptr;
    int*              tet_id_to_body_id        = nullptr;

    // Per-body motor parameters: [axis_x, axis_y, axis_z, speed, strength, 0]
    // Packed as double6 (reusing __GEIGEN__::Matrix3x3d storage is tricky, so use raw double*)
    // Layout: body_id * 5 + [0..4] = {axis_x, axis_y, axis_z, speed, strength}
    double* body_motor_params = nullptr;

    // Collision exclusion matrix: flat NxN int array (N = collision_body_num).
    // collision_skip_matrix[i * collision_body_num + j] = 1 means skip collision between body i and j.
    // Symmetric: matrix[i*N+j] == matrix[j*N+i].
    int*    collision_skip_matrix = nullptr;
    int     collision_body_num    = 0;

    // [multi-FEM-bodyid] Per-body flag (size = collision_body_num).
    //   body_id_to_is_fem[i] = 1 -> body i is FEM (self-collision allowed)
    //   body_id_to_is_fem[i] = 0 -> body i is ABD (skip same-body checks)
    // Replaces the legacy "_bodyId == -1" sentinel that aliased all FEM
    // bodies into one matrix slot.
    int*    body_id_to_is_fem = nullptr;

    int*    ground_skip_body = nullptr;

    // [FEM-pin] hard-constraint pin GPU arrays (size = n_fem_pins).
    // Allocated only if pin count > 0. Apply kernel reads these to
    // directly project pinned FEM vertex positions to abd_anchor + offset.
    // [M1 substitution method] device-side pin registry
    //   d_fem_pin_fem_vertex[i]    FEM vertex idx (global)
    //   d_fem_pin_abd_body_id[i]   ABD body the vertex follows
    //   d_fem_pin_abd_local_pos[i] vertex's position in ABD rest frame.
    //                              kernel: world = q.t + R(q) * local_pos
    int*     d_fem_pin_fem_vertex   = nullptr;
    int*     d_fem_pin_abd_body_id  = nullptr;
    double3* d_fem_pin_abd_local_pos = nullptr;
    int*     d_fem_pin_abd_anchor   = nullptr;  // legacy/diag
    double3* d_fem_pin_rest_offset  = nullptr;  // legacy/diag
    int      n_fem_pins             = 0;
    // [M2 substitution method] per-vertex bitmap, size = vertexNum.
    //   is_pinned_vertex[v] = 1 iff v is a pinned FEM vertex.
    // Read by FEM elasticity / barrier kernels to skip writing
    // pinned row/col to the global Hessian.
    int*     is_pinned_vertex       = nullptr;
    // [M3.5 substitution method] per-vertex pin index map, size = vertexNum.
    //   vertex_to_pin_idx[v] = pin_idx if v is pinned, -1 otherwise.
    // Used by the chain-rule kernel to fetch body_id and lo (local_pos)
    // for a given pinned vertex without searching d_fem_pin_fem_vertex.
    int*     vertex_to_pin_idx      = nullptr;

    // BVH-skip #3: filtered face/edge index lists. bvh_active_face_idx[t] holds
    // the original face index of the t-th non-isolated face; same for edges.
    // Sized n_active_face / n_active_edge (<= surface count / edge count).
    int*    bvh_active_face_idx = nullptr;
    int     bvh_active_face_num = 0;
    int*    bvh_active_edge_idx = nullptr;
    int     bvh_active_edge_num = 0;

    int                   m_soft_num   = 0;
    int                   m_vertex_num = 0;
    std::vector<uint32_t> host_target_indices;
    std::vector<double3>  host_target_vertices;
    std::function<double3(double3 vertex, int step_id, double ipc_dt)> update_soft_constraint_functor =
        nullptr;

    // Stitch spring support: for soft constraint i, if stitch_paired_vertex[i] >= 0,
    // the target position is: position_of(stitch_paired_vertex[i]) + stitch_rest_offset[i]
    // This gives a spring with rest length = |stitch_rest_offset[i]|.
    // Bilateral coupling: gradient+Hessian on both FEM and ABD sides + cross-Hessian.
    std::vector<int>     stitch_paired_vertex;  // size = m_soft_num, -1 = use functor
    std::vector<double3> stitch_rest_offset;    // size = m_soft_num, rest offset from ABD to FEM
    std::vector<int>     stitch_abd_body_id;    // size = m_soft_num, ABD body id

    // GPU copies of stitch data (allocated in Malloc_DEVICE_MEM if softNum > 0)
    int*     d_stitch_paired_vertex = nullptr;  // GPU: ABD unique point id per spring (-1 if functor)
    double3* d_stitch_rest_offset   = nullptr;  // GPU: rest offset per spring
    int*     d_stitch_abd_body_id   = nullptr;  // GPU: ABD body id per spring

    // Per-frame callback to update body_motor_params on the GPU.
    // Called at the beginning of each timestep with (step_id, ipc_dt, body_motor_params_gpu, body_count).
    // The functor should cudaMemcpy the updated params to the GPU pointer.
    std::function<void(int step_id, double ipc_dt, double* body_motor_params_gpu, int body_count)>
        pre_step_functor = nullptr;
    int m_body_count = 0;  // total body count for pre_step_functor

    void update_soft_constraint_target_position(int step_id, double ipc_dt);


  public:
    device_TetraData() {}
    ~device_TetraData();
    void Malloc_DEVICE_MEM(const int& vertex_num,
                           const int& tetradedra_num,
                           const int& triangle_num,
                           const int& softNum,
                           const int& tri_edgeNum,
                           const int& bodyNum);
    void FREE_DEVICE_MEM();
};


#endif  // ! __DEVICE_FEM_MESHES_CUH__
