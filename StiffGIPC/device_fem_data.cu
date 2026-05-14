//
// device_fem_data.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "cuda_tools/cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num,
                                         const int& tetradedra_num,
                                         const int& triangle_num,
                                         const int& softNum,
                                         const int& tri_edgeNum,
                                         const int& bodyNum)
{
    m_vertex_num   = vertex_num;
    int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
    CUDA_SAFE_CALL(cudaMalloc((void**)&vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&o_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&velocities, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rest_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp_double3Mem, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&xTilta, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&fb, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&totalForce, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&shape_grads, vertex_num * sizeof(double3)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&tetrahedras, tetradedra_num * sizeof(uint4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tempTetrahedras, tetradedra_num * sizeof(uint4)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edges, tri_edgeNum * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edge_adj_vertex, tri_edgeNum * sizeof(uint2)));

#ifdef USE_QUADRATIC_BENDING
    CUDA_SAFE_CALL(cudaMalloc((void**)&quad_bending_Q, tri_edgeNum * sizeof(Eigen::Matrix4d)));
#endif

    CUDA_SAFE_CALL(cudaMalloc((void**)&volum, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&masses, vertex_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lengthRate, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&volumeRate, tetradedra_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&apply_gravity, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tempDouble, maxNumbers * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&BoundaryType, vertex_num * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemset(BoundaryType, 0, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&DmInverses,
                              tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d)));

    m_soft_num = softNum;
    CUDA_SAFE_CALL(cudaMalloc((void**)&targetIndex, softNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&targetVert, softNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triDmInverses,
                              triangle_num * sizeof(__GEIGEN__::Matrix2x2d)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&area, triangle_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triangles, triangle_num * sizeof(uint4)));

    
    CUDA_SAFE_CALL(cudaMalloc((void**)&body_id_to_boundary_type, bodyNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&point_id_to_body_id, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tet_id_to_body_id, tetradedra_num * sizeof(int)));
    // Per-body motor params: 5 doubles per body [axis_x, axis_y, axis_z, speed, strength]
    CUDA_SAFE_CALL(cudaMalloc((void**)&body_motor_params, bodyNum * 5 * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(body_motor_params, 0, bodyNum * 5 * sizeof(double)));

    // Collision exclusion matrix: NxN flat array (default = all zeros = no exclusions)
    collision_body_num = bodyNum;
    if(bodyNum > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&collision_skip_matrix, bodyNum * bodyNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(collision_skip_matrix, 0, bodyNum * bodyNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&ground_skip_body, bodyNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(ground_skip_body, 0, bodyNum * sizeof(int)));
        // [multi-FEM-bodyid] per-body FEM flag table
        CUDA_SAFE_CALL(cudaMalloc((void**)&body_id_to_is_fem, bodyNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(body_id_to_is_fem, 0, bodyNum * sizeof(int)));
    }

    // Stitch spring GPU arrays (bilateral coupling)
    if(softNum > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stitch_paired_vertex, softNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stitch_rest_offset, softNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_stitch_abd_body_id, softNum * sizeof(int)));
        // -1 means "no stitch partner"; prevents garbage from triggering stitch logic
        CUDA_SAFE_CALL(cudaMemset(d_stitch_paired_vertex, 0xFF, softNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(d_stitch_rest_offset, 0, softNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(d_stitch_abd_body_id, 0xFF, softNum * sizeof(int)));
    }
}

device_TetraData::~device_TetraData()
{
    FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM()
{
    CUDA_SAFE_CALL(cudaFree(vertexes));
    CUDA_SAFE_CALL(cudaFree(o_vertexes));
    CUDA_SAFE_CALL(cudaFree(temp_double3Mem));
    CUDA_SAFE_CALL(cudaFree(velocities));
    CUDA_SAFE_CALL(cudaFree(rest_vertexes));
    CUDA_SAFE_CALL(cudaFree(xTilta));
    CUDA_SAFE_CALL(cudaFree(fb));
    CUDA_SAFE_CALL(cudaFree(apply_gravity));
    CUDA_SAFE_CALL(cudaFree(shape_grads));
    CUDA_SAFE_CALL(cudaFree(tetrahedras));
    CUDA_SAFE_CALL(cudaFree(tempTetrahedras));
    CUDA_SAFE_CALL(cudaFree(volum));
    CUDA_SAFE_CALL(cudaFree(masses));
    CUDA_SAFE_CALL(cudaFree(lengthRate));
    CUDA_SAFE_CALL(cudaFree(volumeRate));
    CUDA_SAFE_CALL(cudaFree(DmInverses));
    CUDA_SAFE_CALL(cudaFree(tempDouble));
    CUDA_SAFE_CALL(cudaFree(BoundaryType));

    CUDA_SAFE_CALL(cudaFree(totalForce));
    CUDA_SAFE_CALL(cudaFree(targetIndex));
    CUDA_SAFE_CALL(cudaFree(targetVert));
    CUDA_SAFE_CALL(cudaFree(triDmInverses));
    CUDA_SAFE_CALL(cudaFree(area));
    CUDA_SAFE_CALL(cudaFree(triangles));

    CUDA_SAFE_CALL(cudaFree(tri_edges));
    CUDA_SAFE_CALL(cudaFree(tri_edge_adj_vertex));

#ifdef USE_QUADRATIC_BENDING
    CUDA_SAFE_CALL(cudaFree(quad_bending_Q));
#endif

    CUDA_SAFE_CALL(cudaFree(body_id_to_boundary_type));
    CUDA_SAFE_CALL(cudaFree(point_id_to_body_id));
    CUDA_SAFE_CALL(cudaFree(tet_id_to_body_id));
    CUDA_SAFE_CALL(cudaFree(body_motor_params));
    CUDA_SAFE_CALL(cudaFree(collision_skip_matrix));
    CUDA_SAFE_CALL(cudaFree(ground_skip_body));
    CUDA_SAFE_CALL(cudaFree(bvh_active_face_idx));
    CUDA_SAFE_CALL(cudaFree(bvh_active_edge_idx));

    // Stitch spring GPU arrays
    CUDA_SAFE_CALL(cudaFree(d_stitch_paired_vertex));
    CUDA_SAFE_CALL(cudaFree(d_stitch_rest_offset));
    CUDA_SAFE_CALL(cudaFree(d_stitch_abd_body_id));

    // [Hybrid mesh] per-tet ABD body assignment.
    CUDA_SAFE_CALL(cudaFree(d_tet_to_abd_body));
}

void device_TetraData::update_soft_constraint_target_position(int step_id, double ipc_dt)
{
    // Call pre-step functor to update body_motor_params if set
    if(pre_step_functor && m_body_count > 0)
    {
        pre_step_functor(step_id, ipc_dt, body_motor_params, m_body_count);
    }

    if(m_soft_num < 1)
        return;

    std::vector<double3> host_vertexes(m_vertex_num);
    CUDA_SAFE_CALL(cudaMemcpy(
        host_vertexes.data(), vertexes, m_vertex_num * sizeof(double3), cudaMemcpyDeviceToHost));

    for(int i = 0; i < m_soft_num; i++)
    {
        // Bilateral stitch spring: target is computed dynamically in GPU kernel
        // from current ABD vertex position + rest_offset.  No pre-computed target needed.
        if(!stitch_paired_vertex.empty() && stitch_paired_vertex[i] >= 0)
        {
            // Still update targetVert as fallback (not used by bilateral kernel,
            // but keeps the array valid for debugging / energy printout).
            auto abd_pos = host_vertexes[stitch_paired_vertex[i]];
            auto off     = stitch_rest_offset[i];
            host_target_vertices[i] = make_double3(
                abd_pos.x + off.x, abd_pos.y + off.y, abd_pos.z + off.z);
        }
        else if(update_soft_constraint_functor != nullptr)
        {
            host_target_vertices[i] = update_soft_constraint_functor(
                host_vertexes[host_target_indices[i]], step_id, ipc_dt);
        }
        else
        {
            host_target_vertices[i] = host_vertexes[host_target_indices[i]];
        }
    }

    CUDA_SAFE_CALL(cudaMemcpy(targetVert,
                              host_target_vertices.data(),
                              m_soft_num * sizeof(double3),
                              cudaMemcpyHostToDevice));
}
