//
// mlbvh.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _MLBVH_CUH_
#define _MLBVH_CUH_
#include <cstdint>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Set the pair-emit overflow caps (logical buffer capacities). Emits whose index
// reaches the cap are redirected to a trash slot (buffers allocated with +1), so
// detection can never write out of bounds; the host then grows + redoes.
void set_emit_caps(int dcd_cap, int ccd_cap);
void set_ee_nodedup(int v);
void set_ee_canon(int v);
void set_ee_nomollify(int v);
void set_ee_trace(int v);
void set_ee_tgt(int a, int b);
void set_bvh_envmajor(int v);
void set_ee_detgate(int v);
void set_bvh_envpart(int v);  // [env-part B] enable env-id subtree pruning in broad-phase
void set_self_p2g(const int* p);  // [perenv-par] per-vertex env id; cross-env self-collision pairs skipped at emission (null = off)
// [multi-env subscene, v0.6.7 API] alias of set_self_p2g — points the broad-phase
// env filter at a device array of per-vertex env ids. nullptr disables.
void mlbvh_set_vertex_env_id(const int* d_vertex_env_id);
struct Node;
void computeNodeEnv(int* node_env, const Node* _nodes, const int* prim_env, uint32_t* flags, int number, cudaStream_t stream = 0);
void reset_max_stack();
int get_max_stack();
void set_ee_vloc(const int* p);

struct AABB
{
  public:
    double3             upper;
    double3             lower;
    __host__ __device__ AABB();
    __host__ __device__ void combines(const double& x, const double& y, const double& z);
    __host__ __device__ void    combines(const double& x,
                                         const double& y,
                                         const double& z,
                                         const double& xx,
                                         const double& yy,
                                         const double& zz);
    __host__ __device__ void    combines(const AABB& aabb);
    __host__ __device__ double3 center();
};

struct Node
{
  public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class lbvh
{
  public:
    uint32_t  vert_number;
    double3*  _vertexes;
    AABB*     _bvs;
    AABB*     _tempLeafBox;
    Node*     _nodes;
    uint64_t* _MChash;
    uint32_t* _indices;
    int4*     _collisionPair;
    int4*     _ccd_collisionPair;
    uint32_t* _cpNum;
    int*      _MatIndex;
    uint32_t* _flags;
    AABB      scene;
    int*      _btype;
    int*      _bodyId;
    int*      _collision_skip_matrix = nullptr;  // NxN exclusion matrix (null if none)
    int       _collision_body_count = 0;        // N dimension of exclusion matrix
    // [multi-FEM-bodyid] per-body flag table (size = _collision_body_count).
    // Set by GIPC::initBVH after wiring d_tetMesh.body_id_to_is_fem.
    // 1 = FEM body (allow self-collision), 0 = ABD body (skip self).
    int*      _body_id_to_is_fem    = nullptr;

    // BVH-skip #3: filter input faces/edges. _active_idx[t] = original face/edge index
    // for t-th active leaf (i.e. excludes isolated body's faces/edges entirely from
    // sort + tree build). face_number_active = active count. When set, Construct()
    // operates on n_active leaves instead of full face_number.
    int*      _active_idx           = nullptr;
    int       face_number_active    = 0;
    // [env-det] per-prim env id (indexed by global prim index) for env-major Morton (merged path).
    const int* m_prim_env           = nullptr;
    const int* m_prim_localid       = nullptr;  // [env-det] env-local prim rank for Morton low bits
    const double3* m_env_offset      = nullptr;  // [env-det] LIVE per-vertex env offset (read at build)
    const uint32_t* m_prim_v0        = nullptr;  // [env-det] per-prim first vertex (static)
    // [env-part B] per-NODE env id (size 2N-1): leaves = prim env, internal = uniform env or -1 (mixed).
    // Computed when env-major. Lets the broad-phase prune other-env subtrees by env-id ⇒ no cross-env
    // candidates (fast) while AABBs stay LOCAL (overlap mirror ⇒ bit-identical). Allocated in MALLOC.
    int*       m_node_env            = nullptr;

    // [perenv-parallel #2] cub radix-sort scratch (per instance / per pool slot, pre-allocated):
    // the per-env active-path Morton sort must do NO cudaMalloc/cudaFree — thrust's internal
    // alloc/free are device-wide syncs that serialized the per-env pool streams (the detect
    // kernels could never overlap). Swapped by GIPC's pool swapIn alongside the other scratch.
    void*     _sort_tmp       = nullptr;   // cub temp storage
    size_t    _sort_tmp_bytes = 0;         // byte capacity of _sort_tmp
    uint64_t* _mch_alt        = nullptr;   // out-of-place key buffer
    uint32_t* _idx_alt        = nullptr;   // out-of-place value buffer
    int       _sort_cap       = 0;         // element capacity of the alt buffers
    void ensure_sort_scratch(int N);       // (re)alloc to fit N (syncing malloc; pre-size pool slots)

  public:
    lbvh() {}
    ~lbvh();
    void MALLOC_DEVICE_MEM(const int& number);
    void FREE_DEVICE_MEM();
    //void Construct();
};


class lbvh_f : public lbvh
{
  public:
    uint32_t  face_number;
    uint3*    _faces;
    uint32_t* _surfVerts;

  public:
    void   init(int*       _bodyID,
                int*       _btype,
                double3*   _mVerts,
                uint3*     _mFaces,
                uint32_t*  _mSurfVert,
                int4*      _mCollisonPairs,
                int4*      _ccd_mCollisonPairs,
                uint32_t*  _mcpNum,
                int*       _mMatIndex,
                const int& faceNum,
                const int& vertNum,
                int*       collision_skip_matrix = nullptr,
                int        collision_body_count  = 0);
    double Construct(cudaStream_t stream = 0);
    AABB*  getSceneSize();
    double ConstructFullCCD(const double3* moveDir, const double& alpha, cudaStream_t stream = 0,
                            const double* alpha_dev = nullptr);
    void   SelfCollitionDetect(double dHat, cudaStream_t stream = 0);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                 cudaStream_t stream = 0, const double* alpha_dev = nullptr);
};

class lbvh_e : public lbvh
{
  public:
    double3* _rest_vertexes;
    uint32_t edge_number;
    uint2*   _edges;

  public:
    void   init(int*       _bodyID,
                int*       _btype,
                double3*   _mVerts,
                double3*   _rest_vertexes,
                uint2*     _mEdges,
                int4*      _mCollisonPairs,
                int4*      _ccd_mCollisonPairs,
                uint32_t*  _mcpNum,
                int*       _mMatIndex,
                const int& edgeNum,
                const int& vertNum,
                int*       collision_skip_matrix = nullptr,
                int        collision_body_count  = 0);
    double Construct(cudaStream_t stream = 0);
    double ConstructFullCCD(const double3* moveDir, const double& alpha, cudaStream_t stream = 0,
                            const double* alpha_dev = nullptr);
    void   SelfCollitionDetect(double dHat, cudaStream_t stream = 0);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha,
                                 cudaStream_t stream = 0, const double* alpha_dev = nullptr);
};

__device__ void _d_PP(const double3& v0, const double3& v1, double& d);

__device__ void _d_PT(const double3& v0,
                      const double3& v1,
                      const double3& v2,
                      const double3& v3,
                      double&        d);

__device__ void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d);

__device__ void _d_EE(const double3& v0,
                      const double3& v1,
                      const double3& v2,
                      const double3& v3,
                      double&        d);

__device__ void _d_EEParallel(const double3& v0,
                              const double3& v1,
                              const double3& v2,
                              const double3& v3,
                              double&        d);

__device__ double _compute_epx(const double3& v0,
                               const double3& v1,
                               const double3& v2,
                               const double3& v3);

#endif