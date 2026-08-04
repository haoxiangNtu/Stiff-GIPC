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
#include <vector>
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
int get_bvh_stack_capacity();
void set_bvh_audit(int v);  // [audit-gate] enable the per-pop stack-depth probe (STIFF_STACK_DIAG)
void set_ee_vloc(const int* p);

// Validation-only traversal census.  The hot kernels contain no census code
// unless the translation unit is built with
// STIFF_BVH_TRAVERSAL_AUDIT_BUILD; the normal simulator binary is therefore
// unaffected.  A "primitive_test" is a leaf pair that survived body/env/
// adjacency filtering and reached exact PT/EE classification (or CCD emit).
struct BvhTraversalAudit
{
    unsigned long long queries;
    unsigned long long node_pops;
    unsigned long long overlapping_children;
    unsigned long long primitive_tests;
};
constexpr int kBvhAuditBodyCapacity = 128;
void set_bvh_traversal_audit(int v);
void set_bvh_pair_work_audit(int v);
void set_bvh_traversal_margin_scale(double scale);
void reset_bvh_traversal_audit();
void get_bvh_traversal_audit(BvhTraversalAudit out[4]);
void get_bvh_traversal_body_audit(
    BvhTraversalAudit out[4][kBvhAuditBodyCapacity]);
void get_bvh_traversal_pair_primitive_audit(
    unsigned long long out[4][kBvhAuditBodyCapacity]
                               [kBvhAuditBodyCapacity]);
void print_bvh_traversal_audit();

// Validation-only VF-DCD body-pair raw-candidate cache.  All setters become
// no-ops unless the owning GIPC instance explicitly arms the cache.
void set_bvh_vf_pair_cache(const unsigned char* pair_valid,
                           const int*           pair_index,
                           int                  body_count,
                           int                  pair_count,
                           int2*                candidates,
                           uint32_t*            counts,
                           int                  segment_capacity,
                           int*                 overflow);
void set_bvh_vf_pair_front(uint32_t* front_nodes,
                           uint32_t* front_counts,
                           int       front_capacity,
                           int*      front_overflow);
void rebuild_bvh_vf_pair_front(const Node* nodes,
                               const int*  node_body,
                               int         primitive_count,
                               cudaStream_t stream = 0);
void reset_bvh_vf_pair_cache_counts(cudaStream_t stream = 0);
void replay_bvh_vf_pair_cache(const double3* vertexes,
                              const uint3*   faces,
                              uint32_t*      cp_num,
                              int*           mat_index,
                              int4*          collision_pair,
                              int4*          ccd_collision_pair,
                              double         d_hat,
                              cudaStream_t   stream = 0);

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
    uint32_t  vert_number = 0;
    double3*  _vertexes = nullptr;
    AABB*     _bvs = nullptr;
    AABB*     _tempLeafBox = nullptr;
    Node*     _nodes = nullptr;
    uint64_t* _MChash = nullptr;
    uint32_t* _indices = nullptr;
    int4*     _collisionPair = nullptr;
    int4*     _ccd_collisionPair = nullptr;
    uint32_t* _cpNum = nullptr;
    int*      _MatIndex = nullptr;
    uint32_t* _flags = nullptr;
    AABB      scene;
    int*      _btype = nullptr;
    int*      _bodyId = nullptr;
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
    // Uniform collision body for each subtree, or -1 for a mixed subtree.
    // Allocated only by the experimental body-pair cache path.
    int*       m_node_body           = nullptr;
    const int* m_prim_body           = nullptr;
    // Validation candidate: maximum ORIGINAL primitive index in each subtree.
    // The default EE ownership rule emits only obj_idx >= self_eid; this bound
    // lets the range-pruned traversal discard an entire all-lower subtree
    // while preserving exactly that directed-pair contract.
    uint32_t*  m_node_max_element    = nullptr;

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

    // Validation candidate: each point-swapped per-env scratch allocation
    // owns an independent refit generation.  One scalar state on lbvh would
    // see slot0/slot1/slot2 as a topology change on every host iteration and
    // therefore rebuild forever.  The node allocation is the stable slot key;
    // active-list identity/count still invalidate that slot if another env is
    // mapped onto it.
    struct RefitTopologyState
    {
        Node*      nodes_identity  = nullptr;
        const int* active_identity = nullptr;
        int        number          = 0;
        int        since_rebuild   = 0;
        bool       topology_ready  = false;
        unsigned long long capture_id = 0;
        unsigned long long reuses      = 0;
        unsigned long long rebuilds    = 0;
    };
    std::vector<RefitTopologyState> m_refit_states;
    void invalidateRefitTopology();

  public:
    lbvh() {}
    ~lbvh();
    void MALLOC_DEVICE_MEM(const int& number, bool allocate_node_max = false);
    void FREE_DEVICE_MEM();
    //void Construct();
};


class lbvh_f : public lbvh
{
  public:
    uint32_t  face_number = 0;
    uint3*    _faces = nullptr;
    uint32_t* _surfVerts = nullptr;

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
    double3* _rest_vertexes = nullptr;
    uint32_t edge_number = 0;
    uint2*   _edges = nullptr;

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
