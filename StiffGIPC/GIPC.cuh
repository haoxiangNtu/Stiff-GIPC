//
// GIPC.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _GIPC_H_
#define _GIPC_H_
#include <memory>
#include "mlbvh.cuh"
#include "device_fem_data.cuh"

#include "PCG_SOLVER.cuh"
#include <gipc/abd_fem_count_info.h>
namespace gipc
{
class ABDSimData;
class ABDSystem;
class GlobalLinearSystem;
}  // namespace gipc

class GIPC
{
  public:
    bool      animation      = false;
    double3*  _vertexes      = nullptr;
    double3*  _rest_vertexes = nullptr;
    uint3*    _faces         = nullptr;
    uint2*    _edges         = nullptr;
    uint32_t* _surfVerts     = nullptr;


    double3*  targetVert  = nullptr;
    uint32_t* targetInd   = nullptr;
    uint32_t  softNum     = 0;
    uint32_t  triangleNum = 0;

    // Bilateral stitch spring GPU pointers (point into device_TetraData's d_stitch_* arrays)
    int*     m_d_stitch_paired_vertex = nullptr;
    double3* m_d_stitch_rest_offset   = nullptr;  // [stitch local-frame fix]
                                                  // After finalize, contents
                                                  // are in ABD body rest
                                                  // frame (R_finalize^T * world).
                                                  // Kernel uses target =
                                                  // anchor_world + R_now * lo.
    int*     m_d_stitch_abd_body_id   = nullptr;
    // [stitch local-frame fix] Pointer to ABDSystem's per-body q array
    // (Vector12 each: t.xyz, axis_x.xyz, axis_y.xyz, axis_z.xyz).
    // Wired in finalize from m_abd_sim_data->device.body_id_to_q.
    void*    m_d_abd_body_q           = nullptr;  // void* to avoid Vector12 forward-decl pain


    double3* _moveDir = nullptr;
    lbvh_f   bvh_f;
    lbvh_e   bvh_e;

    PCG_Data pcg_data;

    int4*     _collisonPairs     = nullptr;
    int4*     _ccd_collisonPairs = nullptr;
    uint32_t* _cpNum             = nullptr;
    int*      _MatIndex          = nullptr;
    uint32_t* _close_cpNum       = nullptr;
    // On-demand reduction scratch: reductions launch ceil(count/default_threads)
    // blocks each writing one double. pcg_data.squeue is only sized to the mesh
    // (max(vertexNum,tetra)), so reductions over a COLLISION/CCD PAIR count
    // (h_cpNum / h_ccd_cpNum, up to MAX_*_PAIRS) overflow it. This buffer grows
    // to ceil(count/default_threads) on demand so no pair-count reduction can
    // ever overflow; after warmup the capacity stabilizes (no further realloc).
    double*   m_reduce_scratch   = nullptr;
    size_t    m_reduce_cap       = 0;

    uint32_t* _environment_collisionPair = nullptr;

    uint32_t* _closeConstraintID  = nullptr;
    double*   _closeConstraintVal = nullptr;

    int4*   _closeMConstraintID  = nullptr;
    double* _closeMConstraintVal = nullptr;

    uint32_t* _gpNum       = nullptr;
    uint32_t* _close_gpNum = nullptr;
    //uint32_t* _cpNum;
    uint32_t h_cpNum[5]  = {0, 0, 0, 0, 0};
    uint32_t h_ccd_cpNum = 0;
    uint32_t h_gpNum     = 0;

    uint32_t h_close_cpNum = 0;
    uint32_t h_close_gpNum = 0;

    double   Kappa         = 0.0;
    double   dHat          = 0.0;
    double   fDhat         = 0.0;
    double   bboxDiagSize2 = 0.0;
    double   relative_dhat = 0.0;
    // Absolute contact distance (meters). >0 overrides the scene-bbox-derived
    // dHat so the contact thickness does NOT inflate with scene/env count.
    double   absolute_dhat = 0.0;
    double   dTol          = 0.0;
    double   minKappaCoef  = 0.0;
    double   IPC_dt        = 0.0;
    double3  gravity       = make_double3(0, -9.8, 0);
    double   Step          = 0.0;
    double   meanMass      = 0.0;
    double   meanVolumn    = 0.0;
    double3* _groundNormal = nullptr;
    double*  _groundOffset = nullptr;

    double3 ground_normal_cfg = make_double3(0, 1, 0);
    double  ground_offset_cfg = -1.0;
    std::string assets_dir_cfg;

    // for friction
    double*                 lambda_lastH_scalar  = nullptr;
    double2*                distCoord            = nullptr;
    __GEIGEN__::Matrix3x2d* tanBasis             = nullptr;
    int4*                   _collisonPairs_lastH = nullptr;
    uint32_t                h_cpNum_last[5]      = {0, 0, 0, 0, 0};
    int*                    _MatIndex_last       = nullptr;

    double*   lambda_lastH_scalar_gd  = nullptr;
    uint32_t* _collisonPairs_lastH_gd = nullptr;
    uint32_t  h_gpNum_last;

    uint32_t vertexNum      = 0;
    uint32_t surf_vertexNum = 0;
    uint32_t edge_Num       = 0;
    uint32_t tri_edge_num   = 0;
    uint32_t surface_Num    = 0;
    uint32_t tetrahedraNum  = 0;

    GIPCTripletMatrix gipc_global_triplet;
    AABB     SceneSize;
    int      MAX_COLLITION_PAIRS_NUM     = 0;
    int      MAX_CCD_COLLITION_PAIRS_NUM = 0;
    // internal Hessian-triplet margin (set from cfg.triplet_internal_margin; 32
    // = historical hardcoded value). The dominant per-env triplet over-reserve.
    double   m_triplet_internal_margin   = 32.0;

    double RestNHEnergy       = 0.0;
    double animation_subRate  = 0.0;
    double animation_fullRate = 0.0;


    double bendStiff = 0.0;


    double density                 = 0.0;
    double YoungModulus            = 0.0;
    double PoissonRate             = 0.0;
    double lengthRateLame          = 0.0;
    double volumeRateLame          = 0.0;
    double lengthRate              = 0.0;
    double volumeRate              = 0.0;
    double frictionRate            = 0.0;
    double gd_frictionRate         = 0.0;
    double clothThickness          = 0.0;
    double clothYoungModulus       = 0.0;
    double bendYoungModulus        = 0.0;
    double stretchStiff            = 0.0;
    double shearStiff              = 0.0;
    double strainRate              = 0.0;
    double clothDensity            = 0.0;
    double softMotionRate          = 0.0;
    double Newton_solver_threshold = 0.0;
    double pcg_threshold           = 0.0;

    gipc::ABDFEMCountInfo abd_fem_count_info{};
    int                   num_joint_constraints = 0;  // set from tetMesh

    bool m_skip_all_collision = false;

    // Semi-implicit early exit (ref: https://arxiv.org/abs/2512.12151, Algorithm 1)
    bool   semi_implicit_enabled  = false;
    double semi_implicit_beta_tol = 1e-3;
    int    semi_implicit_min_iter = 1;

    int    newton_iter_cap = 1000;

    int* _point_body_id     = nullptr;
    int* _ground_skip_body  = nullptr;
    int  _ground_body_count = 0;

    // [multi-FEM-bodyid] Per-body FEM flag table (size = _collision_body_num
    // when initBVH() is called with a non-null table). Owned by
    // device_TetraData; only borrowed pointer here.
    int* _body_id_to_is_fem = nullptr;

    // [M2 substitution method] Per-vertex pinned-mask (size = vertexNum).
    // is_pinned_vertex[v] == 1 if FEM vertex v is hard-pinned to ABD.
    // Used by elasticity/barrier kernels to skip writing pinned vertex's
    // row/col to the global Hessian (so PCG sees them as disconnected DOFs).
    int* m_d_is_pinned_vertex = nullptr;

    // Auxiliary stream for overlapping bvh_e collision detection with
    // bvh_f (default stream). Created lazily; destroyed in dtor.
    cudaStream_t m_aux_stream = nullptr;

  public:
    GIPC();
    ~GIPC();
    uint64_t getHashCode(double3 p, uint32_t i);
    void     build_gipc_system(device_TetraData& tet);

    void MALLOC_DEVICE_MEM();

    void tempMalloc_closeConstraint();
    void tempFree_closeConstraint();

    void FREE_DEVICE_MEM();
    // Returns a reduction scratch buffer guaranteed to hold ceil(count/default_threads)
    // doubles. Grows on demand; use for ANY reduction whose element count is a
    // collision/CCD pair count instead of pcg_data.squeue (which is mesh-sized).
    double* ensure_reduce_scratch(int count);
    void initBVH(int* _btype, int* _bodyId, int* _collision_skip_matrix = nullptr, int _collision_body_count = 0);
    void init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer, double buffScale = 1);

    void buildCP();
    void buildFullCP(const double& alpha);
    void buildBVH();

    AABB* calcuMaxSceneSize();

    void buildBVH_FULLCCD(const double& alpha);
    void step_forward(device_TetraData& TetMesh, double alpha = 1.0, bool move_boundary = false);


    void GroundCollisionDetect();
    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
    void calBarrierHessian();
    void calBarrierGradient(double3* _gradient, double mKap);
    void calFrictionHessian(device_TetraData& TetMesh);
    void calFrictionGradient(double3* _gradient, device_TetraData& TetMesh);

    int calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type = 0);
    float computeGradientAndHessian(device_TetraData& TetMesh);
    void  computeGroundGradientAndHessian(double3* _gradient);

    void partitionContactHessian();

    void  computeGroundGradient(double3* _gradient, double mKap);
    void computeSoftConstraintGradientAndHessian(double3* _gradient,
                                                 int global_hessian_fem_offset);

    void getTotalForce(double3* _gradient, double3* _gradient2);

    void   computeSoftConstraintGradient(double3* _gradient);
    double computeEnergy(device_TetraData& TetMesh);

    double Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh);

    double ground_largestFeasibleStepSize(double slackness, double* mqueue);

    double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);

    double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);

    double cfl_largestSpeed(double* mqueue);

    bool lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha);
    void postLineSearch(device_TetraData& TetMesh, double alpha);

    bool checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh);
    bool isIntersected(device_TetraData& TetMesh);
    bool checkGroundIntersection();

    void computeCloseGroundVal();
    void computeSelfCloseVal();

    bool checkCloseGroundVal();
    bool checkSelfCloseVal();

    double2 minMaxGroundDist();
    double2 minMaxSelfDist();

    void updateVelocities(device_TetraData& TetMesh);
    void updateBoundary(device_TetraData& TetMesh, double alpha);
    void updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha, int fid);
    void updateBoundary2(device_TetraData& TetMesh);
    void computeXTilta(device_TetraData& TetMesh, const double& rate);

    void initKappa(device_TetraData& TetMesh);
    void suggestKappa(double& kappa);
    void upperBoundKappa(double& kappa);
    int  solve_subIP(device_TetraData& TetMesh,
                     double&           time0,
                     double&           time1,
                     double&           time2,
                     double&           time3,
                     double&           time4);
    void IPC_Solver(device_TetraData& TetMesh);
    void sortMesh(device_TetraData& TetMesh, int updateVertNum);
    void buildFrictionSets();

    void create_LinearSystem(device_TetraData& tet);

  public:
    void                                      init_abd_system();
    /// Set up surface mesh body data in ABDSystem before init_abd_system().
    void                                      setup_surface_mesh_bodies(class tetrahedra_obj& tetMesh);
    // Note: defined in gipc.cu which has load_mesh.h included
    void                                      init_joint_constraints_from_mesh(class tetrahedra_obj& tetMesh);
    /// Update revolute driving joint target angles from tetMesh.joint_angle_controls.
    /// Call each frame before IPC_Solver when using interactive joint control.
    void                                      update_joint_angle_targets_from_mesh(class tetrahedra_obj& tetMesh);
    std::unique_ptr<gipc::ABDSimData>         m_abd_sim_data;
    std::unique_ptr<gipc::ABDSystem>          m_abd_system;
    std::unique_ptr<gipc::GlobalLinearSystem> m_global_linear_system;
};

#endif