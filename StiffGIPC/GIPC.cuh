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
#include <unordered_map>
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
    // [multi-env determinism 4.1+4.2] per-vertex env offset (default 0). The BVH builds on
    // d_bvh_vertexes = _vertexes + d_env_offset (envs spatially separated → efficient
    // broad-phase), while narrow-phase/CCD/solve read local _vertexes (identical per env →
    // deterministic). Zero offset ⇒ d_bvh_vertexes == _vertexes ⇒ no behavior change.
    double3*  d_env_offset   = nullptr;
    double3*  d_bvh_vertexes = nullptr;
    // [multi-env P2 / per-env BVH] per-env face/edge index lists (the BVH _active_idx subset).
    // Faces/edges grouped by env via d_point_to_group[face.x]; per-env [offset,count) so the
    // BVH Construct/Detect can be looped per env on LOCAL _vertexes (full precision, no cross-
    // env candidates) — the root fix for cross-env divergence. Built once (topology static).
    int*              d_perenv_face_idx = nullptr;   // face indices, env-contiguous
    int*              d_perenv_edge_idx = nullptr;   // edge indices, env-contiguous
    std::vector<int>  h_perenv_face_off, h_perenv_face_cnt;  // per-env [off,cnt) into face_idx
    std::vector<int>  h_perenv_edge_off, h_perenv_edge_cnt;
    std::vector<int>  h_perenv_active;               // env ids with ≥1 face or edge (skip empty NG slots)
    int               m_perenv_bvh_groups = 0;       // NG once built; 0 = not built
    bool              m_perenv_bvh = false;          // STIFF_PERENV_BVH gate
    // [perenv-parallel #1] scratch pool for CONCURRENT per-env BVH builds. Per env i we point-swap
    // bvh_f/bvh_e's scratch to pool slot i%K and run Construct+Detect on stream i%K (Construct captures
    // the swapped pointers at host-launch time, so different slots overlap; same-slot envs serialize on
    // the same stream). Shared I/O (verts/edges/_collisionPair/_cpNum...) stays on bvh_f/bvh_e.
    struct BvhScratch { Node* nodes=nullptr; AABB* bvs=nullptr; uint64_t* mch=nullptr;
                        uint32_t* idx=nullptr; AABB* tmp=nullptr; uint32_t* flags=nullptr; int* node_env=nullptr;
                        // [perenv-parallel #2] per-slot cub sort scratch: the Morton sort must not
                        // cudaMalloc/cudaFree (device-wide syncs serialized the pool streams).
                        void* sort_tmp=nullptr; size_t sort_bytes=0;
                        uint64_t* mch_alt=nullptr; uint32_t* idx_alt=nullptr; int sort_cap=0; };
    std::vector<BvhScratch>   m_pool_f, m_pool_e;
    std::vector<cudaStream_t> m_pool_streams;
    int               m_pool_K = 0;                  // 0 = pool not allocated
    void allocPerEnvPool(int K);                     // alloc K scratch sets + streams (once)
    const int*        m_d_p2g = nullptr;             // captured TetMesh.d_point_to_group (for lazy index build)
    // [multi-env cross-env DIAGNOSTIC] find the upstream env-asymmetry seed: compare env0 vs env1
    // (identical envs, local-frame coords) of any per-vertex buffer via a per-env local-id
    // correspondence. STIFF_XENV gates. xenvDiff returns max|env0[k]-env1[k]|.
    int*              d_xenv_lid     = nullptr;       // per-vertex local id within its env (-1 if env>1)
    double*           d_xenv_buf     = nullptr;       // [2 * maxlocal * 3] scatter target
    int               m_xenv_maxlocal = 0;
    bool              m_xenv_ready    = false;
    int*              m_d_vloc        = nullptr;  // [env-det] global->env-local vert id (canon tiebreak)
    bool              m_vloc_built    = false;
    double            xenvDiff(const double3* buf, const char* label);
    void              xenvPairClassify(const int4* pairs, int n, const char* label);
    int*              d_face_env = nullptr;
    int*              d_edge_env = nullptr;
    int*              d_face_localid = nullptr;
    int*              d_edge_localid = nullptr;
    uint32_t*         d_face_v0 = nullptr;
    uint32_t*         d_edge_v0 = nullptr;
    void              enableEnvMajorBVH(const int* p2g);
    // [multi-env per-group κ] per-group barrier stiffness. The DYNAMIC κ doubling (postLineSearch
    // Kappa*=2 on a GLOBAL close-contact bool) is the cross-env bifurcation coupling; per-group κ
    // + per-group close-val decouples it. nullptr/false → scalar Kappa (baseline, bit-identical).
    double*           m_kappa_group = nullptr;        // device [NG]
    std::vector<double> h_kappa_group;                // host mirror [NG]
    bool              m_pergroup_kappa = false;       // STIFF_PERGROUP_KAPPA gate
    int*              m_d_close_grp = nullptr;        // device [NG] per-group close-val isChange flag
    // [multi-env determinism 4.3] binned (reproducible) FP accumulator for the contact+friction
    // gradient: K bins per (vertex,component), each an EXACT fixed-point slice of one exponent
    // band. Atomic deposits are order-independent (each bin's adds are exact) ⇒ bit-identical
    // run-to-run AND across identical envs, with full dynamic range (no single-scale overflow).
    // Layout: ((v*3 + comp)*BINNED_K + k). Combined back into contact_grads each Newton iter.
    double*   g_grad_binned  = nullptr;
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

    // ②-D2H: persistent 9-slot device buffer for batched energy reductions.
    // computeEnergy() previously did 9 blocking cudaMemcpy(D2H) — one per
    // Energy_Add_Reduction_Algorithm call. Now each reduction writes its
    // final scalar into m_energy_slots[i] via D2D (queued, async), then
    // ONE blocking D2H grabs all 9 doubles at the end.
    static constexpr int kEnergySlotCount = 9;
    double* m_energy_slots = nullptr;
    // ②-D2H: 2-slot device buffer for batching ground+self largestFeasibleStepSize
    // reductions (called back-to-back at the top of each line search). One D2H
    // of 2 doubles instead of 2 separate blocking D2Hs.
    double* m_alpha_slots = nullptr;

    // [0be8da3-port, grow-only] element capacities of the persistent friction /
    // close-constraint buffers. cudaMalloc/cudaFree device-sync, so the per-step
    // alloc/free choreography is replaced by grow-on-demand (25% headroom);
    // capacity-sized upfront allocation was rejected: ~2GB standing VRAM at
    // N=20 (22.4/24GB) would cut the max env count.
    size_t m_fric_cp_cap  = 0;  // 5 cp-sized friction buffers
    size_t m_fric_gd_cap  = 0;  // 2 ground-sized friction buffers
    size_t m_close_gp_cap = 0;  // 2 gp-sized close buffers
    size_t m_close_cp_cap = 0;  // 2 cp-sized close buffers
    void   ensure_frictionBuffers();

    // [multi-env S1] per-env (per-group) feasible line-search step substrate.
    // d_env_alpha[g] = the largest feasible alpha for env g this Newton iter
    // (per-env CFL + per-env CCD min). NG fixed slots; envs are dense 0..ng-1.
    // PHYSICS-NEUTRAL until S2: computed + validated only, the actual step still
    // uses the global scalar alpha. Gated by env STIFF_PERENV_ALPHA. h_env_alpha
    // is the host mirror S2 will read to drive per-env step_forward.
    static constexpr int kEnvAlphaSlots = 256;
    double*             m_env_alpha   = nullptr;   // device, size kEnvAlphaSlots
    std::vector<double> h_env_alpha;               // host mirror
    // scratch for the per-env feasibility reductions split across two phases of
    // one Newton iter: regions [0]=ground [1]=self-narrow [2]=refined-self
    // [3]=cfl-maxspeed, each kEnvAlphaSlots wide. Phase A (pre-buildFullCP)
    // fills ground+self-narrow; Phase B fills refined+cfl and combines.
    double*             m_env_scratch = nullptr;   // device, size 4*kEnvAlphaSlots
    // [multi-env S2] per-env step apply. When m_perenv_apply is true, step_forward
    // moves FEM vert v by m_env_alpha[point_to_group[v]] and ABD body b by
    // m_abd_body_alpha[b] (gathered = m_env_alpha[body_to_group[b]]). The scalar
    // alpha arg is the fallback for ungrouped DOFs / disabled mode. Gated by
    // STIFF_PERENV_ALPHA; the caller (lineSearch) flips m_perenv_apply off to
    // fall back to a uniform global step when the global energy safety-check fails.
    bool                m_perenv_apply    = false;
    double*             m_abd_body_alpha  = nullptr;  // device, size abd_body_num
    // set true by the S1 block each Newton iter once m_env_alpha is freshly
    // populated; lineSearch only does the per-env try when this is true (guards
    // against applying stale per-env alpha on iters where S1 didn't run).
    bool                m_env_alpha_valid = false;
    // [multi-env S4] per-env active flag (1=active/solve, 0=masked/converged).
    // An env is masked once its Newton max-move < thr*margin; periodically all
    // are unmasked + re-checked to catch non-monotonic bounce-back. Read by the
    // assembly/PCG/SpMV masking (RHS-zero + triplet-skip) to skip converged envs.
    // Gated STIFF_PERENV_MASK. h_env_active mirror; m_recheck_counter drives the
    // periodic full re-check.
    int*                m_env_active      = nullptr;  // device, size kEnvAlphaSlots
    std::vector<int>    h_env_active;                 // host mirror
    int                 m_recheck_counter = 0;

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
    // [P1-dyn] Dynamic triplet-buffer sizing for non-hybrid scenes. m_fixed_triplet_base
    // = topology-fixed internal triplet count (captured in init()); each step the global
    // triplet buffer is grown to 2*length (length = fixed + ACTUAL contact triplets from
    // h_cpNum) — the 2x is the converter's documented [length:2*length) scratch/output
    // region (global_linear_system.cu). Never overflows: length is exact.
    long long m_fixed_triplet_base       = 0;
    bool      m_dynamic_triplet          = false;

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
    double newton_velocity_tol     = 0.0;   // [uipc-style opt-in] 0 = legacy exit
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
    // [multi-env P2] build the per-env face/edge index lists (once; topology static). NG = #groups.
    void buildPerEnvBVHIndex(int NG, const int* d_point_to_group);
    // [multi-env P2] per-env Construct+Detect loop (DCD). Replaces buildBVH()+buildCP() when
    // m_perenv_bvh: each env builds its tree on LOCAL _vertexes via _active_idx, queries, appends.
    void buildBVH_and_CP_perenv(double dHat);
    // [multi-env P2] per-env CCD Construct+FullDetect loop (line-search feasible-alpha). Same
    // idea on the swept BVH so the per-env feasible alpha is full-precision / per-env identical.
    void buildBVH_and_CP_perenv_CCD(double alpha);

    AABB* calcuMaxSceneSize();

    void buildBVH_FULLCCD(const double& alpha);
    void step_forward(device_TetraData& TetMesh, double alpha = 1.0, bool move_boundary = false);


    void GroundCollisionDetect();
    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
    void calBarrierHessian();
    void calBarrierGradient(double3* _gradient, double mKap,
                            int2* ec_pair = nullptr, double3* ec_force = nullptr,
                            const int* ec_pbid = nullptr, double ec_inv_dt2 = 0.0);

    // [Step B] per-contact force export for the Newton ContactSensor. Fills
    // out_pair[i]=(bodyA,bodyB) (bodyB=-1 for ground) and out_force[i]=world
    // contact force (N) on bodyA. Returns count = h_cpNum[0] + h_gpNum.
    int exportContacts(int2* out_pair, double3* out_force);
    // [Step B] scratch gradient for exportContacts (per-vertex, grow-only).
    double3*  _ec_grad_scratch = nullptr;
    int       _ec_scratch_cap  = 0;
    // [4.3] binned-gradient helpers: zero before / combine after any barrier/friction
    // gradient kernel (they scatter to the binned accumulator, not their _gradient arg).
    void zeroBinnedGrad();
    void combineBinnedGrad(double3* out);
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
    // [backport] standalone per-env energy dispatcher: writes the reduced global
    // scalar to a device slot (D2D) and, when out_penv != nullptr, buckets per-env.
    // Used only by computeEnergy_perenv. (Not the full ②-D2H computeEnergy rewrite.)
    void   Energy_Add_Reduction_Algorithm_DeviceOut(int type,
                                                    device_TetraData& TetMesh,
                                                    double* out_slot,
                                                    double* out_penv = nullptr);
    // [multi-env S3] per-env total energy E_g into env_out[kEnvAlphaSlots]
    // (host array). Validates Sum_g E_g == global computeEnergy. Returns global E.
    double computeEnergy_perenv(device_TetraData& TetMesh, std::vector<double>& env_out);
    // [de-CPU S3] device-resident variant: per-env energies land in d_Eg[kEnvAlphaSlots] on DEVICE
    // (same term kernels + a device combine replicating the host order/factors -> bit-identical
    // values). NO D2H. Used by the S3 per-env backtrack decision kernel.
    void computeEnergy_perenv_dev(device_TetraData& TetMesh, double* d_Eg);
    // [de-CPU S3] shared term-launcher: fills the static pe_all slice block on device (layout in
    // GIPC.cu) and reports whether per-env kappa rescale applies. Used by both variants above.
    double* _launch_perenv_energy_terms(device_TetraData& TetMesh, bool& perenv_k_out);
    // ②-D2H batched variants — write minValue (NOT 1.0/minValue) to slot.
    // Caller does the 1.0/x and the m_skip_all_collision / numbers<1 guards.
    void   ground_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, double* out_slot);
    void   self_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, int numbers, double* out_slot);

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

    // [decouple debug] Full-state checkpoint: save/restore the cross-frame persistent state
    // (FEM vertexes/o_vertexes/velocities/xTilta + ABD q/q_prev/q_v + Kappa) so a mid-trajectory
    // restart is bit-identical (friction/contact is ephemeral, rebuilt each step from positions).
    // Enables fast iteration on deep-grasp Hessian-FP debugging: checkpoint frame N once, then
    // load+step frame N+1 repeatedly instead of replaying frames 0..N each time.
    void save_checkpoint(device_TetraData& tm, const char* path);
    void load_checkpoint(device_TetraData& tm, const char* path);

    // Pending per-body density overrides (body_id -> density), stashed by
    // SimEngine::set_abd_body_density before finalize. build_gipc_system
    // transfers these into m_abd_system right after it is created and before
    // the per-body mass setup runs.
    std::unordered_map<int, double>           m_pending_abd_density;

    // Pending per-body inertial overrides (body_id -> {mass, com[3],
    // inertia[3x3 row-major]}). Stashed by SimEngine::set_abd_body_inertia
    // before finalize; transferred to m_abd_system right after it is created.
    struct PendingInertia { double mass; double com[3]; double inertia[9]; };
    std::unordered_map<int, PendingInertia>   m_pending_abd_inertia;
};

#endif