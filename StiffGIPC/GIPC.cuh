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
#include <vector>
#include "mlbvh.cuh"
#include "device_fem_data.cuh"

#include "PCG_SOLVER.cuh"
#include "device_common/device_buffer.cuh"  // [3d] RAII device-buffer owner
#include "device_common/mirrors.h"           // [3b] audited host mirrors
#include "multienv/mode_config.h"            // [C1] finalize-time mode snapshot
#include "energy/energy_terms.h"             // [E2] term registry (X-macro)
#include "frame_fsm/frame_status.cuh"         // [Phase C] frame-boundary ABI

#ifdef GIPC_ENABLE_DIAGNOSTICS
// Intrusive derivative diagnostics. These declarations and their Python
// bindings are absent from production builds unless explicitly enabled.
struct FdCheckResult
{
    double max_rel  = 0.0;
    double mean_rel = 0.0;
    double worst_fd = 0.0;
    double worst_an = 0.0;
    double sign     = 0.0;
    double p50 = 0.0, p95 = 0.0;
    int    worst_v = -1, worst_axis = -1, n = 0, n_nonfinite = 0;
};
struct FdHessianResult
{
    double max_rel = 0.0, mean_rel = 0.0, sign = 0.0;
    double p50 = 0.0, p95 = 0.0;
    int worst_v = -1, worst_axis = -1, n = 0, n_nonfinite = 0;
};
struct FdActivityResult
{
    uint32_t fem_tets = 0, triangles = 0, bending_edges = 0, soft = 0;
    uint32_t contact = 0, ground = 0, friction = 0, ground_friction = 0;
};
#endif
#include "multienv/mode_contract.h"          // [C2] the promise table (doc-only)
#include <gipc/abd_fem_count_info.h>
namespace gipc
{
class ABDSimData;
class ABDSystem;
class GlobalLinearSystem;
struct RevoluteDrivingControlPacked;
struct PrismaticDrivingControlPacked;
}  // namespace gipc

class GIPC
{
  public:
    bool      animation      = false;
    // Per-engine telemetry/state. These used to be translation-unit globals,
    // causing one SimEngine to inherit another engine's frame index and
    // counters (checkpoint soft-target timing included).
    int       m_total_newton_iters    = 0;
    // [rl-reset] step-health telemetry: line-search budget exhaustions and the
    // subset with non-finite incremental potential. An RL loop diffs these
    // across step() to detect and discard silently-degraded episodes (merged
    // mode WARNs and continues by contract).
    int       m_ls_exhausted_total    = 0;
    int       m_ls_nonfinite_total    = 0;
    // [B3 device-count] trial mode: contact-count-dependent energy kernels
    // read the LIVE count from _cpNum on device (grid sized from a slacked
    // iteration-start bound) instead of a per-trial host mirror refresh.
    // Default off = legacy behavior everywhere.
    bool      m_energy_use_device_counts = false;
    // [B3 trial-defer] iteration-start contact-count bounds (slacked) for
    // trial-mode energy grids; buildCP skips its counts D2H + overflow check
    // while m_ls_defer_counts is set — overflow is detected via the monotone
    // device counter piggybacked on the line-search decision read, and the
    // rare trip re-runs buildCP in legacy mode (full grow+redo machinery).
    bool      m_ls_defer_counts       = false;
    uint32_t* m_scr_gp_friction       = nullptr;  // [B3 s7] friction-era gp count, device-stashed
    double*   m_d_ls_alpha            = nullptr;  // [C-1] device-resident trial alpha
    cudaGraphExec_t m_ls_graph_exec   = nullptr;  // [C-1] cached trial-body self-tail graph
    // Captured host values: pointer generation, budget, the two device-count
    // energy launch bounds, and the frozen DCD snapshot copy length.
    long long m_ls_graph_sig[5]       = {-1, -1, -1, -1, -1};
    // [Phase C] Opaque owner of conditional graphs, transaction snapshots and
    // pinned frame-boundary packets.  Kept opaque here so the public solver
    // header exposes only the stable FrameStatus ABI.
    void*                   m_frame_graph_context = nullptr;
    // [Phase D] Opaque owner of an episode-resident outer WHILE graph,
    // pre-uploaded action sequences, and two pinned observation slots.  The
    // episode context borrows the frame transaction snapshots, so it is
    // destroyed before m_frame_graph_context.
    void*                   m_episode_graph_context = nullptr;
    bool                    m_frame_graph_active  = false;
    bool                    m_frame_terminal_emitted = false;
    frame_fsm::FrameStatus  m_last_frame_status{};
    bool      m_ccd_defer_counts      = false;
    int       m_energy_bound_cp       = 0;
    int       m_energy_bound_gp       = 0;
    unsigned  m_pair_overflow_seen    = 0u;
    void      refresh_pair_counts();
    void      handleGroundCollapse(int collapsed);
    // [descriptor phase-0.3] solver scratch — was function-static device
    // allocations shared process-wide (leak + cross-engine sharing + dangling
    // after device reset). Instance-owned; the use sites keep their lazy
    // `if(!ptr) cudaMalloc` pattern through reference aliases; freed in
    // FREE_DEVICE_MEM. All are tiny fixed-size (scalars / kEnvAlphaSlots).
    double* m_scr_ls_eg0 = nullptr;
    double* m_scr_ls_eg1 = nullptr;
    int*    m_scr_ls_decision_counts = nullptr;
    double* m_scr_maxk = nullptr;
    double* m_scr_sq_a = nullptr;
    int*    m_scr_cnt_a = nullptr;
    double* m_scr_mxm = nullptr;
    double* m_scr_mm = nullptr;
    int*    m_scr_ct = nullptr;
    double* m_scr_mx = nullptr;
    int*    m_scr_env_cnt = nullptr;
    double* m_scr_sq_b = nullptr;
    double* m_scr_perenv_ta = nullptr;
    int*    m_scr_xenv4 = nullptr;
    double* m_scr_gsum_bin = nullptr;
    double* m_scr_gsnorm_bin = nullptr;
    double* m_scr_gsum_g = nullptr;
    double* m_scr_gsnorm_g = nullptr;
    int       m_total_frames          = 0;
    double    m_total_pcg_iters       = 0.0;
    double    m_total_collision_pairs = 0.0;
    double    m_max_collision_pairs   = 0.0;
    double    m_total_time_ms         = 0.0;
    double    m_phase_time_ms[5]      = {0.0, 0.0, 0.0, 0.0, 0.0};
    double    m_time_make_pd_ms       = 0.0;
    bool      m_update_boundary       = false;
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
                        uint32_t* node_max_element=nullptr;
                        // [perenv-parallel #2] per-slot cub sort scratch: the Morton sort must not
                        // cudaMalloc/cudaFree (device-wide syncs serialized the pool streams).
                        void* sort_tmp=nullptr; size_t sort_bytes=0;
                        uint64_t* mch_alt=nullptr; uint32_t* idx_alt=nullptr; int sort_cap=0; };
    std::vector<BvhScratch>   m_pool_f, m_pool_e;
    std::vector<cudaStream_t> m_pool_streams;
    int               m_pool_K = 0;                  // 0 = pool not allocated
    void allocPerEnvPool(int K);                     // alloc K scratch sets + streams (once)
    const int*        m_d_p2g = nullptr;             // captured TetMesh.d_point_to_group (for lazy index build)
    // [iron-law] captured for mid-run env quarantine: body→group map + body count
    // (to mark a quarantined env's bodies in the ground-skip table), plus
    // ownership of a lazily-allocated skip table (normally d_tetMesh owns it).
    const int*        m_d_b2g = nullptr;             // captured TetMesh.d_body_to_group
    // [C1] runtime truth of the multi-env mode, captured once at finalize
    ModeConfig m_mode_config;
    int               m_collision_body_count = 0;    // captured TetMesh.collision_body_num
    bool              m_ground_skip_owned = false;   // we cudaMalloc'ed _ground_skip_body
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
    // [env-scale convergence] per-env STATIC bbox diag^2 (rest state, built at finalize
    // from p2g). Convergence thresholds: merged uses the AVG over active envs (N-invariant,
    // no absolute/relative_dhat in the exit path); per-env freeze uses each env's OWN value
    // (batch-invariant for strict). Empty/0 -> legacy whole-scene bboxDiagSize2.
    double*             d_env_bbox2     = nullptr;   // device [NG]
    std::vector<double> h_env_bbox2;                 // host mirror [NG]
    double              m_avg_env_bbox2 = 0.0;       // avg over active envs
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
    DeviceBuffer<uint3>    _faces;      // [B1] owner; mlbvh holds raw views
    DeviceBuffer<uint2>    _edges;
    DeviceBuffer<uint32_t> _surfVerts;


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

#ifdef STIFF_BVH_COHERENCE_AUDIT_BUILD
    // Shadow-only model of a DCD Verlet candidate list.  It never changes the
    // solver's pair set: every buildCP still runs the exhaustive LBVH query.
    // The audit only snapshots positions and counts how many successive
    // queries a list grown by STIFF_BVH_MARGIN_SCALE could safely serve under
    // the max-vertex-displacement completeness proof.
    std::vector<double3> m_bvh_coherence_reference;
    std::vector<double3> m_bvh_coherence_current;
    unsigned long long   m_bvh_coherence_observations = 0;
    unsigned long long   m_bvh_coherence_builds       = 0;
    unsigned long long   m_bvh_coherence_reuses       = 0;
    unsigned long long   m_bvh_coherence_invalidations = 0;
    unsigned long long   m_bvh_coherence_current_span = 0;
    unsigned long long   m_bvh_coherence_max_span     = 0;
    double               m_bvh_coherence_max_displacement = 0.0;
    struct BvhCoherenceStat
    {
        unsigned long long observations = 0;
        unsigned long long builds = 0;
        unsigned long long reuses = 0;
        unsigned long long invalidations = 0;
        unsigned long long current_span = 0;
        unsigned long long max_span = 0;
        double max_displacement = 0.0;
    };
    struct BvhCoherencePairState
    {
        int body_a = -1;
        int body_b = -1;
        std::vector<double3> reference_a;
        std::vector<double3> reference_b;
        BvhCoherenceStat stat;
        // Validity of the raw margin list used by the immediately following
        // broad-phase query.  The validation-only workload census associates
        // each query's actual primitive tests with this exact decision.
        bool reusable_this_observation = false;
        unsigned long long primitive_tests[4] = {};
        unsigned long long reusable_primitive_tests[4] = {};
    };
    struct BvhSweptPairState
    {
        int body_a = -1;
        int body_b = -1;
        std::vector<double3> reference_start_a;
        std::vector<double3> reference_end_a;
        std::vector<double3> reference_start_b;
        std::vector<double3> reference_end_b;
        BvhCoherenceStat stat;
        bool reusable_this_observation = false;
        unsigned long long primitive_tests[2] = {};
        unsigned long long reusable_primitive_tests[2] = {};
    };
    bool                     m_bvh_family_coherence_initialized = false;
    std::vector<int>         m_bvh_coherence_body_id;
    std::vector<int>         m_bvh_coherence_body_is_fem;
    std::vector<int>         m_bvh_coherence_skip_matrix;
    std::vector<std::vector<int>> m_bvh_coherence_body_vertices;
    std::vector<double3>     m_bvh_coherence_body_reference;
    std::vector<BvhCoherenceStat> m_bvh_coherence_body_stats;
    std::vector<BvhCoherencePairState> m_bvh_coherence_pair_states;
    std::vector<unsigned long long> m_bvh_pair_work_snapshot;
    std::vector<BvhSweptPairState> m_bvh_swept_pair_states;
    std::vector<unsigned long long> m_bvh_swept_pair_work_snapshot;
    std::vector<double3> m_bvh_swept_current_start;
    std::vector<double3> m_bvh_swept_current_end;
    bool m_bvh_swept_coherence_initialized = false;
    bool                     m_bvh_vf_cache_ready = false;
    int                      m_bvh_vf_cache_pair_count = 0;
    int                      m_bvh_vf_cache_segment_capacity = 0;
    unsigned char*           m_bvh_vf_cache_valid = nullptr;
    int*                     m_bvh_vf_cache_index = nullptr;
    int2*                    m_bvh_vf_cache_candidates = nullptr;
    uint32_t*                m_bvh_vf_cache_counts = nullptr;
    int*                     m_bvh_vf_cache_overflow = nullptr;
    int2*                    m_bvh_ee_cache_candidates = nullptr;
    uint32_t*                m_bvh_ee_cache_counts = nullptr;
    int*                     m_bvh_ee_cache_overflow = nullptr;
    int                      m_bvh_ee_cache_segment_capacity = 0;
    unsigned char*           m_bvh_ccd_cache_valid = nullptr;
    int2*                    m_bvh_vf_ccd_cache_candidates = nullptr;
    uint32_t*                m_bvh_vf_ccd_cache_counts = nullptr;
    int*                     m_bvh_vf_ccd_cache_overflow = nullptr;
    int2*                    m_bvh_ee_ccd_cache_candidates = nullptr;
    uint32_t*                m_bvh_ee_ccd_cache_counts = nullptr;
    int*                     m_bvh_ee_ccd_cache_overflow = nullptr;
    int                      m_bvh_ccd_cache_segment_capacity = 0;
    double3*                 m_bvh_ccd_cache_reference_start = nullptr;
    double3*                 m_bvh_ccd_cache_reference_end = nullptr;
    unsigned long long*      m_bvh_ccd_cache_device_stats = nullptr;
    bool                     m_bvh_ccd_cache_seen = false;
    int                      m_bvh_pair_cache_mask = 0;
    uint32_t*                m_bvh_vf_cache_ref_offsets = nullptr;
    int*                     m_bvh_vf_cache_ref_vertices = nullptr;
    double3*                 m_bvh_vf_cache_references = nullptr;
    unsigned long long*      m_bvh_vf_cache_device_stats = nullptr;
    int                      m_bvh_vf_cache_ref_entry_count = 0;
    bool                     m_bvh_vf_cache_device_validity = false;
    uint32_t*                m_bvh_vf_front_nodes = nullptr;
    uint32_t*                m_bvh_vf_front_counts = nullptr;
    int*                     m_bvh_vf_front_overflow = nullptr;
    uint32_t*                m_bvh_ee_front_nodes = nullptr;
    uint32_t*                m_bvh_ee_front_counts = nullptr;
    int*                     m_bvh_ee_front_overflow = nullptr;
    int                      m_bvh_vf_front_capacity = 128;
    unsigned long long       m_bvh_vf_cache_queries = 0;
    unsigned long long       m_bvh_vf_cache_valid_pair_uses = 0;
    unsigned long long       m_bvh_vf_cache_pair_uses = 0;
    unsigned long long       m_bvh_vf_cache_replay_candidates = 0;
    unsigned long long       m_bvh_vf_front_fallbacks = 0;
    uint32_t                 m_bvh_vf_front_max_roots = 0;
    uint32_t                 m_bvh_vf_front_max_body_roots = 0;
    int*                     m_bvh_face_body = nullptr;
    int*                     m_bvh_edge_body = nullptr;
    void collectBvhPairWorkload();
    void updateBvhVfPairCache();
    void updateBvhCcdPairCache(const double& alpha,
                               const double* alpha_dev);
    void auditBvhTemporalCoherence();
    void auditBvhSweptTemporalCoherence(const double& alpha,
                                         const double* alpha_dev);
    void collectBvhSweptPairWorkload();
    void printBvhSweptTemporalCoherence();
    void printBvhTemporalCoherence();
#endif

    PCG_Data pcg_data;

    int4*     _collisonPairs     = nullptr;
    int4*     _ccd_collisonPairs = nullptr;
    DeviceBuffer<uint32_t> _cpNum;  // [B1] 6 slots: cp[0:5] + gp[5]; _gpNum is a VIEW (=_cpNum+5)
    // Read-only detection snapshots used by capacity-launched contact
    // assembly. _cpNum is reused as an atomic rank scratch during assembly,
    // so no guarded kernel may consume it directly.
    DeviceBuffer<uint32_t> m_pair_snap_cur;   // cp[0:5] + gp[5], current DCD
    DeviceBuffer<uint32_t> m_pair_snap_last;  // lagged friction counts
    int*      _MatIndex          = nullptr;
    uint32_t* _close_cpNum       = nullptr;
    // On-demand reduction scratch: reductions launch ceil(count/default_threads)
    // blocks each writing one double. pcg_data.squeue is only sized to the mesh
    // (max(vertexNum,tetra)), so reductions over a COLLISION/CCD PAIR count
    // (h_cpNum / h_ccd_cpNum, up to MAX_*_PAIRS) overflow it. This buffer grows
    // to ceil(count/default_threads) on demand so no pair-count reduction can
    // ever overflow; after warmup the capacity stabilizes (no further realloc).
    DeviceBuffer<double> m_reduce_scratch;  // [3d-2] grow-only reduce scratch
    size_t    m_reduce_cap       = 0;

    uint32_t* _environment_collisionPair = nullptr;

    uint32_t* _closeConstraintID  = nullptr;
    double*   _closeConstraintVal = nullptr;

    int4*   _closeMConstraintID  = nullptr;
    double* _closeMConstraintVal = nullptr;

    uint32_t* _gpNum       = nullptr;
    uint32_t* _close_gpNum = nullptr;
    // Ground-distance invariant flag: zero = none; otherwise the first vertex
    // with a non-finite or non-positive distance encoded as -(id + 1).
    int*      _gdCollapse  = nullptr;
    // Effective CCD-invalid bits: global/per-env x ground/narrow/refined.
    // Refined candidates are first recorded in m_ccd_refined_invalid and are
    // promoted here only when the corresponding refinement gate consumes them.
    int*      m_ccd_alpha_invalid   = nullptr;
    int*      m_ccd_refined_invalid = nullptr;  // [0]=global, [1+g]=per-env raw status
    int*      m_ground_trial_invalid = nullptr;
    int*      m_env_ground_trial_invalid = nullptr;
    // [per-body friction] per-vertex mu tables (device, size vertexNum), built
    // at finalize from SimEngine's pending per-body overrides. nullptr = feature
    // unused -> every friction kernel takes its legacy scalar path
    // (bit-identical to v0.8.3). Self-contact pairs combine the two sides'
    // representative-vertex mu geometrically; ground pairs use the vertex's own
    // ground-mu directly.
    double*   d_vert_mu    = nullptr;
    double*   d_vert_mu_gd = nullptr;
    // [per-env productization] per-env solve telemetry, reset each solve_subIP.
    // Filled by the HOST S1 path (per_env_exit / STIFF_PERENV_ALPHA without the
    // dev-mask fast path). frozen_iter[g]: Newton iter at which env g froze
    // (-1 = ran to loop end). status[g]: 0 active/absent, 1 converged,
    // 2 timeout (env_newton_iter_cap), 3 diverged (NaN/inf max-move).
    std::vector<int> m_env_frozen_iter;
    std::vector<int> m_env_status;
    // [iron-law] PERSISTENT per-env quarantine (unlike m_env_status, survives
    // across frames): set when an env becomes ground-infeasible MID-RUN — the
    // env is pinned (alpha=0, status=3) every iteration of every later solve
    // instead of a whole-process throw killing the healthy envs. Empty until
    // first quarantine. Init-time violations (before per-env machinery is
    // live) still throw, preserving the finalize-validation contract.
    std::vector<uint8_t> m_env_quarantined;
    // [iron-law completion] device mirror of m_env_quarantined + per-env
    // non-finite-direction scan flags (both lazy, kEnvAlphaSlots ints).
    DeviceBuffer<int> m_d_env_quarantined;  // [3d-2]
    DeviceBuffer<int> m_d_env_dirnan;       // [3d-2]
    int              env_newton_iter_cap = 0;  // per-env iter budget; 0 = off
    // [T1] line-search backtracking budget (halvings); 0 = engine default (64).
    int              line_search_max_iter = 64;
    double           energy_abs_tol       = 0.0;
    double           energy_rel_tol       = 0.0;
    uint64_t         energy_tolerance_accept_count = 0;
    void      throwIfGroundDistanceInvalid();
    // [iron-law] mid-run env quarantine (see GIPC.cu): demote an env-attributable
    // ground infeasibility to a persistent per-env freeze instead of a throw.
    bool      quarantineEnvOfVertex(int vertex, double distance);
    bool      quarantineEnv(int env, int vertex, double distance);
    // [rl-reset] inverse of quarantineEnv for episode resets; self-correcting
    // (a still-broken env is re-quarantined within one frame by the scans).
    bool      reviveEnv(int env);
    void      quarantineGroundInfeasibleAtFrameStart();
    bool      perEnvIsolationLive();   // [2c] THE availability gate (multienv/isolation.cuh)
    void      throwIfInvalidCcdAlpha(const char* context);
    int       groundTrialStatus(const int* point_to_group, int group_count);
    void      halveGroundInvalidEnvAlpha(int group_count);
    //uint32_t* _cpNum;
    // [3b pilot] counts-after-build mirrors: audited via STIFF_MIRROR_AUDIT=1
    // (writers invalidate at build entries, refresh at the D2H copy-backs)
    HostMirrorArray<uint32_t, 5> h_cpNum{"h_cpNum"};
    HostMirror<uint32_t>         h_ccd_cpNum{"h_ccd_cpNum"};
    // Last fully adjudicated swept-CCD count. DCD legitimately invalidates
    // h_ccd_cpNum by reusing _cpNum, but frame-boundary telemetry must still
    // report the most recent CCD high-water without bypassing mirror audits.
    uint32_t m_last_ccd_pair_count = 0;
    // [C6-b] Swept counts swing wildly WITHIN a frame (foldshirt: 58k at the
    // last Newton iteration, 390k at the peak). Training from the last value
    // seeds a tier that the very first recorded frame overflows, forcing a
    // boundary retry every frame. Train from the peak instead.
    uint32_t m_peak_ccd_pair_count = 0;
    // [C6-b] Same story on the DCD/ground axes: h_cpNum/h_gpNum hold the LAST
    // Newton iteration's census, which is far below the frame's peak.
    uint32_t m_peak_cpNum[5] = {0, 0, 0, 0, 0};
    uint32_t m_peak_gpNum    = 0;
    // [C6-b] Set by frame_graph_finish_terminal when a capacity tier actually
    // grew. A frame that overflowed can only be retried if the retry will run
    // at a LARGER tier; otherwise it would replay the identical failure.
    bool m_graph_tier_grew = false;
    void note_pair_census_peak()
    {
        for(int s = 0; s < 5; ++s)
            if(h_cpNum[s] > m_peak_cpNum[s])
                m_peak_cpNum[s] = h_cpNum[s];
        if(static_cast<uint32_t>(h_gpNum) > m_peak_gpNum)
            m_peak_gpNum = static_cast<uint32_t>(h_gpNum);
    }
    // [narrow-self snapshot] immutable copy of the DCD-time CCD pair mirror.
    // The DCD detect kernels write _collisionPair AND _ccd_collisionPair at the
    // same atomic slot, so right after buildCP the first h_cpNum[0] entries of
    // _ccd_collisonPairs ARE the DCD pair set — but buildFullCP later OVERWRITES
    // the same buffer with the swept list, so by the time S1's narrow-self runs,
    // the "prefix" is an unrelated slice of the swept emission (race-ordered,
    // env-unbalanced: the measured strict cross-env asymmetry + N=8 run-to-run
    // instability). buildCP therefore snapshots the mirror into this dedicated
    // buffer, and narrow-self consumes ONLY the snapshot (full count, stable
    // content). Structural fix per the two-agent root-cause analysis.
    int4*    _dcd_ccd_snapshot = nullptr;
    uint32_t m_dcd_snap_count  = 0;
    int      m_dcd_snap_cap    = 0;
    void     snapshotDcdCcdPairs();
    HostMirror<uint32_t>         h_gpNum{"h_gpNum"};

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
    // [3d pilot] friction lastH family: RAII-owned (implicit T* views keep
    // every kernel-arg/memset site unchanged; policy caps m_fric_*_cap below)
    DeviceBuffer<double>                 lambda_lastH_scalar;
    DeviceBuffer<double2>                distCoord;
    DeviceBuffer<__GEIGEN__::Matrix3x2d> tanBasis;
    DeviceBuffer<int4>                   _collisonPairs_lastH;
    // [3b-2] lagged-friction snapshot mirrors: value semantics = "as of the
    // last buildFrictionSets snapshot" — no invalidation sites by design
    // (never stale between snapshots); wrapped for uniformity + type-check.
    HostMirrorArray<uint32_t, 5> h_cpNum_last{"h_cpNum_last"};

    DeviceBuffer<double>   lambda_lastH_scalar_gd;
    DeviceBuffer<uint32_t> _collisonPairs_lastH_gd;
    HostMirror<uint32_t> h_gpNum_last{"h_gpNum_last"};  // [3b-2] was UNINITIALIZED — T{} now

    // Persistent device slots for 9 FEM/contact terms plus 6 ABD terms. The
    // line-search path combines these on device; diagnostic computeEnergy()
    // performs one batched D2H instead of one transfer per term.
    static constexpr int kEnergySlotCount = 15;
    double* m_energy_slots = nullptr;
    // E0/Etrial belong exclusively to line search. Compatibility callers use a
    // separate scalar so diagnostics cannot overwrite an in-flight E0.
    double* m_line_search_energy      = nullptr;
    double* m_compatibility_energy    = nullptr;
    // Line-search decision only: 0=descent, 1=retry, 2=tolerance acceptance.
    int*    m_line_search_decision    = nullptr;
    // [descriptor phase-0b] per-env energy slices + DeviceOut scalar sink for
    // the host energy dispatch — were function-static device allocations
    // shared across engines in one process; instance-owned, freed in
    // FREE_DEVICE_MEM. Lazily allocated on first per-env energy evaluation.
    double* m_pe_all                  = nullptr;
    double* m_energy_sink             = nullptr;
    // Newton convergence only: 0=continue, 1=converged.
    int*    m_newton_convergence_decision = nullptr;
    // CCD device-control state: ground, narrow-self, temp alpha, max speed,
    // refined-self, final alpha, CFL alpha, effective-invalid snapshot.
    double* m_ccd_alpha_slots = nullptr;

    // [0be8da3-port, grow-only] element capacities of the persistent friction /
    // close-constraint buffers. cudaMalloc/cudaFree device-sync, so the per-step
    // alloc/free choreography is replaced by grow-on-demand (25% headroom);
    // capacity-sized upfront allocation was rejected: ~2GB standing VRAM at
    // N=20 (22.4/24GB) would cut the max env count.
    size_t m_fric_cp_cap  = 0;  // 5 cp-sized friction buffers
    size_t m_fric_gd_cap  = 0;  // 2 ground-sized friction buffers
    size_t m_close_gp_cap = 0;  // 2 gp-sized close buffers
    size_t m_close_cp_cap = 0;  // 2 cp-sized close buffers
    // [C4-b] in-graph close-set doubling flag (device int) + arming bit for
    // the device-kappa injection (set only while the whole-frame graph
    // records a collision body).
    int* m_d_close_flag = nullptr;
    bool m_graph_kappa_armed = false;
    void   ensure_frictionBuffers();

    // [multi-env S1] per-env (per-group) feasible line-search step substrate.
    // d_env_alpha[g] = the largest feasible alpha for env g this Newton iter
    // (per-env CFL + per-env CCD min). Storage has fixed capacity, while kernels
    // receive m_active_group_count; envs are validated dense 0..ng-1.
    // PHYSICS-NEUTRAL until S2: computed + validated only, the actual step still
    // uses the global scalar alpha. Gated by env STIFF_PERENV_ALPHA. h_env_alpha
    // is the host mirror S2 will read to drive per-env step_forward.
    static constexpr int kEnvAlphaSlots = device_TetraData::kGroupSlotCapacity;
    int                 m_active_group_count = 0;
    DeviceBuffer<double> m_env_alpha;    // [3d-2] device, size kEnvAlphaSlots
    std::vector<double> h_env_alpha;               // host mirror
    // scratch for the per-env feasibility reductions split across two phases of
    // one Newton iter: direct-alpha regions [0]=ground [1]=self-narrow
    // [2]=refined-self, [3]=surface cfl-maxspeed, [4]=all-vertex Newton max-move.
    // Alpha regions are initialized to 1 and MIN-reduced; max regions start at 0.
    DeviceBuffer<double> m_env_scratch;  // [3d-2] device, size 5*kEnvAlphaSlots
    // [multi-env S2] per-env step apply. When m_perenv_apply is true, step_forward
    // moves FEM vert v by m_env_alpha[point_to_group[v]] and ABD body b by
    // m_abd_body_alpha[b] (gathered = m_env_alpha[body_to_group[b]]). The scalar
    // alpha arg is the fallback for ungrouped DOFs / disabled mode. Gated by
    // STIFF_PERENV_ALPHA; the caller (lineSearch) flips m_perenv_apply off to
    // fall back to a uniform global step when the global energy safety-check fails.
    bool                m_perenv_apply    = false;
    DeviceBuffer<double> m_abd_body_alpha;  // [3d-2] device, size abd_body_num
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
    DeviceBuffer<int>    m_env_active;      // [3d-2] device, size kEnvAlphaSlots
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

    // Auxiliary stream for overlapping bvh_e collision detection with bvh_f.
    // Persistent events connect it to the per-thread default stream without
    // per-Newton event allocation/destruction or a host-wide stream sync.
    cudaStream_t m_aux_stream      = nullptr;
    cudaEvent_t  m_aux_reset_event = nullptr;
    cudaEvent_t  m_aux_done_event  = nullptr;

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
    void buildFullCP(const double& alpha, const double* alpha_dev = nullptr);
    void buildBVH();
    // [multi-env P2] build the per-env face/edge index lists (once; topology static). NG = #groups.
    void buildPerEnvBVHIndex(int NG, const int* d_point_to_group);
    // [multi-env P2] per-env Construct+Detect loop (DCD). Replaces buildBVH()+buildCP() when
    // m_perenv_bvh: each env builds its tree on LOCAL _vertexes via _active_idx, queries, appends.
    void buildBVH_and_CP_perenv(double dHat);
    // [multi-env P2] per-env CCD Construct+FullDetect loop (line-search feasible-alpha). Same
    // idea on the swept BVH so the per-env feasible alpha is full-precision / per-env identical.
    void buildBVH_and_CP_perenv_CCD(double alpha,
                                    const double* alpha_dev = nullptr);

    AABB* calcuMaxSceneSize();

    void buildBVH_FULLCCD(const double& alpha,
                          const double* alpha_dev = nullptr);
    void step_forward(device_TetraData& TetMesh, double alpha = 1.0, bool move_boundary = false, const double* alpha_dev = nullptr);


    void GroundCollisionDetect();
    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
    void calBarrierHessian();
    void calBarrierGradient(double3* _gradient, double mKap,
                            int2* ec_pair = nullptr, double3* ec_force = nullptr,
                            const int* ec_pbid = nullptr, double ec_inv_dt2 = 0.0,
                            bool use_group_kappa = true);

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

    void  computeGroundGradient(double3* _gradient, double mKap,
                                bool use_group_kappa = true);
    void computeSoftConstraintGradientAndHessian(double3* _gradient,
                                                 int global_hessian_fem_offset);

    void getTotalForce(double3* _gradient, double3* _gradient2);

    void   computeSoftConstraintGradient(double3* _gradient);
    double computeEnergy(device_TetraData& TetMesh);
    // Queue all FEM/contact/ABD reductions and the exact-order device combine
    // into out_scalar. No D2H or host synchronization.
    void computeEnergy_DeviceOut(device_TetraData& TetMesh, double* out_scalar);

    double Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh);
    // [FD gate] test-only: central-difference E vs assembled analytic gradient
#ifdef GIPC_ENABLE_DIAGNOSTICS
    FdCheckResult fd_gradient_check(device_TetraData& TetMesh,
                                    double h,
                                    int nprobes,
                                    unsigned seed);
    FdHessianResult fd_hessian_diagonal_check(device_TetraData& TetMesh,
                                              double h,
                                              int nprobes,
                                              unsigned seed);
    FdActivityResult fd_activity();
#endif
    // [E2] per-term registry members (defined in each term's energy/ file)
#define GIPC_ENERGY_TERM_DECL(id, name)                                        \
    int  energy_size_##name();                                                 \
    void energy_launch_##name(device_TetraData&, double*, int, int,            \
                              unsigned int, unsigned int, double*, const int*, \
                              int, int, int, double);
    GIPC_ENERGY_TERMS(GIPC_ENERGY_TERM_DECL)
#undef GIPC_ENERGY_TERM_DECL
    // [backport] standalone per-env energy dispatcher: writes the reduced global
    // scalar to a device slot (D2D) and, when out_penv != nullptr, buckets per-env.
    // Used only by computeEnergy_perenv. (Not the full ②-D2H computeEnergy rewrite.)
    void   Energy_Add_Reduction_Algorithm_DeviceOut(int type,
                                                    device_TetraData& TetMesh,
                                                    double* out_slot,
                                                    double* out_penv = nullptr,
                                                    double energy_kappa = -1.0);
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
    // ②-D2H batched variants — write direct feasible alpha to each slot.
    // Caller applies MIN and handles m_skip_all_collision / numbers<1 guards.
    void   ground_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, double* out_slot);
    void   self_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, int numbers, double* out_slot);
    void   self_full_largestFeasibleStepSize_DeviceOut(double slackness,
                                                       double* mqueue,
                                                       int numbers,
                                                       double* out_slot,
                                                       const uint32_t* d_live = nullptr);
    void   cfl_largestSpeed_DeviceOut(double* mqueue, double* out_slot);

    double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);

    double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);

    double cfl_largestSpeed(double* mqueue);

    bool lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha);
    // [C4-a] collision chain inside the whole-frame conditional graph.
    // Contract: kappa and the close-set stay frozen at their frame-boundary
    // values (postLineSearch does not run inside the graph) and friction
    // sets are whatever the last synchronous frame built — scenes must keep
    // friction coefficients at zero for sync-equivalence until C4-c.
    void train_collision_graph_capacities();
    void snapshotDcdCcdPairsCapture();
    void enqueue_ccd_alpha_conditional();
    void enqueue_pair_tier_guard();
    // [C4-b] non-null only while the whole-frame conditional graph records a
    // collision body: points at FrameDeviceState::kappa so barrier/ground
    // G/H and energy read the device-advanced kappa instead of the value
    // baked at capture. Every other path keeps by-value semantics.
    const double* graph_kappa_dev() const;
    // [C4-b] the postLineSearch equivalent enqueued at the Newton-step tail:
    // close-set check (device counts) -> conditional kappa doubling into
    // FrameDeviceState::kappa -> close-set rebuild at capacity grids.
    void enqueue_post_ls_kappa_conditional();
    // [C5] isolated-mode graph body pieces. The per-env CCD-alpha chain (S1
    // phases A/B + the device freeze decision into FrameDeviceState) and the
    // per-env S3 line-search WHILE loop. Contract: the recorded detection
    // pipeline is the merged tree with env-id emission filtering — the pair
    // SET matches the per-env-tree path (zero cross-env contacts) while the
    // build stays capture-safe; per-env solver decisions are all on device.
    void enqueue_perenv_ccd_alpha_conditional(device_TetraData& TetMesh);
    void enqueue_s3_line_search_conditional(device_TetraData& TetMesh);
    // [C5] frame-boundary training for the isolated bodies: every lazily
    // allocated per-env scratch reaches final size before capture.
    void train_perenv_graph_capacities(device_TetraData& TetMesh);
    // [C5] force the merged detection pipeline while the whole-frame graph
    // records/executes an isolated-mode frame (per-env BVH host loops are
    // not capture-safe; isolation is preserved by emission filtering).
    bool m_graph_merged_detect = false;

    // [C6] Graph training capacity. C4 originally shaped every recorded
    // launch from MAX_COLLITION_PAIRS_NUM — the WORST-CASE emission capacity.
    // That is fine for the small gate scenes (MAX_PAIRS ~2k) but catastrophic
    // for real ones: foldshirt's MAX_PAIRS is 737k, whose triplet envelope
    // needs ~44 GB. The recorded shape must instead follow the counts the
    // scene ACTUALLY produces, with headroom; overflow past the trained tier
    // is already adjudicated in-graph (OVF_* -> boundary retry -> re-record),
    // which is exactly the machinery that makes a smaller tier safe.
    // Per-ARITY extents. A single "pairs" number cannot size the triplet
    // stream: the assembly cost is tier(n4)*M12 + tier(n3)*M9 + tier(n2)*M6,
    // and treating every pair as if it were all three arities at once
    // over-allocates ~3x on top of the worst-case error.
    int m_graph_train_cp[5]  = {0, 0, 0, 0, 0};
    // [C6-o] Per-axis growth-streak escalation. A contact ratchet (towel
    // crumple) crosses ONE axis's tier per frame, and each crossing costs a
    // failed attempt + a whole-frame re-record -- 19 storm frames on the
    // A800. When the SAME axis re-crosses within an 8-frame window, its next
    // growth doubles once more per consecutive crossing (capped at 4x extra),
    // collapsing the storm to a few growth events. Axes: 0..4 = cp slots,
    // 5 = ccd, 6..9 = contact classes, 10 = abd unique blocks.
    int64_t m_axis_grow_last[11]   = {};
    int     m_axis_grow_streak[11] = {};
    // [C6-o/C6-aa] pinned {result, error_code, invalid_bits} snapshot the
    // host Newton loop polls; per-engine (freed in ~GIPC).
    int* m_capacity_poll_host = nullptr;
    int m_graph_train_pairs  = 0;   // trained DCD pair extent (slot 0)
    int m_last_assembled_triplets = 0;  // measured length of a real frame
    int m_graph_train_ground = 0;   // trained ground pair extent
    // Growth factor applied to the observed counts when training.
    // [C6-g] Tier headroom. This is a DETERMINISM knob, not just a memory one.
    // Contact counts are emitted by racy atomicAdd across two streams, so a
    // count sitting near a tier boundary crosses it on some runs and not
    // others; the frame then retries on some runs and not others, and a retry
    // is not physics-neutral (it perturbs the trajectory at ~1e-6, which
    // chaotic contact amplifies). Symptoms of headroom=2: G18 passes or fails
    // marginally run to run (squeeze/friction velocities 2.9e-7..3.5e-6 against
    // a 2.45e-7 envelope, with OVF_TRIPLETS retries firing intermittently), and
    // towel_scramble's crumple metric spreads 0.77..1.03 where the host is a
    // deterministic 0.905. Overridable so the trade-off stays measurable.
    static int graph_train_headroom_num()
    {
        if(const char* e = getenv("STIFF_GRAPH_TIER_HEADROOM"))
        {
            const int v = atoi(e);
            if(v >= 1 && v <= 64)
                return v;
        }
        // [C6-p] Default flipped 2 -> 1. Tier width is a launch-time
        // constant inside the recorded graph, and the 2x headroom doubled
        // every capacity-wide pass (converter unique-block reduction alone
        // was 86ms/frame at 8.4M slots over a 4.2M payload). Measured step
        // ratios vs graph-off, honest total wall:
        //   forcegrip 4090: 2.08x (h2) -> 1.39x (h1); A800: 3.25x -> 1.75x
        //   beaker    4090: 1.61x (h2) -> 1.21x (h1)
        // The old 2x bought fewer growth re-records; C6-o's per-axis streak
        // escalation now bounds ratchet storms, and the measured cost is
        // 1-2 extra growth frames per run (97% -> 94% coverage).
        return 1;
    }
    int graph_trained_pair_extent() const { return m_graph_train_pairs; }
    int graph_trained_ground_extent() const { return m_graph_train_ground; }
    // Swept (CCD) extent is trained INDEPENDENTLY of the DCD extent. They are
    // wildly different in practice — foldshirt emits ~27k DCD pairs but ~324k
    // swept ones — so deriving CCD from DCD by the buffer ratio forced a 4x
    // DCD inflation to satisfy a CCD need, and the triplet envelope (which
    // scales with the DCD side) blew past device memory.
    int m_graph_train_ccd = 0;
    int graph_trained_ccd_extent() const
    {
        if(m_graph_train_ccd <= 0)
            return MAX_CCD_COLLITION_PAIRS_NUM;
        return std::max(256,
                        std::min(m_graph_train_ccd,
                                 MAX_CCD_COLLITION_PAIRS_NUM));
    }
    void ensure_graph_friction_capacity();
    // [C6-l canon-slots] Deterministic pair-slot order under ee_canon: the
    // emission's atomicAdd slot assignment is racy, and every order-dependent
    // consumer (the LS energy trees above all) inherits that raciness in the
    // recorded replay. One stable lexicographic sort after each DCD build
    // makes the slot order a pure function of the pair SET.
    uint64_t* m_canon_keys[2]   = {nullptr, nullptr};
    uint32_t* m_canon_idx[2]    = {nullptr, nullptr};
    int4*     m_canon_pairs_tmp = nullptr;
    int4*     m_canon_ccd_tmp   = nullptr;
    int*      m_canon_mat_tmp   = nullptr;
    uint32_t* m_canon_gp_tmp    = nullptr;
    void*     m_canon_sort_tmp  = nullptr;
    size_t    m_canon_sort_tmp_bytes = 0;
    bool      m_canon_ready     = false;
    // [C6-l] True only while an EPISODE graph is being trained/captured. The
    // zero-observation bake clamp is episode-scoped: episodes have no per-
    // frame fallback (a failed frame aborts the episode transactionally), so
    // they must bake overflow-proof; step-mode frames prefer the C6-i host
    // fallback, whose parity with the baseline is what G19 asserts.
    bool      m_episode_capture = false;
    void      canonicalizePairSlots();
    void update_graph_training_capacity();
    void self_largestFeasibleStepSize_DeviceOut_Masked(double slackness,
                                                       double* mqueue,
                                                       int capacity,
                                                       double* out_slot,
                                                       const uint32_t* d_live);
    void lineSearchConditional(device_TetraData& TetMesh,
                               const double* alpha_device,
                               bool save_temp = true);
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
    void enqueue_frame_graph_body(device_TetraData& TetMesh);
    void IPC_Solver(device_TetraData& TetMesh);
    void IPC_Solver_FrameGraph(device_TetraData& TetMesh);
    void prepare_frame_graph(device_TetraData& TetMesh);
    void destroy_frame_graph();
    void prepare_episode_graph(
        device_TetraData& TetMesh,
        int frame_count,
        const gipc::RevoluteDrivingControlPacked* revolute_actions,
        int revolute_count,
        const gipc::PrismaticDrivingControlPacked* prismatic_actions,
        int prismatic_count,
        bool device_native = false);
    void launch_episode_graph_async(device_TetraData& TetMesh,
                                    int64_t base_frame_id);
    void launch_gpu_rl_graph_async(uintptr_t cuda_stream);
    // [D2/full-episode] Launch a pre-recorded multi-frame device-native
    // episode once. Unlike launch_gpu_rl_graph_async(), this path does not
    // require one host cudaGraphLaunch per frame; the captured conditional
    // episode loop consumes the pre-uploaded action slab on device.
    void launch_gpu_rl_episode_graph_async(uintptr_t cuda_stream);
    int  gpu_rl_episode_frame_count() const;
    bool gpu_rl_graph_prepared() const;
    bool gpu_rl_graph_ready() const;
    void synchronize_gpu_rl_graph() const;
    uintptr_t gpu_rl_revolute_actions_device_ptr() const;
    uintptr_t gpu_rl_prismatic_actions_device_ptr() const;
    uintptr_t gpu_rl_positions_device_ptr() const;
    uintptr_t gpu_rl_velocities_device_ptr() const;
    uintptr_t gpu_rl_statuses_device_ptr() const;
    uintptr_t gpu_rl_frame_counter_device_ptr() const;
    // [D2] packed {angle,rate}/{disp,rate} joint observations, refreshed by
    // the frame graph itself, plus the in-stream reset-to-snapshot replay.
    uintptr_t gpu_rl_joint_observations_device_ptr() const;
    int       gpu_rl_joint_observation_count() const;
    void      launch_gpu_rl_reset_async(uintptr_t cuda_stream = 0);
    void      launch_gpu_rl_reset_masked_async(uintptr_t d_env_mask,
                                                uintptr_t cuda_stream = 0);
    int gpu_rl_graph_node_count() const;
    int gpu_rl_graph_h2d_count() const;
    int gpu_rl_graph_d2h_count() const;
    bool episode_graph_in_flight() const;
    bool episode_observation_ready(int slot) const;
    void wait_episode_observation(int slot) const;
    int episode_slot_first_frame(int slot) const;
    int episode_slot_frame_count(int slot) const;
    int episode_attempted_frame_count() const;
    void copy_episode_observation_slot(
        int slot,
        double3* positions,
        double3* velocities,
        frame_fsm::FrameStatus* statuses,
        int frame_capacity) const;
    int finish_episode_graph();
    void destroy_episode_graph();
    void frame_graph_begin(device_TetraData& TetMesh,
                           int64_t frame_id,
                           int attempt = 0,
                           uint32_t retry_invalid_bits = 0);
    void frame_graph_enqueue_terminal(device_TetraData& TetMesh,
                                      int result,
                                      int error_code,
                                      uint32_t invalid_bits = 0,
                                      int err_env = -1,
                                      int err_primitive = -1);
    int frame_graph_finish_terminal();
    frame_fsm::FrameDeviceState* frame_graph_device_state() const;
    // [C6] graph-coverage telemetry (see frame_status.cu). Always counted;
    // STIFF_GRAPH_STATS=1 prints the summary at teardown.
    int  m_frames_committed  = 0;
    int  m_frames_full_graph = 0;
    int  m_frames_two_graph  = 0;
    void note_frame_graph_coverage(const frame_fsm::FrameStatus& status);
    void print_frame_graph_coverage(const char* tag = nullptr) const;
    void record_legacy_frame_status(bool graph_requested,
                                    bool callback_fallback,
                                    int newton_iterations = 0);
    const frame_fsm::FrameStatus& get_frame_status() const
    {
        return m_last_frame_status;
    }
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
    void update_joint_angle_targets_from_mesh(class tetrahedra_obj& tetMesh,
                                              double substep_ratio = 1.0);
    // [drive-substep] host mesh remembered by the per-frame target update so
    // solve_subIP can re-ramp targets each Newton iteration (STIFF_DRIVE_SUBSTEP=N).
    class tetrahedra_obj* m_drive_substep_mesh = nullptr;
    std::unique_ptr<gipc::ABDSimData>         m_abd_sim_data;
    std::unique_ptr<gipc::ABDSystem>          m_abd_system;
    std::unique_ptr<gipc::GlobalLinearSystem> m_global_linear_system;

    // Versioned frame-boundary integrator checkpoint. It restores FEM/ABD
    // motion, external force, kappa, isolation/quarantine and frame state;
    // ephemeral contact/friction sets are rebuilt from restored positions.
    void save_checkpoint(device_TetraData& tm, const char* path);
    void load_checkpoint(device_TetraData& tm, const char* path);

    // Pending per-body density overrides (body_id -> density), stashed by
    // SimEngine::set_abd_body_density before finalize. build_gipc_system
    // transfers these into m_abd_system right after it is created and before
    // the per-body mass setup runs.
    std::unordered_map<int, double>           m_pending_abd_density;

    // Pending total-mass overrides (body_id -> kilograms). Kept distinct from
    // density so external scene APIs cannot silently confuse kg with kg/m^3.
    std::unordered_map<int, double>           m_pending_abd_mass;

    // Pending per-body inertial overrides (body_id -> {mass, com[3],
    // inertia[3x3 row-major]}). Stashed by SimEngine::set_abd_body_inertia
    // before finalize; transferred to m_abd_system right after it is created.
    struct PendingInertia { double mass; double com[3]; double inertia[9]; };
    std::unordered_map<int, PendingInertia>   m_pending_abd_inertia;
};

// Internal regression hook: exercises the real device CCD tail with a NaN
// max-speed candidate and must throw before an invalid step can be accepted.
void stiff_test_ccd_nan_max_speed_fail_fast();

#endif
