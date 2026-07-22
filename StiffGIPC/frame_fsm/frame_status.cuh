#pragma once
// [frame-fsm P0] Device-resident frame control state and the per-frame status
// block. This is the ABI between the tail-launch state-machine graphs (P1..P3)
// and the host: during a frame the GPU owns FrameDeviceState; at the frame
// boundary the terminal graph serializes ONE FrameStatus which the host reads
// after its single stream synchronization.
//
// Contract (docs/FRAME_FSM_PLAN.md):
//  - normal committed frame: 1 root launch + 1 stream sync + 1 FrameStatus;
//  - overflow attempt: device-side rollback to the frame-begin snapshot,
//    status RETRY_REQUIRED with required_* capacities; the host grows buffers,
//    rebuilds affected tier graphs and relaunches the SAME frame id;
//  - fatal error: first-failure context recorded via fsm_record_error()
//    (atomicCAS keeps the FIRST failure only), successors are skipped by the
//    state transition, the rollback terminal still emits one FrameStatus.
//
// POD-only, fixed layout, no Eigen/no virtuals: the block is copied D2H as raw
// bytes and must be stable across TUs and (later) across the Python binding.

#include <cstdint>

namespace frame_fsm
{

// ---- result / phase codes --------------------------------------------------

enum FrameResult : int32_t
{
    FRAME_OK             = 0,   // frame committed
    FRAME_RETRY_REQUIRED = 1,   // capacity overflow: rolled back, host must grow + relaunch
    FRAME_FATAL          = 2,   // physics/numerical invariant violated (see error ctx)
    FRAME_RUNTIME_ERROR  = 3,   // device-side launch/API failure (see launch_status)
};

enum FramePhase : int32_t
{
    PHASE_IDLE          = 0,
    PHASE_FRAME_BEGIN   = 1,   // snapshot / input copy / initKappa / friction
    PHASE_ASSEMBLY      = 2,   // gradient/Hessian + sparse convert (tiered)
    PHASE_PCG           = 3,   // PCG batch self-tail loop
    PHASE_NEWTON_DECIDE = 4,
    PHASE_CCD           = 5,
    PHASE_LINE_SEARCH   = 6,   // LS trial self-tail loop
    PHASE_POST_LS       = 7,
    PHASE_COMMIT        = 8,
    PHASE_ROLLBACK      = 9,
};

// Overflow / invalid origin bits (extends the CCD six-bit scheme).
enum FrameInvalidBits : uint32_t
{
    INV_CCD_GROUND         = 1u << 0,
    INV_CCD_NARROW_SELF    = 1u << 1,
    INV_CCD_REFINED_SELF   = 1u << 2,
    INV_CCD_PE_GROUND      = 1u << 3,
    INV_CCD_PE_NARROW      = 1u << 4,
    INV_CCD_PE_REFINED     = 1u << 5,
    INV_LS_BUDGET          = 1u << 6,   // line-search budget exhausted (accepted loudly)
    INV_START_INTERSECTING = 1u << 7,   // isIntersected budget: infeasible start
    INV_NAN_STATE          = 1u << 8,   // non-finite positions/velocities detected
    OVF_DCD_PAIRS          = 1u << 16,
    OVF_CCD_PAIRS          = 1u << 17,
    OVF_TRIPLETS           = 1u << 18,
    OVF_UNIQUE_BLOCKS      = 1u << 19,
    OVF_MAS_CLUSTERS       = 1u << 20,
};

// Execution-path facts are reported separately from the result.  In
// particular, P3b-1 deliberately exposes its host Newton boundaries instead
// of pretending that the whole-frame one-root contract has already been met.
enum FramePathFlags : uint32_t
{
    PATH_GRAPH_REQUESTED       = 1u << 0,
    PATH_GRAPH_ACTIVE          = 1u << 1,
    PATH_LEGACY_FALLBACK       = 1u << 2,
    PATH_P3B1_HOST_NEWTON      = 1u << 3,
    PATH_P3B2_FULL_TAIL        = 1u << 4,
    PATH_TERMINAL_ROLLBACK     = 1u << 5,
    PATH_TEST_INJECTION        = 1u << 6,
};

enum FrameErrorCode : int32_t
{
    ERR_NONE              = 0,
    ERR_CAPACITY          = 1,
    ERR_NONFINITE_STATE   = 2,
    ERR_SOLVER_EXCEPTION  = 3,
    ERR_RETRY_EXHAUSTED   = 4,
    ERR_GRAPH_LAUNCH      = 5,
};

// ---- per-frame status block (device writes, host reads once per frame) -----

struct alignas(16) FrameStatus
{
    // outcome
    int32_t  result;          // FrameResult
    int32_t  phase;           // FramePhase at termination
    uint32_t invalid_bits;    // FrameInvalidBits
    int32_t  launch_status;   // device-side cudaGraphLaunch/cudaError code, 0 = OK

    // first-failure context (valid when result != FRAME_OK)
    int32_t  err_env;         // -1 = global
    int32_t  err_primitive;   // pair/vertex/edge/face id, -1 = n/a
    int32_t  err_newton_iter;
    int32_t  err_ls_iter;

    int32_t  error_code;      // FrameErrorCode; first failure wins
    uint32_t path_flags;      // FramePathFlags
    int32_t  graph_launches;  // root + terminal graph launches for this attempt
    int32_t  host_boundaries; // host-controlled Newton/phase boundaries (audit)

    // work counters (always valid; device-accumulated, host reads at boundary)
    int32_t  substeps;
    int32_t  newton_iters;
    int32_t  pcg_iters;
    int32_t  ls_trials;

    // capacity high-water + requirements (RETRY_REQUIRED fills required_*)
    int32_t  hw_dcd_pairs;
    int32_t  hw_ccd_pairs;
    int32_t  hw_triplets;
    int32_t  hw_unique_blocks;
    int32_t  required_dcd_pairs;
    int32_t  required_ccd_pairs;
    int32_t  required_triplets;
    int32_t  required_unique_blocks;
    int32_t  hw_mas_clusters;
    int32_t  required_mas_clusters;

    // Graph audit.  A compliant graph has no D2H in root and exactly one D2H
    // node in its terminal executable.
    int32_t  root_graph_nodes;
    int32_t  root_d2h_nodes;
    int32_t  terminal_graph_nodes;
    int32_t  terminal_d2h_nodes;

    // key scalars for observability (last committed Newton iteration)
    double   final_alpha;
    double   final_energy;
    double   max_movement;
    double   cfl_alpha;
    double   kappa;

    // frame identity (host writes before launch, device echoes back)
    int64_t  frame_id;
    int32_t  attempt;         // 0 = first attempt, >0 = retry count
    int32_t  _pad0;
};
static_assert(sizeof(FrameStatus) <= 256, "FrameStatus must stay one D2H packet");

// ---- device-resident control state (GPU-owned during a frame) --------------

struct alignas(16) FrameDeviceState
{
    // loop control
    int32_t  newton_iter;
    int32_t  pcg_iter_total;
    int32_t  ls_trial;
    int32_t  substep;
    int32_t  newton_converged;   // 0/1 (owned by the Newton decide kernel)
    int32_t  ls_decision;        // 0=accept 1=halve 2=tolerance-accept
    int32_t  phase;              // FramePhase (transition kernels advance it)
    int32_t  result;             // FrameResult, FRAME_OK until proven otherwise

    // error slot: first failure wins (fsm_record_error)
    int32_t  error_code;         // 0 = none; else FrameResult-compatible code
    uint32_t invalid_bits;
    int32_t  err_env;
    int32_t  err_primitive;
    int32_t  err_newton_iter;
    int32_t  err_ls_iter;
    uint32_t path_flags;
    int32_t  host_boundaries;

    // live counts (device truth; capacity guards compare against tier caps)
    int32_t  cp_count;
    int32_t  gp_count;
    int32_t  ccd_count;
    int32_t  triplet_count;
    int32_t  unique_count;
    int32_t  hw_dcd_pairs;
    int32_t  hw_ccd_pairs;
    int32_t  hw_triplets;
    int32_t  hw_unique_blocks;
    int32_t  hw_mas_clusters;
    int32_t  required_dcd_pairs;
    int32_t  required_ccd_pairs;
    int32_t  required_triplets;
    int32_t  required_unique_blocks;
    int32_t  required_mas_clusters;

    // control scalars
    double   alpha;
    double   cfl_alpha;
    double   energy_E0;
    double   energy_trial;
    double   max_movement;
    double   kappa;

    int64_t  frame_id;
    int32_t  attempt;
    int32_t  _pad1;
};

#if defined(__CUDACC__)
// Record the FIRST failure only; later failures keep the original context so
// the terminal status points at the root cause, not a cascade symptom.
__device__ __forceinline__ void fsm_record_error(FrameDeviceState* st,
                                                 int32_t           code,
                                                 uint32_t          bits,
                                                 int32_t           env,
                                                 int32_t           prim)
{
    if(atomicCAS(&st->error_code, 0, code) == 0)
    {
        st->err_env       = env;
        st->err_primitive = prim;
        st->err_newton_iter = st->newton_iter;
        st->err_ls_iter     = st->ls_trial;
    }
    atomicOr(&st->invalid_bits, bits);
}
#endif

}  // namespace frame_fsm
