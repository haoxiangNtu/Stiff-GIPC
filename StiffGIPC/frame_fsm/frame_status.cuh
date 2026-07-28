#pragma once
// Stable host/device ABI for Phase C whole-frame CUDA Graph execution.
//
// During a graph attempt the GPU owns FrameDeviceState.  The terminal node
// serializes exactly one FrameStatus packet, which is the only normal-path
// D2H transfer needed to adjudicate a frame at its boundary.

#include <cstdint>

namespace frame_fsm
{

enum FrameResult : int32_t
{
    FRAME_OK             = 0,
    FRAME_RETRY_REQUIRED = 1,
    FRAME_FATAL          = 2,
    FRAME_RUNTIME_ERROR  = 3,
};

enum FramePhase : int32_t
{
    PHASE_IDLE          = 0,
    PHASE_FRAME_BEGIN   = 1,
    PHASE_ASSEMBLY      = 2,
    PHASE_PCG           = 3,
    PHASE_NEWTON_DECIDE = 4,
    PHASE_CCD           = 5,
    PHASE_LINE_SEARCH   = 6,
    PHASE_POST_LS       = 7,
    PHASE_COMMIT        = 8,
    PHASE_ROLLBACK      = 9,
};

enum FrameInvalidBits : uint32_t
{
    INV_CCD_GROUND         = 1u << 0,
    INV_CCD_NARROW_SELF    = 1u << 1,
    INV_CCD_REFINED_SELF   = 1u << 2,
    INV_CCD_PE_GROUND      = 1u << 3,
    INV_CCD_PE_NARROW      = 1u << 4,
    INV_CCD_PE_REFINED     = 1u << 5,
    INV_LS_BUDGET          = 1u << 6,
    INV_START_INTERSECTING = 1u << 7,
    INV_NAN_STATE          = 1u << 8,
    OVF_DCD_PAIRS          = 1u << 16,
    OVF_CCD_PAIRS          = 1u << 17,
    OVF_TRIPLETS           = 1u << 18,
    OVF_UNIQUE_BLOCKS      = 1u << 19,
    OVF_MAS_CLUSTERS       = 1u << 20,
};

enum FramePathFlags : uint32_t
{
    PATH_GRAPH_REQUESTED         = 1u << 0,
    PATH_GRAPH_ACTIVE            = 1u << 1,
    PATH_LEGACY_FALLBACK         = 1u << 2,
    PATH_HOST_PHASE_BRIDGE       = 1u << 3,
    PATH_FULL_CONDITIONAL_GRAPH  = 1u << 4,
    PATH_TERMINAL_ROLLBACK       = 1u << 5,
    PATH_TEST_INJECTION          = 1u << 6,
    PATH_RETRIED                 = 1u << 7,
    PATH_PCG_DEVICE_CONTINUATION = 1u << 8,
    PATH_LS_DEVICE_LOOP          = 1u << 9,
    PATH_EPISODE_RESIDENT        = 1u << 10,
};

enum FrameErrorCode : int32_t
{
    ERR_NONE             = 0,
    ERR_CAPACITY         = 1,
    ERR_NONFINITE_STATE  = 2,
    ERR_SOLVER_EXCEPTION = 3,
    ERR_RETRY_EXHAUSTED  = 4,
    ERR_GRAPH_LAUNCH     = 5,
    ERR_CCD_INVALID      = 6,
};

struct alignas(16) FrameStatus
{
    int32_t  result;
    int32_t  phase;
    uint32_t invalid_bits;
    int32_t  launch_status;

    int32_t err_env;
    int32_t err_primitive;
    int32_t err_newton_iter;
    int32_t err_ls_iter;

    int32_t  error_code;
    uint32_t path_flags;
    int32_t  graph_launches;
    int32_t  host_boundaries;

    int32_t substeps;
    int32_t newton_iters;
    int32_t pcg_iters;
    int32_t ls_trials;

    int32_t hw_dcd_pairs;
    int32_t hw_ccd_pairs;
    int32_t hw_triplets;
    int32_t hw_unique_blocks;
    int32_t required_dcd_pairs;
    int32_t required_ccd_pairs;
    int32_t required_triplets;
    int32_t required_unique_blocks;
    int32_t hw_mas_clusters;
    int32_t required_mas_clusters;

    int32_t root_graph_nodes;
    int32_t root_d2h_nodes;
    int32_t terminal_graph_nodes;
    int32_t terminal_d2h_nodes;

    double final_alpha;
    double final_energy;
    double max_movement;
    double cfl_alpha;
    double kappa;

    int64_t  frame_id;
    int32_t  attempt;
    int32_t  retry_count;
    uint32_t retry_invalid_bits;
    int32_t  _pad0;
};
static_assert(sizeof(FrameStatus) <= 256,
              "FrameStatus must remain a single small D2H packet");

struct alignas(16) FrameDeviceState
{
    int32_t newton_iter;
    int32_t pcg_iter_total;
    int32_t ls_trial;
    int32_t substep;
    int32_t newton_converged;
    int32_t ls_decision;
    int32_t phase;
    int32_t result;

    int32_t  error_code;
    uint32_t invalid_bits;
    int32_t  err_env;
    int32_t  err_primitive;
    int32_t  err_newton_iter;
    int32_t  err_ls_iter;
    uint32_t path_flags;
    int32_t  host_boundaries;

    int32_t cp_count;
    int32_t gp_count;
    int32_t ccd_count;
    int32_t triplet_count;
    int32_t unique_count;
    int32_t hw_dcd_pairs;
    int32_t hw_ccd_pairs;
    int32_t hw_triplets;
    int32_t hw_unique_blocks;
    int32_t hw_mas_clusters;
    int32_t required_dcd_pairs;
    int32_t required_ccd_pairs;
    int32_t required_triplets;
    int32_t required_unique_blocks;
    int32_t required_mas_clusters;

    double alpha;
    double cfl_alpha;
    double energy_E0;
    double energy_trial;
    double max_movement;
    double kappa;

    int64_t frame_id;
    int32_t attempt;
    int32_t retry_count;
    uint32_t retry_invalid_bits;
    int32_t _pad1;
};

#if defined(__CUDACC__)
__device__ __forceinline__ void fsm_record_error(FrameDeviceState* state,
                                                 int32_t code,
                                                 uint32_t bits,
                                                 int32_t env,
                                                 int32_t primitive)
{
    if(atomicCAS(&state->error_code, ERR_NONE, code) == ERR_NONE)
    {
        state->err_env         = env;
        state->err_primitive   = primitive;
        state->err_newton_iter = state->newton_iter;
        state->err_ls_iter     = state->ls_trial;
    }
    atomicOr(&state->invalid_bits, bits);
}
#endif

}  // namespace frame_fsm
