#include <chrono>
#include "GIPC.cuh"

#include "abd_system/abd_sim_data.h"
#include "abd_system/abd_system.h"
#include "cuda_tools/cuda_tools.h"
#include "frame_fsm/conditional_graph.h"
#include "gipc/statistics.h"
#include "linear_system/linear_system/global_linear_system.h"
#include "linear_system/utils/capacity_tier.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{
struct alignas(16) FrameBeginInput
{
    int64_t  frame_id           = 0;
    int32_t  attempt            = 0;
    uint32_t path_flags         = 0;
    uint32_t retry_invalid_bits = 0;
    int32_t  _pad0              = 0;
    double   kappa              = 0.0;
};

struct alignas(16) FrameTerminalInput
{
    int32_t  result        = frame_fsm::FRAME_OK;
    int32_t  error_code    = frame_fsm::ERR_NONE;
    uint32_t invalid_bits  = 0;
    int32_t  launch_status = 0;

    int32_t err_env       = -1;
    int32_t err_primitive = -1;
    int32_t newton_iters  = 0;
    int32_t pcg_iters     = 0;
    int32_t ls_trials     = 0;

    int32_t hw_dcd_pairs     = 0;
    int32_t hw_ccd_pairs     = 0;
    int32_t hw_triplets      = 0;
    int32_t hw_unique_blocks = 0;

    int32_t required_dcd_pairs     = 0;
    int32_t required_ccd_pairs     = 0;
    int32_t required_triplets      = 0;
    int32_t required_unique_blocks = 0;

    int32_t root_graph_nodes     = 0;
    int32_t root_d2h_nodes       = 0;
    int32_t terminal_graph_nodes = 0;
    int32_t terminal_d2h_nodes   = 0;
    int32_t graph_launches       = 2;
    int32_t host_boundaries      = 1;

    double final_alpha  = 1.0;
    double final_energy = 0.0;
    double max_movement = 0.0;
    double cfl_alpha    = 1.0;
    double kappa        = 0.0;
};

struct HostAttemptSnapshot
{
    int    total_frames          = 0;
    int    total_newton_iters    = 0;
    double total_pcg_iters       = 0.0;
    double total_collision_pairs = 0.0;
    double max_collision_pairs   = 0.0;
    double total_time_ms         = 0.0;
    double phase_time_ms[5]      = {0.0, 0.0, 0.0, 0.0, 0.0};
    double time_make_pd_ms       = 0.0;
    double animation_full_rate   = 0.0;
    double kappa                 = 0.0;
    uint32_t cp_count[5]         = {0, 0, 0, 0, 0};
    uint32_t cp_last[5]          = {0, 0, 0, 0, 0};
    uint32_t gp_count            = 0;
    uint32_t gp_last             = 0;
    uint32_t ccd_count           = 0;
    uint32_t last_ccd_count      = 0;
    uint32_t dcd_snapshot_count  = 0;
    int ls_exhausted_total       = 0;
    int ls_nonfinite_total       = 0;
    int triplet_offset           = 0;
    int collision_triplet_offset = 0;
};

struct FrameGraphContext
{
    cudaGraphExec_t root_exec     = nullptr;
    cudaGraphExec_t terminal_exec = nullptr;
    cudaGraphExec_t full_exec     = nullptr;

    frame_fsm::FrameDeviceState* d_state  = nullptr;
    frame_fsm::FrameStatus*      d_status = nullptr;
    FrameBeginInput*             d_begin  = nullptr;
    FrameTerminalInput*          d_terminal = nullptr;

    FrameBeginInput*             h_begin    = nullptr;
    FrameTerminalInput*          h_terminal = nullptr;
    frame_fsm::FrameStatus*      h_status   = nullptr;

    double3* fem_vertexes   = nullptr;
    double3* fem_o_vertexes = nullptr;
    double3* fem_velocities = nullptr;
    double3* fem_x_tilta    = nullptr;

    gipc::Vector12* abd_q       = nullptr;
    gipc::Vector12* abd_q_prev  = nullptr;
    gipc::Vector12* abd_q_v     = nullptr;
    gipc::Vector12* abd_q_tilde = nullptr;

    double* kappa_group          = nullptr;
    double* kappa_snapshot       = nullptr;

    size_t vertex_count = 0;
    size_t abd_count    = 0;
    size_t group_count  = 0;

    int root_nodes     = 0;
    int root_d2h       = 0;
    int terminal_nodes = 0;
    int terminal_d2h   = 0;
    int full_nodes     = 0;
    int full_d2h       = 0;
    int newton_begin   = 0;
    long long full_generation = -1;
    bool full_capture_failed   = false;

    // [C4-a] pair counts the collision executable was tiered against
    // (slots 0..5 = live DCD block, 6..11 = lagged friction block). The
    // boundary refresh compares live tiers to these and drops the executable
    // when a tier boundary is crossed, so masked kernels never under-launch.
    uint32_t captured_pair_counts[12] = {0};
    bool     captured_pair_counts_valid = false;

    HostAttemptSnapshot host_snapshot;
};

struct alignas(16) EpisodeRuntimeInput
{
    int64_t  base_frame_id = 0;
    int32_t  frame_count   = 0;
    uint32_t path_flags    = 0;
    int32_t  graph_nodes   = 0;
    int32_t  graph_d2h     = 0;
    int32_t  _pad0         = 0;
    double   kappa         = 0.0;
};

struct alignas(16) EpisodeDeviceControl
{
    int32_t frame_index      = 0;
    int32_t attempted_frames = 0;
    int32_t successful_frames = 0;
    int32_t terminal_result  = frame_fsm::FRAME_OK;
};

struct EpisodeGraphContext
{
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t slot_event[2] = {nullptr, nullptr};
    cudaEvent_t completion_event = nullptr;
    cudaStream_t launch_stream = nullptr;

    EpisodeRuntimeInput*  d_input   = nullptr;
    EpisodeDeviceControl* d_control = nullptr;
    int*                  d_ready_one = nullptr;
    int64_t*              d_frame_counter = nullptr;

    // [D2] device joint observations ({angle,rate} per revolute then
    // {disp,rate} per prismatic) refreshed by the frame graph itself.
    double* d_joint_obs      = nullptr;
    int     joint_obs_count  = 0;

    // [D2] in-stream reset snapshot: the committed dynamic state captured at
    // prepare time. launch_gpu_rl_reset_async replays it with pure D2D
    // copies on the caller's stream — no host synchronization.
    double3*        reset_fem_x    = nullptr;
    double3*        reset_fem_ox   = nullptr;
    double3*        reset_fem_v    = nullptr;
    double3*        reset_fem_xt   = nullptr;
    gipc::Vector12* reset_abd_q      = nullptr;
    gipc::Vector12* reset_abd_q_prev = nullptr;
    gipc::Vector12* reset_abd_q_v    = nullptr;
    gipc::Vector12* reset_abd_q_tilde = nullptr;
    size_t          reset_vertex_count = 0;
    int             reset_abd_count    = 0;
    // Live mesh buffers the reset replays into (grabbed at prepare time;
    // buffer generation churn forces a re-prepare before they can move).
    double3* mesh_vertexes   = nullptr;
    double3* mesh_o_vertexes = nullptr;
    double3* mesh_velocities = nullptr;
    double3* mesh_x_tilta    = nullptr;
    // [D3 mask-reset] env maps captured at prepare for the masked variant.
    const int* reset_p2g = nullptr;   // vertex -> env group (-1 = unowned)
    const int* reset_b2g = nullptr;   // ABD body -> env group

    EpisodeRuntimeInput* h_input = nullptr;
    int* h_ready            = nullptr;
    int* h_slot_attempted   = nullptr;

    unsigned char* d_actions = nullptr;
    gipc::RevoluteDrivingControlPacked* d_revolute_actions = nullptr;
    gipc::PrismaticDrivingControlPacked* d_prismatic_actions = nullptr;
    double3* d_positions  = nullptr;
    double3* d_velocities = nullptr;
    frame_fsm::FrameStatus* d_statuses = nullptr;

    double3* h_positions[2]  = {nullptr, nullptr};
    double3* h_velocities[2] = {nullptr, nullptr};
    frame_fsm::FrameStatus* h_statuses[2] = {nullptr, nullptr};

    int frame_count     = 0;
    int split_frame     = 0;
    int revolute_count  = 0;
    int prismatic_count = 0;
    size_t vertex_count = 0;
    int graph_nodes     = 0;
    int graph_h2d       = 0;
    int graph_d2h       = 0;
    size_t action_bytes = 0;
    size_t prismatic_action_offset = 0;
    long long generation = -1;
    bool in_flight       = false;
    bool device_native   = false;

    HostAttemptSnapshot host_snapshot;
};

bool knob_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] && std::atoi(value) != 0;
}

bool full_graph_eligible(const GIPC& ipc,
                         const device_TetraData& mesh,
                         const FrameGraphContext& context,
                         std::string& reason,
                         bool allow_abd = false)
{
    if(knob_enabled("STIFF_FRAME_FORCE_ROLLBACK")
       || knob_enabled("STIFF_FRAME_FORCE_UNIQUE_TIER"))
    {
        reason = "transaction fault injection requires the two-graph gate";
        return false;
    }
    if(std::getenv("NAN_DIAG"))
    {
        reason = "NAN_DIAG requests post-frame host diagnostics";
        return false;
    }
    for(const char* diagnostic : {
            "STIFF_MIRROR_AUDIT",
            "STIFF_SLOT_AUDIT",
            "STIFF_PHASE_TIME",
            "STIFF_KSUM",
            "STIFF_MAS_DUMP",
            "STIFF_MAS_FUSE_VALIDATE",
            "STIFF_STACK_DIAG"})
    {
        if(std::getenv(diagnostic))
        {
            reason =
                std::string(diagnostic)
                + " performs capture-incompatible host diagnostics";
            return false;
        }
    }
    // [C6] ABD in the ordinary step() whole-frame graph. The original gate
    // predates C4/D4: ABD + collision + friction is now proven to run inside
    // the graph and to REUSE one executable across frames (the GPU-native RL
    // path does 43 frames on a single 1089-node exec, zero transfers), and
    // the frame transaction already snapshots q / q_prev / q_v / q_tilde for
    // rollback. Opt-in first so the claim gets measured before it becomes the
    // default, exactly how C4/C5 were introduced.
    if(context.abd_count != 0 && !allow_abd
       && !knob_enabled("STIFF_C6_ABD_STEP_GRAPH"))
    {
        reason =
            "ABD bodies in ordinary step() need STIFF_C6_ABD_STEP_GRAPH=1";
        return false;
    }
    if(context.abd_count != 0
       && ipc.gipc_global_triplet.m_abd_unique_tier[1] <= 0)
    {
        reason =
            "ABD final assembly tier was not trained by the warm-up frame";
        return false;
    }
    if(mesh.n_fem_pins != 0)
    {
        reason = "FEM-to-ABD pins require the ABD transaction snapshot";
        return false;
    }
    if(!ipc.m_skip_all_collision)
    {
        if(!knob_enabled("STIFF_C4_COLLISION_GRAPH"))
        {
            reason =
                "collision in the whole-frame graph requires "
                "STIFF_C4_COLLISION_GRAPH=1 (kappa/close-set and friction "
                "sets stay frozen at frame-boundary values)";
            return false;
        }
        if(ipc.MAX_COLLITION_PAIRS_NUM <= 0
           || ipc.MAX_CCD_COLLITION_PAIRS_NUM <= 0)
        {
            reason = "collision pair tiers were never allocated";
            return false;
        }
    }
    if(ipc.m_update_boundary)
    {
        reason = "moving boundaries are active";
        return false;
    }
    // [C6] Soft targets are only a blocker when the HOST owns them. Bilateral
    // stitch springs (every gripper scene) are recomputed on device inside the
    // assembly kernel, so they are graph-safe; a user functor or a plain
    // pinned target still needs the per-frame host pass and stays out.
    if(ipc.softNum != 0 && !mesh.soft_targets_are_device_resident())
    {
        reason = "host-owned soft targets are active";
        return false;
    }
    if(ipc.semi_implicit_enabled)
    {
        reason = "semi-implicit beta is still host-owned";
        return false;
    }
    if(ipc.animation_subRate < 0.999999)
    {
        reason = "multi-substep animation needs an outer device loop";
        return false;
    }

    const ModeConfig& mode = ipc.m_mode_config;
    // [C5] isolated mode is eligible as a COMPLETE bundle. strict stays out:
    // its promise is bitwise reproducibility, and capacity-grid reductions
    // legally reassociate the sums — admitting strict would be an anchor
    // change, not a residency change.
    // [C6-l] spmv_det is admitted. It is a numeric-path choice (deterministic
    // binned SpMV/gradient reductions), not the strict anchor: strict's
    // bitwise promise is what capacity-grid reassociation breaks, and that
    // rejection stays below. Its lazy binned accumulators are primed outside
    // capture by the pre-capture dry run (the same pattern the segmented PCG
    // device loop already uses), so nothing allocates inside the recording.
    // Measured payoff: det-SpMV kills the armed-layout SpMV order-raciness
    // (towel pre-contact divergence gone, C6-k), making this the determinism
    // lever for the shipped whole-frame config.
    // ee_canon is likewise admitted (C6-l): canonical pair ORDER is exactly
    // what makes the in-graph contact-energy sums order-free -- without it the
    // recorded replay sums the same pair multiset in a racy slot permutation
    // and one towel run in two explodes at first contact (newton=1000,
    // ls=20586, alpha=1e-25) before the C6-i fallback rescues it.
    if(mode.mode == ModeConfig::Strict || mode.ee_detgate || mode.ccd_canon)
    {
        reason = "strict determinism controls are active";
        return false;
    }
    if(mode.mode == ModeConfig::Isolated)
    {
        if(!knob_enabled("STIFF_C5_ISOLATED_GRAPH"))
        {
            reason =
                "isolated whole-frame graph requires "
                "STIFF_C5_ISOLATED_GRAPH=1";
            return false;
        }
        if(ipc.m_skip_all_collision)
        {
            reason =
                "isolated mode without collision has no per-env alpha chain";
            return false;
        }
        if(!knob_enabled("STIFF_C4_COLLISION_GRAPH"))
        {
            reason =
                "isolated mode implies collision in the graph "
                "(STIFF_C4_COLLISION_GRAPH=1)";
            return false;
        }
        if(ipc.m_active_group_count <= 0)
        {
            reason = "isolated mode without declared environment groups";
            return false;
        }
        // Telemetry/iteration-cap overlays route the solver through the host
        // diagnostic S1 path (5 x NG D2H + host loop) — no recorded form.
        if(mode.perenv_telem || ipc.env_newton_iter_cap > 0)
        {
            reason =
                "per-env telemetry / iteration cap use the host diagnostic "
                "S1 path";
            return false;
        }
        return true;
    }
    const bool partial_overlay =
        mode.bvh_envdet || mode.perenv_bvh
        || mode.decouple_thresh || mode.pergroup_kappa
        || mode.segmented_pcg || mode.perenv_alpha || mode.perenv_par
        || mode.perenv_telem
        || mode.perenv_mask || mode.perenv_mask_dev;
    if(partial_overlay)
    {
        reason = "partial per-env solver overlay on merged mode";
        return false;
    }
    return true;
}

template <typename T>
void device_alloc(T*& pointer, size_t count)
{
    if(count == 0)
        return;
    CUDA_SAFE_CALL(
        cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
}

template <typename T>
void device_free(T*& pointer)
{
    if(pointer)
        cudaFree(pointer);
    pointer = nullptr;
}

struct GraphAudit
{
    int node_count = 0;
    int h2d_count  = 0;
    int d2h_count  = 0;
    int host_count = 0;
};

void accumulate_graph_audit(
    cudaGraph_t                         graph,
    GraphAudit&                         audit,
    const std::vector<cudaGraphNode_t>* conditional_nodes = nullptr)
{
    size_t count = 0;
    CUDA_SAFE_CALL(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    if(count)
        CUDA_SAFE_CALL(cudaGraphGetNodes(graph, nodes.data(), &count));
    audit.node_count += static_cast<int>(count);
    for(cudaGraphNode_t node : nodes)
    {
        // CUDA 12.8 may return cudaErrorUnknown when querying the type of a
        // conditional node obtained from cudaGraphAddNode, even though the
        // graph is valid and instantiates successfully.  The recorder already
        // owns the authoritative handles, so audit their bodies separately
        // and avoid passing those opaque nodes back through GetType.
        if(conditional_nodes
           && std::find(conditional_nodes->begin(),
                        conditional_nodes->end(),
                        node)
                  != conditional_nodes->end())
            continue;
        cudaGraphNodeType type{};
        CUDA_SAFE_CALL(cudaGraphNodeGetType(node, &type));
        if(type == cudaGraphNodeTypeHost)
        {
            ++audit.host_count;
            continue;
        }
        if(type == cudaGraphNodeTypeMemcpy)
        {
            cudaMemcpy3DParms params{};
            CUDA_SAFE_CALL(cudaGraphMemcpyNodeGetParams(node, &params));
            if(params.kind == cudaMemcpyHostToDevice)
                ++audit.h2d_count;
            else if(params.kind == cudaMemcpyDeviceToHost)
                ++audit.d2h_count;
            continue;
        }
        if(type == cudaGraphNodeTypeGraph)
        {
            cudaGraph_t child = nullptr;
            CUDA_SAFE_CALL(
                cudaGraphChildGraphNodeGetGraph(node, &child));
            if(child)
                accumulate_graph_audit(
                    child, audit, conditional_nodes);
        }
    }
}

void audit_graph(cudaGraph_t graph, int& node_count, int& d2h_count)
{
    GraphAudit audit;
    accumulate_graph_audit(graph, audit);
    node_count = audit.node_count;
    d2h_count  = audit.d2h_count;
}

__global__ void frame_begin_init(frame_fsm::FrameDeviceState* state,
                                 const FrameBeginInput* input)
{
    if(blockIdx.x || threadIdx.x)
        return;
    *state                  = frame_fsm::FrameDeviceState{};
    state->phase            = frame_fsm::PHASE_FRAME_BEGIN;
    state->result           = frame_fsm::FRAME_OK;
    state->err_env          = -1;
    state->err_primitive    = -1;
    state->err_newton_iter  = -1;
    state->err_ls_iter      = -1;
    state->alpha            = 1.0;
    state->cfl_alpha        = 1.0;
    state->frame_id         = input->frame_id;
    state->attempt          = input->attempt;
    state->retry_count      = input->attempt;
    state->retry_invalid_bits = input->retry_invalid_bits;
    state->path_flags       = input->path_flags;
    state->kappa            = input->kappa;
}

__global__ void frame_terminal_apply(
    frame_fsm::FrameDeviceState* state,
    const FrameTerminalInput* input)
{
    if(blockIdx.x || threadIdx.x)
        return;
    state->newton_iter      = input->newton_iters;
    state->pcg_iter_total  += input->pcg_iters;
    state->ls_trial         = input->ls_trials;
    state->hw_dcd_pairs =
        state->hw_dcd_pairs > input->hw_dcd_pairs
            ? state->hw_dcd_pairs
            : input->hw_dcd_pairs;
    state->hw_ccd_pairs =
        state->hw_ccd_pairs > input->hw_ccd_pairs
            ? state->hw_ccd_pairs
            : input->hw_ccd_pairs;
    state->hw_triplets =
        state->hw_triplets > input->hw_triplets
            ? state->hw_triplets
            : input->hw_triplets;
    state->hw_unique_blocks =
        state->hw_unique_blocks > input->hw_unique_blocks
            ? state->hw_unique_blocks
            : input->hw_unique_blocks;
    state->required_dcd_pairs =
        state->required_dcd_pairs > input->required_dcd_pairs
            ? state->required_dcd_pairs
            : input->required_dcd_pairs;
    state->required_ccd_pairs =
        state->required_ccd_pairs > input->required_ccd_pairs
            ? state->required_ccd_pairs
            : input->required_ccd_pairs;
    state->required_triplets =
        state->required_triplets > input->required_triplets
            ? state->required_triplets
            : input->required_triplets;
    state->required_unique_blocks =
        state->required_unique_blocks > input->required_unique_blocks
            ? state->required_unique_blocks
            : input->required_unique_blocks;
    state->alpha            = input->final_alpha;
    state->energy_trial     = input->final_energy;
    state->max_movement     = input->max_movement;
    state->cfl_alpha        = input->cfl_alpha;
    state->kappa            = input->kappa;

    if(input->error_code != frame_fsm::ERR_NONE)
    {
        frame_fsm::fsm_record_error(state,
                                    input->error_code,
                                    input->invalid_bits,
                                    input->err_env,
                                    input->err_primitive);
        state->result = input->result;
    }
    else if(input->invalid_bits)
    {
        atomicOr(&state->invalid_bits, input->invalid_bits);
        state->result = input->result;
    }
    else if(state->result == frame_fsm::FRAME_OK)
    {
        state->result = input->result;
    }
    if(state->result != frame_fsm::FRAME_OK)
        state->path_flags |= frame_fsm::PATH_TERMINAL_ROLLBACK;
    state->phase  = state->result == frame_fsm::FRAME_OK
                        ? frame_fsm::PHASE_COMMIT
                        : frame_fsm::PHASE_ROLLBACK;
}

__global__ void frame_terminal_finalize_full(
    frame_fsm::FrameDeviceState* state)
{
    if(blockIdx.x || threadIdx.x)
        return;
    state->host_boundaries = 1;
    if(state->result != frame_fsm::FRAME_OK)
        state->path_flags |= frame_fsm::PATH_TERMINAL_ROLLBACK;
    state->phase = state->result == frame_fsm::FRAME_OK
                       ? frame_fsm::PHASE_COMMIT
                       : frame_fsm::PHASE_ROLLBACK;
}

__global__ void frame_validate_finite(frame_fsm::FrameDeviceState* state,
                                      const double3* positions,
                                      const double3* velocities,
                                      int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count || state->result != frame_fsm::FRAME_OK)
        return;
    const double3 p = positions[index];
    const double3 v = velocities[index];
    const bool invalid =
        !isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)
        || !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z);
    if(!invalid)
        return;
    frame_fsm::fsm_record_error(state,
                                frame_fsm::ERR_NONFINITE_STATE,
                                frame_fsm::INV_NAN_STATE,
                                -1,
                                index);
    atomicCAS(&state->result,
              frame_fsm::FRAME_OK,
              frame_fsm::FRAME_FATAL);
    state->phase = frame_fsm::PHASE_ROLLBACK;
}

__global__ void frame_restore_fem(
    const frame_fsm::FrameDeviceState* state,
    double3* positions,
    double3* old_positions,
    double3* velocities,
    double3* x_tilta,
    const double3* snap_positions,
    const double3* snap_old_positions,
    const double3* snap_velocities,
    const double3* snap_x_tilta,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count || state->result == frame_fsm::FRAME_OK)
        return;
    positions[index]     = snap_positions[index];
    old_positions[index] = snap_old_positions[index];
    velocities[index]    = snap_velocities[index];
    x_tilta[index]       = snap_x_tilta[index];
}

__global__ void frame_restore_abd(
    const frame_fsm::FrameDeviceState* state,
    gipc::Vector12* q,
    gipc::Vector12* q_prev,
    gipc::Vector12* q_v,
    gipc::Vector12* q_tilde,
    const gipc::Vector12* snap_q,
    const gipc::Vector12* snap_q_prev,
    const gipc::Vector12* snap_q_v,
    const gipc::Vector12* snap_q_tilde,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count || state->result == frame_fsm::FRAME_OK)
        return;
    q[index]       = snap_q[index];
    q_prev[index]  = snap_q_prev[index];
    q_v[index]     = snap_q_v[index];
    q_tilde[index] = snap_q_tilde[index];
}

__global__ void frame_restore_kappa(
    const frame_fsm::FrameDeviceState* state,
    double* live,
    const double* snapshot,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < count && state->result != frame_fsm::FRAME_OK)
        live[index] = snapshot[index];
}

__global__ void frame_serialize_status(
    const frame_fsm::FrameDeviceState* state,
    const FrameTerminalInput* input,
    frame_fsm::FrameStatus* status)
{
    if(blockIdx.x || threadIdx.x)
        return;
    frame_fsm::FrameStatus out{};
    out.result          = state->result;
    out.phase           = state->phase;
    out.invalid_bits    = state->invalid_bits;
    out.launch_status   = input->launch_status;
    out.err_env         = state->err_env;
    out.err_primitive   = state->err_primitive;
    out.err_newton_iter = state->err_newton_iter;
    out.err_ls_iter     = state->err_ls_iter;
    out.error_code      = state->error_code;
    out.path_flags      = state->path_flags;
    out.graph_launches  = input->graph_launches;
    out.host_boundaries = input->host_boundaries;
    out.newton_iters    = state->newton_iter;
    out.pcg_iters       = state->pcg_iter_total;
    out.ls_trials       = state->ls_trial;
    out.hw_dcd_pairs    = state->hw_dcd_pairs;
    out.hw_ccd_pairs    = state->hw_ccd_pairs;
    out.hw_triplets     = state->hw_triplets;
    out.hw_unique_blocks = state->hw_unique_blocks;
    out.required_dcd_pairs = state->required_dcd_pairs;
    out.required_ccd_pairs = state->required_ccd_pairs;
    out.required_triplets = state->required_triplets;
    out.required_unique_blocks = state->required_unique_blocks;
    for(int i = 0; i < 4; ++i)
        out.contact_class_count[i] = state->contact_class_count[i];
    out.root_graph_nodes = input->root_graph_nodes;
    out.root_d2h_nodes = input->root_d2h_nodes;
    out.terminal_graph_nodes = input->terminal_graph_nodes;
    out.terminal_d2h_nodes = input->terminal_d2h_nodes;
    out.final_alpha    = state->alpha;
    out.final_energy   = state->energy_trial;
    out.max_movement   = state->max_movement;
    out.cfl_alpha      = state->cfl_alpha;
    out.kappa          = state->kappa;
    out.frame_id       = state->frame_id;
    out.attempt        = state->attempt;
    out.retry_count    = state->retry_count;
    out.retry_invalid_bits = state->retry_invalid_bits;
    *status = out;
}

__global__ void episode_initialize(EpisodeDeviceControl* control)
{
    if(blockIdx.x || threadIdx.x)
        return;
    *control = EpisodeDeviceControl{};
}

__global__ void episode_frame_begin(
    frame_fsm::FrameDeviceState* state,
    const EpisodeDeviceControl* control,
    const EpisodeRuntimeInput* input,
    const int64_t* frame_counter)
{
    if(blockIdx.x || threadIdx.x)
        return;
    *state                 = frame_fsm::FrameDeviceState{};
    state->phase           = frame_fsm::PHASE_FRAME_BEGIN;
    state->result          = frame_fsm::FRAME_OK;
    state->err_env         = -1;
    state->err_primitive   = -1;
    state->err_newton_iter = -1;
    state->err_ls_iter     = -1;
    state->alpha           = 1.0;
    state->cfl_alpha       = 1.0;
    state->frame_id =
        (input->path_flags & frame_fsm::PATH_GPU_NATIVE_RL)
            ? *frame_counter + control->frame_index
            : input->base_frame_id + control->frame_index;
    state->path_flags = input->path_flags;
    state->kappa      = input->kappa;
}

__global__ void episode_advance_frame_counter(
    const EpisodeDeviceControl* control,
    int64_t* frame_counter)
{
    if(blockIdx.x || threadIdx.x)
        return;
    *frame_counter += control->attempted_frames;
}

__global__ void episode_store_frame(
    const frame_fsm::FrameDeviceState* state,
    EpisodeDeviceControl* control,
    const EpisodeRuntimeInput* input,
    const double3* positions,
    const double3* velocities,
    double3* episode_positions,
    double3* episode_velocities,
    frame_fsm::FrameStatus* statuses,
    int vertex_count)
{
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame  = control->frame_index;
    if(vertex < vertex_count)
    {
        const size_t offset =
            static_cast<size_t>(frame) * vertex_count + vertex;
        episode_positions[offset]  = positions[vertex];
        episode_velocities[offset] = velocities[vertex];
    }
    if(vertex != 0)
        return;

    frame_fsm::FrameStatus out{};
    out.result          = state->result;
    out.phase           = state->phase;
    out.invalid_bits    = state->invalid_bits;
    out.launch_status   = 0;
    out.err_env         = state->err_env;
    out.err_primitive   = state->err_primitive;
    out.err_newton_iter = state->err_newton_iter;
    out.err_ls_iter     = state->err_ls_iter;
    out.error_code      = state->error_code;
    out.path_flags      = state->path_flags;
    out.graph_launches  = 1;
    out.host_boundaries = 0;
    out.substeps        = 1;
    out.newton_iters    = state->newton_iter;
    out.pcg_iters       = state->pcg_iter_total;
    out.ls_trials       = state->ls_trial;
    out.hw_dcd_pairs    = state->hw_dcd_pairs;
    out.hw_ccd_pairs    = state->hw_ccd_pairs;
    out.hw_triplets     = state->hw_triplets;
    out.hw_unique_blocks = state->hw_unique_blocks;
    out.required_dcd_pairs = state->required_dcd_pairs;
    out.required_ccd_pairs = state->required_ccd_pairs;
    out.required_triplets = state->required_triplets;
    out.required_unique_blocks = state->required_unique_blocks;
    out.hw_mas_clusters = state->hw_mas_clusters;
    out.required_mas_clusters = state->required_mas_clusters;
    for(int i = 0; i < 4; ++i)
        out.contact_class_count[i] = state->contact_class_count[i];
    out.root_graph_nodes = input->graph_nodes;
    out.root_d2h_nodes   = input->graph_d2h;
    out.terminal_graph_nodes = 0;
    out.terminal_d2h_nodes   = 0;
    out.final_alpha    = state->alpha;
    out.final_energy   = state->energy_trial;
    out.max_movement   = state->max_movement;
    out.cfl_alpha      = state->cfl_alpha;
    out.kappa          = state->kappa;
    out.frame_id       = state->frame_id;
    out.attempt        = 0;
    out.retry_count    = 0;
    out.retry_invalid_bits = 0;
    statuses[frame] = out;

    control->attempted_frames = frame + 1;
    if(state->result == frame_fsm::FRAME_OK)
        control->successful_frames = frame + 1;
    else
        control->terminal_result = state->result;
}

__global__ void episode_remaining_predicate(
    const EpisodeDeviceControl* control,
    const EpisodeRuntimeInput* input,
    cudaGraphConditionalHandle handle)
{
    if(blockIdx.x || threadIdx.x)
        return;
    cudaGraphSetConditional(
        handle,
        control->terminal_result == frame_fsm::FRAME_OK
                && control->attempted_frames < input->frame_count
            ? 1u
            : 0u);
}

__global__ void episode_tail(
    const frame_fsm::FrameDeviceState* state,
    EpisodeDeviceControl* control,
    const EpisodeRuntimeInput* input,
    int iteration_limit,
    cudaGraphConditionalHandle handle)
{
    if(blockIdx.x || threadIdx.x)
        return;
    const int next = control->frame_index + 1;
    const bool keep_running =
        state->result == frame_fsm::FRAME_OK
        && next < iteration_limit;
    if(state->result == frame_fsm::FRAME_OK
       && next < input->frame_count)
        control->frame_index = next;
    cudaGraphSetConditional(handle, keep_running ? 1u : 0u);
}

void destroy_context(FrameGraphContext* context)
{
    if(!context)
        return;
    if(context->root_exec)
        cudaGraphExecDestroy(context->root_exec);
    if(context->terminal_exec)
        cudaGraphExecDestroy(context->terminal_exec);
    if(context->full_exec)
        cudaGraphExecDestroy(context->full_exec);
    device_free(context->d_state);
    device_free(context->d_status);
    device_free(context->d_begin);
    device_free(context->d_terminal);
    device_free(context->fem_vertexes);
    device_free(context->fem_o_vertexes);
    device_free(context->fem_velocities);
    device_free(context->fem_x_tilta);
    device_free(context->abd_q);
    device_free(context->abd_q_prev);
    device_free(context->abd_q_v);
    device_free(context->abd_q_tilde);
    device_free(context->kappa_group);
    device_free(context->kappa_snapshot);
    if(context->h_begin)
        cudaFreeHost(context->h_begin);
    if(context->h_terminal)
        cudaFreeHost(context->h_terminal);
    if(context->h_status)
        cudaFreeHost(context->h_status);
    delete context;
}

template <typename T>
void pinned_alloc(T*& pointer, size_t count)
{
    if(count == 0)
        return;
    CUDA_SAFE_CALL(cudaHostAlloc(
        reinterpret_cast<void**>(&pointer),
        count * sizeof(T),
        cudaHostAllocPortable));
}

template <typename T>
void pinned_free(T*& pointer)
{
    if(pointer)
        cudaFreeHost(pointer);
    pointer = nullptr;
}

size_t checked_product(size_t lhs, size_t rhs, const char* label)
{
    if(lhs && rhs > std::numeric_limits<size_t>::max() / lhs)
        throw std::overflow_error(
            std::string("[episode-graph] size overflow: ") + label);
    return lhs * rhs;
}

void destroy_episode_context(EpisodeGraphContext* context)
{
    if(!context)
        return;
    if(context->exec)
        cudaGraphExecDestroy(context->exec);
    if(context->completion_event)
        cudaEventDestroy(context->completion_event);
    for(cudaEvent_t& event : context->slot_event)
    {
        if(event)
            cudaEventDestroy(event);
        event = nullptr;
    }
    device_free(context->d_input);
    device_free(context->d_control);
    device_free(context->d_ready_one);
    device_free(context->d_frame_counter);
    device_free(context->d_joint_obs);
    device_free(context->reset_fem_x);
    device_free(context->reset_fem_ox);
    device_free(context->reset_fem_v);
    device_free(context->reset_fem_xt);
    device_free(context->reset_abd_q);
    device_free(context->reset_abd_q_prev);
    device_free(context->reset_abd_q_v);
    device_free(context->reset_abd_q_tilde);
    device_free(context->d_actions);
    device_free(context->d_positions);
    device_free(context->d_velocities);
    device_free(context->d_statuses);
    pinned_free(context->h_input);
    pinned_free(context->h_ready);
    pinned_free(context->h_slot_attempted);
    for(int slot = 0; slot < 2; ++slot)
    {
        pinned_free(context->h_positions[slot]);
        pinned_free(context->h_velocities[slot]);
        pinned_free(context->h_statuses[slot]);
    }
    delete context;
}

FrameGraphContext& graph_context(GIPC& ipc)
{
    auto* context =
        static_cast<FrameGraphContext*>(ipc.m_frame_graph_context);
    if(!context)
        throw std::logic_error("frame graph transaction is not prepared");
    return *context;
}

EpisodeGraphContext& episode_context(GIPC& ipc)
{
    auto* context =
        static_cast<EpisodeGraphContext*>(ipc.m_episode_graph_context);
    if(!context)
        throw std::logic_error("episode graph is not prepared");
    return *context;
}

const EpisodeGraphContext& episode_context(const GIPC& ipc)
{
    auto* context =
        static_cast<const EpisodeGraphContext*>(
            ipc.m_episode_graph_context);
    if(!context)
        throw std::logic_error("episode graph is not prepared");
    return *context;
}

void capture_root_graph(GIPC& ipc,
                        device_TetraData& mesh,
                        FrameGraphContext& context)
{
    cudaGraph_t graph = nullptr;
    CUDA_SAFE_CALL(cudaStreamBeginCapture(
        cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    CUDA_SAFE_CALL(cudaMemcpyAsync(context.d_begin,
                                   context.h_begin,
                                   sizeof(FrameBeginInput),
                                   cudaMemcpyHostToDevice,
                                   cudaStreamPerThread));
    frame_begin_init<<<1, 1, 0, cudaStreamPerThread>>>(
        context.d_state, context.d_begin);

    const size_t fem_bytes = context.vertex_count * sizeof(double3);
    if(fem_bytes)
    {
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.fem_vertexes,
                                       mesh.vertexes,
                                       fem_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.fem_o_vertexes,
                                       mesh.o_vertexes,
                                       fem_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.fem_velocities,
                                       mesh.velocities,
                                       fem_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.fem_x_tilta,
                                       mesh.xTilta,
                                       fem_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    }
    if(context.abd_count)
    {
        auto& device = ipc.m_abd_sim_data->device;
        const size_t bytes = context.abd_count * sizeof(gipc::Vector12);
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.abd_q,
                                       device.body_id_to_q.data(),
                                       bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.abd_q_prev,
                                       device.body_id_to_q_prev.data(),
                                       bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.abd_q_v,
                                       device.body_id_to_q_v.data(),
                                       bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.abd_q_tilde,
                                       device.body_id_to_q_tilde.data(),
                                       bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    }
    if(context.group_count)
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.kappa_snapshot,
                                       ipc.m_kappa_group,
                                       context.group_count * sizeof(double),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    CUDA_SAFE_CALL(
        cudaStreamEndCapture(cudaStreamPerThread, &graph));
    audit_graph(graph, context.root_nodes, context.root_d2h);
    CUDA_SAFE_CALL(
        cudaGraphInstantiate(&context.root_exec, graph, nullptr, nullptr, 0));
    CUDA_SAFE_CALL(cudaGraphUpload(
        context.root_exec, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
}

void capture_terminal_graph(GIPC& ipc,
                            device_TetraData& mesh,
                            FrameGraphContext& context)
{
    cudaGraph_t graph = nullptr;
    CUDA_SAFE_CALL(cudaStreamBeginCapture(
        cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    CUDA_SAFE_CALL(cudaMemcpyAsync(context.d_terminal,
                                   context.h_terminal,
                                   sizeof(FrameTerminalInput),
                                   cudaMemcpyHostToDevice,
                                   cudaStreamPerThread));
    frame_terminal_apply<<<1, 1, 0, cudaStreamPerThread>>>(
        context.d_state, context.d_terminal);
    if(context.vertex_count)
    {
        const int blocks =
            static_cast<int>((context.vertex_count + 255) / 256);
        frame_validate_finite<<<blocks, 256, 0, cudaStreamPerThread>>>(
            context.d_state,
            mesh.vertexes,
            mesh.velocities,
            static_cast<int>(context.vertex_count));
        frame_restore_fem<<<blocks, 256, 0, cudaStreamPerThread>>>(
            context.d_state,
            mesh.vertexes,
            mesh.o_vertexes,
            mesh.velocities,
            mesh.xTilta,
            context.fem_vertexes,
            context.fem_o_vertexes,
            context.fem_velocities,
            context.fem_x_tilta,
            static_cast<int>(context.vertex_count));
    }
    if(context.abd_count)
    {
        auto& device = ipc.m_abd_sim_data->device;
        const int blocks =
            static_cast<int>((context.abd_count + 255) / 256);
        frame_restore_abd<<<blocks, 256, 0, cudaStreamPerThread>>>(
            context.d_state,
            device.body_id_to_q.data(),
            device.body_id_to_q_prev.data(),
            device.body_id_to_q_v.data(),
            device.body_id_to_q_tilde.data(),
            context.abd_q,
            context.abd_q_prev,
            context.abd_q_v,
            context.abd_q_tilde,
            static_cast<int>(context.abd_count));
    }
    if(context.group_count)
        frame_restore_kappa<<<
            static_cast<int>((context.group_count + 255) / 256),
            256,
            0,
            cudaStreamPerThread>>>(
            context.d_state,
            ipc.m_kappa_group,
            context.kappa_snapshot,
            static_cast<int>(context.group_count));
    frame_serialize_status<<<1, 1, 0, cudaStreamPerThread>>>(
        context.d_state, context.d_terminal, context.d_status);
    CUDA_SAFE_CALL(cudaMemcpyAsync(context.h_status,
                                   context.d_status,
                                   sizeof(frame_fsm::FrameStatus),
                                   cudaMemcpyDeviceToHost,
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(
        cudaStreamEndCapture(cudaStreamPerThread, &graph));
    audit_graph(graph, context.terminal_nodes, context.terminal_d2h);
    CUDA_SAFE_CALL(cudaGraphInstantiate(
        &context.terminal_exec, graph, nullptr, nullptr, 0));
    CUDA_SAFE_CALL(cudaGraphUpload(
        context.terminal_exec, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
}

void capture_full_graph(GIPC& ipc,
                        device_TetraData& mesh,
                        FrameGraphContext& context)
{
    if(context.full_exec)
    {
        CUDA_SAFE_CALL(cudaGraphExecDestroy(context.full_exec));
        context.full_exec = nullptr;
    }

    cudaGraph_t graph = nullptr;
    bool        root_capture_ended_cleanly = false;
    CUDA_SAFE_CALL(cudaGraphCreate(&graph, 0));
    try
    {
        CUDA_SAFE_CALL(cudaStreamBeginCaptureToGraph(
            cudaStreamPerThread,
            graph,
            nullptr,
            nullptr,
            0,
            cudaStreamCaptureModeThreadLocal));
        CUDA_SAFE_CALL(cudaMemcpyAsync(context.d_begin,
                                       context.h_begin,
                                       sizeof(FrameBeginInput),
                                       cudaMemcpyHostToDevice,
                                       cudaStreamPerThread));
        frame_begin_init<<<1, 1, 0, cudaStreamPerThread>>>(
            context.d_state, context.d_begin);

        const size_t fem_bytes =
            context.vertex_count * sizeof(double3);
        if(fem_bytes)
        {
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                context.fem_vertexes,
                mesh.vertexes,
                fem_bytes,
                cudaMemcpyDeviceToDevice,
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                context.fem_o_vertexes,
                mesh.o_vertexes,
                fem_bytes,
                cudaMemcpyDeviceToDevice,
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                context.fem_velocities,
                mesh.velocities,
                fem_bytes,
                cudaMemcpyDeviceToDevice,
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                context.fem_x_tilta,
                mesh.xTilta,
                fem_bytes,
                cudaMemcpyDeviceToDevice,
                cudaStreamPerThread));
        }

        std::vector<cudaGraph_t> conditional_bodies;
        std::vector<cudaGraphNode_t> conditional_nodes;
        {
            frame_fsm::ConditionalGraphRecorder recorder(
                graph, cudaStreamPerThread);
            frame_fsm::ConditionalGraphRecorderScope scope(recorder);
            ipc.enqueue_frame_graph_body(mesh);
            conditional_bodies = recorder.conditional_bodies();
            conditional_nodes  = recorder.conditional_nodes();
        }

        ipc.updateVelocities(mesh);
        ipc.computeXTilta(mesh, 1);
        if(context.vertex_count)
        {
            const int blocks = static_cast<int>(
                (context.vertex_count + 255) / 256);
            frame_validate_finite<<<
                blocks, 256, 0, cudaStreamPerThread>>>(
                context.d_state,
                mesh.vertexes,
                mesh.velocities,
                static_cast<int>(context.vertex_count));
        }
        frame_terminal_finalize_full<<<
            1, 1, 0, cudaStreamPerThread>>>(context.d_state);
        if(context.vertex_count)
        {
            const int blocks = static_cast<int>(
                (context.vertex_count + 255) / 256);
            frame_restore_fem<<<
                blocks, 256, 0, cudaStreamPerThread>>>(
                context.d_state,
                mesh.vertexes,
                mesh.o_vertexes,
                mesh.velocities,
                mesh.xTilta,
                context.fem_vertexes,
                context.fem_o_vertexes,
                context.fem_velocities,
                context.fem_x_tilta,
                static_cast<int>(context.vertex_count));
        }

        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.d_terminal,
            context.h_terminal,
            sizeof(FrameTerminalInput),
            cudaMemcpyHostToDevice,
            cudaStreamPerThread));
        frame_serialize_status<<<
            1, 1, 0, cudaStreamPerThread>>>(
            context.d_state,
            context.d_terminal,
            context.d_status);
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.h_status,
            context.d_status,
            sizeof(frame_fsm::FrameStatus),
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));

        cudaGraph_t captured = nullptr;
        CUDA_SAFE_CALL(cudaStreamEndCapture(
            cudaStreamPerThread, &captured));
        root_capture_ended_cleanly = true;
        if(captured != graph)
            throw std::runtime_error(
                "full frame capture returned a different root graph");

        GraphAudit audit;
        accumulate_graph_audit(
            graph, audit, &conditional_nodes);
        for(cudaGraph_t body : conditional_bodies)
            accumulate_graph_audit(
                body, audit, &conditional_nodes);
        context.full_nodes = audit.node_count;
        context.full_d2h   = audit.d2h_count;
        if(audit.d2h_count != 1 || audit.host_count != 0)
        {
            std::ostringstream message;
            message << "full frame graph audit requires exactly one D2H "
                    << "and no host nodes; got d2h=" << audit.d2h_count
                    << " host=" << audit.host_count;
            throw std::runtime_error(message.str());
        }

        CUDA_SAFE_CALL(cudaGraphInstantiate(
            &context.full_exec, graph, nullptr, nullptr, 0));
        CUDA_SAFE_CALL(cudaGraphUpload(
            context.full_exec, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaGraphDestroy(graph));
        graph = nullptr;
        context.full_generation = pcg_buffer_generation();
    }
    catch(...)
    {
        cudaStreamCaptureStatus status =
            cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(
               cudaStreamPerThread, &status) == cudaSuccess
           && status != cudaStreamCaptureStatusNone)
        {
            cudaGraph_t abandoned = nullptr;
            const cudaError_t end_result = cudaStreamEndCapture(
                cudaStreamPerThread, &abandoned);
            if(end_result == cudaSuccess)
            {
                root_capture_ended_cleanly = abandoned == graph;
                if(abandoned && abandoned != graph)
                    cudaGraphDestroy(abandoned);
            }
        }
        cudaGetLastError();
        if(context.full_exec)
        {
            cudaGraphExecDestroy(context.full_exec);
            context.full_exec = nullptr;
        }
        // An invalidated stream capture can leave the explicit root graph in
        // an opaque driver-owned state.  Destroying that handle has crashed
        // inside libcuda on affected drivers, so only release it after a
        // successful cudaStreamEndCapture.  The failed construction is
        // permanently disabled for this context, limiting the leak to one
        // graph handle while preserving the legacy fallback.
        if(graph && root_capture_ended_cleanly)
            cudaGraphDestroy(graph);
        throw;
    }
}

void snapshot_host_attempt(GIPC& ipc, HostAttemptSnapshot& snapshot)
{
    snapshot.total_frames          = ipc.m_total_frames;
    snapshot.total_newton_iters    = ipc.m_total_newton_iters;
    snapshot.total_pcg_iters       = ipc.m_total_pcg_iters;
    snapshot.total_collision_pairs = ipc.m_total_collision_pairs;
    snapshot.max_collision_pairs   = ipc.m_max_collision_pairs;
    snapshot.total_time_ms         = ipc.m_total_time_ms;
    std::memcpy(snapshot.phase_time_ms,
                ipc.m_phase_time_ms,
                sizeof(snapshot.phase_time_ms));
    snapshot.time_make_pd_ms     = ipc.m_time_make_pd_ms;
    snapshot.animation_full_rate = ipc.animation_fullRate;
    snapshot.kappa               = ipc.Kappa;
    for(int i = 0; i < 5; ++i)
    {
        snapshot.cp_count[i] = ipc.h_cpNum[i];
        snapshot.cp_last[i]  = ipc.h_cpNum_last[i];
    }
    snapshot.gp_count           = ipc.h_gpNum;
    snapshot.gp_last            = ipc.h_gpNum_last;
    // DCD reuses _cpNum and intentionally invalidates h_ccd_cpNum after the
    // swept count has already been adjudicated. A frame-boundary transaction
    // must snapshot the dedicated stable cache, never bypass that audit.
    snapshot.ccd_count          = ipc.m_last_ccd_pair_count;
    snapshot.last_ccd_count     = ipc.m_last_ccd_pair_count;
    snapshot.dcd_snapshot_count = ipc.m_dcd_snap_count;
    snapshot.ls_exhausted_total = ipc.m_ls_exhausted_total;
    snapshot.ls_nonfinite_total = ipc.m_ls_nonfinite_total;
    snapshot.triplet_offset =
        ipc.gipc_global_triplet.global_triplet_offset;
    snapshot.collision_triplet_offset =
        ipc.gipc_global_triplet.global_collision_triplet_offset;
}

void restore_host_attempt(GIPC& ipc, const HostAttemptSnapshot& snapshot)
{
    ipc.m_total_frames          = snapshot.total_frames;
    ipc.m_total_newton_iters    = snapshot.total_newton_iters;
    ipc.m_total_pcg_iters       = snapshot.total_pcg_iters;
    ipc.m_total_collision_pairs = snapshot.total_collision_pairs;
    ipc.m_max_collision_pairs   = snapshot.max_collision_pairs;
    ipc.m_total_time_ms         = snapshot.total_time_ms;
    std::memcpy(ipc.m_phase_time_ms,
                snapshot.phase_time_ms,
                sizeof(snapshot.phase_time_ms));
    ipc.m_time_make_pd_ms     = snapshot.time_make_pd_ms;
    ipc.animation_fullRate    = snapshot.animation_full_rate;
    ipc.Kappa                 = snapshot.kappa;
    std::memcpy(ipc.h_cpNum.refresh_dst(),
                snapshot.cp_count,
                sizeof(snapshot.cp_count));
    std::memcpy(ipc.h_cpNum_last.refresh_dst(),
                snapshot.cp_last,
                sizeof(snapshot.cp_last));
    ipc.h_gpNum              = snapshot.gp_count;
    ipc.h_gpNum_last         = snapshot.gp_last;
    ipc.h_ccd_cpNum          = snapshot.ccd_count;
    ipc.m_last_ccd_pair_count = snapshot.last_ccd_count;
    ipc.m_dcd_snap_count     = snapshot.dcd_snapshot_count;
    ipc.m_ls_exhausted_total = snapshot.ls_exhausted_total;
    ipc.m_ls_nonfinite_total = snapshot.ls_nonfinite_total;
    ipc.gipc_global_triplet.global_triplet_offset =
        snapshot.triplet_offset;
    ipc.gipc_global_triplet.global_collision_triplet_offset =
        snapshot.collision_triplet_offset;
}

void arm_full_graph_attempt(GIPC& ipc, FrameGraphContext& context)
{
    if(ipc.m_global_linear_system)
        ipc.m_global_linear_system->set_frame_device_state(
            context.d_state);
    ipc.gipc_global_triplet.m_frame_device_state = context.d_state;
    ipc.gipc_global_triplet.m_contact_partition_txn_ok = true;
    ipc.gipc_global_triplet.m_abd_tier_txn_ok =
        context.abd_count != 0;
    ipc.gipc_global_triplet.m_abd_unique_test_tier = 0;
    ipc.m_frame_graph_active      = true;
    ipc.m_frame_terminal_emitted = false;
}

void disarm_full_graph_attempt(GIPC& ipc)
{
    if(ipc.m_global_linear_system)
        ipc.m_global_linear_system->set_frame_device_state(nullptr);
    ipc.gipc_global_triplet.m_frame_device_state = nullptr;
    ipc.gipc_global_triplet.m_abd_unique_test_tier = 0;
    ipc.gipc_global_triplet.m_abd_tier_txn_ok = false;
    ipc.gipc_global_triplet.m_contact_partition_txn_ok = false;
    ipc.m_frame_graph_active      = false;
    ipc.m_frame_terminal_emitted = false;
}

void enqueue_episode_slot_copy(EpisodeGraphContext& context, int slot)
{
    const int first = slot == 0 ? 0 : context.split_frame;
    const int count =
        slot == 0 ? context.split_frame
                  : context.frame_count - context.split_frame;
    const size_t vertex_offset =
        checked_product(static_cast<size_t>(first),
                        context.vertex_count,
                        "slot vertex offset");
    const size_t vertex_elements =
        checked_product(static_cast<size_t>(count),
                        context.vertex_count,
                        "slot vertex count");
    if(vertex_elements)
    {
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.h_positions[slot],
            context.d_positions + vertex_offset,
            vertex_elements * sizeof(double3),
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.h_velocities[slot],
            context.d_velocities + vertex_offset,
            vertex_elements * sizeof(double3),
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));
    }
    if(count)
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.h_statuses[slot],
            context.d_statuses + first,
            static_cast<size_t>(count)
                * sizeof(frame_fsm::FrameStatus),
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));

    const auto* attempted = reinterpret_cast<const int*>(
        reinterpret_cast<const unsigned char*>(context.d_control)
        + offsetof(EpisodeDeviceControl, attempted_frames));
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        context.h_slot_attempted + slot,
        attempted,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread));
    // The event is ordered after all payload DMA.  Publish the host-ready
    // sentinel only after the event record node has executed, preventing a
    // completed event from a prior episode from being mistaken for this one.
    // A normal event record made during stream capture is an internal graph
    // dependency and is not a host-queryable completion fence. Request an
    // external event node explicitly so poll/wait APIs observe this execution.
    CUDA_SAFE_CALL(cudaEventRecordWithFlags(
        context.slot_event[slot],
        cudaStreamPerThread,
        cudaEventRecordExternal));
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        context.h_ready + slot,
        context.d_ready_one,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        cudaStreamPerThread));
}

void capture_episode_graph(GIPC& ipc,
                           device_TetraData& mesh,
                           FrameGraphContext& frame,
                           EpisodeGraphContext& episode)
{
    cudaGraph_t graph = nullptr;
    bool capture_ended_cleanly = false;
    CUDA_SAFE_CALL(cudaGraphCreate(&graph, 0));
    try
    {
        CUDA_SAFE_CALL(cudaStreamBeginCaptureToGraph(
            cudaStreamPerThread,
            graph,
            nullptr,
            nullptr,
            0,
            cudaStreamCaptureModeThreadLocal));
        if(!episode.device_native)
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                episode.d_input,
                episode.h_input,
                sizeof(EpisodeRuntimeInput),
                cudaMemcpyHostToDevice,
                cudaStreamPerThread));
        const size_t observation_elements =
            checked_product(
                static_cast<size_t>(episode.frame_count),
                episode.vertex_count,
                "episode observations");
        if(observation_elements)
        {
            CUDA_SAFE_CALL(cudaMemsetAsync(
                episode.d_positions,
                0,
                observation_elements * sizeof(double3),
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(
                episode.d_velocities,
                0,
                observation_elements * sizeof(double3),
                cudaStreamPerThread));
        }
        CUDA_SAFE_CALL(cudaMemsetAsync(
            episode.d_statuses,
            0,
            static_cast<size_t>(episode.frame_count)
                * sizeof(frame_fsm::FrameStatus),
            cudaStreamPerThread));
        episode_initialize<<<1, 1, 0, cudaStreamPerThread>>>(
            episode.d_control);

        std::vector<cudaGraph_t> conditional_bodies;
        std::vector<cudaGraphNode_t> conditional_nodes;
        {
            frame_fsm::ConditionalGraphRecorder recorder(
                graph, cudaStreamPerThread);
            frame_fsm::ConditionalGraphRecorderScope scope(recorder);
            auto enqueue_iteration =
                [&](cudaGraphConditionalHandle episode_handle,
                    int iteration_limit)
                {
                    episode_frame_begin<<<
                        1, 1, 0, cudaStreamPerThread>>>(
                        frame.d_state,
                        episode.d_control,
                        episode.d_input,
                        episode.d_frame_counter);

                    const size_t vertex_bytes =
                        frame.vertex_count * sizeof(double3);
                    if(vertex_bytes)
                    {
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.fem_vertexes,
                            mesh.vertexes,
                            vertex_bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.fem_o_vertexes,
                            mesh.o_vertexes,
                            vertex_bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.fem_velocities,
                            mesh.velocities,
                            vertex_bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.fem_x_tilta,
                            mesh.xTilta,
                            vertex_bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                    }
                    if(frame.abd_count)
                    {
                        auto& device = ipc.m_abd_sim_data->device;
                        const size_t bytes =
                            frame.abd_count * sizeof(gipc::Vector12);
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.abd_q,
                            device.body_id_to_q.data(),
                            bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.abd_q_prev,
                            device.body_id_to_q_prev.data(),
                            bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.abd_q_v,
                            device.body_id_to_q_v.data(),
                            bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.abd_q_tilde,
                            device.body_id_to_q_tilde.data(),
                            bytes,
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));
                    }
                    if(frame.group_count)
                        CUDA_SAFE_CALL(cudaMemcpyAsync(
                            frame.kappa_snapshot,
                            ipc.m_kappa_group,
                            frame.group_count * sizeof(double),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamPerThread));

                    if(episode.revolute_count
                       || episode.prismatic_count)
                    {
                        ipc.m_abd_system
                            ->enqueue_episode_driving_targets(
                                *ipc.m_abd_sim_data,
                                episode.d_revolute_actions,
                                episode.d_prismatic_actions,
                                &episode.d_control->frame_index);
                    }

                    ipc.enqueue_frame_graph_body(mesh);
                    ipc.updateVelocities(mesh);
                    ipc.computeXTilta(mesh, 1);
                    if(frame.vertex_count)
                    {
                        const int blocks = static_cast<int>(
                            (frame.vertex_count + 255) / 256);
                        frame_validate_finite<<<
                            blocks, 256, 0, cudaStreamPerThread>>>(
                            frame.d_state,
                            mesh.vertexes,
                            mesh.velocities,
                            static_cast<int>(frame.vertex_count));
                    }
                    frame_terminal_finalize_full<<<
                        1, 1, 0, cudaStreamPerThread>>>(
                        frame.d_state);
                    if(frame.vertex_count)
                    {
                        const int blocks = static_cast<int>(
                            (frame.vertex_count + 255) / 256);
                        frame_restore_fem<<<
                            blocks, 256, 0, cudaStreamPerThread>>>(
                            frame.d_state,
                            mesh.vertexes,
                            mesh.o_vertexes,
                            mesh.velocities,
                            mesh.xTilta,
                            frame.fem_vertexes,
                            frame.fem_o_vertexes,
                            frame.fem_velocities,
                            frame.fem_x_tilta,
                            static_cast<int>(frame.vertex_count));
                    }
                    if(frame.abd_count)
                    {
                        auto& device = ipc.m_abd_sim_data->device;
                        const int blocks = static_cast<int>(
                            (frame.abd_count + 255) / 256);
                        frame_restore_abd<<<
                            blocks, 256, 0, cudaStreamPerThread>>>(
                            frame.d_state,
                            device.body_id_to_q.data(),
                            device.body_id_to_q_prev.data(),
                            device.body_id_to_q_v.data(),
                            device.body_id_to_q_tilde.data(),
                            frame.abd_q,
                            frame.abd_q_prev,
                            frame.abd_q_v,
                            frame.abd_q_tilde,
                            static_cast<int>(frame.abd_count));
                    }
                    if(frame.group_count)
                        frame_restore_kappa<<<
                            static_cast<int>(
                                (frame.group_count + 255) / 256),
                            256,
                            0,
                            cudaStreamPerThread>>>(
                            frame.d_state,
                            ipc.m_kappa_group,
                            frame.kappa_snapshot,
                            static_cast<int>(frame.group_count));

                    const int observation_blocks = std::max(
                        1,
                        static_cast<int>(
                            (frame.vertex_count + 255) / 256));
                    episode_store_frame<<<
                        observation_blocks,
                        256,
                        0,
                        cudaStreamPerThread>>>(
                        frame.d_state,
                        episode.d_control,
                        episode.d_input,
                        mesh.vertexes,
                        mesh.velocities,
                        episode.d_positions,
                        episode.d_velocities,
                        episode.d_statuses,
                        static_cast<int>(frame.vertex_count));

                    episode_tail<<<
                        1, 1, 0, cudaStreamPerThread>>>(
                        frame.d_state,
                        episode.d_control,
                        episode.d_input,
                        iteration_limit,
                        episode_handle);
                };

            recorder.while_loop(
                1,
                cudaGraphCondAssignDefault,
                [&](cudaGraphConditionalHandle first_half)
                {
                    enqueue_iteration(
                        first_half, episode.split_frame);
                });
            if(!episode.device_native)
                enqueue_episode_slot_copy(episode, 0);

            // Recording the solver body advances capture-time host mirrors
            // (PCG counters, kappa, assembly offsets, and animation rate).
            // The second WHILE must be built from the same frame-boundary
            // state as the first one; otherwise the first frame after the
            // observation split captures different launch parameters.
            restore_host_attempt(ipc, episode.host_snapshot);
            ipc.animation_fullRate = ipc.animation_subRate;

            recorder.if_then(
                [&](cudaGraphConditionalHandle remaining)
                {
                    episode_remaining_predicate<<<
                        1, 1, 0, cudaStreamPerThread>>>(
                        episode.d_control,
                        episode.d_input,
                        remaining);
                },
                [&](cudaGraphConditionalHandle)
                {
                    recorder.while_loop(
                        1,
                        cudaGraphCondAssignDefault,
                        [&](cudaGraphConditionalHandle second_half)
                        {
                            enqueue_iteration(
                                second_half,
                                episode.frame_count);
                        });
                });
            if(!episode.device_native)
                enqueue_episode_slot_copy(episode, 1);
            else
            {
                episode_advance_frame_counter<<<
                    1, 1, 0, cudaStreamPerThread>>>(
                    episode.d_control,
                    episode.d_frame_counter);
                // [D2] refresh the packed joint observations from the
                // committed state — pure device work inside the graph.
                if(episode.d_joint_obs)
                    ipc.m_abd_system->enqueue_joint_observations(
                        *ipc.m_abd_sim_data,
                        episode.d_joint_obs,
                        ipc.IPC_dt);
            }
            conditional_bodies = recorder.conditional_bodies();
            conditional_nodes  = recorder.conditional_nodes();
        }

        cudaGraph_t captured = nullptr;
        CUDA_SAFE_CALL(cudaStreamEndCapture(
            cudaStreamPerThread, &captured));
        capture_ended_cleanly = true;
        if(captured != graph)
            throw std::runtime_error(
                "episode capture returned a different root graph");

        GraphAudit audit;
        accumulate_graph_audit(
            graph, audit, &conditional_nodes);
        for(cudaGraph_t body : conditional_bodies)
            accumulate_graph_audit(
                body, audit, &conditional_nodes);
        episode.graph_nodes = audit.node_count;
        episode.graph_h2d   = audit.h2d_count;
        episode.graph_d2h   = audit.d2h_count;
        if(episode.device_native
           && (audit.host_count != 0
               || audit.h2d_count != 0
               || audit.d2h_count != 0))
        {
            std::ostringstream message;
            message
                << "GPU-native RL graph audit requires zero host nodes "
                   "and zero host/device transfers; got h2d="
                << audit.h2d_count
                << " d2h=" << audit.d2h_count
                << " host=" << audit.host_count;
            throw std::runtime_error(message.str());
        }
        if(!episode.device_native
           && (audit.host_count != 0 || audit.d2h_count < 4))
        {
            std::ostringstream message;
            message
                << "episode graph audit requires no host nodes and "
                   "two asynchronous observation slots; got d2h="
                << audit.d2h_count
                << " host=" << audit.host_count;
            throw std::runtime_error(message.str());
        }

        CUDA_SAFE_CALL(cudaGraphInstantiate(
            &episode.exec, graph, nullptr, nullptr, 0));
        CUDA_SAFE_CALL(cudaGraphUpload(
            episode.exec, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaGraphDestroy(graph));
        graph = nullptr;
        episode.generation = pcg_buffer_generation();
    }
    catch(...)
    {
        cudaStreamCaptureStatus status =
            cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(
               cudaStreamPerThread, &status) == cudaSuccess
           && status != cudaStreamCaptureStatusNone)
        {
            cudaGraph_t abandoned = nullptr;
            const cudaError_t end_result =
                cudaStreamEndCapture(
                    cudaStreamPerThread, &abandoned);
            if(end_result == cudaSuccess)
            {
                capture_ended_cleanly = abandoned == graph;
                if(abandoned && abandoned != graph)
                    cudaGraphDestroy(abandoned);
            }
        }
        cudaGetLastError();
        if(episode.exec)
        {
            cudaGraphExecDestroy(episode.exec);
            episode.exec = nullptr;
        }
        if(graph && capture_ended_cleanly)
            cudaGraphDestroy(graph);
        throw;
    }
}

// [C4-a] all in-graph collision buffers must reach final capacity before
// capture. Explicit tier math first, then a dry-run of the assembly+solve
// host builders with capacity mirrors: every lazy workspace (converter
// staging, radix-sort temp, preconditioner levels, ABD assembly tiers)
// grows on its own real path here, outside capture, so the recorded pass
// allocates nothing. Must run after arm_full_graph_attempt — the dry run
// has to exercise the same transactional partition branch the recording
// will take. Its garbage triplets/gradients are overwritten by the graph's
// own prologue at runtime, and host bookkeeping is rolled back by the same
// snapshot machinery the recording itself relies on. Shared by the step
// whole-frame path and the episode/GPU-native RL capture path.
void train_collision_for_capture(GIPC& ipc, device_TetraData& mesh)
{
    if(ipc.m_skip_all_collision)
        return;
    const bool isolated_body =
        ipc.m_mode_config.mode == ModeConfig::Isolated;
    ipc.train_collision_graph_capacities();
    if(isolated_body)
        ipc.train_perenv_graph_capacities(mesh);
    HostAttemptSnapshot training_snapshot;
    snapshot_host_attempt(ipc, training_snapshot);
    const bool defer_before  = ipc.m_ls_defer_counts;
    const bool energy_before = ipc.m_energy_use_device_counts;
    const bool merged_detect_before = ipc.m_graph_merged_detect;
    // [C5] the dry run must take the SAME detection path the recording will
    // take, otherwise the per-env tree's extents (not the merged tree's)
    // would be the ones trained.
    ipc.m_graph_merged_detect = isolated_body;
    // The PCG epilogue records into stats["newton"].back(); give the dry
    // run the per-iteration context solve_subIP would have set up, and
    // restore the frame's stats object afterwards.
    auto& frame_stats = gipc::Statistics::instance().at_current_frame();
    const gipc::Json stats_backup = frame_stats;
    try
    {
        frame_stats["newton"] = gipc::Json::array();
        frame_stats["newton"].push_back(gipc::Json::object());
        ipc.m_ls_defer_counts           = true;
        ipc.m_energy_use_device_counts = true;
        // [C6-b] Contamination probe. The dry run is HOST-executed, so the
        // capacity-mirror assembly can be compared against the exact-count
        // assembly at the identical state without involving the graph at
        // all. If the search directions differ here, the bug is in the
        // capacity-shaped assembly, not in graph capture.
        std::vector<double> exact_dir;
        const bool probe_dir = std::getenv("STIFF_GRAPH_DIRPROBE") != nullptr;
        if(probe_dir)
        {
            const bool defer_save  = ipc.m_ls_defer_counts;
            const bool energy_save = ipc.m_energy_use_device_counts;
            ipc.m_ls_defer_counts           = false;
            ipc.m_energy_use_device_counts = false;
            ipc.refresh_pair_counts();          // exact mirrors
            ipc.computeGradientAndHessian(mesh);
            ipc.calculateMovingDirection(mesh, 0, ipc.pcg_data.P_type);
            exact_dir.resize(3 * ipc.vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(exact_dir.data(),
                                      ipc._moveDir,
                                      exact_dir.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            ipc.m_ls_defer_counts           = defer_save;
            ipc.m_energy_use_device_counts = energy_save;
        }
        // [C6-d] Did the rollback actually restore? Compare the live state with
        // the snapshot the terminal graph restores from. Zero means restored.
        FrameGraphContext& restore_ctx = graph_context(ipc);
        if(std::getenv("STIFF_FRAME_GRAPH_DIAG") && restore_ctx.vertex_count)
        {
            std::vector<double3> live(restore_ctx.vertex_count);
            std::vector<double3> snap(restore_ctx.vertex_count);
            const size_t bytes = restore_ctx.vertex_count * sizeof(double3);
            CUDA_SAFE_CALL(cudaMemcpy(live.data(), mesh.vertexes, bytes,
                                      cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(snap.data(), restore_ctx.fem_vertexes,
                                      bytes, cudaMemcpyDeviceToHost));
            double worst = 0.0;
            int    worst_i = -1;
            for(size_t i = 0; i < live.size(); ++i)
            {
                const double d =
                    std::max(std::max(std::fabs(live[i].x - snap[i].x),
                                      std::fabs(live[i].y - snap[i].y)),
                             std::fabs(live[i].z - snap[i].z));
                if(d > worst) { worst = d; worst_i = (int)i; }
            }
            fprintf(stderr,
                    "[restore] max|live-snap|=%.6e at vertex %d of %d\n",
                    worst, worst_i, (int)restore_ctx.vertex_count);
        }
        // [C6] trained extent, not worst-case capacity.
        for(int slot = 0; slot < 5; ++slot)
            ipc.h_cpNum.refresh_dst()[slot] =
                static_cast<uint32_t>(ipc.m_graph_train_cp[slot]);
        ipc.h_gpNum =
            static_cast<uint32_t>(ipc.graph_trained_ground_extent());
#ifdef USE_FRICTION
        for(int slot = 0; slot < 5; ++slot)
            ipc.h_cpNum_last.refresh_dst()[slot] =
                static_cast<uint32_t>(ipc.m_graph_train_cp[slot]);
        ipc.h_gpNum_last =
            static_cast<uint32_t>(ipc.graph_trained_ground_extent());
#endif
        ipc.computeGradientAndHessian(mesh);
        ipc.calculateMovingDirection(mesh, 0, ipc.pcg_data.P_type);
        if(probe_dir && !exact_dir.empty())
        {
            std::vector<double> mirror_dir(exact_dir.size());
            CUDA_SAFE_CALL(cudaMemcpy(mirror_dir.data(),
                                      ipc._moveDir,
                                      mirror_dir.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            double n_exact = 0.0, n_mirror = 0.0, n_diff = 0.0;
            double worst = 0.0;
            int worst_i = -1;
            int nonfinite_mirror = 0;
            for(size_t i = 0; i < exact_dir.size(); ++i)
            {
                const double a = exact_dir[i], b = mirror_dir[i];
                if(!std::isfinite(b))
                    ++nonfinite_mirror;
                n_exact += a * a;
                n_mirror += b * b;
                const double d = std::fabs(a - b);
                n_diff += d * d;
                if(d > worst)
                {
                    worst = d;
                    worst_i = static_cast<int>(i);
                }
            }
            fprintf(stderr,
                    "[dirprobe] |exact|=%.6e |mirror|=%.6e |diff|=%.6e "
                    "ratio=%.3f worst=%.3e at dof %d (vertex %d) "
                    "nonfinite_mirror=%d\n",
                    std::sqrt(n_exact), std::sqrt(n_mirror),
                    std::sqrt(n_diff),
                    n_exact > 0.0 ? std::sqrt(n_mirror / n_exact) : 0.0,
                    worst, worst_i, worst_i / 3, nonfinite_mirror);
        }
        // [C6-e] Tier-invariance probe. A tier is only a launch extent over
        // zero-padded slots, so assembling the SAME state at two different
        // tiers must give the same search direction. Frame 11 of foldshirt says
        // otherwise: growing cp[4] from 1024 to 8192 turned a clean 50-iteration
        // attempt into a 3.8M-pair swept explosion on the very next attempt.
        // Each (tier, pre-zero) combination is assembled from identical state;
        // if the two tiers disagree only when the buffer is NOT pre-zeroed, the
        // pad span of the grown class is what is leaking.
        if(std::getenv("STIFF_GRAPH_TIERPROBE"))
        {
            const int  saved4 = ipc.m_graph_train_cp[4];
            const int  dofs   = 3 * static_cast<int>(ipc.vertexNum);
            std::vector<double> pass[4];
            long long offsets[4] = {0, 0, 0, 0};
            for(int variant = 0; variant < 4; ++variant)
            {
                const bool big       = (variant % 2) != 0;
                const bool pre_zero  = variant >= 2;
                ipc.m_graph_train_cp[4] = big ? saved4 * 8 : saved4;
                for(int slot = 0; slot < 5; ++slot)
                    ipc.h_cpNum.refresh_dst()[slot] =
                        static_cast<uint32_t>(ipc.m_graph_train_cp[slot]);
                ipc.h_gpNum = static_cast<uint32_t>(
                    ipc.graph_trained_ground_extent());
#ifdef USE_FRICTION
                for(int slot = 0; slot < 5; ++slot)
                    ipc.h_cpNum_last.refresh_dst()[slot] =
                        static_cast<uint32_t>(ipc.m_graph_train_cp[slot]);
                ipc.h_gpNum_last = static_cast<uint32_t>(
                    ipc.graph_trained_ground_extent());
#endif
                if(pre_zero)
                {
                    auto& tri = ipc.gipc_global_triplet;
                    const size_t cap = tri.triplet_capacity();
                    CUDA_SAFE_CALL(cudaMemset(tri.block_row_indices(0), 0,
                                              cap * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMemset(tri.block_col_indices(0), 0,
                                              cap * sizeof(int)));
                    CUDA_SAFE_CALL(cudaMemset(tri.block_values(0), 0,
                                              cap * sizeof(Eigen::Matrix3d)));
                }
                ipc.computeGradientAndHessian(mesh);
                ipc.calculateMovingDirection(mesh, 0, ipc.pcg_data.P_type);
                offsets[variant] =
                    ipc.gipc_global_triplet.global_triplet_offset;
                pass[variant].resize(dofs);
                CUDA_SAFE_CALL(cudaMemcpy(pass[variant].data(),
                                          ipc._moveDir,
                                          static_cast<size_t>(dofs)
                                              * sizeof(double),
                                          cudaMemcpyDeviceToHost));
            }
            ipc.m_graph_train_cp[4] = saved4;
            auto report = [&](const char* label, int a, int b) {
                double na = 0.0, nb = 0.0, nd = 0.0;
                for(int i = 0; i < dofs; ++i)
                {
                    na += pass[a][i] * pass[a][i];
                    nb += pass[b][i] * pass[b][i];
                    const double d = pass[a][i] - pass[b][i];
                    nd += d * d;
                }
                fprintf(stderr,
                        "[tierprobe] %s cp4=%d vs %d: |small|=%.6e "
                        "|big|=%.6e |diff|=%.6e rel=%.3e "
                        "offsets=%lld vs %lld\n",
                        label, saved4, saved4 * 8,
                        std::sqrt(na), std::sqrt(nb), std::sqrt(nd),
                        na > 0.0 ? std::sqrt(nd / na) : 0.0,
                        offsets[a], offsets[b]);
            };
            report("raw      ", 0, 1);
            report("pre-zeroed", 2, 3);
        }
        // The recording replays the same mirrors and the same path, so the
        // dry run's extents ARE the capture-time extents. The a-priori
        // bound formula cannot see ground/ABD contributions, so re-derive
        // the class-tier envelope from the MEASURED collision payload; if
        // any tier grew, run the dry pass once more so staging layouts and
        // every workspace re-train at the final shape (extents depend only
        // on the capacity mirrors, so one correction converges).
        const int measured_payload =
            ipc.gipc_global_triplet.global_collision_triplet_offset;
        const int payload_tier =
            gipc::assembly_capacity_tier(measured_payload);
        const bool has_abd = ipc.abd_fem_count_info.abd_body_num > 0;
        const bool has_fem = ipc.abd_fem_count_info.fem_point_num > 0;
        // Class-tier order: fem_fem, abd_fem, fem_abd, abd_abd; class 0 is
        // always possible — capacity pads land there (see the train-side
        // comment in train_collision_graph_capacities).
        const bool class_possible[4] = {
            true, has_abd && has_fem, has_abd && has_fem, has_abd};
        (void)has_fem;
        // [C6] The class tiers are trained from observed per-class counts in
        // train_collision_graph_capacities. Re-deriving them here from the
        // capacity-inflated payload is what produced the compounding blowup
        // (246k -> 2.6M -> 21M triplets), so this pass no longer re-tiers;
        // genuine past-tier overflow is adjudicated in-graph and grows the
        // tier through the OVF retry path.
        // [D4-b] One narrow re-tier is back, for the case the bottom-up model
        // provably cannot see: in a PURE-ABD scene every padded slot of the
        // collision payload classifies into abd_abd (index 0 is an ABD body,
        // so the hash(0,0) pads follow), and the pad-side training models the
        // payload as contact_tier (4864 on D4) while the dry run MEASURES the
        // real padded payload (10240). The dry run reproduces the record
        // layout, so its measured payload is authoritative. Class tiers do
        // not feed the pair emission, so this converges in one pass -- the
        // compounding loop C6 killed ran through *_contact_num being read
        // back as observations, which the honest census already cut. FEM
        // scenes keep the no-retier behaviour bit-for-bit.
        bool retier = false;
        if(!has_fem && has_abd
           && ipc.gipc_global_triplet.m_contact_class_tier[3]
                  < measured_payload)
        {
            ipc.gipc_global_triplet.m_contact_class_tier[3] =
                gipc::assembly_capacity_tier(measured_payload);
            retier = true;
        }
        (void)class_possible;
        (void)payload_tier;
        // [C6] The ABD contraction reads [offset + fem_fem, 2*offset), so the
        // triplet buffer must hold TWICE the assembled length. The a-priori
        // bound cannot predict that length (it does not model ABD body
        // Hessians, joints or stitch springs), so guarantee it from the
        // MEASURED length here, at the legal boundary.
        const size_t contraction_need =
            2 * static_cast<size_t>(
                    ipc.gipc_global_triplet.global_triplet_offset)
            + 4096;
        bool grew_for_contraction = false;
        if(ipc.gipc_global_triplet.triplet_capacity() < contraction_need)
        {
            ipc.gipc_global_triplet.global_triplet_offset = 0;
            ipc.gipc_global_triplet.global_collision_triplet_offset = 0;
            ipc.gipc_global_triplet.open_discard_window();
            ipc.gipc_global_triplet.ensure_capacity_discard(
                contraction_need + contraction_need / 4);
            grew_for_contraction = true;
        }
        // [D4 fix] The ABD system's converter3x3 owns its OWN merge bins, and
        // they grow with cudaFree/cudaMalloc -- illegal inside capture. This
        // reserve used to live only in the retier branch below, so a scene
        // whose dry-run needed no retier (D4: retier=0, no contraction growth)
        // reached capture with floor-sized bins (256*36) and died with error
        // 900 when the 4190-wide ABD final convert arrived. Reserve
        // unconditionally; ensure_capacity is a no-op when already large.
        // 2x: exact baked counts drift a few percent between the dry-run and
        // the recording pass (measured 4206 -> 4352), so a same-tier reserve
        // can be a few slots short (need 4190*36 vs cap 4096*36).
        if(ipc.m_abd_system)
        {
            if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
                fprintf(stderr, "[reserve] converter3x3=%p tier=%d\n",
                        (void*)&ipc.m_abd_system->converter3x3,
                        2 * payload_tier);
            ipc.m_abd_system->converter3x3.ensure_capacity(2 * payload_tier);
        }
        if(retier || grew_for_contraction)
        {
            ++pcg_buffer_generation();
            ipc.computeGradientAndHessian(mesh);
            ipc.calculateMovingDirection(mesh, 0, ipc.pcg_data.P_type);
        }
        // [D4 fix] Reserve the converter merge bins at the widest extent any
        // in-capture convert can present: the padded payload tier and both ABD
        // unique tiers. Growth inside capture is a hard 900.
        if(ipc.m_global_linear_system)
            ipc.m_global_linear_system->train_converter_capacity(2 * std::max(
                {payload_tier,
                 ipc.gipc_global_triplet.m_abd_unique_tier[0],
                 ipc.gipc_global_triplet.m_abd_unique_tier[1]}));
        if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
            fprintf(stderr,
                    "[train-capture] dry-run offset=%d payload=%d "
                    "payload_tier=%d retier=%d\n",
                    ipc.gipc_global_triplet.global_triplet_offset,
                    measured_payload,
                    payload_tier,
                    static_cast<int>(retier));
        if(ipc.m_global_linear_system)
            ipc.m_global_linear_system->train_converter_capacity(
                gipc::assembly_capacity_tier(
                    ipc.gipc_global_triplet.global_triplet_offset));
    }
    catch(...)
    {
        frame_stats                     = stats_backup;
        ipc.m_ls_defer_counts           = defer_before;
        ipc.m_energy_use_device_counts = energy_before;
        ipc.m_graph_merged_detect       = merged_detect_before;
        restore_host_attempt(ipc, training_snapshot);
        disarm_full_graph_attempt(ipc);
        throw;
    }
    frame_stats                     = stats_backup;
    ipc.m_ls_defer_counts           = defer_before;
    ipc.m_energy_use_device_counts = energy_before;
    ipc.m_graph_merged_detect       = merged_detect_before;
    restore_host_attempt(ipc, training_snapshot);
    CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
}

// [C6-n] Whole-frame graphs price in fixed capacity-width launches plus
// re-record storms (a capture costs 20-30s on weak-host pods); a sub-1k
// vertex scene pays 10-80x the release frame for that structure (A800
// towel_scramble: 2126ms graph-on vs 25.8ms off). And the two-graph gate
// is no refuge: its inner graphs re-record on every capacity-generation
// bump, which turns a hard crumple frame into 15.7s of captures for the
// SAME 567 Newton iterations the release path solves in 0.5s. Ordinary
// step() therefore lands tiny scenes on the pure release path (layout
// machinery off, no captures). Episodes are exempt on purpose: their
// contract is zero host blocking, not per-frame wall time.
// STIFF_FULL_GRAPH_MIN_VERTS overrides the default of 1024; the gates pin
// it to 0 so 16-vertex fixtures keep exercising the graph machinery.
static bool tiny_scene_for_full_graph(uint32_t vertex_count)
{
    long min_verts = 1024;
    if(const char* value = std::getenv("STIFF_FULL_GRAPH_MIN_VERTS"))
        min_verts = std::atol(value);
    return min_verts > 0 && (long)vertex_count < min_verts;
}

// [C6-m/C6-n] Scoped device_count_mode() kill switch: while armed, every
// layout-mode consumer (assembly extents, partition, friction sets,
// converter) takes the pure release path. RAII so throws restore the mode.
struct LayoutOverrideOff
{
    bool armed;
    explicit LayoutOverrideOff(bool on) : armed(on)
    {
        if(armed)
            GIPCTripletMatrix::s_layout_override_off = true;
    }
    ~LayoutOverrideOff()
    {
        if(armed)
            GIPCTripletMatrix::s_layout_override_off = false;
    }
};

bool try_launch_full_graph(GIPC& ipc,
                           device_TetraData& mesh,
                           int64_t frame_id,
                           int attempt,
                           uint32_t retry_invalid_bits)
{
    const auto t_enter = std::chrono::steady_clock::now();
    ipc.prepare_frame_graph(mesh);
    FrameGraphContext& context = graph_context(ipc);
    std::string reason;
    // [C6-n->C6-q] Tiny scenes decline the FULL graph only: capture storms
    // and capacity-width replay price a sub-1k scene out (A800 towel: 82x),
    // but the two-graph transaction must remain -- it is where tier training
    // and the ABD unique-tier warm-up live, which the episode contract
    // ("run one step(), then prepare") depends on. The original C6-n
    // full-bypass broke exactly that (G17a: prepare threw 'tier not trained'
    // on sub-1k scenes); the 15.7s two-graph runaway that motivated it was
    // C6-o's starved spin, cured at the root since (towel two-graph now
    // lands within 4-8% of the release wall). Gates that audit the FULL
    // graph on tiny fixtures pin STIFF_FULL_GRAPH_MIN_VERTS=0.
    if(tiny_scene_for_full_graph(ipc.vertexNum))
    {
        static bool announced = false;
        if(!announced && knob_enabled("STIFF_FRAME_GRAPH_DIAG"))
        {
            announced = true;
            std::fprintf(stderr,
                         "[frame-full-graph] declined: scene has %u vertices "
                         "(< STIFF_FULL_GRAPH_MIN_VERTS, default 1024); "
                         "two-graph transaction stays active\n",
                         ipc.vertexNum);
        }
        return false;
    }
    if(context.full_capture_failed
       || !full_graph_eligible(ipc, mesh, context, reason))
    {
        if(knob_enabled("STIFF_FRAME_GRAPH_DIAG"))
            std::fprintf(stderr,
                         "[frame-full-graph] not eligible: %s\n",
                         context.full_capture_failed
                             ? "a previous capture failed permanently"
                             : reason.c_str());
        return false;
    }

    // [C4-b] frame-boundary kappa re-initialization, exactly as the release
    // IPC_Solver opens each frame (gradient-projection initKappa consuming
    // the previous frame's pair state). Legal boundary host work; the value
    // enters the graph through the per-launch pinned FrameBeginInput and is
    // idempotent, so a later fallback to IPC_Solver stays value-identical.
    // [C6-e] EVERY attempt, not just attempt 0. A capacity retry re-runs the
    // whole frame from its restored start state, so the frame boundary has to be
    // rebuilt too -- inheriting attempt 0's boundary was what turned foldshirt's
    // grasp-closing frame into a 3.8M-pair swept explosion on the first retry.
    if(!ipc.m_skip_all_collision
       && knob_enabled("STIFF_C4_COLLISION_GRAPH"))
    {
        // [C6-h] Kappa derivation runs on attempt 0 ONLY. initKappa derives
        // kappa from the gradient buffers, and on a retry those hold the FAILED
        // attempt's garbage: beaker frame 1 went kappa 614.4 -> 8.63 (71x too
        // soft) between attempts 2 and 3, the barrier collapsed, the sweep hit
        // 17.3M pairs and the retry budget burned. A retry replays the SAME
        // frame, so it must run at the same kappa: restore_host_attempt has
        // already put attempt 0's post-initKappa value back in ipc.Kappa (the
        // snapshot is taken after this block), and frame_restore_kappa restored
        // m_kappa_group on device. The friction rebuild below stays
        // unconditional (C6-e): it reads the bit-exactly restored vertices.
        if(attempt == 0)
        {
            ipc.upperBoundKappa(ipc.Kappa);
            if(ipc.Kappa < 1e-16)
                ipc.suggestKappa(ipc.Kappa);
            ipc.initKappa(mesh);
        }
#ifdef USE_FRICTION
        // [C4-c] rebuild the lagged friction sets at the frame boundary,
        // exactly where the release IPC_Solver does (frame-start snapshot
        // semantics; the sets are frozen for the whole frame). Also refreshes
        // h_cpNum_last/h_gpNum_last and m_pair_snap_last.
        ipc.ensure_frictionBuffers();
        // Grow to final capacity BEFORE building: the growth is resize_discard,
        // and training used to run it right after this block, discarding the
        // lagged sets it had just built.
        ipc.ensure_graph_friction_capacity();
        ipc.buildFrictionSets();
#endif
    }

    snapshot_host_attempt(ipc, context.host_snapshot);
    context.newton_begin = ipc.m_total_newton_iters;

    *context.h_begin = FrameBeginInput{};
    context.h_begin->frame_id = frame_id;
    context.h_begin->attempt  = attempt;
    context.h_begin->retry_invalid_bits = retry_invalid_bits;
    context.h_begin->path_flags =
        frame_fsm::PATH_GRAPH_REQUESTED
        | frame_fsm::PATH_GRAPH_ACTIVE
        | frame_fsm::PATH_FULL_CONDITIONAL_GRAPH
        | frame_fsm::PATH_PCG_DEVICE_CONTINUATION
        | frame_fsm::PATH_LS_DEVICE_LOOP;
    if(attempt > 0)
        context.h_begin->path_flags |= frame_fsm::PATH_RETRIED;
    context.h_begin->kappa = ipc.Kappa;

    *context.h_terminal = FrameTerminalInput{};
    context.h_terminal->graph_launches  = 1;
    context.h_terminal->host_boundaries = 1;
    context.h_terminal->kappa = ipc.Kappa;

    // The release solver assigns this before its (normally single) substep.
    // It is a capture-time launch parameter for stitch/soft terms; eligibility
    // above rejects the host-driven variants but preserving the assignment
    // keeps plain FEM semantics identical.
    ipc.animation_fullRate = ipc.animation_subRate;
    arm_full_graph_attempt(ipc, context);

    // [C6-b] Hypothesis probe: ABD's intermediate converter offsets are host
    // values baked into the recorded launches, and they advance with every
    // committed q/layout state — the original "ABD bodies are not yet
    // device-conditional" gate said exactly this. If forcing a re-record per
    // frame fixes ABD scenes, the executable REUSE is the defect, not the
    // recording, and the fix is to make those offsets device-resident.
    const bool force_rerecord =
        knob_enabled("STIFF_C6_RERECORD") && context.abd_count != 0;
    if(force_rerecord && context.full_exec)
    {
        cudaGraphExecDestroy(context.full_exec);
        context.full_exec = nullptr;
    }
    if(!context.full_exec
       || context.full_generation != pcg_buffer_generation())
    {
        train_collision_for_capture(ipc, mesh);
        try
        {
            capture_full_graph(ipc, mesh, context);
            // Recording runs the host launch-builders once. None of their
            // bookkeeping is a completed physical frame.
            restore_host_attempt(ipc, context.host_snapshot);
            ipc.animation_fullRate = ipc.animation_subRate;
            // [C4-a/c] tier baseline for executable reuse — recorded from
            // the REAL boundary mirrors (after restore), never the capacity
            // mirrors the recording ran with.
            for(int i = 0; i < 5; ++i)
                context.captured_pair_counts[i] = ipc.h_cpNum[i];
            context.captured_pair_counts[5] = ipc.h_gpNum;
#ifdef USE_FRICTION
            for(int i = 0; i < 5; ++i)
                context.captured_pair_counts[6 + i] =
                    ipc.h_cpNum_last[i];
            context.captured_pair_counts[11] = ipc.h_gpNum_last;
#endif
            context.captured_pair_counts_valid =
                !ipc.m_skip_all_collision;
        }
        catch(const std::exception& error)
        {
            restore_host_attempt(ipc, context.host_snapshot);
            disarm_full_graph_attempt(ipc);
            context.full_capture_failed = true;
            std::fprintf(
                stderr,
                "[frame-full-graph] capture disabled for this engine; "
                "falling back to the audited two-graph transaction: %s\n",
                error.what());
            return false;
        }
    }

    context.h_terminal->root_graph_nodes = context.full_nodes;
    context.h_terminal->root_d2h_nodes   = context.full_d2h;
    context.h_terminal->terminal_graph_nodes = 0;
    context.h_terminal->terminal_d2h_nodes   = 0;

    const auto t_launch = std::chrono::steady_clock::now();
    const cudaError_t launch =
        cudaGraphLaunch(context.full_exec, cudaStreamPerThread);
    if(launch != cudaSuccess)
    {
        cudaGetLastError();
        restore_host_attempt(ipc, context.host_snapshot);
        disarm_full_graph_attempt(ipc);
        throw std::runtime_error(
            std::string("full frame graph launch failed: ")
            + cudaGetErrorString(launch));
    }
    ipc.m_frame_terminal_emitted = true;

    const cudaError_t boundary =
        cudaStreamSynchronize(cudaStreamPerThread);
    if(getenv("STIFF_FRAME_SECTION_TIME"))
    {
        const auto t_done = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr,
                "[frame-sections] pre_launch=%.1fms graph_wait=%.1fms\n",
                ms(t_enter, t_launch),
                ms(t_launch, t_done));
    }
    if(boundary != cudaSuccess)
    {
        cudaGetLastError();
        restore_host_attempt(ipc, context.host_snapshot);
        disarm_full_graph_attempt(ipc);
        throw std::runtime_error(
            std::string("full frame graph boundary failed: ")
            + cudaGetErrorString(boundary));
    }
    return true;
}

std::string status_error(const frame_fsm::FrameStatus& status,
                         const char* cause)
{
    std::ostringstream message;
    message << "[frame-fsm] frame " << status.frame_id
            << " failed: result=" << status.result
            << " phase=" << status.phase
            << " error=" << status.error_code
            << " invalid=0x" << std::hex << status.invalid_bits
            << std::dec
            << " primitive=" << status.err_primitive;
    if(cause && cause[0])
        message << " cause=" << cause;
    return message.str();
}
}  // namespace

void GIPC::destroy_episode_graph()
{
    auto* context =
        static_cast<EpisodeGraphContext*>(m_episode_graph_context);
    if(!context)
        return;
    if(context->in_flight)
    {
        // GPU-native consumers may enqueue observation/reward/policy kernels
        // after the graph's completion event on the bound stream. Teardown is
        // an explicit host boundary, so drain the entire stream before freeing
        // any exported device buffer.
        const cudaError_t result = context->device_native
            ? cudaStreamSynchronize(context->launch_stream)
            : cudaStreamSynchronize(cudaStreamPerThread);
        if(result != cudaSuccess)
            cudaGetLastError();
        disarm_full_graph_attempt(*this);
        context->in_flight = false;
    }
    destroy_episode_context(context);
    m_episode_graph_context = nullptr;
}

void GIPC::prepare_episode_graph(
    device_TetraData& mesh,
    int frame_count,
    const gipc::RevoluteDrivingControlPacked* revolute_actions,
    int revolute_count,
    const gipc::PrismaticDrivingControlPacked* prismatic_actions,
    int prismatic_count,
    bool device_native)
{
    if(frame_count <= 0)
        throw std::invalid_argument(
            "[episode-graph] frame_count must be positive");
    if(revolute_count < 0 || prismatic_count < 0)
        throw std::invalid_argument(
            "[episode-graph] joint counts must be non-negative");
    if(m_total_frames == 0)
        throw std::logic_error(
            "[episode-graph] one synchronous warm-up step is required "
            "before graph capture");
    // [C6-l] Episode-scoped bake behaviour (see the clamp in 13_kappa): an
    // RAII guard so every exit path -- returns and throws alike -- clears it.
    struct EpisodeCaptureFlag
    {
        bool& flag;
        explicit EpisodeCaptureFlag(bool& f) : flag(f) { flag = true; }
        ~EpisodeCaptureFlag() { flag = false; }
    } episode_capture_guard{m_episode_capture};
    if(m_frame_graph_active)
        throw std::logic_error(
            "[episode-graph] another graph transaction is active");
    if(const char* drive_substep = std::getenv("STIFF_DRIVE_SUBSTEP");
       drive_substep && std::atoi(drive_substep) > 1)
        throw std::runtime_error(
            "[episode-graph] STIFF_DRIVE_SUBSTEP>1 is not device-resident");

    prepare_frame_graph(mesh);
    FrameGraphContext& frame = graph_context(*this);
    std::string reason;
    if(!full_graph_eligible(
           *this, mesh, frame, reason, /*allow_abd=*/true))
        throw std::runtime_error(
            std::string("[episode-graph] unsupported scene: ") + reason);
    if(frame.full_capture_failed)
        throw std::runtime_error(
            "[episode-graph] whole-frame capture was previously disabled");

    const int expected_revolute =
        m_abd_system ? m_abd_system->m_num_revolute_driving : 0;
    const int expected_prismatic =
        m_abd_system ? m_abd_system->m_num_prismatic_driving : 0;
    if(revolute_count != expected_revolute
       || prismatic_count != expected_prismatic)
    {
        std::ostringstream message;
        message
            << "[episode-graph] action shape mismatch: expected "
            << expected_revolute << " revolute and "
            << expected_prismatic << " prismatic controls, got "
            << revolute_count << " and " << prismatic_count;
        throw std::invalid_argument(message.str());
    }
    if(!device_native
       && ((revolute_count && !revolute_actions)
           || (prismatic_count && !prismatic_actions)))
        throw std::invalid_argument(
            "[episode-graph] non-empty controls require action data");

    auto upload_actions = [&](EpisodeGraphContext& context)
    {
        if(context.device_native || context.action_bytes == 0)
            return;
        std::vector<unsigned char> staging(context.action_bytes);
        if(context.prismatic_action_offset)
            std::memcpy(
                staging.data(),
                revolute_actions,
                context.prismatic_action_offset);
        const size_t prismatic_bytes =
            context.action_bytes
            - context.prismatic_action_offset;
        if(prismatic_bytes)
            std::memcpy(
                staging.data()
                    + context.prismatic_action_offset,
                prismatic_actions,
                prismatic_bytes);
        CUDA_SAFE_CALL(cudaMemcpy(
            context.d_actions,
            staging.data(),
            context.action_bytes,
            cudaMemcpyHostToDevice));
    };

    auto* existing =
        static_cast<EpisodeGraphContext*>(m_episode_graph_context);
    if(existing)
    {
        if(existing->in_flight)
            throw std::logic_error(
                "[episode-graph] the previous episode is still in flight");
        const bool compatible =
            existing->frame_count == frame_count
            && existing->vertex_count == frame.vertex_count
            && existing->revolute_count == revolute_count
            && existing->prismatic_count == prismatic_count
            && existing->device_native == device_native
            && existing->generation == pcg_buffer_generation();
        if(compatible)
        {
            upload_actions(*existing);
            return;
        }
        destroy_episode_context(existing);
        m_episode_graph_context = nullptr;
    }

    auto* context = new EpisodeGraphContext{};
    m_episode_graph_context = context;
    try
    {
        context->frame_count     = frame_count;
        context->split_frame     = (frame_count + 1) / 2;
        context->revolute_count  = revolute_count;
        context->prismatic_count = prismatic_count;
        context->vertex_count    = frame.vertex_count;
        context->device_native   = device_native;

        const size_t observations = checked_product(
            static_cast<size_t>(frame_count),
            context->vertex_count,
            "episode observations");
        const size_t revolute_elements = checked_product(
            static_cast<size_t>(frame_count),
            static_cast<size_t>(revolute_count),
            "revolute actions");
        const size_t prismatic_elements = checked_product(
            static_cast<size_t>(frame_count),
            static_cast<size_t>(prismatic_count),
            "prismatic actions");
        const size_t revolute_bytes = checked_product(
            revolute_elements,
            sizeof(gipc::RevoluteDrivingControlPacked),
            "revolute action bytes");
        const size_t prismatic_bytes = checked_product(
            prismatic_elements,
            sizeof(gipc::PrismaticDrivingControlPacked),
            "prismatic action bytes");
        if(prismatic_bytes
           > std::numeric_limits<size_t>::max()
                 - revolute_bytes)
            throw std::overflow_error(
                "[episode-graph] size overflow: packed actions");
        static_assert(
            alignof(gipc::RevoluteDrivingControlPacked)
            == alignof(gipc::PrismaticDrivingControlPacked));
        static_assert(
            sizeof(gipc::RevoluteDrivingControlPacked)
                == 3 * sizeof(gipc::Float)
            && sizeof(gipc::PrismaticDrivingControlPacked)
                   == 3 * sizeof(gipc::Float),
            "GPU-native RL actions require three tightly packed scalars");
        context->prismatic_action_offset = revolute_bytes;
        context->action_bytes =
            revolute_bytes + prismatic_bytes;
        device_alloc(context->d_input, 1);
        device_alloc(context->d_control, 1);
        device_alloc(context->d_frame_counter, 1);
        if(!device_native)
            device_alloc(context->d_ready_one, 1);
        device_alloc(
            context->d_actions, context->action_bytes);
        if(context->d_actions)
        {
            context->d_revolute_actions =
                reinterpret_cast<
                    gipc::RevoluteDrivingControlPacked*>(
                    context->d_actions);
            context->d_prismatic_actions =
                reinterpret_cast<
                    gipc::PrismaticDrivingControlPacked*>(
                    context->d_actions
                    + context->prismatic_action_offset);
            CUDA_SAFE_CALL(cudaMemset(
                context->d_actions, 0, context->action_bytes));
        }
        device_alloc(context->d_positions, observations);
        device_alloc(context->d_velocities, observations);
        device_alloc(
            context->d_statuses,
            static_cast<size_t>(frame_count));

        if(device_native)
        {
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(
                &context->completion_event,
                cudaEventDisableTiming));
            // [D2] joint observation block + in-stream reset snapshot.
            context->joint_obs_count =
                2 * revolute_count + 2 * prismatic_count;
            if(context->joint_obs_count > 0)
            {
                device_alloc(context->d_joint_obs,
                             static_cast<size_t>(
                                 context->joint_obs_count));
                CUDA_SAFE_CALL(cudaMemset(
                    context->d_joint_obs,
                    0,
                    static_cast<size_t>(context->joint_obs_count)
                        * sizeof(double)));
            }
            context->reset_vertex_count = frame.vertex_count;
            context->reset_abd_count =
                static_cast<int>(frame.abd_count);
            context->mesh_vertexes   = mesh.vertexes;
            context->mesh_o_vertexes = mesh.o_vertexes;
            context->mesh_velocities = mesh.velocities;
            context->mesh_x_tilta    = mesh.xTilta;
            context->reset_p2g = mesh.d_point_to_group;
            context->reset_b2g = mesh.d_body_to_group;
            if(context->reset_vertex_count)
            {
                device_alloc(context->reset_fem_x,
                             context->reset_vertex_count);
                device_alloc(context->reset_fem_ox,
                             context->reset_vertex_count);
                device_alloc(context->reset_fem_v,
                             context->reset_vertex_count);
                device_alloc(context->reset_fem_xt,
                             context->reset_vertex_count);
            }
            if(context->reset_abd_count)
            {
                device_alloc(context->reset_abd_q,
                             static_cast<size_t>(
                                 context->reset_abd_count));
                device_alloc(context->reset_abd_q_prev,
                             static_cast<size_t>(
                                 context->reset_abd_count));
                device_alloc(context->reset_abd_q_v,
                             static_cast<size_t>(
                                 context->reset_abd_count));
                device_alloc(context->reset_abd_q_tilde,
                             static_cast<size_t>(
                                 context->reset_abd_count));
            }
        }
        else
        {
            pinned_alloc(context->h_input, 1);
            pinned_alloc(context->h_ready, 2);
            pinned_alloc(context->h_slot_attempted, 2);
            std::memset(context->h_ready, 0, 2 * sizeof(int));
            std::memset(
                context->h_slot_attempted, 0, 2 * sizeof(int));
            for(int slot = 0; slot < 2; ++slot)
            {
                const int count =
                    slot == 0
                        ? context->split_frame
                        : frame_count - context->split_frame;
                const size_t elements = checked_product(
                    static_cast<size_t>(count),
                    context->vertex_count,
                    "pinned observation slot");
                pinned_alloc(context->h_positions[slot], elements);
                pinned_alloc(context->h_velocities[slot], elements);
                pinned_alloc(
                    context->h_statuses[slot],
                    static_cast<size_t>(count));
                CUDA_SAFE_CALL(cudaEventCreateWithFlags(
                    &context->slot_event[slot],
                    cudaEventDisableTiming));
            }
            const int one = 1;
            CUDA_SAFE_CALL(cudaMemcpy(
                context->d_ready_one,
                &one,
                sizeof(one),
                cudaMemcpyHostToDevice));
        }
        CUDA_SAFE_CALL(cudaMemset(
            context->d_frame_counter, 0, sizeof(int64_t)));
        upload_actions(*context);

        EpisodeRuntimeInput initial_input{};
        initial_input.frame_count = frame_count;
        initial_input.path_flags =
            frame_fsm::PATH_GRAPH_REQUESTED
            | frame_fsm::PATH_GRAPH_ACTIVE
            | frame_fsm::PATH_FULL_CONDITIONAL_GRAPH
            | frame_fsm::PATH_PCG_DEVICE_CONTINUATION
            | frame_fsm::PATH_LS_DEVICE_LOOP
            | frame_fsm::PATH_EPISODE_RESIDENT
            | (device_native ? frame_fsm::PATH_GPU_NATIVE_RL : 0u);
        initial_input.kappa = Kappa;
        if(device_native)
            CUDA_SAFE_CALL(cudaMemcpy(
                context->d_input,
                &initial_input,
                sizeof(initial_input),
                cudaMemcpyHostToDevice));
        else
            *context->h_input = initial_input;

        snapshot_host_attempt(*this, context->host_snapshot);
        animation_fullRate = animation_subRate;
        arm_full_graph_attempt(*this, frame);
        try
        {
            // [C4] collision workspaces must be trained before the episode
            // capture too (same contract as the step whole-frame path).
            train_collision_for_capture(*this, mesh);
            capture_episode_graph(*this, mesh, frame, *context);
            restore_host_attempt(*this, context->host_snapshot);
            disarm_full_graph_attempt(*this);
            if(device_native)
            {
                initial_input.graph_nodes = context->graph_nodes;
                initial_input.graph_d2h   = context->graph_d2h;
                CUDA_SAFE_CALL(cudaMemcpy(
                    context->d_input,
                    &initial_input,
                    sizeof(initial_input),
                    cudaMemcpyHostToDevice));
                // [D2] pre-fill the joint observation block twice so the
                // buffer-difference rate starts at an exact zero.
                if(context->d_joint_obs)
                {
                    m_abd_system->enqueue_joint_observations(
                        *m_abd_sim_data, context->d_joint_obs, IPC_dt);
                    m_abd_system->enqueue_joint_observations(
                        *m_abd_sim_data, context->d_joint_obs, IPC_dt);
                }
                // [D2] capture the reset snapshot from the committed state
                // (setup boundary; synchronous copies are fine here).
                if(context->reset_vertex_count)
                {
                    const size_t bytes = context->reset_vertex_count
                                         * sizeof(double3);
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_fem_x, mesh.vertexes, bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_fem_ox, mesh.o_vertexes, bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_fem_v, mesh.velocities, bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_fem_xt, mesh.xTilta, bytes,
                        cudaMemcpyDeviceToDevice));
                }
                if(context->reset_abd_count)
                {
                    auto& abd = m_abd_sim_data->device;
                    const size_t bytes =
                        static_cast<size_t>(context->reset_abd_count)
                        * sizeof(gipc::Vector12);
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_abd_q,
                        abd.body_id_to_q.data(), bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_abd_q_prev,
                        abd.body_id_to_q_prev.data(), bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_abd_q_v,
                        abd.body_id_to_q_v.data(), bytes,
                        cudaMemcpyDeviceToDevice));
                    CUDA_SAFE_CALL(cudaMemcpy(
                        context->reset_abd_q_tilde,
                        abd.body_id_to_q_tilde.data(), bytes,
                        cudaMemcpyDeviceToDevice));
                }
                CUDA_SAFE_CALL(
                    cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
        catch(...)
        {
            restore_host_attempt(*this, context->host_snapshot);
            disarm_full_graph_attempt(*this);
            throw;
        }
    }
    catch(...)
    {
        destroy_episode_context(context);
        m_episode_graph_context = nullptr;
        throw;
    }
}

void GIPC::launch_episode_graph_async(device_TetraData&,
                                      int64_t base_frame_id)
{
    EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[episode-graph] use launch_gpu_rl_graph_async() for a "
            "GPU-native RL graph");
    if(context.in_flight)
        throw std::logic_error(
            "[episode-graph] an episode is already in flight");
    if(!context.exec
       || context.generation != pcg_buffer_generation())
        throw std::runtime_error(
            "[episode-graph] graph buffers changed; prepare the episode again");

    FrameGraphContext& frame = graph_context(*this);
    snapshot_host_attempt(*this, context.host_snapshot);
    *context.h_input = EpisodeRuntimeInput{};
    context.h_input->base_frame_id = base_frame_id;
    context.h_input->frame_count   = context.frame_count;
    context.h_input->path_flags =
        frame_fsm::PATH_GRAPH_REQUESTED
        | frame_fsm::PATH_GRAPH_ACTIVE
        | frame_fsm::PATH_FULL_CONDITIONAL_GRAPH
        | frame_fsm::PATH_PCG_DEVICE_CONTINUATION
        | frame_fsm::PATH_LS_DEVICE_LOOP
        | frame_fsm::PATH_EPISODE_RESIDENT;
    context.h_input->graph_nodes = context.graph_nodes;
    context.h_input->graph_d2h   = context.graph_d2h;
    context.h_input->kappa       = Kappa;
    std::memset(context.h_ready, 0, 2 * sizeof(int));
    std::memset(
        context.h_slot_attempted, 0, 2 * sizeof(int));

    animation_fullRate = animation_subRate;
    arm_full_graph_attempt(*this, frame);
    const cudaError_t launch =
        cudaGraphLaunch(context.exec, cudaStreamPerThread);
    if(launch != cudaSuccess)
    {
        cudaGetLastError();
        restore_host_attempt(*this, context.host_snapshot);
        disarm_full_graph_attempt(*this);
        throw std::runtime_error(
            std::string("[episode-graph] launch failed: ")
            + cudaGetErrorString(launch));
    }
    context.in_flight = true;
    m_frame_terminal_emitted = true;
}

void GIPC::launch_gpu_rl_graph_async(uintptr_t cuda_stream)
{
    EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error(
            "[gpu-rl] prepare a GPU-native RL graph before launch");
    if(context.frame_count != 1)
        throw std::logic_error(
            "[gpu-rl] closed-loop execution requires a one-frame graph");
    if(!context.exec
       || context.generation != pcg_buffer_generation())
        throw std::runtime_error(
            "[gpu-rl] graph buffers changed; prepare the graph again");

    cudaStream_t stream = cuda_stream
        ? reinterpret_cast<cudaStream_t>(cuda_stream)
        : cudaStreamPerThread;
    if(context.in_flight && context.launch_stream != stream)
        throw std::logic_error(
            "[gpu-rl] repeated launches must use the same CUDA stream; "
            "end_gpu_rl() before changing streams");

    const bool first_launch = !context.in_flight;
    if(first_launch)
    {
        FrameGraphContext& frame = graph_context(*this);
        snapshot_host_attempt(*this, context.host_snapshot);
        animation_fullRate = animation_subRate;
        arm_full_graph_attempt(*this, frame);
        context.launch_stream = stream;
    }

    const cudaError_t launch = cudaGraphLaunch(context.exec, stream);
    if(launch != cudaSuccess)
    {
        cudaGetLastError();
        if(first_launch)
        {
            restore_host_attempt(*this, context.host_snapshot);
            disarm_full_graph_attempt(*this);
            context.launch_stream = nullptr;
        }
        throw std::runtime_error(
            std::string("[gpu-rl] graph launch failed: ")
            + cudaGetErrorString(launch));
    }
    const cudaError_t record =
        cudaEventRecord(context.completion_event, stream);
    if(record != cudaSuccess)
    {
        cudaGetLastError();
        cudaStreamSynchronize(stream);
        restore_host_attempt(*this, context.host_snapshot);
        disarm_full_graph_attempt(*this);
        context.launch_stream = nullptr;
        context.in_flight = false;
        throw std::runtime_error(
            std::string("[gpu-rl] completion event record failed: ")
            + cudaGetErrorString(record));
    }
    context.in_flight = true;
    m_frame_terminal_emitted = true;
}

bool GIPC::gpu_rl_graph_prepared() const
{
    const auto* context =
        static_cast<const EpisodeGraphContext*>(
            m_episode_graph_context);
    return context && context->device_native && context->exec;
}

bool GIPC::gpu_rl_graph_ready() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    if(!context.in_flight)
        return false;
    const cudaError_t query =
        cudaEventQuery(context.completion_event);
    if(query == cudaSuccess)
        return true;
    if(query == cudaErrorNotReady)
        return false;
    cudaGetLastError();
    throw std::runtime_error(
        std::string("[gpu-rl] completion query failed: ")
        + cudaGetErrorString(query));
}

void GIPC::synchronize_gpu_rl_graph() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    if(!context.in_flight)
        throw std::logic_error("[gpu-rl] no GPU-native RL step was launched");
    const cudaError_t result =
        cudaEventSynchronize(context.completion_event);
    if(result != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(
            std::string("[gpu-rl] completion wait failed: ")
            + cudaGetErrorString(result));
    }
}

uintptr_t GIPC::gpu_rl_revolute_actions_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_revolute_actions);
}

uintptr_t GIPC::gpu_rl_prismatic_actions_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_prismatic_actions);
}

uintptr_t GIPC::gpu_rl_positions_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_positions);
}

uintptr_t GIPC::gpu_rl_velocities_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_velocities);
}

uintptr_t GIPC::gpu_rl_statuses_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_statuses);
}

uintptr_t GIPC::gpu_rl_frame_counter_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_frame_counter);
}

uintptr_t GIPC::gpu_rl_joint_observations_device_ptr() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return reinterpret_cast<uintptr_t>(context.d_joint_obs);
}

int GIPC::gpu_rl_joint_observation_count() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return context.joint_obs_count;
}

// [D2] in-stream selective reset: replay the prepare-time committed state
// with pure D2D copies on the caller's stream. No host synchronization; the
// next frame graph launch rebuilds collision/contact state from scratch in
// its own prologue. The device frame counter is intentionally left running
// (episode bookkeeping is the caller's policy decision).
// [D3 mask-reset] Selective per-env reset: envs whose mask entry is nonzero
// snap back to the prepare-time state; every other env is untouched. Vertices
// and bodies with no group (-1) never reset through the masked path.
__global__ void _masked_reset_fem(double3* x, double3* ox, double3* v,
                                  double3* xt, const double3* sx,
                                  const double3* sox, const double3* sv,
                                  const double3* sxt, const int* p2g,
                                  const int* mask, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    const int g = p2g ? p2g[i] : -1;
    if(g < 0 || !mask[g]) return;
    x[i] = sx[i]; ox[i] = sox[i]; v[i] = sv[i]; xt[i] = sxt[i];
}

__global__ void _masked_reset_abd(gipc::Vector12* q, gipc::Vector12* qp,
                                  gipc::Vector12* qv, gipc::Vector12* qt,
                                  const gipc::Vector12* sq,
                                  const gipc::Vector12* sqp,
                                  const gipc::Vector12* sqv,
                                  const gipc::Vector12* sqt, const int* b2g,
                                  const int* mask, int n)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= n) return;
    const int g = b2g ? b2g[b] : -1;
    if(g < 0 || !mask[g]) return;
    q[b] = sq[b]; qp[b] = sqp[b]; qv[b] = sqv[b]; qt[b] = sqt[b];
}

void GIPC::launch_gpu_rl_reset_masked_async(uintptr_t d_env_mask,
                                            uintptr_t cuda_stream)
{
    EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error(
            "[gpu-rl] prepare a GPU-native RL graph before masked reset");
    if(!d_env_mask)
        throw std::invalid_argument("[gpu-rl] null device env mask");
    if(!context.reset_p2g && context.reset_vertex_count)
        throw std::logic_error(
            "[gpu-rl] masked reset needs per-env vertex groups "
            "(point_to_group is null -- single ungrouped scene?)");
    if(context.in_flight && context.launch_stream && cuda_stream
       && reinterpret_cast<cudaStream_t>(cuda_stream)
              != context.launch_stream)
        throw std::logic_error(
            "[gpu-rl] masked reset must use the bound CUDA stream");
    cudaStream_t stream = cuda_stream
        ? reinterpret_cast<cudaStream_t>(cuda_stream)
        : cudaStreamPerThread;
    const int* mask = reinterpret_cast<const int*>(d_env_mask);
    if(context.reset_vertex_count)
    {
        const int n = static_cast<int>(context.reset_vertex_count);
        _masked_reset_fem<<<(n + 255) / 256, 256, 0, stream>>>(
            context.mesh_vertexes, context.mesh_o_vertexes,
            context.mesh_velocities, context.mesh_x_tilta,
            context.reset_fem_x, context.reset_fem_ox, context.reset_fem_v,
            context.reset_fem_xt, context.reset_p2g, mask, n);
    }
    if(context.reset_abd_count)
    {
        auto& abd = m_abd_sim_data->device;
        const int n = context.reset_abd_count;
        _masked_reset_abd<<<(n + 255) / 256, 256, 0, stream>>>(
            abd.body_id_to_q.data(), abd.body_id_to_q_prev.data(),
            abd.body_id_to_q_v.data(), abd.body_id_to_q_tilde.data(),
            context.reset_abd_q, context.reset_abd_q_prev,
            context.reset_abd_q_v, context.reset_abd_q_tilde,
            context.reset_b2g, mask, n);
    }
}

void GIPC::launch_gpu_rl_reset_async(uintptr_t cuda_stream)
{
    EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error(
            "[gpu-rl] prepare a GPU-native RL graph before reset");
    if(context.in_flight && context.launch_stream
       && cuda_stream
       && reinterpret_cast<cudaStream_t>(cuda_stream)
              != context.launch_stream)
        throw std::logic_error(
            "[gpu-rl] reset must use the bound CUDA stream");
    cudaStream_t stream = cuda_stream
        ? reinterpret_cast<cudaStream_t>(cuda_stream)
        : cudaStreamPerThread;
    if(context.reset_vertex_count)
    {
        const size_t bytes =
            context.reset_vertex_count * sizeof(double3);
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.mesh_vertexes, context.reset_fem_x, bytes,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.mesh_o_vertexes, context.reset_fem_ox, bytes,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.mesh_velocities, context.reset_fem_v, bytes,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            context.mesh_x_tilta, context.reset_fem_xt, bytes,
            cudaMemcpyDeviceToDevice, stream));
    }
    if(context.reset_abd_count)
    {
        auto& abd = m_abd_sim_data->device;
        const size_t bytes =
            static_cast<size_t>(context.reset_abd_count)
            * sizeof(gipc::Vector12);
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            abd.body_id_to_q.data(), context.reset_abd_q, bytes,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            abd.body_id_to_q_prev.data(), context.reset_abd_q_prev,
            bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            abd.body_id_to_q_v.data(), context.reset_abd_q_v, bytes,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            abd.body_id_to_q_tilde.data(), context.reset_abd_q_tilde,
            bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

int GIPC::gpu_rl_graph_node_count() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return context.graph_nodes;
}

int GIPC::gpu_rl_graph_h2d_count() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return context.graph_h2d;
}

int GIPC::gpu_rl_graph_d2h_count() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(!context.device_native)
        throw std::logic_error("[gpu-rl] no GPU-native RL graph is prepared");
    return context.graph_d2h;
}

bool GIPC::episode_graph_in_flight() const
{
    const auto* context =
        static_cast<const EpisodeGraphContext*>(
            m_episode_graph_context);
    return context && context->in_flight;
}

bool GIPC::episode_observation_ready(int slot) const
{
    if(slot < 0 || slot > 1)
        throw std::out_of_range(
            "[episode-graph] observation slot must be 0 or 1");
    const EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[gpu-rl] device observations must be consumed through the "
            "device ABI");
    const volatile int* ready = context.h_ready + slot;
    if(*ready == 0)
        return false;
    const cudaError_t query =
        cudaEventQuery(context.slot_event[slot]);
    if(query == cudaSuccess)
        return true;
    if(query == cudaErrorNotReady)
        return false;
    cudaGetLastError();
    throw std::runtime_error(
        std::string("[episode-graph] event query failed: ")
        + cudaGetErrorString(query));
}

void GIPC::wait_episode_observation(int slot) const
{
    if(slot < 0 || slot > 1)
        throw std::out_of_range(
            "[episode-graph] observation slot must be 0 or 1");
    const EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[gpu-rl] device observations have no host observation slot");
    const volatile int* ready = context.h_ready + slot;
    if(*ready == 0 && !context.in_flight)
        throw std::logic_error(
            "[episode-graph] no episode is producing this slot");
    while(*ready == 0)
        std::this_thread::yield();
    const cudaError_t wait =
        cudaEventSynchronize(context.slot_event[slot]);
    if(wait != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(
            std::string("[episode-graph] observation fence failed: ")
            + cudaGetErrorString(wait));
    }
}

int GIPC::episode_slot_first_frame(int slot) const
{
    if(slot < 0 || slot > 1)
        throw std::out_of_range(
            "[episode-graph] observation slot must be 0 or 1");
    const EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[gpu-rl] device observations have no host observation slot");
    return slot == 0 ? 0 : context.split_frame;
}

int GIPC::episode_attempted_frame_count() const
{
    const EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[gpu-rl] read the device frame counter through the device ABI");
    int attempted = 0;
    for(int slot = 0; slot < 2; ++slot)
    {
        const volatile int* ready = context.h_ready + slot;
        if(*ready)
            attempted = std::max(
                attempted, context.h_slot_attempted[slot]);
    }
    return std::clamp(attempted, 0, context.frame_count);
}

int GIPC::episode_slot_frame_count(int slot) const
{
    if(!episode_observation_ready(slot))
        throw std::logic_error(
            "[episode-graph] observation slot is not ready");
    const EpisodeGraphContext& context = episode_context(*this);
    const int attempted = std::clamp(
        context.h_slot_attempted[slot],
        0,
        context.frame_count);
    if(slot == 0)
        return std::min(attempted, context.split_frame);
    return std::max(
        0,
        attempted - context.split_frame);
}

void GIPC::copy_episode_observation_slot(
    int slot,
    double3* positions,
    double3* velocities,
    frame_fsm::FrameStatus* statuses,
    int frame_capacity) const
{
    if(!episode_observation_ready(slot))
        throw std::logic_error(
            "[episode-graph] observation slot is not ready");
    const EpisodeGraphContext& context = episode_context(*this);
    const int count = episode_slot_frame_count(slot);
    if(frame_capacity < count)
        throw std::invalid_argument(
            "[episode-graph] output frame capacity is too small");
    const size_t elements = checked_product(
        static_cast<size_t>(count),
        context.vertex_count,
        "observation copy");
    if(elements && (!positions || !velocities))
        throw std::invalid_argument(
            "[episode-graph] null observation output");
    if(count && !statuses)
        throw std::invalid_argument(
            "[episode-graph] null status output");
    if(elements)
    {
        std::memcpy(
            positions,
            context.h_positions[slot],
            elements * sizeof(double3));
        std::memcpy(
            velocities,
            context.h_velocities[slot],
            elements * sizeof(double3));
    }
    if(count)
        std::memcpy(
            statuses,
            context.h_statuses[slot],
            static_cast<size_t>(count)
                * sizeof(frame_fsm::FrameStatus));
}

int GIPC::finish_episode_graph()
{
    EpisodeGraphContext& context = episode_context(*this);
    if(context.device_native)
        throw std::logic_error(
            "[gpu-rl] use end_gpu_rl() to leave GPU-native RL mode");
    if(!context.in_flight)
        throw std::logic_error(
            "[episode-graph] no episode is in flight");
    try
    {
        wait_episode_observation(1);
    }
    catch(...)
    {
        restore_host_attempt(*this, context.host_snapshot);
        disarm_full_graph_attempt(*this);
        context.in_flight = false;
        throw;
    }

    const int attempted = std::clamp(
        context.h_slot_attempted[1],
        0,
        context.frame_count);
    int successful = 0;
    int newton_iters = 0;
    int pcg_iters    = 0;
    for(int frame = 0; frame < attempted; ++frame)
    {
        const frame_fsm::FrameStatus& status =
            frame < context.split_frame
                ? context.h_statuses[0][frame]
                : context.h_statuses[1][
                      frame - context.split_frame];
        if(status.result != frame_fsm::FRAME_OK)
            break;
        ++successful;
        newton_iters += status.newton_iters;
        pcg_iters    += status.pcg_iters;
    }
    if(attempted)
    {
        const int last = attempted - 1;
        m_last_frame_status =
            last < context.split_frame
                ? context.h_statuses[0][last]
                : context.h_statuses[1][
                      last - context.split_frame];
    }

    m_total_frames =
        context.host_snapshot.total_frames + successful;
    m_total_newton_iters =
        context.host_snapshot.total_newton_iters + newton_iters;
    m_total_pcg_iters =
        context.host_snapshot.total_pcg_iters + pcg_iters;
    animation_fullRate = animation_subRate;
    disarm_full_graph_attempt(*this);
    context.in_flight = false;
    return successful;
}

void GIPC::destroy_frame_graph()
{
    // Episode nodes borrow the frame snapshots and state packet.
    destroy_episode_graph();
    destroy_context(
        static_cast<FrameGraphContext*>(m_frame_graph_context));
    m_frame_graph_context   = nullptr;
    m_frame_graph_active    = false;
    m_frame_terminal_emitted = false;
}

void GIPC::prepare_frame_graph(device_TetraData& mesh)
{
    if(m_frame_graph_context)
        return;

    auto* context = new FrameGraphContext{};
    m_frame_graph_context = context;
    try
    {
        context->vertex_count = vertexNum;
        context->abd_count =
            m_abd_sim_data
                ? m_abd_sim_data->device.body_id_to_q.size()
                : 0;
        context->group_count =
            m_pergroup_kappa && m_kappa_group
                ? static_cast<size_t>(m_active_group_count)
                : 0;

        device_alloc(context->d_state, 1);
        device_alloc(context->d_status, 1);
        device_alloc(context->d_begin, 1);
        device_alloc(context->d_terminal, 1);
        device_alloc(context->fem_vertexes, context->vertex_count);
        device_alloc(context->fem_o_vertexes, context->vertex_count);
        device_alloc(context->fem_velocities, context->vertex_count);
        device_alloc(context->fem_x_tilta, context->vertex_count);
        device_alloc(context->abd_q, context->abd_count);
        device_alloc(context->abd_q_prev, context->abd_count);
        device_alloc(context->abd_q_v, context->abd_count);
        device_alloc(context->abd_q_tilde, context->abd_count);
        device_alloc(context->kappa_group, context->group_count);
        device_alloc(context->kappa_snapshot, context->group_count);

        CUDA_SAFE_CALL(cudaHostAlloc(
            reinterpret_cast<void**>(&context->h_begin),
            sizeof(FrameBeginInput),
            cudaHostAllocPortable));
        CUDA_SAFE_CALL(cudaHostAlloc(
            reinterpret_cast<void**>(&context->h_terminal),
            sizeof(FrameTerminalInput),
            cudaHostAllocPortable));
        CUDA_SAFE_CALL(cudaHostAlloc(
            reinterpret_cast<void**>(&context->h_status),
            sizeof(frame_fsm::FrameStatus),
            cudaHostAllocPortable));
        *context->h_begin    = FrameBeginInput{};
        *context->h_terminal = FrameTerminalInput{};
        *context->h_status   = frame_fsm::FrameStatus{};

        capture_root_graph(*this, mesh, *context);
        capture_terminal_graph(*this, mesh, *context);
        if(context->root_d2h != 0 || context->terminal_d2h != 1)
            throw std::runtime_error(
                "frame graph audit requires root D2H=0 and terminal D2H=1");
    }
    catch(...)
    {
        destroy_frame_graph();
        throw;
    }
}

void GIPC::frame_graph_begin(device_TetraData& mesh,
                             int64_t frame_id,
                             int attempt,
                             uint32_t retry_invalid_bits)
{
    int contact_stable_count = 0;
    for(int s = 0; s < 4; ++s)
        contact_stable_count +=
            gipc_global_triplet.m_contact_class_tier[s];
    if(contact_stable_count > 0)
    {
        const int sort_capacity =
            gipc::assembly_capacity_tier(contact_stable_count);
        const size_t need =
            static_cast<size_t>(sort_capacity)
            + static_cast<size_t>(contact_stable_count);
        if(gipc_global_triplet.triplet_capacity() < need)
        {
            gipc_global_triplet.global_triplet_offset = 0;
            gipc_global_triplet.global_collision_triplet_offset = 0;
            gipc_global_triplet.open_discard_window();
            gipc_global_triplet.ensure_capacity_discard(need);
        }
        if(gipc_global_triplet.global_external_max_capcity
           < sort_capacity)
        {
            gipc_global_triplet.resize_collision_hash_size(
                static_cast<size_t>(sort_capacity));
            gipc_global_triplet.global_external_max_capcity =
                sort_capacity;
            ++pcg_buffer_generation();
        }
    }
    prepare_frame_graph(mesh);
    FrameGraphContext& context = graph_context(*this);
    snapshot_host_attempt(*this, context.host_snapshot);
    context.newton_begin = m_total_newton_iters;

    *context.h_begin = FrameBeginInput{};
    context.h_begin->frame_id = frame_id;
    context.h_begin->attempt  = attempt;
    context.h_begin->retry_invalid_bits = retry_invalid_bits;
    context.h_begin->path_flags =
        frame_fsm::PATH_GRAPH_REQUESTED
        | frame_fsm::PATH_GRAPH_ACTIVE
        | frame_fsm::PATH_HOST_PHASE_BRIDGE
        | frame_fsm::PATH_PCG_DEVICE_CONTINUATION;
    if(attempt > 0)
        context.h_begin->path_flags |= frame_fsm::PATH_RETRIED;
    gipc_global_triplet.m_abd_unique_test_tier = 0;
    bool force_unique_tier = false;
    if(const char* forced_tier =
           std::getenv("STIFF_FRAME_FORCE_UNIQUE_TIER"))
    {
        const int tier = std::atoi(forced_tier);
        if(tier > 0)
        {
            force_unique_tier = true;
            if(attempt == 0)
                gipc_global_triplet.m_abd_unique_test_tier = tier;
            context.h_begin->path_flags |=
                frame_fsm::PATH_TEST_INJECTION;
        }
    }
    context.h_begin->kappa = Kappa;

    const cudaError_t launch =
        cudaGraphLaunch(context.root_exec, cudaStreamPerThread);
    if(launch != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(
            std::string("frame root graph launch failed: ")
            + cudaGetErrorString(launch));
    }
    if(m_global_linear_system)
        m_global_linear_system->set_frame_device_state(context.d_state);
    gipc_global_triplet.m_frame_device_state = context.d_state;
    gipc_global_triplet.m_contact_partition_txn_ok = true;
    const char* abd_tier = std::getenv("STIFF_ABD_TIER");
    gipc_global_triplet.m_abd_tier_txn_ok =
        force_unique_tier
        || (abd_tier && abd_tier[0] && std::atoi(abd_tier) != 0);
    m_frame_graph_active     = true;
    m_frame_terminal_emitted = false;
}

void GIPC::frame_graph_enqueue_terminal(device_TetraData&,
                                        int result,
                                        int error_code,
                                        uint32_t invalid_bits,
                                        int err_env,
                                        int err_primitive)
{
    FrameGraphContext& context = graph_context(*this);
    if(m_frame_terminal_emitted)
        throw std::logic_error("frame terminal graph emitted twice");

    *context.h_terminal = FrameTerminalInput{};
    FrameTerminalInput& input = *context.h_terminal;
    input.result        = result;
    input.error_code    = error_code;
    input.invalid_bits  = invalid_bits;
    input.err_env       = err_env;
    input.err_primitive = err_primitive;
    input.newton_iters  =
        std::max(0, m_total_newton_iters - context.newton_begin);
    input.pcg_iters = static_cast<int>(std::max(
        0.0,
        m_total_pcg_iters - context.host_snapshot.total_pcg_iters));
    input.hw_dcd_pairs  = static_cast<int>(h_cpNum[0]);
    input.hw_ccd_pairs  = static_cast<int>(m_last_ccd_pair_count);
    input.hw_triplets   = gipc_global_triplet.global_triplet_offset;
    input.hw_unique_blocks =
        static_cast<int>(gipc_global_triplet.h_unique_key_number);
    input.kappa = Kappa;
    input.root_graph_nodes     = context.root_nodes;
    input.root_d2h_nodes       = context.root_d2h;
    input.terminal_graph_nodes = context.terminal_nodes;
    input.terminal_d2h_nodes   = context.terminal_d2h;

    const char* force = std::getenv("STIFF_FRAME_FORCE_ROLLBACK");
    if(input.result == frame_fsm::FRAME_OK && force && force[0]
       && force[0] != '0')
    {
        input.result       = frame_fsm::FRAME_RETRY_REQUIRED;
        input.error_code   = frame_fsm::ERR_CAPACITY;
        input.invalid_bits = frame_fsm::OVF_DCD_PAIRS;
    }

    const cudaError_t launch =
        cudaGraphLaunch(context.terminal_exec, cudaStreamPerThread);
    if(launch != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error(
            std::string("frame terminal graph launch failed: ")
            + cudaGetErrorString(launch));
    }
    m_frame_terminal_emitted = true;
}

int GIPC::frame_graph_finish_terminal()
{
    FrameGraphContext& context = graph_context(*this);
    if(!m_frame_terminal_emitted)
        throw std::logic_error("frame terminal graph was not emitted");
    m_last_frame_status = *context.h_status;
    m_graph_tier_grew   = false;   // [C6-b] per-adjudication
    // [C6-e] Mirror the host's step-health counters: an in-graph line search
    // that exhausted its budget accepted a non-descent step, exactly as the
    // host's WARN path does, and an RL loop diffs these across step().
    if(m_last_frame_status.invalid_bits & frame_fsm::INV_LS_BUDGET)
    {
        ++m_ls_exhausted_total;
        if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
            fprintf(stderr,
                    "[line-search][WARN] in-graph budget exhausted on frame "
                    "%lld: step accepted anyway (host policy)\n",
                    (long long)m_last_frame_status.frame_id);
    }
    // [C6-d] Decode a ground-collapse report: which vertex, which body, is that
    // body on the ground skip list, and what is its actual signed distance.
    // The skip mask is a host pointer + host count baked into the recorded
    // launch, so a body that is legitimately allowed below the plane would be
    // reported as collapsed if the recording captured a count of zero.
    if(std::getenv("STIFF_FRAME_GRAPH_DIAG")
       && (m_last_frame_status.invalid_bits
           & frame_fsm::INV_START_INTERSECTING)
       && m_last_frame_status.err_primitive < 0)
    {
        const int vertex = -m_last_frame_status.err_primitive - 1;
        double3   position = make_double3(0.0, 0.0, 0.0);
        double3   normal   = make_double3(0.0, 0.0, 0.0);
        double    offset   = 0.0;
        int       body     = -1;
        int       skip     = -1;
        if(vertex >= 0 && vertex < static_cast<int>(vertexNum))
        {
            CUDA_SAFE_CALL(cudaMemcpy(&position, _vertexes + vertex,
                                      sizeof(double3), cudaMemcpyDeviceToHost));
            if(_point_body_id)
                CUDA_SAFE_CALL(cudaMemcpy(&body, _point_body_id + vertex,
                                          sizeof(int), cudaMemcpyDeviceToHost));
        }
        if(_ground_skip_body && body >= 0 && body < _ground_body_count)
            CUDA_SAFE_CALL(cudaMemcpy(&skip, _ground_skip_body + body,
                                      sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&normal, _groundNormal, sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&offset, _groundOffset, sizeof(double),
                                  cudaMemcpyDeviceToHost));
        const double distance = normal.x * position.x + normal.y * position.y
                                + normal.z * position.z - offset;
        fprintf(stderr,
                "[collapse] vertex=%d/%d body=%d skip=%d body_count=%d "
                "pos=(%.6e,%.6e,%.6e) dist=%.6e vertexNum=%d\n",
                vertex, m_last_frame_status.err_primitive, body, skip,
                _ground_body_count, position.x, position.y, position.z,
                distance, (int)vertexNum);
    }
    if(std::getenv("STIFF_FRAME_GRAPH_DIAG") && m_ccd_alpha_slots)
    {
        // [C6-c] Which alpha lane collapsed? 0=ground 1=narrow-self
        // 2=initial-combine 3=cfl 4=refined-swept. Read on the host after the
        // graph has run; the slots persist in device memory.
        double slots[9] = {0};
        CUDA_SAFE_CALL(cudaMemcpy(slots, m_ccd_alpha_slots, sizeof(slots),
                                  cudaMemcpyDeviceToHost));
        fprintf(stderr,
                "[alpha-lanes] ground=%.6e narrow=%.6e initial=%.6e cfl=%.6e "
                "refined=%.6e invalid=%.0f\n",
                slots[0], slots[1], slots[2], slots[3], slots[4], slots[7]);
    }
    if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
        fprintf(stderr,
                "[graph-frame] result=%d err=%d inv=0x%x newton=%d pcg=%d ls=%d "
                "alpha=%.6e "
                "cfl=%.6e kappa=%.6e move=%.6e dcd=%d ccd=%d "
                "class=[%d/%d,%d/%d,%d/%d,%d/%d]\n",
                m_last_frame_status.result,
                m_last_frame_status.error_code,
                (unsigned)m_last_frame_status.invalid_bits,
                m_last_frame_status.newton_iters,
                m_last_frame_status.pcg_iters,
                m_last_frame_status.ls_trials,
                m_last_frame_status.final_alpha,
                m_last_frame_status.cfl_alpha,
                m_last_frame_status.kappa,
                m_last_frame_status.max_movement,
                m_last_frame_status.hw_dcd_pairs,
                m_last_frame_status.hw_ccd_pairs,
                m_last_frame_status.contact_class_count[0],
                gipc_global_triplet.m_contact_class_tier[0],
                m_last_frame_status.contact_class_count[1],
                gipc_global_triplet.m_contact_class_tier[1],
                m_last_frame_status.contact_class_count[2],
                gipc_global_triplet.m_contact_class_tier[2],
                m_last_frame_status.contact_class_count[3],
                gipc_global_triplet.m_contact_class_tier[3]);
    note_frame_graph_coverage(m_last_frame_status);   // [C6]
    // [C6-o] growth-streak escalation (see GIPC.cuh): consecutive re-crossings
    // of the SAME axis within 8 frames earn 2x per streak step, capped at 4x.
    auto growth_escalation = [&](int axis) -> int {
        int64_t& last   = m_axis_grow_last[axis];
        int&     streak = m_axis_grow_streak[axis];
        if(last != 0 && m_total_frames - last <= 8)
            streak = std::min(streak + 1, 2);
        else
            streak = 0;
        last = m_total_frames;
        return streak;
    };
    if((m_last_frame_status.invalid_bits
        & frame_fsm::OVF_UNIQUE_BLOCKS)
       && m_last_frame_status.required_unique_blocks > 0)
    {
        // [C6-h] Growing the unique tier without bumping the buffer generation
        // was dead code for the recorded path: the retry replayed the SAME
        // executable with the SAME baked unique extents, so beaker overflowed
        // unique blocks on attempt 2, grew, and overflowed identically on
        // attempt 3 (inv=0x80040) -- a wasted retry that then cascaded. A tier
        // that shapes recorded launches must force a re-record, exactly like
        // the pair/CCD tiers below.
        const int tier = gipc::assembly_capacity_tier(
            static_cast<int>(std::min<long long>(
                INT_MAX / 2,
                static_cast<long long>(
                    m_last_frame_status.required_unique_blocks)
                    << growth_escalation(10))));
        const int prev0 = gipc_global_triplet.m_abd_unique_tier[0];
        const int prev1 = gipc_global_triplet.m_abd_unique_tier[1];
        gipc_global_triplet.m_abd_unique_tier[0] = std::max(prev0, tier);
        gipc_global_triplet.m_abd_unique_tier[1] = std::max(prev1, tier);
        if(gipc_global_triplet.m_abd_unique_tier[0] > prev0
           || gipc_global_triplet.m_abd_unique_tier[1] > prev1)
        {
            m_graph_tier_grew = true;
            ++pcg_buffer_generation();   // force a re-record at the new tier
        }
    }
    // [C6] Pair-capacity overflow must GROW the trained extents, otherwise the
    // retry replays the same too-small tier and burns the whole retry budget.
    // The status packet carries what the frame actually needed.
    if(m_last_frame_status.invalid_bits
       & (frame_fsm::OVF_DCD_PAIRS | frame_fsm::OVF_CCD_PAIRS))
    {
        int needed_dcd = std::max(
            m_last_frame_status.required_dcd_pairs,
            m_last_frame_status.hw_dcd_pairs);
        int needed_ccd = std::max(
            m_last_frame_status.required_ccd_pairs,
            m_last_frame_status.hw_ccd_pairs);
        // The swept extent is derived from the DCD extent by the buffers'
        // DCD:CCD ratio, so express the CCD requirement in DCD terms.
        // Grow the swept extent on its own axis — inflating the DCD side to
        // cover a swept need multiplies the triplet envelope for nothing.
        // The in-graph emission is capacity-clamped, so on overflow the device
        // counter saturates AT the tier and `needed` never exceeds it — the
        // growth test below could never fire. When the overflow bit is set but
        // the reported need is not larger, we know it overflowed and not by how
        // much: grow geometrically.
        if((m_last_frame_status.invalid_bits & frame_fsm::OVF_CCD_PAIRS)
           && needed_ccd <= m_graph_train_ccd)
            needed_ccd = m_graph_train_ccd + 1;
        if(m_last_frame_status.invalid_bits & frame_fsm::OVF_CCD_PAIRS)
            needed_ccd = static_cast<int>(std::min<long long>(
                MAX_CCD_COLLITION_PAIRS_NUM,
                static_cast<long long>(needed_ccd)
                    << growth_escalation(5)));
        if((m_last_frame_status.invalid_bits & frame_fsm::OVF_DCD_PAIRS)
           && needed_dcd <= m_graph_train_pairs)
            needed_dcd = m_graph_train_pairs + 1;
        if(needed_ccd > m_graph_train_ccd)
        {
            const int grown_ccd = std::min(
                MAX_CCD_COLLITION_PAIRS_NUM,
                gipc::assembly_capacity_tier(
                    needed_ccd * GIPC::graph_train_headroom_num()));
            if(grown_ccd > m_graph_train_ccd)
            {
                m_graph_tier_grew = true;
                if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
                    fprintf(stderr,
                            "[graph-train] swept overflow: needed=%d -> "
                            "trained ccd %d -> %d\n",
                            needed_ccd, m_graph_train_ccd, grown_ccd);
                m_graph_train_ccd = grown_ccd;
                ++pcg_buffer_generation();
            }
        }
        // Only grow the DCD side when the DCD side actually overflowed.
        // Multiplying the CURRENT tier by the headroom grew it on every
        // swept-only overflow, doubling the triplet envelope for nothing.
        // [C6-d] Grow only the axes the device reported as crossed. The guard
        // encodes them as a bitmask in err_primitive over its own entry order:
        // 0 -> cp[0], 1 -> cp[2], 2 -> cp[3], 3 -> cp[4], 4 -> ground (which is
        // already at worst case and must never cross). Each crossed axis is
        // sized from the reported requirement, and unrelated axes are left
        // alone -- inflating them all in lockstep is what blew up the triplet
        // envelope.
        static const int kEntryToSlot[5] = {0, 2, 3, 4, -1};
        // fsm_record_error keeps only the FIRST error's primitive (atomicCAS on
        // error_code), so when another error beat the tier guard to it the mask
        // never reaches us. Falling back to "grow every axis" there is actively
        // harmful: the capacity-mirror assembly emits abd_abd triplets over the
        // TIERED pair extents, so inflating cp[2..4] multiplies the triplet
        // stream, overflows the abd_abd class, grows that tier, and the next
        // attempt emits more still -- 512k -> 2.77M -> 11.8M triplets across
        // three retries, ending in OOM. With no mask we grow only cp[0], the one
        // axis whose requirement is reported as an honest count.
        const int packed = m_last_frame_status.err_primitive;
        const int mask   = packed > 0 ? ((packed >> 15) & 0x1F) : 0x01;
        bool      grew_any = false;
        for(int entry = 0; entry < 5; ++entry)
        {
            if(!(mask & (1 << entry)))
                continue;
            const int slot = kEntryToSlot[entry];
            if(slot < 0)
            {
                fprintf(stderr,
                        "[graph-train][WARN] ground extent crossed its trained "
                        "value -- it is supposed to be at worst case "
                        "(surf_vertexNum) and cannot be grown further\n");
                continue;
            }
            int shift = packed > 0 ? ((packed >> (3 * entry)) & 7) : 0;
            if(shift < 1)
                shift = 1;
            shift += growth_escalation(entry);
            const long long reach =
                static_cast<long long>(m_graph_train_cp[slot]) << shift;
            const int want = static_cast<int>(std::min<long long>(
                MAX_COLLITION_PAIRS_NUM,
                gipc::assembly_capacity_tier(static_cast<int>(std::min<long long>(
                    MAX_COLLITION_PAIRS_NUM,
                    reach * GIPC::graph_train_headroom_num())))));
            if(want <= m_graph_train_cp[slot])
                continue;
            if(std::getenv("STIFF_FRAME_GRAPH_DIAG"))
                fprintf(stderr,
                        "[graph-train] pair overflow: needed dcd=%d -> "
                        "trained cp[%d] %d -> %d\n",
                        needed_dcd, slot, m_graph_train_cp[slot], want);
            m_graph_train_cp[slot] = want;
            grew_any               = true;
        }
        if(grew_any)
        {
            m_graph_tier_grew   = true;
            m_graph_train_pairs = m_graph_train_cp[0];
            ++pcg_buffer_generation();   // force a re-record at the new tier
        }
    }
    if(m_last_frame_status.invalid_bits & frame_fsm::OVF_TRIPLETS)
    {
        bool grew_contact_class = false;
        for(int s = 0; s < 4; ++s)
        {
            const int exact =
                m_last_frame_status.contact_class_count[s];
            if(exact <= gipc_global_triplet.m_contact_class_tier[s])
                continue;
            gipc_global_triplet.m_contact_class_tier[s] =
                gipc::assembly_capacity_tier(static_cast<int>(
                    std::min<long long>(INT_MAX / 2,
                                        static_cast<long long>(exact)
                                            << growth_escalation(6 + s))));
            grew_contact_class = true;
        }
        if(grew_contact_class)
        {
            m_graph_tier_grew = true;
            int stable_count = 0;
            for(int s = 0; s < 4; ++s)
                stable_count +=
                    gipc_global_triplet.m_contact_class_tier[s];
            const int sort_capacity = gipc::assembly_capacity_tier(
                std::max(stable_count,
                         m_last_frame_status.required_triplets));
            const int staging_base =
                std::max(sort_capacity, stable_count);
            const size_t need =
                static_cast<size_t>(staging_base)
                + static_cast<size_t>(stable_count);
            // The failed attempt has already been restored and its triplets
            // are dead. This is the legal frame-boundary discard/grow window.
            gipc_global_triplet.global_triplet_offset = 0;
            gipc_global_triplet.global_collision_triplet_offset = 0;
            gipc_global_triplet.open_discard_window();
            gipc_global_triplet.ensure_capacity_discard(need);
            gipc_global_triplet.resize_collision_hash_size(
                static_cast<size_t>(sort_capacity));
            gipc_global_triplet.global_external_max_capcity =
                std::max(gipc_global_triplet.global_external_max_capcity,
                         sort_capacity);
            ++pcg_buffer_generation();
        }
    }
    if(m_last_frame_status.result != frame_fsm::FRAME_OK)
        restore_host_attempt(*this, context.host_snapshot);
    else if(m_last_frame_status.path_flags
            & frame_fsm::PATH_FULL_CONDITIONAL_GRAPH)
    {
        // The device-owned Newton/PCG loops never touch host telemetry.
        // Commit their exact terminal counters once, at the sole frame
        // boundary, just as IPC_Solver would have done incrementally.
        m_total_frames =
            context.host_snapshot.total_frames + 1;
        m_total_newton_iters =
            context.host_snapshot.total_newton_iters
            + m_last_frame_status.newton_iters;
        m_total_pcg_iters =
            context.host_snapshot.total_pcg_iters
            + m_last_frame_status.pcg_iters;
        animation_fullRate = animation_subRate;
        // [C4-b] commit the device-advanced kappa: the terminal packet
        // carries FrameDeviceState::kappa after any in-graph doublings, and
        // the next frame's boundary initKappa starts from it exactly as the
        // host solver's would.
        if(!m_skip_all_collision && m_last_frame_status.kappa > 0.0)
            Kappa = m_last_frame_status.kappa;
        if(!m_skip_all_collision && context.full_exec)
        {
            // [C4-a] the one deliberate boundary read of the collision
            // frame: refresh the frozen host mirrors from the device pair
            // snapshot. The recorded launch extents are shaped by full
            // trained capacity, so any live count <= capacity replays
            // correctly — no executable invalidation is needed here (true
            // past-capacity overflow is adjudicated in-graph and grows the
            // tiers through the OVF retry path, which bumps the buffer
            // generation and forces a re-record).
            uint32_t live[6] = {0, 0, 0, 0, 0, 0};
            CUDA_SAFE_CALL(cudaMemcpy(live,
                                      m_pair_snap_cur,
                                      sizeof(live),
                                      cudaMemcpyDeviceToHost));
            std::memcpy(
                h_cpNum.refresh_dst(), live, 5 * sizeof(uint32_t));
            h_gpNum          = live[5];
            m_dcd_snap_count = live[0];
        }
    }
    else
        m_total_pcg_iters =
            context.host_snapshot.total_pcg_iters
            + m_last_frame_status.pcg_iters;
    if(m_global_linear_system)
        m_global_linear_system->set_frame_device_state(nullptr);
    gipc_global_triplet.m_frame_device_state = nullptr;
    gipc_global_triplet.m_abd_unique_test_tier = 0;
    gipc_global_triplet.m_abd_tier_txn_ok    = false;
    gipc_global_triplet.m_contact_partition_txn_ok = false;
    m_frame_graph_active     = false;
    m_frame_terminal_emitted = false;
    return m_last_frame_status.result;
}

frame_fsm::FrameDeviceState* GIPC::frame_graph_device_state() const
{
    auto* context =
        static_cast<FrameGraphContext*>(m_frame_graph_context);
    return m_frame_graph_active && context ? context->d_state : nullptr;
}

void GIPC::IPC_Solver_FrameGraph(device_TetraData& mesh)
{
    // Frame zero remains the allocation/lazy-workspace warm-up boundary. It
    // runs the release solver and publishes an explicit fallback packet; no
    // graph caches a pointer until those first-use allocations have settled.
    //
    // [C6-g] REJECTED: gating additionally on "a frame has observed contact"
    // (so the capacity tiers never train from a contact-free frame). It did not
    // fix what it targeted -- towel's crumple spread stayed at 0.955..1.031 --
    // cost coverage (98% -> 93%), and moved G18 off its calibrated noise
    // envelope by changing WHICH frames are graphed. The specific cliffs it was
    // aimed at are already handled: the ground axis is trained to worst case
    // (it has no OVF bit) and the contact-class tiers grow through OVF_TRIPLETS.
    if(m_total_frames == 0)
    {
        const int before = m_total_newton_iters;
        IPC_Solver(mesh);
        record_legacy_frame_status(
            true, true, m_total_newton_iters - before);
        return;
    }

    int max_retries = 3;
    if(const char* configured = std::getenv("STIFF_FRAME_MAX_RETRIES"))
        max_retries = std::clamp(std::atoi(configured), 0, 16);
    const char* force_rollback =
        std::getenv("STIFF_FRAME_FORCE_ROLLBACK");
    const bool diagnostic_rollback =
        force_rollback && force_rollback[0]
        && force_rollback[0] != '0';
    constexpr uint32_t capacity_bits =
        frame_fsm::OVF_DCD_PAIRS
        | frame_fsm::OVF_CCD_PAIRS
        | frame_fsm::OVF_TRIPLETS
        | frame_fsm::OVF_UNIQUE_BLOCKS
        | frame_fsm::OVF_MAS_CLUSTERS;
    const int64_t physical_frame_id = m_total_frames;
    uint32_t retry_invalid_bits = 0;
    // [C6-i] After a capacity overflow the frame is NOT retried in-graph.
    // Re-recording at a grown tier changes grid shapes and reduction order, so
    // whether a retry fired -- which depends on racy atomicAdd counts sitting
    // near a tier boundary -- injected a ~1e-6 run-to-run perturbation that
    // chaotic contact amplified (towel: crumple 0.77..1.11 in-graph vs a
    // deterministic 0.905 on the host). Instead the frame is FINISHED on the
    // release solver from its bit-exactly restored start state -- the exact
    // code graph-off runs, deterministic where the host is -- and the tier
    // growth adjudicated at this frame boundary shapes the NEXT frame's
    // re-record. This is the iron law's own split: growth decisions live on
    // the host at frame boundaries, the graph only covers the frame interior.
    // STIFF_GRAPH_INGRAPH_RETRY=1 restores the old replay-in-graph behaviour.
    const bool ingraph_retry = knob_enabled("STIFF_GRAPH_INGRAPH_RETRY");
    bool       capacity_fallback = false;

    // [C6-e] Rollback audit against the FRAME's start state, captured here,
    // outside the retry loop. The earlier check compared the live state with
    // context.fem_vertexes, which the begin graph refreshes at every attempt --
    // "live == snapshot" is then trivially true and can never detect a failed
    // restore.
    const bool audit_rollback = std::getenv("STIFF_FRAME_GRAPH_DIAG") != nullptr;
    std::vector<double3> frame_start;
    std::vector<gipc::Vector12> abd_start;
    const int abd_n = m_abd_sim_data
                          ? (int)m_abd_sim_data->device.body_id_to_q.size()
                          : 0;
    if(audit_rollback && vertexNum)
    {
        frame_start.resize(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(frame_start.data(),
                                  mesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
    }
    if(audit_rollback && abd_n > 0)
    {
        abd_start.resize(abd_n);
        CUDA_SAFE_CALL(cudaMemcpy(abd_start.data(),
                                  m_abd_sim_data->device.body_id_to_q.data(),
                                  (size_t)abd_n * sizeof(gipc::Vector12),
                                  cudaMemcpyDeviceToHost));
    }

    for(int attempt = 0; attempt <= max_retries; ++attempt)
    {
        if(audit_rollback && attempt > 0 && !frame_start.empty())
        {
            std::vector<double3> live(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(live.data(),
                                      mesh.vertexes,
                                      vertexNum * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
            double worst = 0.0;
            int    worst_i = -1;
            for(size_t i = 0; i < live.size(); ++i)
            {
                const double d = std::max(
                    std::max(std::fabs(live[i].x - frame_start[i].x),
                             std::fabs(live[i].y - frame_start[i].y)),
                    std::fabs(live[i].z - frame_start[i].z));
                if(d > worst) { worst = d; worst_i = (int)i; }
            }
            double abd_worst = 0.0;
            int    abd_worst_i = -1;
            if(!abd_start.empty())
            {
                std::vector<gipc::Vector12> q(abd_n);
                CUDA_SAFE_CALL(cudaMemcpy(
                    q.data(),
                    m_abd_sim_data->device.body_id_to_q.data(),
                    (size_t)abd_n * sizeof(gipc::Vector12),
                    cudaMemcpyDeviceToHost));
                for(int b = 0; b < abd_n; ++b)
                    for(int c = 0; c < 12; ++c)
                    {
                        const double d =
                            std::fabs(q[b](c) - abd_start[b](c));
                        if(d > abd_worst)
                        {
                            abd_worst   = d;
                            abd_worst_i = b;
                        }
                    }
            }
            fprintf(stderr,
                    "[rollback-audit] attempt %d: max|live-frameStart|=%.6e "
                    "at vertex %d | max|q-qStart|=%.6e at body %d of %d\n",
                    attempt, worst, worst_i, abd_worst, abd_worst_i, abd_n);
        }
        try
        {
            const bool full_launched =
                !capacity_fallback
                && knob_enabled("STIFF_FRAME_FULL_GRAPH")
                && try_launch_full_graph(
                    *this,
                    mesh,
                    physical_frame_id,
                    attempt,
                    retry_invalid_bits);
            if(!full_launched)
            {
                // [C6-m] A fallback attempt runs with the capacity-layout
                // machinery fully OFF, so it is bit-for-bit the graph-off
                // frame the C6-i contract promises: exact partition, no tier
                // guard, no truncation. RAII so throws restore the mode.
                LayoutOverrideOff layout_off{attempt > 0};
                // [C6-l] A fallback attempt inherits pair arrays clobbered by
                // the aborted recording: a truncated, racy SUBSET emitted at
                // TRIAL positions. The release solver's own frame boundary
                // builds the lagged friction sets straight from those arrays
                // (buildFrictionSets reads h_cpNum[0]/_collisonPairs, no
                // fresh detection), which re-injected nondeterminism into an
                // otherwise bit-exact fallback -- measured fallback Newton 17
                // vs 19 across two runs with identical restored state.
                // Rebuild the pair set from the restored frame-entry geometry
                // first: same positions give the same SET, and the canonical
                // slot order makes the arrays bitwise reproducible.
                if(attempt > 0 && !m_skip_all_collision
                   && !getenv("STIFF_FALLBACK_NO_REBUILD"))
                {
                    buildBVH();
                    buildCP();
                }
                frame_graph_begin(
                    mesh,
                    physical_frame_id,
                    attempt,
                    retry_invalid_bits);
                IPC_Solver(mesh);
                frame_graph_enqueue_terminal(
                    mesh,
                    frame_fsm::FRAME_OK,
                    frame_fsm::ERR_NONE);
                CUDA_SAFE_CALL(
                    cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
        catch(const std::exception& error)
        {
            bool terminal_adjudicated = false;
            if(m_frame_graph_active && !m_frame_terminal_emitted)
            {
                frame_graph_enqueue_terminal(
                    mesh,
                    frame_fsm::FRAME_FATAL,
                    frame_fsm::ERR_SOLVER_EXCEPTION);
                CUDA_SAFE_CALL(
                    cudaStreamSynchronize(cudaStreamPerThread));
                frame_graph_finish_terminal();
                terminal_adjudicated = true;
            }
            if(!terminal_adjudicated)
                throw;
            throw std::runtime_error(
                status_error(m_last_frame_status, error.what()));
        }

        const int result = frame_graph_finish_terminal();
        if(result == frame_fsm::FRAME_OK)
            return;

        retry_invalid_bits |= m_last_frame_status.invalid_bits;
        // A too-small tier starves the solver (dropped pairs -> wrong energy ->
        // line-search budget exhaustion), and THAT error is what gets recorded,
        // not the capacity bit. Retry on any outcome that carries a capacity bit
        // as long as a tier actually grew — the grow flag is what makes this
        // terminate: no growth, no retry.
        const bool retryable =
            (result == frame_fsm::FRAME_RETRY_REQUIRED || m_graph_tier_grew)
            && (m_last_frame_status.invalid_bits & capacity_bits)
            && !diagnostic_rollback;
        if(retryable && attempt < max_retries)
        {
            if(!ingraph_retry)
                capacity_fallback = true;   // [C6-i] finish on the release solver
            continue;
        }

        if(retryable)
        {
            m_last_frame_status.error_code =
                frame_fsm::ERR_RETRY_EXHAUSTED;
            m_last_frame_status.retry_count = attempt;
            m_last_frame_status.retry_invalid_bits =
                retry_invalid_bits;
        }
        throw std::runtime_error(
            status_error(m_last_frame_status,
                         retryable
                             ? "capacity retry budget exhausted"
                             : "terminal rollback"));
    }
}
