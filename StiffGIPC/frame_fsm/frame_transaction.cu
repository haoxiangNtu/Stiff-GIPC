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

    // [C4-a] pair counts the collision executable was tiered against. The
    // boundary refresh compares live tiers to these and drops the executable
    // when a tier boundary is crossed, so masked kernels never under-launch.
    uint32_t captured_pair_counts[6] = {0, 0, 0, 0, 0, 0};
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
    if(context.abd_count != 0 && !allow_abd)
    {
        reason = "ABD bodies are not yet device-conditional";
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
        if(ipc.frictionRate != 0.0 || ipc.gd_frictionRate != 0.0)
        {
            reason =
                "friction sets are rebuilt only at synchronous frames; "
                "nonzero friction is not sync-equivalent until C4-c";
            return false;
        }
    }
    if(ipc.m_update_boundary || ipc.softNum != 0)
    {
        reason = "moving boundaries or host soft targets are active";
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
    const bool mode_overlay =
        mode.mode != ModeConfig::Merged
        || mode.bvh_envdet || mode.perenv_bvh
        || mode.decouple_thresh || mode.pergroup_kappa
        || mode.segmented_pcg || mode.perenv_alpha || mode.perenv_par
        || mode.ee_canon || mode.ee_detgate || mode.ccd_canon
        || mode.spmv_det || mode.perenv_telem
        || mode.perenv_mask || mode.perenv_mask_dev;
    if(mode_overlay)
    {
        reason = "isolated/strict/per-env solver controls are active";
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
        for(int i = 0; i < 5; ++i)
            context.captured_pair_counts[i] = ipc.h_cpNum[i];
        context.captured_pair_counts[5] = ipc.h_gpNum;
        context.captured_pair_counts_valid = !ipc.m_skip_all_collision;
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
                episode_advance_frame_counter<<<
                    1, 1, 0, cudaStreamPerThread>>>(
                    episode.d_control,
                    episode.d_frame_counter);
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

bool try_launch_full_graph(GIPC& ipc,
                           device_TetraData& mesh,
                           int64_t frame_id,
                           int attempt,
                           uint32_t retry_invalid_bits)
{
    ipc.prepare_frame_graph(mesh);
    FrameGraphContext& context = graph_context(ipc);
    std::string reason;
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

    if(!context.full_exec
       || context.full_generation != pcg_buffer_generation())
    {
        // [C4-a] all in-graph collision buffers must reach final capacity
        // before capture. Explicit tier math first, then a dry-run of the
        // assembly+solve host builders with capacity mirrors: every lazy
        // workspace (converter staging, radix-sort temp, preconditioner
        // levels) grows on its own real path here, outside capture, so the
        // recorded pass allocates nothing. Running after
        // arm_full_graph_attempt matters — the dry run must exercise the
        // same transactional partition branch the recording will take. Its
        // garbage triplets/gradients are overwritten by the graph's own
        // prologue at runtime, and host bookkeeping is rolled back by the
        // same snapshot machinery the recording itself relies on.
        if(!ipc.m_skip_all_collision)
        {
            ipc.train_collision_graph_capacities();
            HostAttemptSnapshot training_snapshot;
            snapshot_host_attempt(ipc, training_snapshot);
            const bool defer_before  = ipc.m_ls_defer_counts;
            const bool energy_before = ipc.m_energy_use_device_counts;
            // The PCG epilogue records into stats["newton"].back(); give
            // the dry run the per-iteration context solve_subIP would have
            // set up, and restore the frame's stats object afterwards.
            auto& frame_stats =
                gipc::Statistics::instance().at_current_frame();
            const gipc::Json stats_backup = frame_stats;
            try
            {
                frame_stats["newton"] = gipc::Json::array();
                frame_stats["newton"].push_back(gipc::Json::object());
                ipc.m_ls_defer_counts           = true;
                ipc.m_energy_use_device_counts = true;
                for(int slot = 0; slot < 5; ++slot)
                    ipc.h_cpNum.refresh_dst()[slot] =
                        static_cast<uint32_t>(
                            ipc.MAX_COLLITION_PAIRS_NUM);
                ipc.h_gpNum =
                    static_cast<uint32_t>(ipc.surf_vertexNum);
                ipc.computeGradientAndHessian(mesh);
                ipc.calculateMovingDirection(
                    mesh, 0, ipc.pcg_data.P_type);
            }
            catch(...)
            {
                frame_stats                     = stats_backup;
                ipc.m_ls_defer_counts           = defer_before;
                ipc.m_energy_use_device_counts = energy_before;
                restore_host_attempt(ipc, training_snapshot);
                disarm_full_graph_attempt(ipc);
                throw;
            }
            frame_stats                     = stats_backup;
            ipc.m_ls_defer_counts           = defer_before;
            ipc.m_energy_use_device_counts = energy_before;
            restore_host_attempt(ipc, training_snapshot);
            CUDA_SAFE_CALL(
                cudaStreamSynchronize(cudaStreamPerThread));
        }
        try
        {
            capture_full_graph(ipc, mesh, context);
            // Recording runs the host launch-builders once. None of their
            // bookkeeping is a completed physical frame.
            restore_host_attempt(ipc, context.host_snapshot);
            ipc.animation_fullRate = ipc.animation_subRate;
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
    if((m_last_frame_status.invalid_bits
        & frame_fsm::OVF_UNIQUE_BLOCKS)
       && m_last_frame_status.required_unique_blocks > 0)
    {
        // Boundary-only tier growth.  No graph body allocation or host count
        // refresh is needed on the retry: the terminal packet already carried
        // the exact required count.
        const int tier = gipc::assembly_capacity_tier(
            m_last_frame_status.required_unique_blocks);
        gipc_global_triplet.m_abd_unique_tier[0] = std::max(
            gipc_global_triplet.m_abd_unique_tier[0], tier);
        gipc_global_triplet.m_abd_unique_tier[1] = std::max(
            gipc_global_triplet.m_abd_unique_tier[1], tier);
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
                gipc::assembly_capacity_tier(exact);
            grew_contact_class = true;
        }
        if(grew_contact_class)
        {
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
        if(!m_skip_all_collision && context.captured_pair_counts_valid
           && context.full_exec)
        {
            // [C4-a] the one deliberate boundary read of the collision
            // frame: refresh the frozen host mirrors from the device pair
            // snapshot and drop the executable when any launch tier was
            // crossed — masked kernels can mask down but never launch up.
            uint32_t live[6] = {0, 0, 0, 0, 0, 0};
            CUDA_SAFE_CALL(cudaMemcpy(live,
                                      m_pair_snap_cur,
                                      sizeof(live),
                                      cudaMemcpyDeviceToHost));
            std::memcpy(
                h_cpNum.refresh_dst(), live, 5 * sizeof(uint32_t));
            h_gpNum          = live[5];
            m_dcd_snap_count = live[0];
            bool tier_crossed = false;
            for(int i = 0; i < 6 && !tier_crossed; ++i)
            {
                const int captured = static_cast<int>(
                    context.captured_pair_counts[i]);
                const int now = static_cast<int>(live[i]);
                if(gipc::assembly_capacity_tier(now)
                   != gipc::assembly_capacity_tier(captured))
                    tier_crossed = true;
            }
            if(tier_crossed)
            {
                CUDA_SAFE_CALL(
                    cudaGraphExecDestroy(context.full_exec));
                context.full_exec = nullptr;
                context.captured_pair_counts_valid = false;
            }
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

    for(int attempt = 0; attempt <= max_retries; ++attempt)
    {
        try
        {
            const bool full_launched =
                knob_enabled("STIFF_FRAME_FULL_GRAPH")
                && try_launch_full_graph(
                    *this,
                    mesh,
                    physical_frame_id,
                    attempt,
                    retry_invalid_bits);
            if(!full_launched)
            {
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
        const bool retryable =
            result == frame_fsm::FRAME_RETRY_REQUIRED
            && (m_last_frame_status.invalid_bits & capacity_bits)
            && !diagnostic_rollback;
        if(retryable && attempt < max_retries)
            continue;

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
