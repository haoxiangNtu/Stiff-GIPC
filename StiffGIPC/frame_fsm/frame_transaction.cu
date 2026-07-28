#include "GIPC.cuh"

#include "abd_system/abd_sim_data.h"
#include "abd_system/abd_system.h"
#include "cuda_tools/cuda_tools.h"
#include "linear_system/linear_system/global_linear_system.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
struct alignas(16) FrameBeginInput
{
    int64_t  frame_id   = 0;
    int32_t  attempt    = 0;
    uint32_t path_flags = 0;
    double   kappa      = 0.0;
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
};

struct FrameGraphContext
{
    cudaGraphExec_t root_exec     = nullptr;
    cudaGraphExec_t terminal_exec = nullptr;

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
    int newton_begin   = 0;

    HostAttemptSnapshot host_snapshot;
};

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

void audit_graph(cudaGraph_t graph, int& node_count, int& d2h_count)
{
    size_t count = 0;
    CUDA_SAFE_CALL(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    if(count)
        CUDA_SAFE_CALL(cudaGraphGetNodes(graph, nodes.data(), &count));
    node_count = static_cast<int>(count);
    d2h_count  = 0;
    for(cudaGraphNode_t node : nodes)
    {
        cudaGraphNodeType type{};
        CUDA_SAFE_CALL(cudaGraphNodeGetType(node, &type));
        if(type != cudaGraphNodeTypeMemcpy)
            continue;
        cudaMemcpy3DParms params{};
        CUDA_SAFE_CALL(cudaGraphMemcpyNodeGetParams(node, &params));
        if(params.kind == cudaMemcpyDeviceToHost)
            ++d2h_count;
    }
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
    state->hw_dcd_pairs     = input->hw_dcd_pairs;
    state->hw_ccd_pairs     = input->hw_ccd_pairs;
    state->hw_triplets      = input->hw_triplets;
    state->hw_unique_blocks = input->hw_unique_blocks;
    state->required_dcd_pairs = input->required_dcd_pairs;
    state->required_ccd_pairs = input->required_ccd_pairs;
    state->required_triplets = input->required_triplets;
    state->required_unique_blocks = input->required_unique_blocks;
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
    state->phase  = state->result == frame_fsm::FRAME_OK
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
    out.graph_launches  = 2;
    out.host_boundaries = 1;
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
    *status = out;
}

void destroy_context(FrameGraphContext* context)
{
    if(!context)
        return;
    if(context->root_exec)
        cudaGraphExecDestroy(context->root_exec);
    if(context->terminal_exec)
        cudaGraphExecDestroy(context->terminal_exec);
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

FrameGraphContext& graph_context(GIPC& ipc)
{
    auto* context =
        static_cast<FrameGraphContext*>(ipc.m_frame_graph_context);
    if(!context)
        throw std::logic_error("frame graph transaction is not prepared");
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

void GIPC::destroy_frame_graph()
{
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
                             int attempt)
{
    prepare_frame_graph(mesh);
    FrameGraphContext& context = graph_context(*this);
    snapshot_host_attempt(*this, context.host_snapshot);
    context.newton_begin = m_total_newton_iters;

    *context.h_begin = FrameBeginInput{};
    context.h_begin->frame_id = frame_id;
    context.h_begin->attempt  = attempt;
    context.h_begin->path_flags =
        frame_fsm::PATH_GRAPH_REQUESTED
        | frame_fsm::PATH_GRAPH_ACTIVE
        | frame_fsm::PATH_HOST_PHASE_BRIDGE
        | frame_fsm::PATH_PCG_DEVICE_CONTINUATION;
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
    if(m_last_frame_status.result != frame_fsm::FRAME_OK)
        restore_host_attempt(*this, context.host_snapshot);
    else
        m_total_pcg_iters =
            context.host_snapshot.total_pcg_iters
            + m_last_frame_status.pcg_iters;
    if(m_global_linear_system)
        m_global_linear_system->set_frame_device_state(nullptr);
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

    frame_graph_begin(mesh, m_total_frames, 0);
    try
    {
        IPC_Solver(mesh);
        frame_graph_enqueue_terminal(
            mesh, frame_fsm::FRAME_OK, frame_fsm::ERR_NONE);
        CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
        const int result = frame_graph_finish_terminal();
        if(result != frame_fsm::FRAME_OK)
            throw std::runtime_error(
                status_error(m_last_frame_status, "terminal rollback"));
    }
    catch(const std::exception& error)
    {
        if(m_frame_graph_active && !m_frame_terminal_emitted)
        {
            frame_graph_enqueue_terminal(mesh,
                                         frame_fsm::FRAME_FATAL,
                                         frame_fsm::ERR_SOLVER_EXCEPTION);
            CUDA_SAFE_CALL(
                cudaStreamSynchronize(cudaStreamPerThread));
            frame_graph_finish_terminal();
        }
        throw std::runtime_error(
            status_error(m_last_frame_status, error.what()));
    }
}
