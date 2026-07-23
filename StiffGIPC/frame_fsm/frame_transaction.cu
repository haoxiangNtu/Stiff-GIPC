#include "GIPC.cuh"

#include "abd_system/abd_sim_data.h"
#include "abd_system/abd_system.h"
#include "cuda_tools/cuda_tools.h"
#include "linear_system/linear_system/global_linear_system.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

extern int    totalNT;
extern double totalTime;
extern int    total_Frames;
extern double maxCOllisionPairNum;
extern double totalCollisionPairs;
extern double total_Cg_count;
extern double timemakePd;
extern int    g_dec_frame;

namespace
{
struct alignas(16) FrameBeginInput
{
    int64_t  frame_id  = 0;
    int32_t  attempt   = 0;
    uint32_t path_flags = 0;
    double   kappa     = 0.0;
    double   animation_full_rate = 0.0;
};

struct alignas(16) FrameTerminalInput
{
    int32_t  result       = frame_fsm::FRAME_OK;
    int32_t  error_code   = frame_fsm::ERR_NONE;
    uint32_t invalid_bits = 0;
    int32_t  launch_status = 0;
    int32_t  err_env       = -1;
    int32_t  err_primitive = -1;
    int32_t  substeps      = 0;
    int32_t  newton_iters  = 0;
    int32_t  ls_trials     = 0;
    int32_t  host_boundaries = 0;
    int32_t  hw_dcd_pairs  = 0;
    int32_t  hw_ccd_pairs  = 0;
    int32_t  hw_triplets   = 0;
    int32_t  hw_unique_blocks = 0;
    int32_t  hw_mas_clusters = 0;
    int32_t  required_dcd_pairs = 0;
    int32_t  required_ccd_pairs = 0;
    int32_t  required_triplets = 0;
    int32_t  required_unique_blocks = 0;
    int32_t  required_mas_clusters = 0;
    int32_t  graph_launches = 2;
    int32_t  root_graph_nodes = 0;
    int32_t  root_d2h_nodes = 0;
    int32_t  terminal_graph_nodes = 0;
    int32_t  terminal_d2h_nodes = 0;
    int32_t  retry_count = 0;
    uint32_t retry_invalid_bits = 0;
    int32_t  inject_nan_vertex = -1;
    uint32_t path_flags_or = 0;
    int32_t  _pad_path = 0;
    double   final_alpha   = 1.0;
    double   final_energy  = 0.0;
    double   max_movement  = 0.0;
    double   cfl_alpha     = 1.0;
    double   kappa         = 0.0;
};

struct alignas(16) FrameAuxState
{
    double animation_full_rate = 0.0;
    double drive_ratio         = 1.0;
    int32_t substep            = 0;
    int32_t _pad               = 0;
};

struct HostAttemptSnapshot
{
    int    total_nt       = 0;
    double total_time     = 0.0;
    double max_pairs      = 0.0;
    double total_pairs    = 0.0;
    double total_pcg      = 0.0;
    double make_pd        = 0.0;
    int    dec_frame      = -1;
    double animation      = 0.0;
    uint64_t tolerance_accepts = 0;
    uint32_t cp_count[5]  = {0, 0, 0, 0, 0};
    uint32_t gp_count     = 0;
    uint32_t ccd_count    = 0;
    uint32_t dcd_snapshot_count = 0;
    uint32_t cp_last[5]   = {0, 0, 0, 0, 0};
    uint32_t gp_last      = 0;
    std::vector<double> kappa_group;
    std::vector<int> env_frozen;
    std::vector<int> env_status;
};

struct FrameGraphContext
{
    cudaGraphExec_t root_exec     = nullptr;
    cudaGraphExec_t terminal_exec = nullptr;
    cudaGraphExec_t pcg_continuation_exec = nullptr;

    frame_fsm::FrameDeviceState* d_state  = nullptr;
    frame_fsm::FrameStatus*      d_status = nullptr;
    FrameBeginInput*             d_begin  = nullptr;
    FrameTerminalInput*          d_terminal = nullptr;
    FrameAuxState*               d_aux      = nullptr;
    FrameAuxState*               d_aux_snapshot = nullptr;
    double*                      d_kappa = nullptr;
    double*                      d_kappa_snapshot = nullptr;

    FrameBeginInput*    h_begin    = nullptr;
    FrameTerminalInput* h_terminal = nullptr;
    frame_fsm::FrameStatus* h_status = nullptr;

    double3* fem_vertexes   = nullptr;
    double3* fem_o_vertexes = nullptr;
    double3* fem_velocities = nullptr;
    double3* fem_x_tilta    = nullptr;

    gipc::Vector12* abd_q       = nullptr;
    gipc::Vector12* abd_q_prev  = nullptr;
    gipc::Vector12* abd_q_v     = nullptr;
    gipc::Vector12* abd_q_tilde = nullptr;

    double* kappa_group = nullptr;
    gipc::RevoluteDrivingGPUData* revolute = nullptr;
    gipc::PrismaticDrivingGPUData* prismatic = nullptr;

    int4*     collision_pairs = nullptr;
    int4*     dcd_snapshot    = nullptr;
    uint32_t* ground_pairs    = nullptr;

    size_t vertex_count   = 0;
    size_t abd_count      = 0;
    size_t group_count    = 0;
    size_t revolute_count = 0;
    size_t prismatic_count = 0;
    size_t collision_capacity = 0;
    size_t dcd_snapshot_capacity = 0;
    size_t ground_capacity = 0;

    int root_nodes = 0;
    int root_d2h   = 0;
    int terminal_nodes = 0;
    int terminal_d2h   = 0;
    int attempt = 0;

    int pending_result = frame_fsm::FRAME_OK;
    int pending_error  = frame_fsm::ERR_NONE;
    uint32_t pending_bits = 0;
    int pending_err_env = -1;
    int pending_err_primitive = -1;
    int required_dcd = 0;
    int required_ccd = 0;
    int required_triplets = 0;
    int required_unique = 0;
    int required_mas = 0;

    int substeps = 0;
    int newton_iters = 0;
    int ls_trials = 0;
    int host_boundaries = 0;
    int hw_dcd = 0;
    int hw_ccd = 0;
    int hw_triplets = 0;
    int hw_unique = 0;
    int hw_mas = 0;
    uint32_t observed_invalid_bits = 0;
    double alpha = 1.0;
    double cfl_alpha = 1.0;
    double energy = 0.0;
    double max_movement = 0.0;

    HostAttemptSnapshot host_snapshot;
};

template <typename T>
void device_alloc(T*& ptr, size_t count)
{
    if(count == 0) return;
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
}

template <typename T>
void device_free(T*& ptr)
{
    if(ptr) cudaFree(ptr);
    ptr = nullptr;
}

void destroy_context(FrameGraphContext* ctx)
{
    if(!ctx) return;
    if(ctx->root_exec) cudaGraphExecDestroy(ctx->root_exec);
    if(ctx->terminal_exec) cudaGraphExecDestroy(ctx->terminal_exec);
    if(ctx->pcg_continuation_exec)
        cudaGraphExecDestroy(ctx->pcg_continuation_exec);
    device_free(ctx->d_state);
    device_free(ctx->d_status);
    device_free(ctx->d_begin);
    device_free(ctx->d_terminal);
    device_free(ctx->d_aux);
    device_free(ctx->d_aux_snapshot);
    device_free(ctx->d_kappa);
    device_free(ctx->d_kappa_snapshot);
    device_free(ctx->fem_vertexes);
    device_free(ctx->fem_o_vertexes);
    device_free(ctx->fem_velocities);
    device_free(ctx->fem_x_tilta);
    device_free(ctx->abd_q);
    device_free(ctx->abd_q_prev);
    device_free(ctx->abd_q_v);
    device_free(ctx->abd_q_tilde);
    device_free(ctx->kappa_group);
    device_free(ctx->revolute);
    device_free(ctx->prismatic);
    device_free(ctx->collision_pairs);
    device_free(ctx->dcd_snapshot);
    device_free(ctx->ground_pairs);
    if(ctx->h_begin) cudaFreeHost(ctx->h_begin);
    if(ctx->h_terminal) cudaFreeHost(ctx->h_terminal);
    if(ctx->h_status) cudaFreeHost(ctx->h_status);
    delete ctx;
}

void audit_graph(cudaGraph_t graph, int& nodes, int& d2h)
{
    size_t n = 0;
    CUDA_SAFE_CALL(cudaGraphGetNodes(graph, nullptr, &n));
    std::vector<cudaGraphNode_t> list(n);
    if(n) CUDA_SAFE_CALL(cudaGraphGetNodes(graph, list.data(), &n));
    nodes = static_cast<int>(n);
    d2h  = 0;
    for(cudaGraphNode_t node : list)
    {
        cudaGraphNodeType type{};
        CUDA_SAFE_CALL(cudaGraphNodeGetType(node, &type));
        if(type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms p{};
        CUDA_SAFE_CALL(cudaGraphMemcpyNodeGetParams(node, &p));
        if(p.kind == cudaMemcpyDeviceToHost) ++d2h;
    }
}

__global__ void frame_begin_init(frame_fsm::FrameDeviceState* state,
                                 const FrameBeginInput* input,
                                 double* kappa,
                                 FrameAuxState* aux)
{
    if(blockIdx.x || threadIdx.x) return;
    *state = frame_fsm::FrameDeviceState{};
    state->phase       = frame_fsm::PHASE_FRAME_BEGIN;
    state->result      = frame_fsm::FRAME_OK;
    state->err_env     = -1;
    state->err_primitive = -1;
    state->err_newton_iter = -1;
    state->err_ls_iter = -1;
    state->alpha       = 1.0;
    state->cfl_alpha   = 1.0;
    state->frame_id    = input->frame_id;
    state->attempt     = input->attempt;
    state->path_flags  = input->path_flags;
    state->kappa       = input->kappa;
    *kappa             = input->kappa;
    aux->animation_full_rate = input->animation_full_rate;
    aux->drive_ratio = 1.0;
    aux->substep = 0;
}

__global__ void frame_post_pcg_continuation(
    frame_fsm::FrameDeviceState* state)
{
    if(blockIdx.x || threadIdx.x || state->result != frame_fsm::FRAME_OK) return;
    state->phase = frame_fsm::PHASE_NEWTON_DECIDE;
}

__global__ void frame_terminal_apply(frame_fsm::FrameDeviceState* state,
                                     const FrameTerminalInput* input,
                                     double* kappa,
                                     FrameAuxState* aux)
{
    if(blockIdx.x || threadIdx.x) return;
    state->newton_iter     = input->newton_iters;
    state->ls_trial        = input->ls_trials;
    state->substep         = input->substeps;
    state->host_boundaries = input->host_boundaries;
    state->path_flags |= input->path_flags_or;
    state->hw_dcd_pairs    = max(state->hw_dcd_pairs, input->hw_dcd_pairs);
    state->hw_ccd_pairs    = max(state->hw_ccd_pairs, input->hw_ccd_pairs);
    state->hw_triplets     = max(state->hw_triplets, input->hw_triplets);
    state->hw_unique_blocks = max(state->hw_unique_blocks, input->hw_unique_blocks);
    state->hw_mas_clusters = max(state->hw_mas_clusters, input->hw_mas_clusters);
    state->required_dcd_pairs = input->required_dcd_pairs;
    state->required_ccd_pairs = input->required_ccd_pairs;
    state->required_triplets = input->required_triplets;
    state->required_unique_blocks = input->required_unique_blocks;
    state->required_mas_clusters = input->required_mas_clusters;
    state->alpha        = input->final_alpha;
    state->cfl_alpha    = input->cfl_alpha;
    state->energy_trial = input->final_energy;
    state->max_movement = input->max_movement;
    state->kappa        = input->kappa;
    *kappa              = input->kappa;
    aux->substep        = input->substeps;

    if(input->error_code != frame_fsm::ERR_NONE)
    {
        if(atomicCAS(&state->error_code, frame_fsm::ERR_NONE,
                     input->error_code) == frame_fsm::ERR_NONE)
        {
            state->err_env          = input->err_env;
            state->err_primitive    = input->err_primitive;
            state->err_newton_iter  = input->newton_iters;
            state->err_ls_iter      = input->ls_trials;
        }
    }
    atomicOr(&state->invalid_bits, input->invalid_bits);
    if(input->result != frame_fsm::FRAME_OK)
        state->result = input->result;
    state->phase = state->result == frame_fsm::FRAME_OK
                       ? frame_fsm::PHASE_COMMIT
                       : frame_fsm::PHASE_ROLLBACK;
}

__global__ void frame_validate_finite(frame_fsm::FrameDeviceState* state,
                                      const double3* positions,
                                      const double3* velocities,
                                      int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    const double3 p = positions[i];
    const double3 v = velocities[i];
    const bool bad = !isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)
                  || !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z);
    if(!bad) return;
    frame_fsm::fsm_record_error(state,
                                frame_fsm::ERR_NONFINITE_STATE,
                                frame_fsm::INV_NAN_STATE,
                                -1,
                                i);
    atomicCAS(&state->result,
              frame_fsm::FRAME_OK,
              frame_fsm::FRAME_FATAL);
    state->phase = frame_fsm::PHASE_ROLLBACK;
}

__global__ void frame_inject_nonfinite_for_test(double3* positions,
                                                int count,
                                                const FrameTerminalInput* input)
{
    if(blockIdx.x || threadIdx.x) return;
    const int vertex = input->inject_nan_vertex;
    if(vertex >= 0 && vertex < count)
        positions[vertex].x = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL));
}

__global__ void frame_restore_fem(frame_fsm::FrameDeviceState* state,
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count || state->result == frame_fsm::FRAME_OK) return;
    positions[i]     = snap_positions[i];
    old_positions[i] = snap_old_positions[i];
    velocities[i]    = snap_velocities[i];
    x_tilta[i]       = snap_x_tilta[i];
}

__global__ void frame_restore_abd(frame_fsm::FrameDeviceState* state,
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count || state->result == frame_fsm::FRAME_OK) return;
    q[i]       = snap_q[i];
    q_prev[i]  = snap_q_prev[i];
    q_v[i]     = snap_q_v[i];
    q_tilde[i] = snap_q_tilde[i];
}

template <typename T>
__global__ void frame_restore_array(frame_fsm::FrameDeviceState* state,
                                    T* live,
                                    const T* snapshot,
                                    int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < count && state->result != frame_fsm::FRAME_OK)
        live[i] = snapshot[i];
}

__global__ void frame_restore_scalars(frame_fsm::FrameDeviceState* state,
                                      double* kappa,
                                      const double* kappa_snapshot,
                                      FrameAuxState* aux,
                                      const FrameAuxState* aux_snapshot,
                                      uint32_t* cp_counts,
                                      uint32_t* close_cp,
                                      uint32_t* close_gp)
{
    if(blockIdx.x || threadIdx.x || state->result == frame_fsm::FRAME_OK) return;
    *kappa       = *kappa_snapshot;
    state->kappa = *kappa_snapshot;
    *aux         = *aux_snapshot;
    if(cp_counts)
        for(int i = 0; i < 6; ++i) cp_counts[i] = 0;
    if(close_cp) *close_cp = 0;
    if(close_gp) *close_gp = 0;
    state->path_flags |= frame_fsm::PATH_TERMINAL_ROLLBACK;
}

__global__ void frame_serialize_status(const frame_fsm::FrameDeviceState* state,
                                       const FrameTerminalInput* input,
                                       frame_fsm::FrameStatus* status)
{
    if(blockIdx.x || threadIdx.x) return;
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
    out.host_boundaries = state->host_boundaries;
    out.substeps        = state->substep;
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
    out.retry_count    = input->retry_count;
    out.retry_invalid_bits = input->retry_invalid_bits;
    *status = out;
}

void capture_root_graph(GIPC& ipc,
                        device_TetraData& mesh,
                        FrameGraphContext& ctx)
{
    cudaGraph_t graph = nullptr;
    CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,
                                          cudaStreamCaptureModeThreadLocal));
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.d_begin,
                                   ctx.h_begin,
                                   sizeof(FrameBeginInput),
                                   cudaMemcpyHostToDevice,
                                   cudaStreamPerThread));
    frame_begin_init<<<1, 1, 0, cudaStreamPerThread>>>(
        ctx.d_state, ctx.d_begin, ctx.d_kappa, ctx.d_aux);

    const size_t fem_bytes = ctx.vertex_count * sizeof(double3);
    if(fem_bytes)
    {
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.fem_vertexes, mesh.vertexes, fem_bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.fem_o_vertexes, mesh.o_vertexes, fem_bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.fem_velocities, mesh.velocities, fem_bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.fem_x_tilta, mesh.xTilta, fem_bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    if(ctx.abd_count)
    {
        auto& d = ipc.m_abd_sim_data->device;
        const size_t bytes = ctx.abd_count * sizeof(gipc::Vector12);
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.abd_q, d.body_id_to_q.data(), bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.abd_q_prev, d.body_id_to_q_prev.data(), bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.abd_q_v, d.body_id_to_q_v.data(), bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.abd_q_tilde, d.body_id_to_q_tilde.data(), bytes,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.d_kappa_snapshot,
                                   ctx.d_kappa,
                                   sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.d_aux_snapshot,
                                   ctx.d_aux,
                                   sizeof(FrameAuxState),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    if(ctx.group_count)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.kappa_group,
                                       ipc.m_kappa_group,
                                       ctx.group_count * sizeof(double),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    if(ctx.revolute_count)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.revolute,
                                       ipc.m_abd_system->m_revolute_driving_data.data(),
                                       ctx.revolute_count * sizeof(gipc::RevoluteDrivingGPUData),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    if(ctx.prismatic_count)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.prismatic,
                                       ipc.m_abd_system->m_prismatic_driving_data.data(),
                                       ctx.prismatic_count * sizeof(gipc::PrismaticDrivingGPUData),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    if(ctx.collision_capacity)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.collision_pairs,
                                       ipc._collisonPairs,
                                       ctx.collision_capacity * sizeof(int4),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    if(ctx.dcd_snapshot_capacity)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.dcd_snapshot,
                                       ipc._dcd_ccd_snapshot,
                                       ctx.dcd_snapshot_capacity * sizeof(int4),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    if(ctx.ground_capacity)
        CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.ground_pairs,
                                       ipc._environment_collisionPair,
                                       ctx.ground_capacity * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));

    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    audit_graph(graph, ctx.root_nodes, ctx.root_d2h);
    CUDA_SAFE_CALL(cudaGraphInstantiate(&ctx.root_exec, graph, nullptr, nullptr, 0));
    CUDA_SAFE_CALL(cudaGraphUpload(ctx.root_exec, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
}

void capture_pcg_continuation_graph(FrameGraphContext& ctx)
{
    cudaGraph_t graph = nullptr;
    CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,
                                          cudaStreamCaptureModeThreadLocal));
    frame_post_pcg_continuation<<<1, 1, 0, cudaStreamPerThread>>>(ctx.d_state);
    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    CUDA_SAFE_CALL(cudaGraphInstantiateWithFlags(
        &ctx.pcg_continuation_exec,
        graph,
        cudaGraphInstantiateFlagDeviceLaunch));
    CUDA_SAFE_CALL(cudaGraphUpload(ctx.pcg_continuation_exec,
                                  cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
}

void capture_terminal_graph(GIPC& ipc,
                            device_TetraData& mesh,
                            FrameGraphContext& ctx)
{
    cudaGraph_t graph = nullptr;
    CUDA_SAFE_CALL(cudaStreamBeginCapture(cudaStreamPerThread,
                                          cudaStreamCaptureModeThreadLocal));
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.d_terminal,
                                   ctx.h_terminal,
                                   sizeof(FrameTerminalInput),
                                   cudaMemcpyHostToDevice,
                                   cudaStreamPerThread));
    frame_terminal_apply<<<1, 1, 0, cudaStreamPerThread>>>(
        ctx.d_state, ctx.d_terminal, ctx.d_kappa, ctx.d_aux);
    if(ctx.vertex_count)
    {
        const int blocks = static_cast<int>((ctx.vertex_count + 255) / 256);
        frame_inject_nonfinite_for_test<<<1, 1, 0, cudaStreamPerThread>>>(
            mesh.vertexes,
            static_cast<int>(ctx.vertex_count),
            ctx.d_terminal);
        frame_validate_finite<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ctx.d_state,
            mesh.vertexes,
            mesh.velocities,
            static_cast<int>(ctx.vertex_count));
        frame_restore_fem<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ctx.d_state,
            mesh.vertexes,
            mesh.o_vertexes,
            mesh.velocities,
            mesh.xTilta,
            ctx.fem_vertexes,
            ctx.fem_o_vertexes,
            ctx.fem_velocities,
            ctx.fem_x_tilta,
            static_cast<int>(ctx.vertex_count));
    }
    if(ctx.abd_count)
    {
        auto& d = ipc.m_abd_sim_data->device;
        const int blocks = static_cast<int>((ctx.abd_count + 255) / 256);
        frame_restore_abd<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ctx.d_state,
            d.body_id_to_q.data(),
            d.body_id_to_q_prev.data(),
            d.body_id_to_q_v.data(),
            d.body_id_to_q_tilde.data(),
            ctx.abd_q,
            ctx.abd_q_prev,
            ctx.abd_q_v,
            ctx.abd_q_tilde,
            static_cast<int>(ctx.abd_count));
    }
    if(ctx.group_count)
        frame_restore_array<<<static_cast<int>((ctx.group_count + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state, ipc.m_kappa_group, ctx.kappa_group,
            static_cast<int>(ctx.group_count));
    if(ctx.revolute_count)
        frame_restore_array<<<static_cast<int>((ctx.revolute_count + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state,
            ipc.m_abd_system->m_revolute_driving_data.data(),
            ctx.revolute,
            static_cast<int>(ctx.revolute_count));
    if(ctx.prismatic_count)
        frame_restore_array<<<static_cast<int>((ctx.prismatic_count + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state,
            ipc.m_abd_system->m_prismatic_driving_data.data(),
            ctx.prismatic,
            static_cast<int>(ctx.prismatic_count));
    if(ctx.collision_capacity)
        frame_restore_array<<<static_cast<int>((ctx.collision_capacity + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state, ipc._collisonPairs, ctx.collision_pairs,
            static_cast<int>(ctx.collision_capacity));
    if(ctx.dcd_snapshot_capacity)
        frame_restore_array<<<static_cast<int>((ctx.dcd_snapshot_capacity + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state, ipc._dcd_ccd_snapshot, ctx.dcd_snapshot,
            static_cast<int>(ctx.dcd_snapshot_capacity));
    if(ctx.ground_capacity)
        frame_restore_array<<<static_cast<int>((ctx.ground_capacity + 255) / 256),
                              256, 0, cudaStreamPerThread>>>(
            ctx.d_state, ipc._environment_collisionPair, ctx.ground_pairs,
            static_cast<int>(ctx.ground_capacity));

    frame_restore_scalars<<<1, 1, 0, cudaStreamPerThread>>>(
        ctx.d_state,
        ctx.d_kappa,
        ctx.d_kappa_snapshot,
        ctx.d_aux,
        ctx.d_aux_snapshot,
        ipc._cpNum,
        ipc._close_cpNum,
        ipc._close_gpNum);
    frame_serialize_status<<<1, 1, 0, cudaStreamPerThread>>>(
        ctx.d_state, ctx.d_terminal, ctx.d_status);
    CUDA_SAFE_CALL(cudaMemcpyAsync(ctx.h_status,
                                   ctx.d_status,
                                   sizeof(frame_fsm::FrameStatus),
                                   cudaMemcpyDeviceToHost,
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    audit_graph(graph, ctx.terminal_nodes, ctx.terminal_d2h);
    CUDA_SAFE_CALL(cudaGraphInstantiate(&ctx.terminal_exec, graph, nullptr, nullptr, 0));
    CUDA_SAFE_CALL(cudaGraphUpload(ctx.terminal_exec, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaGraphDestroy(graph));
}

FrameGraphContext& context(GIPC& ipc)
{
    auto* ctx = static_cast<FrameGraphContext*>(ipc.m_frame_graph_context);
    if(!ctx) throw std::logic_error("frame graph has not been prepared");
    return *ctx;
}

void snapshot_host_attempt(GIPC& ipc, FrameGraphContext& ctx)
{
    HostAttemptSnapshot& h = ctx.host_snapshot;
    h.total_nt       = totalNT;
    h.total_time     = totalTime;
    h.max_pairs      = maxCOllisionPairNum;
    h.total_pairs    = totalCollisionPairs;
    h.total_pcg      = total_Cg_count;
    h.make_pd        = timemakePd;
    h.dec_frame      = g_dec_frame;
    h.animation      = ipc.animation_fullRate;
    h.tolerance_accepts = ipc.energy_tolerance_accept_count;
    std::memcpy(h.cp_count, ipc.h_cpNum, sizeof(h.cp_count));
    h.gp_count = ipc.h_gpNum;
    h.ccd_count = ipc.h_ccd_cpNum;
    h.dcd_snapshot_count = ipc.m_dcd_snap_count;
    std::memcpy(h.cp_last, ipc.h_cpNum_last, sizeof(h.cp_last));
    h.gp_last = ipc.h_gpNum_last;
    h.kappa_group = ipc.h_kappa_group;
    h.env_frozen  = ipc.m_env_frozen_iter;
    h.env_status  = ipc.m_env_status;
}

void restore_host_attempt(GIPC& ipc, FrameGraphContext& ctx)
{
    const HostAttemptSnapshot& h = ctx.host_snapshot;
    totalNT              = h.total_nt;
    totalTime            = h.total_time;
    maxCOllisionPairNum  = h.max_pairs;
    totalCollisionPairs  = h.total_pairs;
    total_Cg_count       = h.total_pcg;
    timemakePd           = h.make_pd;
    g_dec_frame          = h.dec_frame;
    ipc.animation_fullRate = h.animation;
    ipc.energy_tolerance_accept_count = h.tolerance_accepts;
    std::memcpy(ipc.h_cpNum, h.cp_count, sizeof(h.cp_count));
    ipc.h_gpNum = h.gp_count;
    ipc.h_ccd_cpNum = h.ccd_count;
    ipc.m_dcd_snap_count = h.dcd_snapshot_count;
    std::memcpy(ipc.h_cpNum_last, h.cp_last, sizeof(h.cp_last));
    ipc.h_gpNum_last = h.gp_last;
    ipc.h_kappa_group = h.kappa_group;
    ipc.m_env_frozen_iter = h.env_frozen;
    ipc.m_env_status = h.env_status;
}

void grow_requested_capacities(GIPC& ipc,
                               int required_dcd,
                               int required_ccd,
                               int required_triplets)
{
    if(required_dcd > ipc.MAX_COLLITION_PAIRS_NUM)
    {
        const int newcap = required_dcd + required_dcd / 2 + 1;
        CUDA_SAFE_CALL(cudaFree(ipc._collisonPairs));
        CUDA_SAFE_CALL(cudaFree(ipc._MatIndex));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ipc._collisonPairs),
                                  (static_cast<size_t>(newcap) + 1) * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ipc._MatIndex),
                                  (static_cast<size_t>(newcap) + 1) * sizeof(int)));
        ipc.MAX_COLLITION_PAIRS_NUM = newcap;
        ipc.bvh_f._collisionPair = ipc.bvh_e._collisionPair = ipc._collisonPairs;
        ipc.bvh_f._MatIndex = ipc.bvh_e._MatIndex = ipc._MatIndex;
    }
    const int mirror_required = std::max(required_ccd,
                                         ipc.MAX_COLLITION_PAIRS_NUM);
    if(mirror_required > ipc.MAX_CCD_COLLITION_PAIRS_NUM)
    {
        const int newcap = mirror_required + mirror_required / 2 + 1;
        CUDA_SAFE_CALL(cudaFree(ipc._ccd_collisonPairs));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ipc._ccd_collisonPairs),
                                  (static_cast<size_t>(newcap) + 1) * sizeof(int4)));
        ipc.MAX_CCD_COLLITION_PAIRS_NUM = newcap;
        ipc.bvh_f._ccd_collisionPair = ipc.bvh_e._ccd_collisionPair =
            ipc._ccd_collisonPairs;
    }
    set_emit_caps(ipc.MAX_COLLITION_PAIRS_NUM,
                  ipc.MAX_CCD_COLLITION_PAIRS_NUM);

    if(required_triplets > 0
       && ipc.gipc_global_triplet.triplet_capacity()
              < static_cast<size_t>(required_triplets))
    {
        const size_t cap = static_cast<size_t>(required_triplets)
                         + static_cast<size_t>(required_triplets) / 2 + 1;
        ipc.gipc_global_triplet.reserve_triplets(cap);
        if(ipc.gipc_global_triplet.global_external_max_capcity
           < required_triplets)
        {
            ipc.gipc_global_triplet.resize_collision_hash_size(cap);
            ipc.gipc_global_triplet.global_external_max_capcity =
                static_cast<int>(cap);
        }
    }
}

std::string status_error(const frame_fsm::FrameStatus& st,
                         const std::string& cause)
{
    std::ostringstream out;
    out << "[frame-fsm] frame " << st.frame_id
        << " attempt " << st.attempt
        << " failed: result=" << st.result
        << " phase=" << st.phase
        << " error=" << st.error_code
        << " invalid=0x" << std::hex << st.invalid_bits << std::dec
        << " env=" << st.err_env
        << " primitive=" << st.err_primitive
        << " newton=" << st.err_newton_iter
        << " ls=" << st.err_ls_iter;
    if(!cause.empty()) out << " cause=" << cause;
    return out.str();
}
}  // namespace

void GIPC::destroy_frame_graph()
{
    if(m_global_linear_system)
        m_global_linear_system->set_solver_device_continuation(nullptr);
    destroy_context(static_cast<FrameGraphContext*>(m_frame_graph_context));
    m_frame_graph_context = nullptr;
    m_frame_graph_active = false;
    m_frame_terminal_emitted = false;
}

void GIPC::prepare_frame_graph(device_TetraData& mesh)
{
    if(m_frame_graph_context) return;

    // Per-group kappa must exist before the first root snapshot.  Legacy still
    // allocates lazily; the opt-in graph path moves that allocation to its
    // finalize/warm-up boundary.
    if(std::getenv("STIFF_PERGROUP_KAPPA") && mesh.d_point_to_group
       && mesh.h_groups_present && !m_pergroup_kappa)
    {
        m_pergroup_kappa = true;
        m_d_p2g = mesh.d_point_to_group;
        m_active_group_count = mesh.h_group_count;
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_kappa_group),
                                  static_cast<size_t>(m_active_group_count)
                                      * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_d_close_grp),
                                  static_cast<size_t>(m_active_group_count)
                                      * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(m_kappa_group,
                                  0,
                                  static_cast<size_t>(m_active_group_count)
                                      * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(m_d_close_grp,
                                  0,
                                  static_cast<size_t>(m_active_group_count)
                                      * sizeof(int)));
        h_kappa_group.assign(m_active_group_count, 0.0);
    }

    // snapshotDcdCcdPairs() is legacy-lazy.  A graph may retain its pointer, so
    // reserve its maximum DCD extent before capture and preserve any warm-up
    // contents while replacing the allocation.
    const int required_dcd_snapshot_cap = MAX_COLLITION_PAIRS_NUM + 1;
    if(m_dcd_snap_cap < required_dcd_snapshot_cap)
    {
        int4* replacement = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&replacement),
                                  static_cast<size_t>(required_dcd_snapshot_cap)
                                      * sizeof(int4)));
        if(_dcd_ccd_snapshot && m_dcd_snap_count)
        {
            const size_t preserved = std::min(
                static_cast<size_t>(m_dcd_snap_count),
                static_cast<size_t>(m_dcd_snap_cap));
            CUDA_SAFE_CALL(cudaMemcpy(replacement,
                                      _dcd_ccd_snapshot,
                                      preserved * sizeof(int4),
                                      cudaMemcpyDeviceToDevice));
        }
        if(_dcd_ccd_snapshot)
            CUDA_SAFE_CALL(cudaFree(_dcd_ccd_snapshot));
        _dcd_ccd_snapshot = replacement;
        m_dcd_snap_cap = required_dcd_snapshot_cap;
    }

    auto* ctx = new FrameGraphContext{};
    m_frame_graph_context = ctx;
    try
    {
        ctx->vertex_count = vertexNum;
        ctx->abd_count = m_abd_sim_data
                             ? m_abd_sim_data->device.body_id_to_q.size()
                             : 0;
        ctx->group_count = m_pergroup_kappa
                               ? static_cast<size_t>(m_active_group_count)
                               : 0;
        ctx->revolute_count = m_abd_system
                                  ? m_abd_system->m_revolute_driving_data.size()
                                  : 0;
        ctx->prismatic_count = m_abd_system
                                   ? m_abd_system->m_prismatic_driving_data.size()
                                   : 0;
        ctx->collision_capacity = static_cast<size_t>(MAX_COLLITION_PAIRS_NUM) + 1;
        ctx->dcd_snapshot_capacity = static_cast<size_t>(m_dcd_snap_cap);
        ctx->ground_capacity = surf_vertexNum;

        device_alloc(ctx->d_state, 1);
        device_alloc(ctx->d_status, 1);
        device_alloc(ctx->d_begin, 1);
        device_alloc(ctx->d_terminal, 1);
        device_alloc(ctx->d_aux, 1);
        device_alloc(ctx->d_aux_snapshot, 1);
        device_alloc(ctx->d_kappa, 1);
        device_alloc(ctx->d_kappa_snapshot, 1);
        device_alloc(ctx->fem_vertexes, ctx->vertex_count);
        device_alloc(ctx->fem_o_vertexes, ctx->vertex_count);
        device_alloc(ctx->fem_velocities, ctx->vertex_count);
        device_alloc(ctx->fem_x_tilta, ctx->vertex_count);
        device_alloc(ctx->abd_q, ctx->abd_count);
        device_alloc(ctx->abd_q_prev, ctx->abd_count);
        device_alloc(ctx->abd_q_v, ctx->abd_count);
        device_alloc(ctx->abd_q_tilde, ctx->abd_count);
        device_alloc(ctx->kappa_group, ctx->group_count);
        device_alloc(ctx->revolute, ctx->revolute_count);
        device_alloc(ctx->prismatic, ctx->prismatic_count);
        device_alloc(ctx->collision_pairs, ctx->collision_capacity);
        device_alloc(ctx->dcd_snapshot, ctx->dcd_snapshot_capacity);
        device_alloc(ctx->ground_pairs, ctx->ground_capacity);
        CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(&ctx->h_begin),
                                     sizeof(FrameBeginInput),
                                     cudaHostAllocPortable));
        CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(&ctx->h_terminal),
                                     sizeof(FrameTerminalInput),
                                     cudaHostAllocPortable));
        CUDA_SAFE_CALL(cudaHostAlloc(reinterpret_cast<void**>(&ctx->h_status),
                                     sizeof(frame_fsm::FrameStatus),
                                     cudaHostAllocPortable));
        *ctx->h_begin = FrameBeginInput{};
        *ctx->h_terminal = FrameTerminalInput{};
        *ctx->h_status = frame_fsm::FrameStatus{};

        capture_pcg_continuation_graph(*ctx);
        if(m_global_linear_system)
            m_global_linear_system->set_solver_device_continuation(
                ctx->pcg_continuation_exec);
        capture_root_graph(*this, mesh, *ctx);
        capture_terminal_graph(*this, mesh, *ctx);
        if(ctx->root_d2h != 0 || ctx->terminal_d2h != 1)
            throw std::runtime_error(
                "frame graph audit failed: expected root D2H=0 and terminal D2H=1");
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
    FrameGraphContext& ctx = context(*this);
    snapshot_host_attempt(*this, ctx);
    ctx.attempt = attempt;
    if(attempt == 0)
    {
        m_frame_retry_bits = 0;
        m_frame_retry_count = 0;
        m_frame_retry_required_dcd = 0;
        m_frame_retry_required_ccd = 0;
        m_frame_retry_required_triplets = 0;
        m_frame_retry_required_unique = 0;
        m_frame_retry_required_mas = 0;
    }
    ctx.pending_result = frame_fsm::FRAME_OK;
    ctx.pending_error = frame_fsm::ERR_NONE;
    ctx.pending_bits = 0;
    ctx.pending_err_env = -1;
    ctx.pending_err_primitive = -1;
    ctx.required_dcd = ctx.required_ccd = ctx.required_triplets = 0;
    ctx.required_unique = ctx.required_mas = 0;
    ctx.substeps = ctx.newton_iters = ctx.ls_trials = 0;
    ctx.host_boundaries = 0;
    ctx.hw_dcd = static_cast<int>(h_cpNum[0]);
    ctx.hw_ccd = static_cast<int>(h_ccd_cpNum);
    ctx.hw_triplets = ctx.hw_unique = ctx.hw_mas = 0;
    ctx.observed_invalid_bits = 0;
    ctx.alpha = ctx.cfl_alpha = 1.0;
    ctx.energy = ctx.max_movement = 0.0;

    *ctx.h_begin = FrameBeginInput{};
    ctx.h_begin->frame_id = frame_id;
    ctx.h_begin->attempt  = attempt;
    ctx.h_begin->path_flags = frame_fsm::PATH_GRAPH_REQUESTED
                            | frame_fsm::PATH_GRAPH_ACTIVE
                            | frame_fsm::PATH_HOST_PHASE_BRIDGE
                            | frame_fsm::PATH_PCG_DEVICE_CONTINUATION;
    if(attempt > 0) ctx.h_begin->path_flags |= frame_fsm::PATH_RETRIED;
    ctx.h_begin->kappa = Kappa;
    ctx.h_begin->animation_full_rate = animation_fullRate;

    const cudaError_t status = cudaGraphLaunch(ctx.root_exec,
                                                cudaStreamPerThread);
    if(status != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error("frame root graph launch failed: "
                                 + std::to_string(static_cast<int>(status)));
    }
    m_frame_graph_active = true;
    m_frame_terminal_emitted = false;
}

void GIPC::frame_graph_request_retry(uint32_t bits,
                                     int required_dcd_pairs,
                                     int required_ccd_pairs,
                                     int required_triplets_in,
                                     int required_unique_blocks,
                                     int required_mas_clusters)
{
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    ctx.pending_result = frame_fsm::FRAME_RETRY_REQUIRED;
    ctx.pending_error  = frame_fsm::ERR_CAPACITY;
    ctx.pending_bits  |= bits;
    ctx.required_dcd = std::max(ctx.required_dcd, required_dcd_pairs);
    ctx.required_ccd = std::max(ctx.required_ccd, required_ccd_pairs);
    ctx.required_triplets = std::max(ctx.required_triplets, required_triplets_in);
    ctx.required_unique = std::max(ctx.required_unique, required_unique_blocks);
    ctx.required_mas = std::max(ctx.required_mas, required_mas_clusters);
    throw std::runtime_error("frame capacity retry requested");
}

void GIPC::frame_graph_note_pairs(int dcd_pairs, int ccd_pairs)
{
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    ctx.hw_dcd = std::max(ctx.hw_dcd, dcd_pairs);
    ctx.hw_ccd = std::max(ctx.hw_ccd, ccd_pairs);
}

void GIPC::frame_graph_guard_pairs(int dcd_pairs, int ccd_pairs)
{
    frame_graph_note_pairs(dcd_pairs, ccd_pairs);
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    int logical_dcd = MAX_COLLITION_PAIRS_NUM;
    int logical_ccd = MAX_CCD_COLLITION_PAIRS_NUM;
    if(ctx.attempt == 0)
    {
        if(const char* value = std::getenv("STIFF_FRAME_TEST_DCD_CAP"))
            logical_dcd = std::max(0, std::atoi(value));
        if(const char* value = std::getenv("STIFF_FRAME_TEST_CCD_CAP"))
            logical_ccd = std::max(0, std::atoi(value));
    }
    uint32_t bits = 0;
    if(dcd_pairs > logical_dcd) bits |= frame_fsm::OVF_DCD_PAIRS;
    if(ccd_pairs > logical_ccd) bits |= frame_fsm::OVF_CCD_PAIRS;
    if(bits)
        frame_graph_request_retry(bits,
                                  dcd_pairs > logical_dcd ? dcd_pairs : 0,
                                  ccd_pairs > logical_ccd ? ccd_pairs : 0);
}

void GIPC::frame_graph_note_assembly(int triplets,
                                     int unique_blocks,
                                     int mas_clusters)
{
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    ctx.hw_triplets = std::max(ctx.hw_triplets, triplets);
    ctx.hw_unique = std::max(ctx.hw_unique, unique_blocks);
    ctx.hw_mas = std::max(ctx.hw_mas, mas_clusters);
}

void GIPC::frame_graph_guard_triplets(int required_triplets_in)
{
    frame_graph_note_assembly(required_triplets_in);
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    size_t logical_cap = gipc_global_triplet.triplet_capacity();
    if(ctx.attempt == 0)
        if(const char* value = std::getenv("STIFF_FRAME_TEST_TRIPLET_CAP"))
            logical_cap = static_cast<size_t>(std::max(0, std::atoi(value)));
    if(static_cast<size_t>(required_triplets_in) > logical_cap)
        frame_graph_request_retry(frame_fsm::OVF_TRIPLETS,
                                  0,
                                  0,
                                  required_triplets_in);
}

void GIPC::frame_graph_note_newton(int iterations,
                                   double movement,
                                   double accepted_alpha,
                                   double accepted_cfl)
{
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    ++ctx.substeps;
    ctx.newton_iters += std::max(0, iterations);
    ctx.max_movement = movement;
    ctx.alpha = accepted_alpha;
    ctx.cfl_alpha = accepted_cfl;
}

void GIPC::frame_graph_note_line_search(int trials,
                                        uint32_t invalid_bits,
                                        double accepted_alpha,
                                        double /*energy0*/,
                                        double energy1)
{
    if(!m_frame_graph_active) return;
    FrameGraphContext& ctx = context(*this);
    ctx.ls_trials += std::max(0, trials);
    ctx.observed_invalid_bits |= invalid_bits;
    ctx.alpha = accepted_alpha;
    ctx.energy = energy1;
}

frame_fsm::FrameDeviceState* GIPC::frame_graph_device_state() const
{
    auto* ctx = static_cast<FrameGraphContext*>(m_frame_graph_context);
    return m_frame_graph_active && ctx ? ctx->d_state : nullptr;
}

int GIPC::frame_graph_read_phase()
{
    FrameGraphContext& ctx = context(*this);
    int phase = frame_fsm::PHASE_IDLE;
    CUDA_SAFE_CALL(cudaMemcpy(&phase,
                              &ctx.d_state->phase,
                              sizeof(phase),
                              cudaMemcpyDeviceToHost));
    ++ctx.host_boundaries;
    return phase;
}

void GIPC::frame_graph_enqueue_terminal(device_TetraData& /*mesh*/,
                                        int result,
                                        int error_code,
                                        uint32_t invalid_bits,
                                        int err_env,
                                        int err_primitive)
{
    FrameGraphContext& ctx = context(*this);
    if(m_frame_terminal_emitted)
        throw std::logic_error("frame terminal emitted twice");

    if(ctx.pending_result != frame_fsm::FRAME_OK)
    {
        result = ctx.pending_result;
        error_code = ctx.pending_error;
        invalid_bits |= ctx.pending_bits;
        err_env = ctx.pending_err_env;
        err_primitive = ctx.pending_err_primitive;
    }
    const char* force = std::getenv("STIFF_FRAME_FORCE_ROLLBACK");
    if(result == frame_fsm::FRAME_OK && ctx.attempt == 0 && force
       && force[0] && force[0] != '0')
    {
        result = frame_fsm::FRAME_RETRY_REQUIRED;
        error_code = frame_fsm::ERR_CAPACITY;
        invalid_bits |= frame_fsm::OVF_DCD_PAIRS;
        ctx.h_begin->path_flags |= frame_fsm::PATH_TEST_INJECTION;
        // The begin packet has already been consumed by the root graph; carry
        // the injection fact through the terminal packet as well.
        ctx.required_dcd = std::max(ctx.required_dcd,
                                    static_cast<int>(h_cpNum[0]));
    }

    *ctx.h_terminal = FrameTerminalInput{};
    FrameTerminalInput& in = *ctx.h_terminal;
    in.result = result;
    in.error_code = error_code;
    in.invalid_bits = invalid_bits | ctx.observed_invalid_bits;
    in.err_env = err_env;
    in.err_primitive = err_primitive;
    in.substeps = ctx.substeps;
    in.newton_iters = ctx.newton_iters;
    in.ls_trials = ctx.ls_trials;
    in.host_boundaries = ctx.host_boundaries;
    in.hw_dcd_pairs = ctx.hw_dcd;
    in.hw_ccd_pairs = ctx.hw_ccd;
    in.hw_triplets = ctx.hw_triplets;
    in.hw_unique_blocks = ctx.hw_unique;
    in.hw_mas_clusters = ctx.hw_mas;
    in.required_dcd_pairs = std::max(ctx.required_dcd,
                                     m_frame_retry_required_dcd);
    in.required_ccd_pairs = std::max(ctx.required_ccd,
                                     m_frame_retry_required_ccd);
    in.required_triplets = std::max(ctx.required_triplets,
                                    m_frame_retry_required_triplets);
    in.required_unique_blocks = std::max(ctx.required_unique,
                                         m_frame_retry_required_unique);
    in.required_mas_clusters = std::max(ctx.required_mas,
                                        m_frame_retry_required_mas);
    in.retry_count = m_frame_retry_count;
    in.retry_invalid_bits = m_frame_retry_bits;
    in.final_alpha = ctx.alpha;
    in.final_energy = ctx.energy;
    in.max_movement = ctx.max_movement;
    in.cfl_alpha = ctx.cfl_alpha;
    in.kappa = Kappa;
    if(result == frame_fsm::FRAME_OK && ctx.attempt == 0)
    {
        if(const char* value = std::getenv("STIFF_FRAME_TEST_NAN_VERTEX"))
        {
            in.inject_nan_vertex = std::atoi(value);
            in.path_flags_or |= frame_fsm::PATH_TEST_INJECTION;
        }
    }
    in.root_graph_nodes = ctx.root_nodes;
    in.root_d2h_nodes = ctx.root_d2h;
    in.terminal_graph_nodes = ctx.terminal_nodes;
    in.terminal_d2h_nodes = ctx.terminal_d2h;
    if(result == frame_fsm::FRAME_RETRY_REQUIRED && ctx.attempt == 0
       && force && force[0] && force[0] != '0')
        in.path_flags_or |= frame_fsm::PATH_TEST_INJECTION;

    if(m_global_linear_system)
        m_global_linear_system->enqueue_frame_solver_stats(ctx.d_state,
                                                            cudaStreamPerThread);
    const cudaError_t status = cudaGraphLaunch(ctx.terminal_exec,
                                                cudaStreamPerThread);
    if(status != cudaSuccess)
    {
        cudaGetLastError();
        throw std::runtime_error("frame terminal graph launch failed: "
                                 + std::to_string(static_cast<int>(status)));
    }
    m_frame_terminal_emitted = true;
}

int GIPC::frame_graph_finish_terminal()
{
    FrameGraphContext& ctx = context(*this);
    if(!m_frame_terminal_emitted)
        throw std::logic_error("frame terminal not emitted");
    m_last_frame_status = *ctx.h_status;
    Kappa = m_last_frame_status.kappa;
    if(m_last_frame_status.result == frame_fsm::FRAME_OK)
        total_Cg_count += m_last_frame_status.pcg_iters;
    else
    {
        restore_host_attempt(*this, ctx);
        if(m_last_frame_status.result == frame_fsm::FRAME_RETRY_REQUIRED)
        {
            ++m_frame_retry_count;
            m_frame_retry_bits |= m_last_frame_status.invalid_bits;
            m_frame_retry_required_dcd = std::max(
                m_frame_retry_required_dcd,
                m_last_frame_status.required_dcd_pairs);
            m_frame_retry_required_ccd = std::max(
                m_frame_retry_required_ccd,
                m_last_frame_status.required_ccd_pairs);
            m_frame_retry_required_triplets = std::max(
                m_frame_retry_required_triplets,
                m_last_frame_status.required_triplets);
            m_frame_retry_required_unique = std::max(
                m_frame_retry_required_unique,
                m_last_frame_status.required_unique_blocks);
            m_frame_retry_required_mas = std::max(
                m_frame_retry_required_mas,
                m_last_frame_status.required_mas_clusters);
        }
    }
    m_frame_graph_active = false;
    return m_last_frame_status.result;
}

void GIPC::record_legacy_frame_status(bool graph_requested,
                                      bool callback_fallback,
                                      int newton_iterations)
{
    frame_fsm::FrameStatus st{};
    st.result = frame_fsm::FRAME_OK;
    st.phase  = frame_fsm::PHASE_COMMIT;
    st.err_env = st.err_primitive = -1;
    st.err_newton_iter = st.err_ls_iter = -1;
    st.path_flags = graph_requested ? frame_fsm::PATH_GRAPH_REQUESTED : 0;
    if(callback_fallback) st.path_flags |= frame_fsm::PATH_LEGACY_FALLBACK;
    st.newton_iters = newton_iterations;
    st.hw_dcd_pairs = static_cast<int>(h_cpNum[0]);
    st.hw_ccd_pairs = static_cast<int>(h_ccd_cpNum);
    st.kappa = Kappa;
    st.frame_id = total_Frames > 0 ? total_Frames - 1 : 0;
    m_last_frame_status = st;
}

void GIPC::IPC_Solver_FrameGraph(device_TetraData& mesh)
{
    constexpr int kMaxAttempts = 8;
    for(int attempt = 0; attempt < kMaxAttempts; ++attempt)
    {
        frame_graph_begin(mesh, total_Frames, attempt);
        try
        {
            IPC_Solver(mesh);
            return;
        }
        catch(const std::exception& e)
        {
            const std::string cause = e.what();
            if(!m_frame_terminal_emitted)
            {
                const FrameGraphContext& ctx = context(*this);
                const bool retry = ctx.pending_result
                                 == frame_fsm::FRAME_RETRY_REQUIRED;
                frame_graph_enqueue_terminal(
                    mesh,
                    retry ? frame_fsm::FRAME_RETRY_REQUIRED
                          : frame_fsm::FRAME_FATAL,
                    retry ? frame_fsm::ERR_CAPACITY
                          : frame_fsm::ERR_SOLVER_EXCEPTION,
                    retry ? ctx.pending_bits : 0,
                    -1,
                    -1);
                CUDA_SAFE_CALL(cudaStreamSynchronize(cudaStreamPerThread));
                frame_graph_finish_terminal();
            }

            if(m_last_frame_status.result
               == frame_fsm::FRAME_RETRY_REQUIRED)
            {
                const int req_dcd = m_last_frame_status.required_dcd_pairs;
                const int req_ccd = m_last_frame_status.required_ccd_pairs;
                const int req_tri = m_last_frame_status.required_triplets;
                destroy_frame_graph();
                grow_requested_capacities(*this, req_dcd, req_ccd, req_tri);
                continue;
            }
            throw std::runtime_error(status_error(m_last_frame_status, cause));
        }
    }

    m_last_frame_status.result = frame_fsm::FRAME_FATAL;
    m_last_frame_status.error_code = frame_fsm::ERR_RETRY_EXHAUSTED;
    throw std::runtime_error(status_error(m_last_frame_status,
                                          "capacity retry budget exhausted"));
}
