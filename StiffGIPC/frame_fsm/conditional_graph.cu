#include "frame_fsm/conditional_graph.h"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace frame_fsm
{
thread_local ConditionalGraphRecorder*
    ConditionalGraphRecorder::s_current = nullptr;

namespace
{
[[noreturn]] void graph_error(cudaError_t error, const char* operation)
{
    std::ostringstream message;
    message << "[conditional-graph] " << operation << ": "
            << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

void check(cudaError_t error, const char* operation)
{
    if(error != cudaSuccess)
        graph_error(error, operation);
}

void end_active_capture(cudaStream_t stream) noexcept
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(stream, &status) != cudaSuccess
       || status == cudaStreamCaptureStatusNone)
    {
        cudaGetLastError();
        return;
    }

    cudaGraph_t abandoned = nullptr;
    cudaStreamEndCapture(stream, &abandoned);
    // For capture-to-existing-graph, `abandoned` is owned by the caller/root.
    // Never destroy it here; the top-level builder discards the owner graph.
    cudaGetLastError();
}
}  // namespace

ConditionalGraphRecorder::ConditionalGraphRecorder(cudaGraph_t owner,
                                                   cudaStream_t stream)
    : m_owner(owner)
    , m_stream(stream)
{
    if(!m_owner)
        throw std::invalid_argument(
            "[conditional-graph] owner graph is null");
}

ConditionalGraphRecorder::~ConditionalGraphRecorder()
{
    deactivate();
}

void ConditionalGraphRecorder::activate()
{
    if(m_active)
        return;
    if(s_current)
        throw std::logic_error(
            "[conditional-graph] a recorder is already active on this thread");
    s_current = this;
    m_active  = true;
}

void ConditionalGraphRecorder::deactivate() noexcept
{
    if(!m_active)
        return;
    if(s_current == this)
        s_current = nullptr;
    m_active = false;
}

ConditionalGraphRecorder* ConditionalGraphRecorder::current() noexcept
{
    return s_current;
}

cudaGraphConditionalHandle ConditionalGraphRecorder::while_loop(
    unsigned int default_value,
    unsigned int handle_flags,
    const Body& body)
{
    if(!m_active || s_current != this)
        throw std::logic_error(
            "[conditional-graph] WHILE requested without active recorder");
    if(!body)
        throw std::invalid_argument(
            "[conditional-graph] WHILE body is empty");

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long      capture_id     = 0;
    cudaGraph_t             current_graph  = nullptr;
    const cudaGraphNode_t*  frontier       = nullptr;
    size_t                  frontier_count = 0;
    // Use the public overload name. CUDA maps it to the v2/PTSZ entry point
    // when per-thread default streams are enabled; spelling `_v2` directly is
    // not portable across nvcc's host-pass macro boundary.
    check(cudaStreamGetCaptureInfo(m_stream,
                                   &capture_status,
                                   &capture_id,
                                   &current_graph,
                                   &frontier,
                                   &frontier_count),
          "query capture frontier");
    if(capture_status != cudaStreamCaptureStatusActive || !current_graph)
        throw std::logic_error(
            "[conditional-graph] stream is not actively capturing");

    std::vector<cudaGraphNode_t> dependencies;
    if(frontier_count)
        dependencies.assign(frontier, frontier + frontier_count);

    cudaGraph_t ended_graph = nullptr;
    check(cudaStreamEndCapture(m_stream, &ended_graph),
          "end prefix capture");
    if(ended_graph != current_graph)
        throw std::logic_error(
            "[conditional-graph] capture returned a different graph");

    cudaGraphConditionalHandle handle{};
    cudaGraphNode_t             conditional_node = nullptr;
    cudaGraph_t                 body_graph       = nullptr;
    try
    {
        check(cudaGraphConditionalHandleCreate(
                  &handle,
                  m_owner,
                  default_value,
                  handle_flags),
              "create WHILE handle");

        cudaGraphNodeParams parameters{};
        parameters.type               = cudaGraphNodeTypeConditional;
        parameters.conditional.handle = handle;
        parameters.conditional.type   = cudaGraphCondTypeWhile;
        parameters.conditional.size   = 1;
        check(cudaGraphAddNode(&conditional_node,
                               current_graph,
                               dependencies.empty()
                                   ? nullptr
                                   : dependencies.data(),
                               dependencies.size(),
                               &parameters),
              "add WHILE node");
        body_graph = parameters.conditional.phGraph_out[0];
        if(!body_graph)
            throw std::runtime_error(
                "[conditional-graph] CUDA returned a null WHILE body");

        check(cudaStreamBeginCaptureToGraph(
                  m_stream,
                  body_graph,
                  nullptr,
                  nullptr,
                  0,
                  cudaStreamCaptureModeThreadLocal),
              "begin WHILE body capture");
        try
        {
            body(handle);
        }
        catch(...)
        {
            end_active_capture(m_stream);
            throw;
        }

        cudaGraph_t ended_body = nullptr;
        check(cudaStreamEndCapture(m_stream, &ended_body),
              "end WHILE body capture");
        if(ended_body != body_graph)
            throw std::logic_error(
                "[conditional-graph] body capture returned a different graph");

        check(cudaStreamBeginCaptureToGraph(
                  m_stream,
                  current_graph,
                  &conditional_node,
                  nullptr,
                  1,
                  cudaStreamCaptureModeThreadLocal),
              "resume capture after WHILE");
    }
    catch(...)
    {
        end_active_capture(m_stream);
        throw;
    }
    return handle;
}

}  // namespace frame_fsm
