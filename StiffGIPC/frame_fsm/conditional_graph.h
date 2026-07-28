#pragma once

// Host-side construction helper for CUDA conditional graph nodes.
//
// A graph executable cannot be launched while its stream is being captured.
// Phase C therefore composes PCG and line-search loops directly into the
// currently captured graph:
//
//   capture prefix -> add WHILE -> capture CUDA-owned body -> resume capture
//
// Conditional handles are associated with the top-level owner graph, even
// when the node itself is nested in another conditional body.

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace frame_fsm
{

class ConditionalGraphRecorder
{
  public:
    using Body = std::function<void(cudaGraphConditionalHandle)>;

    ConditionalGraphRecorder(cudaGraph_t owner, cudaStream_t stream);
    ~ConditionalGraphRecorder();

    ConditionalGraphRecorder(const ConditionalGraphRecorder&) = delete;
    ConditionalGraphRecorder& operator=(
        const ConditionalGraphRecorder&) = delete;

    // Only one recorder may be active on a host thread.  Activation does not
    // begin capture; the caller owns the top-level capture lifetime.
    void activate();
    void deactivate() noexcept;

    // Insert a WHILE node at the current capture frontier.  On return, capture
    // has resumed in the original graph with the conditional node as its sole
    // dependency.  Throws after safely ending capture on any construction
    // failure so the caller can discard the incomplete owner graph.
    cudaGraphConditionalHandle while_loop(
        unsigned int default_value,
        unsigned int handle_flags,
        const Body& body);

    // Capture `predicate(handle)` immediately before an IF node. The
    // predicate must set the handle from device code on every execution.
    cudaGraphConditionalHandle if_then(const Body& predicate,
                                       const Body& body);

    cudaGraph_t owner() const noexcept { return m_owner; }
    cudaStream_t stream() const noexcept { return m_stream; }
    const std::vector<cudaGraph_t>& conditional_bodies() const noexcept
    {
        return m_conditional_bodies;
    }
    const std::vector<cudaGraphNode_t>& conditional_nodes() const noexcept
    {
        return m_conditional_nodes;
    }

    static ConditionalGraphRecorder* current() noexcept;
    static bool active() noexcept { return current() != nullptr; }

  private:
    cudaGraph_t  m_owner  = nullptr;
    cudaStream_t m_stream = nullptr;
    bool         m_active = false;
    std::vector<cudaGraph_t> m_conditional_bodies;
    std::vector<cudaGraphNode_t> m_conditional_nodes;

    static thread_local ConditionalGraphRecorder* s_current;
};

class ConditionalGraphRecorderScope
{
  public:
    explicit ConditionalGraphRecorderScope(ConditionalGraphRecorder& recorder)
        : m_recorder(recorder)
    {
        m_recorder.activate();
    }

    ~ConditionalGraphRecorderScope() { m_recorder.deactivate(); }

    ConditionalGraphRecorderScope(const ConditionalGraphRecorderScope&) =
        delete;
    ConditionalGraphRecorderScope& operator=(
        const ConditionalGraphRecorderScope&) = delete;

  private:
    ConditionalGraphRecorder& m_recorder;
};

}  // namespace frame_fsm
