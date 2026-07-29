#include "GIPC.cuh"

#include <algorithm>

void GIPC::record_legacy_frame_status(bool graph_requested,
                                      bool callback_fallback,
                                      int newton_iterations)
{
    frame_fsm::FrameStatus status{};
    status.result             = frame_fsm::FRAME_OK;
    status.phase              = frame_fsm::PHASE_COMMIT;
    status.err_env            = -1;
    status.err_primitive      = -1;
    status.err_newton_iter    = -1;
    status.err_ls_iter        = -1;
    status.path_flags         = graph_requested
                                    ? frame_fsm::PATH_GRAPH_REQUESTED
                                    : 0u;
    if(callback_fallback)
        status.path_flags |= frame_fsm::PATH_LEGACY_FALLBACK;
    status.newton_iters       = std::max(0, newton_iterations);
    status.hw_dcd_pairs       = static_cast<int>(h_cpNum[0]);
    status.hw_ccd_pairs       = static_cast<int>(m_last_ccd_pair_count);
    status.final_alpha        = 1.0;
    status.cfl_alpha          = 1.0;
    status.kappa              = Kappa;
    status.frame_id           = m_total_frames > 0 ? m_total_frames - 1 : 0;
    m_last_frame_status       = status;
    note_frame_graph_coverage(status);
}

// [C6] Graph-coverage telemetry. Every committed frame passes through here or
// through the transaction's terminal, so ANY scene — including the replay
// examples, which each have their own loop — reports how much of its run
// actually executed as one whole-frame graph. STIFF_GRAPH_STATS=1 prints the
// summary; the counters are always maintained (two adds per frame) so a gate
// or a Python caller can read them without re-running.
void GIPC::note_frame_graph_coverage(const frame_fsm::FrameStatus& status)
{
    ++m_frames_committed;
    if(status.path_flags & frame_fsm::PATH_FULL_CONDITIONAL_GRAPH)
        ++m_frames_full_graph;
    else if(status.path_flags & frame_fsm::PATH_GRAPH_ACTIVE)
        ++m_frames_two_graph;
}

void GIPC::print_frame_graph_coverage(const char* tag) const
{
    if(!getenv("STIFF_GRAPH_STATS") || m_frames_committed == 0)
        return;
    printf("[graph-stats]%s%s frames=%d full_graph=%d (%.0f%%) "
           "two_graph=%d legacy=%d\n",
           tag ? " " : "", tag ? tag : "",
           m_frames_committed,
           m_frames_full_graph,
           100.0 * m_frames_full_graph / m_frames_committed,
           m_frames_two_graph,
           m_frames_committed - m_frames_full_graph - m_frames_two_graph);
    fflush(stdout);
}
