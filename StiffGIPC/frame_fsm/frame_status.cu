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
    status.hw_ccd_pairs       = static_cast<int>(h_ccd_cpNum);
    status.final_alpha        = 1.0;
    status.cfl_alpha          = 1.0;
    status.kappa              = Kappa;
    status.frame_id           = m_total_frames > 0 ? m_total_frames - 1 : 0;
    m_last_frame_status       = status;
}
