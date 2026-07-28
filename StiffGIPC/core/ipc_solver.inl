// ============================================================================
// core/ipc_solver.inl — frame orchestration, single owner (v0.8.6 Phase 2d).
//
// The ONLY place that knows how a frame walks: GIPC::IPC_Solver (per frame),
// GIPC::solve_subIP (Newton loop), GIPC::lineSearch / GIPC::postLineSearch.
// Stage order is documented in core/frame_pipeline.h — change them TOGETHER.
//
// Since Phase 2d step 2 this file is the body of core/ipc_solver.cu — its OWN
// translation unit (the refactor's first physical TU separation). The .cu
// shell provides the context: extern decls for every kernel launched here
// (definitions stay in their mechanism modules, composite TU) and for the
// composite-owned globals (g_gipc_log_level, g_dec_k/g_dec_frame). Bodies are
// verbatim moves from gipc_modules/14. Runtime counters are GIPC members so
// multiple engines cannot contaminate one another's telemetry/frame state.
//
// Mechanics live with their owners and are CALLED from here, never inlined
// here: pair growth = contact/pair_buffers.cuh, iron-law quarantine =
// multienv/isolation.cuh, reductions = device_common/reductions.cuh,
// triplet growth = linear_system ensure_capacity_{preserve,discard}.
// ============================================================================
#include "frame_pipeline.h"
#include "errors.h"  // [error-taxonomy] GeometryError for frame-0 infeasibility
#include "device_common/nvtx_ranges.h"  // [B3] phase attribution

// ── verbatim from gipc_modules/14 (pre-2d lines 962..1374) ──
// [phase-time] lineSearch inner split (per frame): energy evals vs buildBVH+intersect vs buildCP vs step.
static double g_ls_e_ms = 0.0, g_ls_bvh_ms = 0.0, g_ls_cp_ms = 0.0, g_ls_step_ms = 0.0;
// gated event-pair stopwatch (STIFF_PHASE_TIME only; events are stream-ordered, no extra syncs —
// callers place it around ops that already end host-synchronous).
struct _LsTimer {
    bool on; cudaEvent_t a, b; double* acc;
    _LsTimer(double* accum) : on(getenv("STIFF_PHASE_TIME") != nullptr), acc(accum)
    { if(on){ cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a); } }
    void stop()
    { if(on){ cudaEventRecord(b); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b);
              *acc += m; cudaEventDestroy(a); cudaEventDestroy(b); on=false; } }
};

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    bool   stopped       = false;
    const char* device_ls_env = getenv("STIFF_DEVICE_LINESEARCH");
    const bool device_ls = !device_ls_env || !device_ls_env[0]
                        || device_ls_env[0] != '0';
    // The preceding CCD/buildFullCP path already joins its auxiliary stream and
    // reads its counts on the host.  All energy work below is ordered on PTDS,
    // so the former full-device wait here was redundant.
    double lastEnergyVal = 0.0;
    if(device_ls)
        computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 0);
    else
        lastEnergyVal = computeEnergy(TetMesh);
    bool perenv_try = (m_env_alpha_valid && m_env_alpha && TetMesh.d_point_to_group
                       && TetMesh.h_groups_present
                       && abd_fem_count_info.fem_point_num > 0
                       && m_mode_config.perenv_alpha);

    // [multi-env S3] validate per-env energy decomposition (Sum_g E_g(FEM) ==
    // global FEM). Read-only; gated. Run a couple times then it's confirmed.
    if(getenv("STIFF_S3_VALIDATE") && TetMesh.d_point_to_group)
    {
        std::vector<double> eg;
        computeEnergy_perenv(TetMesh, eg);  // prints [S3-energy] when STIFF_PENV_STATS
    }

    double c1m         = 0.0;
    double armijoParam = 0;
    if(armijoParam > 0.0)
    {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                              TetMesh.vertexes,
                              vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);


    double alpha_SL = alpha;
    const int line_search_budget =
        line_search_max_iter > 0 ? line_search_max_iter : 64;

    // [multi-env S2/S3] RIGOROUS per-env line search. Step each env by its own
    // CCD-feasible alpha (m_env_alpha, S1-validated safe), then enforce PER-ENV
    // energy DESCENT: any env whose own energy E_g rose halves its alpha_g and
    // re-steps (independent per-env backtracking) — NOT the global-energy heuristic
    // (which can mask one env's increase). Per-env energy is validated exact
    // (Sum_g E_g == global, machine precision). Accept when every env satisfies
    // E_g(alpha_g) <= E_g(0) with no intersection. Falls back to the standard
    // uniform search only if an env can't descend within maxBT halvings. Gated.
    if(perenv_try)   // (perenv_try hoisted to the top — see the lazy lastEnergyVal comment)
    {
        const int NG = TetMesh.h_group_count;
        const int abdN = (int)abd_fem_count_info.abd_body_num;
        const int maxBT = 8;
        // [de-CPU S3] energies + decision fully DEVICE-resident: computeEnergy_perenv_dev fills
        // d_Eg0/d_Eg1 (no 22KB D2H), _s3_decide halves m_env_alpha IN PLACE (the true state; the
        // old host loop operated on the stale h_env_alpha mirror). Host reads ONE int per round —
        // required: it decides whether to re-run the step/rebuild/energy round (host loop control).
        double*& d_Eg0 = m_scr_ls_eg0; double*& d_Eg1 = m_scr_ls_eg1;
        int*& d_decision_counts = m_scr_ls_decision_counts;
        if(!d_Eg0)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg0, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg1, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_decision_counts, 2 * sizeof(int)));
        }
        _LsTimer _t0(&g_ls_e_ms);
        computeEnergy_perenv_dev(TetMesh, d_Eg0);   // per-env energy at START (temp) config
        _t0.stop();
        bool accepted = false;
        for(int bt = 0; bt <= maxBT; ++bt)
        {
            if(abdN > 0 && TetMesh.d_body_to_group)   // refresh per-body ABD alpha
            {
                if(!m_abd_body_alpha)
                    m_abd_body_alpha.resize_discard(abdN);  // [3d-2]
                int tn = 256, bn = (abdN + tn - 1) / tn;
                _gather_abd_body_alpha<<<bn, tn>>>(TetMesh.d_body_to_group, m_env_alpha,
                                                   m_abd_body_alpha, abdN, NG);
            }
            _LsTimer _ts(&g_ls_step_ms);
            m_perenv_apply = true;
            step_forward(TetMesh, alpha, false);   // per-env step from temp
            m_perenv_apply = false;
            _ts.stop();
            _LsTimer _tb(&g_ls_bvh_ms);
            buildBVH();
            int ground_trial_status = groundTrialStatus(TetMesh.d_point_to_group, NG);
            if(ground_trial_status != 0)
            {
                _tb.stop();
                if((ground_trial_status & 2) == 0)
                {
                    halveGroundInvalidEnvAlpha(NG);
                    continue;
                }
                break;
            }
            bool _isect = isIntersected(TetMesh);
            _tb.stop();
            if(_isect)   // CCD-safe alpha should prevent this; safety net
            {
                _s3_halve_all<<<(NG + 255) / 256, 256>>>(m_env_alpha, NG);
                continue;
            }
            _LsTimer _tc(&g_ls_cp_ms);
            buildCP();
            _tc.stop();
            // [backport] buildCP() already D2H-syncs h_cpNum/h_gpNum and the energy
            // kernels consume unsorted pairs directly, so the perf branch's
            // sync_cpNum()/sort_collision_pairs_by_type() are unneeded in this path.
            _LsTimer _t1(&g_ls_e_ms);
            computeEnergy_perenv_dev(TetMesh, d_Eg1);   // per-env energy AFTER step
            _t1.stop();
            CUDA_SAFE_CALL(cudaMemsetAsync(d_decision_counts, 0, 2 * sizeof(int)));
            _s3_decide<<<(NG + 255) / 256, 256>>>(
                d_Eg0, d_Eg1, m_env_alpha, d_decision_counts, NG,
                energy_abs_tol, energy_rel_tol);
            int decision_counts[2] = {0, 0};
            CUDA_SAFE_CALL(cudaMemcpy(decision_counts, d_decision_counts,
                                      2 * sizeof(int), cudaMemcpyDeviceToHost));
            int nfail = decision_counts[0];
            if(nfail == 0)
            {
                energy_tolerance_accept_count +=
                    static_cast<uint64_t>(decision_counts[1]);
                accepted = true;
                if(getenv("STIFF_PENV_STATS")) printf("[S3-accept] per-env descent ok (bt=%d)\n", bt);
                break;
            }
            // (m_env_alpha already halved in place on device — no H2D)
            if(getenv("STIFF_PENV_STATS")) printf("[S3-bt] bt=%d halved %d envs\n", bt, nfail);
        }
        if(accepted) return false;   // accepted; collision state already rebuilt
        if(getenv("STIFF_PENV_STATS")) printf("[S3-fallback] per-env descent not reached -> global\n");
        // fall through to standard uniform-alpha line search (re-steps from temp); the Armijo
        // reference lastEnergyVal is the eager entry computeEnergy (bit-exact original semantics).
    }

    step_forward(TetMesh, alpha, false);

    bool rehash = true;

    buildBVH();

    int ground_trial_backtracks = 0;
    int ground_trial_status = groundTrialStatus(nullptr, 0);
    while(ground_trial_status != 0 && ground_trial_backtracks < line_search_budget)
    {
        alpha *= 0.5;
        ++ground_trial_backtracks;
        step_forward(TetMesh, alpha, false);
        buildBVH();
        ground_trial_status = groundTrialStatus(nullptr, 0);
    }
    if(ground_trial_status != 0)
        throw std::runtime_error(
            "[StiffGIPC] ground trial step remained outside the strict barrier "
            "domain after line-search backtracking");

    int numOfIntersect = 0;
    int insectNum      = 0;

    bool checkInterset = true;

    while(checkInterset && isIntersected(TetMesh))
    {
        if(numOfIntersect >= line_search_budget)
        {
            // After `line_search_budget` halvings the trial state is numerically
            // the start state: the Newton step BEGAN intersecting (IPC feasibility
            // invariant already broken upstream) — no step size can fix that.
            // Fail fast like the ground trial above instead of spinning forever.
            throw std::runtime_error(
                "[StiffGIPC] mesh intersection persists after line-search "
                "backtracking (start state likely already intersecting; IPC "
                "feasibility invariant broken in an earlier step)");
        }
        printf("type 0 intersection happened 0:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = std::min(cfl_alpha, alpha);
        step_forward(TetMesh, alpha, false);
        buildBVH();
        //break;
    }

    // [B3 trial-defer] merged path only (the per-env buildCP refreshes counts
    // inside its own pipeline): contact/ground energy grids use slacked
    // iteration-start bounds + device live counts during trials; the mirror
    // stays invalid (MIRROR_AUDIT enforces no stale reader) and is refreshed
    // once at line-search exit. Overflow rides the decision read.
    const bool ls_defer = device_ls && !(m_perenv_bvh && m_d_p2g);
    if(ls_defer)
    {
        m_energy_bound_cp = (int)h_cpNum[0] + (int)h_cpNum[0] / 4 + 64;
        m_energy_bound_gp = (int)h_gpNum + (int)h_gpNum / 4 + 64;
        m_ls_defer_counts          = true;
        m_energy_use_device_counts = true;
    }
    buildCP();

    double testingE = 0.0;

    auto evaluate_trial_energy = [&](double trial_alpha) {
        if(device_ls)
        {
            computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 1);
            _global_ls_decide<<<1, 1>>>(m_line_search_energy + 0,
                                        m_line_search_energy + 1,
                                        c1m,
                                        trial_alpha,
                                        energy_abs_tol,
                                        energy_rel_tol,
                                        m_line_search_decision,
                                        _gdCollapse,
                                        nullptr,
                                        nullptr);
            int dec_of[3] = {0, 0, 0};
            CUDA_SAFE_CALL(cudaMemcpy(dec_of,
                                      m_line_search_decision,
                                      3 * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            int decision = dec_of[0];
            // [B3 trial-defer] monotone pair-overflow counter piggybacked on
            // the decision read. A bump means this trial's detection hit the
            // emission caps: re-run buildCP in legacy mode (full counts +
            // grow + redo machinery), refresh the bounds, and re-evaluate the
            // trial energy once. False positives (counter drift from a legacy
            // overflow elsewhere) just repeat this benign recovery.
            if(m_ls_defer_counts
               && (unsigned)dec_of[1] != m_pair_overflow_seen)
            {
                m_pair_overflow_seen       = (unsigned)dec_of[1];
                m_ls_defer_counts          = false;
                buildCP();
                m_ls_defer_counts          = true;
                m_energy_bound_cp = (int)h_cpNum[0] + (int)h_cpNum[0] / 4 + 64;
                m_energy_bound_gp = (int)h_gpNum + (int)h_gpNum / 4 + 64;
                computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 1);
                _global_ls_decide<<<1, 1>>>(m_line_search_energy + 0,
                                            m_line_search_energy + 1,
                                            c1m,
                                            trial_alpha,
                                            energy_abs_tol,
                                            energy_rel_tol,
                                            m_line_search_decision,
                                            _gdCollapse,
                                            nullptr,
                                            nullptr);
                CUDA_SAFE_CALL(cudaMemcpy(dec_of,
                                          m_line_search_decision,
                                          3 * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                decision             = dec_of[0];
                m_pair_overflow_seen = (unsigned)dec_of[1];
            }
            // [B3 trial-defer] deferred ground-collapse response: same trial,
            // one call, idempotent (0 = clean). Legacy mode (defer off) already
            // handled it inside buildCP, so gate on the flag to avoid doubling.
            if(m_ls_defer_counts && dec_of[2] < 0)
                handleGroundCollapse(dec_of[2]);
            if(getenv("STIFF_DEVICE_LINESEARCH_VALIDATE"))
            {
                double h_energy[2] = {0.0, 0.0};
                CUDA_SAFE_CALL(cudaMemcpy(h_energy,
                                          m_line_search_energy,
                                          sizeof(h_energy),
                                          cudaMemcpyDeviceToHost));
                const double rhs = h_energy[0] + c1m * trial_alpha;
                const double tol = energy_abs_tol
                                 + energy_rel_tol * fabs(h_energy[0]);
                const int host_decision =
                    !std::isfinite(h_energy[1])   // mirror the kernel's NaN guard
                        ? 1
                        : (h_energy[1] > rhs + tol ? 1
                                                   : (h_energy[1] > rhs ? 2 : 0));
                if(host_decision != decision)
                    throw std::runtime_error(
                        "[line-search] device and host decisions differ");
                static bool first_match = true;
                if(first_match)
                {
                    printf("[line-search-device-validate] first decision matched (%d)\n",
                           decision);
                    first_match = false;
                }
            }
            return decision;
        }

        testingE = computeEnergy(TetMesh);
        const double rhs = lastEnergyVal + c1m * trial_alpha;
        const double tol = energy_abs_tol + energy_rel_tol * fabs(lastEnergyVal);
        // [NaN-quarantine gap fix] see _global_ls_decide: NaN must read as
        // "not a descent" (backtrack), never as silent acceptance.
        if(!std::isfinite(testingE))
            return 1;
        return testingE > rhs + tol ? 1 : (testingE > rhs ? 2 : 0);
    };
    int energy_decision = evaluate_trial_energy(alpha);

    int    numOfLineSearch = 0;
    double LFStepSize      = alpha;

    // [C-1 ls-graph] backtracking as a device self-tail-launch graph: the trial
    // body (halve -> step -> BVH -> CP(defer) -> energy -> decide) re-launches
    // itself on device while status[0]==1 with budget left; the host does ONE
    // packed 24B read after the loop instead of a 12B sync per trial. Overflow
    // and ground-collapse handling move post-loop (the packed read carries
    // both); a truncated-emission false accept is re-adjudicated by the legacy
    // evaluate below. Cached across line searches on the buffer generation.
    {
        static int s_ls_graph = -1;
        if(s_ls_graph < 0)
        { const char* e = getenv("STIFF_LS_GRAPH"); s_ls_graph = e ? atoi(e) : 0; }
        if(energy_decision == 1 && s_ls_graph && device_ls && m_ls_defer_counts
           && m_total_frames >= 1 /* frame 0 warms every lazy alloc */)
        {
            const long long sig0 = pcg_buffer_generation();
            if(m_ls_graph_exec
               && (m_ls_graph_sig[0] != sig0
                   || m_ls_graph_sig[1] != (long long)line_search_budget))
            {
                cudaGraphExecDestroy(m_ls_graph_exec);
                m_ls_graph_exec = nullptr;
            }
            if(!m_ls_graph_exec)
            {
                cudaGraph_t lg = nullptr;
                bool record_threw = false;
                if(cudaStreamBeginCapture(cudaStreamPerThread,
                                          cudaStreamCaptureModeThreadLocal)
                   == cudaSuccess)
                {
                    // A capture-illegal call inside the body (sync memcpy, alloc,
                    // muda wait) surfaces as a C++ throw: terminate the capture,
                    // fall back to the host loop PERMANENTLY, never crash.
                    const char* _cap_stage = "begin";
                    // [C-1] the recorded energy grids must outlive this frame's
                    // pair counts: record with capacity bounds (kernels mask by
                    // d_live; zero-padded sums are bitwise-neutral), restore after.
                    const int _save_bcp = m_energy_bound_cp;
                    const int _save_bgp = m_energy_bound_gp;
                    m_energy_bound_cp   = MAX_COLLITION_PAIRS_NUM;
                    m_energy_bound_gp   = surf_vertexNum;
                    m_ls_recording      = true;
                    try
                    {
                        _ls_trial_begin<<<1, 1>>>(m_d_ls_alpha, m_line_search_decision);
                        _cap_stage = "step_forward";
                        step_forward(TetMesh, 0.0, false, m_d_ls_alpha);
                        _cap_stage = "buildBVH";
                        buildBVH();
                        _cap_stage = "buildCP";
                        buildCP();
                        _cap_stage = "energy";
                        computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 1,
                                                m_d_ls_scalars + 1);
                        _cap_stage = "decide";
                        _global_ls_decide<<<1, 1>>>(m_line_search_energy + 0,
                                                    m_line_search_energy + 1,
                                                    0.0,
                                                    0.0,
                                                    energy_abs_tol,
                                                    energy_rel_tol,
                                                    m_line_search_decision,
                                                    _gdCollapse,
                                                    m_d_ls_alpha,
                                                    m_d_ls_scalars + 0);
                        _ls_trial_tail<<<1, 1>>>(m_line_search_decision,
                                                 line_search_budget,
                                                 m_d_ls_alpha);
                    }
                    catch(const std::exception& ex)
                    {
                        record_threw = true;
                        s_ls_graph   = 0;   // permanent session fallback
                        cudaGraph_t junk = nullptr;
                        cudaStreamEndCapture(cudaStreamPerThread, &junk);
                        if(junk) cudaGraphDestroy(junk);
                        cudaGetLastError();
                        fprintf(stderr,
                                "[ls-graph] trial body not capturable at stage=%s (%s) -> "
                                "host loop fallback for this session\n",
                                _cap_stage,
                                ex.what());
                    }
                    m_energy_bound_cp = _save_bcp;
                    m_energy_bound_gp = _save_bgp;
                    m_ls_recording    = false;
                    if(!record_threw
                       && cudaStreamEndCapture(cudaStreamPerThread, &lg) == cudaSuccess && lg
                       && cudaGraphInstantiateWithFlags(&m_ls_graph_exec,
                                                        lg,
                                                        cudaGraphInstantiateFlagDeviceLaunch)
                              == cudaSuccess
                       && m_ls_graph_exec)
                    {
                        m_ls_graph_sig[0] = sig0;
                        m_ls_graph_sig[1] = (long long)line_search_budget;
                        if(getenv("STIFF_LS_GRAPH_DIAG"))
                        {
                            static int _ncap = 0;
                            printf("[ls-graph-diag] capture #%d gen=%lld frame=%u\n",
                                   ++_ncap, sig0, m_total_frames);
                        }
                        static bool onceg = false;
                        if(!onceg)
                        {
                            onceg = true;
                            printf("[ls-graph] trial self-tail graph active (budget=%d)\n",
                                   line_search_budget);
                        }
                    }
                    else
                        m_ls_graph_exec = nullptr;
                }
                if(lg) cudaGraphDestroy(lg);
                cudaGetLastError();   // clear any sticky capture error
            }
            if(m_ls_graph_exec)
            {
                _ls_seed<<<1, 1>>>(m_d_ls_alpha, alpha, m_line_search_decision,
                                   m_d_ls_scalars, c1m, Kappa);
                CUDA_SAFE_CALL(cudaGraphUpload(m_ls_graph_exec, cudaStreamPerThread));
                CUDA_SAFE_CALL(cudaGraphLaunch(m_ls_graph_exec, cudaStreamPerThread));
                int s6[6] = {0, 0, 0, 0, 0, 0};
                CUDA_SAFE_CALL(cudaMemcpy(s6,
                                          m_line_search_decision,
                                          6 * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                energy_decision = s6[0];
                numOfLineSearch = s6[3];
                if(getenv("STIFF_LS_GRAPH_DIAG"))
                {
                    static int _dbg = 0;
                    if(_dbg++ < 6)
                    {
                        double e01[2] = {0, 0};
                        CUDA_SAFE_CALL(cudaMemcpy(e01, m_line_search_energy,
                                                  2 * sizeof(double),
                                                  cudaMemcpyDeviceToHost));
                        double a_dev = 0;
                        CUDA_SAFE_CALL(cudaMemcpy(&a_dev, m_d_ls_alpha,
                                                  sizeof(double),
                                                  cudaMemcpyDeviceToHost));
                        printf("[ls-graph-diag] dec=%d trials=%d a_dev=%.6e "
                               "a_host_in=%.6e E0=%.9e E1=%.9e\n",
                               s6[0], s6[3], a_dev, LFStepSize, e01[0], e01[1]);
                    }
                }
                const unsigned long long bits =
                    ((unsigned long long)(unsigned)s6[5] << 32)
                    | (unsigned long long)(unsigned)s6[4];
                memcpy(&alpha, &bits, sizeof(alpha));
                if(s6[2] < 0)
                    handleGroundCollapse(s6[2]);
                if((unsigned)s6[1] != m_pair_overflow_seen)
                    energy_decision = evaluate_trial_energy(alpha);
            }
        }
    }

    std::cout.precision(18);
    // A larger configurable budget prevents the former hard-coded eight-step
    // limit from silently accepting a non-descent step in difficult contact.
    // Exhaustion remains loud because the current engine policy accepts the
    // final candidate so callers can decide whether to abort the simulation.
    while(energy_decision == 1 && numOfLineSearch < line_search_budget)
    {
        //std::cout << "[" << numOfLineSearch << "]   testE:    " << testingE
        //          << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        step_forward(TetMesh, alpha, false);
        buildBVH();
        buildCP();
        energy_decision = evaluate_trial_energy(alpha);
    }
    // [B3 trial-defer] restore mirror freshness once for everything after
    // the trial loop (postLineSearch, next-iteration GH, close constraints).
    if(m_ls_defer_counts)
    {
        m_ls_defer_counts          = false;
        m_energy_use_device_counts = false;
        refresh_pair_counts();
    }
    const bool line_search_exhausted = energy_decision == 1;
    if(energy_decision == 2)
        ++energy_tolerance_accept_count;
    if(line_search_exhausted)
    {
        if(device_ls)
        {
            double h_energy[2] = {0.0, 0.0};
            CUDA_SAFE_CALL(cudaMemcpy(h_energy,
                                      m_line_search_energy,
                                      2 * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            lastEnergyVal = h_energy[0];
            testingE      = h_energy[1];
        }
        // [T1] Monotonicity NOT achieved within budget: the accepted step raises
        // the incremental potential. Loud and unconditional — a silent
        // non-descent step corrupts contact state downstream.
        fprintf(stderr,
                "[line-search][WARN] budget exhausted (%d halvings, alpha=%.3e): "
                "energy did NOT decrease (E=%.9e > E0=%.9e). Step accepted anyway "
                "-- POTENTIAL SOLVER ERROR: expect contact drift / collapsed "
                "barrier distances / iteration blow-up in later frames. Raise "
                "Config.line_search_max_iter, reduce dt, or soften the drive.\n",
                numOfLineSearch, alpha, testingE, lastEnergyVal);
        // [rl-reset] step-health telemetry: an RL loop diffs these counters
        // across step() to detect and discard degraded episodes.
        ++m_ls_exhausted_total;
        if(!std::isfinite(testingE) || !std::isfinite(lastEnergyVal))
            ++m_ls_nonfinite_total;
        // [error-taxonomy] frame 0 + non-finite incremental potential at every
        // trial alpha = the INITIAL configuration is infeasible (interpenetrating
        // bodies at spawn — log-barrier of a negative distance). Historically a
        // silent NaN cascade; mid-run policy (WARN + accept, isolated-mode
        // quarantine) is deliberately unchanged — this fires only before any
        // valid frame ever existed, where "keep going" can only produce garbage.
        if(m_total_frames == 0
           && (!std::isfinite(testingE) || !std::isfinite(lastEnergyVal)))
        {
            throw gipc::GeometryError(
                "frame 0 line search exhausted with non-finite incremental "
                "potential — the initial configuration is infeasible for IPC "
                "(bodies interpenetrating at spawn, or spawned through the "
                "ground). Fix the spawn transforms; IPC requires a "
                "penetration-free initial state.");
        }
    }


    if(alpha < LFStepSize)
    {
        bool needRecomputeCS = false;
        while(checkInterset && isIntersected(TetMesh))
        {
            if(numOfIntersect >= line_search_budget)
            {
                throw std::runtime_error(
                    "[StiffGIPC] mesh intersection persists after energy "
                    "line-search backtracking (start state likely already "
                    "intersecting; IPC feasibility invariant broken in an "
                    "earlier step)");
            }
            printf("type 1 intersection happened 1:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = std::min(cfl_alpha, alpha);

            step_forward(TetMesh, alpha, false);
            buildBVH();
            needRecomputeCS = true;
        }
        if(needRecomputeCS)
        {
            buildCP();
        }
    }

    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if(Kappa == 0.0)
    {
        initKappa(TetMesh);
    }
    else
    {
        if(m_pergroup_kappa && m_kappa_group && m_d_close_grp)
        {
            // [multi-env per-group κ] run BOTH close-val checks (no short-circuit) to populate the
            // per-group flags, then double ONLY the groups that hit a close contact. This decouples
            // the cross-env bifurcation (a global κ doubling was the dominant cross-env coupling).
            int NG = m_active_group_count;
            CUDA_SAFE_CALL(cudaMemset(m_d_close_grp, 0, NG * sizeof(int)));
            (void)checkCloseGroundVal();   // populates m_d_close_grp (global bool ignored)
            (void)checkSelfCloseVal();
            // [perf/device-residence] double the close groups' κ ON DEVICE (in m_kappa_group), taking the
            // envelope via atomicMax — replaces the per-Newton D2H(close flags) + 256-env host loop +
            // H2D(κ). kappaMax is a host scalar (env-independent), the doubling+cap+max are all order-free
            // → strict bit-identical. h_kappa_group is NOT touched here (it is re-seeded by initKappa each
            // frame and read nowhere else during the frame; device m_kappa_group is the in-frame truth).
            double kappaMax = 1e300;
            upperBoundKappa(kappaMax);     // kappaMax = env-independent cap (was recomputed per group)
            double*& d_maxK = m_scr_maxk;
            if(!d_maxK) CUDA_SAFE_CALL(cudaMalloc(&d_maxK, sizeof(double)));
            CUDA_SAFE_CALL(cudaMemcpy(d_maxK, &Kappa, sizeof(double), cudaMemcpyHostToDevice));  // envelope init
            {
                int bs = 256;
                const double* frozen_alpha =
                    (m_mode_config.decouple_thresh && m_env_alpha_valid)
                        ? m_env_alpha.data()
                        : nullptr;
                _per_group_kappa_double<<<(NG + bs - 1) / bs, bs>>>(
                    m_kappa_group, m_d_close_grp, frozen_alpha, NG, kappaMax, d_maxK);
            }
            CUDA_SAFE_CALL(cudaMemcpy(&Kappa, d_maxK, sizeof(double), cudaMemcpyDeviceToHost));  // scalar envelope
            tempFree_closeConstraint();
            tempMalloc_closeConstraint();
            CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));
            computeCloseGroundVal();
            computeSelfCloseVal();
            return;
        }

        bool updateKappa = checkCloseGroundVal();
        if(!updateKappa)
        {
            updateKappa = checkSelfCloseVal();
        }
        if(updateKappa)
        {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
}

// ── verbatim from gipc_modules/14 (pre-2d lines 1448..2543) ──
#include <vector>
#include <fstream>
std::vector<int> iterV;

// [phase-time] time3 sub-split (per frame): S1 per-env-alpha block vs lineSearch proper.
static double g_t3_s1_ms = 0.0, g_t3_ls_ms = 0.0;

int              GIPC::solve_subIP(device_TetraData& TetMesh,
                      double&           time0,
                      double&           time1,
                      double&           time2,
                      double&           time3,
                      double&           time4)
{
    auto& stats_at_current_frame = gipc::Statistics::instance().at_current_frame();
    if(g_gipc_log_level >= 1)
        std::cout << "solve_subIP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                  << std::endl;

    stats_at_current_frame["newton"] = gipc::Json::array();
    g_t3_s1_ms = 0.0; g_t3_ls_ms = 0.0;   // [phase-time] time3 sub-split, per frame
    g_ls_e_ms = 0.0; g_ls_bvh_ms = 0.0; g_ls_cp_ms = 0.0; g_ls_step_ms = 0.0;
    const int active_group_count = TetMesh.h_group_count;
    m_active_group_count = active_group_count;
    // [env-det] capture p2g at entry so the canon env-local-id tiebreak (g_vloc) is built BEFORE the
    // first buildCP of this frame (buildCP runs before computeGradientAndHessian sets m_d_p2g).
    if(m_mode_config.ee_canon && TetMesh.d_point_to_group) m_d_p2g = TetMesh.d_point_to_group;

    int iterCap = newton_iter_cap, k = 0;
    double semi_beta = 1.0;
    // [per-env productization] reset per-env telemetry for this solve
    m_env_frozen_iter.assign(kEnvAlphaSlots, -1);
    m_env_status.assign(kEnvAlphaSlots, 0);
    // [semi-implicit per-env] beta_g for the decoupled/host path: each env
    // accumulates its OWN line-search progress (candidate alpha_g from S1)
    // and exits by freezing ITSELF — the per-env analogue of Alg.1 that the
    // global beta cannot provide without re-coupling the batch.
    std::vector<double> semi_beta_env(kEnvAlphaSlots, 1.0);
    // [drive-substep] uipc-style animation substepping for joint driving: with
    // STIFF_DRIVE_SUBSTEP=S (>1), the per-frame driving target ramps linearly
    // over the first S Newton iterations (ratio=(k+1)/S) instead of dumping the
    // whole frame's driving energy into iteration 0 — that lump budget is what
    // the global monotone line search can "spend" on flipping a light resting
    // body (the kick). Convergence exits are suppressed until the ramp
    // completes (uipc: animation_reach_target).
    static const int s_drive_substep =
        getenv("STIFF_DRIVE_SUBSTEP") ? atoi(getenv("STIFF_DRIVE_SUBSTEP")) : 0;
    const bool drive_substep_on = s_drive_substep > 1 && m_drive_substep_mesh;
    double     drive_ratio      = drive_substep_on ? 0.0 : 1.0;

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    double totalTimeStep = 0;

    // [multi-env S4] inject per-env mask into the linear system: solve_linear_system
    // zeros masked envs' RHS, spmv skips their triplets. Reset all-active at frame
    // start (detection updates m_env_active each iter); cleared after the loop.
    const bool s4_mask_on = (m_env_active && TetMesh.d_dof_to_group
                             && TetMesh.d_point_to_group
                             && TetMesh.h_groups_present
                             && (m_mode_config.perenv_mask
                                 || m_mode_config.perenv_mask_dev));
    // [S4-dev] device-derived mask (from m_env_alpha, zero D2H). Requires the per-env alpha
    // machinery (m_env_alpha filled by the S1 line-search block each iter).
    const bool s4_dev_mask =
        s4_mask_on && m_env_alpha && m_mode_config.perenv_mask_dev
                             && m_mode_config.perenv_alpha;
    if(s4_mask_on)
    {
        std::fill_n(h_env_active.begin(), active_group_count, 1);
        CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                  active_group_count * sizeof(int), cudaMemcpyHostToDevice));
        m_global_linear_system->set_env_mask(
            m_env_active, TetMesh.d_dof_to_group, active_group_count);
    }
    // [multi-env P3] register the DOF→group map for the SEGMENTED block-diagonal PCG even when
    // masking is off (the PCG reads m_s4_dof_to_group/m_s4_ng). active=nullptr ⇒ no RHS masking.
    else if(m_mode_config.segmented_pcg && TetMesh.d_dof_to_group
            && TetMesh.d_point_to_group && TetMesh.h_groups_present)
        m_global_linear_system->set_env_mask(
            nullptr, TetMesh.d_dof_to_group, active_group_count);
    else
        m_global_linear_system->set_env_mask(nullptr, nullptr, 0);

    int& s_dec_frame = g_dec_frame;   // [decouple probe] per-solve_subIP-call counter (~frame)
    s_dec_frame++;

    // [decouple] per-env convergence latch for the loop-exit override. The global gradVanish can
    // fire while a per-env env is still UNDER-converged (its mates converged fast → loop ends → that
    // env is cut off at a batch-dependent iter → drift). When DECOUPLE_THRESH, the loop exits only
    // once ALL present envs are per-env frozen (each reached ITS OWN convergence), so an env's final
    // state is independent of the mates / loop length. Updated at the end of S1 Phase B each iter.
    bool all_env_frozen = false;
    // Merged/global Newton convergence must be evaluated on the direction
    // solved at the CURRENT state. The legacy placement checked the previous
    // iteration's direction before the PCG solve, so a newly converged tiny
    // direction still entered line search and could spend 64 halvings fighting
    // energy-reduction roundoff. Per-env modes retain their existing freeze
    // pipeline for now and will be handled separately.
    const bool current_global_exit = !m_mode_config.decouple_thresh;
    // Seven per-iteration events are diagnostic-only. Production must not
    // create/destroy them or force a device-wide synchronization.
    const bool phase_time = (getenv("STIFF_PHASE_TIME") != nullptr);

    for(; k < iterCap; ++k)
    {
        if(g_gipc_log_level >= 1 && k > 0 && k % 10 == 0)
            printf("  Newton iter %d ...\n", k);
        stats_at_current_frame["newton"].push_back(gipc::Json::object());

        // [drive-substep] ramp the joint driving targets across the solve.
        // theta_prev/d_prev derive from q_prev (frame-start state, constant
        // within the solve), so re-invoking per iteration is deterministic.
        if(drive_substep_on && drive_ratio < 1.0)
        {
            drive_ratio = std::min(1.0, double(k + 1) / s_drive_substep);
            update_joint_angle_targets_from_mesh(*m_drive_substep_mesh, drive_ratio);
        }

        // [S4-dev] periodic all-active recheck (bounce-back detection): every RECHECK iters, unmask
        // ALL envs so masked (frozen) envs get one REAL solve — if a κ doubling / friction update
        // moved a frozen env off its optimum, its hmx exceeds thr and _per_env_alpha_compute
        // un-freezes it (mask follows at end of this iter). Placed BEFORE the solve so the recheck
        // iter solves the full system. Fixed cadence → deterministic.
        if(s4_dev_mask && (k % 4 == 0))
            _mask_fill<<<(active_group_count + 255) / 256, 256>>>(
                m_env_active, 1, active_group_count);

        m_total_collision_pairs += h_cpNum[0];
        m_max_collision_pairs =
            (m_max_collision_pairs > h_cpNum[0]) ? m_max_collision_pairs : h_cpNum[0];
        cudaEvent_t start = nullptr, end0 = nullptr, end1 = nullptr, end2 = nullptr;
        cudaEvent_t end3 = nullptr, end4 = nullptr, e2b = nullptr;
        if(phase_time)
        {
            CUDA_SAFE_CALL(cudaEventCreate(&start));
            CUDA_SAFE_CALL(cudaEventCreate(&end0));
            CUDA_SAFE_CALL(cudaEventCreate(&end1));
            CUDA_SAFE_CALL(cudaEventCreate(&end2));
            CUDA_SAFE_CALL(cudaEventCreate(&end3));
            CUDA_SAFE_CALL(cudaEventCreate(&end4));
            CUDA_SAFE_CALL(cudaEventCreate(&e2b));
        }
        auto destroy_iteration_events = [&]()
        {
            if(!phase_time) return;
            CUDA_SAFE_CALL(cudaEventDestroy(start));
            CUDA_SAFE_CALL(cudaEventDestroy(end0));
            CUDA_SAFE_CALL(cudaEventDestroy(end1));
            CUDA_SAFE_CALL(cudaEventDestroy(end2));
            CUDA_SAFE_CALL(cudaEventDestroy(end3));
            CUDA_SAFE_CALL(cudaEventDestroy(end4));
            CUDA_SAFE_CALL(cudaEventDestroy(e2b));
        };

        //printf("\n\n\ncollision num  %d\n\n\n", h_cpNum[0]+h_gpNum);

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(start));
        g_dec_k = (int)k;   // [decouple probe] expose k to computeGradientAndHessian's stage dumps
        gipc_nvtx_push("GH_assembly");
        // [C-3 probe] STIFF_NEWTON_GRAPH=1: record THIS iteration's assembly
        // chain into a CUDA graph and launch it (record -> instantiate ->
        // launch -> destroy). Host bookkeeping runs at record time, so the
        // semantics are identical each iteration; what this buys today is a
        // machine-checked capturability proof of the whole GH chain with a
        // stage-marked fallback that names the first remaining blocker.
        // Cross-iteration exec reuse (the real win) lands once the ext/ABD
        // host bookkeeping is device-closed. Single-env merged gate as C-2.
        {
            static int s_ng = -1;
            if(s_ng < 0)
            { const char* e = getenv("STIFF_NEWTON_GRAPH"); s_ng = e ? atoi(e) : 0; }
            bool gh_done = false;
            // [C-3] the device snapshot seeds at the CALLSITE every iteration
            // (fresh mirrors + kappa) — never from inside the recorded graph.
            // Gated to the armed single-env merged population: per-env modes
            // have different mirror semantics (and the audit-armed gates read
            // trip on merged-formula mirror access there).
            static int s_c2 = -1;
            if(s_c2 < 0)
            { const char* e = getenv("STIFF_C2_OFFSET_DEV"); s_c2 = e ? atoi(e) : 0; }
            if((s_ng || s_c2) && !m_perenv_bvh && m_active_group_count <= 1)
                seed_gh_snapshot();
            if(s_ng && !m_perenv_bvh && m_active_group_count <= 1
               && m_total_frames >= 1)
            {
                // [C-3] the signature covers pointer generations AND the host
                // branch state baked into the recorded graph (the partition /
                // assembly paths branch on contact presence).
                const long long sig0 = (pcg_buffer_generation() << 2)
                                       | ((h_cpNum[0] > 0) ? 1 : 0)
                                       | ((h_gpNum > 0) ? 2 : 0);
                if(m_gh_graph_exec && m_gh_graph_sig != sig0)
                {
                    cudaGraphExecDestroy(m_gh_graph_exec);
                    m_gh_graph_exec = nullptr;
                }
                // [C-3] reuse is EXPERIMENTAL (=2): remaining baked host
                // branches inside GH (probe-enumerated so far: partition
                // segment converts, SPLIT_GH, pin blocks) make replays unsound
                // until each is device-closed. =1 keeps the validated
                // per-iteration record path.
                if(s_ng >= 2 && m_gh_graph_exec)
                {
                    // [C-3 reuse] REPLAY: shared host prologue (P1-dyn grow can
                    // bump the generation -> fall through to a re-record), then
                    // bookkeeping recompute via the observed FEM-tail length,
                    // then ONE graph launch replaces the whole assembly chain.
                    gh_pregrow();
                    if(((pcg_buffer_generation() << 2)
                        | ((h_cpNum[0] > 0) ? 1 : 0)
                        | ((h_gpNum > 0) ? 2 : 0)) != sig0)
                    {
                        cudaGraphExecDestroy(m_gh_graph_exec);
                        m_gh_graph_exec = nullptr;
                    }
                    else
                    {
                        const long long ct = gh_contact_total(nullptr, nullptr);
                        gipc_global_triplet.global_collision_triplet_offset = (int)ct;
                        gipc_global_triplet.global_triplet_offset = (int)(ct + m_gh_tail_len);
                        CUDA_SAFE_CALL(cudaGraphLaunch(m_gh_graph_exec, cudaStreamPerThread));
                        gh_done = true;
                        static bool oncer = false;
                        if(!oncer)
                        {
                            oncer = true;
                            printf("[gh-graph] cross-iteration REUSE active "
                                   "(assembly = one graph launch)\n");
                        }
                    }
                }
                if(!gh_done)
                {
                    cudaGraph_t gg    = nullptr;
                    bool        threw = false;
                    if(cudaStreamBeginCapture(cudaStreamPerThread,
                                              cudaStreamCaptureModeThreadLocal)
                       == cudaSuccess)
                    {
                        try
                        {
                            m_gh_recording          = true;
                            cuda_safe_call_throws() = true;
                            gipc_in_graph_capture() = true;
                            m_time_make_pd_ms += computeGradientAndHessian(TetMesh);
                            cuda_safe_call_throws() = false;
                            gipc_in_graph_capture() = false;
                        }
                        catch(const std::exception& ex)
                        {
                            cuda_safe_call_throws() = false;
                            gipc_in_graph_capture() = false;
                            threw = true;
                            s_ng  = 0;
                            cudaGraph_t junk = nullptr;
                            cudaStreamEndCapture(cudaStreamPerThread, &junk);
                            if(junk) cudaGraphDestroy(junk);
                            cudaGetLastError();
                            fprintf(stderr,
                                    "[gh-graph] assembly not capturable (%s) -> "
                                    "plain path for this session\n",
                                    ex.what());
                        }
                        m_gh_recording = false;
                        if(!threw
                           && cudaStreamEndCapture(cudaStreamPerThread, &gg) == cudaSuccess
                           && gg)
                        {
                            if(cudaGraphInstantiate(&m_gh_graph_exec, gg, nullptr, nullptr, 0)
                               == cudaSuccess && m_gh_graph_exec)
                            {
                                CUDA_SAFE_CALL(cudaGraphLaunch(m_gh_graph_exec,
                                                               cudaStreamPerThread));
                                gh_done = true;
                                // Record-time bookkeeping ran on the host: cache
                                // the generation + the observed FEM-tail length
                                // (offset - contact total, scene-constant for the
                                // gated no-ABD single-env population).
                                m_gh_graph_sig = (pcg_buffer_generation() << 2)
                                                 | ((h_cpNum[0] > 0) ? 1 : 0)
                                                 | ((h_gpNum > 0) ? 2 : 0);
                                m_gh_tail_len  = (long long)gipc_global_triplet.global_triplet_offset
                                                 - gh_contact_total(nullptr, nullptr);
                                static bool onceg = false;
                                if(!onceg)
                                {
                                    onceg = true;
                                    printf("[gh-graph] assembly chain captured + "
                                           "graph-launched (exec cached)\n");
                                }
                            }
                            cudaGraphDestroy(gg);
                        }
                        else if(!threw)
                            cudaGetLastError();
                    }
                    if(!gh_done && !threw)
                        s_ng = 0;   // capture path unavailable: permanent fallback
                    if(!gh_done)
                        m_time_make_pd_ms += computeGradientAndHessian(TetMesh);
                    gh_done = true;
                }
            }
            if(!gh_done)
                m_time_make_pd_ms += computeGradientAndHessian(TetMesh);
        }
        gipc_nvtx_pop();

        // [decouple probe] PRE-SOLVE gradient dump (shape_grads + fb hold the CLEAN gradient here,
        // before calculateMovingDirection clobbers shape_grads as scratch). frame STIFF_DUMP_FRAME,
        // any k if STIFF_PROBE_K unset → use k==0. Python compares env0 across batches.
        if(getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            std::vector<double3> hsh(vertexNum), hfb(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hfb.data(), TetMesh.fb, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PRE");
            FILE* a=fopen((std::string(fn)+".shape").c_str(),"wb"); if(a){fwrite(hsh.data(),sizeof(double3),vertexNum,a);fclose(a);}
            FILE* b=fopen((std::string(fn)+".fb").c_str(),"wb");    if(b){fwrite(hfb.data(),sizeof(double3),vertexNum,b);fclose(b);}
            FILE* c=fopen((std::string(fn)+".grp").c_str(),"wb");   if(c){fwrite(hp.data(),sizeof(int),vertexNum,c);fclose(c);}
            printf("[grad-pre] dumped shape+fb+grp @frame %d k=%d\n", s_dec_frame, (int)k);
        }

        const char* merged_diag_frame = getenv("STIFF_MERGED_DIAG_FRAME");
        const bool merged_diag_sample = merged_diag_frame
                                     && s_dec_frame == atoi(merged_diag_frame)
                                     && ((int)k < 20 || ((int)k % 10) == 0);
        double distToOpt_PN = DBL_MAX;

        // The merged path keeps its historical BVH-scene scale. Merely declaring body groups must
        // not change merged-mode convergence; group-local scales belong to the decoupled path only.
        double thr_bbox2 = bboxDiagSize2;   // legacy: whole scene (== the env for ungrouped scenes)
        // The scalar fallback used by the decoupled path is the average complete-environment scale;
        // normal freeze sites use each environment's own bbox. A physical velocity tolerance, when
        // configured, overrides both relative scales.
        if(m_mode_config.decouple_thresh
           && TetMesh.h_groups_present && m_avg_env_bbox2 > 0.0)
            thr_bbox2 = m_avg_env_bbox2;

        // [uipc-style opt-in] newton_velocity_tol>0: physical exit (max step displacement
        // <= v_tol*dt), scene-size/env-count independent, relative_dhat fully inert.
        double _newton_thr = (newton_velocity_tol > 0.0)
                                 ? (newton_velocity_tol * IPC_dt)
                                 : sqrt(Newton_solver_threshold * Newton_solver_threshold
                                        * thr_bbox2 * IPC_dt * IPC_dt);
        auto device_newton_converged = [&](bool retain_movement) {
            calcMinMovement_DeviceOut(_moveDir, pcg_data.squeue, vertexNum);
            _newton_convergence_decide<<<1, 1>>>(pcg_data.squeue,
                                                 _newton_thr,
                                                 m_newton_convergence_decision);
            int converged = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&converged,
                                      m_newton_convergence_decision,
                                      sizeof(int),
                                      cudaMemcpyDeviceToHost));
            if(retain_movement)
                CUDA_SAFE_CALL(cudaMemcpy(&distToOpt_PN,
                                          pcg_data.squeue,
                                          sizeof(double),
                                          cudaMemcpyDeviceToHost));
            return converged != 0;
        };
        bool gradVanish = current_global_exit
                              ? false
                              : device_newton_converged(merged_diag_sample);

        // [multi-env P3a step2] per-env Newton convergence tracking (precursor to
        // mask early-exit). The merged Newton loop currently breaks on the GLOBAL
        // move norm = the HARDEST env. Here we measure per-env move RMS each Newton
        // iter so we can see envs converge at DIFFERENT k (the early-exit premise,
        // on real merged-run data). Read-only diagnostic, gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && TetMesh.h_groups_present && getenv("STIFF_PENV_STATS"))
        {
            const int NG = active_group_count;
            double*& d_sq = m_scr_sq_a; int*& d_cnt = m_scr_cnt_a;
            if(!d_sq) { cudaMalloc((void**)&d_sq, kEnvAlphaSlots*sizeof(double));
                        cudaMalloc((void**)&d_cnt, kEnvAlphaSlots*sizeof(int)); }
            cudaMemset(d_sq, 0, NG*sizeof(double)); cudaMemset(d_cnt, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                              d_sq, d_cnt, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hs(NG); std::vector<int> hc(NG);
            cudaMemcpy(hs.data(), d_sq, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.data(), d_cnt, NG*sizeof(int), cudaMemcpyDeviceToHost);
            double thr = sqrt(Newton_solver_threshold * Newton_solver_threshold
                              * bboxDiagSize2 * IPC_dt * IPC_dt);
            // [S4 probe] per-env MAX move (the real Newton-exit metric), not RMS.
            double*& d_mxm = m_scr_mxm;
            if(!d_mxm) cudaMalloc((void**)&d_mxm, kEnvAlphaSlots*sizeof(double));
            cudaMemset(d_mxm, 0, NG*sizeof(double));
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mxm, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG);
            cudaMemcpy(hmm.data(), d_mxm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            printf("[P3a-newton] k=%d thr=%.3e per-env maxMove:", k, thr);
            int n_conv = 0, n_present = 0;
            for(int g = 0; g < NG; ++g) if(hc[g] > 0) {
                ++n_present;
                bool conv = (k && hmm[g] < thr);   // matches gradVanish (max move)
                if(conv) ++n_conv;
                printf(" g%d=%.2e%s", g, hmm[g], conv ? "*" : "");
            }
            printf("  (%d/%d done-by-maxmove)\n", n_conv, n_present);
        }

        // [multi-env S4] per-env active-mask DETECTION (foundation; the assembly/
        // PCG/SpMV skips read m_env_active). Mask env once its Newton max-move <
        // thr*margin; every RECHECK iters unmask ALL present envs + re-check
        // (catches non-monotonic bounce-back). Self-contained; gated STIFF_PERENV_MASK.
        // Skip detection on early Newton iters: nothing converges before ~k=MINK
        // (measured), so the per-iter D2H+sync overhead there is pure waste.
        if(m_env_active && TetMesh.d_point_to_group
           && m_mode_config.perenv_mask
           && !s4_dev_mask && k >= 4)   // [S4-dev] device-derived mask supersedes host detection
        {
            const int NG = active_group_count;
            const int RECHECK = 4;
            const double margin = 0.5;
            double thr = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));   // [decouple] batch-invariant; velocity_tol opt-in
            double*& d_mm = m_scr_mm; int*& d_ct = m_scr_ct;
            if(!d_mm) { cudaMalloc((void**)&d_mm, kEnvAlphaSlots*sizeof(double));
                        cudaMalloc((void**)&d_ct, kEnvAlphaSlots*sizeof(int)); }
            cudaMemset(d_mm, 0, NG*sizeof(double)); cudaMemset(d_ct, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mm, vertexNum, NG);
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, nullptr, nullptr, d_ct, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG); std::vector<int> hct(NG);
            cudaMemcpy(hmm.data(), d_mm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hct.data(), d_ct, NG*sizeof(int), cudaMemcpyDeviceToHost);
            const bool recheck = (k % RECHECK == 0);
            int n_active = 0, n_present = 0;
            for(int g = 0; g < NG; ++g)
            {
                if(hct[g] <= 0) { h_env_active[g] = 0; continue; }  // absent env
                ++n_present;
                if(recheck || k == 0) h_env_active[g] = 1;          // periodic full re-check
                else if(h_env_active[g]
                        && hmm[g] < ((newton_velocity_tol > 0.0 || h_env_bbox2.empty() || h_env_bbox2[g] <= 0.0)
                                         ? thr
                                         : Newton_solver_threshold * IPC_dt * sqrt(h_env_bbox2[g]))
                                        * margin)   // [env-scale] own-bbox; deeply converged -> mask
                    h_env_active[g] = 0;
                if(h_env_active[g]) ++n_active;
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                      NG * sizeof(int), cudaMemcpyHostToDevice));
            if(getenv("STIFF_PENV_STATS"))
                printf("[S4-mask] k=%d active=%d/%d%s\n", k, n_active, n_present,
                       recheck ? " (recheck)" : "");
        }

        //double distToOpt_PN = calcMinMovement(TetMesh.totalForce, pcg_data.squeue, vertexNum);
        //printf("disToopt:  %f        %f\n",
        //       distToOpt_PN,
        //       2 * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //           * IPC_dt * IPC_dt);

        //bool gradVanish =
        //    (distToOpt_PN < 1
        //                        * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //                        * IPC_dt * IPC_dt);

        // [decouple] DECOUPLE_THRESH: exit only when ALL envs are per-env frozen (each converged),
        // NOT on the global gradVanish (which can cut off an under-converged env when its mates
        // finish first → batch-dependent final state). all_env_frozen is from the prev iter's S1
        // Phase B. Baseline (off) keeps the global gradVanish exit.
        // [robustness] the frozen-exit needs the S1 per-env-alpha machinery to actually run
        // (m_env_alpha_valid, set by the prev iter's S1). With DECOUPLE_THRESH but WITHOUT
        // STIFF_PERENV_ALPHA, all_env_frozen stays false forever → the loop ran to iterCap every
        // frame (pathological, found by the flag ablation). Fall back to gradVanish in that case.
        bool do_break = !current_global_exit
                     && ((m_mode_config.decouple_thresh && m_env_alpha_valid)
                             ? (k && all_env_frozen)
                             : (k && gradVanish));
        // [drive-substep] no convergence exit until the driving ramp completes
        // (uipc: animation_reach_target gates convergence_check).
        do_break = do_break && drive_ratio >= 1.0;
        if(do_break)
        {
            destroy_iteration_events();
            break;
        }
        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end0));

        gipc_nvtx_push("linear_solve");
        auto cg_count = calculateMovingDirection(TetMesh, h_cpNum[0], pcg_data.P_type);
        gipc_nvtx_pop();
        //std::cout << "[" << k << "]"
        //          << "cg_count = " << cg_count << std::endl;
        m_total_pcg_iters += cg_count;

        // [iron-law completion] make quarantined envs fully INERT before any
        // downstream consumer of this iteration's direction. The CCD alpha
        // kernels flag !isfinite(dot(n, dir)) into the shared invalid mask and
        // the device-chain validation would THROW — killing healthy envs.
        // (1) scan the fresh PCG direction per env: a naturally-diverging env
        //     is quarantined HERE, before the CCD chain trips on its NaNs
        //     (the NaN max-move quarantine in the freeze loop runs AFTER the
        //     CCD chain — too late for the throw);
        // (2) zero every quarantined env's direction: positions stay frozen
        //     (alpha==0 keeps temp verbatim), CCD candidates go neutral, the
        //     global convergence norm no longer sees the dead env.
        // Same gate as the NaN/timeout quarantine (host telemetry path);
        // the pure-device fast path is the documented contract: isolation
        // promises require per_env_exit / STIFF_PERENV_TELEM.
        {
            const bool quar_gate = m_d_p2g && perEnvIsolationLive();
            if(quar_gate)
            {
                const int NGq = m_active_group_count;
                if(!m_d_env_dirnan)
                    m_d_env_dirnan.resize_discard(kEnvAlphaSlots);  // [3d-2]
                CUDA_SAFE_CALL(cudaMemsetAsync(m_d_env_dirnan, 0, NGq * sizeof(int), 0));
                {
                    int bs = 256, gs = ((int)vertexNum + bs - 1) / bs;
                    _scan_dir_nonfinite<<<gs, bs>>>(_moveDir, m_d_p2g,
                                                    m_d_env_dirnan, (int)vertexNum);
                }
                std::vector<int> hnan(NGq);
                CUDA_SAFE_CALL(cudaMemcpy(hnan.data(), m_d_env_dirnan,
                                          NGq * sizeof(int), cudaMemcpyDeviceToHost));
                for(int g = 0; g < NGq; ++g)
                    if(hnan[g] && (m_env_quarantined.empty() || !m_env_quarantined[g]))
                        quarantineEnv(g, -1, 0.0);
                bool any_quar = false;
                if(!m_env_quarantined.empty())
                    for(int g = 0; g < NGq; ++g)
                        any_quar |= (m_env_quarantined[g] != 0);
                if(any_quar && m_d_env_quarantined)
                {
                    int bs = 256, gs = ((int)vertexNum + bs - 1) / bs;
                    _zero_dir_quarantined<<<gs, bs>>>(_moveDir, m_d_p2g,
                                                      m_d_env_quarantined, (int)vertexNum);
                }
            }
        }
        if(current_global_exit)
        {
            gradVanish = device_newton_converged(merged_diag_sample);
            if(k && gradVanish && drive_ratio >= 1.0)
            {
                destroy_iteration_events();
                break;
            }
        }
        // [decouple probe] full-precision moveDir dump (engine order) + d_point_to_group (engine
        // order, aligned). At frame STIFF_DUMP_FRAME, Newton iter STIFF_PROBE_K. Python masks
        // verts[grp==g] and compares matesA vs matesB to find the fine per-step coupling seed.
        if(getenv("STIFF_GRAD_PROBE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 1))
        {
            std::vector<double3> hmd(vertexNum), hgr(vertexNum), hsh(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hmd.data(), _moveDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hgr.data(), TetMesh.fb, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // contact+ground grad
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // kinetic+elastic
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PROBE");
            FILE* f = fopen(fn, "wb"); if(f){ fwrite(hmd.data(), sizeof(double3), vertexNum, f); fclose(f); }
            std::string gradn = std::string(fn) + ".grad";
            FILE* fg = fopen(gradn.c_str(), "wb"); if(fg){ fwrite(hgr.data(), sizeof(double3), vertexNum, fg); fclose(fg); }
            std::string shn = std::string(fn) + ".shape";
            FILE* fs = fopen(shn.c_str(), "wb"); if(fs){ fwrite(hsh.data(), sizeof(double3), vertexNum, fs); fclose(fs); }
            std::string gn = std::string(fn) + ".grp";
            FILE* g = fopen(gn.c_str(), "wb"); if(g){ fwrite(hp.data(), sizeof(int), vertexNum, g); fclose(g); }
            printf("[grad-probe] dumped moveDir+grad+grp @frame %d k=%d (%d verts) -> %s\n",
                   s_dec_frame, (int)k, vertexNum, fn);
            // [decouple] ABD body pose dump (verify the gripper/arm pose drifts across batches → the
            // FEM-pin seed). m_d_abd_body_q = Vector12 (12 doubles) per body; d_body_to_group = env id.
            if(getenv("STIFF_ABD_DUMP") && m_abd_sim_data && TetMesh.d_body_to_group)
            {
                int nb = (int)abd_fem_count_info.abd_body_num;
                const double* qptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_q.data());
                const double* dqptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_dq.data());
                std::vector<double> hq(12 * nb), hdq(12 * nb); std::vector<int> hbg(nb);
                CUDA_SAFE_CALL(cudaMemcpy(hq.data(), qptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(hdq.data(), dqptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                FILE* dq=fopen((std::string(fn)+".abddq").c_str(),"wb"); if(dq){fwrite(hdq.data(),sizeof(double),12*nb,dq);fclose(dq);}
                CUDA_SAFE_CALL(cudaMemcpy(hbg.data(), TetMesh.d_body_to_group, nb * sizeof(int), cudaMemcpyDeviceToHost));
                FILE* q=fopen((std::string(fn)+".abdq").c_str(),"wb"); if(q){fwrite(hq.data(),sizeof(double),12*nb,q);fclose(q);}
                FILE* b=fopen((std::string(fn)+".abdg").c_str(),"wb"); if(b){fwrite(hbg.data(),sizeof(int),nb,b);fclose(b);}
                printf("[abd-dump] %d bodies @frame %d k=%d\n", nb, s_dec_frame, (int)k);
            }
        }
        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end1));
        double alpha = 1.0, slackness_a = 0.9, slackness_m = 0.8;
        double diag_ground_alpha = 1.0;
        double diag_narrow_alpha = 1.0;
        double diag_refined_alpha = 1.0;
        int    diag_narrow_pairs = h_cpNum[0];
        bool   diag_refine_used = false;

        // Keep the complete scalar CCD chain on device. The first two
        // reductions feed temp_alpha directly into swept BVH construction.
        const bool g_did = !m_skip_all_collision && surf_vertexNum >= 1;
        // Gate/count from the stable DCD snapshot, not the live CCD buffer.
        const bool s_did = !m_skip_all_collision && m_dcd_snap_count >= 1;
        gipc_nvtx_push("ccd_alpha");
        CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_alpha_invalid, 0, sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(
            m_ccd_refined_invalid, 0, (1 + kEnvAlphaSlots) * sizeof(int)));
        if(g_did)
            ground_largestFeasibleStepSize_DeviceOut(
                slackness_a, pcg_data.squeue, m_ccd_alpha_slots + 0);
        if(s_did)
            self_largestFeasibleStepSize_DeviceOut(
                slackness_m,
                ensure_reduce_scratch(m_dcd_snap_count),
                m_dcd_snap_count,
                m_ccd_alpha_slots + 1);
        _ccd_initial_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                             g_did ? 1 : 0,
                                             s_did ? 1 : 0,
                                             m_ccd_alpha_invalid);
        //alpha = std::min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        double temp_alpha = 1.0;
        double alpha_CFL  = 1.0;

        double ccd_size = 1.0;
        //#ifdef USE_FRICTION
        //        ccd_size = 0.6;
        //#endif

        // [multi-env S1 Phase A] per-env temp_alpha terms (ground + narrow-self),
        // computed HERE because buildFullCP below overwrites _ccd_collisonPairs.
        // Mirrors the engine's temp_alpha reductions (lines above) but per-env.
        // Regions: m_env_scratch[0*NG]=ground alpha, [1*NG]=narrow-self alpha.
        m_env_alpha_valid = false;  // reset each Newton iter; S1 sets true below
        const bool s1_on = (m_env_scratch && TetMesh.d_point_to_group
                            // [N=1 guard] all -1 p2g = zero env coverage: S1 would flag itself
                            // valid, all_env_frozen unreachable -> Newton pegs at iterCap.
                            && TetMesh.h_groups_present
                            && surf_vertexNum >= 1 && !m_skip_all_collision
                            && m_mode_config.perenv_alpha);
        if(s1_on)
        {
            const int NG = active_group_count, bs = 256;
            _fill_double<<<(2 * NG + bs - 1) / bs, bs>>>(
                m_env_scratch, 1.0, 2 * NG);
            _per_env_groundAlpha_min<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir,
                TetMesh.d_point_to_group, m_env_scratch + 0*NG, slackness_a,
                surf_vertexNum, _point_body_id, _ground_skip_body, _ground_body_count, NG,
                m_ccd_alpha_invalid);
            // [narrow-self snapshot] sweep the FULL DCD-time snapshot (stable
            // content, env-balanced by construction) — never the live CCD buffer,
            // whose prefix is a race-ordered slice of last iteration's swept
            // emission (the strict cross-env asymmetry root cause).
            if(m_dcd_snap_count >= 1)
                _per_env_selfAlpha_min<<<(m_dcd_snap_count+bs-1)/bs, bs>>>(
                    _vertexes, _dcd_ccd_snapshot, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 1*NG, slackness_m, m_dcd_snap_count, NG,
                    m_mode_config.ccd_canon ? m_d_vloc : nullptr,
                    m_ccd_alpha_invalid,
                    kCcdInvalidPerEnvNarrow,
                    nullptr);
        }

        // [B3 ccd-defer] merged path: the count refresh inside buildFullCP is
        // deferred; count + past-capacity overflow signal ride the scalar-chain
        // read below. Per-env keeps the legacy immediate refresh.
        const bool ccd_defer =
            !m_skip_all_collision
            && !(m_perenv_bvh && m_d_p2g && m_perenv_bvh_groups > 0);
        m_ccd_defer_counts = ccd_defer;
        buildBVH_FULLCCD(1.0, m_ccd_alpha_slots + 2);
        buildFullCP(1.0, m_ccd_alpha_slots + 2);
        m_ccd_defer_counts = false;
        int ccd_cnt = 0;
        if(ccd_defer)
        {
            // Capacity grid + in-kernel live mask: min is exact, so regridding
            // is bitwise-neutral; count==0 degenerates to identity everywhere
            // and the combine gate (device-read) leaves it unconsumed.
            cfl_largestSpeed_DeviceOut(pcg_data.squeue, m_ccd_alpha_slots + 3);
            self_full_largestFeasibleStepSize_DeviceOut(
                slackness_m,
                ensure_reduce_scratch(MAX_CCD_COLLITION_PAIRS_NUM),
                MAX_CCD_COLLITION_PAIRS_NUM,
                m_ccd_alpha_slots + 4,
                _cpNum);
            _ccd_final_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                               0,
                                               dHat,
                                               ccd_size,
                                               m_ccd_alpha_invalid,
                                               m_ccd_refined_invalid,
                                               _cpNum);
        }
        else
        {
            ccd_cnt = (int)h_ccd_cpNum;
            if(ccd_cnt > 0)
            {
                cfl_largestSpeed_DeviceOut(pcg_data.squeue, m_ccd_alpha_slots + 3);
                // Launch the refined reduction unconditionally. Its raw invalid
                // status becomes effective only if the exact refinement gate fires.
                self_full_largestFeasibleStepSize_DeviceOut(
                    slackness_m,
                    ensure_reduce_scratch(ccd_cnt),
                    ccd_cnt,
                    m_ccd_alpha_slots + 4);
            }
            _ccd_final_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                               ccd_cnt > 0 ? 1 : 0,
                                               dHat,
                                               ccd_size,
                                               m_ccd_alpha_invalid,
                                               m_ccd_refined_invalid,
                                               nullptr);
        }

        // One scalar-chain D2H after every global decision and validation bit.
        double h_ccd_state[9] = {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0};
        CUDA_SAFE_CALL(cudaMemcpy(h_ccd_state,
                                  m_ccd_alpha_slots,
                                  sizeof(h_ccd_state),
                                  cudaMemcpyDeviceToHost));
        if(ccd_defer)
        {
            ccd_cnt = (int)h_ccd_state[8];
            if(ccd_cnt > MAX_CCD_COLLITION_PAIRS_NUM)
            {
                // Overflow: emits past capacity went to the trash slot, so the
                // deferred reduction saw a subset. Fall back to the legacy
                // machinery (refresh + grow + redo detection), recompute the
                // refined term over the full set, and re-read. Rare by design;
                // the discarded subset alpha was never consumed.
                buildFullCP(1.0, m_ccd_alpha_slots + 2);
                ccd_cnt = (int)h_ccd_cpNum;
                if(ccd_cnt > 0)
                    self_full_largestFeasibleStepSize_DeviceOut(
                        slackness_m,
                        ensure_reduce_scratch(ccd_cnt),
                        ccd_cnt,
                        m_ccd_alpha_slots + 4);
                _ccd_final_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                                   ccd_cnt > 0 ? 1 : 0,
                                                   dHat,
                                                   ccd_size,
                                                   m_ccd_alpha_invalid,
                                                   m_ccd_refined_invalid,
                                                   nullptr);
                CUDA_SAFE_CALL(cudaMemcpy(h_ccd_state,
                                          m_ccd_alpha_slots,
                                          sizeof(h_ccd_state),
                                          cudaMemcpyDeviceToHost));
            }
        }
        validateFinalCcdStateOrThrow(h_ccd_state, "device CCD chain");
        if(getenv("STIFF_CCD_VALIDATE"))
        {
            const double host_temp = h_ccd_state[0] < h_ccd_state[1]
                                         ? h_ccd_state[0]
                                         : h_ccd_state[1];
            double host_cfl   = host_temp;
            double host_alpha = host_temp;
            if(ccd_cnt > 0)
            {
                host_cfl = sqrt(dHat) / h_ccd_state[3] * 0.5;
                host_alpha = host_temp < host_cfl ? host_temp : host_cfl;
                if(host_temp > 2.0 * host_cfl)
                {
                    const double refined = h_ccd_state[4] * ccd_size;
                    host_alpha = host_temp < refined ? host_temp : refined;
                    host_alpha = host_alpha > host_cfl ? host_alpha : host_cfl;
                }
            }
            const bool temp_exact = std::memcmp(
                &host_temp, &h_ccd_state[2], sizeof(double)) == 0;
            const bool cfl_exact = std::memcmp(
                &host_cfl, &h_ccd_state[6], sizeof(double)) == 0;
            const bool alpha_exact = std::memcmp(
                &host_alpha, &h_ccd_state[5], sizeof(double)) == 0;
            if(!temp_exact || !cfl_exact || !alpha_exact)
                throw std::runtime_error(
                    "[CCD] device alpha chain differs from host formula");
            static bool first_ccd_match = true;
            if(first_ccd_match)
            {
                printf("[ccd-device-validate] temp/CFL/final alpha exact\n");
                first_ccd_match = false;
            }
        }
        diag_ground_alpha  = h_ccd_state[0];
        diag_narrow_alpha  = h_ccd_state[1];
        temp_alpha         = h_ccd_state[2];
        diag_refined_alpha = h_ccd_state[4];
        alpha              = h_ccd_state[5];
        alpha_CFL          = h_ccd_state[6];
        diag_refine_used   = ccd_cnt > 0 && temp_alpha > 2.0 * alpha_CFL;
        gipc_nvtx_pop();

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end2));
        //printf("alpha:  %f\n", alpha);

        // [multi-env P3a] read-only: per-env alpha_CFL spread. The global alpha
        // above is ONE scalar (= min over ALL envs of CCD/CFL feasible step). If
        // per-env maxSpeed differs a lot, the global alpha is dragged by the
        // fastest env -> slower envs forced to over-small steps -> per-env
        // line-search would decouple them. Decides whether per-env alpha is worth
        // building. Gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && TetMesh.h_groups_present
           && surf_vertexNum >= 1 && getenv("STIFF_PENV_STATS"))
        {
            const int NG = active_group_count;
            double*& d_mx = m_scr_mx;
            if(!d_mx) cudaMalloc((void**)&d_mx, kEnvAlphaSlots * sizeof(double));
            cudaMemset(d_mx, 0, NG * sizeof(double));
            int bs = 256, gs = (surf_vertexNum + bs - 1) / bs;
            _per_env_max_cfl<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                         _surfVerts, d_mx, surf_vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmx(NG);
            cudaMemcpy(hmx.data(), d_mx, NG * sizeof(double), cudaMemcpyDeviceToHost);
            double sq = sqrt(dHat);
            printf("[P3a-cfl] k=%d global_alpha=%.3e per-env alpha_CFL=", k, alpha);
            for(int g = 0; g < NG; ++g) if(hmx[g] > 0.0)
                printf(" g%d=%.3e", g, sq / hmx[g] * 0.5);
            printf("\n");
        }

        // [multi-env S1 Phase B] per-env feasible-alpha substrate (PHYSICS-NEUTRAL).
        // Phase A filled ground+narrow-self; here add refined-self (over the NEW
        // _ccd_collisonPairs) + CFL, then combine per the engine's exact logic:
        //   temp_alpha_env = min(1, ground_env, narrowSelf_env)
        //   if ccd pairs: alpha_env = min(temp_alpha_env, alpha_CFL_env);
        //                 if temp_alpha_env > 2*alpha_CFL_env:
        //                     alpha_env = max(min(temp_alpha_env, refinedSelf_env*ccd_size), alpha_CFL_env)
        // Each CCD term is a direct per-env MIN(alpha), matching the global path.
        // This per-env substrate does not alter the global scalar `alpha`;
        // it only fills the values S2 consumes. Invariant
        // (validated): min_g(m_env_alpha) == global feasibility alpha; N=1 -> one
        // group -> m_env_alpha[g0] == global alpha.
        if(s1_on)
        {
            const int NG = active_group_count, bs = 256;
            _fill_double<<<(NG + bs - 1) / bs, bs>>>(
                m_env_scratch + 2*NG, 1.0, NG);  // refined-alpha region
            cudaMemset(m_env_scratch + 3*NG, 0, 2 * NG * sizeof(double));  // cfl + Newton max regions
            if(ccd_cnt > 0)  // refined-self over NEW _ccd_collisonPairs[0..ccd_cnt)
                _per_env_selfAlpha_min<<<(ccd_cnt+bs-1)/bs, bs>>>(
                    _vertexes, _ccd_collisonPairs, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 2*NG, slackness_m, ccd_cnt, NG,
                    m_mode_config.ccd_canon ? m_d_vloc : nullptr,
                    m_ccd_alpha_invalid,
                    kCcdInvalidPerEnvRefined,
                    m_ccd_refined_invalid + 1);
            _per_env_max_cfl<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                TetMesh.d_point_to_group, _moveDir, _surfVerts, m_env_scratch + 3*NG,
                surf_vertexNum, NG);
            _per_env_max_move<<<(vertexNum+bs-1)/bs, bs>>>(
                TetMesh.d_point_to_group, _moveDir, m_env_scratch + 4*NG,
                vertexNum, NG);
            // [perf] DEVICE-SIDE per-env alpha + freeze (no cudaDeviceSynchronize, no 5x256 D2H, no
            // host loop, no H2D) — writes m_env_alpha directly + one 3-int D2H
            // (two freeze counters and the already-needed CCD status word).
            // Bit-identical to the host loop (same per-env formulas). Host path kept only under a
            // diagnostic flag.
            const double _sq_    = sqrt(dHat);
            const double _thrcv_ = m_mode_config.decouple_thresh ? ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt)) : 0.0;
            const bool _s1diag_ = getenv("STIFF_PENV_STATS") || getenv("STIFF_A0_DUMP")
                               || getenv("STIFF_S1_DEBUG") || getenv("STIFF_ALPHA_DBG")
                               // [per-env productization] telemetry (freeze iters/
                               // status), the per-env iter budget and the NaN
                               // quarantine live in the host loop — route there
                               // when any of them is requested.
                               || env_newton_iter_cap > 0
                               || m_mode_config.perenv_telem;
            if(!_s1diag_)
            {
                int*& d_env_cnt = m_scr_env_cnt;
                if(!d_env_cnt)
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_env_cnt, 3 * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemsetAsync(d_env_cnt, 0, 3 * sizeof(int)));
                _per_env_alpha_compute<<<(NG + bs - 1) / bs, bs>>>(
                    m_env_alpha, m_env_scratch, NG, _sq_, 1.0, (ccd_cnt > 0) ? 1 : 0,
                    temp_alpha, alpha_CFL, m_mode_config.decouple_thresh ? 1 : 0,
                    getenv("STIFF_NO_REFINE") ? 1 : 0, _thrcv_,
                    d_env_bbox2, Newton_solver_threshold * IPC_dt, newton_velocity_tol * IPC_dt,
                    m_ccd_refined_invalid + 1,
                    m_ccd_alpha_invalid,
                    d_env_cnt);
                int _hc_[3];
                CUDA_SAFE_CALL(cudaMemcpy(
                    _hc_, d_env_cnt, 3 * sizeof(int), cudaMemcpyDeviceToHost));
                throwForInvalidCcdMask(_hc_[2], "per-env CCD reduction");
                m_env_alpha_valid = true;
                all_env_frozen    = (_hc_[0] > 0 && _hc_[1] == _hc_[0]);
            }
            else
            {
            _promote_per_env_refined_invalid<<<(NG + bs - 1) / bs, bs>>>(
                m_env_scratch,
                NG,
                (ccd_cnt > 0) ? 1 : 0,
                _sq_,
                temp_alpha,
                alpha_CFL,
                m_mode_config.decouple_thresh ? 1 : 0,
                getenv("STIFF_NO_REFINE") ? 1 : 0,
                m_ccd_refined_invalid + 1,
                m_ccd_alpha_invalid);
            cudaDeviceSynchronize();
            throwIfInvalidCcdAlpha("per-env CCD reduction");
            std::vector<double> hg(NG), hs(NG), hr(NG), hmx(NG), hnm(NG);
            // [semi-implicit beta timing fix] update beta_g from the PREVIOUS
            // iteration's ACCEPTED per-env alpha (m_env_alpha still holds it —
            // this S1 pass overwrites it further below), not from this
            // iteration's pre-line-search candidate. Candidate-based decay
            // overestimated progress whenever backtracking halved the step
            // (audit leftover #5). Threshold crossing only arms a freeze flag
            // consumed in the loop below (freeze-next semantics).
            static std::vector<char> semi_freeze_next;
            if(semi_implicit_enabled && (int)k >= semi_implicit_min_iter + 1)
            {
                std::vector<double> h_acc(NG);
                cudaMemcpy(h_acc.data(), m_env_alpha, NG*sizeof(double), cudaMemcpyDeviceToHost);
                if((int)semi_freeze_next.size() < NG) semi_freeze_next.assign(NG, 0);
                for(int g = 0; g < NG; ++g)
                {
                    if(m_env_status[g] != 0) continue;         // already frozen/diverged
                    semi_beta_env[g] *= std::max(0.0, 1.0 - h_acc[g]);
                    if(semi_beta_env[g] <= semi_implicit_beta_tol)
                        semi_freeze_next[g] = 1;
                }
            }
            else if((int)semi_freeze_next.size() < NG)
                semi_freeze_next.assign(NG, 0);
            if((int)k == 0) std::fill(semi_freeze_next.begin(), semi_freeze_next.end(), 0);
            cudaMemcpy(hg.data(),  m_env_scratch + 0*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(),  m_env_scratch + 1*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hr.data(),  m_env_scratch + 2*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hmx.data(), m_env_scratch + 3*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hnm.data(), m_env_scratch + 4*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            const double sq       = sqrt(dHat);
            const double ccd_size = 1.0;
            const bool   have_ccd = (ccd_cnt > 0);
            double       min_env  = 1e30, max_env = 0.0;
            int          n_env    = 0, n_frozen = 0;
            for(int g = 0; g < NG; ++g)
            {
                bool present = hnm[g] > 0.0
                            || (!h_env_bbox2.empty() && h_env_bbox2[g] > 0.0);
                if(!present) continue;
                double ta = std::min(hg[g], hs[g]);  // direct temp_alpha_env
                double a = ta;
                if(have_ccd && hmx[g] > 0.0)
                {
                    double acfl = sq / hmx[g] * 0.5;
                    a = std::min(ta, acfl);
                    // The refinement GATE: the engine gates on the GLOBAL temp_alpha/alpha_CFL,
                    // which makes env_0's branch decision (enter refined CCD or not) depend on the
                    // MATES (global temp_alpha = min over all envs) → BREAKS batch-invariance: at a
                    // frame where global temp_alpha diverges across batches, env_0 enters refinement
                    // in one batch but not the other → env_0's feasible alpha differs → drift →
                    // chaos amplifies. STIFF_DECOUPLE_THRESH gates per-env (ta_g vs acfl_g) so env_0's
                    // branch depends only on env_0 (batch-invariant). Off → exact engine behavior.
                    // [decouple TEST] STIFF_NO_REFINE: skip refinement entirely (env0's a=min(ta,acfl)
                    // → fully per-env/batch-invariant; hr — built from global-temp_alpha CCD pairs — is
                    // the confirmed last leak). isIntersected safety net in lineSearch catches any
                    // resulting penetration. Used to verify hr is the only remaining batch-coupling.
                    double gate_lhs = m_mode_config.decouple_thresh ? ta   : temp_alpha;
                    double gate_rhs = m_mode_config.decouple_thresh ? acfl : alpha_CFL;
                    if(!getenv("STIFF_NO_REFINE") && gate_lhs > 2.0 * gate_rhs)
                    {
                        a = std::min(ta, hr[g] * ccd_size);
                        a = std::max(a, acfl);
                    }
                }
                h_env_alpha[g] = a;
                // [decouple] FREEZE env g once IT has converged (per-env max-move < the per-env Newton
                // threshold), so env g stops stepping at ITS OWN convergence iter — NOT the global loop
                // count. The merged Newton loop runs a BATCH-DEPENDENT number of iters (harder mates →
                // more iters: measured A=13 vs B=16 at frame0); during the extra iters an already-
                // converged env's ABD (dq small but nonzero) keeps stepping → its gripper/arm pose
                // drifts batch-dependently → FEM-pin seed. Freezing at the env's own convergence makes
                // env g's total steps batch-invariant. Block-diagonal per-env solve ⇒ monotonic ⇒ no
                // re-activation needed. Gated STIFF_DECOUPLE_THRESH.
                if(m_mode_config.decouple_thresh)
                {
                    double thr_cv = (newton_velocity_tol > 0.0)
                        ? (newton_velocity_tol * IPC_dt)
                        : ((!h_env_bbox2.empty() && h_env_bbox2[g] > 0.0)
                               ? Newton_solver_threshold * IPC_dt * sqrt(h_env_bbox2[g])   // [env-scale] own bbox
                               : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    if(hnm[g] < thr_cv) h_env_alpha[g] = 0.0;
                }
                // [per-env productization] quarantine + budget, evaluated per iter
                // (no latch needed: NaN persists, k only grows).
                if(!m_env_quarantined.empty() && g < (int)m_env_quarantined.size()
                   && m_env_quarantined[g])
                {   // [iron-law] persistently quarantined env: pin it in EVERY
                    // iteration of EVERY later solve. Checked OUTSIDE the
                    // alpha!=0 gate — the direction-zero pass makes a
                    // quarantined env's max-move 0, so the convergence freeze
                    // above zeroes its alpha first and would otherwise mislabel
                    // it status 1 (converged); status 3 must win the telemetry.
                    h_env_alpha[g] = 0.0;
                    if(m_env_status[g] != 3)
                    {
                        m_env_status[g] = 3;
                        if(m_env_frozen_iter[g] < 0)
                            m_env_frozen_iter[g] = (int)k;
                    }
                }
                else if(h_env_alpha[g] != 0.0)
                {
                    if(std::isnan(hnm[g]) || std::isinf(hnm[g]))
                    {   // diverged env: freeze it so its NaN cannot poison the batch
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] < 2)
                        {
                            m_env_status[g]      = 3;
                            m_env_frozen_iter[g] = (int)k;
                            printf("  [per-env] env %d DIVERGED (NaN/inf max-move) at iter %d -> frozen\n",
                                   g, (int)k);
                        }
                    }
                    else if(env_newton_iter_cap > 0 && (int)k + 1 >= env_newton_iter_cap)
                    {   // per-env iteration budget: give up on THIS env only
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] < 2)
                        {
                            m_env_status[g]      = 2;
                            m_env_frozen_iter[g] = (int)k;
                            printf("  [per-env] env %d TIMEOUT (cap=%d) at iter %d -> frozen\n",
                                   g, env_newton_iter_cap, (int)k);
                        }
                    }
                    else if(semi_implicit_enabled && semi_freeze_next[g])
                    {   // [semi-implicit per-env, beta timing fix] the freeze flag
                        // was armed from the PREVIOUS iteration's accepted alpha
                        // (see the beta update above the loop); consume it here.
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] == 0)
                        {
                            m_env_status[g]      = 1;   // converged (semi-implicit accept)
                            m_env_frozen_iter[g] = (int)k;
                        }
                    }
                }
                if(h_env_alpha[g] == 0.0) ++n_frozen;   // [decouple] per-env converged (frozen)
                if(h_env_alpha[g] == 0.0 && m_env_frozen_iter[g] < 0)
                {   // [per-env productization] first freeze = convergence iter
                    m_env_frozen_iter[g] = (int)k;
                    if(m_env_status[g] == 0) m_env_status[g] = 1;
                }
                min_env = std::min(min_env, a);
                max_env = std::max(max_env, a);
                ++n_env;
                if(getenv("STIFF_A0_DUMP") && g == 0
                   && (!getenv("STIFF_DUMP_FRAME") || g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))))
                {   // [decouple] env0 per-iter: applied alpha (h_env_alpha[g], post-freeze) + hmx (max-move) + frozen?
                    double thr_cv = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    printf("[a0] frame=%d k=%d a_applied=%.17e a_feasible=%.17e "
                           "newtonMax=%.17e cflMax=%.17e thr_cv=%.6e frozen=%d\n",
                           g_dec_frame, (int)k, h_env_alpha[g], a, hnm[g], hmx[g],
                           thr_cv, (int)(h_env_alpha[g] == 0.0));
                }
                if(getenv("STIFF_S1_DEBUG") && alpha > 0.99 && a < 0.9)
                    printf("  [S1-dbg] g%d a=%.4e ta=%.4e ground=%.4e narrow=%.4e "
                           "refined=%.4e acfl=%.4e ccdN=%d alphaCFLglob=%.4e (global=%.4e)\n",
                           g, a, ta, hg[g], hs[g], hr[g], have_ccd?sq/hmx[g]*0.5:9.99,
                           ccd_cnt, alpha_CFL, alpha);
            }
            // [env-det dbg] cross-env alpha mismatch (STIFF_ALPHA_DBG): pin the component that differs.
            if(getenv("STIFF_ALPHA_DBG") && NG >= 2 && hmx[0] > 0.0 && hmx[1] > 0.0
               && (h_env_alpha[0] != h_env_alpha[1] || hg[0] != hg[1] || hs[0] != hs[1]
                   || hr[0] != hr[1] || hmx[0] != hmx[1]))
            {
                static int _ad = 0;
                if(_ad++ < 12)
                    printf("[alpha-dbg] a0=%.17e a1=%.17e | hg %.17e/%.17e hs %.17e/%.17e "
                           "hr %.17e/%.17e hmx %.17e/%.17e ccdN=%d\n",
                           h_env_alpha[0], h_env_alpha[1], hg[0], hg[1], hs[0], hs[1],
                           hr[0], hr[1], hmx[0], hmx[1], ccd_cnt);
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_alpha, h_env_alpha.data(),
                                      NG * sizeof(double), cudaMemcpyHostToDevice));
            m_env_alpha_valid = true;  // m_env_alpha fresh -> lineSearch may use it
            // [decouple] all present envs per-env frozen (converged)? → drives the loop-exit override
            // so the loop runs until env_0 (and every env) reaches ITS OWN convergence, batch-independent.
            all_env_frozen = (n_env > 0 && n_frozen == n_env);
            if(getenv("STIFF_PENV_STATS"))
                printf("[S1-envalpha] k=%d global_alpha=%.6e min_env=%.6e max_env=%.6e "
                       "n_env=%d rel=%.2e (neutral-check; headroom=max_env/global)\n",
                       k, alpha, min_env, max_env, n_env,
                       fabs(alpha - min_env) / std::max(alpha, 1e-30));
            }   // [perf] end diagnostic host path
        }

        // The current move direction has just been tested against every environment's Newton
        // threshold. Exit immediately, exactly like the merged path, rather than performing a
        // full zero-active assembly/solve on the next iteration.
        if(m_mode_config.decouple_thresh && m_env_alpha_valid
           && all_env_frozen && k && drive_ratio >= 1.0)
        {
            destroy_iteration_events();
            break;
        }

        // [S4-dev] derive next iter's active mask from the freeze decision already on device
        // (m_env_alpha == 0 ⇔ env converged this iter). Zero D2H — replaces the S4 host detection.
        // Frozen set is final here: the S3 per-env backtrack only halves nonzero alphas (never → 0).
        if(s4_dev_mask && m_env_alpha_valid)
            _mask_from_env_alpha<<<(active_group_count + 255) / 256, 256>>>(
                m_env_active, m_env_alpha, active_group_count);

        if(phase_time)
            CUDA_SAFE_CALL(cudaEventRecord(e2b));  // end S1 per-env-alpha / start lineSearch
        double alpha_before_line_search = alpha;
        if(merged_diag_sample)
        {
            std::vector<double3> positions(vertexNum), directions(vertexNum);
            std::vector<int> body_ids(vertexNum);
            std::vector<uint32_t> surface_ids(surf_vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(positions.data(), _vertexes,
                                      vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(directions.data(), _moveDir,
                                      vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(body_ids.data(), _point_body_id,
                                      vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(surface_ids.data(), _surfVerts,
                                      surf_vertexNum * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            double3 normal;
            double offset;
            CUDA_SAFE_CALL(cudaMemcpy(
                &normal, _groundNormal, sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(
                &offset, _groundOffset, sizeof(double), cudaMemcpyDeviceToHost));

            int max_vertex = -1;
            double max_component = -1.0;
            for(int vertex = 0; vertex < static_cast<int>(vertexNum); ++vertex)
            {
                const double3& direction = directions[vertex];
                const double component = std::max(
                    std::max(fabs(direction.x), fabs(direction.y)), fabs(direction.z));
                if(component > max_component)
                {
                    max_component = component;
                    max_vertex = vertex;
                }
            }

            int limiting_vertex = -1;
            double limiting_alpha = 1.0;
            double limiting_distance = 0.0;
            double limiting_coefficient = 0.0;
            for(uint32_t vertex : surface_ids)
            {
                const double3& position = positions[vertex];
                const double3& direction = directions[vertex];
                const double distance = normal.x * position.x + normal.y * position.y
                                      + normal.z * position.z - offset;
                const double coefficient = normal.x * direction.x + normal.y * direction.y
                                         + normal.z * direction.z;
                if(distance > 0.0 && coefficient > 0.0)
                {
                    const double candidate = std::min(
                        1.0, slackness_a * (distance / coefficient));
                    if(candidate < limiting_alpha)
                    {
                        limiting_alpha = candidate;
                        limiting_vertex = static_cast<int>(vertex);
                        limiting_distance = distance;
                        limiting_coefficient = coefficient;
                    }
                }
            }
            const double3 max_direction = max_vertex >= 0
                                              ? directions[max_vertex]
                                              : make_double3(0.0, 0.0, 0.0);
            printf("[merged-alpha-detail] frame=%d k=%d maxV=%d maxB=%d "
                   "maxDir=(%.17e,%.17e,%.17e) groundV=%d groundB=%d "
                   "groundAlpha=%.17e groundDist=%.17e groundCoef=%.17e\n",
                   s_dec_frame,
                   (int)k,
                   max_vertex,
                   max_vertex >= 0 ? body_ids[max_vertex] : -1,
                   max_direction.x,
                   max_direction.y,
                   max_direction.z,
                   limiting_vertex,
                   limiting_vertex >= 0 ? body_ids[limiting_vertex] : -1,
                   limiting_alpha,
                   limiting_distance,
                   limiting_coefficient);
        }
        gipc_nvtx_push("line_search");
        lineSearch(TetMesh, alpha, alpha_CFL);
        gipc_nvtx_pop();

        if(merged_diag_sample)
            printf("[merged-alpha] frame=%d k=%d move=%.17e thr=%.17e "
                   "ground=%.17e narrow=%.17e narrowN=%d temp=%.17e "
                   "ccdN=%d cfl=%.17e refined=%.17e refine=%d "
                   "preLS=%.17e postLS=%.17e lsRatio=%.17e cp=%d gp=%d\n",
                   s_dec_frame, (int)k, distToOpt_PN, _newton_thr,
                   diag_ground_alpha, diag_narrow_alpha, diag_narrow_pairs, temp_alpha,
                   ccd_cnt, alpha_CFL, diag_refined_alpha, (int)diag_refine_used,
                   alpha_before_line_search, alpha,
                   alpha_before_line_search > 0.0 ? alpha / alpha_before_line_search : 0.0,
                   (int)h_cpNum[0], (int)h_gpNum);

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end3));
        gipc_nvtx_push("post_ls");
        postLineSearch(TetMesh, alpha);
        gipc_nvtx_pop();
        //computeGradientAndHessian(TetMesh);
        if(phase_time)
        {
            CUDA_SAFE_CALL(cudaEventRecord(end4));
            // Waiting for the final timing event is sufficient; avoid stalling
            // unrelated streams even in diagnostic mode.
            CUDA_SAFE_CALL(cudaEventSynchronize(end4));
            float time00 = 0, time11 = 0, time22 = 0, time33 = 0, time44 = 0;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time00, start, end0));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time11, end0, end1));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time22, end1, end2));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time33, end2, end3));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time44, end3, end4));
            {   // time3 sub-split: S1 per-env alpha vs lineSearch
                float t3a = 0, t3b = 0;
                CUDA_SAFE_CALL(cudaEventElapsedTime(&t3a, end2, e2b));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&t3b, e2b, end3));
                g_t3_s1_ms += t3a;
                g_t3_ls_ms += t3b;
            }
            time0 += time00;
            time1 += time11;
            time2 += time22;
            time3 += time33;
            time4 += time44;
            destroy_iteration_events();
        }
        totalTimeStep += alpha;

        // Semi-implicit early exit (ref: arXiv 2512.12151, Algorithm 1)
        // beta tracks cumulative line-search progress; when alpha≈1 (good step),
        // beta decays fast -> early exit.  When alpha is small, beta stays large.
        // [semi-implicit x multi-env] beta/alpha are GLOBAL. Under the per-env
        // decoupled exit (DECOUPLE_THRESH) a global early break would re-couple the
        // batch: one env's good steps cut off still-unconverged mates at a
        // batch-dependent iter — the exact drift DECOUPLE_THRESH exists to prevent.
        // The decoupled path therefore ignores the semi-implicit exit (its per-env
        // frozen check already exits as soon as every env converged). A true per-env
        // beta belongs with the per-env productization work.
        const bool semi_decoupled = m_mode_config.decouple_thresh && m_env_alpha_valid;
        if(semi_implicit_enabled && semi_decoupled)
        {
            static bool noted = false;
            if(!noted)
            {
                printf("  [semi-implicit] NOTE: per-env decoupled exit active -> "
                       "beta runs PER-ENV (each env freezes at its own beta<=tol); "
                       "the global early exit is disabled.\n");
                noted = true;
            }
        }
        if(semi_implicit_enabled && k >= semi_implicit_min_iter && !semi_decoupled
           && drive_ratio >= 1.0)  // [drive-substep] ramp not done -> no early exit
        {
            if(TetMesh.h_groups_present)  // [N=1 guard] p2g is allocated (all -1) even single-env
            {   // multi-env scene, merged exit: legal but batch-coupled — say so once.
                static bool warned = false;
                if(!warned)
                {
                    printf("  [semi-implicit] WARNING: multi-env scene with a GLOBAL "
                           "semi-implicit exit — every env stops at the same Newton "
                           "iter, so a fast env can terminate a still-unconverged "
                           "mate (batch-coupled result). For per-env convergence "
                           "run with STIFF_DECOUPLE_THRESH=1 STIFF_PERENV_ALPHA=1.\n");
                    warned = true;
                }
            }
            semi_beta *= fmax(0.0, 1.0 - alpha);  // clamp: alpha>1 must not flip beta's sign
            if(semi_beta <= semi_implicit_beta_tol)
            {
                printf("  [semi-implicit] early exit at Newton iter %d (beta=%.6e, tol=%.6e)\n",
                       k, semi_beta, semi_implicit_beta_tol);
                k++;
                break;
            }
        }
    }
    // [multi-env S4] clear the linear-system mask so later/other solves are unmasked
    if(s4_mask_on) m_global_linear_system->set_env_mask(nullptr, nullptr, 0);
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for(int ii = 0; ii < iterV.size(); ii++)
    //{
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    if(g_gipc_log_level >= 1)
        printf("\n\n      Kappa: %f                               iteration k:  %d\n", Kappa, k);
    // [phase-time] cumulative GPU ms per phase (across frames). time0=Hessian/grad assembly,
    // time1=PCG linear solve, time2=CCD-BVH build, time3=line-search(+per-env alpha), time4=κ update.
    // Compare modes to localize the isolated/strict slowdown. Also reports Newton iters this frame.
    if(getenv("STIFF_PHASE_TIME"))
        printf("[phase-time cum-ms] Hess=%.0f PCG=%.0f ccdBVH=%.0f lineSearch=%.0f kappaUpd=%.0f | ls-split[s1=%.0f ls=%.0f] ls-inner[e=%.0f bvh=%.0f cp=%.0f step=%.0f] | this-frame-newton-iters=%d cum-pcg-iters=%lld\n",
               time0, time1, time2, time3, time4, g_t3_s1_ms, g_t3_ls_ms,
               g_ls_e_ms, g_ls_bvh_ms, g_ls_cp_ms, g_ls_step_ms, k, (long long)m_total_pcg_iters);
    return k;
}

// ── verbatim from gipc_modules/14 (pre-2d lines 2674..2897) ──
void   GIPC::IPC_Solver(device_TetraData& TetMesh)
{
    GipcNvtxScope _nvtx_frame("frame");
    //double animation_fullRate = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);
    //    if(isRotate&&m_total_frames*IPC_dt>=2.2){
    //        isRotate = false;
    //        updateBoundary2(TetMesh);
    //    }
    if(m_update_boundary)
    {
        updateBoundaryMoveDir(TetMesh, alpha, m_total_frames);
        buildBVH_FULLCCD(alpha);
        buildFullCP(alpha);
        if(h_ccd_cpNum > 0)
        {
            double slackness_m = 0.8;
            CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_alpha_invalid, 0, sizeof(int)));
            alpha              = std::min(alpha,
                             self_largestFeasibleStepSize(slackness_m, ensure_reduce_scratch(h_ccd_cpNum), h_ccd_cpNum));
        }
        //updateBoundary(TetMesh, alpha);

        CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                                  TetMesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToDevice));
        updateBoundaryMoveDir(TetMesh, alpha, m_total_frames);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);
        //step_forward(TetMesh, 1, true);

        bool rehash = true;

        buildBVH();
        int       numOfIntersect        = 0;
        const int boundary_isect_budget = line_search_max_iter > 0 ? line_search_max_iter : 64;
        while(isIntersected(TetMesh))
        {
            if(numOfIntersect >= boundary_isect_budget)
            {
                throw std::runtime_error(
                    "[StiffGIPC] boundary-move intersection persists after "
                    "backtracking (pre-move state likely already intersecting)");
            }
            printf("type 6 intersection happened:    %f\n", alpha);
            alpha /= 2.0;
            updateBoundaryMoveDir(TetMesh, alpha, m_total_frames);
            numOfIntersect++;
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        1,
                        true,
                        vertexNum);
            //step_forward(TetMesh, 1, true);
            buildBVH();
        }

        buildCP();
        printf("boundary alpha: %f\n  finished a step\n", alpha);
    }

    TetMesh.update_soft_constraint_target_position(m_total_frames + 1, IPC_dt);
    //suggestKappa(Kappa);
    upperBoundKappa(Kappa);
    if(Kappa < 1e-16)
    {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);
    //Kappa = 1e4;
#ifdef USE_FRICTION
    ensure_frictionBuffers();  // [0be8da3-port] grow-only, no per-frame malloc
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    int    k           = 0;
    double time0       = 0;
    double time1       = 0;
    double time2       = 0;
    double time3       = 0;
    double time4       = 0;
    // [iron-law] frame-start quarantine scan — must precede every CCD-alpha
    // kernel of this frame (see quarantineGroundInfeasibleAtFrameStart).
    quarantineGroundInfeasibleAtFrameStart();

    while(true)
    {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        m_total_newton_iters += solve_subIP(TetMesh, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = std::min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = std::max(minMaxDist1.y, minMaxDist2.y);


        bool finishMotion = animation_fullRate > 0.99 ? true : false;

        if(finishMotion)
        {
            tempFree_closeConstraint();
            break;
            //}
        }
        else
        {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        //updateVelocities(TetMesh);

        //computeXTilta(TetMesh, 1);
#ifdef USE_FRICTION
        ensure_frictionBuffers();  // [0be8da3-port] grow-only, no sub-iter realloc
        buildFrictionSets();
#endif
    }

#ifdef USE_FRICTION
    // [0be8da3-port] friction buffers persist across frames; freed in FREE_DEVICE_MEM.
#endif

    updateVelocities(TetMesh);

    computeXTilta(TetMesh, 1);
    cudaEventRecord(end0);
    // Engine.step remains synchronous, but only waits for this PTDS chain.
    CUDA_SAFE_CALL(cudaEventSynchronize(end0));
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    cudaEventDestroy(start);
    cudaEventDestroy(end0);
    m_total_time_ms += tttime;
    m_total_frames++;
    if(g_gipc_log_level >= 1)
        printf("average time cost:     %f,    frame id:   %d\n",
               m_total_newton_iters > 0
                   ? m_total_time_ms / m_total_newton_iters
                   : 0.0,
               m_total_frames);

    // [multi-env P3a] validate the segmented per-env reduction primitive on real
    // data: per-env vertex count (must match the substrate, e.g. 11433/env) and a
    // real per-env quantity (velocity norm). Read-only diagnostic, gated.
    if(getenv("STIFF_PENV_STATS") && TetMesh.d_point_to_group && TetMesh.h_groups_present)
    {
        const int NG = TetMesh.h_group_count;
        double*& d_sq = m_scr_sq_b;
        static int*    d_cnt = nullptr;
        if(!d_sq)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_sq, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cnt, kEnvAlphaSlots * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemset(d_sq, 0, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_cnt, 0, NG * sizeof(int)));
        int bs = 256, gs = (vertexNum + bs - 1) / bs;
        _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, TetMesh.velocities,
                                          d_sq, d_cnt, vertexNum, NG);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::vector<double> h_sq(NG);
        std::vector<int> h_cnt(NG);
        CUDA_SAFE_CALL(cudaMemcpy(h_sq.data(), d_sq, NG * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_cnt.data(), d_cnt, NG * sizeof(int), cudaMemcpyDeviceToHost));
        printf("[P3a-reduce] frame %d per-env:", m_total_frames);
        for(int g = 0; g < NG; ++g)
            if(h_cnt[g] > 0) printf(" g%d[n=%d |v|=%.4e]", g, h_cnt[g], sqrt(h_sq[g]));
        printf("\n");
    }

    m_phase_time_ms[0] += time0;
    m_phase_time_ms[1] += time1;
    m_phase_time_ms[2] += time2;
    m_phase_time_ms[3] += time3;
    m_phase_time_ms[4] += time4;


    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << m_phase_time_ms[0] / 1000.0 << std::endl;
    outTime << "time1: " << m_phase_time_ms[1] / 1000.0 << std::endl;
    outTime << "time2: " << m_phase_time_ms[2] / 1000.0 << std::endl;
    outTime << "time3: " << m_phase_time_ms[3] / 1000.0 << std::endl;
    outTime << "time4: " << m_phase_time_ms[4] / 1000.0 << std::endl;
    outTime << "time_makePD: " << m_time_make_pd_ms / 1000.0 << std::endl;

    outTime << "totalTime: " << m_total_time_ms / 1000.0 << std::endl;
    outTime << "total iter: " << m_total_newton_iters << std::endl;
    outTime << "frames: " << m_total_frames << std::endl;
    outTime << "totalCollisionNum: " << m_total_collision_pairs << std::endl;
    outTime << "averageCollision: "
            << (m_total_newton_iters > 0
                    ? m_total_collision_pairs / m_total_newton_iters
                    : 0.0)
            << std::endl;
    outTime << "maxCOllisionPairNum: " << m_max_collision_pairs << std::endl;
    outTime << "totalCgTime: " << m_total_pcg_iters << std::endl;
    outTime.close();


    auto& stats = gipc::Statistics::instance();

    stats.at_current_frame()["timer"] =
        gipc::GlobalTimer::current()->report_merged_as_json();
    if(g_gipc_log_level >= 1)
        gipc::GlobalTimer::current()->print_merged_timings();
    gipc::GlobalTimer::current()->clear();
    stats.write_to_file(std::string{gipc::output_dir()} + "/stats.json");

    auto f = stats.frame();
    stats.frame(f + 1);
}
