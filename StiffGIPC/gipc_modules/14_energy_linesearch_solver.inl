#include "frame_fsm/conditional_graph.h"

__global__ void _s3_decide(const double* Eg0,
                           const double* Eg1,
                           double*       env_alpha,
                           int*          decision_counts,
                           int           ng,
                           double        energy_abs_tol,
                           double        energy_rel_tol,
                           const int*    ground_gate = nullptr)
{
    // [C5] recorded S3 body: a ground-invalid trial round must not consume
    // its (garbage) trial energies — the tail halves the offending envs and
    // retries instead.
    if(ground_gate && *ground_gate != 0)
        return;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] <= 0.0) return;               // absent (or frozen) env
    double tol = energy_abs_tol + energy_rel_tol * fabs(Eg0[g]);
    if(Eg1[g] > Eg0[g] + tol)
    {
        env_alpha[g] *= 0.5;
        atomicAdd(decision_counts, 1);
    }
    else if(Eg1[g] > Eg0[g])
        atomicAdd(decision_counts + 1, 1);
}

// Standard (uniform-alpha) line-search decision.  Status: 0=descent,
// 1=retry/exhausted, 2=accepted only by configured roundoff tolerance.
extern __device__ uint32_t g_pair_overflow_count;
__global__ void _global_ls_decide(const double* energy0,
                                  const double* energy1,
                                  double        c1m,
                                  double        alpha,
                                  double        energy_abs_tol,
                                  double        energy_rel_tol,
                                  int*          status,
                                  const int*    gd_collapse,
                                  const double* alpha_dev)
{
    // [C-1 ls-graph] device-resident trial alpha when armed.
    if(alpha_dev)
        alpha = *alpha_dev;
    const double e0  = *energy0;
    const double e1  = *energy1;
    const double rhs = __dadd_rn(e0, __dmul_rn(c1m, alpha));
    const double tol = __dadd_rn(energy_abs_tol,
                                 __dmul_rn(energy_rel_tol, fabs(e0)));
    // [NaN-quarantine gap fix] any comparison against NaN is false, so a
    // NaN trial energy used to fall through to status 0 = "accepted descent"
    // — the one path where a diverged env's NaN could slip past the per-env
    // quarantine. Non-finite trial ⇒ status 1 (keep backtracking; on budget
    // exhaustion the loud non-descent warning fires instead of silence).
    status[0] = !isfinite(e1) ? 1
              : (e1 > __dadd_rn(rhs, tol) ? 1 : (e1 > rhs ? 2 : 0));
    // [B3 trial-defer] piggyback the pair-overflow counter into the SAME
    // 8-byte host read that fetches the decision — zero extra round trips.
    status[1] = (int)g_pair_overflow_count;
    // [B3 trial-defer] ground-collapse flag rides the same read; the host
    // response (quarantine / typed throw) fires only on a negative value.
    status[2] = gd_collapse ? *gd_collapse : 0;
}

// [C-1 ls-graph] trial-body head: halve the device alpha, count the trial.
__global__ void _ls_trial_begin(double* alpha_dev, int* status)
{
    *alpha_dev *= 0.5;
    ++status[3];
}
// [C-1 ls-graph] seed before graph launch: start alpha + zeroed trial count.
__global__ void _ls_seed(double* alpha_dev, double alpha0, int* status)
{
    *alpha_dev = alpha0;
    status[3]  = 0;
}
// [C-1 ls-graph] trial-body tail: publish alpha bits for the single post-loop
// host read, then self-relaunch while still backtracking with budget left
// (same device tail-launch idiom as pcg_graph_tail_relaunch).
__global__ void _ls_trial_tail(int* status, int budget, const double* alpha_dev)
{
    const unsigned long long bits =
        (unsigned long long)__double_as_longlong(*alpha_dev);
    status[4] = (int)(bits & 0xffffffffull);
    status[5] = (int)(bits >> 32);
    if(status[0] == 1 && status[3] < budget)
        cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

// [C-3 conditional LS] The first WHILE iteration evaluates the CCD-selected
// alpha verbatim; subsequent iterations halve before evaluating, matching the
// legacy "first trial outside the backtracking loop" order.
__global__ void _ls_conditional_seed(double* alpha_dev,
                                     const double* alpha0_dev,
                                     int* status,
                                     frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;
    *alpha_dev = *alpha0_dev;
    status[0]  = 1;
    status[1]  = static_cast<int>(g_pair_overflow_count);
    status[2]  = 0;
    status[3]  = 0;
    status[6]  = status[1];
    if(frame)
    {
        frame->alpha       = *alpha_dev;
        frame->ls_decision = 1;
        frame->phase       = frame_fsm::PHASE_LINE_SEARCH;
    }
}

__global__ void _ls_conditional_trial_begin(double* alpha_dev, int* status)
{
    if(blockIdx.x || threadIdx.x)
        return;
    if(status[3] != 0)
        *alpha_dev *= 0.5;
    ++status[3];
}

__global__ void _ls_conditional_tail(
    int* status,
    int budget,
    const double* alpha_dev,
    const double* energy_trial,
    cudaGraphConditionalHandle handle,
    frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;

    const bool pair_overflow = status[1] != status[6];
    const bool collapse      = status[2] < 0;
    const bool exhausted     = status[0] == 1 && status[3] >= budget;
    if(frame)
    {
        frame->alpha        = *alpha_dev;
        frame->energy_trial = *energy_trial;
        frame->ls_decision  = status[0];
        ++frame->ls_trial;
        frame->phase = frame_fsm::PHASE_LINE_SEARCH;
        if(pair_overflow)
        {
            frame_fsm::fsm_record_error(
                frame,
                frame_fsm::ERR_CAPACITY,
                frame_fsm::OVF_DCD_PAIRS,
                -1,
                -1);
            frame->result = frame_fsm::FRAME_RETRY_REQUIRED;
            frame->phase  = frame_fsm::PHASE_ROLLBACK;
        }
        if(collapse)
        {
            frame_fsm::fsm_record_error(
                frame,
                frame_fsm::ERR_SOLVER_EXCEPTION,
                frame_fsm::INV_START_INTERSECTING,
                -1,
                status[2]);
            frame->result = frame_fsm::FRAME_FATAL;
            frame->phase  = frame_fsm::PHASE_ROLLBACK;
        }
        if(exhausted)
        {
            frame_fsm::fsm_record_error(
                frame,
                frame_fsm::ERR_SOLVER_EXCEPTION,
                frame_fsm::INV_LS_BUDGET,
                -1,
                -1);
            frame->result = frame_fsm::FRAME_FATAL;
            frame->phase  = frame_fsm::PHASE_ROLLBACK;
        }
    }
    const bool healthy = !frame || frame->result == frame_fsm::FRAME_OK;
    cudaGraphSetConditional(
        handle,
        healthy && status[0] == 1 && status[3] < budget ? 1u : 0u);
}

__global__ void _newton_step_predicate(
    frame_fsm::FrameDeviceState* frame,
    cudaGraphConditionalHandle handle)
{
    if(blockIdx.x || threadIdx.x)
        return;
    const bool healthy = frame->result == frame_fsm::FRAME_OK;
    // Preserve the legacy `k && converged` rule: the first solved direction
    // always takes one line-search step, even if it is already below the
    // threshold. Later converged directions exit before stepping.
    frame->newton_step_active =
        healthy
        && (frame->newton_iter == 0 || !frame->newton_converged);
    cudaGraphSetConditional(
        handle, frame->newton_step_active ? 1u : 0u);
}

__global__ void _newton_iteration_begin(
    frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;
    frame->phase              = frame_fsm::PHASE_ASSEMBLY;
    frame->newton_step_active = 0;
}

__global__ void _newton_unit_alpha(
    double* alpha_slots,
    frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;
    alpha_slots[5] = 1.0;
    alpha_slots[6] = 1.0;
    frame->alpha     = 1.0;
    frame->cfl_alpha = 1.0;
}

__global__ void _newton_tail_conditional(
    frame_fsm::FrameDeviceState* frame,
    int iteration_cap,
    cudaGraphConditionalHandle handle)
{
    if(blockIdx.x || threadIdx.x)
        return;
    const int previous_completed = frame->newton_iter;
    if(frame->newton_step_active
       && frame->result == frame_fsm::FRAME_OK)
        ++frame->newton_iter;

    const bool first_step_completed =
        previous_completed == 0 && frame->newton_step_active;
    const bool keep_running =
        frame->result == frame_fsm::FRAME_OK
        && frame->newton_iter < iteration_cap
        && (first_step_completed || !frame->newton_converged);
    frame->phase = keep_running
                       ? frame_fsm::PHASE_ASSEMBLY
                       : (frame->result == frame_fsm::FRAME_OK
                              ? frame_fsm::PHASE_POST_LS
                              : frame_fsm::PHASE_ROLLBACK);
    cudaGraphSetConditional(handle, keep_running ? 1u : 0u);
}
// [de-CPU S3] intersect-safety halving (was: host loop over the stale mirror + H2D).
__global__ void _s3_halve_all(double* env_alpha, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] > 0.0) env_alpha[g] *= 0.5;
}

// [de-CPU S3] shared term-launcher for the per-env energy: fills the pe_all slice block on device
// (see slice layout below) and reports whether per-env kappa rescale applies. Used by BOTH the
// host-combine (validation) and the device-combine (S3 decision) variants.
static void _dbg_ksum(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t acc = 1469598103934665603ULL;
    for(uint64_t v : h) { acc ^= v; acc *= 1099511628211ULL; }
    printf("[ksum] %-14s %016llx\n", name, (unsigned long long)acc);
}
// COMMUTATIVE checksum (order-independent): sum+xor of bit-patterns. Tells value-non-det from
// order-non-det on the raw triplets (which sit at non-deterministic slots).
static void _dbg_ksum_comm(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t s = 0, x = 0;
    for(uint64_t v : h) { s += v; x ^= v; }
    printf("[ksum] %-14s sum=%016llx xor=%016llx\n", name, (unsigned long long)s, (unsigned long long)x);
}

// [decouple probe] env0-masked COMMUTATIVE (order-free) hash of the RAW Hessian triplet VALUES.
// Masks triplets whose row block is a FEM vertex in env0 (p2g[row]==0). Order-free integer
// bit-sum ⇒ batch-dependent triplet ORDER is irrelevant; only the env0 VALUE multiset + COUNT
// matter. Compare A-vs-B: ntrip differs⇒contact-pair SET differs; count same+hash differs⇒a
// per-pair value differs; both same⇒merge/preconditioner is the seed (values are bit-exact).
static void _dbg_hess_env0(const void* vals9, const int* rows, const int* cols, long ntrip, const int* p2g_dev, int vN,
                           long b_end = -1, long f_end = -1, long g_end = -1)
{
    std::vector<double> hv((size_t)ntrip * 9);
    std::vector<int>    hr(ntrip), hc(ntrip), hp(vN);
    cudaMemcpy(hv.data(), vals9, (size_t)ntrip * 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), rows, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hc.data(), cols, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hp.data(), p2g_dev, (size_t)vN * sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t src_sum[4] = {0,0,0,0}; long src_cnt[4] = {0,0,0,0};  // 0=BARRIER 1=FRICTION 2=GROUND 3=FEM
    // split: INTRA = row∈env0 AND col∈env0 ; CROSS = row∈env0 AND col∉env0 (cross-env leak)
    uint64_t si = 0, xi = 0, sc = 0, xc = 0; long ci = 0, cc = 0;
    long col_neg1 = 0, col_oob = 0, col_mate = 0;  // CROSS col classification
    long mate_min = (long)ntrip + 1, mate_max = -1;  // triplet-index range of cross-env-mate triplets
    for(long t = 0; t < ntrip; ++t)
    {
        int r = hr[t], c = hc[t];
        bool re = (r >= 0 && r < vN && hp[r] == 0);
        if(!re) continue;
        bool ce = (c >= 0 && c < vN && hp[c] == 0);
        uint64_t hh = 0, hx = 0;
        for(int e = 0; e < 9; ++e) { uint64_t b; memcpy(&b, &hv[(size_t)t * 9 + e], 8); hh += b; hx ^= b; }
        if(ce) { ++ci; si += hh; xi ^= hx;
                 if(b_end >= 0) { int s = (t < b_end) ? 0 : (t < f_end) ? 1 : (t < g_end) ? 2 : 3;
                                  src_sum[s] += hh; ++src_cnt[s]; } }
        else
        {
            ++cc; sc += hh; xc ^= hx;
            if(c < 0 || c >= vN) ++col_oob;        // col out-of-range (ABD body dof? ground?)
            else if(hp[c] < 0) ++col_neg1;          // col ungrouped (-1): env0 gripper/static
            else { ++col_mate;                       // col in a DIFFERENT env (1/2/3): TRUE cross-env
                   if(t < mate_min) mate_min = t; if(t > mate_max) mate_max = t; }
        }
    }
    printf("[ksum-env0] INTRA: ntrip=%ld sum=%016llx | CROSS: ntrip=%ld | by-src BARRIER(n=%ld,s=%016llx) FRICTION(n=%ld,s=%016llx) GROUND(n=%ld,s=%016llx) FEM(n=%ld,s=%016llx)\n",
           ci, (unsigned long long)si, cc,
           src_cnt[0], (unsigned long long)src_sum[0], src_cnt[1], (unsigned long long)src_sum[1],
           src_cnt[2], (unsigned long long)src_sum[2], src_cnt[3], (unsigned long long)src_sum[3]);
}

int GIPC::calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type)
{
    gipc::Timer timer{"solve_linear_system"};
    auto        iter = 0;

    if(getenv("STIFF_HESS_ENV0") && TetMesh.d_point_to_group
       && g_dec_frame == (getenv("STIFF_DUMP_FRAME") ? atoi(getenv("STIFF_DUMP_FRAME")) : -1)
       && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
    {
        cudaDeviceSynchronize();
        { extern unsigned long long get_xskip(); printf("[xskip] cross-env pairs skipped so far = %llu\n", get_xskip());
          long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          printf("[tri-bounds] BARRIER=[0,%ld) FRICTION=[%ld,%ld) GROUND=[%ld,%ld) FEM=[%ld,end) total=%lld\n",
                 Bb, Bb, Bb+Ff, Bb+Ff, Bb+Ff+(long)h_gpNum, Bb+Ff+(long)h_gpNum, (long long)gipc_global_triplet.global_triplet_offset); }
        // [corrected] mask by d_dof_to_group (the BLOCK index space the triplet row/col live in:
        // ABD blocks first, then FEM vertex blocks), NOT d_point_to_group (per-vertex, no ABD prefix).
        { long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          _dbg_hess_env0(gipc_global_triplet.block_values(), gipc_global_triplet.block_row_indices(),
                       gipc_global_triplet.block_col_indices(),
                       (long)gipc_global_triplet.global_triplet_offset, TetMesh.d_dof_to_group, TetMesh.dof_block_count,
                       Bb, Bb+Ff, Bb+Ff+(long)h_gpNum); }
    }

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        _dbg_ksum("fb_in",      TetMesh.fb,          (size_t)vertexNum * sizeof(double3));
        _dbg_ksum("shapegrad_in", TetMesh.shape_grads, (size_t)vertexNum * sizeof(double3));
        // COMMUTATIVE hash of the RAW triplets (block values + row/col), order-independent →
        // tells if the Hessian VALUE-multiset is deterministic (vs just non-det slot order).
        _dbg_ksum_comm("rawHess_comm", gipc_global_triplet.block_values(),
                       (size_t)gipc_global_triplet.global_triplet_offset * 9 * sizeof(double));
        _dbg_ksum_comm("rawRow_comm", gipc_global_triplet.block_row_indices(),
                       (size_t)gipc_global_triplet.global_triplet_offset * sizeof(int));
        printf("[ksum] cpNum=%d Kappa=%.17g toff=%lld\n",
               h_cpNum[0], Kappa, (long long)gipc_global_triplet.global_triplet_offset);
    }

    // [xenv] gradient RHS asymmetry — is fb already env-asymmetric BEFORE the solve?
    // (fb is assembled by computeGradientAndHessian from identical co-located verts.)
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(TetMesh.fb, "fb(grad) pre-solve");

    iter = m_global_linear_system->solve_linear_system();

    // [xenv] search-direction asymmetry — did the SOLVE (SpMV/dot/precond) introduce it?
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(_moveDir, "moveDir post-solve");

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        // merged matrix (converter output, what the spmv used) + the solve result
        _dbg_ksum("matrix_merged", gipc_global_triplet.block_values(),
                  (size_t)gipc_global_triplet.h_unique_key_number * 9 * sizeof(double));
        printf("[ksum] nuniq_merged=%d\n", gipc_global_triplet.h_unique_key_number.get());
        _dbg_ksum("moveDir_out", _moveDir, (size_t)vertexNum * sizeof(double3));
    }


    if(!frame_fsm::ConditionalGraphRecorder::active())
    {
        auto& json = gipc::Statistics::instance().at_current_frame();
        json["newton"].back()["pcg"]["iterations"] = iter;
    }
    return iter;
}


bool edgeTriIntersectionQuery(const int*     _bodyId,
                              const int*     _btype,
                              const double3* _vertexes,
                              const uint2*   _edges,
                              const uint3*   _faces,
                              const AABB*    _edge_bvs,
                              const Node*    _edge_nodes,
                              double         dHat,
                              int            number,
                              const int*     _collision_skip_matrix,
                              int            _collision_body_count,
                              const int*     _body_id_to_is_fem)
{
    int numbers = number;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    int*               _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery<<<blockNum, threadNum>>>(
        _bodyId, _btype, _vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    return edgeTriIntersectionQuery(bvh_e._bodyId,
                                    bvh_e._btype,
                                    TetMesh.vertexes,
                                    bvh_e._edges,
                                    bvh_f._faces,
                                    bvh_e._bvs,
                                    bvh_e._nodes,
                                    dHat,
                                    bvh_f.face_number,
                                    bvh_e._collision_skip_matrix,
                                    bvh_e._collision_body_count,
                                    _body_id_to_is_fem);
}

bool GIPC::checkGroundIntersection()
{
    int numbers = h_gpNum;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

int GIPC::groundTrialStatus(const int* point_to_group, int group_count)
{
    if(m_skip_all_collision || surf_vertexNum == 0 || !m_ground_trial_invalid)
        return 0;
    const bool per_env = point_to_group && m_env_ground_trial_invalid && group_count > 0;
    CUDA_SAFE_CALL(cudaMemsetAsync(m_ground_trial_invalid, 0, sizeof(int)));
    if(per_env)
        CUDA_SAFE_CALL(cudaMemsetAsync(
            m_env_ground_trial_invalid, 0, group_count * sizeof(int)));
    const int threads = 256;
    _markGroundTrialInvalid<<<(surf_vertexNum + threads - 1) / threads, threads>>>(
        _vertexes,
        _surfVerts,
        _groundOffset,
        _groundNormal,
        _point_body_id,
        _ground_skip_body,
        _ground_body_count,
        per_env ? point_to_group : nullptr,
        per_env ? m_env_ground_trial_invalid : nullptr,
        m_ground_trial_invalid,
        group_count,
        surf_vertexNum);
    int status = 0;
    CUDA_SAFE_CALL(cudaMemcpy(
        &status, m_ground_trial_invalid, sizeof(int), cudaMemcpyDeviceToHost));
    return status;
}

void GIPC::halveGroundInvalidEnvAlpha(int group_count)
{
    if(group_count <= 0 || !m_env_alpha || !m_env_ground_trial_invalid)
        return;
    const int threads = 256;
    _halveGroundInvalidEnvAlpha<<<(group_count + threads - 1) / threads, threads>>>(
        m_env_alpha, m_env_ground_trial_invalid, group_count);
}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if(m_skip_all_collision)
        return false;

    // CCD line-search already constrains alpha to a non-intersecting step.
    // The line-search-tail isIntersected() check is a paranoid second pass
    // that re-runs _edgeTriIntersectionQuery (42% of GPU time in case39)
    // for every line-search alpha bisection.  On smooth-contact scenes it
    // never fires (verified across 9/9 paired runs on case39).
    //
    // Default: SKIP the recheck (was opt-in via STIFF_SKIP_CCD_SANITY=1 in
    // dc11e10).  Set GIPC_FORCE_CCD_SANITY=1 to restore the v0.6-and-earlier
    // behavior of running the check.  STIFF_SKIP_CCD_SANITY=0 also restored
    // (back-compat); any other value or unset = skip.
    // [fail-fast][2026-07-08] We tried default-ON: on a plain scene (single FEM
    // bunny dropped on the ground) the edge-tri recheck reports intersections
    // every bisection ("type 0 intersection happened" x260k) and the alpha loop
    // never exits -> hang. The checker is not usable as a default in its current
    // state (false positives / self-intersection sensitivity), which is the real
    // reason it was disabled — document this instead of hiding it. Default stays
    // OFF; GIPC_FORCE_CCD_SANITY=1 opts in for debugging. The ground-collapse
    // invariant is enforced separately by the buildCP d-floor throw.
    static const bool keep_sanity = []{
        const char* v_force = std::getenv("GIPC_FORCE_CCD_SANITY");
        if(v_force && v_force[0] && v_force[0] != '0') return true;
        return false;
    }();
    if(!keep_sanity) return false;

    if(checkGroundIntersection())
    {
        return true;
    }

    if(checkEdgeTriIntersectionIfAny(TetMesh))
    {
        std::cout << "is edge triangle\n";
        return true;
    }
    return false;
}




void GIPC::tempMalloc_closeConstraint()
{
    // [0be8da3-port, grow-only] buffers persist across sub-iterations/frames;
    // (re)allocate only on growth. Kernels touch [0, h_gpNum)/[0, h_cpNum[0]).
    if((size_t)h_gpNum > m_close_gp_cap)
    {
        if(_closeConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
        }
        size_t n = (size_t)h_gpNum + h_gpNum / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID, n * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal, n * sizeof(double)));
        m_close_gp_cap = n;
    }
    if((size_t)h_cpNum[0] > m_close_cp_cap)
    {
        if(_closeMConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
        }
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID, n * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal, n * sizeof(double)));
        m_close_cp_cap = n;
    }
}

void GIPC::tempFree_closeConstraint()
{
    // [0be8da3-port] no-op — buffers persist; freed in FREE_DEVICE_MEM.
}

void GIPC::ensure_frictionBuffers()
{
    // [0be8da3-port, grow-only] replaces the per-frame/per-sub-iter free+malloc
    // of the 7 friction lastH buffers. distCoord's live range [0, h_cpNum[0])
    // is re-zeroed on EVERY call (a fresh cudaMalloc'd buffer was memset the
    // same way), keeping the [4.3] frame-0 lag fix value-identical.
    if((size_t)h_cpNum[0] > m_fric_cp_cap)
    {
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;   // growth policy stays HERE
        lambda_lastH_scalar.resize_discard(n);            // [3d] release-then-alloc
        distCoord.resize_discard(n);
        tanBasis.resize_discard(n);
        _collisonPairs_lastH.resize_discard(n);
        m_fric_cp_cap = n;
    }
    if((size_t)h_gpNum > m_fric_gd_cap)
    {
        size_t n = (size_t)h_gpNum + h_gpNum / 4;         // growth policy stays HERE
        lambda_lastH_scalar_gd.resize_discard(n);
        _collisonPairs_lastH_gd.resize_discard(n);
        m_fric_gd_cap = n;
    }
    if(h_cpNum[0])
        CUDA_SAFE_CALL(cudaMemset(distCoord, 0, h_cpNum[0] * sizeof(double2)));  // [4.3] frame-0 lag uninit
}

void GIPC::updateVelocities(device_TetraData& TetMesh)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);

    m_abd_system->update_velocity(*m_abd_sim_data);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha, int fid)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundaryMoveDir<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers, fid);
}


void GIPC::computeXTilta(device_TetraData& TetMesh, const double& rate)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(TetMesh.BoundaryType,
                                            TetMesh.velocities,
                                            TetMesh.o_vertexes,
                                            TetMesh.xTilta,
                                            TetMesh.apply_gravity,
                                            IPC_dt,
                                            rate,
                                            gravity,
                                            numbers);

    m_abd_system->cal_q_tilde(*m_abd_sim_data);
}

// ============================================================================
// [C5] isolated-mode whole-frame graph bodies.
//
// Placed at the tail of the composite TU so every per-env kernel (11), the S3
// decision (this file), the ground-trial marker (06), the ABD alpha gather
// (08) and the device per-env energy dispatcher (energy/01) are visible.
//
// Isolation contract inside the graph: the recorded detection pipeline is the
// MERGED tree with env-id emission filtering (set_self_p2g), which yields the
// same pair SET as the per-env-tree path — zero cross-env contacts — while
// staying capture-safe (the per-env build is a host loop over envs with
// variable launch extents and host-side BVH object mutation). Every per-env
// SOLVER decision (alpha, freeze, S3 backtracking, kappa) is device-resident.
// ============================================================================

// Frame-boundary training: all per-env scratch that the host path allocates
// lazily must exist at final size before capture.
void GIPC::train_perenv_graph_capacities(device_TetraData& TetMesh)
{
    if(!m_scr_env_cnt)
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_scr_env_cnt, 3 * sizeof(int)));
    if(!m_scr_ls_eg0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_scr_ls_eg0,
                                  kEnvAlphaSlots * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_scr_ls_eg1,
                                  kEnvAlphaSlots * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_scr_ls_decision_counts,
                                  4 * sizeof(int)));
    }
    const int abdN = static_cast<int>(abd_fem_count_info.abd_body_num);
    if(abdN > 0 && !m_abd_body_alpha)
        m_abd_body_alpha.resize_discard(abdN);

    // Prime the MERGED detection pipeline at the boundary. Two distinct
    // reasons, both fatal inside capture if skipped:
    //   1. every previous frame ran the per-env trees, so the merged tree's
    //      lazily sized sort scratch was never allocated at full extent;
    //   2. the device-symbol setters (EE dedup/canon/env-part/self-p2g/...)
    //      are VALUE-CACHED — the per-env and merged paths publish different
    //      values, so the first merged call after a per-env frame fires a
    //      synchronous cudaMemcpyToSymbol, which is illegal in capture.
    // Running one merged build+detect here settles both.
    const bool restore_merged_detect = m_graph_merged_detect;
    m_graph_merged_detect = true;
    buildBVH();
    buildCP();
    m_graph_merged_detect = restore_merged_detect;

    // The per-env energy dispatcher's slice block + the ABD per-env bin are
    // lazily sized on first use; run one dispatch at the boundary so the
    // recorded pass finds them resident.
    if(m_env_alpha && m_scr_ls_eg0)
        computeEnergy_perenv_dev(TetMesh, m_scr_ls_eg0);
}

// The per-env CCD-alpha chain, enqueue form: S1 phase A (direct terms over the
// DCD-time snapshot), the merged swept chain (C4, capture-safe), S1 phase B
// (refined / cfl / max-move) and the device freeze decision publishing
// all_env_frozen into FrameDeviceState::newton_converged. No host reads.
void GIPC::enqueue_perenv_ccd_alpha_conditional(device_TetraData& TetMesh)
{
    frame_fsm::FrameDeviceState* frame = frame_graph_device_state();
    const int    NG          = m_active_group_count;
    const int    bs          = 256;
    const double slackness_a = 0.9;
    const double slackness_m = 0.8;

    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_ccd_alpha_invalid, 0, sizeof(int), cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_refined_invalid,
                                   0,
                                   (1 + kEnvAlphaSlots) * sizeof(int),
                                   cudaStreamPerThread));

    // ---- S1 phase A: direct per-env ground + narrow-self terms ----
    _fill_double<<<(2 * NG + bs - 1) / bs, bs, 0, cudaStreamPerThread>>>(
        m_env_scratch, 1.0, 2 * NG);
    if(surf_vertexNum >= 1)
        _per_env_groundAlpha_min<<<(surf_vertexNum + bs - 1) / bs,
                                   bs,
                                   0,
                                   cudaStreamPerThread>>>(
            _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir,
            TetMesh.d_point_to_group, m_env_scratch + 0 * NG, slackness_a,
            surf_vertexNum, _point_body_id, _ground_skip_body,
            _ground_body_count, NG, m_ccd_alpha_invalid);
    if(m_dcd_snap_cap > 0)
        _per_env_selfAlpha_min<<<(m_dcd_snap_cap + bs - 1) / bs,
                                 bs,
                                 0,
                                 cudaStreamPerThread>>>(
            _vertexes, _dcd_ccd_snapshot, _moveDir, TetMesh.d_point_to_group,
            m_env_scratch + 1 * NG, slackness_m, m_dcd_snap_cap, NG,
            nullptr, m_ccd_alpha_invalid, kCcdInvalidPerEnvNarrow, nullptr,
            m_pair_snap_cur.data() + 0);

    // ---- the global scalar chain: also drives the swept build ----
    // (its slots seed the uniform fallback alpha; the per-env path consumes
    //  the fresh _ccd_collisonPairs it produces).
    enqueue_ccd_alpha_conditional();

    // ---- S1 phase B: refined-self / cfl / Newton max-move ----
    _fill_double<<<(NG + bs - 1) / bs, bs, 0, cudaStreamPerThread>>>(
        m_env_scratch + 2 * NG, 1.0, NG);
    CUDA_SAFE_CALL(cudaMemsetAsync(m_env_scratch + 3 * NG,
                                   0,
                                   2 * NG * sizeof(double),
                                   cudaStreamPerThread));
    const int ccd_train = graph_trained_ccd_extent();   // [C6]
    _per_env_selfAlpha_min<<<(ccd_train + bs - 1) / bs,
                             bs,
                             0,
                             cudaStreamPerThread>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, TetMesh.d_point_to_group,
        m_env_scratch + 2 * NG, slackness_m, ccd_train, NG,
        nullptr, m_ccd_alpha_invalid, kCcdInvalidPerEnvRefined,
        m_ccd_refined_invalid + 1,
        _cpNum);
    _per_env_max_cfl<<<(surf_vertexNum + bs - 1) / bs, bs, 0,
                       cudaStreamPerThread>>>(
        TetMesh.d_point_to_group, _moveDir, _surfVerts,
        m_env_scratch + 3 * NG, surf_vertexNum, NG);
    _per_env_max_move<<<(vertexNum + bs - 1) / bs, bs, 0,
                        cudaStreamPerThread>>>(
        TetMesh.d_point_to_group, _moveDir, m_env_scratch + 4 * NG,
        vertexNum, NG);

    // ---- device per-env alpha + freeze + Newton-exit publication ----
    // decouple=1 by contract (isolated bundle), so the refinement gate is
    // per-env and the host temp_alpha/alpha_CFL arguments are dead.
    double thr_bbox2 = bboxDiagSize2;
    if(TetMesh.h_groups_present && m_avg_env_bbox2 > 0.0)
        thr_bbox2 = m_avg_env_bbox2;
    const double sq = sqrt(dHat);
    const double thr_cv =
        (newton_velocity_tol > 0.0)
            ? (newton_velocity_tol * IPC_dt)
            : sqrt(Newton_solver_threshold * Newton_solver_threshold
                   * thr_bbox2 * IPC_dt * IPC_dt);
    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_scr_env_cnt, 0, 3 * sizeof(int), cudaStreamPerThread));
    _per_env_alpha_compute<<<(NG + bs - 1) / bs, bs, 0, cudaStreamPerThread>>>(
        m_env_alpha, m_env_scratch, NG, sq, 1.0, 1,
        0.0, 0.0, 1,
        0, thr_cv,
        d_env_bbox2, Newton_solver_threshold * IPC_dt,
        newton_velocity_tol * IPC_dt,
        m_ccd_refined_invalid + 1, m_ccd_alpha_invalid,
        m_scr_env_cnt,
        _cpNum);
    _perenv_newton_decide<<<1, 1, 0, cudaStreamPerThread>>>(
        m_scr_env_cnt, frame);
}

// The per-env S3 line search as a conditional WHILE loop: per-env step from
// the temp config, capture-safe rebuild in deferred-count mode, device per-env
// energies, in-place halving, loop control on a conditional handle.
void GIPC::enqueue_s3_line_search_conditional(device_TetraData& TetMesh)
{
    auto* recorder = frame_fsm::ConditionalGraphRecorder::current();
    if(!recorder)
        throw std::logic_error(
            "[s3-conditional] no active conditional recorder");
    frame_fsm::FrameDeviceState* frame = frame_graph_device_state();
    const int NG    = m_active_group_count;
    const int abdN  = static_cast<int>(abd_fem_count_info.abd_body_num);
    const int bs    = 256;
    const int maxBT = 8;

    computeEnergy_perenv_dev(TetMesh, m_scr_ls_eg0);
    CUDA_SAFE_CALL(cudaMemcpyAsync(TetMesh.temp_double3Mem,
                                   TetMesh.vertexes,
                                   vertexNum * sizeof(double3),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);

    _s3_conditional_seed<<<1, 1, 0, cudaStreamPerThread>>>(
        m_scr_ls_decision_counts, frame);
    recorder->while_loop(
        1,
        cudaGraphCondAssignDefault,
        [&](cudaGraphConditionalHandle handle)
        {
            _s3_round_begin<<<1, 1, 0, cudaStreamPerThread>>>(
                m_scr_ls_decision_counts);
            if(abdN > 0 && TetMesh.d_body_to_group)
                _gather_abd_body_alpha<<<(abdN + bs - 1) / bs, bs, 0,
                                         cudaStreamPerThread>>>(
                    TetMesh.d_body_to_group, m_env_alpha, m_abd_body_alpha,
                    abdN, NG);
            m_perenv_apply = true;
            step_forward(TetMesh, 0.0, false);
            m_perenv_apply = false;
            buildBVH();
            // Ground-trial adjudication in device form: the host reads one
            // int here; the recorded body publishes a flag word that gates
            // the energy decision and steers the tail.
            CUDA_SAFE_CALL(cudaMemsetAsync(
                m_ground_trial_invalid, 0, sizeof(int), cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(m_env_ground_trial_invalid,
                                           0,
                                           NG * sizeof(int),
                                           cudaStreamPerThread));
            if(surf_vertexNum >= 1)
                _markGroundTrialInvalid<<<(surf_vertexNum + bs - 1) / bs, bs,
                                          0, cudaStreamPerThread>>>(
                    _vertexes, _surfVerts, _groundOffset, _groundNormal,
                    _point_body_id, _ground_skip_body, _ground_body_count,
                    TetMesh.d_point_to_group, m_env_ground_trial_invalid,
                    m_ground_trial_invalid, NG, surf_vertexNum);
            buildCP();
            computeEnergy_perenv_dev(TetMesh, m_scr_ls_eg1);
            _s3_decide<<<(NG + bs - 1) / bs, bs, 0, cudaStreamPerThread>>>(
                m_scr_ls_eg0, m_scr_ls_eg1, m_env_alpha,
                m_scr_ls_decision_counts, NG,
                energy_abs_tol, energy_rel_tol,
                m_ground_trial_invalid);
            _s3_tail_conditional<<<1, 1, 0, cudaStreamPerThread>>>(
                m_scr_ls_decision_counts, m_ground_trial_invalid,
                m_env_alpha, m_env_ground_trial_invalid, NG, maxBT,
                handle, frame);
        });
}
