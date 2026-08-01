__global__ void _ccd_initial_alpha_combine(double* slots,
                                           int have_ground,
                                           int have_self,
                                           int* invalid)
{
    const double ground = have_ground ? slots[0] : 1.0;
    const double self   = have_self ? slots[1] : 1.0;
    slots[0] = ground;
    slots[1] = self;
    if(have_ground && (!isfinite(ground) || ground <= 0.0 || ground > 1.0))
        atomicOr(invalid, kCcdInvalidGlobalGround);
    if(have_self && (!isfinite(self) || self <= 0.0 || self > 1.0))
        atomicOr(invalid, kCcdInvalidGlobalNarrow);
    slots[2] = ground < self ? ground : self;
}

__global__ void _ccd_final_alpha_combine(double* slots,
                                         int have_ccd_pairs,
                                         double d_hat,
                                         double ccd_size,
                                         int* invalid,
                                         const int* refined_invalid,
                                         const uint32_t* d_ccd_count,
                                         frame_fsm::FrameDeviceState* frame,
                                         int ccd_capacity = 0)
{
    // [B3 ccd-defer] when armed, the pair gate comes from the live device
    // count and the raw count rides slot 8 of the same scalar-chain read
    // (exact in double well past 2^32). Null keeps legacy semantics.
    if(d_ccd_count)
    {
        have_ccd_pairs = (*d_ccd_count > 0u) ? 1 : 0;
        slots[8]       = (double)(*d_ccd_count);
    }
    const double temp_alpha = slots[2];
    double refined   = 1.0;
    double alpha_cfl = temp_alpha;
    double alpha     = temp_alpha;
    if(have_ccd_pairs)
    {
        const double max_speed = slots[3];
        refined  = slots[4];
        alpha_cfl = __dmul_rn(__ddiv_rn(__dsqrt_rn(d_hat), max_speed), 0.5);
        // Keep comparison semantics deliberate: a NaN alpha_cfl propagates to
        // alpha and is rejected by the single host validation below.
        alpha = temp_alpha < alpha_cfl ? temp_alpha : alpha_cfl;
        if(temp_alpha > __dmul_rn(2.0, alpha_cfl))
        {
            if((refined_invalid && (*refined_invalid & kCcdRawInvalid))
               || !isfinite(refined) || refined <= 0.0 || refined > 1.0)
                atomicOr(invalid, kCcdInvalidGlobalRefined);
            const double refined_scaled = __dmul_rn(refined, ccd_size);
            alpha = temp_alpha < refined_scaled ? temp_alpha : refined_scaled;
            alpha = alpha > alpha_cfl ? alpha : alpha_cfl;
        }
    }
    slots[4] = refined;
    slots[5] = alpha;
    slots[6] = alpha_cfl;
    const uint32_t invalid_bits =
        static_cast<uint32_t>(*invalid & kCcdInvalidEffectiveMask);
    slots[7] = static_cast<double>(invalid_bits);
    if(frame)
    {
        frame->alpha     = alpha;
        frame->cfl_alpha = alpha_cfl;
        frame->ccd_count = d_ccd_count
                               ? static_cast<int>(*d_ccd_count)
                               : have_ccd_pairs;
        // [C6-e] Keep the frame's PEAK swept count, not just the overflow
        // value. Without it the telemetry reports 0 for every frame that
        // happens to fit, which makes it impossible to tell an anomalous sweep
        // from a normal one. The growth path only consults this when an OVF bit
        // is set, so recording it unconditionally changes no policy.
        if(frame->ccd_count > frame->hw_ccd_pairs)
            frame->hw_ccd_pairs = frame->ccd_count;
        frame->phase = frame_fsm::PHASE_LINE_SEARCH;
        // [C4-a] in-graph replacement for the host legacy grow-redo: past-
        // capacity raw counts mean the deferred refined reduction saw only a
        // subset, so the frame must retry from a boundary with grown tiers.
        if(ccd_capacity > 0 && d_ccd_count
           && *d_ccd_count > static_cast<uint32_t>(ccd_capacity))
        {
            frame->hw_ccd_pairs = static_cast<int>(*d_ccd_count);
            frame_fsm::fsm_record_error(
                frame,
                frame_fsm::ERR_CAPACITY,
                frame_fsm::OVF_CCD_PAIRS,
                -1,
                -1);
            frame->result = frame_fsm::FRAME_RETRY_REQUIRED;
            frame->phase  = frame_fsm::PHASE_ROLLBACK;
        }
        if(invalid_bits || !isfinite(alpha) || alpha <= 0.0 || alpha > 1.0)
        {
            frame_fsm::fsm_record_error(
                frame,
                frame_fsm::ERR_CCD_INVALID,
                invalid_bits ? invalid_bits
                             : static_cast<uint32_t>(
                                   frame_fsm::INV_NAN_STATE),
                -1,
                -1);
            frame->result = frame_fsm::FRAME_FATAL;
            frame->phase  = frame_fsm::PHASE_ROLLBACK;
        }
    }
}


double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets)
{

    int numbers = tetrahedraNum;
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, tets, _moveDir, mqueue, slackness, errorRate, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

// [B3 trial-defer] the counts-after-build refresh, extracted so the
// line-search exit can restore mirror freshness after deferred trials.
void GIPC::refresh_pair_counts()
{
    CUDA_SAFE_CALL(cudaMemcpyAsync(m_pair_snap_cur,
                                   _cpNum,
                                   6 * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    uint32_t cp_gp_buf[6];
    CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    memcpy(h_cpNum.refresh_dst(), cp_gp_buf, 5 * sizeof(uint32_t));
    h_gpNum = cp_gp_buf[5];
    note_pair_census_peak();
}

void GIPC::buildCP()
{
    // [3b] device truth changes below: DCD re-emission rewrites _cpNum/_gpNum
    // AND clobbers the _ccd_collisonPairs prefix (the documented narrow-self
    // hazard) — all three mirrors stale until their refresh points.
    h_cpNum.invalidate();
    h_gpNum.invalidate();
    h_ccd_cpNum.invalidate();
    if(m_skip_all_collision)
    {
        memset(h_cpNum.refresh_dst(), 0, 5 * sizeof(uint32_t));
        h_gpNum = 0;
        CUDA_SAFE_CALL(cudaMemsetAsync(m_pair_snap_cur,
                                       0,
                                       6 * sizeof(uint32_t),
                                       cudaStreamPerThread));
        return;
    }

    // [env-det] EE detection settings + env-local vertex map MUST be set BEFORE the per-env branch,
    // else the per-env path (which returns early) runs the EE dedup with GLOBAL edge indices (not
    // env-local) → cross-env asymmetric. Idempotent; the merged path below re-runs harmlessly.
    set_ee_nodedup(getenv("STIFF_EE_NODEDUP") ? 1 : 0);
    set_ee_detgate(m_mode_config.ee_detgate ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);
    set_bvh_audit(getenv("STIFF_STACK_DIAG") ? 1 : 0);  // [audit-gate] per-pop depth probe, diag only
    // [perenv-par] per-vertex cross-env skip at self-collision emission (robust where BVH env-part is
    // bypassed by env-MIXED co-located nodes). Gated STIFF_DECOUPLE_THRESH; null = off (legacy path).
    set_self_p2g((m_mode_config.decouple_thresh && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(m_mode_config.ee_canon ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    if(m_mode_config.ee_canon && m_d_p2g && !m_vloc_built)
    {
        std::vector<int> hp(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec;
        for(int v=0; v<vertexNum; v++){ int g=hp[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_vloc, (size_t)vertexNum*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(m_d_vloc, vloc.data(), (size_t)vertexNum*sizeof(int), cudaMemcpyHostToDevice));
        set_ee_vloc(m_d_vloc); m_vloc_built = true;
    }
    // g_vloc is a process-global CUDA symbol. A previous Engine may have
    // published an instance-owned map that reset() has since freed. Always
    // republish this Engine's current view (including nullptr) before any EE
    // kernel, rather than relying on the symbol's stale prior value.
    set_ee_vloc((m_mode_config.ee_canon && m_vloc_built) ? m_d_vloc : nullptr);

    // [multi-env P2] per-env BVH path: build each env's tree on LOCAL verts + detect, looped.
    // [C5] the whole-frame graph forces the merged tree: the per-env build is
    // a host loop over envs with variable launch extents and host-side BVH
    // object mutation (not capture-safe). Isolation is preserved because
    // set_self_p2g (armed under decouple_thresh) filters cross-env pairs at
    // emission, so the pair SET is the same either way.
    if(m_perenv_bvh && m_d_p2g && !m_graph_merged_detect)
    {
        if(m_perenv_bvh_groups == 0)
            buildPerEnvBVHIndex(m_active_group_count, m_d_p2g);
        buildBVH_and_CP_perenv(dHat);
        return;
    }

    if(!m_aux_stream)
        CUDA_SAFE_CALL(cudaStreamCreate(&m_aux_stream));
    if(!m_aux_reset_event)
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(
            &m_aux_reset_event, cudaEventDisableTiming));
    if(!m_aux_done_event)
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(
            &m_aux_done_event, cudaEventDisableTiming));

    // Memsets on default stream. Use an event so aux stream observes them
    // before its kernel reads/atomicAdds _cpNum.
    CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, 5 * sizeof(uint32_t), 0));
    CUDA_SAFE_CALL(cudaMemsetAsync(_gpNum, 0, sizeof(uint32_t), 0));
    CUDA_SAFE_CALL(cudaMemsetAsync(_gdCollapse, 0, sizeof(int), 0));  // [d-floor fail-fast] reset per detection
    CUDA_SAFE_CALL(cudaEventRecord(m_aux_reset_event, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(m_aux_stream, m_aux_reset_event, 0));

    // bvh_f on default stream, bvh_e on aux stream -> overlap.
    // Both atomicAdd into _cpNum & _collisionPair; CUDA atomics handle
    // cross-stream contention correctly. Pair-set order doesn't matter
    // to consumers (they iterate 0..h_cpNum[0]).
    // [xenv pin] isolate the two detection passes to localize the asymmetry:
    //   STIFF_SKIP_F=1 → only edge-edge (tests EE ownership obj_idx<self_eid)
    //   STIFF_SKIP_E=1 → only point-triangle (tests Morton/candidate; PT has no index dedup)
    set_ee_nodedup(getenv("STIFF_EE_NODEDUP") ? 1 : 0);
    set_ee_detgate(m_mode_config.ee_detgate ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);  // [env-part B]
    // [perenv-par] per-vertex cross-env skip at self-collision emission (see note above). null = off.
    set_self_p2g((m_mode_config.decouple_thresh && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(m_mode_config.ee_canon ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    { static int _tc = 0; set_ee_trace((getenv("STIFF_EE_TRACE") && _tc++ == 0) ? 1 : 0); }  // first buildCP (iter0) only
    set_ee_tgt(getenv("STIFF_BAR_TGT0")?atoi(getenv("STIFF_BAR_TGT0")):-1, getenv("STIFF_BAR_TGT1")?atoi(getenv("STIFF_BAR_TGT1")):-1);
    // [env-det] build the global→env-local vertex id map once (canon total-order tie-break).
    if(m_mode_config.ee_canon && m_d_p2g && !m_vloc_built)
    {
        std::vector<int> hp(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec;
        for(int v=0; v<vertexNum; v++){ int g=hp[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_vloc, (size_t)vertexNum*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(m_d_vloc, vloc.data(), (size_t)vertexNum*sizeof(int), cudaMemcpyHostToDevice));
        set_ee_vloc(m_d_vloc); m_vloc_built = true;
    }
    if(getenv("STIFF_STACK_DIAG")) reset_max_stack();
    // [env-det dump] one-shot dump of the edge BVH Morton hashes + edges to verify env0/env1 trees
    // are byte-identical modulo the env bit. STIFF_MCDUMP.
    { static int _mdc=0;
      if(getenv("STIFF_MCDUMP") && _mdc++==1){   // fire on 2nd buildCP (frame-0 pre-solve, still mirror)
        int nE=(int)bvh_e.edge_number;
        std::vector<uint64_t> hm(nE); std::vector<uint2> he(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hm.data(), bvh_e._MChash, (size_t)nE*sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
        std::vector<uint32_t> hi(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hi.data(), bvh_e._indices, (size_t)nE*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        FILE*f=fopen("/tmp/xd_mch.bin","wb"); fwrite(hm.data(),sizeof(uint64_t),nE,f); fclose(f);
        f=fopen("/tmp/xd_edges2.bin","wb"); fwrite(he.data(),sizeof(uint2),nE,f); fclose(f);
        f=fopen("/tmp/xd_idx.bin","wb"); fwrite(hi.data(),sizeof(uint32_t),nE,f); fclose(f);
        printf("[mcdump] dumped %d edge MChash + edges + indices\n", nE); } }
    if(!getenv("STIFF_SKIP_F")) bvh_f.SelfCollitionDetect(dHat);
    if(!getenv("STIFF_SKIP_E")) bvh_e.SelfCollitionDetect(dHat, m_aux_stream);
    if(getenv("STIFF_STACK_DIAG")) { CUDA_SAFE_CALL(cudaDeviceSynchronize());
        static int _sd=0; if(_sd++<3) printf("[stack] max traversal depth = %d (cap 2048)\n", get_max_stack()); }
    GroundCollisionDetect();
    // Join the auxiliary detector back into PTDS without blocking the host.
    // The following count D2H remains the algorithmic host-control wait.
    CUDA_SAFE_CALL(cudaEventRecord(m_aux_done_event, m_aux_stream));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(
        cudaStreamPerThread, m_aux_done_event, 0));

    // [C6-l canon-slots] must run before the snapshot and any consumer.
    if(m_mode_config.ee_canon)
        canonicalizePairSlots();

    if(m_ls_defer_counts)
    {   // [B3 trial-defer] mirror stays invalid during trials; energies use
        // the slacked bounds + device live counts; overflow via the monotone
        // counter on the decision read. MIRROR_AUDIT proves no stale reader.
        CUDA_SAFE_CALL(cudaMemcpyAsync(m_pair_snap_cur,
                                       _cpNum,
                                       6 * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        return;
    }
    {   // [9d28824-port] contiguous _cpNum[0:5]+_gpNum[5]: one 6-int D2H.
        uint32_t cp_gp_buf[6];
        CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        memcpy(h_cpNum.refresh_dst(), cp_gp_buf, 5 * sizeof(uint32_t));
        h_gpNum = cp_gp_buf[5];
        note_pair_census_peak();
    }

    // Overflow → grow DCD pair buffers + redo detection (BVH unchanged, no pairs
    // lost; emits were redirected to the trash slot so nothing was corrupted).
    while((int)h_cpNum[0] > MAX_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_cpNum[0] + h_cpNum[0] / 2) + 1;
        printf("[DCD-grow] h_cpNum=%u > cap=%d -> grow to %d, redo detection\n",
               h_cpNum[0], MAX_COLLITION_PAIRS_NUM, newcap);
        // [v0.8.6 2b] mechanics (alloc, +1 trash, DCD<=CCD lockstep, cap
        // publication) live in contact/pair_buffers.cuh; policy stays here.
        pair_buffers_grow_dcd(PairBuffers{_collisonPairs, _MatIndex, _ccd_collisonPairs, MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM}, newcap);
        bvh_f._collisionPair     = bvh_e._collisionPair     = _collisonPairs;
        bvh_f._MatIndex          = bvh_e._MatIndex          = _MatIndex;
        bvh_f._ccd_collisionPair = bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, 5 * sizeof(uint32_t), 0));
        CUDA_SAFE_CALL(cudaMemsetAsync(_gpNum, 0, sizeof(uint32_t), 0));
        CUDA_SAFE_CALL(cudaMemsetAsync(_gdCollapse, 0, sizeof(int), 0));  // [d-floor fail-fast] reset per detection
        // Preserve the v0.8.4.2 grow-redo ordering, using the same persistent
        // PTDS↔aux event pair as the first pass.
        CUDA_SAFE_CALL(cudaEventRecord(m_aux_reset_event, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(m_aux_stream, m_aux_reset_event, 0));
        bvh_f.SelfCollitionDetect(dHat);
        bvh_e.SelfCollitionDetect(dHat, m_aux_stream);
        GroundCollisionDetect();
        CUDA_SAFE_CALL(cudaEventRecord(m_aux_done_event, m_aux_stream));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(
            cudaStreamPerThread, m_aux_done_event, 0));
        {   // [9d28824-port] one 6-int D2H
            uint32_t cp_gp_buf[6];
            CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            memcpy(h_cpNum.refresh_dst(), cp_gp_buf, 5 * sizeof(uint32_t));
            h_gpNum = cp_gp_buf[5];
            note_pair_census_peak();
        }
    }

    CUDA_SAFE_CALL(cudaMemcpyAsync(m_pair_snap_cur,
                                   _cpNum,
                                   6 * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    snapshotDcdCcdPairs();   // [narrow-self snapshot] before buildFullCP clobbers the mirror
    // [B3 trial-defer] during line-search trials the collapse flag rides the
    // piggybacked decision read instead (handleGroundCollapse at the consumer);
    // every non-trial caller keeps the immediate blocking check.
    if(!m_ls_defer_counts)
        throwIfGroundDistanceInvalid();
}

// [narrow-self snapshot] copy the DCD-time CCD mirror (first h_cpNum[0] slots of
// _ccd_collisonPairs, written by the DCD detect kernels at the same atomic slot
// as _collisionPair) into a dedicated immutable buffer. See GIPC.cuh for why.

// ============================================================================
// [C6-l canon-slots] Deterministic pair-slot order.
//
// The DCD emission assigns slots with atomicAdd, so the slot PERMUTATION of an
// identical pair set varies run to run. Every order-dependent consumer
// inherits that: the deterministic gradient deposits and the order-free
// duplicate-block merge are immune, but the LS energy reductions sum pair
// energies in slot order, and in the recorded whole-frame replay the schedule
// jitter is large enough that one towel run in two flipped a knife-edge LS
// decision at first contact and exploded (newton=1000, ls=20586).
//
// Fix at the source: one stable two-pass lexicographic radix sort of the pair
// slots (plus the ground list) right after emission. Slot order becomes a pure
// function of the pair SET. Everything is fixed-shape at MAX capacity with
// device-side live masking (pads key to MAX and sink to the tail), so the
// recorded launches are identical every frame and nothing allocates in
// capture -- the lazy allocations below are trained by the pre-capture dry
// run, exactly like every other capacity axis.
// ============================================================================
__device__ __forceinline__ unsigned int _canon_ord32(int v)
{
    return static_cast<unsigned int>(v) ^ 0x80000000u;
}

__global__ void _canon_keys_zw(const int4*     pairs,
                               const uint32_t* live,
                               uint64_t*       keys,
                               uint32_t*       index,
                               int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity)
        return;
    index[i] = static_cast<uint32_t>(i);
    if(i < static_cast<int>(*live))
    {
        const int4 q = pairs[i];
        keys[i] = (static_cast<uint64_t>(_canon_ord32(q.z)) << 32)
                  | _canon_ord32(q.w);
    }
    else
        keys[i] = ~0ull;
}

__global__ void _canon_keys_xy_gather(const int4*     pairs,
                                      const uint32_t* live,
                                      const uint32_t* index,
                                      uint64_t*       keys,
                                      int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity)
        return;
    const uint32_t j = index[i];
    if(j < *live)
    {
        const int4 q = pairs[j];
        keys[i] = (static_cast<uint64_t>(_canon_ord32(q.x)) << 32)
                  | _canon_ord32(q.y);
    }
    else
        keys[i] = ~0ull;
}

__global__ void _canon_gather(const int4*     pairs,
                              const int*      mat,
                              const int4*     ccd,
                              const uint32_t* index,
                              const uint32_t* live,
                              int4*           pairs_out,
                              int*            mat_out,
                              int4*           ccd_out,
                              int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity || i >= static_cast<int>(*live))
        return;
    const uint32_t j = index[i];
    pairs_out[i] = pairs[j];
    mat_out[i]   = mat[j];
    ccd_out[i]   = ccd[j];
}

__global__ void _canon_writeback(const int4*     pairs_in,
                                 const int*      mat_in,
                                 const int4*     ccd_in,
                                 const uint32_t* live,
                                 int4*           pairs,
                                 int*            mat,
                                 int4*           ccd,
                                 int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity || i >= static_cast<int>(*live))
        return;
    pairs[i] = pairs_in[i];
    mat[i]   = mat_in[i];
    ccd[i]   = ccd_in[i];
}

__global__ void _canon_gp_keys(const uint32_t* gp,
                               const uint32_t* live,
                               uint64_t*       keys,
                               uint32_t*       index,
                               int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity)
        return;
    index[i] = static_cast<uint32_t>(i);
    keys[i] = i < static_cast<int>(*live) ? static_cast<uint64_t>(gp[i])
                                          : ~0ull;
}

__global__ void _canon_gp_gather(const uint32_t* gp,
                                 const uint32_t* index,
                                 const uint32_t* live,
                                 uint32_t*       gp_out,
                                 int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity || i >= static_cast<int>(*live))
        return;
    gp_out[i] = gp[index[i]];
}

__global__ void _canon_gp_writeback(const uint32_t* gp_in,
                                    const uint32_t* live,
                                    uint32_t*       gp,
                                    int             capacity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= capacity || i >= static_cast<int>(*live))
        return;
    gp[i] = gp_in[i];
}

void GIPC::canonicalizePairSlots()
{
    const int cap    = MAX_COLLITION_PAIRS_NUM;
    const int gp_cap = static_cast<int>(surf_vertexNum);
    if(cap <= 0)
        return;
    if(!m_canon_ready)
    {
        // Lazy one-time allocation; the pre-capture dry run takes this branch
        // outside capture, so the recorded path never allocates.
        for(int b = 0; b < 2; ++b)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_keys[b],
                                      (size_t)cap * sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_idx[b],
                                      (size_t)cap * sizeof(uint32_t)));
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_pairs_tmp,
                                  (size_t)cap * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_ccd_tmp,
                                  (size_t)cap * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_mat_tmp,
                                  (size_t)cap * sizeof(int)));
        if(gp_cap > 0)
            CUDA_SAFE_CALL(cudaMalloc((void**)&m_canon_gp_tmp,
                                      (size_t)gp_cap * sizeof(uint32_t)));
        size_t bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, bytes,
                                        m_canon_keys[0], m_canon_keys[1],
                                        m_canon_idx[0], m_canon_idx[1],
                                        cap);
        m_canon_sort_tmp_bytes = bytes + 256;
        CUDA_SAFE_CALL(cudaMalloc(&m_canon_sort_tmp,
                                  m_canon_sort_tmp_bytes));
        m_canon_ready = true;
    }
    const int tn = 256;
    const int bn = (cap + tn - 1) / tn;
    // Pass 1: minor key (z, w). Radix sort is stable, so sorting minor first
    // and major second yields the full lexicographic (x, y, z, w) order.
    _canon_keys_zw<<<bn, tn, 0, cudaStreamPerThread>>>(
        _collisonPairs, _cpNum, m_canon_keys[0], m_canon_idx[0], cap);
    size_t bytes = m_canon_sort_tmp_bytes;
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
        m_canon_sort_tmp, bytes, m_canon_keys[0], m_canon_keys[1],
        m_canon_idx[0], m_canon_idx[1], cap, 0, 64, cudaStreamPerThread));
    // Pass 2: major key (x, y), gathered through pass 1's permutation.
    _canon_keys_xy_gather<<<bn, tn, 0, cudaStreamPerThread>>>(
        _collisonPairs, _cpNum, m_canon_idx[1], m_canon_keys[0], cap);
    bytes = m_canon_sort_tmp_bytes;
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
        m_canon_sort_tmp, bytes, m_canon_keys[0], m_canon_keys[1],
        m_canon_idx[1], m_canon_idx[0], cap, 0, 64, cudaStreamPerThread));
    // m_canon_idx[0] now holds the lexicographic permutation.
    _canon_gather<<<bn, tn, 0, cudaStreamPerThread>>>(
        _collisonPairs, _MatIndex, _ccd_collisonPairs, m_canon_idx[0],
        _cpNum, m_canon_pairs_tmp, m_canon_mat_tmp, m_canon_ccd_tmp, cap);
    _canon_writeback<<<bn, tn, 0, cudaStreamPerThread>>>(
        m_canon_pairs_tmp, m_canon_mat_tmp, m_canon_ccd_tmp, _cpNum,
        _collisonPairs, _MatIndex, _ccd_collisonPairs, cap);
    if(gp_cap > 0 && m_canon_gp_tmp)
    {
        const int gbn = (gp_cap + tn - 1) / tn;
        _canon_gp_keys<<<gbn, tn, 0, cudaStreamPerThread>>>(
            _environment_collisionPair, _gpNum, m_canon_keys[0],
            m_canon_idx[0], gp_cap);
        bytes = m_canon_sort_tmp_bytes;
        CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
            m_canon_sort_tmp, bytes, m_canon_keys[0], m_canon_keys[1],
            m_canon_idx[0], m_canon_idx[1], gp_cap, 0, 64,
            cudaStreamPerThread));
        _canon_gp_gather<<<gbn, tn, 0, cudaStreamPerThread>>>(
            _environment_collisionPair, m_canon_idx[1], _gpNum,
            m_canon_gp_tmp, gp_cap);
        _canon_gp_writeback<<<gbn, tn, 0, cudaStreamPerThread>>>(
            m_canon_gp_tmp, _gpNum, _environment_collisionPair, gp_cap);
    }
}

void GIPC::snapshotDcdCcdPairs()
{
    m_dcd_snap_count = h_cpNum[0];
    if(m_dcd_snap_count == 0)
        return;
    if((int)m_dcd_snap_count > m_dcd_snap_cap)
    {
        ++pcg_buffer_generation();  // [C-1] captured snapshot pointer moves
        if(_dcd_ccd_snapshot) CUDA_SAFE_CALL(cudaFree(_dcd_ccd_snapshot));
        m_dcd_snap_cap = (int)(m_dcd_snap_count + m_dcd_snap_count / 2) + 1;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_dcd_ccd_snapshot,
                                  (size_t)m_dcd_snap_cap * sizeof(int4)));
    }
    // [C-1 capture-safe] async D2D: stream-ordered like every consumer, and
    // the sync variant is illegal inside graph capture. The grow branch above
    // cannot fire mid-line-search (the mirror is frozen for the whole LS).
    CUDA_SAFE_CALL(cudaMemcpyAsync(_dcd_ccd_snapshot, _ccd_collisonPairs,
                                   (size_t)m_dcd_snap_count * sizeof(int4),
                                   cudaMemcpyDeviceToDevice, 0));
}

// [C4-a] device-count snapshot: the host never learns the live DCD count
// inside the frame graph, so the copy masks itself against the pair-count
// snapshot block instead of a host extent.
__global__ void _snapshot_pairs_masked(const int4*     source,
                                       int4*           destination,
                                       const uint32_t* d_pair_counts,
                                       int             capacity)
{
    const uint32_t live  = d_pair_counts[0];
    const uint32_t bound = live < (uint32_t)capacity ? live : (uint32_t)capacity;
    for(uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        index < bound;
        index += gridDim.x * blockDim.x)
        destination[index] = source[index];
}

// [C4-a] frame-boundary training: everything the in-graph collision chain
// touches must be at final capacity before capture (no cudaMalloc inside a
// captured stream). Growth here bumps the buffer generation so stale
// executables are re-recorded.
// [C6] Choose the extents the recorded launches will be shaped by. Driven by
// the counts the scene actually produced (this frame's mirrors), doubled for
// headroom, tier-rounded, and clamped by the hard emission capacity. Never
// MAX_COLLITION_PAIRS_NUM itself: that is the worst case the buffers COULD
// hold, not what the scene needs, and shaping the triplet envelope from it
// costs tens of GB on real scenes.
void GIPC::update_graph_training_capacity()
{
    // [C6-q] Episodes have no per-frame fallback: an overflow mid-episode
    // aborts the whole episode. The C6-p default of headroom 1 is a step-mode
    // throughput choice; episode captures keep the old 2x margin.
    const int headroom =
        std::max(graph_train_headroom_num(), m_episode_capture ? 2 : 1);
    for(int slot = 0; slot < 5; ++slot)
    {
        const int want = gipc::assembly_capacity_tier(
            std::max(256,
                     static_cast<int>(std::max(m_peak_cpNum[slot],
                                               h_cpNum[slot]))
                         * headroom));
        m_graph_train_cp[slot] =
            std::max(m_graph_train_cp[slot],
                     std::min(want, MAX_COLLITION_PAIRS_NUM));
    }
    // [C6-f] Slot 0 stays TIERED (peak * headroom, from the loop above).
    //
    // C6-d pinned it to MAX_COLLITION_PAIRS_NUM on the argument that it does not
    // feed the triplet envelope, so widening it costs no memory. That is true of
    // memory and false of time: it is the launch extent for every per-pair
    // contact kernel, so the recorded frame ran 737196-wide grids over a scene
    // with ~28k live pairs. Paired measurement on foldshirt, same binary, same
    // machine state, 60 frames: graph-off 341.5 ms/frame vs graph-on 629.5 ms
    // -- the graph was 1.84x SLOWER, and foldshirt_finray 276.4 -> 692.3 ms.
    //
    // The reason C6-d had to widen it is now gone: the tier guard reports WHICH
    // axis crossed with a per-axis deficit (see _pair_tier_guard), growth
    // targets that axis alone, and a retry rebuilds the frame boundary, so a
    // saturating slot 0 recovers in one retry instead of climbing 65536 ->
    // 262144 -> 737196 one attempt at a time.
    // [C6-g] Slot 0 defaults back to worst case. Tiering it is what we WANT --
    // it is the launch extent for every per-pair contact kernel, so pinning it
    // costs 1.6x..10.8x on the replay scenes and OOMs small ones (towel, 961
    // verts, asked for 737196-pair capacity) -- but it is blocked on a deeper
    // defect: with tiered extents the contact counts sit near tier boundaries,
    // the counts are racy (two-stream atomicAdd emission), so a frame overflows
    // and RETRIES on some runs and not others, and a retry perturbs the
    // trajectory at ~1e-6. That shows up as G18 passing or failing marginally
    // run to run (2.9e-7..3.5e-6 against a 2.45e-7 envelope) and as
    // towel_scramble's crumple metric spreading 0.77..1.03 where the host is a
    // deterministic 0.905. Raising the headroom to 4 did not stabilise it.
    //
    // Default is TIERED, because the alternative is a hard crash: pinned to
    // worst case, towel_scramble (961 verts) and case39 both OOM at frame 1.
    // A hard crash beats a marginal numeric drift, and G18 turns out to be
    // marginally flaky INDEPENDENT of this knob -- reproducing the exact
    // pre-C6-g configuration still gives friction:kappas 0.360 against a 0.338
    // envelope on some runs. Until that envelope is re-derived from a multi-run
    // baseline the gate cannot adjudicate this trade-off. Worst case stays
    // available under STIFF_GRAPH_UNTIER_PAIR_SLOT0=1.
    if(getenv("STIFF_GRAPH_UNTIER_PAIR_SLOT0"))
        m_graph_train_cp[0] = MAX_COLLITION_PAIRS_NUM;
    m_graph_train_pairs = m_graph_train_cp[0];
    // Remember what a REAL frame actually assembled. The a-priori bound below
    // models FEM/contact/ground/friction but not ABD body Hessians, joints or
    // stitch springs, so for gripper scenes it underestimates badly; the
    // measured length is the only honest starting point.
    if(gipc_global_triplet.global_triplet_offset > m_last_assembled_triplets)
        m_last_assembled_triplets =
            gipc_global_triplet.global_triplet_offset;
    const int observed_ccd = static_cast<int>(
        std::max(m_peak_ccd_pair_count, m_last_ccd_pair_count));
    const int want_ccd = gipc::assembly_capacity_tier(
        std::max(1024, observed_ccd * headroom));
    m_graph_train_ccd =
        std::max(m_graph_train_ccd,
                 std::min(want_ccd, MAX_CCD_COLLITION_PAIRS_NUM));
    // [C6-c] The ground axis is trained to its WORST CASE, never to an observed
    // count. Unlike the DCD and swept axes it has no OVF_* bit, so an in-graph
    // truncation is undetectable and can never be adjudicated into a retry with
    // a larger tier. It is also the axis most likely to be observed as zero:
    // only frame 0 runs on the host, and scenes routinely start with the cloth
    // in the air (h_gpNum == 0), which pinned the extent at the 256 floor for
    // the whole run. foldshirt then touched down around frame 8, the recorded
    // ground assembly covered 256 of the live constraints, the rest lost their
    // barrier, vertices sank into the plane, and the ground CCD lane collapsed
    // from ~0.1 (host) to 2.7e-9 — which is what killed the line search.
    //
    // Worst case is one constraint per surface vertex, which is what the
    // ground buffers and the assembly envelope (contact_scalar_extent) are
    // already sized for, so this costs launch width and nothing else.
    m_graph_train_ground = static_cast<int>(surf_vertexNum);
    if(getenv("STIFF_FRAME_GRAPH_DIAG"))
        fprintf(stderr,
                "[graph-train] observed cp=[%u,%u,%u,%u,%u] gp=%u -> trained "
                "cp=[%d,%d,%d,%d,%d] gp=%d (caps pair=%d ccd=%d)\n",
                h_cpNum[0], h_cpNum[1], h_cpNum[2], h_cpNum[3], h_cpNum[4],
                (unsigned)h_gpNum,
                m_graph_train_cp[0], m_graph_train_cp[1], m_graph_train_cp[2],
                m_graph_train_cp[3], m_graph_train_cp[4], m_graph_train_ground,
                MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
}

// [C6-e] friction lastH family at final capacity. Hoisted out of
// train_collision_graph_capacities so the frame-boundary buildFrictionSets can
// call it FIRST: these are resize_DISCARD growths, so running them after the
// boundary build threw away the lagged sets that had just been computed. The
// lastH emission is an unbounded atomicAdd, which is why the capacity has to be
// the worst case rather than a trained tier.
void GIPC::ensure_graph_friction_capacity()
{
#ifdef USE_FRICTION
    if(static_cast<size_t>(MAX_COLLITION_PAIRS_NUM) > m_fric_cp_cap)
    {
        ++pcg_buffer_generation();
        const size_t n = static_cast<size_t>(MAX_COLLITION_PAIRS_NUM);
        lambda_lastH_scalar.resize_discard(n);
        distCoord.resize_discard(n);
        tanBasis.resize_discard(n);
        _collisonPairs_lastH.resize_discard(n);
        m_fric_cp_cap = n;
    }
    if(static_cast<size_t>(surf_vertexNum) > m_fric_gd_cap)
    {
        ++pcg_buffer_generation();
        const size_t n = static_cast<size_t>(surf_vertexNum);
        lambda_lastH_scalar_gd.resize_discard(n);
        _collisonPairs_lastH_gd.resize_discard(n);
        m_fric_gd_cap = n;
    }
#endif
}

void GIPC::train_collision_graph_capacities()
{
    update_graph_training_capacity();
    if(m_dcd_snap_cap < MAX_COLLITION_PAIRS_NUM)
    {
        ++pcg_buffer_generation();
        if(_dcd_ccd_snapshot)
            CUDA_SAFE_CALL(cudaFree(_dcd_ccd_snapshot));
        m_dcd_snap_cap = MAX_COLLITION_PAIRS_NUM;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_dcd_ccd_snapshot,
                                  (size_t)m_dcd_snap_cap * sizeof(int4)));
    }
    int scratch_extent = MAX_CCD_COLLITION_PAIRS_NUM;
    if(scratch_extent < m_dcd_snap_cap)
        scratch_extent = m_dcd_snap_cap;
    if(scratch_extent < static_cast<int>(surf_vertexNum))
        scratch_extent = static_cast<int>(surf_vertexNum);
    (void)ensure_reduce_scratch(scratch_extent);

    // [C4-b] close-set buffers and the doubling flag reach final capacity
    // here so the in-graph postLineSearch equivalent never allocates.
    if(static_cast<size_t>(surf_vertexNum) > m_close_gp_cap)
    {
        ++pcg_buffer_generation();
        if(_closeConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
        }
        m_close_gp_cap = static_cast<size_t>(surf_vertexNum);
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID,
                                  m_close_gp_cap * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal,
                                  m_close_gp_cap * sizeof(double)));
    }
    // Close-set emission is an unbounded atomicAdd (no capacity clamp in
    // _calSelfCloseVal), so this buffer MUST cover the worst case — training
    // it down corrupts memory the moment live pairs exceed the tier. It is
    // small (24 B/pair); the envelope that actually had to shrink is the
    // triplet stream below.
    if(static_cast<size_t>(MAX_COLLITION_PAIRS_NUM) > m_close_cp_cap)
    {
        ++pcg_buffer_generation();
        if(_closeMConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
        }
        m_close_cp_cap = static_cast<size_t>(MAX_COLLITION_PAIRS_NUM);
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID,
                                  m_close_cp_cap * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal,
                                  m_close_cp_cap * sizeof(double)));
    }
    if(!m_d_close_flag)
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_close_flag, sizeof(int)));

#ifdef USE_FRICTION
    ensure_graph_friction_capacity();
#endif

    // The recorded assembly is shaped by capacity mirrors (all pair counts at
    // MAX), so every allocation the recording would otherwise trigger inside
    // capture must land here, at the legal frame-boundary discard window.
    // Mirrors 13's dynamic frame-start grow evaluated at capacity counts.
    if(m_dynamic_triplet)
    {
        // [C6] per-ARITY envelope. The old form multiplied ONE pair tier by
        // all three arity strides, i.e. it budgeted as if every pair were
        // simultaneously PP, PE and PT — 3x over the true bound, on top of
        // using the worst-case pair count.
        const long long contact_tier =
            static_cast<long long>(
                gipc::assembly_capacity_tier(m_graph_train_cp[4])) * M12_Off
            + static_cast<long long>(
                gipc::assembly_capacity_tier(m_graph_train_cp[3])) * M9_Off
            + static_cast<long long>(
                gipc::assembly_capacity_tier(m_graph_train_cp[2])) * M6_Off;
        long long bound = m_fixed_triplet_base
                          + static_cast<long long>(
                              abd_fem_count_info.fem_point_num)
                          + contact_tier
                          + contact_scalar_extent(
                              static_cast<int>(surf_vertexNum));
#ifdef USE_FRICTION
        // [C4-c] the recording shapes friction extents from capacity
        // mirrors too — budget the lagged tiers at the same MAX counts.
        bound += contact_tier + contact_scalar_extent(
                     static_cast<int>(surf_vertexNum));
#endif
        bound += 4096;

        // Partition staging envelope at capacity payload (13's txn block):
        // sort capacity + stable region + staging region must all fit.
        const int sort_capacity = gipc::assembly_capacity_tier(
            static_cast<int>(contact_tier));
        // [C4/D4] capacity-shaped recordings sort over zero-padded tails,
        // and the pads land in the partition's class-0 census — so the
        // in-graph OVF_TRIPLETS guard sees counts up to the full payload.
        // Train the class tiers to the capacity envelope once, here, so a
        // reused executable can never trip a tier the boundary retry path
        // (absent inside an episode) would have had to grow. Pads carry
        // neutral zero triplets, so oversized segments stay value-exact.
        // Only classes the scene can actually populate are trained: the
        // ABD lifting kernels launch from their class extents and would
        // dereference null Jacobi tables in a pure-FEM scene (sanitizer-
        // confirmed) if an impossible class were given a nonzero tier.
        // [C6] Train each contact class from ITS OWN observed count, not from
        // the (already capacity-inflated) payload tier. The old rule set every
        // class to the full payload tier, which compounds: each training pass
        // fed the previous pass's inflated payload back in, taking foldshirt's
        // assembled stream from 246k to 2.6M to 21M triplets across two passes
        // until the contraction ran past a 41M-element buffer.
        const bool has_abd = abd_fem_count_info.abd_body_num > 0;
        const bool has_fem = abd_fem_count_info.fem_point_num > 0;
        // [C6-b] Read the honest census, NOT the *_contact_num fields: under a
        // capacity mirror those hold class_tier, so training from them feeds
        // the previous tier back in as an observation and doubles the class-0
        // segment every frame (foldshirt: 242k -> 524k -> 1048k -> OOM, with
        // the ABD contraction pushed past a 2.9M-element triplet buffer).
        const int observed_class[4] = {
            gipc_global_triplet.m_observed_class_count[0],
            gipc_global_triplet.m_observed_class_count[1],
            gipc_global_triplet.m_observed_class_count[2],
            gipc_global_triplet.m_observed_class_count[3]};
        // m_contact_class_tier order: fem_fem, abd_fem, fem_abd, abd_abd
        // (see partitionContactHessian's class_num assignments). Class 0
        // is ALWAYS possible: capacity-grid zero pads carry hash(0,0) and
        // land in the first sorted class regardless of scene composition —
        // its consumer is a plain 3x3 pass-through, so oversizing it never
        // dereferences ABD tables. The ABD lifting classes stay gated on
        // actual scene population (null-Jacobi launch otherwise).
        // [D4-b] The zero pads carry hash(0,0), and (0,0) classifies by the
        // ID layout: index 0 is a FEM vertex when the scene has FEM points, so
        // the pads land in class 0 -- but in a PURE-ABD scene index 0 is an
        // ABD body and the pads classify as abd_abd (class 3). Granting the
        // pad allowance to class 0 unconditionally left class 3 at its 256
        // floor on the D4 scene, and the first recorded frame overflowed on
        // its own pads: class_counts=[0,0,0,10240] against tier 256, contact
        // load irrelevant. The allowance follows the pads' real class, and
        // that class is always trainable (has_abd holds when pad_class==3).
        const int pad_class = has_fem ? 0 : 3;
        if(getenv("STIFF_FRAME_GRAPH_DIAG"))
            fprintf(stderr,
                    "[class-train] has_fem=%d has_abd=%d pad_class=%d "
                    "sort_cap=%d contact_tier=%lld obs=[%d,%d,%d,%d] "
                    "tier_in=[%d,%d,%d,%d]\n",
                    (int)has_fem, (int)has_abd, pad_class, sort_capacity,
                    (long long)contact_tier, observed_class[0],
                    observed_class[1], observed_class[2], observed_class[3],
                    gipc_global_triplet.m_contact_class_tier[0],
                    gipc_global_triplet.m_contact_class_tier[1],
                    gipc_global_triplet.m_contact_class_tier[2],
                    gipc_global_triplet.m_contact_class_tier[3]);
        const bool class_possible[4] = {
            true, has_abd && has_fem, has_abd && has_fem, has_abd};
        for(int s = 0; s < 4; ++s)
        {
            if(!class_possible[s])
                continue;
            long long want = static_cast<long long>(observed_class[s])
                             * std::max(graph_train_headroom_num(),
                                        m_episode_capture ? 2 : 1);
            if(s == pad_class)
            {
                if(pad_class == 0)
                {
                    // FEM scenes: the C6-b allowance, unchanged.
                    want += std::max(0, sort_capacity
                                            - static_cast<int>(contact_tier));
                }
                else
                {
                    // Pure-ABD scenes: with zero real contact the ENTIRE
                    // padded payload is pads and every one of them lands in
                    // abd_abd, so the allowance must span all non-real slots
                    // of the sorted extent (D4: class_counts=[0,0,0,10240]
                    // against a 256-floor tier; the sort-tail-only allowance
                    // left required=10496 unreachable).
                    long long observed_total = 0;
                    for(int c = 0; c < 4; ++c)
                        observed_total += observed_class[c];
                    want += std::max<long long>(
                        0, sort_capacity - observed_total);
                }
            }
            const int tier = gipc::assembly_capacity_tier(
                static_cast<int>(std::max<long long>(256, want)));
            if(gipc_global_triplet.m_contact_class_tier[s] < tier)
            {
                gipc_global_triplet.m_contact_class_tier[s] = tier;
                ++pcg_buffer_generation();
            }
        }
        long long stable_count = 0;
        for(int s = 0; s < 4; ++s)
            stable_count += gipc_global_triplet.m_contact_class_tier[s];
        const long long staging_base =
            sort_capacity > stable_count ? sort_capacity : stable_count;
        const long long staging_need = staging_base + stable_count;

        long long target = 2 * bound;
        if(target < staging_need)
            target = staging_need;
        // The ABD contraction reads [offset + fem_fem, 2*offset): the buffer
        // must hold twice the assembled length. Scale the measured length by
        // the capacity-mirror inflation the recording will apply.
        if(m_last_assembled_triplets > 0)
        {
            const int observed = std::max(1, (int)h_cpNum[0]);
            const double inflation =
                std::max(1.0, (double)m_graph_train_cp[0] / observed);
            const long long measured_need =
                2 * (long long)(m_last_assembled_triplets * inflation) + 8192;
            if(target < measured_need)
                target = measured_need;
        }
        if(gipc_global_triplet.triplet_capacity()
           < static_cast<size_t>(target))
        {
            gipc_global_triplet.open_discard_window();
            gipc_global_triplet.ensure_capacity_discard(
                static_cast<size_t>(target));
            const size_t cap = gipc_global_triplet.triplet_capacity();
            CUDA_SAFE_CALL(cudaMemsetAsync(
                gipc_global_triplet.block_values(),
                0,
                cap * 9 * sizeof(double),
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(
                gipc_global_triplet.block_row_indices(),
                0,
                cap * sizeof(int),
                cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaMemsetAsync(
                gipc_global_triplet.block_col_indices(),
                0,
                cap * sizeof(int),
                cudaStreamPerThread));
        }
        long long hash_need =
            static_cast<long long>(static_cast<double>(bound) * 1.1);
        if(hash_need < sort_capacity)
            hash_need = sort_capacity;
        if(gipc_global_triplet.global_external_max_capcity < hash_need)
        {
            gipc_global_triplet.resize_collision_hash_size(
                static_cast<size_t>(hash_need));
            gipc_global_triplet.global_external_max_capcity =
                static_cast<int>(hash_need);
        }

        // muda's radix sort grows its temp workspace on demand — train it
        // here with the exact capture-time extent so the recorded SortPairs
        // reuses the workspace instead of allocating inside capture. Buffer
        // contents are irrelevant; only the size signature matters.
        if(sort_capacity > 0)
            muda::DeviceRadixSort().SortPairs(
                gipc_global_triplet.block_hash_value(),
                gipc_global_triplet.block_sort_hash_value(),
                gipc_global_triplet.block_index(),
                gipc_global_triplet.block_sort_index(),
                sort_capacity);

        // The final converter's merge-bin workspace is sized from the
        // capture-time tier of the full triplet stream — grow it here, not
        // inside the recorded convert.
        if(m_global_linear_system)
            m_global_linear_system->train_converter_capacity(
                gipc::assembly_capacity_tier(static_cast<int>(bound)));
        // The ABD slice converter's merge bin sees abd_abd_contact_num at
        // the capacity-trained class tier during recording — train it to
        // the same envelope.
        if(m_abd_system)
            m_abd_system->converter3x3.ensure_capacity(
                gipc::assembly_capacity_tier(
                    gipc_global_triplet.m_contact_class_tier[3]));
    }
}

void GIPC::snapshotDcdCcdPairsCapture()
{
    if(m_dcd_snap_cap <= 0)
        return;
    const int block = 256;
    const int grid =
        std::min(1024, (m_dcd_snap_cap + block - 1) / block);
    _snapshot_pairs_masked<<<grid, block, 0, cudaStreamPerThread>>>(
        _ccd_collisonPairs,
        _dcd_ccd_snapshot,
        m_pair_snap_cur,
        m_dcd_snap_cap);
}

// [C4-a] capacity-grid twin of self_largestFeasibleStepSize_DeviceOut: sweeps
// the DCD-time snapshot over its trained capacity with the live device count
// as the in-kernel mask (OOB lanes hold the identity 1.0, so regridding stays
// bitwise-neutral — same argument as the refined reduction).
void GIPC::self_largestFeasibleStepSize_DeviceOut_Masked(double slackness,
                                                         double* mqueue,
                                                         int capacity,
                                                         double* out_slot,
                                                         const uint32_t* d_live)
{
    const unsigned int threadNum = default_threads;
    int numbers  = capacity;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes,
        _dcd_ccd_snapshot,
        _moveDir,
        mqueue,
        slackness,
        numbers,
        m_ccd_alpha_invalid,
        kCcdInvalidGlobalNarrow,
        d_live);
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_min_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}

// [C4-a] final-count tier guard: assembly and energy kernels were recorded
// with launch bounds tiered from capture-time counts. They mask down against
// live device counts but can never launch up, so a frame whose final counts
// cross any captured tier must retry from a boundary (step() re-records; an
// episode consumer sees the per-frame RETRY status).
__global__ void _pair_tier_guard(const uint32_t* d_pair_counts,
                                 int t0,
                                 int t2,
                                 int t3,
                                 int t4,
                                 int tg,
                                 frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x || !frame)
        return;
    const int tier[5] = {t0, t2, t3, t4, tg};
    const int live[5] = {static_cast<int>(d_pair_counts[0]),
                         static_cast<int>(d_pair_counts[2]),
                         static_cast<int>(d_pair_counts[3]),
                         static_cast<int>(d_pair_counts[4]),
                         static_cast<int>(d_pair_counts[5])};
    frame->hw_dcd_pairs = live[0];
    frame->cp_count     = live[0];
    frame->gp_count     = live[4];
    // [C6-c] Report a requirement the host can ACT on. The crossing is usually
    // a sub-slot (PT/EE/PE/PP or ground), whose tier is far below the total
    // tier, so reporting the total left needed_dcd < m_graph_train_pairs and the
    // growth test never fired -- the frame retried at the identical tier until
    // the budget ran out. The host grows the total and then sets each sub-slot
    // to grown/2, so the total must reach twice the crossing count for the
    // sub-slot to actually clear it.
    // [C6-c/d] Report WHICH axis crossed, as a bitmask over the entries above
    // (0 -> cp[0], 1 -> cp[2], 2 -> cp[3], 3 -> cp[4], 4 -> ground), plus the
    // worst crossing count. Growing every sub-slot to half the total tier
    // instead -- the first attempt at this -- inflated cp[4] from 1024 to
    // 131072 on a 425-pair observation, and since that slot contributes a 12x12
    // block per pair the triplet envelope exploded from 367k to 4.74M and
    // overflowed the class tier on the very next attempt. Growth has to target
    // the axis that actually failed.
    // Pack, per entry, BOTH "did it cross" and "by how much" -- the deficit as
    // a power-of-two shift. Sizing every crossed axis from the single worst
    // crossing count still cross-contaminates: cp[4] crossed by a few hundred
    // pairs and was sized from cp[0]'s 65537, taking it from 1024 to 262144 on
    // a 425-pair observation. Layout: bits 0..14 hold five 3-bit shifts,
    // bits 15..19 hold the crossed mask. No ABI change -- it all rides
    // err_primitive.
    bool crossed      = false;
    int  crossed_mask = 0;
    int  shifts       = 0;
    int  worst_live   = 0;
    for(int i = 0; i < 5; ++i)
    {
        if(live[i] <= tier[i])
            continue;
        crossed = true;
        crossed_mask |= (1 << i);
        int shift = 1;
        int reach = tier[i] > 0 ? tier[i] : 1;
        while(reach < live[i] && shift < 7)
        {
            reach <<= 1;
            ++shift;
        }
        shifts |= (shift & 7) << (3 * i);
        if(live[i] > worst_live)
            worst_live = live[i];
    }
    if(!crossed)
        return;
    frame->required_dcd_pairs =
        live[0] > worst_live ? live[0] : worst_live;

    frame_fsm::fsm_record_error(frame,
                                frame_fsm::ERR_CAPACITY,
                                frame_fsm::OVF_DCD_PAIRS,
                                -1,
                                shifts | (crossed_mask << 15));
    atomicCAS(&frame->result,
              frame_fsm::FRAME_OK,
              frame_fsm::FRAME_RETRY_REQUIRED);
}

void GIPC::enqueue_pair_tier_guard()
{
    frame_fsm::FrameDeviceState* frame = frame_graph_device_state();
    if(!frame)
        return;
    _pair_tier_guard<<<1, 1, 0, cudaStreamPerThread>>>(
        m_pair_snap_cur,
        gipc::assembly_capacity_tier(static_cast<int>(h_cpNum[0])),
        gipc::assembly_capacity_tier(static_cast<int>(h_cpNum[2])),
        gipc::assembly_capacity_tier(static_cast<int>(h_cpNum[3])),
        gipc::assembly_capacity_tier(static_cast<int>(h_cpNum[4])),
        gipc::assembly_capacity_tier(static_cast<int>(h_gpNum)),
        frame);
}

// [C4-b] device-kappa injection point: non-null only while the whole-frame
// graph records a collision body. Points at FrameDeviceState::kappa, which
// frame_begin_init seeds from the per-launch pinned input and the in-graph
// close-set doubling advances between Newton iterations.
const double* GIPC::graph_kappa_dev() const
{
    if(!m_graph_kappa_armed)
        return nullptr;
    frame_fsm::FrameDeviceState* frame = frame_graph_device_state();
    return frame ? &frame->kappa : nullptr;
}

// [C4-b] conditional in-graph kappa doubling: the host postLineSearch does
// Kappa *= 2 followed by upperBoundKappa's scene-constant cap.
__global__ void _post_ls_kappa_double(const int* flag,
                                      double     kappa_max,
                                      frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x || !frame)
        return;
    if(*flag)
    {
        const double doubled = frame->kappa * 2.0;
        frame->kappa = doubled > kappa_max ? kappa_max : doubled;
    }
}

// [C4-b] the postLineSearch equivalent, enqueue form (merged scalar-kappa
// path): check the previous iteration's close set entirely on device, double
// FrameDeviceState::kappa when any constraint tightened, then rebuild the
// close set at capacity grids with device live counts. Mirrors the host
// sequence check -> double+cap -> reset counters -> recompute.
void GIPC::enqueue_post_ls_kappa_conditional()
{
    frame_fsm::FrameDeviceState* frame = frame_graph_device_state();
    if(!frame || !m_d_close_flag)
        return;
    const unsigned int threadNum = default_threads;

    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_d_close_flag, 0, sizeof(int), cudaStreamPerThread));
    if(m_close_gp_cap > 0)
    {
        const int capacity = static_cast<int>(m_close_gp_cap);
        const int blocks = (capacity + threadNum - 1) / threadNum;
        _checkGroundCloseVal<<<blocks, threadNum, 0, cudaStreamPerThread>>>(
            _vertexes,
            _groundOffset,
            _groundNormal,
            m_d_close_flag,
            _closeConstraintID,
            _closeConstraintVal,
            capacity,
            nullptr,
            nullptr,
            _close_gpNum);
    }
    if(m_close_cp_cap > 0)
    {
        const int capacity = static_cast<int>(m_close_cp_cap);
        const int blocks = (capacity + threadNum - 1) / threadNum;
        _checkSelfCloseVal<<<blocks, threadNum, 0, cudaStreamPerThread>>>(
            _vertexes,
            m_d_close_flag,
            _closeMConstraintID,
            _closeMConstraintVal,
            capacity,
            nullptr,
            nullptr,
            _close_cpNum);
    }

    // upperBoundKappa's cap is a scene constant — evaluate it once here.
    // (defined later in this composite TU, module 12)
    void compute_H_b(double d, double dHat, double& H);
    double H_b;
    double bb = bboxDiagSize2;
    if(!getenv("STIFF_DIAG_KAPPA_MERGEDBB") && absolute_dhat > 0.0
       && relative_dhat > 0.0)
        bb = (absolute_dhat * absolute_dhat)
             / (relative_dhat * relative_dhat);
    compute_H_b(1.0e-16 * bb, dHat, H_b);
    double kappa_max = 100 * minKappaCoef * meanMass / (4.0e-16 * bb * H_b);
    if(meanMass == 0.0)
        kappa_max = 100 * minKappaCoef / (4.0e-16 * bb * H_b);
    // [C6-b] The host's in-frame kappa doubling is INERT: checkCloseGroundVal
    // and checkSelfCloseVal gate on h_close_gpNum / h_close_cpNum, and neither
    // mirror is written anywhere in the tree, so both always return false and
    // Kappa never moves within a frame. That is a deliberate upstream decision
    // ("do not revive the legacy in-frame doubling path without a separately
    // validated adaptive-contact redesign", computeSelfCloseVal).
    //
    // This enqueue form reads the DEVICE counters, which the recompute below
    // does populate — so it faithfully revived the retired path and diverged
    // from the solver it is supposed to reproduce. On foldshirt's grasp-closing
    // frame it doubled kappa six times (41.73 -> 2670.70), which forced the
    // line search to collapse alpha to 2.4e-31 over 127 trials and killed the
    // frame; the host clears the same frame in 31 Newton iterations with kappa
    // pinned at 41.729624.
    //
    // Default is host-equivalent (no doubling). The kernel stays behind a knob
    // so the redesign has something to measure against.
    if(getenv("STIFF_GRAPH_LEGACY_KAPPA_DOUBLE"))
        _post_ls_kappa_double<<<1, 1, 0, cudaStreamPerThread>>>(
            m_d_close_flag, kappa_max, frame);

    CUDA_SAFE_CALL(cudaMemsetAsync(
        _close_gpNum, 0, sizeof(uint32_t), cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(
        _close_cpNum, 0, sizeof(uint32_t), cudaStreamPerThread));
    if(m_close_gp_cap > 0)
    {
        const int capacity = static_cast<int>(m_close_gp_cap);
        const int blocks = (capacity + threadNum - 1) / threadNum;
        _computeGroundCloseVal<<<blocks, threadNum, 0, cudaStreamPerThread>>>(
            _vertexes,
            _groundOffset,
            _groundNormal,
            _environment_collisionPair,
            dTol,
            _closeConstraintID,
            _closeConstraintVal,
            _close_gpNum,
            capacity,
            m_pair_snap_cur.data() + 5);
    }
    if(m_close_cp_cap > 0)
    {
        const int capacity = static_cast<int>(m_close_cp_cap);
        const int blocks = (capacity + threadNum - 1) / threadNum;
        _calSelfCloseVal<<<blocks, threadNum, 0, cudaStreamPerThread>>>(
            _vertexes,
            _collisonPairs,
            _closeMConstraintID,
            _closeMConstraintVal,
            _close_cpNum,
            dTol,
            capacity,
            m_pair_snap_cur.data() + 0);
    }
}

// [C4-a] the merged scalar-chain CCD alpha, enqueue form: no host reads, no
// grow-redo — validation and past-capacity retries ride FrameDeviceState.
void GIPC::enqueue_ccd_alpha_conditional()
{
    const double slackness_a = 0.9;
    const double slackness_m = 0.8;
    const double ccd_size    = 1.0;

    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_ccd_alpha_invalid, 0, sizeof(int), cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_refined_invalid,
                                   0,
                                   (1 + kEnvAlphaSlots) * sizeof(int),
                                   cudaStreamPerThread));
    const int g_did = surf_vertexNum >= 1 ? 1 : 0;
    if(g_did)
        ground_largestFeasibleStepSize_DeviceOut(
            slackness_a, pcg_data.squeue, m_ccd_alpha_slots + 0);
    // Count==0 degenerates every lane to the identity, so the combine may
    // consume the slot unconditionally.
    self_largestFeasibleStepSize_DeviceOut_Masked(
        slackness_m,
        ensure_reduce_scratch(m_dcd_snap_cap),
        m_dcd_snap_cap,
        m_ccd_alpha_slots + 1,
        m_pair_snap_cur);
    _ccd_initial_alpha_combine<<<1, 1>>>(
        m_ccd_alpha_slots, g_did, 1, m_ccd_alpha_invalid);

    m_ccd_defer_counts = true;
    buildBVH_FULLCCD(1.0, m_ccd_alpha_slots + 2);
    buildFullCP(1.0, m_ccd_alpha_slots + 2);
    m_ccd_defer_counts = false;

    cfl_largestSpeed_DeviceOut(pcg_data.squeue, m_ccd_alpha_slots + 3);
    // [C6] swept grid at the trained CCD extent; a past-extent swept count
    // is adjudicated in-graph (OVF_CCD_PAIRS -> boundary retry -> re-record
    // at a larger tier), so this never silently drops pairs.
    const int ccd_train = graph_trained_ccd_extent();
    self_full_largestFeasibleStepSize_DeviceOut(
        slackness_m,
        ensure_reduce_scratch(ccd_train),
        ccd_train,
        m_ccd_alpha_slots + 4,
        _cpNum);
    _ccd_final_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                       0,
                                       dHat,
                                       ccd_size,
                                       m_ccd_alpha_invalid,
                                       m_ccd_refined_invalid,
                                       _cpNum,
                                       frame_graph_device_state(),
                                       ccd_train);
}

void GIPC::throwIfGroundDistanceInvalid()
{
    if(!_gdCollapse)
        return;
    int collapsed = 0;
    if(h_gpNum > 0)
        CUDA_SAFE_CALL(cudaMemcpy(&collapsed, _gdCollapse, sizeof(int), cudaMemcpyDeviceToHost));
    handleGroundCollapse(collapsed);
}

// [B3 trial-defer] the collapse RESPONSE, callable with a value that arrived
// via the piggybacked decision read (no dedicated D2H). Rare path: position/
// normal readbacks below only run on an actual violation.
void GIPC::handleGroundCollapse(int collapsed)
{
    if(collapsed < 0)
    {
        const int vertex = -collapsed - 1;
        int body = -1;
        double3 position = make_double3(0.0, 0.0, 0.0);
        double3 normal = make_double3(0.0, 0.0, 0.0);
        double offset = 0.0;
        if(vertex >= 0 && vertex < static_cast<int>(vertexNum))
        {
            CUDA_SAFE_CALL(cudaMemcpy(
                &position, _vertexes + vertex, sizeof(double3), cudaMemcpyDeviceToHost));
            if(_point_body_id)
                CUDA_SAFE_CALL(cudaMemcpy(
                    &body, _point_body_id + vertex, sizeof(int), cudaMemcpyDeviceToHost));
        }
        CUDA_SAFE_CALL(cudaMemcpy(
            &normal, _groundNormal, sizeof(double3), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(
            &offset, _groundOffset, sizeof(double), cudaMemcpyDeviceToHost));
        const double distance = normal.x * position.x + normal.y * position.y
                              + normal.z * position.z - offset;
        // [iron-law demotion] Mid-run, with per-env freeze machinery live, a
        // ground-infeasible vertex is ENV-ATTRIBUTABLE — quarantine that env
        // instead of throwing and killing the healthy envs. Init-time
        // violations (before per-env machinery is live) still throw below.
        if(quarantineEnvOfVertex(vertex, distance))
        {
            CUDA_SAFE_CALL(cudaMemset(_gdCollapse, 0, sizeof(int)));
            return;
        }
        char message[1024];
        const char* reason = "ground distance is non-finite or non-positive";
        snprintf(message,
                 sizeof(message),
                 "[StiffGIPC] IPC invariant violation: %s. vertex=%d body=%d "
                 "position=(%.17e, %.17e, %.17e) normal=(%.17e, %.17e, %.17e) "
                 "offset=%.17e distance=%.17e.%s",
                 reason,
                 vertex,
                 body,
                 position.x,
                 position.y,
                 position.z,
                 normal.x,
                 normal.y,
                 normal.z,
                 offset,
                 distance,
                 " The logarithmic barrier requires strict distance > 0.");
        // [error-taxonomy] typed: derives from std::runtime_error, so existing
        // catch sites are unaffected; python surfaces pystiffgipc.GeometryError.
        throw gipc::GeometryError(message);
    }
}

static std::string ccdInvalidSources(int invalid)
{
    struct Source
    {
        int bit;
        const char* name;
    };
    static constexpr Source sources[] = {
        {kCcdInvalidGlobalGround, "global-ground"},
        {kCcdInvalidGlobalNarrow, "global-narrow-self"},
        {kCcdInvalidGlobalRefined, "global-refined-self"},
        {kCcdInvalidPerEnvGround, "per-env-ground"},
        {kCcdInvalidPerEnvNarrow, "per-env-narrow-self"},
        {kCcdInvalidPerEnvRefined, "per-env-refined-self"},
    };
    std::string result;
    for(const Source& source : sources)
    {
        if(!(invalid & source.bit)) continue;
        if(!result.empty()) result += "+";
        result += source.name;
    }
    return result.empty() ? "unknown" : result;
}

void throwForInvalidCcdMask(int invalid, const char* context)
{
    invalid &= kCcdInvalidEffectiveMask;
    if(invalid == 0) return;
    throw std::runtime_error(
        std::string("[StiffGIPC] invalid direct CCD alpha in ") + context
        + " (source=" + ccdInvalidSources(invalid)
        + "): candidate was non-finite, non-positive, or outside [0,1]. "
          "Refined candidates are effective only when their refinement gate consumes them; "
          "aborting instead of accepting an invalid Newton step.");
}

void validateFinalCcdStateOrThrow(const double* state, const char* context)
{
    throwForInvalidCcdMask(static_cast<int>(state[7]), context);
    const double alpha = state[5];
    if(std::isfinite(alpha) && alpha > 0.0 && alpha <= 1.0) return;

    char message[768];
    snprintf(message,
             sizeof(message),
             "[StiffGIPC] invalid final CCD alpha in %s: alpha=%.17e "
             "temp=%.17e maxSpeed=%.17e refined=%.17e alphaCFL=%.17e. "
             "A non-finite maxSpeed/CFL result is intentionally fail-fast; "
             "the step will not continue with a discarded NaN.",
             context,
             state[5],
             state[2],
             state[3],
             state[4],
             state[6]);
    throw std::runtime_error(message);
}

void GIPC::throwIfInvalidCcdAlpha(const char* context)
{
    if(!m_ccd_alpha_invalid)
        return;
    int invalid = 0;
    CUDA_SAFE_CALL(cudaMemcpy(
        &invalid, m_ccd_alpha_invalid, sizeof(int), cudaMemcpyDeviceToHost));
    throwForInvalidCcdMask(invalid, context);
}

void stiff_test_ccd_nan_max_speed_fail_fast()
{
    double host_state[8] = {
        0.5,
        0.5,
        0.5,
        std::numeric_limits<double>::quiet_NaN(),
        1.0,
        1.0,
        1.0,
        0.0,
    };
    double* device_state = nullptr;
    int* device_invalid = nullptr;
    int* device_refined_invalid = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_state, sizeof(host_state)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_invalid, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_refined_invalid, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(
        device_state, host_state, sizeof(host_state), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(device_invalid, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(device_refined_invalid, 0, sizeof(int)));
    _ccd_final_alpha_combine<<<1, 1>>>(device_state,
                                      1,
                                      1.0,
                                      1.0,
                                      device_invalid,
                                      device_refined_invalid,
                                      nullptr,
                                      nullptr);
    CUDA_SAFE_CALL(cudaMemcpy(
        host_state, device_state, sizeof(host_state), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(device_refined_invalid));
    CUDA_SAFE_CALL(cudaFree(device_invalid));
    CUDA_SAFE_CALL(cudaFree(device_state));
    validateFinalCcdStateOrThrow(host_state, "NaN max-speed regression");
    throw std::runtime_error(
        "[StiffGIPC] NaN max-speed regression did not trigger fail-fast");
}

// [multi-env P3a] segmented per-env reduction PRIMITIVE — the core machinery the
// block-diagonal solve needs (per-env residual / dot-products). For each entry i,
// route its |vec[i]|^2 (and a unit count) into bucket d_point_to_group[i]. Atomic
// bucketing here is fine for the read-only diagnostic; the in-solver version (P3a
// step2) must use fixed-order segmented reduce (cub::DeviceSegmentedReduce) so the
// per-env sums are order-deterministic.

