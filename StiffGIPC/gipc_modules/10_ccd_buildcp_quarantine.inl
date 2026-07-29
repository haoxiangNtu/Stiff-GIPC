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
    if(m_perenv_bvh && m_d_p2g)
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
void GIPC::train_collision_graph_capacities()
{
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

    // The recorded assembly is shaped by capacity mirrors (all pair counts at
    // MAX), so every allocation the recording would otherwise trigger inside
    // capture must land here, at the legal frame-boundary discard window.
    // Mirrors 13's dynamic frame-start grow evaluated at capacity counts.
    if(m_dynamic_triplet)
    {
        const int pair_tier =
            gipc::assembly_capacity_tier(MAX_COLLITION_PAIRS_NUM);
        const long long contact_tier =
            static_cast<long long>(pair_tier)
            * (M12_Off + M9_Off + M6_Off);
        long long bound = m_fixed_triplet_base
                          + static_cast<long long>(
                              abd_fem_count_info.fem_point_num)
                          + contact_tier
                          + contact_scalar_extent(
                              static_cast<int>(surf_vertexNum));
#ifdef USE_FRICTION
        bound += make_contact_triplet_tier(h_cpNum_last).tier_triplets
                 + contact_scalar_extent(h_gpNum_last);
#endif
        bound += 4096;

        // Partition staging envelope at capacity payload (13's txn block):
        // sort capacity + stable region + staging region must all fit.
        const int sort_capacity = gipc::assembly_capacity_tier(
            static_cast<int>(contact_tier));
        long long stable_count = 0;
        for(int s = 0; s < 4; ++s)
            stable_count += gipc_global_triplet.m_contact_class_tier[s];
        const long long staging_base =
            sort_capacity > stable_count ? sort_capacity : stable_count;
        const long long staging_need = staging_base + stable_count;

        long long target = 2 * bound;
        if(target < staging_need)
            target = staging_need;
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
    bool crossed        = false;
    for(int i = 0; i < 5; ++i)
        crossed = crossed || live[i] > tier[i];
    if(!crossed)
        return;
    frame->required_dcd_pairs = live[0];
    frame_fsm::fsm_record_error(frame,
                                frame_fsm::ERR_CAPACITY,
                                frame_fsm::OVF_DCD_PAIRS,
                                -1,
                                -1);
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
                                       _cpNum,
                                       frame_graph_device_state(),
                                       MAX_CCD_COLLITION_PAIRS_NUM);
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
