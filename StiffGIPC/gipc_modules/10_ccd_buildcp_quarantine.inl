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
                                         const int* refined_invalid)
{
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
    slots[7] = static_cast<double>(*invalid & kCcdInvalidEffectiveMask);
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
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
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

void GIPC::buildCP()
{
    if(m_skip_all_collision)
    {
        memset(h_cpNum, 0, sizeof(h_cpNum));
        h_gpNum = 0;
        return;
    }

    // [env-det] EE detection settings + env-local vertex map MUST be set BEFORE the per-env branch,
    // else the per-env path (which returns early) runs the EE dedup with GLOBAL edge indices (not
    // env-local) → cross-env asymmetric. Idempotent; the merged path below re-runs harmlessly.
    set_ee_nodedup(getenv("STIFF_EE_NODEDUP") ? 1 : 0);
    set_ee_detgate(getenv("STIFF_EE_DETGATE") ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);
    set_bvh_audit(getenv("STIFF_STACK_DIAG") ? 1 : 0);  // [audit-gate] per-pop depth probe, diag only
    // [perenv-par] per-vertex cross-env skip at self-collision emission (robust where BVH env-part is
    // bypassed by env-MIXED co-located nodes). Gated STIFF_DECOUPLE_THRESH; null = off (legacy path).
    set_self_p2g((getenv("STIFF_DECOUPLE_THRESH") && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(getenv("STIFF_EE_CANON") ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    if(getenv("STIFF_EE_CANON") && m_d_p2g && !m_vloc_built)
    {
        std::vector<int> hp(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec;
        for(int v=0; v<vertexNum; v++){ int g=hp[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; }
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_vloc, (size_t)vertexNum*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(m_d_vloc, vloc.data(), (size_t)vertexNum*sizeof(int), cudaMemcpyHostToDevice));
        set_ee_vloc(m_d_vloc); m_vloc_built = true;
    }

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
    set_ee_detgate(getenv("STIFF_EE_DETGATE") ? 1 : 0);
    set_bvh_envpart(getenv("STIFF_BVH_ENVPART") ? 1 : 0);  // [env-part B]
    // [perenv-par] per-vertex cross-env skip at self-collision emission (see note above). null = off.
    set_self_p2g((getenv("STIFF_DECOUPLE_THRESH") && m_d_p2g) ? m_d_p2g : nullptr);
    set_ee_canon(getenv("STIFF_EE_CANON") ? 1 : 0);
    set_ee_nomollify(getenv("STIFF_EE_NOMOLLIFY") ? 1 : 0);
    { static int _tc = 0; set_ee_trace((getenv("STIFF_EE_TRACE") && _tc++ == 0) ? 1 : 0); }  // first buildCP (iter0) only
    set_ee_tgt(getenv("STIFF_BAR_TGT0")?atoi(getenv("STIFF_BAR_TGT0")):-1, getenv("STIFF_BAR_TGT1")?atoi(getenv("STIFF_BAR_TGT1")):-1);
    // [env-det] build the global→env-local vertex id map once (canon total-order tie-break).
    if(getenv("STIFF_EE_CANON") && m_d_p2g && !m_vloc_built)
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

    {   // [9d28824-port] contiguous _cpNum[0:5]+_gpNum[5]: one 6-int D2H.
        uint32_t cp_gp_buf[6];
        CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
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
            memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
            h_gpNum = cp_gp_buf[5];
        }
    }

    snapshotDcdCcdPairs();   // [narrow-self snapshot] before buildFullCP clobbers the mirror
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
        if(_dcd_ccd_snapshot) CUDA_SAFE_CALL(cudaFree(_dcd_ccd_snapshot));
        m_dcd_snap_cap = (int)(m_dcd_snap_count + m_dcd_snap_count / 2) + 1;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_dcd_ccd_snapshot,
                                  (size_t)m_dcd_snap_cap * sizeof(int4)));
    }
    CUDA_SAFE_CALL(cudaMemcpy(_dcd_ccd_snapshot, _ccd_collisonPairs,
                              (size_t)m_dcd_snap_count * sizeof(int4),
                              cudaMemcpyDeviceToDevice));
}

// IPC's invariant is ground distance d > 0, maintained by
// CCD + the log-barrier. Under pathological pressing the distance can collapse
// far below physical validity (observed: healthy um-scale equilibrium ->
// 1e-23 m within two frames), after which the barrier Hessian (~1/d^2) makes
// Newton escape O(d)/iteration -- the vertex is permanently pinned and the
// "solution" is garbage (verified identical at iter caps 200 and 600).
// The engine must not silently solve past a broken invariant: fail loudly.
// (The former `dist2==0 ? 1e-12` clamps that papered over the NaN symptom of
//  this state are removed for the same reason -- clamp-and-continue hides a
//  state the engine cannot actually handle.)
// Cost: the flag is set inside the detection kernel (free ride) and read back
// as 4 bytes here -- no reductions, no allocations in the hot path.
//
// A non-finite or non-positive distance is already outside the barrier domain
// and throws immediately. Every finite positive distance remains feasible;
// there is deliberately no absolute distance floor.
// [iron-law] Quarantine the env owning `vertex` (persistent freeze via
// m_env_quarantined + the solve loop's pin branch) and remove its bodies from
// ground detection AND ground-CCD alpha via the skip table — otherwise the CCD
// alpha fail-fast would fire on the same infeasible vertices one phase later
// and kill the process anyway. Returns true when the env was (or already is)
// quarantined; false when no per-env machinery is live (caller keeps its
// throw). Gates: m_d_p2g set (= at least one computeGradientAndHessian ran →
// NOT initial-state validation), host telemetry path + per-env alpha active
// (isolated/strict). merged mode has no isolation machinery → false.
bool GIPC::quarantineEnv(int env, int vertex, double distance)
{
    if(!(m_active_group_count > 1
         && (env_newton_iter_cap > 0 || getenv("STIFF_PERENV_TELEM"))
         && getenv("STIFF_PERENV_ALPHA")))
        return false;
    if(env < 0 || env >= kEnvAlphaSlots)
        return false;
    if(m_env_quarantined.empty())
        m_env_quarantined.assign(kEnvAlphaSlots, 0);
    if(!m_d_env_quarantined)
    {   // device mirror for the direction-zero kernel
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_env_quarantined, kEnvAlphaSlots * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(m_d_env_quarantined, 0, kEnvAlphaSlots * sizeof(int)));
    }
    if(!m_env_quarantined[env])
    {
        m_env_quarantined[env] = 1;
        const int one = 1;
        CUDA_SAFE_CALL(cudaMemcpy(m_d_env_quarantined + env, &one, sizeof(int),
                                  cudaMemcpyHostToDevice));
        int body = -1;
        if(vertex >= 0 && _point_body_id)
            CUDA_SAFE_CALL(cudaMemcpy(&body, _point_body_id + vertex, sizeof(int), cudaMemcpyDeviceToHost));
        if(vertex >= 0)
            fprintf(stderr,
                    "[per-env][QUARANTINE] env %d became ground-infeasible MID-RUN "
                    "(vertex=%d body=%d distance=%.6e): this env is frozen from now "
                    "on; healthy envs continue. (Initial-state violations still throw.)\n",
                    env, vertex, body, distance);
        else
            fprintf(stderr,
                    "[per-env][QUARANTINE] env %d produced a NON-FINITE Newton "
                    "direction MID-RUN: this env is frozen from now on (direction "
                    "zeroed each iteration); healthy envs continue.\n",
                    env);
        if(m_d_b2g && m_collision_body_count > 0)
        {
            if(!_ground_skip_body)
            {   // lazily allocate when no skip bodies were ever registered
                CUDA_SAFE_CALL(cudaMalloc((void**)&_ground_skip_body,
                                          m_collision_body_count * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemset(_ground_skip_body, 0,
                                          m_collision_body_count * sizeof(int)));
                _ground_body_count  = m_collision_body_count;
                m_ground_skip_owned = true;
            }
            const int nb = _ground_body_count;
            if(nb > 0)
            {
                int bs = 128, gs = (nb + bs - 1) / bs;
                _mark_env_ground_skip<<<gs, bs>>>(_ground_skip_body, m_d_b2g, env, nb);
            }
        }
    }
    return true;
}

bool GIPC::quarantineEnvOfVertex(int vertex, double distance)
{
    if(!(m_d_p2g && vertex >= 0 && vertex < static_cast<int>(vertexNum)))
        return false;
    int env = -1;
    CUDA_SAFE_CALL(cudaMemcpy(&env, m_d_p2g + vertex, sizeof(int), cudaMemcpyDeviceToHost));
    return quarantineEnv(env, vertex, distance);
}

// [iron-law] Frame-start quarantine scan: runs BEFORE any CCD-alpha work of
// the new frame. Teleports/drives can make an env infeasible BETWEEN frames,
// and the collision build is carried over from the previous frame — without
// this probe the ground-CCD fail-fast fires before any detection sees the new
// state, bypassing the demotion. Flag-only probe → no pair-list side effects.
void GIPC::quarantineGroundInfeasibleAtFrameStart()
{
    if(getenv("STIFF_SKIP_GRND"))
        return;
    if(!(_gdCollapse && m_d_p2g && m_d_b2g && m_active_group_count > 1
         && (env_newton_iter_cap > 0 || getenv("STIFF_PERENV_TELEM"))
         && getenv("STIFF_PERENV_ALPHA")))
        return;
    const int n = static_cast<int>(surf_vertexNum);
    if(n < 1)
        return;
    const int bs = 256, gs = (n + bs - 1) / bs;
    for(int round = 0; round <= m_active_group_count; ++round)
    {   // one round per newly-quarantined env (atomicMin reports one winner)
        CUDA_SAFE_CALL(cudaMemset(_gdCollapse, 0, sizeof(int)));
        _probe_ground_infeasible<<<gs, bs>>>(_vertexes, _surfVerts, _groundOffset,
                                             _groundNormal, n, _point_body_id,
                                             _ground_skip_body, _ground_body_count,
                                             _gdCollapse);
        int collapsed = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&collapsed, _gdCollapse, sizeof(int), cudaMemcpyDeviceToHost));
        if(collapsed >= 0)
            break;
        const int vertex = -collapsed - 1;
        double3   position = make_double3(0.0, 0.0, 0.0);
        double3   normal   = make_double3(0.0, 1.0, 0.0);
        double    offset   = 0.0;
        CUDA_SAFE_CALL(cudaMemcpy(&position, _vertexes + vertex, sizeof(double3), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&normal, _groundNormal, sizeof(double3), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&offset, _groundOffset, sizeof(double), cudaMemcpyDeviceToHost));
        const double distance = normal.x * position.x + normal.y * position.y
                              + normal.z * position.z - offset;
        if(!quarantineEnvOfVertex(vertex, distance))
            break;   // unattributable → leave it to the regular throw sites
    }
    CUDA_SAFE_CALL(cudaMemset(_gdCollapse, 0, sizeof(int)));
}

void GIPC::throwIfGroundDistanceInvalid()
{
    if(!_gdCollapse)
        return;
    int collapsed = 0;
    if(h_gpNum > 0)
        CUDA_SAFE_CALL(cudaMemcpy(&collapsed, _gdCollapse, sizeof(int), cudaMemcpyDeviceToHost));
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
        throw std::runtime_error(message);
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

static void throwForInvalidCcdMask(int invalid, const char* context)
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

static void validateFinalCcdStateOrThrow(const double* state, const char* context)
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
                                      device_refined_invalid);
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
