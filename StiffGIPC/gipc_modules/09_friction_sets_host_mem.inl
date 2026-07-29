__global__ void _calFrictionLastH_gd(const double3* _vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     const const uint32_t* _collisionPair_environment,
                                     double*   lambda_lastH_gd,
                                     uint32_t* _collisionPair_last_gd,
                                     double    dHat,
                                     double    Kappa,
                                     int       number,
                                     const double* kappa_grp = nullptr,
                                     const int*    p2g = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    double3 normal = *g_normal;
    int     gidx   = _collisionPair_environment[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    // [d-floor fail-fast] clamp removed; buildCP() throws before d can collapse here.

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    // [decouple] per-group κ so env0's friction normal-force is batch-invariant (global Kappa is a
    // reduction over ALL envs ⇒ batch-dependent; friction Hessian ∝ λ exposes it even at zero sliding).
    double Kp = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa;  /* [-1 guard] */
    lambda_lastH_gd[idx]        = -Kp * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const double3*    _vertexes,
                                             const const int4* _collisionPair,
                                             double*           lambda_lastH,
                                             double2*          distCoord,
                                             __GEIGEN__::Matrix3x2d* tanBasis,
                                             int4*     _collisionPair_last,
                                             double    dHat,
                                             double    Kappa,
                                             uint32_t* _cpNum_last,
                                             int       number,
                                             const double* kappa_grp = nullptr,
                                             const int*    p2g = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dis;
    int    last_index = -1;
    // [decouple] per-group κ for the lagged friction normal-force λ (batch-invariant). gv = pair's
    // representative vertex (same convention as the barrier, GIPC.cu:3250).
    double Kappa_eff = Kappa;
    if(kappa_grp && p2g)
    { int gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); if(gv >= 0) { int _gg = p2g[gv]; if(_gg >= 0) Kappa_eff = kappa_grp[_gg]; } }  /* [-1 guard] */
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_EE(_vertexes[MMCVIDI.x],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PP(
                    _vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y >= 0)
            {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                Friction::computeClosestPoint_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 distCoord[last_index].x);
                distCoord[last_index].y = 0;
                Friction::computeTangentBasis_PE(_vertexes[v0I],
                                                 _vertexes[MMCVIDI.y],
                                                 _vertexes[MMCVIDI.z],
                                                 tanBasis[last_index]);
            }
        }
        else
        {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            Friction::computeClosestPoint_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             distCoord[last_index]);
            Friction::computeTangentBasis_PT(_vertexes[v0I],
                                             _vertexes[MMCVIDI.y],
                                             _vertexes[MMCVIDI.z],
                                             _vertexes[MMCVIDI.w],
                                             tanBasis[last_index]);
        }
    }
    if(last_index >= 0)
    {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if (RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa_eff * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] =
            -Kappa_eff * 2.0 * sqrt(dis)
            * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat)
               + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}

/// <summary>
///  host code
/// </summary>
void GIPC::FREE_DEVICE_MEM()
{
    // [C6] last chance to report how much of the run was one whole-frame
    // graph; every scene reaches teardown, so no example needs patching.
    print_frame_graph_coverage();
    // Whole-frame graphs retain mesh, solver and transaction-snapshot
    // pointers. They must die before any allocation referenced by a node.
    destroy_frame_graph();

    // Captured nodes retain every device pointer used by the trial body.
    // Destroy the executable before any of those allocations are released.
    if(m_ls_graph_exec)
    {
        cudaGraphExecDestroy(m_ls_graph_exec);
        m_ls_graph_exec   = nullptr;
        for(long long& value : m_ls_graph_sig)
            value = -1;
    }

    auto release = [](auto*& pointer)
    {
        if(pointer)
        {
            CUDA_SAFE_CALL(cudaFree(pointer));
            pointer = nullptr;
        }
    };

    // Streams must finish before their scratch allocations are released.
    for(cudaStream_t& stream : m_pool_streams)
    {
        if(stream)
        {
            CUDA_SAFE_CALL(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
    m_pool_streams.clear();
    auto release_pool = [&](std::vector<BvhScratch>& pool)
    {
        for(BvhScratch& scratch : pool)
        {
            release(scratch.nodes);
            release(scratch.bvs);
            release(scratch.mch);
            release(scratch.idx);
            release(scratch.tmp);
            release(scratch.flags);
            release(scratch.node_env);
            release(scratch.sort_tmp);
            release(scratch.mch_alt);
            release(scratch.idx_alt);
            scratch.sort_bytes = 0;
            scratch.sort_cap   = 0;
        }
        pool.clear();
    };
    release_pool(m_pool_f);
    release_pool(m_pool_e);
    m_pool_K = 0;

    pair_buffers_free(PairBuffers{_collisonPairs, _MatIndex, _ccd_collisonPairs, MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM});   // [v0.8.6 2b] guarded + null-set
    m_reduce_scratch.release(); m_reduce_cap = 0;  // [3d-2]
    _cpNum.release();
    m_pair_snap_cur.release();
    m_pair_snap_last.release();
    _gpNum = nullptr;
    release(_close_cpNum);
    release(_close_gpNum);
    release(_gdCollapse);
    if(m_ground_skip_owned && _ground_skip_body)
    {   // [iron-law] only free when WE lazily allocated it (normally d_tetMesh owns it)
        release(_ground_skip_body);
    }
    m_ground_skip_owned = false;
    m_d_env_quarantined.release();
    m_d_env_dirnan.release();
    release(_environment_collisionPair);
    release(_groundNormal);
    release(_groundOffset);

    _faces.release();
    _edges.release();
    _surfVerts.release();

    // [multi-env S1] free per-env line-search substrate
    m_env_alpha.release();
    m_env_scratch.release();
    m_abd_body_alpha.release();
    m_env_active.release();

    // [0be8da3-port] free the persistent (grow-only) friction/close buffers and
    // reset capacities so engine.reset() starts clean.
    lambda_lastH_scalar.release();      // [3d] free+null+idempotence are class
    distCoord.release();                //      invariants now, not a call-site
    tanBasis.release();                 //      discipline
    _collisonPairs_lastH.release();
    lambda_lastH_scalar_gd.release();
    _collisonPairs_lastH_gd.release();
    if(_closeConstraintID)
    {
        release(_closeConstraintID);
        release(_closeConstraintVal);
    }
    if(_closeMConstraintID)
    {
        release(_closeMConstraintID);
        release(_closeMConstraintVal);
    }
    m_fric_cp_cap = 0; m_fric_gd_cap = 0; m_close_gp_cap = 0; m_close_cp_cap = 0;

    // Device-resident energy/control scalars.
    release(m_energy_slots);
    release(m_line_search_energy);
    release(m_compatibility_energy);
    release(m_line_search_decision);
    release(m_newton_convergence_decision);
    release(m_ccd_alpha_slots);
    release(m_d_close_flag);   // [C4-b]
    release(m_ccd_alpha_invalid);
    release(m_ccd_refined_invalid);
    release(_dcd_ccd_snapshot);
    m_dcd_snap_count = 0; m_dcd_snap_cap = 0;
    release(m_ground_trial_invalid);
    release(m_env_ground_trial_invalid);

    // GIPC-owned mode, determinism, material and diagnostic allocations that
    // are created outside the initial allocator.
    release(d_env_offset);
    release(d_bvh_vertexes);
    release(g_grad_binned);
    release(d_perenv_face_idx);
    release(d_perenv_edge_idx);
    release(d_xenv_lid);
    release(d_xenv_buf);
    release(d_face_env);
    release(d_edge_env);
    release(d_face_localid);
    release(d_edge_localid);
    release(d_face_v0);
    release(d_edge_v0);
    release(m_d_vloc);
    release(m_pe_all);        // [descriptor phase-0b] per-env energy slices
    // [descriptor phase-0.3] solver scratch family
    release(m_scr_ls_eg0);   release(m_scr_ls_eg1);
    release(m_scr_gp_friction);   // [B3 s7]
    release(m_d_ls_alpha);        // [C-1]
    release(m_scr_ls_decision_counts);
    release(m_scr_maxk);
    release(m_scr_sq_a);     release(m_scr_cnt_a);
    release(m_scr_mxm);
    release(m_scr_mm);       release(m_scr_ct);
    release(m_scr_mx);
    release(m_scr_env_cnt);
    release(m_scr_sq_b);
    release(m_scr_perenv_ta);
    release(m_scr_xenv4);
    release(m_scr_gsum_bin); release(m_scr_gsnorm_bin);
    release(m_scr_gsum_g);   release(m_scr_gsnorm_g);
    release(m_energy_sink);   // [descriptor phase-0b] DeviceOut scalar sink
    release(d_env_bbox2);
    release(m_kappa_group);
    release(m_d_close_grp);
    release(d_vert_mu);
    release(d_vert_mu_gd);

    pcg_data.FREE_DEVICE_MEM();

    bvh_e.FREE_DEVICE_MEM();
    bvh_f.FREE_DEVICE_MEM();
}

void GIPC::MALLOC_DEVICE_MEM()
{
    // +1 trash slot: pair-emit overflow is redirected to index==cap (see _emit_slot
    // in mlbvh.cu) so detection never writes out of bounds; the host then grows.
    pair_buffers_alloc(PairBuffers{_collisonPairs, _MatIndex, _ccd_collisonPairs, MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM});   // [v0.8.6 2b] mechanics owner
    CUDA_SAFE_CALL(cudaMalloc((void**)&_environment_collisionPair,
                              surf_vertexNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_moveDir, vertexNum * sizeof(double3)));
    // [9d28824-port] one contiguous [6]-uint32 block: _cpNum aliases [0:5],
    // _gpNum aliases [5]. Kernel-side code unchanged (takes uint32_t*); the
    // paired cpNum+gpNum reads become ONE 6-int D2H.
    _cpNum.resize_discard(6);
    _gpNum = _cpNum + 5;
    m_pair_snap_cur.resize_discard(6);
    m_pair_snap_last.resize_discard(6);
    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_pair_snap_cur, 0, 6 * sizeof(uint32_t), cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemsetAsync(
        m_pair_snap_last, 0, 6 * sizeof(uint32_t), cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundNormal, 5 * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_groundOffset, 5 * sizeof(double)));
    double  h_offset[5] = {ground_offset_cfg, -1, 1, -1, 1};
    double3 H_normal[5];
    H_normal[0] = ground_normal_cfg;
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
    CUDA_SAFE_CALL(cudaMemcpy(_groundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_groundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


    _faces.resize_discard(surface_Num);
    _edges.resize_discard(edge_Num);
    _surfVerts.resize_discard(surf_vertexNum);

    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_cpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_close_gpNum, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_gdCollapse, sizeof(int)));   // [d-floor fail-fast]
    CUDA_SAFE_CALL(cudaMemset(_gdCollapse, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_ccd_alpha_invalid, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(m_ccd_alpha_invalid, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(
        (void**)&m_ccd_refined_invalid, (1 + kEnvAlphaSlots) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(
        m_ccd_refined_invalid, 0, (1 + kEnvAlphaSlots) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_ground_trial_invalid, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(m_ground_trial_invalid, 0, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(
        (void**)&m_env_ground_trial_invalid, kEnvAlphaSlots * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(
        m_env_ground_trial_invalid, 0, kEnvAlphaSlots * sizeof(int)));
    {   // [PSD clamp] ground-Hessian projection opt-out (see g_ground_hess_legacy)
        const char* v      = getenv("STIFF_GROUND_HESS_LEGACY");
        int         legacy = (v && v[0] && v[0] != '0') ? 1 : 0;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_ground_hess_legacy, &legacy, sizeof(int)));
    }

    // [multi-env S1] per-env feasible-alpha substrate (physics-neutral until S2).
    m_env_alpha.resize_discard(kEnvAlphaSlots);
    m_env_scratch.resize_discard(5 * kEnvAlphaSlots);
    m_env_active.resize_discard(kEnvAlphaSlots);
    h_env_alpha.assign(kEnvAlphaSlots, 1.0);
    h_env_active.assign(kEnvAlphaSlots, 1);
    // [batch-size hygiene] m_env_alpha starts at 1.0 like the host mirror: the device fast path
    // (2262b33) writes only PRESENT envs' slots — absent slots must not hold cudaMalloc garbage.
    CUDA_SAFE_CALL(cudaMemcpy(m_env_alpha, h_env_alpha.data(),
                              kEnvAlphaSlots * sizeof(double), cudaMemcpyHostToDevice));
    // [hygiene] m_env_scratch starts NEUTRAL (alpha regions 0-2 = 1.0, max
    // regions 3-4 = 0.0), never cudaMalloc garbage: the first per-env swept
    // build reads ta_e from these slots before any S1 pass has filled them.
    // (Independent defect — not the cross-env root cause, but real.)
    { std::vector<double> neutral(5 * kEnvAlphaSlots, 0.0);
      std::fill(neutral.begin(), neutral.begin() + 3 * kEnvAlphaSlots, 1.0);
      CUDA_SAFE_CALL(cudaMemcpy(m_env_scratch, neutral.data(),
                                5 * kEnvAlphaSlots * sizeof(double), cudaMemcpyHostToDevice)); }
    { std::vector<int> ones(kEnvAlphaSlots, 1);
      CUDA_SAFE_CALL(cudaMemcpy(m_env_active, ones.data(), kEnvAlphaSlots * sizeof(int), cudaMemcpyHostToDevice)); }

    // Device energy terms and line-search control state.
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_energy_slots, kEnergySlotCount * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_line_search_energy, 2 * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_compatibility_energy, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_line_search_decision, 7 * sizeof(int)));  // [B3/C-3] {decision, overflow, collapse, trials, alpha_lo, alpha_hi, overflow_baseline}
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_newton_convergence_decision, sizeof(int)));
    // Device-resident CCD alpha/control chain (see slot layout in GIPC.cuh).
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_ccd_alpha_slots, 9 * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_scr_gp_friction, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_ls_alpha, sizeof(double)));  // [C-1]
    CUDA_SAFE_CALL(cudaMemset(m_scr_gp_friction, 0, sizeof(uint32_t)));

    CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

    // [multi-env determinism] per-vertex env offset (0 by default = no-op) + BVH vertex buffer.
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_env_offset, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_bvh_vertexes, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_env_offset, 0, vertexNum * sizeof(double3)));

    // [multi-env determinism 4.3] binned gradient accumulator (BINNED_K bins per vert*comp).
    CUDA_SAFE_CALL(cudaMalloc((void**)&g_grad_binned,
                              3 * (size_t)vertexNum * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_gbin, &g_grad_binned, sizeof(double*)));

    pcg_data.Malloc_DEVICE_MEM(vertexNum, tetrahedraNum);
}


void GIPC::initBVH(int* _btype, int* _bodyId, int* _collision_skip_matrix, int _collision_body_count)
{

    bvh_e.init(_bodyId,
               _btype,
               d_bvh_vertexes,   // [multi-env] BVH on offset-separated verts; narrow-phase uses local _vertexes
               _rest_vertexes,
               _edges,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               edge_Num,
               surf_vertexNum,
               _collision_skip_matrix,
               _collision_body_count);
    bvh_f.init(_bodyId,
               _btype,
               d_bvh_vertexes,   // [multi-env] BVH on offset-separated verts; narrow-phase uses local _vertexes
               _faces,
               _surfVerts,
               _collisonPairs,
               _ccd_collisonPairs,
               _cpNum,
               _MatIndex,
               surface_Num,
               surf_vertexNum,
               _collision_skip_matrix,
               _collision_body_count);
    // [multi-FEM-bodyid] forward the per-body FEM flag table directly
    // (set by sim_engine.cu after this call returns; see ipc._body_id_to_is_fem
    // assignment in do_init_bvh_and_solver).
    bvh_e._body_id_to_is_fem = _body_id_to_is_fem;
    bvh_f._body_id_to_is_fem = _body_id_to_is_fem;
}

void GIPC::init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer, double buffScale)
{
    if(m_skip_all_collision)
    {
        SceneSize.upper = make_double3(maxConer.x, maxConer.y, maxConer.z);
        SceneSize.lower = make_double3(minConer.x, minConer.y, minConer.z);
    }
    else
    {
        // [B3 bbox-async fix] boot-only 48B read of the root box straight from
        // the device. The old code consumed bvh_f.scene — a SIDE EFFECT of the
        // blocking Construct — which surgery ⑥'s async builds no longer refresh;
        // flows that never call getSceneSize() then fed garbage into the dHat
        // derivation (bboxDiagSize2=1e65 → 22M pairs → OOM, 7 gates down).
        CUDA_SAFE_CALL(cudaMemcpy(
            &SceneSize, bvh_f._bvs, sizeof(AABB), cudaMemcpyDeviceToHost));
    }
    bboxDiagSize2 = __GEIGEN__::__squaredNorm(
        __GEIGEN__::__minus(SceneSize.upper, SceneSize.lower));
    // [absolute-dhat fix] The scene-bbox diagonal grows with env count / spacing,
    // which inflates the bbox-derived dHat (contact thickness) — a physics bug
    // and the root cause of super-linear contact growth in multi-env. When
    // absolute_dhat>0, derive an EFFECTIVE bbox so dHat == absolute_dhat^2 and
    // dTol/fDhat stay consistent with a single-env scene of that contact scale.
    double eff_bboxDiagSize2 = bboxDiagSize2;
    if(absolute_dhat > 0.0 && relative_dhat > 0.0)
        eff_bboxDiagSize2 = (absolute_dhat * absolute_dhat)
                            / (relative_dhat * relative_dhat);
    dTol         = 1e-18 * eff_bboxDiagSize2;
    minKappaCoef = 1e11;
    meanMass     = m_meanMass;
    meanVolumn   = m_meanVolumn;
    dHat = relative_dhat * relative_dhat * eff_bboxDiagSize2;  // = absolute_dhat^2 when set
    fDhat = 1e-4 * eff_bboxDiagSize2;
    if(::g_gipc_log_level >= 1)
        printf("[dhat] bboxDiagSize2=%.6g (eff=%.6g)  relative_dhat=%.3g  abs_dhat=%.3g  dHat_sqrt=%.6g%s\n",
               bboxDiagSize2, eff_bboxDiagSize2, relative_dhat, absolute_dhat,
               sqrt(dHat), absolute_dhat > 0.0 ? " (ABSOLUTE)" : " (scene-bbox)");
    if(getenv("STIFF_SEED_DIAG"))
        printf("[seed-diag] bboxDiagSize2=%.17g eff=%.17g meanMass=%.17g meanVolumn=%.17g dHat=%.17g fDhat=%.17g dTol=%.17g scene=[%.17g,%.17g,%.17g]-[%.17g,%.17g,%.17g]\n",
               bboxDiagSize2, eff_bboxDiagSize2, meanMass, meanVolumn, dHat, fDhat, dTol,
               SceneSize.lower.x, SceneSize.lower.y, SceneSize.lower.z,
               SceneSize.upper.x, SceneSize.upper.y, SceneSize.upper.z);


    int global_matrix_block3_size =
        abd_fem_count_info.abd_body_num * 4 + abd_fem_count_info.fem_point_num;


    uint32_t Minimum = 100000 * buffScale;
    int minCollisionBuffer4 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer3 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer2 = std::max(2 * (surf_vertexNum + edge_Num), Minimum);
    int minCollisionBuffer1 = 2 * surf_vertexNum;

    long long unsigned total_internal_triplet_num =
        ((abd_fem_count_info.fem_tet_num + tri_edge_num) * 10 + triangleNum * 6)
        + softNum
        + abd_fem_count_info.abd_body_num * 10
        + num_joint_constraints * 16
        + static_cast<long long>(m_abd_system->m_num_revolute_driving) * 16
        + static_cast<long long>(m_abd_system->m_num_prismatic) * 16
        + static_cast<long long>(m_abd_system->m_num_prismatic_driving) * 16
        + static_cast<long long>(softNum) * 4;
    long long unsigned total_max_collision_triplet_num =
        minCollisionBuffer4 * 16 + minCollisionBuffer3 * 9
        + minCollisionBuffer2 * 4 + minCollisionBuffer1;
    // [Strategy D] M3.5 chain-rule kernel reserves an extension range past
    // the FEM triplets, with capacity = fem_triplet_num * 16 (worst-case
    // diff-body pin-pin expansion).  When rigid region is large (Strategy D
    // hybrid mesh), this 16× factor easily exceeds the previous 2×
    // allocation → CUDA illegal memory access.  Use 32× to give margin
    // (the actual ext_count is usually < 16× but allocation math conservative).
    long long unsigned total_max_global_triplet_num =
        total_internal_triplet_num
            * static_cast<long long unsigned>(m_triplet_internal_margin)
        + total_max_collision_triplet_num;
    // [P1-dyn] Non-hybrid scenes (margin forced to 1 by sim_engine when n_fem_pins==0)
    // size the triplet buffer per-step from ACTUAL contact counts instead of the
    // worst-case 2*(surf+edge)*29 (cp-stats: <1% used). Allocate a small initial buffer;
    // computeGradientAndHessian() grows it to 2*length each step (2x = converter's
    // documented [length:2*length) scratch region). Hybrid keeps the worst-case alloc.
    m_fixed_triplet_base = static_cast<long long>(total_internal_triplet_num);
    m_dynamic_triplet    = (m_triplet_internal_margin <= 1.0);
    long long unsigned init_total = total_internal_triplet_num
        + static_cast<long long unsigned>(abd_fem_count_info.fem_point_num)
        + 2u * static_cast<long long unsigned>(surf_vertexNum) + 100000u;
    long long unsigned triplet_alloc = m_dynamic_triplet
        ? (2u * init_total)            // 2x for the converter scratch/output region
        : (total_max_global_triplet_num * (long long unsigned)buffScale);
    long long unsigned hash_alloc = m_dynamic_triplet
        ? init_total
        : (long long unsigned)((total_internal_triplet_num + total_max_collision_triplet_num) * buffScale);
    if(::g_gipc_log_level >= 1) printf("[buffer] internal=%llu worst=%llu init_alloc=%llu (dynamic=%d, ~%llu MB)\n",
           total_internal_triplet_num, total_max_global_triplet_num, triplet_alloc,
           (int)m_dynamic_triplet, triplet_alloc * 80 / 1024 / 1024);

    gipc_global_triplet.init_var();

    gipc_global_triplet.resize(global_matrix_block3_size,
                               global_matrix_block3_size,
                               triplet_alloc);

    gipc_global_triplet.global_external_max_capcity = hash_alloc;
    gipc_global_triplet.resize_collision_hash_size(hash_alloc);


    m_global_linear_system->gipc_global_triplet = &(gipc_global_triplet);
    m_abd_system->global_triplet                = &(gipc_global_triplet);
    init_abd_system();
}

GIPC::~GIPC()
{
    // The auxiliary detector consumes both GIPC-owned buffers and pointers
    // borrowed from device_TetraData. Complete its work before destroying
    // events/stream or releasing either ownership domain during Engine reset.
    if(m_aux_stream)
        cudaStreamSynchronize(m_aux_stream);
    if(m_aux_done_event)
    {
        cudaEventDestroy(m_aux_done_event);
        m_aux_done_event = nullptr;
    }
    if(m_aux_reset_event)
    {
        cudaEventDestroy(m_aux_reset_event);
        m_aux_reset_event = nullptr;
    }
    if(m_aux_stream)
    {
        cudaStreamDestroy(m_aux_stream);
        m_aux_stream = nullptr;
    }
    FREE_DEVICE_MEM();
}

GIPC::GIPC()
{
    IPC_dt            = 0.01;
    animation_subRate = 1.0;
    animation         = false;

    // h_cpNum_last zero-init happens in the HostMirrorArray member ctor [3b-2]
}

static void _dbg_ksum(const char*, const void*, size_t);  // [4.3 fwd]
void GIPC::buildFrictionSets()
{
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    int                numbers   = h_cpNum[0];
    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        _dbg_ksum("prep_verts", _vertexes, (size_t)vertexNum * sizeof(double3));
    }
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    if(numbers > 0)
    {
        // [multi-env determinism 4.3] zero distCoord first: PP (point-point) lagged pairs write
        // tanBasis but NOT distCoord (no barycentric coords), leaving GARBAGE in their slots.
        // Those slots sit at non-deterministic positions (atomicAdd last_index) → the lagged
        // friction data is non-deterministic run-to-run → the friction Hessian (frame 0) → the
        // whole solve. Zeroing makes PP distCoord deterministically 0 (the friction Hessian for
        // PP doesn't use it; EE/PE/PT overwrite it). THIS is the residual non-atomic source.
        CUDA_SAFE_CALL(cudaMemset(distCoord, 0, (size_t)h_cpNum[0] * sizeof(double2)));
        _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(_vertexes,
                                                              _collisonPairs,
                                                              lambda_lastH_scalar,
                                                              distCoord,
                                                              tanBasis,
                                                              _collisonPairs_lastH,
                                                              dHat,
                                                              Kappa,
                                                              _cpNum,
                                                              h_cpNum[0],
                                                              m_pergroup_kappa ? m_kappa_group : nullptr,
                                                              m_pergroup_kappa ? m_d_p2g : nullptr);
    }
    CUDA_SAFE_CALL(cudaMemcpy(h_cpNum_last.refresh_dst(), _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Preserve the body-contact truth before _cpNum becomes another rank
    // scratch. Ground is different: _gpNum aliases slot 5 and ground Hessian
    // assembly increments it even though the rank is unused. If Newton exits
    // before another buildCP, slot 5 is therefore 2*h_gpNum here. The DCD
    // snapshot is the immutable exact source for the lagged ground count.
    CUDA_SAFE_CALL(cudaMemcpyAsync(m_pair_snap_last,
                                   _cpNum,
                                   5 * sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaMemcpyAsync(m_pair_snap_last.data() + 5,
                                   m_pair_snap_cur.data() + 5,
                                   sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice,
                                   cudaStreamPerThread));
    numbers = h_gpNum;
    if(numbers > 0)
    {

        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionLastH_gd<<<blockNum, threadNum>>>(_vertexes,
                                                      _groundOffset,
                                                      _groundNormal,
                                                      _environment_collisionPair,
                                                      lambda_lastH_scalar_gd,
                                                      _collisonPairs_lastH_gd,
                                                      dHat,
                                                      Kappa,
                                                      h_gpNum,
                                                      m_pergroup_kappa ? m_kappa_group : nullptr,
                                                      m_pergroup_kappa ? m_d_p2g : nullptr);
    }
    h_gpNum_last = h_gpNum;
    // [B3 s7] Restore from the immutable DCD snapshot for the same reason as
    // m_pair_snap_last[5] above; raw _cpNum+5 may already be rank-incremented.
    CUDA_SAFE_CALL(cudaMemcpyAsync(m_scr_gp_friction,
                                   m_pair_snap_cur.data() + 5,
                                   sizeof(uint32_t), cudaMemcpyDeviceToDevice, 0));
}


void GIPC::GroundCollisionDetect()
{
    int numbers = surf_vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _GroundCollisionDetect<<<blockNum, threadNum>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numbers,
        _point_body_id, _ground_skip_body, _ground_body_count, _gdCollapse);
}

void GIPC::computeSoftConstraintGradientAndHessian(double3* _gradient, int global_hessian_fem_offset)
{
    int numbers = softNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        targetVert,
        targetInd,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        softMotionRate,
        animation_fullRate,
        gipc_global_triplet.global_triplet_offset,
        global_hessian_fem_offset,
        m_d_stitch_paired_vertex,
        m_d_stitch_rest_offset,
        m_d_stitch_abd_body_id,
        reinterpret_cast<const __GEIGEN__::Vector12*>(m_d_abd_body_q),
        softNum);
}

void GIPC::getTotalForce(double3* _gradient0, double3* _gradient1)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _getTotalForce<<<blockNum, threadNum>>>(_gradient0, _gradient1, numbers);
}


void GIPC::computeGroundGradientAndHessian(double3* _gradient)
{
#ifndef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if(numbers < 1)
    {
        return;
    }
    const unsigned int threadNum = default_threads;
    const bool tier_mode = contact_tier_layout_mode();
    const int  launch_count = contact_pair_launch_extent(numbers);
    int        blockNum =
        (launch_count + static_cast<int>(threadNum) - 1)
        / static_cast<int>(threadNum);
    if(tier_mode)
        clear_contact_triplet_span(gipc_global_triplet,
                                   gipc_global_triplet.global_triplet_offset,
                                   launch_count);
    _computeGroundGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _groundOffset,
        _groundNormal,
        _environment_collisionPair,
        _gradient,
        _gpNum,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        dHat,
        Kappa,
        gipc_global_triplet.global_triplet_offset,
        numbers,
        m_pergroup_kappa ? m_kappa_group : nullptr,
        m_pergroup_kappa ? m_d_p2g : nullptr,
        tier_mode ? m_pair_snap_cur.data() + 5 : nullptr,
        graph_kappa_dev());   // [C4-b] device kappa inside the frame graph
}

void GIPC::computeCloseGroundVal()
{
    int numbers = h_gpNum;
    if(h_gpNum <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _computeGroundCloseVal<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    dTol,
                                                    _closeConstraintID,
                                                    _closeConstraintVal,
                                                    _close_gpNum,
                                                    numbers);
    // The legacy close-contact Kappa-doubling strategy is intentionally
    // disabled: its raw restore destabilizes coupled ABD contact. Kappa is
    // initialized by the gradient-projection strategy each frame instead.
}

bool GIPC::checkCloseGroundVal()
{
    int numbers = h_close_gpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers,
        m_pergroup_kappa ? m_d_close_grp : nullptr, m_pergroup_kappa ? m_d_p2g : nullptr);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxGroundDist()
{
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient,
                                 double   mKappa,
                                 bool     use_group_kappa)
{
    int numbers = h_gpNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    const int launch_count = contact_pair_launch_extent(numbers);
    int blockNum =
        (launch_count + static_cast<int>(threadNum) - 1)
        / static_cast<int>(threadNum);
    _computeGroundGradient<<<blockNum, threadNum>>>(_vertexes,
                                                    _groundOffset,
                                                    _groundNormal,
                                                    _environment_collisionPair,
                                                    _gradient,
                                                    _gpNum,
                                                    dHat,
                                                    mKappa,
                                                    numbers,
                                                    use_group_kappa && m_pergroup_kappa
                                                        ? m_kappa_group
                                                        : nullptr,
                                                    use_group_kappa && m_pergroup_kappa
                                                        ? m_d_p2g
                                                        : nullptr,
                                                    contact_tier_layout_mode()
                                                        ? m_pair_snap_cur.data() + 5
                                                        : nullptr);
}

void GIPC::computeSoftConstraintGradient(double3* _gradient)
{
    int numbers = softNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeSoftConstraintGradient<<<blockNum, threadNum>>>(
        _vertexes, targetVert, targetInd, _gradient, softMotionRate, animation_fullRate,
        m_d_stitch_paired_vertex, m_d_stitch_rest_offset,
        m_d_stitch_abd_body_id,
        reinterpret_cast<const __GEIGEN__::Vector12*>(m_d_abd_body_q),
        softNum);
}

double* GIPC::ensure_reduce_scratch(int count)
{
    // ceil(count/default_threads) doubles are written by the first reduction pass.
    size_t need = (size_t)((count + default_threads - 1) / default_threads) + 1;
    if(need > m_reduce_cap)
    {
        ++pcg_buffer_generation();   // [C-1] scratch pointer baked in the LS graph moves
        m_reduce_cap = need + need / 2;  // 1.5x slack → no realloc churn after warmup
        m_reduce_scratch.resize_discard(m_reduce_cap);  // [3d-2]
    }
    return m_reduce_scratch;
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers)
{
    if(m_skip_all_collision)
        return 1.0;
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if(numbers < 1)
        return 1;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness, numbers,
        m_ccd_alpha_invalid, kCcdInvalidGlobalRefined, nullptr);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        _reduct_min_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    throwIfInvalidCcdAlpha("self CCD reduction");
    if(!std::isfinite(minValue) || minValue <= 0.0 || minValue > 1.0)
        throw std::runtime_error("[StiffGIPC] invalid direct self-CCD alpha");
    return minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue)
{
    int                numbers   = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, mqueue, _surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

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
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum)
{
    int                numbers   = vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if(type == 0)
    {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    }
    else if(type == 1)
    {
        _reduct_double3Sqn_to_double<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}


// ②-D2H batched variants for the two CCD reductions that fire back-to-back
// at the top of each line search. Each writes a direct feasible alpha via D2D;
// the caller takes MIN without reciprocal conversion.

void GIPC::ground_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, double* out_slot)
{
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_groundAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir, mqueue, slackness, numbers,
        _point_body_id, _ground_skip_body, _ground_body_count, m_ccd_alpha_invalid);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_min_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}

void GIPC::self_largestFeasibleStepSize_DeviceOut(double slackness, double* mqueue, int numbers, double* out_slot)
{
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _dcd_ccd_snapshot /* [narrow-self snapshot] DCD-time mirror, immune to buildFullCP clobbering */, _moveDir, mqueue, slackness, numbers,
        m_ccd_alpha_invalid, kCcdInvalidGlobalNarrow, nullptr);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_min_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}

void GIPC::self_full_largestFeasibleStepSize_DeviceOut(double slackness,
                                                       double* mqueue,
                                                       int numbers,
                                                       double* out_slot,
                                                       const uint32_t* d_live)
{
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    // Refined candidates are provisional: record their raw status separately.
    // The final device gate promotes it only if refinement is actually consumed.
    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes,
        _ccd_collisonPairs,
        _moveDir,
        mqueue,
        slackness,
        numbers,
        m_ccd_refined_invalid,
        kCcdRawInvalid,
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

void GIPC::cfl_largestSpeed_DeviceOut(double* mqueue, double* out_slot)
{
    int                numbers   = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_max_cfl_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, mqueue, _surfVerts, numbers);
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        out_slot, mqueue, sizeof(double), cudaMemcpyDeviceToDevice));
}
