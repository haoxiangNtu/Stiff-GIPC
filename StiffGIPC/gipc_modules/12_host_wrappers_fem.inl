void GIPC::buildBVH_and_CP_perenv(double dHat)
{
    if(m_skip_all_collision) return;
    int NG = m_perenv_bvh_groups;
    // point the BVH at LOCAL verts (the whole reason this kills cross-env divergence)
    double3* saved_f = bvh_f._vertexes;
    double3* saved_e = bvh_e._vertexes;
    bvh_f._vertexes = _vertexes;
    bvh_e._vertexes = _vertexes;
    (void)NG;
    bool skipF = getenv("STIFF_SKIP_F"), skipE = getenv("STIFF_SKIP_E");  // [decomp] per-env skips
    // [perenv-parallel #1] STIFF_PERENV_PAR: run envs concurrently on a K-stream scratch pool.
    // Value-aware (=0 disables) — isolated mode defaults it ON via the Python resolver.
    const char* _pe_par = getenv("STIFF_PERENV_PAR");
    bool par = _pe_par && atoi(_pe_par) != 0;
    int  K   = 1;
    if(par) { int cap = getenv("STIFF_PERENV_K") ? atoi(getenv("STIFF_PERENV_K")) : 8;  // concurrency cap
              K = (int)h_perenv_active.size(); if(K > cap) K = cap; if(K < 1) K = 1; allocPerEnvPool(K); }
    // snapshot bvh scratch so we can point-swap per env + restore at the end.
    BvhScratch of{bvh_f._nodes,bvh_f._bvs,bvh_f._MChash,bvh_f._indices,bvh_f._tempLeafBox,bvh_f._flags,bvh_f.m_node_env,
                  bvh_f._sort_tmp,bvh_f._sort_tmp_bytes,bvh_f._mch_alt,bvh_f._idx_alt,bvh_f._sort_cap};
    BvhScratch oe{bvh_e._nodes,bvh_e._bvs,bvh_e._MChash,bvh_e._indices,bvh_e._tempLeafBox,bvh_e._flags,bvh_e.m_node_env,
                  bvh_e._sort_tmp,bvh_e._sort_tmp_bytes,bvh_e._mch_alt,bvh_e._idx_alt,bvh_e._sort_cap};
    auto swapIn = [](lbvh& b, BvhScratch& s){ b._nodes=s.nodes; b._bvs=s.bvs; b._MChash=s.mch;
        b._indices=s.idx; b._tempLeafBox=s.tmp; b._flags=s.flags; b.m_node_env=s.node_env;
        b._sort_tmp=s.sort_tmp; b._sort_tmp_bytes=s.sort_bytes;   // [perenv-parallel #2]
        b._mch_alt=s.mch_alt; b._idx_alt=s.idx_alt; b._sort_cap=s.sort_cap; };
  perenv_redo:
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
    // [FIX] memsets are on DEFAULT stream; per-env detects on POOL streams. Sync once so the zero is
    // globally visible before any pool-stream detect atomicAdds to _cpNum (else garbage slot -> OOB).
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int i = 0;
    for(int e : h_perenv_active)
    {
        cudaStream_t st = par ? m_pool_streams[i % K] : (cudaStream_t)0;
        if(h_perenv_face_cnt[e] > 0 && !skipF)
        {
            if(par) swapIn(bvh_f, m_pool_f[i % K]);
            bvh_f._active_idx        = d_perenv_face_idx + h_perenv_face_off[e];
            bvh_f.face_number_active = h_perenv_face_cnt[e];
            bvh_f.Construct(st);
            bvh_f.SelfCollitionDetect(dHat, st);
        }
        if(h_perenv_edge_cnt[e] > 0 && !skipE)
        {
            if(par) swapIn(bvh_e, m_pool_e[i % K]);
            bvh_e._active_idx        = d_perenv_edge_idx + h_perenv_edge_off[e];
            bvh_e.face_number_active = h_perenv_edge_cnt[e];  // base member reused as edge active count
            bvh_e.Construct(st);
            bvh_e.SelfCollitionDetect(dHat, st);
        }
        ++i;
    }
    if(par) { for(int k = 0; k < K; ++k) CUDA_SAFE_CALL(cudaStreamSynchronize(m_pool_streams[k]));
              swapIn(bvh_f, of); swapIn(bvh_e, oe); }  // restore original scratch
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, _cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // [perenv-parallel #1 FIX] the per-env path (like the merged path) MUST grow the pair buffers on
    // overflow + redo — else at large N the pair count exceeds the cap and consumers (line-search,
    // gradient) read OOB → illegal access. Per-env DCD fills BOTH _collisonPairs and _ccd_collisonPairs
    // (1:1), so grow both. Emits past cap went to the trash slot, so nothing was corrupted.
    if((int)h_cpNum[0] > MAX_COLLITION_PAIRS_NUM || (int)h_cpNum[0] > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_cpNum[0] + h_cpNum[0] / 2) + 1;
        printf("[perenv DCD-grow] h_cpNum=%u > cap(dcd=%d,ccd=%d) -> grow to %d, redo\n",
               h_cpNum[0], MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        CUDA_SAFE_CALL(cudaFree(_collisonPairs));
        CUDA_SAFE_CALL(cudaFree(_MatIndex));
        CUDA_SAFE_CALL(cudaFree(_ccd_collisonPairs));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs,     ((size_t)newcap + 1) * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex,          ((size_t)newcap + 1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_ccd_collisonPairs, ((size_t)newcap + 1) * sizeof(int4)));
        MAX_COLLITION_PAIRS_NUM     = newcap;
        if(newcap > MAX_CCD_COLLITION_PAIRS_NUM) MAX_CCD_COLLITION_PAIRS_NUM = newcap;
        bvh_f._collisionPair     = bvh_e._collisionPair     = _collisonPairs;
        bvh_f._ccd_collisionPair = bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        bvh_f._MatIndex          = bvh_e._MatIndex          = _MatIndex;
        set_emit_caps(MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM);
        goto perenv_redo;
    }
    bvh_f._active_idx = nullptr; bvh_f.face_number_active = 0;
    bvh_e._active_idx = nullptr; bvh_e.face_number_active = 0;
    bvh_f._vertexes = saved_f;
    bvh_e._vertexes = saved_e;
    CUDA_SAFE_CALL(cudaMemsetAsync(_gdCollapse, 0, sizeof(int), 0));  // [d-floor fail-fast] reset per detection
    if(!getenv("STIFF_SKIP_GRND")) GroundCollisionDetect();
    {   // [9d28824-port] one 6-int D2H
        uint32_t cp_gp_buf[6];
        CUDA_SAFE_CALL(cudaMemcpy(cp_gp_buf, _cpNum, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        memcpy(h_cpNum, cp_gp_buf, 5 * sizeof(uint32_t));
        h_gpNum = cp_gp_buf[5];
    }

    snapshotDcdCcdPairs();   // [narrow-self snapshot] per-env DCD exit (see merged exit)
    throwIfGroundDistanceInvalid();
}

AABB* GIPC::calcuMaxSceneSize()
{
    return bvh_f.getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha, const double* alpha_dev)
{
    if(m_skip_all_collision)
        return;
    // [multi-env P2] per-env mode builds swept trees inside buildFullCP; skip merged build.
    if(m_perenv_bvh && m_perenv_bvh_groups > 0)
        return;
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _addEnvOffset<<<gs, bs>>>(d_bvh_vertexes, _vertexes, d_env_offset, vertexNum); }
    bvh_f.ConstructFullCCD(_moveDir, alpha, 0, alpha_dev);
    bvh_e.ConstructFullCCD(_moveDir, alpha, 0, alpha_dev);
}

void GIPC::calBarrierGradientAndHessian(double3* _gradient, double mKappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(
        _vertexes,
        _rest_vertexes,
        _collisonPairs,
        _gradient,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        _cpNum,
        _MatIndex,
        dHat,
        mKappa,
        h_cpNum[4],
        h_cpNum[3],
        h_cpNum[2],
        numbers,
        m_pergroup_kappa ? m_kappa_group : nullptr,   // [per-group κ] nullptr → scalar
        m_pergroup_kappa ? m_d_p2g : nullptr);
}


void GIPC::calBarrierHessian()
{

    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;   // [split-GH] parity with the fused kernel launch
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    _calBarrierHessian<<<blockNum, threadNum>>>(_vertexes,
                                                _rest_vertexes,
                                                _collisonPairs,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                _cpNum,
                                                _MatIndex,
                                                dHat,
                                                Kappa,
                                                h_cpNum[4],
                                                h_cpNum[3],
                                                h_cpNum[2],
                                                numbers,
                                                m_pergroup_kappa ? m_kappa_group : nullptr,   // [split-GH]
                                                m_pergroup_kappa ? m_d_p2g : nullptr);
}

static void _dbg_ksum(const char*, const void*, size_t);            // [4.3 fwd]
static void _dbg_ksum_comm(const char*, const void*, size_t);       // [4.3 fwd]

void GIPC::calFrictionHessian(device_TetraData& TetMesh)
{
    int numbers = h_cpNum_last[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    if(numbers > 0)
    {
        // [v0.8.5 fix] _calFrictionHessian ranks its M12/M9/M6 slots via
        // atomicAdd(_cpNum+4/3/2), but those counters still hold THIS frame's
        // barrier type counts here, so friction ranks started at n4/n3/n2 and
        // the displaced blocks landed outside the friction segment (silently
        // lost/overwritten). Zero the rank counters first.
        CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum + 2, 0, 3 * sizeof(uint32_t), 0));
        if(getenv("STIFF_KSUM"))
        {
            cudaDeviceSynchronize();
            _dbg_ksum("fric_o_verts", TetMesh.o_vertexes, (size_t)vertexNum * sizeof(double3));
            _dbg_ksum("fric_cur_verts", _vertexes, (size_t)vertexNum * sizeof(double3));
            _dbg_ksum_comm("fric_distCoord", distCoord, (size_t)numbers * sizeof(double2));
            _dbg_ksum_comm("fric_tanBasis", tanBasis, (size_t)numbers * sizeof(__GEIGEN__::Matrix3x2d));
            _dbg_ksum_comm("fric_lambdaH", lambda_lastH_scalar, (size_t)numbers * sizeof(double));
            _dbg_ksum_comm("fric_pairs", _collisonPairs_lastH, (size_t)numbers * sizeof(int4));
        }
        _calFrictionHessian<<<blockNum, threadNum>>>(
            _vertexes,
            TetMesh.o_vertexes,
            _collisonPairs_lastH,
            gipc_global_triplet.block_values(),
            gipc_global_triplet.block_row_indices(),
            gipc_global_triplet.block_col_indices(),
            _cpNum,
            numbers,
            IPC_dt,
            distCoord,
            tanBasis,
            fDhat * IPC_dt * IPC_dt,
            lambda_lastH_scalar,
            frictionRate,
            d_vert_mu,  // [per-body friction]
            h_cpNum[4],
            h_cpNum[3],
            h_cpNum[2],
            h_cpNum_last[4],
            h_cpNum_last[3],
            h_cpNum_last[2]);
    }

    numbers = h_gpNum_last;
    CUDA_SAFE_CALL(cudaMemcpy(_gpNum, &h_gpNum_last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    if(numbers < 1)
        return;

    blockNum = (numbers + threadNum - 1) / threadNum;
    int global_offset = gipc_global_triplet.global_triplet_offset + h_cpNum_last[4] * M12_Off
                        + h_cpNum_last[3] * M9_Off + h_cpNum_last[2] * M6_Off;
    _calFrictionHessian_gd<<<blockNum, threadNum>>>(
        _vertexes,
        TetMesh.o_vertexes,
        _groundNormal,
        _collisonPairs_lastH_gd,
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        numbers,
        IPC_dt,
        fDhat * IPC_dt * IPC_dt,
        lambda_lastH_scalar_gd,
        global_offset,
        gd_frictionRate, d_vert_mu_gd);  // [per-body friction]
}

void GIPC::computeSelfCloseVal()
{
    int numbers = h_cpNum[0];
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _collisonPairs, _closeMConstraintID, _closeMConstraintVal, _close_cpNum, dTol, numbers);
    // See computeCloseGroundVal: do not revive the legacy in-frame doubling
    // path without a separately validated adaptive-contact redesign.
}

bool GIPC::checkSelfCloseVal()
{
    int numbers = h_close_cpNum;
    if(numbers < 1)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    int*               _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(
        _vertexes, _isChange, _closeMConstraintID, _closeMConstraintVal, numbers,
        m_pergroup_kappa ? m_d_close_grp : nullptr, m_pergroup_kappa ? m_d_p2g : nullptr);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist()
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(
        _vertexes, _collisonPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

// void GIPC::calBarrierGradient(double3* _gradient, double mKappa) {
//     int numbers = h_cpNum[0];
//     if (numbers < 1)return;
//     const unsigned int threadNum = 256;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     _calBarrierGradient << <blockNum, threadNum >> > (_vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers);
// }

// ===================== per-contact force export (Step B) =====================
// Ground contacts use the simple distance barrier (lambda * ground_normal).
// Body-body & FEM-coupled contacts reuse the exact I5/NEWF barrier gradient
// via the _calBarrierGradient per-contact hook (see _ec_emit).

__global__ void _exportGroundContactForces(const double3*   _vertexes,
                                           const uint32_t*  envPair,
                                           const double3*   g_normal,
                                           const double*    g_offset,
                                           const int*       _point_body_id,
                                           double Kappa, double dHat, double dt,
                                           int number, int base,
                                           int2* out_pair, double3* out_force)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int    gidx = (int)envPair[idx];
    double3 nrm = g_normal[0];
    double  dist = __GEIGEN__::__v_vec_dot(nrm, _vertexes[gidx]) - g_offset[0];
    double  dis  = dist * dist;
    // [d-floor fail-fast] clamp removed; buildCP() throws before d can collapse here.
    double t      = dis - dHat;
    double g_b    = t * log(dis / dHat) * -2.0 - (t * t) / dis;
    double lambda = -Kappa * 2.0 * sqrt(dis) * g_b;
    double c      = lambda / (dt * dt);
    out_force[base + idx] = make_double3(c * nrm.x, c * nrm.y, c * nrm.z);
    out_pair[base + idx]  = make_int2(_point_body_id[gidx], -1);
}

int GIPC::exportContacts(int2* out_pair, double3* out_force)
{
    int ncp = (int)h_cpNum[0];
    int ngp = (int)h_gpNum;
    const unsigned int threads = 256;
    double inv_dt2 = (IPC_dt > 0.0) ? 1.0 / (IPC_dt * IPC_dt) : 0.0;
    // body-body: reuse the EXACT I5/NEWF barrier gradient (calBarrierGradient)
    // with the per-contact export hook. Needs a per-vertex scratch gradient.
    if(ncp > 0)
    {
        if(vertexNum > _ec_scratch_cap)
        {
            if(_ec_grad_scratch) CUDA_SAFE_CALL(cudaFree(_ec_grad_scratch));
            _ec_scratch_cap = vertexNum;
            CUDA_SAFE_CALL(cudaMalloc((void**)&_ec_grad_scratch, _ec_scratch_cap * sizeof(double3)));
        }
        CUDA_SAFE_CALL(cudaMemset(_ec_grad_scratch, 0, vertexNum * sizeof(double3)));
        // default body-body entries to "skip" (bodyA<0); _ec_emit overwrites
        // the ones it attributes.
        CUDA_SAFE_CALL(cudaMemset(out_pair, 0xFF, ncp * sizeof(int2)));   // -1,-1
        CUDA_SAFE_CALL(cudaMemset(out_force, 0, ncp * sizeof(double3)));
        calBarrierGradient(_ec_grad_scratch, Kappa, out_pair, out_force, _point_body_id, inv_dt2);
    }
    // ground: simple distance barrier (lambda * ground_normal)
    if(ngp > 0)
    {
        int blocks = (ngp + threads - 1) / threads;
        _exportGroundContactForces<<<blocks, threads>>>(
            _vertexes, _environment_collisionPair, _groundNormal, _groundOffset,
            _point_body_id, Kappa, dHat, IPC_dt, ngp, ncp, out_pair, out_force);
    }
    return ncp + ngp;
}

void GIPC::calBarrierGradient(double3* _gradient, double mKappa,
                              int2* ec_pair, double3* ec_force,
                              const int* ec_pbid, double ec_inv_dt2,
                              bool use_group_kappa)
{
    int numbers = h_cpNum[0];
    if(numbers < 1)
        return;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;


    _calBarrierGradient<<<blockNum, threadNum>>>(
        _vertexes, _rest_vertexes, _collisonPairs, _gradient, dHat, mKappa, numbers,
        use_group_kappa && m_pergroup_kappa ? m_kappa_group : nullptr,
        use_group_kappa && m_pergroup_kappa ? m_d_p2g : nullptr,
        ec_pair, ec_force, ec_pbid, ec_inv_dt2);
}

void GIPC::calFrictionGradient(double3* _gradient, device_TetraData& TetMesh)
{
    int                numbers   = h_cpNum_last[0];
    const unsigned int threadNum = 256;
    int                blockNum  = 0;
    if(numbers > 0)
    {
        blockNum = (numbers + threadNum - 1) / threadNum;
        _calFrictionGradient<<<blockNum, threadNum>>>(_vertexes,
                                                      TetMesh.o_vertexes,
                                                      _collisonPairs_lastH,
                                                      _gradient,
                                                      numbers,
                                                      IPC_dt,
                                                      distCoord,
                                                      tanBasis,
                                                      fDhat * IPC_dt * IPC_dt,
                                                      lambda_lastH_scalar,
                                                      frictionRate,
                                                      d_vert_mu);  // [per-body friction]
    }
    numbers = h_gpNum_last;
    if(numbers < 1)
        return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(_vertexes,
                                                     TetMesh.o_vertexes,
                                                     _groundNormal,
                                                     _collisonPairs_lastH_gd,
                                                     _gradient,
                                                     numbers,
                                                     IPC_dt,
                                                     fDhat * IPC_dt * IPC_dt,
                                                     lambda_lastH_scalar_gd,
                                                     gd_frictionRate, d_vert_mu_gd);  // [per-body friction]
}


void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}


void calculate_fem_gradient_hessian(__GEIGEN__::Matrix3x3d* DmInverses,
                                    const double3*          vertexes,
                                    const uint4*            tetrahedras,
                                    const double*           volume,
                                    double3*                gradient,
                                    int                     tetrahedraNum_FEM,
                                    int                     tetrahedraNum_ABD,
                                    const double*           lenRate,
                                    const double*           volRate,
                                    int                     global_offset,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    double                  IPC_dt,
                                    int global_hessian_fem_offset,
                                    const int*              tet_to_abd_body /* nullable, indexed in GLOBAL tet space */)
{
    int numbers = tetrahedraNum_FEM;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian<<<blockNum, threadNum>>>(
        DmInverses + tetrahedraNum_ABD,
        vertexes,
        tetrahedras + tetrahedraNum_ABD,
        volume + tetrahedraNum_ABD,
        gradient,
        numbers,
        lenRate + tetrahedraNum_ABD,
        volRate + tetrahedraNum_ABD,
        //tet_ids,
        global_offset,
        triplet_values,
        row_ids,
        col_ids,
        IPC_dt,
        global_hessian_fem_offset,
        tet_to_abd_body ? tet_to_abd_body + tetrahedraNum_ABD : nullptr);
}

void calculate_triangle_fem_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                             const double3*   vertexes,
                                             const uint3*     triangles,
                                             const double*    area,
                                             double3*         gradient,
                                             int              triangleNum,
                                             double           stretchStiff,
                                             double           shearStiff,
                                             double           strainRate,
                                             int              global_offset,
                                             Eigen::Matrix3d* triplet_values,
                                             int*             row_ids,
                                             int*             col_ids,
                                             double           IPC_dt,
                                             int global_hessian_fem_offset)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient_hessian<<<blockNum, threadNum>>>(triDmInverses,
                                                                      vertexes,
                                                                      triangles,
                                                                      area,
                                                                      gradient,
                                                                      triangleNum,
                                                                      stretchStiff,
                                                                      shearStiff,
                                                                      IPC_dt,
                                                                      global_offset,
                                                                      triplet_values,
                                                                      row_ids,
                                                                      col_ids,
                                                                      strainRate,
                                                                      global_hessian_fem_offset);
}


void calculate_triangle_fem_strain_limiting_gradient_hessian(__GEIGEN__::Matrix2x2d* triDmInverses,
                                                             const double3* vertexes,
                                                             const uint3* triangles,
                                                             __GEIGEN__::Matrix9x9d* Hessians,
                                                             const uint32_t& offset,
                                                             const double* area,
                                                             double3* gradient,
                                                             int triangleNum,
                                                             Eigen::Matrix3d* U3x2,
                                                             Eigen::Matrix2d* V3x2,
                                                             Eigen::Vector2d* S3x2,
                                                             double IPC_dt)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_strain_limiting_gradient_hessian<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, Hessians, offset, area, gradient, triangleNum, U3x2, V3x2, S3x2, IPC_dt);
}


void calculate_triangle_fem_deformationF(__GEIGEN__::Matrix2x2d* triDmInverses,
                                         const double3*          vertexes,
                                         const uint3*            triangles,
                                         int                     triangleNum,
                                         Eigen::Matrix<double, 3, 2>* F3x2)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_deformationF<<<blockNum, threadNum>>>(triDmInverses,
                                                                  vertexes,
                                                                  triangles,

                                                                  triangleNum,
                                                                  F3x2);
}

void calculate_bending_gradient_hessian(const double3*   vertexes,
                                        const double3*   rest_vertexes,
                                        const uint2*     edges,
                                        const uint2*     edges_adj_vertex,
                                        double3*         gradient,
                                        int              edgeNum,
                                        double           bendStiff,
                                        int              global_offset,
                                        Eigen::Matrix3d* triplet_values,
                                        int*             row_ids,
                                        int*             col_ids,
                                        double           IPC_dt,
                                        int global_hessian_fem_offset)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_bending_gradient_hessian<<<blockNum, threadNum>>>(vertexes,
                                                                 rest_vertexes,
                                                                 edges,
                                                                 edges_adj_vertex,
                                                                 gradient,
                                                                 edgeNum,
                                                                 bendStiff,
                                                                 global_offset,
                                                                 triplet_values,
                                                                 row_ids,
                                                                 col_ids,
                                                                 IPC_dt,
                                                                 global_hessian_fem_offset);
}


#ifdef USE_QUADRATIC_BENDING
void calculate_quad_bending_gradient_hessian(const double3* vertexes,
                                             const double3* rest_vertexes,
                                             const uint2*   edges,
                                             const uint2*   edges_adj_vertex,
                                             const Eigen::Matrix4d* quad_bending_Q,
                                             double3*         gradient,
                                             int              edgeNum,
                                             double           bendStiff,
                                             int              global_offset,
                                             Eigen::Matrix3d* triplet_values,
                                             int*             row_ids,
                                             int*             col_ids,
                                             double           IPC_dt,
                                             int global_hessian_fem_offset)
{
    int numbers = edgeNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_quad_bending_gradient_hessian<<<blockNum, threadNum>>>(vertexes,
                                                                      rest_vertexes,
                                                                      edges,
                                                                      edges_adj_vertex,
                                                                      quad_bending_Q,
                                                                      gradient,
                                                                      edgeNum,
                                                                      bendStiff,
                                                                      global_offset,
                                                                      triplet_values,
                                                                      row_ids,
                                                                      col_ids,
                                                                      IPC_dt,
                                                                      global_hessian_fem_offset);
}
#endif


void calculate_fem_gradient(__GEIGEN__::Matrix3x3d* DmInverses,
                            const double3*          vertexes,
                            const uint4*            tetrahedras,
                            const double*           volume,
                            double3*                gradient,
                            int                     tetrahedraNum,
                            double*                 lenRate,
                            double*                 volRate,
                            double                  dt)
{
    int numbers = tetrahedraNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, volume, gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(__GEIGEN__::Matrix2x2d* triDmInverses,
                                     const double3*          vertexes,
                                     const uint3*            triangles,
                                     const double*           area,
                                     double3*                gradient,
                                     int                     triangleNum,
                                     double                  stretchStiff,
                                     double                  shearStiff,
                                     double                  IPC_dt,
                                     double                  strainRate)
{
    int numbers = triangleNum;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt, strainRate);
}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number)
{

    int numbers = number;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void calcMinMovement_DeviceOut(const double3* _moveDir,
                               double* _queue,
                               const int& number)
{
    int numbers = number;
    if(numbers < 1)
    {
        CUDA_SAFE_CALL(cudaMemsetAsync(_queue, 0, sizeof(double)));
        return;
    }
    const unsigned int threadNum   = default_threads;
    int                blockNum    = (numbers + threadNum - 1) / threadNum;
    const unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_max_double3_to_double<<<blockNum, threadNum, sharedMsize>>>(
        _moveDir, _queue, numbers);
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        _reduct_max_double<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
}

__global__ void _newton_convergence_decide(const double* max_movement,
                                            double threshold,
                                            int* converged)
{
    *converged = (*max_movement < threshold) ? 1 : 0;
}

void stepForward(double3* _vertexes,
                 double3* _vertexesTemp,
                 double3* _moveDir,
                 int*     bType,
                 double   alpha,
                 bool     moveBoundary,
                 int      numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(
        _vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}

// [M1 substitution method] Hard-constraint projection kernel.
// world_pos = q.t + A(q) * local_pos
// Canonical q layout per abd_jacobi_matrix.inl operator*(ABDJacobi, Vector12):
//   q.v[0..2]  : translation t
//   q.v[3..5]  : A.row(0)
//   q.v[6..8]  : A.row(1)
//   q.v[9..11] : A.row(2)
// So (A * lp).x = q[3]*lp.x + q[4]*lp.y + q[5]*lp.z, etc.
// Called after ABD step_forward (each line-search alpha try). pinned FEM
// vertices follow ABD's q exactly, so their motion is consistent with the
// ABD body's affine transform, and IPC line search sees a smooth energy.
__global__ void _apply_fem_pins(double3* _vertexes,
                                const int* _pin_fem_vertex,
                                const int* _pin_abd_body_id,
                                const double3* _pin_abd_local_pos,
                                const __GEIGEN__::Vector12* _abd_q,
                                int n_pins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pins) return;
    int fem_v = _pin_fem_vertex[idx];
    int bid   = _pin_abd_body_id[idx];
    const __GEIGEN__::Vector12& q = _abd_q[bid];
    double3 lp = _pin_abd_local_pos[idx];
    // Previous code used q[3]/q[6]/q[9] for the x-component, which is
    // A.col(0)·lp = (A^T·lp)[0] — wrong for non-symmetric A.  Only
    // worked when A ≈ scale·I (e.g. Animated mode where the ABD
    // barely rotates).  Showed up under URDF arms with revolute
    // joints (case_35+): pinned FEM verts visually drift off the
    // rigid sub-mesh as the ABD body rotates.
    _vertexes[fem_v].x = q.v[0] + q.v[3] * lp.x + q.v[4]  * lp.y + q.v[5]  * lp.z;
    _vertexes[fem_v].y = q.v[1] + q.v[6] * lp.x + q.v[7]  * lp.y + q.v[8]  * lp.z;
    _vertexes[fem_v].z = q.v[2] + q.v[9] * lp.x + q.v[10] * lp.y + q.v[11] * lp.z;
}

void apply_fem_pins(double3* _vertexes,
                    const int* _pin_fem_vertex,
                    const int* _pin_abd_body_id,
                    const double3* _pin_abd_local_pos,
                    const void* _abd_q,
                    int n_pins)
{
    if(n_pins <= 0 || _abd_q == nullptr) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (n_pins + threadNum - 1) / threadNum;
    _apply_fem_pins<<<blockNum, threadNum>>>(
        _vertexes, _pin_fem_vertex, _pin_abd_body_id, _pin_abd_local_pos,
        reinterpret_cast<const __GEIGEN__::Vector12*>(_abd_q),
        n_pins);
}

void GIPC::step_forward(device_TetraData& TetMesh, double alpha, bool move_boundary)
{
    auto vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum};
    auto vertexes_temp = muda::BufferView<double3>{TetMesh.temp_double3Mem, vertexNum};
    auto move_dir = muda::BufferView<double3>{_moveDir, vertexNum};
    if(abd_fem_count_info.fem_point_num > 0)
    {
        auto fem_vertexes = vertexes.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);
        auto fem_vertexes_temp =
            vertexes_temp.subview(abd_fem_count_info.fem_point_offset,
                                  abd_fem_count_info.fem_point_num);

        auto fem_move_dir = move_dir.subview(abd_fem_count_info.fem_point_offset,
                                             abd_fem_count_info.fem_point_num);

        auto btype = muda::BufferView<int>{TetMesh.BoundaryType, vertexNum}.subview(
            abd_fem_count_info.fem_point_offset, abd_fem_count_info.fem_point_num);


        // [multi-env S2] per-env FEM step when active (env g's verts step by
        // m_env_alpha[g]); else the standard uniform step.
        if(m_perenv_apply && m_env_alpha && TetMesh.d_point_to_group
           && abd_fem_count_info.fem_point_num > 0)
        {
            const unsigned int tn = default_threads;
            int n = (int)fem_vertexes.size();
            int bn = (n + tn - 1) / tn;
            _stepForward_perenv<<<bn, tn>>>(
                fem_vertexes.data(), fem_vertexes_temp.data(), fem_move_dir.data(),
                btype.data(), TetMesh.d_point_to_group + abd_fem_count_info.fem_point_offset,
                m_env_alpha, alpha, move_boundary, n);
        }
        else
        {
            stepForward(fem_vertexes.data(),
                        fem_vertexes_temp.data(),
                        fem_move_dir.data(),
                        btype.data(),
                        alpha,
                        move_boundary,
                        fem_vertexes.size());
        }
    }
    if(abd_fem_count_info.abd_point_num <= 0)
        return;

    auto abd_vertexes = muda::BufferView<double3>{TetMesh.vertexes, vertexNum}.subview(
        abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num);

    // [multi-env S2] per-body ABD alpha when active (gathered from m_env_alpha via
    // body_to_group); nullptr -> uniform scalar alpha (baseline).
    const double* abd_alpha_ptr = nullptr;
    if(m_perenv_apply && m_abd_body_alpha && TetMesh.d_body_to_group)
        abd_alpha_ptr = m_abd_body_alpha;
    m_abd_system->step_forward(*m_abd_sim_data, abd_vertexes, alpha, abd_alpha_ptr);

    // [M1 substitution method] After ABD step_forward updates q, project
    // pinned FEM vertices to ABD-derived positions: world = q.t + R(q)*lp.
    // Combined with mass=∞ and BoundaryType=Fixed (set in finalize), the
    // pinned vertices' Δx from PCG is ~0 and step_forward leaves them
    // alone; this kernel writes the correct ABD-derived position. IPC
    // line search now sees a smooth energy E(alpha) along the ABD's q
    // direction.
    if(TetMesh.n_fem_pins > 0 && m_d_abd_body_q != nullptr)
    {
        apply_fem_pins(TetMesh.vertexes,
                       TetMesh.d_fem_pin_fem_vertex,
                       TetMesh.d_fem_pin_abd_body_id,
                       TetMesh.d_fem_pin_abd_local_pos,
                       m_d_abd_body_q,
                       TetMesh.n_fem_pins);
    }
}

void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfaces<<<blockNum, threadNum>>>(sortIndex, _faces, offset_num, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& offset_num, const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateEdges<<<blockNum, threadNum>>>(sortIndex, _edges, offset_num, numbers);
}

void updateTriEdges_adjVerts(uint32_t*  sortIndex,
                             uint2*     _tri_edges,
                             uint2*     _adj_verts,
                             const int& offset_num,
                             const int& numbers)
{
    const unsigned int threadNum = default_threads;
    if(numbers < 1)
        return;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateTriEdges_adjVerts<<<blockNum, threadNum>>>(
        sortIndex, _tri_edges, _adj_verts, offset_num, numbers);
}


void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& offset_num, const int& numbers)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateSurfVerts<<<blockNum, threadNum>>>(sortIndex, _sVerts, offset_num, numbers);
}

void updateNeighborInfo(unsigned int*   _neighborList,
                        unsigned int*   d_neighborListInit,
                        unsigned int*   _neighborNum,
                        unsigned int*   _neighborNumInit,
                        unsigned int*   _neighborStart,
                        unsigned int*   _neighborStartTemp,
                        const uint32_t* sortIndex,
                        const uint32_t* sortMapVertIndex,
                        const int&      numbers,
                        const int&      neighborListSize)
{
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateNeighborNum<<<blockNum, threadNum>>>(_neighborNumInit, _neighborNum, sortIndex, numbers);
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(_neighborNum),
                           thrust::device_ptr<unsigned int>(_neighborNum) + numbers,
                           thrust::device_ptr<unsigned int>(_neighborStartTemp));
    _updateNeighborList<<<blockNum, threadNum>>>(d_neighborListInit,
                                                 _neighborList,
                                                 _neighborNum,
                                                 _neighborStart,
                                                 _neighborStartTemp,
                                                 sortIndex,
                                                 sortMapVertIndex,
                                                 numbers);
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit,
                              _neighborList,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart,
                              _neighborStartTemp,
                              numbers * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        _neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

void calcTetMChash(uint64_t*         _MChash,
                   const double3*    _vertexes,
                   uint4*            tets,
                   const const AABB* _MaxBv,
                   const uint32_t*   sortMapVertIndex,
                   int               number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcTetMChash<<<blockNum, threadNum>>>(
        _MChash, _vertexes, tets, _MaxBv, sortMapVertIndex, number);
}

void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber)
{
    int numbers = std::max(traNumber, triNumber);
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTopology<<<blockNum, threadNum>>>(tets, tris, sortMapVertIndex, traNumber, triNumber);
}

void updateVertexes(double3*                      o_vertexes,
                    const double3*                _vertexes,
                    double*                       tempM,
                    const double*                 mass,
                    __GEIGEN__::Matrix3x3d*       tempCons,
                    int*                          tempBtype,
                    const __GEIGEN__::Matrix3x3d* cons,
                    const int*                    bType,
                    const uint32_t*               sortIndex,
                    uint32_t*                     sortMapIndex,
                    int                           number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateVertexes<<<blockNum, threadNum>>>(
        o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}

void updateTetrahedras(uint4*                        o_tetrahedras,
                       uint4*                        tetrahedras,
                       double*                       tempV,
                       const double*                 volum,
                       __GEIGEN__::Matrix3x3d*       tempDmInverse,
                       const __GEIGEN__::Matrix3x3d* dmInverse,
                       const uint32_t*               sortTetIndex,
                       const uint32_t*               sortMapVertIndex,
                       int                           number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _updateTetrahedras<<<blockNum, threadNum>>>(
        o_tetrahedras, tetrahedras, tempV, volum, tempDmInverse, dmInverse, sortTetIndex, sortMapVertIndex, number);
}

void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number)
{
    int numbers = number;
    if(numbers < 1)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash<<<blockNum, threadNum>>>(_MChash, _vertexes, _MaxBv, number);
}

void sortGeometry(device_TetraData& TetMesh,
                  const AABB*       _MaxBv,
                  const int&        vertex_num,
                  const int&        tetradedra_num,
                  const int&        triangle_num)
{


}

////////////////////////TO DO LATER/////////////////////////////////////////


void compute_H_b(double d, double dHat, double& H)
{
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

