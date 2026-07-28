// [C-2] contact-segment triplet total, computed from device-resident counts.
// Mirrors the host accumulation exactly: barrier(live cp 2/3/4) +
// friction(lastH stash 2/3/4 + gd stash, when armed) + ground(live gp slot 5).
__global__ void _calc_contact_triplet_total(int* d_total,
                                            const uint32_t* cp_live,
                                            const uint32_t* cp_fric,
                                            const uint32_t* gp_fric,
                                            int m12, int m9, int m6)
{
    long long t = (long long)cp_live[4] * m12 + (long long)cp_live[3] * m9
                  + (long long)cp_live[2] * m6;
#ifdef USE_FRICTION
    t += (long long)cp_fric[4] * m12 + (long long)cp_fric[3] * m9
         + (long long)cp_fric[2] * m6 + (long long)(*gp_fric);
#endif
    t += (long long)cp_live[5];
    *d_total = (int)t;
}

void GIPC::suggestKappa(double& kappa)
{
    double H_b;
    // [decouple] kappa's only batch-dependent input is the MERGED bboxDiagSize2 (dHat is already
    // abs_dhat-fixed). STIFF_DECOUPLE_THRESH uses the abs_dhat-fixed eff bbox → kappa batch-invariant
    // → env_0's barrier stiffness no longer depends on its batch-mates' contact state.
    double bb = bboxDiagSize2;
    // [abs-kappa consistency] when the user declares an ABSOLUTE contact scale (absolute_dhat>0),
    // κ's scale MUST follow it in ALL modes — deriving κ from the merged scene bbox dilutes the
    // barrier super-linearly with env count/spacing (softer, batch-dependent physics; ablation C2:
    // 258 vs 489 Newton was ENTIRELY this). Same consistency rule as the Newton-exit fix (c1d4d78)
    // and dHat/dTol/fDhat (init). uipc-style: stiffness from a physical contact scale, no bbox.
    // Scenes without absolute_dhat keep the classic bbox derivation.
    // [ablation diag] STIFF_DIAG_KAPPA_MERGEDBB forces the old merged-bbox κ (DIAGNOSTIC ONLY).
    if(!getenv("STIFF_DIAG_KAPPA_MERGEDBB")
       && absolute_dhat > 0.0 && relative_dhat > 0.0)
        bb = (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat);
    compute_H_b(1.0e-16 * bb, dHat, H_b);
    if(meanMass == 0.0)
    {
        kappa = minKappaCoef / (4.0e-16 * bb * H_b);
    }
    else
    {
        kappa = minKappaCoef * meanMass / (4.0e-16 * bb * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    double bb = bboxDiagSize2;   // [abs-kappa consistency] absolute_dhat ⇒ absolute κ scale
    if(!getenv("STIFF_DIAG_KAPPA_MERGEDBB")   // (see suggestKappa; diag = old-bbox escape)
       && absolute_dhat > 0.0 && relative_dhat > 0.0)
        bb = (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat);
    compute_H_b(1.0e-16 * bb, dHat, H_b);
    double kappaMax = 100 * minKappaCoef * meanMass / (4.0e-16 * bb * H_b);
    //printf("max Kappa: %f\n", kappaMax);
    if(meanMass == 0.0)
    {
        kappaMax = 100 * minKappaCoef / (4.0e-16 * bb * H_b);
    }

    if(kappa > kappaMax)
    {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(device_TetraData& TetMesh)
{
    // [batch-size fix] IPC_Solver calls initKappa() BEFORE the first computeGradientAndHessian(),
    // where per-group κ is normally enabled. On frame 0 that left m_pergroup_kappa=false, so env_0
    // fell back to the GLOBAL Kappa (= -gsum/gsnorm, a reduction over ALL envs' verts → N-dependent)
    // → the batch-SIZE divergence seed. Enable per-group κ here too so env_0 uses its OWN per-env κ
    // (binned over d_point_to_group) from the very first step → N-independent.
    if(m_mode_config.pergroup_kappa && TetMesh.d_point_to_group
       && TetMesh.h_groups_present   /* [N=1 guard] wildcard p2g -> kappa_grp[-1] OOB */
       && !m_pergroup_kappa)
    {
        m_pergroup_kappa = true;
        m_d_p2g          = TetMesh.d_point_to_group;
        const int NG     = TetMesh.h_group_count;
        m_active_group_count = NG;
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_kappa_group, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_close_grp, NG * sizeof(int)));
        // [audit lens-E fix] initialize BOTH device buffers at the enable
        // point: whichever enable site fires first, the freshly-malloc'ed
        // m_kappa_group must never be consumed as garbage — seed it with the
        // current scalar Kappa (the per-env initKappa overwrites it when it
        // runs); m_d_close_grp likewise starts as a defined all-zero mask.
        h_kappa_group.assign(NG, Kappa);
        CUDA_SAFE_CALL(cudaMemcpy(m_kappa_group, h_kappa_group.data(),
                                  NG * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(m_d_close_grp, 0, NG * sizeof(int)));
        printf("[pergroup-kappa] enabled (early, in initKappa) NG=%d\n", NG);
    }
    bool perenv_kappa_filled = false;   // [decouple] set when per-env κ replaces the stub broadcast
    if(h_cpNum[0] > 0 || h_gpNum > 0)
    {
        double3* _GE = TetMesh.fb;
        double3* _gc = TetMesh.temp_double3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));
        calKineticGradient(TetMesh.vertexes, TetMesh.xTilta, _GE, TetMesh.masses, vertexNum);
        // [multi-env determinism 4.3] elastic-side binned bracket: FEM + soft → g_gbin → _GE
        // (kinetic already written directly to _GE above).
        zeroBinnedGrad();
        calculate_fem_gradient(TetMesh.DmInverses,
                               TetMesh.vertexes,
                               TetMesh.tetrahedras,
                               TetMesh.volum,
                               _GE,
                               tetrahedraNum,
                               TetMesh.lengthRate,
                               TetMesh.volumeRate,
                               IPC_dt);
        //calculate_triangle_fem_gradient(TetMesh.triDmInverses, TetMesh.vertexes, TetMesh.triangles, TetMesh.area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        // soft constraint is the last elastic-side gradient → g_gbin; close the bracket → _GE:
        computeSoftConstraintGradient(_GE);
        combineBinnedGrad(_GE);
        // ground + barrier → _gc:
        zeroBinnedGrad();
        // Kappa estimation needs the UNIT contact gradient. Reusing the current
        // per-group stiffness here makes the estimate depend on the previous frame
        // (and reads uninitialized device memory on the first frame).
        computeGroundGradient(_gc, 1.0, false);
        calBarrierGradient(_gc, 1.0, nullptr, nullptr, nullptr, 0.0, false);
        combineBinnedGrad(_gc);

        int fem_offset = abd_fem_count_info.fem_point_offset;
        int fem_count  = abd_fem_count_info.fem_point_num;
        double gsum    = 0.0;
        double gsnorm  = 0.0;
        if(fem_count > 0)
        {
            gsum = reduction2Kappa(
                0, _gc + fem_offset, _GE + fem_offset, pcg_data.squeue, fem_count);
            gsnorm = reduction2Kappa(
                1, _gc + fem_offset, _GE + fem_offset, pcg_data.squeue, fem_count);
        }

        double abd_gsum   = 0.0;
        double abd_gsnorm = 0.0;
        if(abd_fem_count_info.abd_body_num > 0)
        {
            m_abd_system->setup_abd_non_contact_gradient(*m_abd_sim_data);
            m_abd_system->temp_system_gradient = m_abd_system->system_gradient;
            m_abd_system->add_abd_contact_gradient(
                *m_abd_sim_data,
                muda::CBufferView<double3>{_gc, abd_fem_count_info.abd_point_num});

            std::vector<double> abd_total_gradient;
            std::vector<double> abd_non_contact_gradient;
            m_abd_system->system_gradient.copy_to(abd_total_gradient);
            m_abd_system->temp_system_gradient.copy_to(abd_non_contact_gradient);
            for(size_t i = 0; i < abd_total_gradient.size(); ++i)
            {
                const double contact =
                    abd_total_gradient[i] - abd_non_contact_gradient[i];
                abd_gsum += contact * abd_non_contact_gradient[i];
                abd_gsnorm += contact * contact;
            }
            gsum += abd_gsum;
            gsnorm += abd_gsnorm;
        }
        if(getenv("STIFF_SEED_DIAG"))
            printf("[seed-kappa-dof] fem(dot=%.17e,norm=%.17e) "
                   "abd(dot=%.17e,norm=%.17e) total(dot=%.17e,norm=%.17e)\n",
                   gsum - abd_gsum, gsnorm - abd_gsnorm,
                   abd_gsum, abd_gsnorm, gsum, gsnorm);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = gsnorm > 0.0 ? -gsum / gsnorm : 0.0;
        if(minKappa > 0.0)
        {
            Kappa = minKappa;
        }
        suggestKappa(minKappa);
        if(Kappa < minKappa)
        {
            Kappa = minKappa;
        }
        upperBoundKappa(Kappa);
        // [decouple] PER-ENV initKappa: the global minKappa = -gsum/gsnorm above is a GLOBAL
        // reduction over ALL envs' DOFs → batch-dependent, and it overrides the batch-invariant
        // suggestKappa. Here we instead set each env's κ from ITS OWN gradient ratio. The DOF
        // representation must match the global path: free FEM vertices plus each ABD body's 12
        // generalized DOFs. ABD collision vertices must not be treated as independent DOFs.
        // suggested (=minKappa after suggestKappa, eff-bbox) is the batch-invariant floor; kmax the cap.
        if(m_mode_config.decouple_thresh && m_pergroup_kappa && m_kappa_group
           && TetMesh.d_point_to_group)
        {
            const int NG = TetMesh.h_group_count;
            double*& d_gsum_bin = m_scr_gsum_bin; double*& d_gsnorm_bin = m_scr_gsnorm_bin;
            double*& d_gsum_g = m_scr_gsum_g;   double*& d_gsnorm_g = m_scr_gsnorm_g;
            if(!d_gsum_bin) {
                cudaMalloc((void**)&d_gsum_bin,   (size_t)kEnvAlphaSlots * BINNED_K * sizeof(double));
                cudaMalloc((void**)&d_gsnorm_bin, (size_t)kEnvAlphaSlots * BINNED_K * sizeof(double));
                cudaMalloc((void**)&d_gsum_g,     kEnvAlphaSlots * sizeof(double));
                cudaMalloc((void**)&d_gsnorm_g,   kEnvAlphaSlots * sizeof(double));
            }
            cudaMemset(d_gsum_bin,   0, (size_t)NG * BINNED_K * sizeof(double));
            cudaMemset(d_gsnorm_bin, 0, (size_t)NG * BINNED_K * sizeof(double));
            int bs = 256;
            if(fem_count > 0)
            {
                int gs = (fem_count + bs - 1) / bs;
                _per_env_kappa_deposit<<<gs, bs>>>(
                    TetMesh.d_point_to_group + fem_offset,
                    _gc + fem_offset,
                    _GE + fem_offset,
                    d_gsum_bin,
                    d_gsnorm_bin,
                    fem_count,
                    NG);
            }
            int abd_body_count = static_cast<int>(abd_fem_count_info.abd_body_num);
            if(abd_body_count > 0 && TetMesh.d_body_to_group)
            {
                int abd_dof_count = abd_body_count * 12;
                int gs = (abd_dof_count + bs - 1) / bs;
                _per_env_abd_kappa_deposit<<<gs, bs>>>(
                    TetMesh.d_body_to_group,
                    m_abd_system->system_gradient.buffer_view().data(),
                    m_abd_system->temp_system_gradient.buffer_view().data(),
                    d_gsum_bin,
                    d_gsnorm_bin,
                    abd_body_count,
                    NG);
            }
            _per_env_kappa_combine<<<(NG + bs - 1) / bs, bs>>>(d_gsum_g, d_gsnorm_g,
                                                               d_gsum_bin, d_gsnorm_bin, NG);
            double suggested = minKappa;   // batch-invariant (suggestKappa wrote it, eff bbox)
            double H_b, bb = (absolute_dhat > 0.0 && relative_dhat > 0.0)
                             ? (absolute_dhat * absolute_dhat) / (relative_dhat * relative_dhat)
                             : bboxDiagSize2;
            compute_H_b(1.0e-16 * bb, dHat, H_b);
            double kmax = 100.0 * minKappaCoef * (meanMass == 0.0 ? 1.0 : meanMass)
                          / (4.0e-16 * bb * H_b);
            // [perf/device-residence] finalize κ per env ON DEVICE (in m_kappa_group) — no D2H(gsum/gsnorm)
            // + host loop + H2D. suggested/kmax are env-independent scalars → bit-identical → strict OK.
            _per_env_kappa_finalize<<<(NG + bs - 1) / bs, bs>>>(d_gsum_g, d_gsnorm_g,
                                                               m_kappa_group, NG, suggested, kmax);
            if(getenv("STIFF_SEED_DIAG"))   // diag only: mirror first entries back for the print below
            {
                if((int)h_kappa_group.size() < NG) h_kappa_group.resize(NG, Kappa);
                CUDA_SAFE_CALL(cudaMemcpy(h_kappa_group.data(), m_kappa_group,
                                          NG * sizeof(double), cudaMemcpyDeviceToHost));
            }
            perenv_kappa_filled = true;
        }
    }

    // [multi-env per-group κ] broadcast the init κ to all groups (STUB: all groups = global κ).
    // SKIPPED when the per-env initKappa above filled m_kappa_group with true per-env values.
    if(m_pergroup_kappa && m_kappa_group && !perenv_kappa_filled)
    {
        const int NG = TetMesh.h_group_count;
        if((int)h_kappa_group.size() < NG) h_kappa_group.resize(NG, Kappa);
        for(int g = 0; g < NG; g++) h_kappa_group[g] = Kappa;
        CUDA_SAFE_CALL(cudaMemcpy(m_kappa_group, h_kappa_group.data(), NG * sizeof(double), cudaMemcpyHostToDevice));
    }
    //printf("Kappa ====== %f\n", Kappa);
    if(getenv("STIFF_SEED_DIAG"))
        printf("[seed-kappa] Kappa=%.17g kappa_group[0]=%.17g kappa_group[1]=%.17g h_cpNum0=%u perenv_filled=%d\n",
               Kappa, (h_kappa_group.size() > 0 ? h_kappa_group[0] : -1.0),
               (h_kappa_group.size() > 1 ? h_kappa_group[1] : -1.0), h_cpNum[0], (int)perenv_kappa_filled);
}


void GIPC::partitionContactHessian()
{
    if(gipc_global_triplet.global_collision_triplet_offset <= 0)
    {
        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
        gipc_global_triplet.abd_abd_contact_num = 0;
        gipc_global_triplet.h_fem_fem_contact_start_id = 0;
        gipc_global_triplet.h_abd_fem_contact_start_id = 0;
        gipc_global_triplet.h_fem_abd_contact_start_id = 0;
        gipc_global_triplet.h_abd_abd_contact_start_id = 0;
        return;
    }

    // Contact partitioning is itself an out-of-place reorder: assembled
    // triplets live in [0,n), while _reorder_triplets writes [n,2n) before
    // the ranges are copied back. Dynamic pre-assembly growth only guarantees
    // [0,n), and the global converter's equivalent safety net runs later.
    // Grow here from the exact contact count while preserving [0,n).
    // [audit lens-A fix] use the GUARDED preserve API: the old bare
    // resize_triplets+reserve_triplets pair silently DESTROYED the live
    // triplets whenever contact_triplet_count exceeded the current capacity
    // (resize() is free->malloc on growth) — reachable in hybrid mode, which
    // skips the [P1-dyn] frame-start bound grow entirely. The guarded call
    // throws loudly instead of ever corrupting the matrix.
    const size_t contact_triplet_count = static_cast<size_t>(
        gipc_global_triplet.global_collision_triplet_offset);
    gipc_global_triplet.ensure_capacity_preserve(contact_triplet_count,
                                                 2 * contact_triplet_count);

    muda::DeviceRadixSort().SortPairs(gipc_global_triplet.block_hash_value(),
                                      gipc_global_triplet.block_sort_hash_value(),
                                      gipc_global_triplet.block_index(),
                                      gipc_global_triplet.block_sort_index(),
                                      gipc_global_triplet.global_collision_triplet_offset);

    int threadNum = 256;

    LaunchCudaKernal_default(
        gipc_global_triplet.global_collision_triplet_offset,
        threadNum,
        0,
        _reorder_triplets,
        gipc_global_triplet.block_row_indices(),
        gipc_global_triplet.block_col_indices(),
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_row_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_col_indices(gipc_global_triplet.global_collision_triplet_offset),
        gipc_global_triplet.block_values(gipc_global_triplet.global_collision_triplet_offset),
        (const uint32_t*)gipc_global_triplet.block_sort_index(),
        gipc_global_triplet.global_collision_triplet_offset);

    //gipc_global_triplet.d_abd_abd_contact_start_id = -1;
    //gipc_global_triplet.d_abd_fem_contact_start_id = -1;
    //gipc_global_triplet.d_fem_abd_contact_start_id = -1;
    //gipc_global_triplet.d_fem_fem_contact_start_id = -1;

    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_abd_fem_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_abd_contact_start_id, -1, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.d_fem_fem_contact_start_id, -1, sizeof(int)));

    size_t shareMem = (threadNum + 1) * sizeof(int);
    LaunchCudaKernal_default(gipc_global_triplet.global_collision_triplet_offset,
                             threadNum,
                             shareMem,
                             _partition_collision_triplets,
                             (const uint64_t*)gipc_global_triplet.block_sort_hash_value(),
                             gipc_global_triplet.d_abd_abd_contact_start_id,
                             gipc_global_triplet.d_abd_fem_contact_start_id,
                             gipc_global_triplet.d_fem_abd_contact_start_id,
                             gipc_global_triplet.d_fem_fem_contact_start_id,
                             //abd_fem_count_info.abd_point_num,
                             gipc_global_triplet.global_collision_triplet_offset);


    //gipc_global_triplet.h_abd_abd_contact_start_id =
    //    gipc_global_triplet.d_abd_abd_contact_start_id;
    //gipc_global_triplet.h_abd_fem_contact_start_id =
    //    gipc_global_triplet.d_abd_fem_contact_start_id;
    //gipc_global_triplet.h_fem_abd_contact_start_id =
    //    gipc_global_triplet.d_fem_abd_contact_start_id;
    //gipc_global_triplet.h_fem_fem_contact_start_id =
    //    gipc_global_triplet.d_fem_fem_contact_start_id;

    // ②-D2H: single batched copy of the 4 contiguous start-ids (block[0..3])
    // replaces 4 separate blocking D2H (each of which drains the GPU).
    int h_csb[4];
    CUDA_SAFE_CALL(cudaMemcpy(h_csb,
                              gipc_global_triplet.d_abd_abd_contact_start_id,
                              4 * sizeof(int),
                              cudaMemcpyDeviceToHost));
    gipc_global_triplet.h_abd_abd_contact_start_id = h_csb[0];
    gipc_global_triplet.h_abd_fem_contact_start_id = h_csb[1];
    gipc_global_triplet.h_fem_abd_contact_start_id = h_csb[2];
    gipc_global_triplet.h_fem_fem_contact_start_id = h_csb[3];


    if(gipc_global_triplet.h_fem_fem_contact_start_id >= 0)
    {
        if(gipc_global_triplet.h_abd_fem_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_fem_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            if(gipc_global_triplet.h_fem_abd_contact_start_id > 0)
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_fem_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_fem_abd_contact_start_id;
            }
            else
            {
                gipc_global_triplet.abd_fem_contact_num =
                    gipc_global_triplet.h_abd_abd_contact_start_id
                    - gipc_global_triplet.h_abd_fem_contact_start_id;

                gipc_global_triplet.fem_abd_contact_num = 0;
            }
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;
        }
        else if(gipc_global_triplet.h_abd_abd_contact_start_id > 0)
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.h_abd_abd_contact_start_id
                - gipc_global_triplet.h_fem_fem_contact_start_id;
            gipc_global_triplet.abd_abd_contact_num =
                gipc_global_triplet.global_collision_triplet_offset
                - gipc_global_triplet.h_abd_abd_contact_start_id;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
        else
        {
            gipc_global_triplet.fem_fem_contact_num =
                gipc_global_triplet.global_collision_triplet_offset;

            gipc_global_triplet.abd_abd_contact_num = 0;

            gipc_global_triplet.abd_fem_contact_num = 0;
            gipc_global_triplet.fem_abd_contact_num = 0;
        }
    }
    else if(gipc_global_triplet.h_abd_abd_contact_start_id >= 0)
    {
        gipc_global_triplet.abd_abd_contact_num =
            gipc_global_triplet.global_collision_triplet_offset;

        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }
    else
    {
        gipc_global_triplet.abd_abd_contact_num = 0;
        gipc_global_triplet.fem_fem_contact_num = 0;
        gipc_global_triplet.abd_fem_contact_num = 0;
        gipc_global_triplet.fem_abd_contact_num = 0;
    }

    gipc_global_triplet.h_fem_fem_contact_start_id = 0;
    gipc_global_triplet.h_abd_fem_contact_start_id =
        gipc_global_triplet.h_fem_fem_contact_start_id + gipc_global_triplet.fem_fem_contact_num;
    gipc_global_triplet.h_fem_abd_contact_start_id =
        gipc_global_triplet.h_abd_fem_contact_start_id + gipc_global_triplet.abd_fem_contact_num;
    gipc_global_triplet.h_abd_abd_contact_start_id =
        gipc_global_triplet.h_fem_abd_contact_start_id + gipc_global_triplet.fem_abd_contact_num;

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_row_indices(),
                   gipc_global_triplet.block_row_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(
        cudaMemcpy(gipc_global_triplet.block_col_indices(),
                   gipc_global_triplet.block_col_indices() + gipc_global_triplet.global_collision_triplet_offset,
                   gipc_global_triplet.global_collision_triplet_offset * sizeof(int),
                   cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(
        gipc_global_triplet.block_values(),
        gipc_global_triplet.block_values() + gipc_global_triplet.global_collision_triplet_offset,
        gipc_global_triplet.global_collision_triplet_offset * sizeof(Eigen::Matrix3d),
        cudaMemcpyDeviceToDevice));
}

static void _dbg_ksum_comm(const char* name, const void* dptr, size_t nbytes);  // [4.3 fwd]
#define KSEG(nm) if(getenv("STIFF_KSUM")) { cudaDeviceSynchronize(); \
    _dbg_ksum_comm(nm, gipc_global_triplet.block_values(), \
        (size_t)gipc_global_triplet.global_triplet_offset * 9 * sizeof(double)); }

// [decouple probe] file-scope frame/k counters so computeGradientAndHessian can gate per-stage
// shape-gradient dumps (set by solve_subIP). STIFF_SHAPE_STAGE dumps shape_grads after kinetic
// (.s1) and after the elastic bracket (.s2) → cross-batch compare splits kinetic vs elastic.
int g_dec_frame = -1;   // [decouple probe] non-static so other TUs (pcg_solver) can gate dumps by frame/k
int g_dec_k     = -1;

float GIPC::computeGradientAndHessian(device_TetraData& TetMesh)
{
    gipc::Timer timer{"cal_gradient_hessian"};

    // [multienv-mode] fast plain-atomic gradient for merged/isolated, binned
    // order-free gradient for strict/default. VALUE-tracked, not once-per-
    // process: a once-latch made a second engine in the same process silently
    // inherit the first engine's determinism mode (descriptor plan phase-0).
    // Republishing only on value change keeps the hot path at zero cost.
    static int s_binned_last = -1;
    {   // [det-gating] POSITIVE gate: determinism machinery is strict-mode opt-in
        // (STIFF_SPMV_DET, set by mode=strict). merged/isolated never pay the
        // bit-identity tax, even when the python resolve layer is bypassed.
        // STIFF_DIAG_BINNED_GRAD=1 forces binned (diagnostics).
        int det = (m_mode_config.spmv_det || getenv("STIFF_DIAG_BINNED_GRAD")) ? 1 : 0;
        if(det != s_binned_last)
        {
            set_binned_on(det);
            set_det_reduce(det);   // fem/MAS/ABD binned_deposit users, centrally
            s_binned_last = det;
        }
    }

    // [multi-env P2] capture d_point_to_group + enable per-env BVH (once). buildCP uses these
    // lazily (it has no TetMesh). STIFF_PERENV_BVH gates; needs grouped envs (d_point_to_group).
    if(m_mode_config.perenv_bvh && TetMesh.d_point_to_group
       && TetMesh.h_groups_present)   // [N=1 guard] all -1 p2g -> per-env index excludes
                                      // EVERY prim (active=0) -> ZERO self-collision
    {
        m_perenv_bvh = true;
        m_d_p2g      = TetMesh.d_point_to_group;
        m_active_group_count = TetMesh.h_group_count;
        m_d_b2g      = TetMesh.d_body_to_group;        // [iron-law] for env quarantine
        m_collision_body_count = TetMesh.collision_body_num;
    }
    // [multi-env cross-env diagnostic] capture p2g + report env0-vs-env1 vertex divergence at the
    // START of each gradient/Hessian (= verts from the previous step's line search). STIFF_XENV.
    if(getenv("STIFF_XENV") && TetMesh.d_point_to_group)
    {
        m_d_p2g = TetMesh.d_point_to_group;
        static int _xc = 0;
        char lbl[48]; snprintf(lbl, sizeof(lbl), "verts call#%d", _xc++);
        xenvDiff(_vertexes, lbl);
    }
    // [multi-env per-group κ] enable + allocate (once). STIFF_PERGROUP_KAPPA gates; needs groups.
    if(m_mode_config.pergroup_kappa && TetMesh.d_point_to_group
       && TetMesh.h_groups_present   /* [N=1 guard] wildcard p2g -> kappa_grp[-1] OOB */
       && !m_pergroup_kappa)
    {
        m_pergroup_kappa = true;
        m_d_p2g          = TetMesh.d_point_to_group;
        const int NG     = TetMesh.h_group_count;
        m_active_group_count = NG;
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_kappa_group, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_close_grp, NG * sizeof(int)));
        // [audit lens-E fix] initialize BOTH device buffers at the enable
        // point: whichever enable site fires first, the freshly-malloc'ed
        // m_kappa_group must never be consumed as garbage — seed it with the
        // current scalar Kappa (the per-env initKappa overwrites it when it
        // runs); m_d_close_grp likewise starts as a defined all-zero mask.
        h_kappa_group.assign(NG, Kappa);
        CUDA_SAFE_CALL(cudaMemcpy(m_kappa_group, h_kappa_group.data(),
                                  NG * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(m_d_close_grp, 0, NG * sizeof(int)));
        printf("[pergroup-kappa] enabled, NG=%d\n", NG);
    }

    CUDA_SAFE_CALL(cudaMemset(TetMesh.fb, 0, vertexNum * sizeof(double3)));
    // [multi-env determinism 4.3] zero the binned contact/friction gradient accumulator
    // (bins start at 0; deposits add exactly). Combined back into contact_grads after ground.
    CUDA_SAFE_CALL(cudaMemset(g_grad_binned, 0,
                              3 * (size_t)vertexNum * BINNED_K * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemset(TetMesh.shape_grads, 0, vertexNum * sizeof(double3)));

    // [multi-env determinism 4.3] zero the WHOLE triplet buffer (block values + row/col) to the
    // reserved capacity. The triplet count is a provable UPPER BOUND (16 slots/pair, but PP/PE/PT
    // write fewer) → reserved-but-unwritten slots otherwise hold GARBAGE that the converter
    // processes → non-deterministic matrix. Zeroing makes them (0,0)=0 (benign + deterministic).
    {
        size_t cap = gipc_global_triplet.triplet_capacity();
        if(cap > 0)
        {
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_values(), 0, cap * 9 * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_row_indices(), 0, cap * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_col_indices(), 0, cap * sizeof(int)));
        }
    }

    //muda::BufferView<double3>{TetMesh.shape_grads, vertexNum}.fill(double3{0, 0, 0});


    auto shape_grads   = TetMesh.shape_grads;
    auto contact_grads = TetMesh.fb;
    {
        gipc::Timer timer{"cal_kinetic_gradient"};
        calKineticGradient(
            TetMesh.vertexes, TetMesh.xTilta, shape_grads, TetMesh.masses, vertexNum);
    }
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(shape_grads, "  a.kinetic");

    // [decouple probe] sub-stage 1: shape_grads = kinetic only (per-vertex-local, expect clean).
    if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
       && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
       && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
    {
        std::vector<double3> h(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(h.data(), shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
        FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".s1").c_str(),"wb");
        if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
        printf("[shape-stage] s1 (kinetic) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
    }

    // [v0.8.5.1 frame-start grow] Capture the previous iteration's EXACT stream length
    // before the reset — it predicts this iteration's converter need (2*length) far
    // more tightly than the M12_Off-blocks-per-pair worst-case bound can.
    const long long prev_triplet_len = gipc_global_triplet.global_triplet_offset;
    gipc_global_triplet.global_triplet_offset = 0;

    // [P1-dyn] Grow the global triplet buffer BEFORE any assembly writes, to a PROVABLE
    // UPPER BOUND on this step's triplet count (so the unchecked assembly kernels can
    // NEVER overflow — not merely "usually fit"). The bound:
    //   - fixed (topology) internal triplets: m_fixed_triplet_base + fem_point (exact)
    //   - collision: h_cpNum[0] is the EXACT number of contact pairs calBarrier iterates;
    //     any pair writes at most M12_Off 3x3 blocks, so M12_Off*h_cpNum[0] >= the real
    //     collision-triplet count for ANY type mix (PP/PE/PT/EE). Same for lagged
    //     friction (h_cpNum_last[0]) and ground (h_gpNum, generous *M6_Off).
    // Non-hybrid only; hybrid keeps the worst-case finalize allocation.
    if(m_dynamic_triplet)
    {
        long long bound = m_fixed_triplet_base
            + static_cast<long long>(abd_fem_count_info.fem_point_num)
            + static_cast<long long>(h_cpNum[0]) * M12_Off     // all contact pairs x max blocks
            + static_cast<long long>(h_gpNum) * M6_Off;        // ground (generous)
#ifdef USE_FRICTION
        bound += static_cast<long long>(h_cpNum_last[0]) * M12_Off
               + static_cast<long long>(h_gpNum_last) * M6_Off;
#endif
        bound += 4096;                                          // fixed slack
        // [v0.8.5.1] This is the ONE point where the triplet buffer provably holds no
        // live data (offset just reset; the previous stream was fully consumed by its
        // solve) — so growth here may legally DISCARD (free→malloc, no copy, no
        // old+new transient): the [P0-mem] memory property, at the location where its
        // "nothing to preserve" premise is actually true. Size for BOTH consumers:
        //   - assembly writes [0:length): 1*bound covers it (provable upper bound);
        //   - the converter needs 2*length_now at the build point. length_now is
        //     unknown here; predict from the previous iteration's EXACT length with
        //     35% jump headroom (2.7 = 2 x 1.35). A plain 2*prev misses exactly the
        //     frames that matter: the towel-strict trigger was a +26% single-frame
        //     contact jump (2*prev=267864 < cap while 2*length_now=336938 > cap).
        // Jumps >35% per Newton iteration fall through to ensure_capacity_preserve
        // at the build point — the correctness backstop: rare, one transient copy,
        // always correct. Margin (30% capped at 512MB) lives in the callee.
        long long conv_pred = 27 * prev_triplet_len / 10;
        long long target    = bound > conv_pred ? bound : conv_pred;
        if(gipc_global_triplet.triplet_capacity() < static_cast<size_t>(target))
        {
            gipc_global_triplet.open_discard_window();  // [A2] THE one legal point
            gipc_global_triplet.ensure_capacity_discard(static_cast<size_t>(target));
            // The whole-buffer determinism memset above ran on the OLD allocation;
            // re-zero the fresh one (grow iterations only, so effectively free).
            size_t cap = gipc_global_triplet.triplet_capacity();
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_values(), 0, cap * 9 * sizeof(double)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_row_indices(), 0, cap * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemset(gipc_global_triplet.block_col_indices(), 0, cap * sizeof(int)));
        }
        if(gipc_global_triplet.global_external_max_capcity < bound)
        {
            gipc_global_triplet.resize_collision_hash_size(static_cast<size_t>(bound * 1.1));
            gipc_global_triplet.global_external_max_capcity = static_cast<int>(bound * 1.1);
        }
    }
    // [3c] offsets are reset and no assembly write has happened yet — the one
    // place arming the slot audit is legal (mirrors the discard-grow legality).
    gipc_global_triplet.slot_audit_arm();

    {
        gipc::Timer timer{"cal_barrier_gradient_hessian"};
        CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
        //calBarrierHessian();
        //calBarrierGradient(contact_grads, Kappa);

        { static int _bc = 0;
          int on = (_bc++ == 0 && getenv("STIFF_BAR_TRACE")) ? 1 : 0;
          int t0 = getenv("STIFF_BAR_TGT0") ? atoi(getenv("STIFF_BAR_TGT0")) : -1;
          int t1 = getenv("STIFF_BAR_TGT1") ? atoi(getenv("STIFF_BAR_TGT1")) : -1;
          set_bar_targets(on, t0, t1); }
        bool _split_gh = getenv("STIFF_SPLIT_GH") != nullptr;
#ifdef SymGH
        if(_split_gh)
        {   // [towel-strict audit] _calBarrierHessian still uses legacy 16/9/4 triplet
            // strides (+ bare I1==0 returns) — incompatible with the SymGH 10/6/3
            // layout: it would deposit garbage slots. Fall back to the fused kernel.
            static bool _split_gh_warned = false;
            if(!_split_gh_warned)
            {
                _split_gh_warned = true;
                printf("[split-GH] DISABLED under SymGH layout; using fused kernel.\n");
            }
            _split_gh = false;
        }
#endif
        if(_split_gh)
        {   // [split-GH experiment] launch gradient(152reg) + hessian-only kernels
            // instead of the fused 254-reg kernel; gradient lands in the SAME _gfx
            // accumulator (both kernels scatter via _gfxAdd). A/B flag, default off.
            int _numbers = h_cpNum[0];
            if(_numbers >= 1)
            {
                const unsigned int _tn = 256;
                int                _bn = (_numbers + _tn - 1) / _tn;
                _calBarrierGradient<<<_bn, _tn>>>(_vertexes, _rest_vertexes,
                    _collisonPairs, contact_grads, dHat, Kappa, _numbers,
                    m_pergroup_kappa ? m_kappa_group : nullptr,
                    m_pergroup_kappa ? m_d_p2g : nullptr);
            }
            calBarrierHessian();
        }
        else
        calBarrierGradientAndHessian(contact_grads, Kappa);
        set_bar_targets(0, -1, -1);
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum[4] * M12_Off + h_cpNum[3] * M9_Off + h_cpNum[2] * M6_Off;
    }
    KSEG("seg_contact")

    float time00 = 0;

#ifdef USE_FRICTION
    {

        gipc::Timer timer{"cal_friction_gradient_hessian"};
        if(!getenv("STIFF_SKIP_FRIC")) {   // [xenv pin] isolate friction's contribution to b
        calFrictionGradient(contact_grads, TetMesh);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        calFrictionHessian(TetMesh);
        }
        gipc_global_triplet.global_triplet_offset +=
            h_cpNum_last[4] * M12_Off + h_cpNum_last[3] * M9_Off
            + h_cpNum_last[2] * M6_Off + h_gpNum_last;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
#endif
    KSEG("seg_thru_friction")
    if(getenv("STIFF_KSUM")) printf("[ksum] cpNumLast=%u gpNumLast=%u\n", h_cpNum_last[0], h_gpNum_last.get());

    computeGroundGradientAndHessian(contact_grads);
    // [multi-env determinism 4.3] combine the binned contact+friction gradient into
    // contact_grads (holds the ground gradient). Order-independent ⇒ bit-identical.
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _gfxToGrad<<<gs, bs>>>(contact_grads, g_grad_binned, vertexNum); }
    if(getenv("STIFF_XENV") && m_d_p2g) {
        printf("[xenv]   (counts: cpNum=%u cpNumLast=%u gpNum=%u gpNumLast=%u)\n",
               h_cpNum[0], h_cpNum_last[0], (unsigned)h_gpNum, (unsigned)h_gpNum_last);
        xenvPairClassify(_collisonPairs, h_cpNum[0], "pairs");
        // [xenv dump] one-shot raw dump of pairs + p2g + verts to localize the differing pair.
        static bool _xdumped = false;
        if(getenv("STIFF_XENV_DUMP") && !_xdumped) {
            _xdumped = true;
            int n = h_cpNum[0];
            std::vector<int4> hp(n); std::vector<int> hg(vertexNum); std::vector<double3> hv(vertexNum);
            cudaMemcpy(hp.data(), _collisonPairs, (size_t)n*sizeof(int4), cudaMemcpyDeviceToHost);
            // [xenv] full-4-vert CCD pairs (no encoding loss) for clean env-local membership compare
            { std::vector<int4> hc(n);
              cudaMemcpy(hc.data(), _ccd_collisonPairs, (size_t)n*sizeof(int4), cudaMemcpyDeviceToHost);
              FILE* fc=fopen("/tmp/xd_ccd.bin","wb"); fwrite(hc.data(),sizeof(int4),n,fc); fclose(fc); }
            cudaMemcpy(hg.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hv.data(), _vertexes, (size_t)vertexNum*sizeof(double3), cudaMemcpyDeviceToHost);
            FILE* f;
            f=fopen("/tmp/xd_pairs.bin","wb"); fwrite(hp.data(),sizeof(int4),n,f); fclose(f);
            f=fopen("/tmp/xd_p2g.bin","wb");   fwrite(hg.data(),sizeof(int),vertexNum,f); fclose(f);
            f=fopen("/tmp/xd_verts.bin","wb"); fwrite(hv.data(),sizeof(double3),vertexNum,f); fclose(f);
            f=fopen("/tmp/xd_meta.txt","w");   fprintf(f,"%d %d %.17g\n",n,vertexNum,dHat); fclose(f);
            // [xenv] dump the EDGE list too — to check if env0/env1 edge sets are mirror-identical
            int nE = (int)bvh_e.edge_number;
            std::vector<uint2> he(nE);
            cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost);
            f=fopen("/tmp/xd_edges.bin","wb"); fwrite(he.data(),sizeof(uint2),nE,f); fclose(f);
            std::vector<double3> hfb(vertexNum);
            cudaMemcpy(hfb.data(), contact_grads, (size_t)vertexNum*sizeof(double3), cudaMemcpyDeviceToHost);
            f=fopen("/tmp/xd_fb.bin","wb"); fwrite(hfb.data(),sizeof(double3),vertexNum,f); fclose(f);
            // per-env edge count (by group of edge.x)
            int e0=0,e1=0; for(auto&e:he){ int g=hg[e.x]; if(g==0)e0++; else if(g==1)e1++; }
            printf("[xenv]   DUMPED %d pairs, %d verts, %d edges (env0=%d env1=%d), dHat=%.17g\n",
                   n, vertexNum, nE, e0, e1, dHat);
        }
        xenvDiff(contact_grads, "  b.barrier+fric+grnd");
    }
    gipc_global_triplet.global_triplet_offset += h_gpNum;
    KSEG("seg_thru_ground")
    gipc_global_triplet.global_collision_triplet_offset =
        gipc_global_triplet.global_triplet_offset;
    // [C-2] device mirror of the contact-segment triplet total: every factor
    // already lives on device (_cpNum live slots + the C-1 friction stashes),
    // so downstream FEM assembly offsets (= this + scene-constant strides) can
    // be read in-kernel — the prerequisite for the Newton-loop graph.
    _calc_contact_triplet_total<<<1, 1>>>(m_d_contact_triplet_total,
                                          _cpNum,
                                          m_scr_cp_friction,
                                          m_scr_gp_friction,
                                          M12_Off, M9_Off, M6_Off);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    gipc_global_triplet.update_hash_value(abd_fem_count_info.abd_point_num);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    partitionContactHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    {
        gipc::Timer timer{"setup_abd_system_gradient_hessian"};

        // Set stitch spring parameters for ABD-side bilateral coupling
        // Only report stitch count when data pointers are valid; otherwise
        // the ABD system reserves triplet slots that are never written,
        // leaving garbage in the preconditioner and causing illegal access.
        m_abd_system->m_stitch_count              = m_d_stitch_paired_vertex ? softNum : 0;
        m_abd_system->m_d_stitch_paired_vertex    = m_d_stitch_paired_vertex;
        m_abd_system->m_d_stitch_rest_offset      = m_d_stitch_rest_offset;
        m_abd_system->m_d_stitch_abd_body_id      = m_d_stitch_abd_body_id;
        m_abd_system->m_d_stitch_fem_vertex_id    = targetInd;
        m_abd_system->m_d_all_vertexes            = _vertexes;
        m_abd_system->m_stitch_motion_rate        = softMotionRate;
        m_abd_system->m_stitch_rate               = animation_fullRate;

        m_abd_system->setup_abd_system_gradient_hessian(
            *m_abd_sim_data,
            TetMesh.BoundaryType,
            muda::BufferView<double3>{TetMesh.fb, vertexNum}.subview(
                abd_fem_count_info.abd_point_offset, abd_fem_count_info.abd_point_num),
            gipc_global_triplet);
    }
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(TetMesh.fb, "  c.+abd");

    int abd_dofs = abd_fem_count_info.abd_body_num * 4;
    int fem_global_hessian_index_offset = -abd_fem_count_info.abd_point_num + abd_dofs;
    {
        muda::ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(gipc_global_triplet.fem_fem_contact_num,
                   [cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    cfem_vals = gipc_global_triplet.block_values(
                        gipc_global_triplet.h_fem_fem_contact_start_id),
                    BDType = TetMesh.BoundaryType,
                    fem_global_hessian_index_offset] __device__(int i) mutable
                   {
                       int row = cfem_rows[i];
                       int col = cfem_cols[i];
                       int btypeA = BDType[row];
                       int btypeB = BDType[col];
                       if(row <= col)
                       {
                           cfem_rows[i] = row + fem_global_hessian_index_offset;
                           cfem_cols[i] = col + fem_global_hessian_index_offset;
                           if(btypeA != 0 || btypeB != 0)
                           {
                               cfem_vals[i].setZero();
                           }
                       }
                       else
                       {
                           cfem_rows[i] = col + fem_global_hessian_index_offset;
                           cfem_cols[i] = row + fem_global_hessian_index_offset;
                           cfem_vals[i].setZero();
                       }
                   });
    }

    {
        gipc::Timer timer{"cal_fem_gradient_hessian"};
        int fem_triplet_start = gipc_global_triplet.global_triplet_offset;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        // [multi-env determinism 4.3] open the elastic-gradient binned bracket: FEM, bending,
        // triangle-FEM, strain-limiting AND soft-constraint all scatter to g_gbin; combined
        // into shape_grads after soft (kinetic is already in shape_grads, written directly).
        zeroBinnedGrad();
        calculate_fem_gradient_hessian(TetMesh.DmInverses,
                                       TetMesh.vertexes,
                                       TetMesh.tetrahedras,
                                       TetMesh.volum,
                                       shape_grads,
                                       abd_fem_count_info.fem_tet_num,
                                       abd_fem_count_info.abd_tet_num,
                                       TetMesh.lengthRate,
                                       TetMesh.volumeRate,
                                       gipc_global_triplet.global_triplet_offset,
                                       gipc_global_triplet.block_values(),
                                       gipc_global_triplet.block_row_indices(),
                                       gipc_global_triplet.block_col_indices(),
                                       IPC_dt,
                                       fem_global_hessian_index_offset,
                                       TetMesh.d_tet_to_abd_body);
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_tet_num * 10;


#ifdef USE_QUADRATIC_BENDING
        calculate_quad_bending_gradient_hessian(TetMesh.vertexes,
                                                TetMesh.rest_vertexes,
                                                TetMesh.tri_edges,
                                                TetMesh.tri_edge_adj_vertex,
                                                TetMesh.quad_bending_Q,
                                                shape_grads,
                                                tri_edge_num,
                                                bendStiff,
                                                gipc_global_triplet.global_triplet_offset,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                IPC_dt,
                                                fem_global_hessian_index_offset);
#else
        calculate_bending_gradient_hessian(TetMesh.vertexes,
                                           TetMesh.rest_vertexes,
                                           TetMesh.tri_edges,
                                           TetMesh.tri_edge_adj_vertex,
                                           shape_grads,
                                           tri_edge_num,
                                           bendStiff,
                                           gipc_global_triplet.global_triplet_offset,
                                           gipc_global_triplet.block_values(),
                                           gipc_global_triplet.block_row_indices(),
                                           gipc_global_triplet.block_col_indices(),
                                           IPC_dt,
                                           fem_global_hessian_index_offset);
#endif
        gipc_global_triplet.global_triplet_offset += tri_edge_num * 10;
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());

        calculate_triangle_fem_gradient_hessian(TetMesh.triDmInverses,
                                                TetMesh.vertexes,
                                                TetMesh.triangles,
                                                TetMesh.area,
                                                shape_grads,
                                                triangleNum,
                                                stretchStiff,
                                                shearStiff,
                                                strainRate,
                                                gipc_global_triplet.global_triplet_offset,
                                                gipc_global_triplet.block_values(),
                                                gipc_global_triplet.block_row_indices(),
                                                gipc_global_triplet.block_col_indices(),
                                                IPC_dt,
                                                fem_global_hessian_index_offset);

        gipc_global_triplet.global_triplet_offset += triangleNum * 6;


        // [decouple probe] sub-stage e_presoft: kinetic+FEM+bending+triangle (NO soft yet).
        // Combine current bins into a scratch copy (does NOT disturb shape_grads or the bins).
        if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            double3* scratch = TetMesh.temp_double3Mem;
            CUDA_SAFE_CALL(cudaMemcpy(scratch, shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToDevice));
            combineBinnedGrad(scratch);   // scratch = kinetic + FEM+bending+triangle (bins so far)
            std::vector<double3> h(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(h.data(), scratch, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".epresoft").c_str(),"wb");
            if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
            printf("[shape-stage] e_presoft (kin+fem+bend+tri, NO soft) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
        }

        // [multi-env determinism 4.3] soft constraint is the LAST elastic-side gradient; it
        // also scatters to g_gbin. Close the bracket: combine all of FEM+bending+triangle+
        // strain+soft into shape_grads (deterministic).
        computeSoftConstraintGradientAndHessian(shape_grads, fem_global_hessian_index_offset);
        combineBinnedGrad(shape_grads);
        if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(shape_grads, "  d.elastic(fem+bend+tri+soft)");
        // [decouple probe] sub-stage 2: shape_grads = kinetic + elastic-bracket (FEM+bend+tri+strain+soft).
        if(getenv("STIFF_SHAPE_STAGE") && getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            std::vector<double3> h(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(h.data(), shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            FILE* a=fopen((std::string(getenv("STIFF_GRAD_PRE"))+".s2").c_str(),"wb");
            if(a){fwrite(h.data(),sizeof(double3),vertexNum,a);fclose(a);}
            printf("[shape-stage] s2 (kinetic+elastic) dumped @frame %d k=%d\n", g_dec_frame, g_dec_k);
        }
        gipc_global_triplet.global_triplet_offset += softNum;
        KSEG("seg_end_soft")

        int fem_triplet_num = gipc_global_triplet.global_triplet_offset - fem_triplet_start;

        // [M3.5 substitution method] Chain-rule pinned-vertex FEM Hessian
        // rows/cols onto the ABD body's q-DOFs.  After this:
        //   pinned-pinned block H_pp  -> J_p^T * H_pp * J_p added at (body*4+r, body*4+c)
        //   pinned-free  block H_pf   -> J_p^T * H_pf added at (body*4+r, free_v_col)
        //   free-pinned  block H_fp   -> H_fp * J_p added at (free_v_row, body*4+c)
        //   in all cases, original (pinned-touching) triplet is zeroed.
        //
        // Without this routing, PCG sees pinned vertex as a free DOF whose
        // dx is later overridden by apply_fem_pins; the free-vertex dx is
        // computed against an incorrect dx_p estimate (elasticity-driven
        // instead of J*dq), and Newton fails to converge under joint
        // motion (k=1000 cap, verified by m5_drive_joint2_test).  After
        // routing, the substituted system has the pinned DOF's contribution
        // baked into the ABD body row, which Newton can solve consistently.
        //
        // Output goes to an "extension range" past the current
        // global_triplet_offset; we use an atomic counter for slot
        // allocation, then bump global_triplet_offset by the final count.
        if(TetMesh.n_fem_pins > 0 && TetMesh.vertex_to_pin_idx != nullptr)
        {
            int ext_start    = gipc_global_triplet.global_triplet_offset;
            // Reserve space (worst case: 16 expansions per pinned-touching triplet).
            // Actual count tracked via atomic counter.
            int ext_capacity = fem_triplet_num * 16;

            // Reset counter device-side.  Reuse the existing
            // d_unique_key_number scratch int* on GIPCTripletMatrix.
            CUDA_SAFE_CALL(cudaMemsetAsync(gipc_global_triplet.d_assembly_scratch_count,
                                           0, sizeof(int)));

            muda::ParallelFor(256)
                .file_line(__FILE__, __LINE__)
                .apply(fem_triplet_num,
                       [cfem_rows = gipc_global_triplet.block_row_indices(fem_triplet_start),
                        cfem_cols = gipc_global_triplet.block_col_indices(fem_triplet_start),
                        triplet_fem = gipc_global_triplet.block_values(fem_triplet_start),
                        ext_rows = gipc_global_triplet.block_row_indices(ext_start),
                        ext_cols = gipc_global_triplet.block_col_indices(ext_start),
                        ext_vals = gipc_global_triplet.block_values(ext_start),
                        ext_count = gipc_global_triplet.d_assembly_scratch_count,
                        BDType   = TetMesh.BoundaryType,
                        v2pin    = TetMesh.vertex_to_pin_idx,
                        pin_body = TetMesh.d_fem_pin_abd_body_id,
                        pin_lo   = TetMesh.d_fem_pin_abd_local_pos,
                        hess_index2fem_index = fem_global_hessian_index_offset,
                        ext_capacity] __device__(int i) mutable
                       {
                           int row = cfem_rows[i];
                           int col = cfem_cols[i];
                           int row_v = row - hess_index2fem_index;
                           int col_v = col - hess_index2fem_index;
                           int btypeA = BDType[row_v];
                           int btypeB = BDType[col_v];
                           if(btypeA == 0 && btypeB == 0)
                               return;  // both free: keep original

                           // Read original block, then zero it (will be replaced
                           // with chain-ruled triplets in extension range).
                           gipc::Matrix3x3 H = triplet_fem[i];
                           triplet_fem[i].setZero();

                           int pin_a = (btypeA != 0) ? v2pin[row_v] : -1;
                           int pin_b = (btypeB != 0) ? v2pin[col_v] : -1;

                           // Helper: append a triplet to extension range.
                           auto append = [&](int r, int c, const gipc::Matrix3x3& V) {
                               int slot = atomicAdd(ext_count, 1);
                               if(slot < ext_capacity) {
                                   ext_rows[slot] = r;
                                   ext_cols[slot] = c;
                                   ext_vals[slot] = V;
                               }
                           };

                           if(pin_a >= 0 && pin_b < 0)
                           {
                               // Row pinned, col free: write 4 triplets at
                               // (body*4+r, col) for r=0..3.
                               // J^T * H (12x3) split into 4 (3x3) sub-blocks.
                               int     body = pin_body[pin_a];
                               double3 lo3  = pin_lo[pin_a];
                               // Sub-block 0 = H itself
                               append(body * 4 + 0, col, H);
                               // Sub-block r (r=1,2,3) = lo * H[r-1, :]^T (outer product)
                               // == column-vec lo times row r-1 of H
                               #pragma unroll
                               for(int r = 1; r < 4; ++r)
                               {
                                   gipc::Matrix3x3 B;
                                   double Hr0 = H(r - 1, 0), Hr1 = H(r - 1, 1), Hr2 = H(r - 1, 2);
                                   B(0, 0) = lo3.x * Hr0; B(0, 1) = lo3.x * Hr1; B(0, 2) = lo3.x * Hr2;
                                   B(1, 0) = lo3.y * Hr0; B(1, 1) = lo3.y * Hr1; B(1, 2) = lo3.y * Hr2;
                                   B(2, 0) = lo3.z * Hr0; B(2, 1) = lo3.z * Hr1; B(2, 2) = lo3.z * Hr2;
                                   int new_row = body * 4 + r;
                                   if(new_row <= col) append(new_row, col, B);
                                   else               append(col, new_row, B.transpose());
                               }
                           }
                           else if(pin_a < 0 && pin_b >= 0)
                           {
                               // Col pinned, row free: write 4 triplets routed to
                               // (row, body*4+c) for c=0..3.  But row > body*4+c
                               // typically (FEM row is in [N_abd*4, ...) range and
                               // body*4+c is in [0, N_abd*4)).  So store at
                               // (body*4+c, row) with TRANSPOSED block to keep
                               // upper-triangle convention.
                               // H * J (3x12) split into 4 (3x3) sub-blocks per col.
                               int     body = pin_body[pin_b];
                               double3 lo3  = pin_lo[pin_b];
                               // Sub-block 0 (cols 0-2) = H itself
                               // Stored at (body*4+0, row) transposed = H.transpose()
                               int new_col = body * 4 + 0;
                               if(new_col <= row) append(new_col, row, H.transpose());
                               else               append(row, new_col, H);
                               // Sub-block c (c=1,2,3) = H[:, c-1] * lo^T (outer)
                               // Stored at (body*4+c, row) transposed = lo * H[:, c-1]^T
                               #pragma unroll
                               for(int c = 1; c < 4; ++c)
                               {
                                   gipc::Matrix3x3 B;  // = H[:, c-1] outer lo
                                   double H0 = H(0, c - 1), H1 = H(1, c - 1), H2 = H(2, c - 1);
                                   B(0, 0) = H0 * lo3.x; B(0, 1) = H0 * lo3.y; B(0, 2) = H0 * lo3.z;
                                   B(1, 0) = H1 * lo3.x; B(1, 1) = H1 * lo3.y; B(1, 2) = H1 * lo3.z;
                                   B(2, 0) = H2 * lo3.x; B(2, 1) = H2 * lo3.y; B(2, 2) = H2 * lo3.z;
                                   // B is original (row, body*4+c). Transposed = (body*4+c, row)
                                   int nc = body * 4 + c;
                                   if(nc <= row) append(nc, row, B.transpose());
                                   else          append(row, nc, B);
                               }
                           }
                           else  // both pinned
                           {
                               // The original triplet (p1, p2, H) with p1<p2
                               // represents H_{p1,p2}=H AND H_{p2,p1}=H^T (sym).
                               // After substitution x_p1=J_a*q_a, x_p2=J_b*q_b:
                               //   y_a += J_a^T * H * J_b * q_b   (path 1)
                               //   y_b += J_b^T * H^T * J_a * q_a (path 2, = transpose of path 1)
                               //
                               // For SAME body (body_a==body_b==body), both paths
                               // target body's diagonal:
                               //   total = J_a^T*H*J_b + (J_a^T*H*J_b)^T (symmetric)
                               // Sym storage upper-tri at body's (r,c) sub-block:
                               //   For r<=c: store M12.block(r,c) + M12.block(c,r)^T
                               //
                               // For DIFFERENT bodies (body_a < body_b), the paths
                               // target distinct off-diagonal block (body_a, body_b)
                               // with M12 stored once; sym SpMV via M^T handles the
                               // implicit (body_b, body_a) direction.
                               int     body_a = pin_body[pin_a];
                               int     body_b = pin_body[pin_b];
                               double3 lo_a   = pin_lo[pin_a];
                               double3 lo_b   = pin_lo[pin_b];
                               gipc::Vector3   xa{lo_a.x, lo_a.y, lo_a.z};
                               gipc::Vector3   xb{lo_b.x, lo_b.y, lo_b.z};
                               gipc::ABDJacobi   Ja(xa), Jb(xb);
                               gipc::Matrix12x12 M12 =
                                   gipc::ABDJacobi::JT_H_J(Ja.T(), H, Jb);
                               if(body_a == body_b)
                               {
                                   #pragma unroll
                                   for(int r = 0; r < 4; ++r)
                                   {
                                       #pragma unroll
                                       for(int c = r; c < 4; ++c)
                                       {
                                           gipc::Matrix3x3 B =
                                               M12.block<3, 3>(r * 3, c * 3)
                                               + M12.block<3, 3>(c * 3, r * 3).transpose();
                                           append(body_a * 4 + r, body_a * 4 + c, B);
                                       }
                                   }
                               }
                               else
                               {
                                   // body_a != body_b: store M12 (no symmetrization)
                                   // at (body_a*4+r, body_b*4+c) for body_a < body_b
                                   // (so always upper-tri); transpose if reversed.
                                   bool a_lt_b = (body_a < body_b);
                                   #pragma unroll
                                   for(int r = 0; r < 4; ++r)
                                   {
                                       #pragma unroll
                                       for(int c = 0; c < 4; ++c)
                                       {
                                           gipc::Matrix3x3 B =
                                               M12.block<3, 3>(r * 3, c * 3);
                                           if(a_lt_b)
                                               append(body_a * 4 + r, body_b * 4 + c, B);
                                           else
                                               append(body_b * 4 + c, body_a * 4 + r, B.transpose());
                                       }
                                   }
                               }
                           }
                       });

            // Read final extension count and bump triplet offset.
            int h_ext_count = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&h_ext_count,
                                      gipc_global_triplet.d_assembly_scratch_count,
                                      sizeof(int),
                                      cudaMemcpyDeviceToHost));
            if(h_ext_count > ext_capacity)
            {
                if(g_gipc_log_level >= 1) printf("[M3.5] WARN ext_count=%d > capacity=%d (truncated; expect "
                       "Newton instability)\n", h_ext_count, ext_capacity);
                h_ext_count = ext_capacity;
            }
            gipc_global_triplet.global_triplet_offset += h_ext_count;
        }
        else
        {
            // No pins: use the original simple zeroing logic for
            // non-zero BoundaryType (e.g. user-defined boundary conditions).
            muda::ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(fem_triplet_num,
                       [cfem_rows = gipc_global_triplet.block_row_indices(fem_triplet_start),
                        cfem_cols = gipc_global_triplet.block_col_indices(fem_triplet_start),
                        triplet_fem = gipc_global_triplet.block_values(fem_triplet_start),
                        BDType   = TetMesh.BoundaryType,
                        hess_index2fem_index = fem_global_hessian_index_offset] __device__(int i) mutable
                       {
                           int row    = cfem_rows[i];
                           int col    = cfem_cols[i];
                           int btypeA = BDType[row - hess_index2fem_index];
                           int btypeB = BDType[col - hess_index2fem_index];
                           if(btypeA != 0 || btypeB != 0)
                           {
                               triplet_fem[i].setZero();
                           }
                       });
        }


        //int massNum =
        muda::ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(abd_fem_count_info.fem_point_num,
                   [mass      = TetMesh.masses,
                    cfem_rows = gipc_global_triplet.block_row_indices(
                        gipc_global_triplet.global_triplet_offset),
                    cfem_cols = gipc_global_triplet.block_col_indices(
                        gipc_global_triplet.global_triplet_offset),
                    triplet_fem = gipc_global_triplet.block_values(
                        gipc_global_triplet.global_triplet_offset),
                    fem_global_hessian_index_offset,
                    fem_pint_start = abd_fem_count_info.abd_point_num,
                    abd_num = abd_fem_count_info.abd_body_num] __device__(int i) mutable
                   {
                       triplet_fem[i] =
                           mass[i + fem_pint_start] * gipc::Matrix3x3::Identity();
                       cfem_rows[i] = i + abd_num * 4;
                       cfem_cols[i] = i + abd_num * 4;
                   });
        gipc_global_triplet.global_triplet_offset += abd_fem_count_info.fem_point_num;

        //cudaMemcpy(TetMesh.totalForce, contact_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice);
        //getTotalForce(shape_grads, TetMesh.totalForce);
    }

    // [M2 substitution method] Chain-rule pinned FEM vertex gradient to ABD body q-DOFs.
    // For each pinned vertex p (body b, rest-frame local_pos lo):
    //   ABD gradient += J_p^T * (shape_grads[p] + fb[p])
    // where J_p^T * g = [g; lo.x*g; lo.y*g; lo.z*g]  (from ABDJacobiT operator*).
    // FEMLinearSubsystem::assemble() already zeros pinned DOFs in the PCG RHS via
    // BoundaryType check, so no double-counting occurs.
    if(TetMesh.n_fem_pins > 0 && m_abd_system && m_d_abd_body_q != nullptr)
    {
        m_abd_system->couple_bin_open((int)m_abd_system->system_gradient.size());  // [4.3] bin the coupling
        muda::ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(TetMesh.n_fem_pins,
                   [sys_grad    = m_abd_system->system_gradient.viewer(),
                    shape_grads = TetMesh.shape_grads,
                    fb          = TetMesh.fb,
                    pin_fem_v   = TetMesh.d_fem_pin_fem_vertex,
                    pin_body_id = TetMesh.d_fem_pin_abd_body_id,
                    pin_lo      = TetMesh.d_fem_pin_abd_local_pos] __device__(int i) mutable
                   {
                       int     fem_v   = pin_fem_v[i];
                       int     body_id = pin_body_id[i];
                       double3 lo      = pin_lo[i];

                       double gx = shape_grads[fem_v].x + fb[fem_v].x;
                       double gy = shape_grads[fem_v].y + fb[fem_v].y;
                       double gz = shape_grads[fem_v].z + fb[fem_v].z;

                       // J_p^T * [gx, gy, gz]:
                       // segment [0:3]  = g
                       // segment [3:6]  = lo * gx
                       // segment [6:9]  = lo * gy
                       // segment [9:12] = lo * gz
                       gipc::Vector12 g12;
                       g12(0)  = gx;        g12(1)  = gy;        g12(2)  = gz;
                       g12(3)  = lo.x * gx; g12(4)  = lo.y * gx; g12(5)  = lo.z * gx;
                       g12(6)  = lo.x * gy; g12(7)  = lo.y * gy; g12(8)  = lo.z * gy;
                       g12(9)  = lo.x * gz; g12(10) = lo.y * gz; g12(11) = lo.z * gz;

                       for(int c = 0; c < 12; ++c)
                           _binDepBase(g_abd_sysbin + ((size_t)body_id * 12 + c) * BINNED_K, g12(c));
                   });
        m_abd_system->couple_bin_close((int)m_abd_system->system_gradient.size());  // [4.3] combine
    }

    // [M3 substitution method] Add J^T * (m * I) * J to the global Hessian at
    // the pinned ABD body's diagonal block.  Writes the UPPER TRIANGLE only
    // (10 triplets per pin), matching write_abd_body_hessian's storage
    // convention.  The CSR converter sums duplicates with the ABD body's
    // own 10 triplets at the same (i,j) positions.
    //
    // KNOWN LIMITATION: only the inertia term (mass*I) is chain-ruled.
    // The FEM elasticity Hessian's cross-terms H_fp * J_p (free-free row,
    // ABD col) and their transposes are NOT chain-ruled — the BoundaryType
    // zeroing at line 10644-10655 drops them.  Without these cross-terms,
    // PCG decouples FEM and ABD: ABD moves q ignoring elasticity pull-back
    // from the free FEM vertices, and Newton fails to converge under
    // joint-driven motion (k=1000 cap hit, verified via m5_drive_joint2_test).
    //
    // For static gripper-close scenarios M2 + M3 inertia is sufficient
    // (Newton k=1-2).  For dynamic joint motion the user should set
    // USE_HARD_PIN=0 (stitch spring) until full elasticity chain-rule is
    // implemented (TODO M3.5).
    if(TetMesh.n_fem_pins > 0)
    {
        int triplet_offset_start = gipc_global_triplet.global_triplet_offset;
        muda::ParallelFor(256)
            .file_line(__FILE__, __LINE__)
            .apply(TetMesh.n_fem_pins,
                   [tri_rows = gipc_global_triplet.block_row_indices(triplet_offset_start),
                    tri_cols = gipc_global_triplet.block_col_indices(triplet_offset_start),
                    tri_vals = gipc_global_triplet.block_values(triplet_offset_start),
                    pin_fem_v   = TetMesh.d_fem_pin_fem_vertex,
                    pin_body_id = TetMesh.d_fem_pin_abd_body_id,
                    pin_lo      = TetMesh.d_fem_pin_abd_local_pos,
                    masses      = TetMesh.masses] __device__(int i) mutable
                   {
                       int     fem_v   = pin_fem_v[i];
                       int     body_id = pin_body_id[i];
                       double3 lo3     = pin_lo[i];
                       double  m       = masses[fem_v];

                       gipc::Vector3   lo{lo3.x, lo3.y, lo3.z};
                       gipc::Matrix3x3 mI = m * gipc::Matrix3x3::Identity();
                       gipc::ABDJacobi   J(lo);
                       gipc::Matrix12x12 H =
                           gipc::ABDJacobi::JT_H_J(J.T(), mI, J);

                       int slot_base = i * 10;
                       int kk = 0;
                       #pragma unroll
                       for(int r = 0; r < 4; ++r)
                       {
                           #pragma unroll
                           for(int c = r; c < 4; ++c)
                           {
                               int slot           = slot_base + kk;
                               tri_rows[slot]     = body_id * 4 + r;
                               tri_cols[slot]     = body_id * 4 + c;
                               tri_vals[slot]     = H.block<3, 3>(r * 3, c * 3);
                               kk++;
                           }
                       }
                   });
        gipc_global_triplet.global_triplet_offset += TetMesh.n_fem_pins * 10;
    }

    // [3c] every triplet slot in [0, global_triplet_offset) must have been
    // written by exactly this pass; survivors throw, tail restored to zeros.
    gipc_global_triplet.slot_audit_check_and_restore("computeGradientAndHessian");

    return time00;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

