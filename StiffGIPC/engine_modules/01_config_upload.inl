void SimEngine::Impl::apply_config_to_ipc()
{
    ipc.density             = cfg.density;
    ipc.PoissonRate         = cfg.poisson_rate;
    ipc.frictionRate        = cfg.friction_rate;
    ipc.gd_frictionRate     = cfg.gd_friction_rate;
    ipc.clothThickness      = cfg.cloth_thickness;
    ipc.clothYoungModulus   = cfg.cloth_young_modulus;
    ipc.bendYoungModulus    = cfg.bend_young_modulus;
    ipc.clothDensity        = cfg.cloth_density;
    ipc.strainRate          = cfg.strain_rate;
    ipc.softMotionRate      = cfg.soft_motion_rate;
    ipc.IPC_dt              = cfg.dt;
    ipc.gravity             = make_double3(cfg.gravity.x(), cfg.gravity.y(), cfg.gravity.z());
    ipc.ground_normal_cfg   = make_double3(cfg.ground_normal.x(), cfg.ground_normal.y(), cfg.ground_normal.z());
    ipc.ground_offset_cfg   = cfg.ground_offset;
    ipc.pcg_threshold       = cfg.pcg_tol;
    ipc.Newton_solver_threshold = cfg.newton_tol;
    ipc.newton_velocity_tol     = cfg.newton_velocity_tol;   // [uipc-style opt-in]
    ipc.relative_dhat       = cfg.relative_dhat;
    ipc.absolute_dhat       = cfg.absolute_dhat;
    ipc.absolute_epsv       = cfg.absolute_epsv;
    ipc.YoungModulus        = cfg.young_modulus;
    ipc.pcg_data.P_type     = cfg.preconditioner_type;
    ipc.assets_dir_cfg      = resolved_assets_dir;

    ipc.semi_implicit_enabled  = cfg.semi_implicit_enabled;
    ipc.semi_implicit_beta_tol = cfg.semi_implicit_beta_tol;
    ipc.semi_implicit_min_iter = cfg.semi_implicit_min_iter;
    ipc.newton_iter_cap        = cfg.newton_iter_cap;
    ipc.env_newton_iter_cap    = cfg.env_newton_iter_cap;  // [per-env productization]
    ipc.line_search_max_iter   = cfg.line_search_max_iter;  // [T1]
    ipc.energy_abs_tol         = cfg.energy_abs_tol;
    ipc.energy_rel_tol         = cfg.energy_rel_tol;

    ipc.m_skip_all_collision = cfg.skip_all_collision;
}

// ---------- FEM initialization (from gl_main.cu::initFEM) ----------
void SimEngine::Impl::do_initFEM()
{
    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.bendStiff    = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                       / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    double massSum   = 0;
    double volumeSum = 0;

    for(int i = 0; i < tetMesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(tetMesh.vertexes.data(), tetMesh.tetrahedras[i], DM);
        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);
        double vlm = calculateVolum(tetMesh.vertexes.data(), tetMesh.tetrahedras[i]);

        // [per-body density] per-tet override (<= 0 / absent = global density)
        const double tet_rho = (i < (int)tetMesh.tet_densities.size()
                                && tetMesh.tet_densities[i] > 0.0)
                                   ? tetMesh.tet_densities[i]
                                   : ipc.density;
        tetMesh.masses[tetMesh.tetrahedras[i].x] += vlm * tet_rho / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].y] += vlm * tet_rho / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].z] += vlm * tet_rho / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].w] += vlm * tet_rho / 4;

        massSum += vlm * tet_rho;
        volumeSum += vlm;
        tetMesh.DM_inverse.push_back(DM_inverse);
        tetMesh.volum.push_back(vlm);

        double lrl = tetMesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double vrl = tetMesh.vert_youngth_modules[i] * ipc.PoissonRate
                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        tetMesh.lengthRate.push_back(4 * lrl / 3);
        tetMesh.volumeRate.push_back(vrl + 5 * lrl / 6);
    }

    for(size_t i = 0; i < tetMesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(tetMesh.vertexes.data(), tetMesh.triangles[i], DM);
        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);
        double area = calculateArea(tetMesh.vertexes.data(), tetMesh.triangles[i]);
        area *= ipc.clothThickness;
        tetMesh.area.push_back(area);

        // [per-body density] per-triangle override (<= 0 / absent = global cloth_density)
        const double tri_rho = (i < tetMesh.tri_densities.size()
                                && tetMesh.tri_densities[i] > 0.0)
                                   ? tetMesh.tri_densities[i]
                                   : ipc.clothDensity;
        tetMesh.masses[tetMesh.triangles[i].x] += tri_rho * area / 3;
        tetMesh.masses[tetMesh.triangles[i].y] += tri_rho * area / 3;
        tetMesh.masses[tetMesh.triangles[i].z] += tri_rho * area / 3;

        massSum += area * tri_rho;
        volumeSum += area;
        tetMesh.tri_DM_inverse.push_back(DM_inverse);
    }

    tetMesh.meanMass  = massSum / tetMesh.vertexNum;
    tetMesh.meanVolum = volumeSum / tetMesh.vertexNum;
    // [batch-size determinism] meanMass sets κ's scale (suggestKappa ∝ meanMass). The global massSum
    // is a serial FP sum over ALL envs' primitives → varies at ~1e-14 with the env COUNT → seeds κ →
    // chaos-amplifies → env_0 not batch-SIZE invariant. Under STIFF_DECOUPLE_THRESH recompute meanMass
    // (and meanVolum) as the INTENSIVE per-vertex mean of ONE env (all envs identical, env-major
    // contiguous layout) → N-invariant by construction. Falls back to the global mean if ungrouped.
    if(ipc.m_mode_config.decouple_thresh && !tetMesh.body_groups.empty()
       && (int)tetMesh.point_id_to_body_id.size() == tetMesh.vertexNum)
    {
        // env_0's verts = { v : body_groups[point_id_to_body_id[v]] == 0 }. The vertex layout is NOT
        // env-major contiguous (bodies grouped by type, not env), so we must select by the group map,
        // not by slicing the first block. env_0's per-vertex-mass mean is the intensive κ scale,
        // identical regardless of env count → batch-SIZE invariant.
        double m0 = 0.0;
        long   c0 = 0;
        for(int v = 0; v < tetMesh.vertexNum; v++)
        {
            int b = tetMesh.point_id_to_body_id[v];
            int g = (b >= 0 && b < (int)tetMesh.body_groups.size()) ? tetMesh.body_groups[b] : -1;
            if(g == 0) { m0 += tetMesh.masses[v]; c0++; }
        }
        if(c0 > 0)
        {
            double new_mean = m0 / (double)c0;
            printf("[batch-inv] meanMass env0(group0, %ld verts)=%.17g (global was %.17g)\n",
                   c0, new_mean, tetMesh.meanMass);
            tetMesh.meanMass = new_mean;
        }
    }
}

// ---------- MAS partition (from gl_main.cu::setMAS_partition) ----------
void SimEngine::Impl::do_setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(size_t i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = static_cast<int>(i);
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
                index = 0;
        }
    }
    index = 0;
    for(size_t i = 0; i < tetMesh.partId_map_real.size(); i++)
    {
        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = static_cast<int>(i);
            index++;
        }
    }
}

// ---------- Upload all host data to GPU ----------
void SimEngine::Impl::do_upload_to_gpu()
{
    cudaSetDevice(cfg.cuda_device);

    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                static_cast<int>(tetMesh.tri_edges.size()),
                                tetMesh.abd_fem_count_info.total_body_num());

    auto safe_copy = [](void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
        if(bytes > 0)
            CUDA_SAFE_CALL(cudaMemcpy(dst, src, bytes, kind));
    };

    safe_copy(d_tetMesh.masses, tetMesh.masses.data(),
              tetMesh.vertexNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.apply_gravity, tetMesh.apply_gravity.data(),
              tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.lengthRate, tetMesh.lengthRate.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.volumeRate, tetMesh.volumeRate.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.volum, tetMesh.volum.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.vertexes, tetMesh.vertexes.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.o_vertexes, tetMesh.vertexes.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tetrahedras, tetMesh.tetrahedras.data(),
              tetMesh.tetrahedraNum * sizeof(uint4), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.DmInverses, tetMesh.DM_inverse.data(),
              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.BoundaryType, tetMesh.boundaryTypies.data(),
              tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.velocities, tetMesh.velocities.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);

    // [per-body friction] expand pending per-body mu overrides into per-vertex
    // device tables. Values are uniform within each body's vertex range, so the
    // within-body metis sort (see stitch note below) cannot misroute them.
    // Nothing pending -> tables stay nullptr -> every friction kernel takes its
    // legacy scalar path (bit-identical to the feature-less build).
    if(!pending_body_mu.empty())
    {
        std::vector<double> h_mu(tetMesh.vertexNum, cfg.friction_rate);
        std::vector<double> h_mu_gd(tetMesh.vertexNum, cfg.gd_friction_rate);
        for(const auto& kv : pending_body_mu)
        {
            const auto& r     = load_records[kv.first];
            const int   v_end = std::min(r.vertex_offset + r.vertex_count,
                                         (int)tetMesh.vertexNum);
            for(int v = r.vertex_offset; v < v_end; ++v)
            {
                h_mu[v] = kv.second.first;
                if(kv.second.second >= 0.0)
                    h_mu_gd[v] = kv.second.second;
            }
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&ipc.d_vert_mu,
                                  tetMesh.vertexNum * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&ipc.d_vert_mu_gd,
                                  tetMesh.vertexNum * sizeof(double)));
        safe_copy(ipc.d_vert_mu, h_mu.data(),
                  tetMesh.vertexNum * sizeof(double), cudaMemcpyHostToDevice);
        safe_copy(ipc.d_vert_mu_gd, h_mu_gd.data(),
                  tetMesh.vertexNum * sizeof(double), cudaMemcpyHostToDevice);
        if(g_gipc_log_level >= 1)
            printf("[per-body-friction] %zu bodies overridden (defaults mu=%.3g gd=%.3g)\n",
                   pending_body_mu.size(), cfg.friction_rate, cfg.gd_friction_rate);
    }
    // [MAS stitch index fix] When MAS is active (preconditioner_type != 0), FEM
    // bodies are loaded in metis-SORTED order: engine vertex (off+i) holds INPUT
    // vertex (off + sort_index[i]); vertex_metis_to_input[engine] = input.
    // Stitch springs are added by the user in INPUT-vertex order, so without
    // this translation they pull the WRONG engine vertices -> garbage forces ->
    // Newton never converges (the case_40 MAS-on bug). P_type==0 => perm is
    // identity => skipped (no-op).
    if(cfg.preconditioner_type != 0 && tetMesh.softNum > 0
       && !tetMesh.vertex_metis_to_input.empty())
    {
        const auto& m2i = tetMesh.vertex_metis_to_input;  // engine_idx -> input_idx
        std::vector<int> i2m(m2i.size(), -1);             // input_idx -> engine_idx
        for(int e = 0; e < (int)m2i.size(); e++)
            if(m2i[e] >= 0 && m2i[e] < (int)i2m.size())
                i2m[m2i[e]] = e;
        auto to_engine = [&](int input_id) -> int {
            return (input_id >= 0 && input_id < (int)i2m.size() && i2m[input_id] >= 0)
                       ? i2m[input_id] : input_id;
        };
        for(auto& v : tetMesh.targetIndex)
            v = static_cast<uint32_t>(to_engine(static_cast<int>(v)));
        for(auto& v : d_tetMesh.stitch_paired_vertex)  // ABD anchors: identity, safe
            v = to_engine(v);
        if(g_gipc_log_level >= 1)
            printf("[MAS-fix] remapped %d stitch FEM indices input->engine (metis) order\n",
                   tetMesh.softNum);
    }
    safe_copy(d_tetMesh.targetIndex, tetMesh.targetIndex.data(),
              tetMesh.softNum * sizeof(uint32_t), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.targetVert, tetMesh.targetPos.data(),
              tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice);

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    safe_copy(d_tetMesh.triDmInverses, tetMesh.tri_DM_inverse.data(),
              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.area, tetMesh.area.data(),
              tetMesh.triangleNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.triangles, tetMesh.triangles.data(),
              tetMesh.triangleNum * sizeof(uint3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tri_edges, tetMesh.tri_edges.data(),
              tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tri_edge_adj_vertex, tetMesh.tri_edges_adj_points.data(),
              tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice);

    safe_copy(d_tetMesh.body_id_to_boundary_type, tetMesh.body_id_to_is_fixed.data(),
              tetMesh.body_id_to_is_fixed.size() * sizeof(int), cudaMemcpyHostToDevice);
    // [multi-FEM-bodyid] upload per-body FEM flag (size = collision_body_num).
    if(d_tetMesh.body_id_to_is_fem != nullptr && !tetMesh.body_id_to_is_fem.empty())
    {
        safe_copy(d_tetMesh.body_id_to_is_fem, tetMesh.body_id_to_is_fem.data(),
                  tetMesh.body_id_to_is_fem.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    safe_copy(d_tetMesh.point_id_to_body_id, tetMesh.point_id_to_body_id.data(),
              tetMesh.point_id_to_body_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tet_id_to_body_id, tetMesh.tet_id_to_body_id.data(),
              tetMesh.tet_id_to_body_id.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Motor params
    if(!tetMesh.body_motor_infos.empty())
    {
        int bodyNum = static_cast<int>(tetMesh.body_motor_infos.size());
        std::vector<double> motor_params(bodyNum * 5, 0.0);
        for(int i = 0; i < bodyNum; i++)
        {
            auto& mi                = tetMesh.body_motor_infos[i];
            motor_params[i * 5 + 0] = mi.axis_x;
            motor_params[i * 5 + 1] = mi.axis_y;
            motor_params[i * 5 + 2] = mi.axis_z;
            motor_params[i * 5 + 3] = mi.speed;
            motor_params[i * 5 + 4] = mi.strength;
        }
        safe_copy(d_tetMesh.body_motor_params, motor_params.data(),
                  bodyNum * 5 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Collision exclusion matrix
    if(::g_gipc_log_level >= 1) printf("[CollisionExclusion] pairs=%d, collision_body_num=%d\n",
           (int)tetMesh.collision_exclusion_pairs.size(), d_tetMesh.collision_body_num);
    const bool groups_declared = !tetMesh.body_groups.empty();
    bool       have_groups     = false;
    int        active_group_count = 0;
    if(groups_declared)
    {
        const int body_count = d_tetMesh.collision_body_num;
        if((int)tetMesh.body_groups.size() != body_count)
            throw std::invalid_argument(
                "set_body_groups: expected exactly " + std::to_string(body_count)
                + " group ids (one per collision body), got "
                + std::to_string(tetMesh.body_groups.size()));

        std::vector<int> seen(device_TetraData::kGroupSlotCapacity, 0);
        bool has_wildcard = false;
        for(int body = 0; body < body_count; ++body)
        {
            int group = tetMesh.body_groups[body];
            if(group < -1 || group >= device_TetraData::kGroupSlotCapacity)
                throw std::invalid_argument(
                    "set_body_groups: group id for body " + std::to_string(body)
                    + " must be -1 or in [0, "
                    + std::to_string(device_TetraData::kGroupSlotCapacity)
                    + "), got " + std::to_string(group));
            if(group < 0)
            {
                has_wildcard = true;
                continue;
            }
            seen[group] = 1;
            active_group_count = std::max(active_group_count, group + 1);
        }
        for(int group = 0; group < active_group_count; ++group)
            if(!seen[group])
                throw std::invalid_argument(
                    "set_body_groups: active group ids must be dense [0, N); missing group "
                    + std::to_string(group));

        const bool isolated_features = ipc.m_mode_config.perenv_bvh
                                    || ipc.m_mode_config.perenv_alpha
                                    || ipc.m_mode_config.pergroup_kappa
                                    || ipc.m_mode_config.segmented_pcg
                                    || ipc.m_mode_config.decouple_thresh;
        if(isolated_features && (active_group_count == 0 || has_wildcard))
            throw std::invalid_argument(
                "isolated/strict mode requires every collision body to have a non-negative "
                "dense group id; wildcard group -1 is only supported by merged mode");
        have_groups = active_group_count > 0;
    }
    d_tetMesh.h_group_count = active_group_count;
    ipc.m_active_group_count = active_group_count;
    if((!tetMesh.collision_exclusion_pairs.empty() || have_groups)
       && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_matrix(N * N, 0);
        for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
        {
            if(a >= 0 && a < N && b >= 0 && b < N)
            {
                host_matrix[a * N + b] = 1;
                host_matrix[b * N + a] = 1;
            }
        }
        // [multi-env] Exclude all cross-group body pairs (both groups >= 0).
        // Guarantees envs never interact regardless of spatial proximity; the
        // existing _is_collision_excluded reads this matrix in every narrow-phase
        // pair-build path, so cross-env candidates are dropped before buffering.
        int n_xgrp = 0;
        if(have_groups)
        {
            const auto& grp = tetMesh.body_groups;
            for(int i = 0; i < N; ++i)
            {
                int gi = (i < (int)grp.size()) ? grp[i] : -1;
                if(gi < 0) continue;
                for(int j = i + 1; j < N; ++j)
                {
                    int gj = (j < (int)grp.size()) ? grp[j] : -1;
                    if(gj < 0 || gj == gi) continue;
                    host_matrix[i * N + j] = 1;
                    host_matrix[j * N + i] = 1;
                    ++n_xgrp;
                }
            }
        }
        safe_copy(d_tetMesh.collision_skip_matrix, host_matrix.data(),
                  N * N * sizeof(int), cudaMemcpyHostToDevice);
        if(::g_gipc_log_level >= 1) printf("[CollisionExclusion] Uploaded %dx%d exclusion matrix (%d pairs + %d cross-group)\n",
               N, N, (int)tetMesh.collision_exclusion_pairs.size(), n_xgrp);
    }

    // [multi-env P2a] upload the group-id substrate (d_body_to_group +
    // d_point_to_group). Foundation for per-group contact segmentation (P2b) and
    // the per-env block-diagonal solve (P3). Default stays -1 (wildcard) when no
    // groups set -> single-env behaviour unchanged.
    if(have_groups && d_tetMesh.collision_body_num > 0)
    {
        d_tetMesh.h_groups_present = true;   // [N=1 guard] per-env machinery master key
        int N = d_tetMesh.collision_body_num;
        std::vector<int> bg(N, -1);
        for(int i = 0; i < N && i < (int)tetMesh.body_groups.size(); ++i)
            bg[i] = tetMesh.body_groups[i];
        safe_copy(d_tetMesh.d_body_to_group, bg.data(),
                  N * sizeof(int), cudaMemcpyHostToDevice);
        const auto& pt2body = tetMesh.point_id_to_body_id;
        std::vector<int> pg(pt2body.size(), -1);
        std::map<int,int> grp_vcount;
        for(size_t v = 0; v < pt2body.size(); ++v)
        {
            int b = pt2body[v];
            int g = (b >= 0 && b < N) ? bg[b] : -1;
            pg[v] = g; grp_vcount[g]++;
        }
        safe_copy(d_tetMesh.d_point_to_group, pg.data(),
                  pg.size() * sizeof(int), cudaMemcpyHostToDevice);
        if(::g_gipc_log_level >= 1)
        {
            printf("[multi-env P2a] group substrate uploaded: %d bodies, %zu verts; per-group vert counts:",
                   N, pt2body.size());
            for(auto& [g, c] : grp_vcount) printf(" g%d=%d", g, c);
            printf("\n");
        }
        // [multi-env P2a] block-level group map (the per-env PCG-reduction key).
        // Block layout: [0, abd_body_num*4) ABD (block b -> body b/4); then
        // fem_point_num FEM blocks (block abd_dofs+j -> vertex fem_point_offset+j).
        const auto& ci = tetMesh.abd_fem_count_info;
        int abd_dofs = (int)ci.abd_body_num * 4;
        int nblk = abd_dofs + (int)ci.fem_point_num;
        std::vector<int> dg(nblk, -1);
        std::map<int,int> grp_bcount;
        for(int b = 0; b < abd_dofs; ++b)
        {
            int body = b / 4;
            int g = (body >= 0 && body < N) ? bg[body] : -1;
            dg[b] = g; grp_bcount[g]++;
        }
        for(int j = 0; j < (int)ci.fem_point_num; ++j)
        {
            int v = (int)ci.fem_point_offset + j;
            int g = (v >= 0 && v < (int)pg.size()) ? pg[v] : -1;
            dg[abd_dofs + j] = g; grp_bcount[g]++;
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_dof_to_group, nblk * sizeof(int)));
        safe_copy(d_tetMesh.d_dof_to_group, dg.data(), nblk * sizeof(int), cudaMemcpyHostToDevice);
        d_tetMesh.dof_block_count = nblk;
        if(::g_gipc_log_level >= 1)
        {
            printf("[multi-env P2a] dof_to_group: %d blocks (%d ABD + %d FEM); per-group block counts:",
                   nblk, abd_dofs, (int)ci.fem_point_num);
            for(auto& [g, c] : grp_bcount) printf(" g%d=%d", g, c);
            printf("\n");
        }
    }

    // Ground skip
    if(!tetMesh.ground_collision_skip_body_ids.empty() && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_flags(N, 0);
        for(int bid : tetMesh.ground_collision_skip_body_ids)
        {
            if(bid >= 0 && bid < N)
                host_flags[bid] = 1;
        }
        safe_copy(d_tetMesh.ground_skip_body, host_flags.data(),
                  N * sizeof(int), cudaMemcpyHostToDevice);
        ipc._ground_skip_body  = d_tetMesh.ground_skip_body;
        ipc._ground_body_count = N;
    }

    // BVH-skip optimization (audit/perf-bvh-skip-isolated): mark "isolated"
    // bodies via the diagonal of collision_skip_matrix. Body i is isolated iff
    // ground_skip_body[i]==1 AND all off-diagonal cells in row i are 1.
    // Kernels detect this with a single matrix lookup at thread entry.
    //
    // Toggle: BVHSKIP2=0 disables this (and #3, since #3 piggy-backs on the
    // same isolation flag). Use to measure #1-only baseline.
    const char* bvhskip2_env_local = std::getenv("BVHSKIP2");
    bool bvhskip2_enabled_local = (bvhskip2_env_local == nullptr) || (std::string(bvhskip2_env_local) != "0");
    if(!bvhskip2_enabled_local) {
        if(::g_gipc_log_level >= 1) printf("[BVHSkip#2] DISABLED via BVHSKIP2=0 (kernels see no isolated diag bits)\n");
    }
    if(bvhskip2_enabled_local && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> ground_flags(N, 0);
        for(int bid : tetMesh.ground_collision_skip_body_ids)
            if(bid >= 0 && bid < N) ground_flags[bid] = 1;
        std::vector<int> matrix(N * N, 0);
        for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
            if(a >= 0 && a < N && b >= 0 && b < N) {
                matrix[a*N+b] = 1; matrix[b*N+a] = 1;
            }
        int n_iso = 0;
        for(int i = 0; i < N; ++i) {
            if(!ground_flags[i]) continue;
            bool all_excluded = true;
            for(int j = 0; j < N; ++j) {
                if(i == j) continue;
                if(matrix[i*N+j] == 0) { all_excluded = false; break; }
            }
            if(all_excluded) {
                int one = 1;
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.collision_skip_matrix + i*N + i,
                                          &one, sizeof(int), cudaMemcpyHostToDevice));
                n_iso++;
            }
        }
        if(::g_gipc_log_level >= 1) printf("[BVHSkip] %d/%d bodies fully isolated → diag[i][i]=1 short-circuit set\n",
               n_iso, N);
    }

    // Stitch springs
    if(tetMesh.softNum > 0 && !d_tetMesh.stitch_paired_vertex.empty())
    {
        safe_copy(d_tetMesh.d_stitch_paired_vertex, d_tetMesh.stitch_paired_vertex.data(),
                  tetMesh.softNum * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_stitch_rest_offset, d_tetMesh.stitch_rest_offset.data(),
                  tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_stitch_abd_body_id, d_tetMesh.stitch_abd_body_id.data(),
                  tetMesh.softNum * sizeof(int), cudaMemcpyHostToDevice);
        ipc.m_d_stitch_paired_vertex = d_tetMesh.d_stitch_paired_vertex;
        ipc.m_d_stitch_rest_offset   = d_tetMesh.d_stitch_rest_offset;
        ipc.m_d_stitch_abd_body_id   = d_tetMesh.d_stitch_abd_body_id;
    }

    // [FEM-pin / M1 substitution] Hard-constraint pin arrays.
    // Each pin: FEM vertex idx + ABD body id + local position in ABD rest frame.
    // The local position is computed from the world-frame offset given to
    // add_fem_pin_to_abd, transformed back through R_finalize^{-1}.
    int n_pins = static_cast<int>(tetMesh.fem_pin_fem_vertex.size());
    if(n_pins > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_fem_vertex,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_body_id,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_local_pos,
                                  n_pins * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_anchor,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_rest_offset,
                                  n_pins * sizeof(double3)));
        d_tetMesh.n_fem_pins = n_pins;

        safe_copy(d_tetMesh.d_fem_pin_fem_vertex, tetMesh.fem_pin_fem_vertex.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_fem_pin_abd_body_id, tetMesh.fem_pin_abd_body_id.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        // (abd_local_pos transform is done later in SimEngine::finalize after ABD q init)
        safe_copy(d_tetMesh.d_fem_pin_abd_anchor, tetMesh.fem_pin_abd_anchor.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_fem_pin_rest_offset, tetMesh.fem_pin_rest_offset.data(),
                  n_pins * sizeof(double3), cudaMemcpyHostToDevice);
        printf("[FEM-pin] allocated %d hard-constraint pins (local_pos transform deferred to finalize)\n", n_pins);
    }
}

// ---------- BVH + solver init ----------
void SimEngine::Impl::do_init_bvh_and_solver()
{
    cudaSetDevice(cfg.cuda_device);
    cudaDeviceSynchronize();

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = static_cast<uint32_t>(tetMesh.surfVerts.size());
    ipc.surface_Num    = static_cast<uint32_t>(tetMesh.surface.size());
    ipc.edge_Num       = static_cast<uint32_t>(tetMesh.surfEdges.size());
    ipc.tri_edge_num   = static_cast<uint32_t>(tetMesh.tri_edges.size());

    if(ipc.m_skip_all_collision)
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM = 1;
        ipc.MAX_COLLITION_PAIRS_NUM     = 1;
    }
    else
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM =
            static_cast<int>(
                1 * cfg.collision_detection_buff_scale
                * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
                   * std::max((ipc.IPC_dt / 0.01), 2.0)));
        ipc.MAX_COLLITION_PAIRS_NUM =
            static_cast<int>(
                (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                * 3 * cfg.collision_detection_buff_scale);
    }

    if(::g_gipc_log_level >= 1) printf("[SimEngine] collision_detection_buff_scale=%.1f  MAX_CCD_PAIRS=%d  MAX_PAIRS=%d\n",
           cfg.collision_detection_buff_scale,
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    ipc.triangleNum = tetMesh.triangleNum;
    ipc.targetVert  = d_tetMesh.targetVert;
    ipc.targetInd   = d_tetMesh.targetIndex;
    ipc.softNum     = tetMesh.softNum;

    ipc.abd_fem_count_info    = tetMesh.abd_fem_count_info;
    ipc.num_joint_constraints = static_cast<int>(tetMesh.joint_constraints.size());

    std::cout << "[SimEngine] BVH init: verts=" << ipc.vertexNum
              << " surface=" << ipc.surface_Num
              << " edges=" << ipc.edge_Num
              << " surf_verts=" << ipc.surf_vertexNum
              << std::endl;
    std::cout.flush();

    ipc.MALLOC_DEVICE_MEM();

    if(ipc.surface_Num > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._faces, tetMesh.surface.data(),
                                  ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    if(ipc.edge_Num > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._edges, tetMesh.surfEdges.data(),
                                  ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    if(ipc.surf_vertexNum > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts, tetMesh.surfVerts.data(),
                                  ipc.surf_vertexNum * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // [multi-FEM-bodyid] hand the per-body FEM flag table to GIPC so the
    // narrow-phase / sanity-check kernels can distinguish FEM vs ABD
    // without the legacy "_bodyId == -1" sentinel. Must be set BEFORE
    // initBVH() so bvh_f/bvh_e read a non-null pointer.
    ipc._body_id_to_is_fem  = d_tetMesh.body_id_to_is_fem;
    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id,
                d_tetMesh.collision_skip_matrix, d_tetMesh.collision_body_num);
    ipc._point_body_id      = d_tetMesh.point_id_to_body_id;

    // BVH-skip #3 wiring is deferred until after ipc.init() — see "[BVHSkip#3-WIRE]" marker.

    // MAS preconditioner setup (must run even for pure-ABD scenes, matching gl_main.cu)
    if(ipc.pcg_data.P_type)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        // [per-env MAS] #envs = #body groups. The MAS aggregation uses this to keep each env's
        // clusters in BANKSIZE-aligned banks at every level (intra-env preconditioner). MUST be
        // set BEFORE initPreconditioner_Neighbor: the hierarchy depth (computeNumLevels) is
        // derived from the PER-ENV node count so it cannot vary with batch size N.
        //
        // HOMOGENEITY GUARD: the segmentation maps env e to warps [e*wpe,(e+1)*wpe), which is
        // only valid when the FEM vertices are env-major contiguous AND every env owns the same
        // vertex count ("warpNum % n_env == 0" alone is NOT sufficient — equal totals can hide
        // unequal groups). Heterogeneous/interleaved scenes fall back to the global hierarchy
        // (m_numEnvs=1, exact v0.8.4.1 topology). Cap 4096 = d_envBase/d_envStart scratch size.
        {
            int  ge   = std::max(1, d_tetMesh.h_group_count);
            bool homo = ge > 1 && ge <= 4096;
            if(homo)
            {
                const int nFem = ipc.vertexNum - tetMesh.abd_vertexOffset;
                if(nFem % ge != 0)
                    homo = false;
                else
                {
                    const int per = nFem / ge;
                    for(int v = 0; v < nFem && homo; v++)
                    {
                        const int gv = tetMesh.abd_vertexOffset + v;
                        const int b  = (gv < (int)tetMesh.point_id_to_body_id.size())
                                           ? tetMesh.point_id_to_body_id[gv] : -1;
                        const int g  = (b >= 0 && b < (int)tetMesh.body_groups.size())
                                           ? tetMesh.body_groups[b] : -1;
                        if(g != v / per)   // env-major contiguity + equal size, one test
                            homo = false;
                    }
                }
                // BANK-level homogeneity: the segmentation unit is the MAS
                // bank (METIS partition), not the vertex. Equal vertex counts
                // do NOT imply equal partition counts (nPart follows graph
                // structure), so verify from the actual partId: every bank
                // single-group, group bank-ranges contiguous, equal banks per
                // group. Anything else -> global hierarchy.
                if(homo)
                {
                    if((int)tetMesh.partId.size() != nFem || tetMesh.part_offset <= 0)
                        homo = false;   // no/partial partition data: cannot prove bank layout
                    else
                    {
                        std::vector<int> bank_group(tetMesh.part_offset, -1);
                        for(int v = 0; v < nFem && homo; v++)
                        {
                            const int bk = (int)tetMesh.partId[v];
                            const int gv = tetMesh.abd_vertexOffset + v;
                            const int bd = tetMesh.point_id_to_body_id[gv];
                            const int g  = tetMesh.body_groups[bd];
                            if(bk < 0 || bk >= tetMesh.part_offset)
                                homo = false;
                            else if(bank_group[bk] == -1)
                                bank_group[bk] = g;
                            else if(bank_group[bk] != g)
                                homo = false;   // one bank spans two envs
                        }
                        if(homo && tetMesh.part_offset % ge != 0)
                            homo = false;       // unequal bank counts per env
                        if(homo)
                        {
                            const int bpe = tetMesh.part_offset / ge;
                            for(int bk = 0; bk < tetMesh.part_offset && homo; bk++)
                                if(bank_group[bk] != bk / bpe)
                                    homo = false;   // non-contiguous / unequal ranges
                        }
                    }
                }
            }
            ipc.pcg_data.MP.m_numEnvs = homo ? std::max(1, d_tetMesh.h_group_count) : 1;
            if(d_tetMesh.h_group_count > 1 && !homo)
                printf("[per-env MAS] heterogeneous/non-contiguous FEM groups -> global "
                       "MAS hierarchy (segmentation disabled)\n");
        }
        ipc.pcg_data.MP.initPreconditioner_Neighbor(
            ipc.vertexNum - tetMesh.abd_vertexOffset,
            tetMesh.abd_vertexOffset,
            neighborListSize,
            ipc._collisonPairs,
            tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;

        if(neighborListSize > 0)
        {
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                      tetMesh.neighborList.data(),
                                      neighborListSize * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                      tetMesh.neighborStart.data(),
                                      (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                      tetMesh.neighborNum.data(),
                                      (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
        }

        if(!tetMesh.partId_map_real.empty())
        {
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                      tetMesh.partId_map_real.data(),
                                      tetMesh.part_offset * BANKSIZE * sizeof(int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                      tetMesh.real_map_partId.data(),
                                      tetMesh.real_map_partId.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
        }

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    // Copy rest vertices
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

#ifdef USE_QUADRATIC_BENDING
    if(!tetMesh.tri_edges.empty())
    {
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),
                            tetMesh.tri_edges.size(),
                            Q_host.data());
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));
    }
#endif

    ipc.buildBVH();
    ipc.setup_surface_mesh_bodies(tetMesh);
    // [P1 triplet-margin] The internal-triplet margin (cfg.triplet_internal_margin,
    // default 32) only exists to reserve headroom for the M3.5 hybrid FEM-ABD pin
    // chain-rule Hessian expansion, which is gated by n_fem_pins>0 (see GIPC.cu M3.5
    // block). For scenes WITHOUT FEM pins the internal Hessian-triplet count is exact
    // (fixed by mesh topology), so margin=1 is sufficient and safe — and it avoids
    // over-allocating the global triplet buffer by up to 32x. Verified: cp-stats shows
    // "ext used 0 / cap 0" for non-hybrid scenes. Hybrid scenes keep the full margin.
    ipc.m_triplet_internal_margin =
        (d_tetMesh.n_fem_pins > 0) ? cfg.triplet_internal_margin : 1.0;
    // [env-scale convergence] per-env world-space bbox diag^2 (+ avg over active envs), once.
    // Derive it from the finalized device state so every import path uses the positions actually
    // consumed by the solver. This is the complete environment scale and need not equal the
    // contact BVH's bboxDiagSize2 (for example, a robot may extend beyond its active contact BVH).
    // Wildcard (-1) verts belong to no env and do not shape any env's scale.
    if(d_tetMesh.h_groups_present
       && (int)tetMesh.point_id_to_body_id.size() == (int)tetMesh.vertexes.size())
    {
        const int NG = d_tetMesh.h_group_count;
        std::vector<double3> lo(NG, make_double3(1e300, 1e300, 1e300));
        std::vector<double3> hi(NG, make_double3(-1e300, -1e300, -1e300));
        std::vector<int>     cnt(NG, 0);
        std::vector<double3> world_vertexes(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(world_vertexes.data(), d_tetMesh.vertexes,
                                  ipc.vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
        const auto& p2b = tetMesh.point_id_to_body_id;
        const auto& bg  = tetMesh.body_groups;
        for(size_t v = 0; v < tetMesh.vertexes.size(); ++v)
        {
            int b = p2b[v];
            int g = (b >= 0 && b < (int)bg.size()) ? bg[b] : -1;
            if(g < 0 || g >= NG) continue;
            const auto& P = world_vertexes[v];
            lo[g].x = std::min(lo[g].x, P.x); hi[g].x = std::max(hi[g].x, P.x);
            lo[g].y = std::min(lo[g].y, P.y); hi[g].y = std::max(hi[g].y, P.y);
            lo[g].z = std::min(lo[g].z, P.z); hi[g].z = std::max(hi[g].z, P.z);
            cnt[g]++;
        }
        ipc.h_env_bbox2.assign(NG, 0.0);
        double sum = 0.0; int nact = 0;
        for(int g = 0; g < NG; ++g)
        {
            if(cnt[g] <= 0) continue;
            double dx = hi[g].x - lo[g].x, dy = hi[g].y - lo[g].y, dz = hi[g].z - lo[g].z;
            ipc.h_env_bbox2[g] = dx * dx + dy * dy + dz * dz;
            sum += ipc.h_env_bbox2[g]; ++nact;
        }
        ipc.m_avg_env_bbox2 = (nact > 0) ? (sum / nact) : 0.0;
        if(!ipc.d_env_bbox2)
            CUDA_SAFE_CALL(cudaMalloc((void**)&ipc.d_env_bbox2, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.d_env_bbox2, ipc.h_env_bbox2.data(),
                                  NG * sizeof(double), cudaMemcpyHostToDevice));
        if(::g_gipc_log_level >= 1)
            printf("[env-scale] per-env bbox: active=%d avg_diag2=%.6g (env0=%.6g)\n",
                   nact, ipc.m_avg_env_bbox2, ipc.h_env_bbox2[0]);
    }
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer,
             cfg.linear_system_buff_scale);

    // [BVHSkip#3-WIRE] Wire _active_idx now that ipc.init() has captured the
    // full-scene bbox into bboxDiagSize2/dHat. Subsequent buildBVH() in step()
    // uses the indirect (filtered) path. (Wiring before ipc.init() shrinks the
    // scene bbox to active leaves only → dHat too small → cloth self-intersect.)
    //
    // Toggle: BVHSKIP3=0 disables (BVH still uses default path, only #1+#2 active).
    // Note: requires BVHSKIP2=1 — #3 reuses the isolation set from #2; if #2
    // is disabled there are no diag bits to read.
    {
        const char* bvhskip3_env = std::getenv("BVHSKIP3");
        bool bvhskip3_enabled = (bvhskip3_env == nullptr) || (std::string(bvhskip3_env) != "0");
        const char* bvhskip2_env = std::getenv("BVHSKIP2");
        bool bvhskip2_enabled = (bvhskip2_env == nullptr) || (std::string(bvhskip2_env) != "0");
        if(!bvhskip3_enabled) {
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] DISABLED via BVHSKIP3=0\n");
        } else if(!bvhskip2_enabled) {
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] AUTO-DISABLED (BVHSKIP2=0 — #3 requires #2's isolation set)\n");
        } else if(d_tetMesh.collision_body_num > 0
                  && (!tetMesh.collision_exclusion_pairs.empty()
                      || !tetMesh.ground_collision_skip_body_ids.empty()))
        {
            const int N = d_tetMesh.collision_body_num;
            std::vector<int> ground_flags(N, 0);
            for(int bid : tetMesh.ground_collision_skip_body_ids)
                if(bid >= 0 && bid < N) ground_flags[bid] = 1;
            std::vector<int> matrix(N * N, 0);
            for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
                if(a >= 0 && a < N && b >= 0 && b < N) {
                    matrix[a*N+b] = 1; matrix[b*N+a] = 1;
                }
            std::vector<int> isolated(N, 0);
            int n_iso = 0;
            for(int i = 0; i < N; ++i) {
                if(!ground_flags[i]) continue;
                bool all_excl = true;
                for(int j = 0; j < N; ++j) {
                    if(i == j) continue;
                    if(matrix[i*N+j] == 0) { all_excl = false; break; }
                }
                if(all_excl) { isolated[i] = 1; n_iso++; }
            }
            // Drop face only if all 3 vertices belong to an isolated body.
            // Drop edge only if both endpoints belong to an isolated body.
            // Conservative: cross-body or cloth-touching primitives stay active.
            auto is_iso = [&](int B) { return (B >= 0 && B < N && isolated[B] != 0); };

            std::vector<int> active_face;
            active_face.reserve(tetMesh.surface.size());
            for(int f = 0; f < (int)tetMesh.surface.size(); ++f) {
                const auto& t = tetMesh.surface[f];
                int Bx = tetMesh.point_id_to_body_id[t.x];
                int By = tetMesh.point_id_to_body_id[t.y];
                int Bz = tetMesh.point_id_to_body_id[t.z];
                if(is_iso(Bx) && is_iso(By) && is_iso(Bz)) continue;
                active_face.push_back(f);
            }
            std::vector<int> active_edge;
            active_edge.reserve(tetMesh.surfEdges.size());
            for(int e = 0; e < (int)tetMesh.surfEdges.size(); ++e) {
                const auto& ed = tetMesh.surfEdges[e];
                int Bx = tetMesh.point_id_to_body_id[ed.x];
                int By = tetMesh.point_id_to_body_id[ed.y];
                if(is_iso(Bx) && is_iso(By)) continue;
                active_edge.push_back(e);
            }
            const int n_af = (int)active_face.size();
            const int n_ae = (int)active_edge.size();
            if(n_af > 0 && n_af < (int)tetMesh.surface.size()) {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.bvh_active_face_idx,
                                          n_af * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.bvh_active_face_idx,
                                          active_face.data(),
                                          n_af * sizeof(int),
                                          cudaMemcpyHostToDevice));
                d_tetMesh.bvh_active_face_num = n_af;
                ipc.bvh_f._active_idx         = d_tetMesh.bvh_active_face_idx;
                ipc.bvh_f.face_number_active  = n_af;
            }
            if(n_ae > 0 && n_ae < (int)tetMesh.surfEdges.size()) {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.bvh_active_edge_idx,
                                          n_ae * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.bvh_active_edge_idx,
                                          active_edge.data(),
                                          n_ae * sizeof(int),
                                          cudaMemcpyHostToDevice));
                d_tetMesh.bvh_active_edge_num = n_ae;
                ipc.bvh_e._active_idx         = d_tetMesh.bvh_active_edge_idx;
                ipc.bvh_e.face_number_active  = n_ae;
            }
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] %d/%d isolated  active faces=%d/%zu  active edges=%d/%zu\n",
                   n_iso, N, n_af, tetMesh.surface.size(),
                   n_ae, tetMesh.surfEdges.size());
            // Re-build BVH so it transitions from default (full) to indirect
            // (active subset). Without this, the next buildCP() launches EE
            // self-query with N=active leaves but BVH leaves are still at the
            // default-path offset → reads stale internal nodes → illegal access.
            ipc.buildBVH();
        }
    }

    // Joint constraints
    if(!tetMesh.joint_constraints.empty() || !tetMesh.prismatic_constraints.empty())
        ipc.init_joint_constraints_from_mesh(tetMesh);

    // [env-det] enable env-major Morton on the merged BVH so co-located identical envs build
    // env-blocked (mirror) trees ⇒ env-symmetric broad-phase enumeration (the last bit-identity layer).
    if(ipc.m_mode_config.bvh_envdet && d_tetMesh.h_groups_present) ipc.enableEnvMajorBVH(d_tetMesh.d_point_to_group);  // [N=1 guard]
    // [multi-env P2] enable per-env BVH EAGERLY (before warm-start buildCP) so the warm-start uses
    // per-env LOCAL trees too — else the warm-start runs the merged path and (at spacing>0) injects
    // the offset-overlap divergence that all later frames inherit. Per-env trees use local verts ⇒
    // bit-identical at any spacing (render separation becomes a pure display offset).
    // [option A] no groups declared but per-env features requested -> they all
    // coherently no-op (exact merged behavior). Warn ONCE so the degradation is
    // never silent (user decision 2026-07-04: fallback + warning, not auto-group).
    if(!d_tetMesh.h_groups_present
       && (ipc.m_mode_config.perenv_bvh || ipc.m_mode_config.perenv_alpha
           || ipc.m_mode_config.pergroup_kappa || ipc.m_mode_config.segmented_pcg
           || ipc.m_mode_config.bvh_envdet || ipc.m_mode_config.decouple_thresh))
        printf("[multienv] WARNING: isolated/strict features requested but NO body groups declared "
               "(set_body_groups never called) — per-env machinery disabled, running merged-equivalent. "
               "Declare groups (all bodies -> 0 for a single env) to engage per-env paths.\n");

    // [N=1 guard] h_groups_present REQUIRED: with the all -1 wildcard table the
    // per-env index excludes EVERY primitive (active=0) -> ZERO self-collision
    // detection -> silently wrong physics (cloth through gripper).
    if(ipc.m_mode_config.perenv_bvh && d_tetMesh.d_point_to_group
       && d_tetMesh.h_groups_present)
    {
        ipc.m_perenv_bvh = true;
        ipc.m_d_p2g = d_tetMesh.d_point_to_group;
        ipc.m_active_group_count = d_tetMesh.h_group_count;
    }
    // Build collision pairs + solver warm-start (mirrors gl_main.cu post-init)
    ipc.buildCP();
    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0;
    ipc.computeXTilta(d_tetMesh, 1);
    ipc.create_LinearSystem(d_tetMesh);
}

// ======================== finalize ========================
