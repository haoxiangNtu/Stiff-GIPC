void SimEngine::finalize()
{
    auto& impl = *m_impl;
    if(impl.finalized)
        throw LifecycleError(
            "finalize() called on an already-finalized SimEngine; call "
            "reset() before building a new scene");
    if(impl.finalize_failed)
        throw LifecycleError(
            "a previous finalize() attempt failed after initialization began; "
            "call reset() before trying again");
    if(!impl.cuda_initialized)
        throw LifecycleError(
            "finalize() requires successful CUDA initialization");
    RuntimeOwnerAttempt runtime_attempt(
        impl.runtime_owner_lease, &impl, impl.finalize_failed);
    impl.apply_config_to_ipc();

    // Build ABD system + linear system
    impl.ipc.build_gipc_system(impl.d_tetMesh);

    // ABD system parms
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.joint_strength_ratio            = impl.cfg.joint_strength_ratio;
        impl.ipc.m_abd_system->parms.revolute_driving_strength_ratio = impl.cfg.revolute_driving_strength_ratio;
        impl.ipc.m_abd_system->parms.prismatic_strength_ratio        = impl.cfg.prismatic_strength_ratio;
        impl.ipc.m_abd_system->parms.prismatic_driving_strength_ratio = impl.cfg.prismatic_driving_strength_ratio;
        impl.ipc.m_abd_system->parms.max_revolute_step_per_frame     = impl.cfg.max_revolute_step_per_frame;
        impl.ipc.m_abd_system->parms.max_prismatic_step_per_frame    = impl.cfg.max_prismatic_step_per_frame;
        impl.ipc.m_abd_system->parms.dt = impl.cfg.dt;
        impl.ipc.m_abd_system->parms.gravity = impl.cfg.gravity;
        impl.ipc.m_abd_system->parms.velocity_damping = impl.cfg.velocity_damping;
    }

    // Prepare metis dir
    std::string metis_dir = impl.resolved_assets_dir + "sorted_mesh/";
    std::filesystem::create_directories(metis_dir);

    impl.do_setMAS_partition();
    impl.tetMesh.getSurface();
    impl.do_initFEM();
    impl.do_upload_to_gpu();
    impl.do_init_bvh_and_solver();

    // [stitch sanity] Warn if soft_motion_rate (stitch spring stiffness) is
    // large compared to FEM Young modulus and the scene has many stitch
    // springs. The stitch Hessian diagonal contribution is
    //   H_stitch_total ≈ stitch_count * soft_motion_rate
    // and gets summed with the FEM elasticity Hessian (~Young per tet) +
    // IPC barrier Hessian (~kappa per contact). If H_stitch dominates by
    // 10x or more, the combined matrix's condition number can blow up
    // and PCG produces NaN. (Verified empirically on case_27_softgripper:
    // motionRate=1e6 + 130 stitch + Young=1e6 -> NaN at step ~131.)
    {
        int stitch_count = impl.tetMesh.softNum;
        double rate = impl.cfg.soft_motion_rate;
        // Average per-vertex Young modulus across all FEM vertices — this
        // is the actual elasticity stiffness the stitch is competing with,
        // unlike cfg.cloth_young_modulus which only applies to dim=2 cloth
        // FEM bodies and may be unset for tet (dim=3) FEM scenes.
        double young_avg = 0.0;
        const auto& yvec = impl.tetMesh.vert_youngth_modules;
        if(!yvec.empty())
        {
            double sum = 0.0;
            for(double y : yvec) sum += y;
            young_avg = sum / static_cast<double>(yvec.size());
        }
        if(stitch_count > 0 && rate > 0.0 && young_avg > 0.0)
        {
            double total_stitch_h = stitch_count * rate;
            double ratio = total_stitch_h / young_avg;
            // Empirical thresholds (case_27_softgripper, 130 stitch, FEM
            // young 1e6):
            //   ratio = 130    (motionRate=1e4) -> stable
            //   ratio = 1.3e4  (motionRate=1e6) -> NaN at step ~131
            // Set threshold = 1000 (~middle in log scale).
            if(ratio > 1000.0)
            {
                printf("\n[SimEngine] *** WARNING: stitch system may be too stiff ***\n");
                printf("[SimEngine]   stitch_count=%d  soft_motion_rate=%.1e  avg FEM Young=%.1e\n",
                       stitch_count, rate, young_avg);
                printf("[SimEngine]   stitch_count * soft_motion_rate / avg_young = %.1f (threshold = 1000)\n",
                       ratio);
                printf("[SimEngine]   Empirically, ratio > 1000 risks PCG NaN under aggressive joint trajectories.\n");
                printf("[SimEngine]   If you hit NaN, try reducing soft_motion_rate (e.g. /100) or raising\n");
                printf("[SimEngine]   per-mesh young_modulus, or run with NAN_DIAG=1 to confirm.\n\n");
                fflush(stdout);
            }
        }
    }

    // DIAG: dump q for first 3 bodies after full finalize
    if(getenv("STIFF_ABD_DBG"))
    {
        int nb = impl.ipc.abd_fem_count_info.abd_body_num;
        int n = std::min(nb, 3);
        if(n > 0 && impl.ipc.m_abd_sim_data)
        {
            using Vec12 = Eigen::Matrix<double, 12, 1>;
            std::vector<Vec12> dbg_q(nb);
            CUDA_SAFE_CALL(cudaMemcpy(dbg_q.data(),
                                      impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                                      nb * sizeof(Vec12), cudaMemcpyDeviceToHost));
            for(int b = 0; b < n; b++)
            {
                auto& q = dbg_q[b];
                std::cout << "[DIAG-CPP] After finalize: body=" << b
                          << " p=[" << q[0] << "," << q[1] << "," << q[2] << "]"
                          << " a1=[" << q[3] << "," << q[4] << "," << q[5] << "]"
                          << " a2=[" << q[6] << "," << q[7] << "," << q[8] << "]"
                          << " a3=[" << q[9] << "," << q[10] << "," << q[11] << "]"
                          << std::endl;
            }
        }
    }

    // [stitch local-frame fix] DISABLED — see commits e0990e6 / pre-substitution-method.
    impl.ipc.m_d_abd_body_q = nullptr;

    // [M1 substitution method] FEM pin transform world→local.
    // After ABD q is initialized, transform pinned FEM vertices' world rest
    // offset into the ABD body's REST frame, store as local_pos. Each step
    // the kernel does world_pos = q.t + R(q) * local_pos.
    // Also wire the ABD q pointer for the apply-pins kernel.
    if(impl.ipc.m_abd_sim_data && impl.d_tetMesh.n_fem_pins > 0)
    {
        int n_pins = impl.d_tetMesh.n_fem_pins;
        int nb     = impl.ipc.abd_fem_count_info.abd_body_num;

        // Read q from GPU
        using Vec12 = Eigen::Matrix<double, 12, 1>;
        std::vector<Vec12> host_q(nb);
        CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                                  impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                                  nb * sizeof(Vec12), cudaMemcpyDeviceToHost));

        // Pull anchor + body_id arrays, compute local_pos
        const auto& fem_v_vec   = impl.tetMesh.fem_pin_fem_vertex;
        const auto& bid_vec     = impl.tetMesh.fem_pin_abd_body_id;
        const auto& anchor_vec  = impl.tetMesh.fem_pin_abd_anchor;

        std::vector<double3> local_pos(n_pins);
        std::vector<double3> host_verts(impl.tetMesh.vertexNum);
        cudaMemcpy(host_verts.data(), impl.d_tetMesh.vertexes,
                   impl.tetMesh.vertexNum * sizeof(double3), cudaMemcpyDeviceToHost);

        for(int i = 0; i < n_pins; i++)
        {
            int bid = bid_vec[i];
            int av  = anchor_vec[i];
            // [Hybrid mesh] anchor == -1 sentinel: caller used
            // add_fem_pins_with_local_pos and provided local_pos directly
            // (already in ABD rest frame).  Pass through verbatim.
            if(av == -1)
            {
                local_pos[i] = impl.tetMesh.fem_pin_abd_local_pos[i];
                continue;
            }
            double3 fem_world = host_verts[fem_v_vec[i]];
            // Compute fem's position in ABD body's rest frame:
            //   world = q.t + R(q) * fem_local
            //   fem_local = R(q)^T * (world - q.t)   (assuming R orthogonal)
            const Vec12& q = host_q[bid];
            double3 d = make_double3(fem_world.x - q[0], fem_world.y - q[1], fem_world.z - q[2]);
            double3 lo;
            lo.x = q[3] * d.x + q[4]  * d.y + q[5]  * d.z;  // R^T row 1 = a1 (q[3..5])
            lo.y = q[6] * d.x + q[7]  * d.y + q[8]  * d.z;
            lo.z = q[9] * d.x + q[10] * d.y + q[11] * d.z;
            local_pos[i] = lo;
        }
        // Upload local_pos to GPU
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.d_fem_pin_abd_local_pos,
                                  local_pos.data(),
                                  n_pins * sizeof(double3), cudaMemcpyHostToDevice));

        // Wire ABD q pointer to GIPC for apply_fem_pins kernel
        impl.ipc.m_d_abd_body_q = reinterpret_cast<void*>(
            impl.ipc.m_abd_sim_data->device.body_id_to_q.data());

        // Mark pinned FEM vertices' BoundaryType = Fixed (=2). step_forward
        // kernel uses btype=0 check to update positions from PCG Δx; setting
        // it to 2 means the line-search position update is skipped, which
        // is what we want — apply_fem_pins kernel will write the correct
        // ABD-derived position right after step_forward.
        //
        // **NOT changing mass** — earlier we tried mass=1e30 to make PCG
        // naturally output Δx_pinned ≈ 0, but that made inertia energy
        // E_kin = ½ m v² explode (1e30 × 5mm² = 1e23) and broke line search.
        // The fix is at the IPC matrix level (M2 below): when assembling
        // the FEM elasticity / barrier / inertia Hessian, skip the
        // pinned vertex's row/col entirely so PCG sees them as
        // disconnected DOFs.
        std::vector<int> btype_host(impl.tetMesh.vertexNum);
        cudaMemcpy(btype_host.data(), impl.d_tetMesh.BoundaryType,
                   impl.tetMesh.vertexNum * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < n_pins; i++)
        {
            int v = fem_v_vec[i];
            btype_host[v] = 2;            // Fixed; PCG Δx update skipped
        }
        cudaMemcpy(impl.d_tetMesh.BoundaryType, btype_host.data(),
                   impl.tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);

        // Build per-vertex pin map for O(1) lookup in elasticity kernels:
        // is_pinned_vertex[v] = 1 if v is a pinned FEM vertex, 0 otherwise.
        // Used in M2 to skip writing pinned row/col to the FEM Hessian.
        std::vector<int> pinned_mask(impl.tetMesh.vertexNum, 0);
        for(int i = 0; i < n_pins; i++)
            pinned_mask[fem_v_vec[i]] = 1;
        if(impl.d_tetMesh.is_pinned_vertex == nullptr)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.is_pinned_vertex,
                                      impl.tetMesh.vertexNum * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.is_pinned_vertex,
                                  pinned_mask.data(),
                                  impl.tetMesh.vertexNum * sizeof(int),
                                  cudaMemcpyHostToDevice));
        // wire the mask into GIPC for kernel access
        impl.ipc.m_d_is_pinned_vertex = impl.d_tetMesh.is_pinned_vertex;

        // [M3.5] Build vertex_to_pin_idx (size = vertexNum) for O(1)
        // lookup of (body_id, lo) given a vertex index.  -1 = not pinned.
        std::vector<int> v2pin_host(impl.tetMesh.vertexNum, -1);
        for(int i = 0; i < n_pins; i++)
            v2pin_host[fem_v_vec[i]] = i;
        if(impl.d_tetMesh.vertex_to_pin_idx == nullptr)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.vertex_to_pin_idx,
                                      impl.tetMesh.vertexNum * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.vertex_to_pin_idx,
                                  v2pin_host.data(),
                                  impl.tetMesh.vertexNum * sizeof(int),
                                  cudaMemcpyHostToDevice));

        // [Hybrid mesh] populate d_tet_to_abd_body (per-tet body assignment).
        // A tet whose 4 verts are ALL pinned to the SAME ABD body is rigid-
        // internal: its Green strain is zero for any rigid motion of the
        // body, so its FEM elasticity is structurally redundant w.r.t. the
        // ABD body's own energy.  Marking it here lets Phase 4's elasticity
        // kernel early-exit, saving a co-rotational SVD per such tet per
        // Newton iter.  Computed once at finalize since pin info is static.
        {
            const auto& tets    = impl.tetMesh.tetrahedras;       // vector<uint4>
            const auto& body_id = impl.tetMesh.fem_pin_abd_body_id; // vector<int>
            std::vector<int> tet_to_abd(impl.tetMesh.tetrahedraNum, -1);
            int n_rigid_tets = 0;
            for(int t = 0; t < impl.tetMesh.tetrahedraNum; ++t)
            {
                const uint4 vs = tets[t];
                const int p0 = v2pin_host[vs.x];
                const int p1 = v2pin_host[vs.y];
                const int p2 = v2pin_host[vs.z];
                const int p3 = v2pin_host[vs.w];
                if(p0 < 0 || p1 < 0 || p2 < 0 || p3 < 0) continue;
                const int b0 = body_id[p0];
                if(body_id[p1] != b0 || body_id[p2] != b0 || body_id[p3] != b0)
                    continue;
                tet_to_abd[t] = b0;
                ++n_rigid_tets;
            }
            if(impl.d_tetMesh.d_tet_to_abd_body == nullptr)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.d_tet_to_abd_body,
                                          impl.tetMesh.tetrahedraNum * sizeof(int)));
            }
            CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.d_tet_to_abd_body,
                                      tet_to_abd.data(),
                                      impl.tetMesh.tetrahedraNum * sizeof(int),
                                      cudaMemcpyHostToDevice));
            if(n_rigid_tets > 0)
            {
                printf("[Hybrid] %d / %d tets are rigid-internal "
                       "(all 4 verts pinned to same body) — Phase 4 will "
                       "skip their elasticity\n",
                       n_rigid_tets, impl.tetMesh.tetrahedraNum);
            }
        }

        printf("[M1+M2+M3.5] %d FEM pins: local_pos transformed, btype=Fixed, "
               "is_pinned_vertex mask + vertex_to_pin_idx uploaded\n", n_pins);
    }

    impl.finalized = true;
    runtime_attempt.commit();
    std::cout << "[SimEngine] Finalized: "
              << impl.ipc.vertexNum << " verts, "
              << impl.ipc.surface_Num << " surface faces, "
              << impl.ipc.edge_Num << " edges" << std::endl;
}

// ======================== step ========================
// [NAN_DIAG] Per-step diagnostic dump (env-gated). Pulls FEM tet volumes,
// vertex velocities and positions to host; reports min/max + NaN counts.
// Useful for pinning down whether NaN is born from tet inversion,
// stitch-spring blowup, kappa overflow, or PCG numerical breakdown.
//
// Activate: NAN_DIAG=1 ./run examples/...
//
// Overhead: ~ vertexNum * 48 bytes D->H copy per step, only when enabled.
namespace {
struct NanDiagState {
    bool enabled = false;
    bool initialized = false;
    int  step_count = 0;
    bool nan_seen   = false;

    void init() {
        if(initialized) return;
        const char* e = std::getenv("NAN_DIAG");
        enabled = (e != nullptr && std::string(e) != "0");
        initialized = true;
        if(enabled) printf("[NAN_DIAG] enabled (env NAN_DIAG=1)\n");
    }
};
static NanDiagState g_diag;

// [NaN-sentinel] opt-in NaN watchdog. One device int gets atomicCAS'd if
// any vertex is NaN/Inf; first occurrence triggers a human-readable
// warning.
//
// **Default OFF** — measured ~163 ms/step overhead on case_27_softgripper
// (12k verts), which is ~150% of a normal 100ms step. The cost comes
// from the synchronous cudaMemcpy(4B, D->H) breaking GPU pipeline
// overlap with the IPC solver. Activate only when debugging NaN:
//
//     NAN_SENTINEL=1 ./run examples/...
//
// Or set NAN_DIAG=1 (which is even more verbose, also opt-in).
struct NanSentinelState {
    int* d_flag = nullptr;
    bool warned = false;
    bool enabled = false;
    bool initialized = false;
    int  step_count = 0;
    void init() {
        if(initialized) return;
        const char* e = std::getenv("NAN_SENTINEL");
        enabled = (e != nullptr && std::string(e) != "0");
        // NAN_DIAG implies NAN_SENTINEL — the diagnostic dump already
        // pulls vertex data, may as well surface a clear warning too.
        const char* diag = std::getenv("NAN_DIAG");
        if(diag != nullptr && std::string(diag) != "0") enabled = true;
        initialized = true;
        if(enabled) printf("[NaN-SENTINEL] enabled (env NAN_SENTINEL=1 or NAN_DIAG=1)\n");
    }
    void ensure_buffer() {
        if(d_flag == nullptr) {
            cudaMalloc(&d_flag, sizeof(int));
        }
    }
};
static NanSentinelState g_sentinel;
}  // namespace

__global__ static void _nan_sentinel_kernel(const double3* verts,
                                            const double3* velocities,
                                            int n,
                                            int* out_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double3 p = verts[idx];
    double3 v = velocities[idx];
    bool bad = isnan(p.x) || isnan(p.y) || isnan(p.z)
               || isinf(p.x) || isinf(p.y) || isinf(p.z)
               || isnan(v.x) || isnan(v.y) || isnan(v.z)
               || isinf(v.x) || isinf(v.y) || isinf(v.z);
    if(bad) atomicCAS(out_flag, 0, 1);
}

static void check_nan_sentinel_(int n_v,
                                const double3* d_vertexes,
                                const double3* d_velocities)
{
    g_sentinel.init();
    if(!g_sentinel.enabled) return;
    g_sentinel.ensure_buffer();
    if(n_v <= 0 || g_sentinel.d_flag == nullptr) return;
    int sc = g_sentinel.step_count++;

    cudaMemset(g_sentinel.d_flag, 0, sizeof(int));
    int blocks = (n_v + 255) / 256;
    _nan_sentinel_kernel<<<blocks, 256>>>(d_vertexes,
                                          d_velocities,
                                          n_v,
                                          g_sentinel.d_flag);
    int h_flag = 0;
    cudaMemcpy(&h_flag, g_sentinel.d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_flag != 0 && !g_sentinel.warned) {
        g_sentinel.warned = true;
        printf("\n========================================================================\n");
        printf("[NaN-SENTINEL] *** NaN/Inf detected in vertex positions or velocities\n");
        printf("[NaN-SENTINEL] *** at step %d. Physics has DIVERGED — subsequent steps\n", sc);
        printf("[NaN-SENTINEL] *** will be garbage and the engine cannot self-recover.\n");
        printf("[NaN-SENTINEL] *** Most common causes:\n");
        printf("[NaN-SENTINEL] ***   1. soft_motion_rate too high vs FEM Young modulus\n");
        printf("[NaN-SENTINEL] ***      (causes Hessian condition number blowup -> PCG NaN)\n");
        printf("[NaN-SENTINEL] ***   2. dt too large for the prescribed joint speed\n");
        printf("[NaN-SENTINEL] ***   3. FEM tet inverted (collision-driven over-compression)\n");
        printf("[NaN-SENTINEL] *** Re-run with NAN_DIAG=1 to see per-step min(tet_vol),\n");
        printf("[NaN-SENTINEL] *** max|v|, max|p|, NaN count + body_id breakdown.\n");
        printf("========================================================================\n\n");
        fflush(stdout);
    }
}

static void dump_nan_diagnostics_(int n_tet, int n_v,
                                  const double* d_volum,
                                  const double3* d_velocities,
                                  const double3* d_vertexes,
                                  const std::vector<int>& point_id_to_body_id)
{
    g_diag.init();
    if(!g_diag.enabled) return;
    int sc = g_diag.step_count++;

    // ---- min(tet_vol) — H1 tet-inverted detector ----
    double min_vol = 0.0;
    int    n_neg_vol = 0;
    if(n_tet > 0)
    {
        std::vector<double> host_vol(n_tet);
        cudaMemcpy(host_vol.data(), d_volum,
                   n_tet * sizeof(double), cudaMemcpyDeviceToHost);
        min_vol = *std::min_element(host_vol.begin(), host_vol.end());
        for(double v : host_vol) if(v < 0.0) ++n_neg_vol;
    }

    // ---- velocities + positions — H2/H4 detectors + NaN tracker ----
    std::vector<double3> host_v(n_v), host_p(n_v);
    cudaMemcpy(host_v.data(), d_velocities,
               n_v * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_p.data(), d_vertexes,
               n_v * sizeof(double3), cudaMemcpyDeviceToHost);

    double max_v2 = 0.0, max_p2 = 0.0;
    int n_nan_v = 0, n_nan_p = 0;
    int first_nan_v = -1, first_nan_p = -1;
    for(int i = 0; i < n_v; i++)
    {
        const double3& v = host_v[i];
        const double3& p = host_p[i];
        bool vn = (std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z)
                   || std::isinf(v.x) || std::isinf(v.y) || std::isinf(v.z));
        bool pn = (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)
                   || std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z));
        if(vn) { ++n_nan_v; if(first_nan_v < 0) first_nan_v = i; }
        if(pn) { ++n_nan_p; if(first_nan_p < 0) first_nan_p = i; }
        if(!vn) {
            double s = v.x*v.x + v.y*v.y + v.z*v.z;
            if(s > max_v2) max_v2 = s;
        }
        if(!pn) {
            double s = p.x*p.x + p.y*p.y + p.z*p.z;
            if(s > max_p2) max_p2 = s;
        }
    }
    double max_v = std::sqrt(max_v2);
    double max_p = std::sqrt(max_p2);

    bool first_nan_step = (n_nan_p + n_nan_v > 0) && !g_diag.nan_seen;
    if(first_nan_step) g_diag.nan_seen = true;

    printf("[NAN_DIAG] step=%4d  min_tet_vol=%+10.3e  neg_vol=%4d  "
           "max|v|=%9.3e  max|p|=%9.3e  nan_v=%4d  nan_p=%4d%s\n",
           sc, min_vol, n_neg_vol, max_v, max_p, n_nan_v, n_nan_p,
           first_nan_step ? "  <-- FIRST NaN HERE" : "");
    if(first_nan_step) {
        if(first_nan_p >= 0) {
            int bid = (first_nan_p < (int)point_id_to_body_id.size())
                      ? point_id_to_body_id[first_nan_p] : -2;
            printf("[NAN_DIAG]   first NaN position vertex idx=%d body_id=%d\n",
                   first_nan_p, bid);
        }
        if(first_nan_v >= 0) {
            int bid = (first_nan_v < (int)point_id_to_body_id.size())
                      ? point_id_to_body_id[first_nan_v] : -2;
            printf("[NAN_DIAG]   first NaN velocity vertex idx=%d body_id=%d\n",
                   first_nan_v, bid);
        }
    }
    fflush(stdout);
}
