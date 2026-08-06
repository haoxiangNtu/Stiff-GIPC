void SimEngine::step()
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError("step() requires a finalized SimEngine");
    cudaSetDevice(impl.cfg.cuda_device);
    if(impl.ipc.gpu_rl_graph_prepared())
    {
        // [step-autoroute] Isaac-style surface: after prepare_gpu_rl() the
        // per-frame step() becomes a thin enqueue of the recorded RL graph
        // instead of a lifecycle error. Host-set joint targets are published
        // to the device action slab on the engine stream, so the launch stays
        // fully asynchronous; device reads (obs ABI, getters that copy from
        // device memory on the same stream) are stream-ordered behind it.
        GIPCTripletMatrix::LayoutForceOnScope layout_force_rl;
        // Honour the launch-stream affinity contract: once an external agent
        // bound a stream via launch_gpu_rl_async, step() publishes and
        // launches on that same stream instead of the per-thread default.
        const uintptr_t    bound = impl.ipc.gpu_rl_bound_stream();
        const cudaStream_t rl_stream =
            bound ? reinterpret_cast<cudaStream_t>(bound)
                  : cudaStreamPerThread;
        const auto& rev = impl.tetMesh.joint_angle_controls;
        const auto& pri = impl.tetMesh.prismatic_drive_controls;
        if(!rev.empty())
        {
            std::vector<RevoluteDrivingControlPacked> packed(rev.size());
            for(size_t i = 0; i < rev.size(); ++i)
                packed[i] = RevoluteDrivingControlPacked{
                    static_cast<Float>(rev[i].target_angle),
                    static_cast<Float>(rev[i].strength_ratio),
                    static_cast<Float>(rev[i].ext_torque)};
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                reinterpret_cast<void*>(
                    impl.ipc.gpu_rl_revolute_actions_device_ptr()),
                packed.data(),
                packed.size() * sizeof(RevoluteDrivingControlPacked),
                cudaMemcpyHostToDevice,
                rl_stream));
        }
        if(!pri.empty())
        {
            std::vector<PrismaticDrivingControlPacked> packed(pri.size());
            for(size_t i = 0; i < pri.size(); ++i)
                packed[i] = PrismaticDrivingControlPacked{
                    static_cast<Float>(pri[i].target_distance),
                    static_cast<Float>(pri[i].strength_ratio),
                    static_cast<Float>(pri[i].ext_force)};
            CUDA_SAFE_CALL(cudaMemcpyAsync(
                reinterpret_cast<void*>(
                    impl.ipc.gpu_rl_prismatic_actions_device_ptr()),
                packed.data(),
                packed.size() * sizeof(PrismaticDrivingControlPacked),
                cudaMemcpyHostToDevice,
                rl_stream));
        }
        impl.ipc.launch_gpu_rl_graph_async(bound);
        impl.step_count++;
        return;
    }
    if(impl.ipc.episode_graph_in_flight())
        throw LifecycleError(
            "step() is unavailable while an episode graph is in flight");

    if(!impl.tetMesh.joint_angle_controls.empty()
       || !impl.tetMesh.prismatic_drive_controls.empty())
    {
        impl.ipc.update_joint_angle_targets_from_mesh(impl.tetMesh);
    }

    const int newton_before = impl.ipc.m_total_newton_iters;
    const char* frame_graph_env = std::getenv("STIFF_FRAME_GRAPH");
    const bool frame_graph_requested =
        frame_graph_env && frame_graph_env[0] && std::atoi(frame_graph_env) != 0;
    if(frame_graph_requested)
    {
        impl.ipc.IPC_Solver_FrameGraph(impl.d_tetMesh);
    }
    else
    {
        impl.ipc.IPC_Solver(impl.d_tetMesh);
        // Legacy remains synchronous at its own terminal event. Keep status
        // publication separate so callers can distinguish it from the graph
        // transaction path without special-casing step().
        impl.ipc.record_legacy_frame_status(
            false,
            false,
            impl.ipc.m_total_newton_iters - newton_before);
    }
    impl.step_count++;

    const bool full_frame_graph =
        (impl.ipc.get_frame_status().path_flags
         & frame_fsm::PATH_FULL_CONDITIONAL_GRAPH)
        != 0;
    if(!full_frame_graph)
    {
        // [NaN-sentinel] always-on lightweight NaN watchdog (~1 atomic int +
        // 4-byte D->H copy per step). The full graph performs the same finite
        // validation before its sole terminal status packet, so repeating the
        // sentinel there would violate the one-boundary contract.
        check_nan_sentinel_(impl.tetMesh.vertexNum,
                            impl.d_tetMesh.vertexes,
                            impl.d_tetMesh.velocities);

        // [NAN_DIAG] env-gated — only runs when NAN_DIAG=1. Full-graph
        // eligibility rejects that diagnostic mode.
        dump_nan_diagnostics_(impl.tetMesh.tetrahedraNum,
                              impl.tetMesh.vertexNum,
                              impl.d_tetMesh.volum,
                              impl.d_tetMesh.velocities,
                              impl.d_tetMesh.vertexes,
                              impl.tetMesh.point_id_to_body_id);
    }
}

#ifdef STIFF_BVH_COHERENCE_AUDIT_BUILD
void SimEngine::print_bvh_coherence_audit() const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    m_impl->ipc.printBvhTemporalCoherence();
}
#endif

void SimEngine::launch_episode_async(
    int frames,
    const double* revolute_actions,
    int revolute_joints,
    const double* prismatic_actions,
    int prismatic_joints)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "launch_episode_async() requires a finalized SimEngine");
    if(impl.ipc.gpu_rl_graph_prepared())
        throw LifecycleError(
            "launch_episode_async() is unavailable in GPU-native RL mode; "
            "call end_gpu_rl() first");
    if(impl.ipc.episode_graph_in_flight())
        throw LifecycleError("an episode graph is already in flight");
    if(frames <= 0)
        throw std::invalid_argument("episode frames must be positive");
    const int expected_revolute =
        static_cast<int>(impl.tetMesh.joint_angle_controls.size());
    const int expected_prismatic =
        static_cast<int>(
            impl.tetMesh.prismatic_drive_controls.size());
    if(revolute_joints != expected_revolute
       || prismatic_joints != expected_prismatic)
        throw std::invalid_argument(
            "episode action joint dimensions do not match the scene");
    if((revolute_joints && !revolute_actions)
       || (prismatic_joints && !prismatic_actions))
        throw std::invalid_argument(
            "non-empty episode actions require data");

    const auto checked_elements = [frames](int joints)
    {
        if(joints < 0
           || static_cast<size_t>(frames)
                  > std::numeric_limits<size_t>::max()
                        / static_cast<size_t>(
                            std::max(1, joints))
                        / 3u)
            throw std::overflow_error(
                "episode action shape is too large");
        return static_cast<size_t>(frames)
               * static_cast<size_t>(joints) * 3u;
    };
    const size_t revolute_values =
        checked_elements(revolute_joints);
    const size_t prismatic_values =
        checked_elements(prismatic_joints);
    impl.episode_revolute_actions.clear();
    impl.episode_prismatic_actions.clear();
    if(revolute_values)
        impl.episode_revolute_actions.assign(
            revolute_actions,
            revolute_actions + revolute_values);
    if(prismatic_values)
        impl.episode_prismatic_actions.assign(
            prismatic_actions,
            prismatic_actions + prismatic_values);

    std::vector<RevoluteDrivingControlPacked> revolute_packed(
        static_cast<size_t>(frames)
        * static_cast<size_t>(revolute_joints));
    std::vector<PrismaticDrivingControlPacked> prismatic_packed(
        static_cast<size_t>(frames)
        * static_cast<size_t>(prismatic_joints));
    for(size_t i = 0; i < revolute_packed.size(); ++i)
    {
        const double target   = revolute_actions[3 * i + 0];
        const double strength = revolute_actions[3 * i + 1];
        const double external = revolute_actions[3 * i + 2];
        if(!std::isfinite(target) || !std::isfinite(strength)
           || !std::isfinite(external))
            throw std::invalid_argument(
                "revolute episode actions must be finite");
        revolute_packed[i] = RevoluteDrivingControlPacked{
            static_cast<Float>(target),
            static_cast<Float>(strength),
            static_cast<Float>(external)};
    }
    for(size_t i = 0; i < prismatic_packed.size(); ++i)
    {
        const double target   = prismatic_actions[3 * i + 0];
        const double strength = prismatic_actions[3 * i + 1];
        const double external = prismatic_actions[3 * i + 2];
        if(!std::isfinite(target) || !std::isfinite(strength)
           || !std::isfinite(external))
            throw std::invalid_argument(
                "prismatic episode actions must be finite");
        prismatic_packed[i] = PrismaticDrivingControlPacked{
            static_cast<Float>(target),
            static_cast<Float>(strength),
            static_cast<Float>(external)};
    }

    cudaSetDevice(impl.cfg.cuda_device);
    impl.ipc.prepare_episode_graph(
        impl.d_tetMesh,
        frames,
        revolute_packed.empty() ? nullptr : revolute_packed.data(),
        revolute_joints,
        prismatic_packed.empty() ? nullptr : prismatic_packed.data(),
        prismatic_joints);
    impl.ipc.launch_episode_graph_async(
        impl.d_tetMesh, impl.ipc.m_total_frames);
    impl.episode_frames = frames;
}

bool SimEngine::episode_in_flight() const
{
    return m_impl->ipc.episode_graph_in_flight();
}

bool SimEngine::episode_observation_ready(int slot) const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    return m_impl->ipc.episode_observation_ready(slot);
}

void SimEngine::wait_episode_observation(int slot) const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    m_impl->ipc.wait_episode_observation(slot);
}

int SimEngine::get_episode_slot_first_frame(int slot) const
{
    return m_impl->ipc.episode_slot_first_frame(slot);
}

int SimEngine::get_episode_slot_frame_count(int slot) const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    return m_impl->ipc.episode_slot_frame_count(slot);
}

int SimEngine::get_episode_attempted_frame_count() const
{
    return m_impl->ipc.episode_attempted_frame_count();
}

void SimEngine::get_episode_observation(
    int slot,
    double* positions,
    double* velocities,
    frame_fsm::FrameStatus* statuses,
    int frame_capacity) const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    const int frame_count =
        m_impl->ipc.episode_slot_frame_count(slot);
    const int vertex_count = m_impl->ipc.vertexNum;
    if(frame_capacity < frame_count)
        throw std::invalid_argument(
            "episode observation capacity is too small");
    if(frame_count > 0
       && vertex_count > 0
       && (!positions || !velocities))
        throw std::invalid_argument(
            "episode observation outputs must not be null");

    const size_t elements =
        static_cast<size_t>(frame_count)
        * static_cast<size_t>(vertex_count);
    std::vector<double3> raw_positions(elements);
    std::vector<double3> raw_velocities(elements);
    m_impl->ipc.copy_episode_observation_slot(
        slot,
        raw_positions.empty() ? nullptr : raw_positions.data(),
        raw_velocities.empty() ? nullptr : raw_velocities.data(),
        statuses,
        frame_capacity);

    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    const bool use_perm =
        !perm.empty()
        && static_cast<int>(perm.size()) >= vertex_count;
    for(int frame = 0; frame < frame_count; ++frame)
    {
        const size_t base =
            static_cast<size_t>(frame) * vertex_count;
        for(int internal = 0; internal < vertex_count; ++internal)
        {
            int output = use_perm ? perm[internal] : internal;
            if(output < 0 || output >= vertex_count)
                output = internal;
            const double3& p = raw_positions[base + internal];
            const double3& v = raw_velocities[base + internal];
            const size_t dst =
                (base + static_cast<size_t>(output)) * 3u;
            positions[dst + 0] = p.x;
            positions[dst + 1] = p.y;
            positions[dst + 2] = p.z;
            velocities[dst + 0] = v.x;
            velocities[dst + 1] = v.y;
            velocities[dst + 2] = v.z;
        }
    }
}

int SimEngine::finish_episode()
{
    auto& impl = *m_impl;
    if(!impl.ipc.episode_graph_in_flight())
        throw LifecycleError("no episode graph is in flight");
    cudaSetDevice(impl.cfg.cuda_device);
    const int successful = impl.ipc.finish_episode_graph();
    impl.step_count += successful;

    if(successful > 0)
    {
        const int action_frame = successful - 1;
        const int revolute_count =
            static_cast<int>(
                impl.tetMesh.joint_angle_controls.size());
        for(int joint = 0; joint < revolute_count; ++joint)
        {
            const size_t offset =
                (static_cast<size_t>(action_frame)
                     * revolute_count
                 + joint)
                * 3u;
            auto& control =
                impl.tetMesh.joint_angle_controls[joint];
            control.target_angle =
                impl.episode_revolute_actions[offset + 0];
            control.strength_ratio =
                impl.episode_revolute_actions[offset + 1];
            control.ext_torque =
                impl.episode_revolute_actions[offset + 2];
        }
        const int prismatic_count =
            static_cast<int>(
                impl.tetMesh.prismatic_drive_controls.size());
        for(int joint = 0; joint < prismatic_count; ++joint)
        {
            const size_t offset =
                (static_cast<size_t>(action_frame)
                     * prismatic_count
                 + joint)
                * 3u;
            auto& control =
                impl.tetMesh.prismatic_drive_controls[joint];
            control.target_distance =
                impl.episode_prismatic_actions[offset + 0];
            control.strength_ratio =
                impl.episode_prismatic_actions[offset + 1];
            control.ext_force =
                impl.episode_prismatic_actions[offset + 2];
        }
    }
    return successful;
}

void SimEngine::prepare_gpu_rl()
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "prepare_gpu_rl() requires a finalized SimEngine");
    if(impl.ipc.episode_graph_in_flight())
        throw LifecycleError(
            "end the active episode before preparing GPU-native RL mode");

    const int revolute_count =
        static_cast<int>(impl.tetMesh.joint_angle_controls.size());
    const int prismatic_count =
        static_cast<int>(
            impl.tetMesh.prismatic_drive_controls.size());
    cudaSetDevice(impl.cfg.cuda_device);
    // [self-contained prepare] Isaac-style contract: no environment knobs.
    // If the user's warm-up frames ran in the pure host layout (no
    // STIFF_FRAME_GRAPH), the capacity tiers were never trained; run ONE
    // training frame with the layout forced on so every axis observes its
    // peak from a representative frame, then capture under the same forced
    // layout. Users who set the env keep the exact previous behaviour.
    const bool layout_was_on = GIPCTripletMatrix::device_count_mode();
    GIPCTripletMatrix::LayoutForceOnScope layout_force;
    if(!layout_was_on)
        this->step();
    impl.ipc.prepare_episode_graph(
        impl.d_tetMesh,
        1,
        nullptr,
        revolute_count,
        nullptr,
        prismatic_count,
        true);
    impl.gpu_rl_layout_forced = !layout_was_on;
    impl.episode_frames = 1;
    impl.episode_revolute_actions.clear();
    impl.episode_prismatic_actions.clear();
}

void SimEngine::prepare_gpu_rl_episode(int frame_count)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "prepare_gpu_rl_episode() requires a finalized SimEngine");
    if(frame_count <= 1)
        throw std::invalid_argument(
            "prepare_gpu_rl_episode() requires frame_count > 1");
    if(impl.ipc.episode_graph_in_flight())
        throw LifecycleError(
            "end the active episode before preparing GPU-native RL mode");

    const int revolute_count = static_cast<int>(
        impl.tetMesh.joint_angle_controls.size());
    const int prismatic_count = static_cast<int>(
        impl.tetMesh.prismatic_drive_controls.size());
    cudaSetDevice(impl.cfg.cuda_device);
    // [self-contained prepare] see prepare_gpu_rl(): train tiers with the
    // layout forced on when the environment never enabled it.
    const bool layout_was_on = GIPCTripletMatrix::device_count_mode();
    GIPCTripletMatrix::LayoutForceOnScope layout_force;
    if(!layout_was_on)
        this->step();
    impl.ipc.prepare_episode_graph(
        impl.d_tetMesh,
        frame_count,
        nullptr,
        revolute_count,
        nullptr,
        prismatic_count,
        true);
    impl.gpu_rl_layout_forced = !layout_was_on;
    impl.episode_frames = frame_count;
    impl.episode_revolute_actions.clear();
    impl.episode_prismatic_actions.clear();
}

void SimEngine::launch_gpu_rl_async(uintptr_t cuda_stream)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "launch_gpu_rl_async() requires a finalized SimEngine");
    cudaSetDevice(impl.cfg.cuda_device);
    // [self-contained prepare] replays and their boundary logic run under
    // the layout the graph was captured with, independent of env knobs.
    GIPCTripletMatrix::LayoutForceOnScope layout_force;
    impl.ipc.launch_gpu_rl_graph_async(cuda_stream);
}

void SimEngine::launch_gpu_rl_episode_async(uintptr_t cuda_stream)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "launch_gpu_rl_episode_async() requires a finalized SimEngine");
    cudaSetDevice(impl.cfg.cuda_device);
    GIPCTripletMatrix::LayoutForceOnScope layout_force;
    impl.ipc.launch_gpu_rl_episode_graph_async(cuda_stream);
}

bool SimEngine::gpu_rl_prepared() const
{
    return m_impl->ipc.gpu_rl_graph_prepared();
}

bool SimEngine::gpu_rl_ready() const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    return m_impl->ipc.gpu_rl_graph_ready();
}

void SimEngine::synchronize_gpu_rl() const
{
    cudaSetDevice(m_impl->cfg.cuda_device);
    GIPCTripletMatrix::LayoutForceOnScope layout_force;
    m_impl->ipc.synchronize_gpu_rl_graph();
}

void SimEngine::end_gpu_rl()
{
    auto& impl = *m_impl;
    if(!impl.ipc.gpu_rl_graph_prepared())
        throw LifecycleError("GPU-native RL mode is not prepared");
    cudaSetDevice(impl.cfg.cuda_device);
    {
        GIPCTripletMatrix::LayoutForceOnScope layout_force;
        impl.ipc.destroy_episode_graph();
    }
    impl.gpu_rl_layout_forced = false;
}

uintptr_t SimEngine::get_gpu_rl_revolute_actions_device_ptr() const
{
    return m_impl->ipc.gpu_rl_revolute_actions_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_prismatic_actions_device_ptr() const
{
    return m_impl->ipc.gpu_rl_prismatic_actions_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_positions_device_ptr() const
{
    return m_impl->ipc.gpu_rl_positions_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_velocities_device_ptr() const
{
    return m_impl->ipc.gpu_rl_velocities_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_statuses_device_ptr() const
{
    return m_impl->ipc.gpu_rl_statuses_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_frame_counter_device_ptr() const
{
    return m_impl->ipc.gpu_rl_frame_counter_device_ptr();
}

uintptr_t SimEngine::get_gpu_rl_joint_observations_device_ptr() const
{
    return m_impl->ipc.gpu_rl_joint_observations_device_ptr();
}

int SimEngine::get_gpu_rl_joint_observation_count() const
{
    return m_impl->ipc.gpu_rl_joint_observation_count();
}

void SimEngine::launch_gpu_rl_reset_async(uintptr_t cuda_stream)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "launch_gpu_rl_reset_async() requires a finalized SimEngine");
    cudaSetDevice(impl.cfg.cuda_device);
    impl.ipc.launch_gpu_rl_reset_async(cuda_stream);
}

void SimEngine::launch_gpu_rl_reset_masked_async(uintptr_t d_env_mask,
                                                 uintptr_t cuda_stream)
{
    auto& impl = *m_impl;
    if(!impl.finalized)
        throw LifecycleError(
            "launch_gpu_rl_reset_masked_async() requires a finalized "
            "SimEngine");
    cudaSetDevice(impl.cfg.cuda_device);
    impl.ipc.launch_gpu_rl_reset_masked_async(d_env_mask, cuda_stream);
}

uintptr_t SimEngine::get_point_to_group_device_ptr() const
{
    return reinterpret_cast<uintptr_t>(
        m_impl->d_tetMesh.d_point_to_group);
}

uintptr_t SimEngine::get_env_quarantined_device_ptr() const
{
    return reinterpret_cast<uintptr_t>(
        m_impl->ipc.m_d_env_quarantined.data());
}

int SimEngine::get_env_group_count() const
{
    return m_impl->d_tetMesh.h_group_count;
}

int SimEngine::get_gpu_rl_graph_node_count() const
{
    return m_impl->ipc.gpu_rl_graph_node_count();
}

int SimEngine::get_gpu_rl_graph_h2d_count() const
{
    return m_impl->ipc.gpu_rl_graph_h2d_count();
}

int SimEngine::get_gpu_rl_graph_d2h_count() const
{
    return m_impl->ipc.gpu_rl_graph_d2h_count();
}

int SimEngine::get_gpu_rl_episode_frame_count() const
{
    return m_impl->ipc.gpu_rl_episode_frame_count();
}

int SimEngine::get_gpu_rl_status_size_bytes() const
{
    return static_cast<int>(sizeof(frame_fsm::FrameStatus));
}

// ======================== state queries ========================
int SimEngine::get_vertex_count() const
{
    return m_impl->ipc.vertexNum;
}

uintptr_t SimEngine::get_vertices_device_ptr() const
{
    // [gpu-direct] Raw device pointer to the global vertex buffer (double3*,
    // length vertexNum). Lets an external GPU framework (Warp) read FEM vertex
    // positions straight from device memory without a host round-trip. Valid
    // after finalize(); contents update after each step().
    return reinterpret_cast<uintptr_t>(m_impl->ipc._vertexes);
}

uintptr_t SimEngine::get_vertex_velocities_device_ptr() const
{
    return reinterpret_cast<uintptr_t>(
        m_impl->d_tetMesh.velocities);
}

int SimEngine::get_surface_face_count() const
{
    return static_cast<int>(m_impl->tetMesh.surface.size());
}

int SimEngine::get_surface_vertex_count() const
{
    return static_cast<int>(m_impl->tetMesh.surfVerts.size());
}

void SimEngine::get_vertex_positions(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] Transparent unscramble. perm[i] = j means engine-internal
    // vertex i corresponds to input-mesh vertex j. We want user-facing
    // output in input order: out[j] = engine_pos[i] for each i.
    // If perm is empty / mismatched, fall back to identity (raw copy).
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(out_xyz, m_impl->ipc._vertexes,
                                  n * sizeof(double3), cudaMemcpyDeviceToHost));
        return;
    }
    // Read engine-order vertices to a temp buffer, then permute into out.
    std::vector<double3> tmp(n);
    CUDA_SAFE_CALL(cudaMemcpy(tmp.data(), m_impl->ipc._vertexes,
                              n * sizeof(double3), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            // Defensive: out-of-range perm entry, fall back to identity for this slot.
            out_xyz[3 * i + 0] = tmp[i].x;
            out_xyz[3 * i + 1] = tmp[i].y;
            out_xyz[3 * i + 2] = tmp[i].z;
            continue;
        }
        out_xyz[3 * j + 0] = tmp[i].x;
        out_xyz[3 * j + 1] = tmp[i].y;
        out_xyz[3 * j + 2] = tmp[i].z;
    }
}

void SimEngine::get_surface_faces(uint32_t* out_idx, int face_count) const
{
    int n = std::min(face_count, static_cast<int>(m_impl->tetMesh.surface.size()));
    if(n <= 0) return;

    // [MAS-perm] Map engine face indices (a, b, c) to input-mesh indices via perm.
    // Output triangle vertices reference vertex_metis_to_input[engine_idx].
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty()
                    && static_cast<int>(perm.size()) >= m_impl->ipc.vertexNum;
    if(!use_perm)
    {
        std::memcpy(out_idx, m_impl->tetMesh.surface.data(), n * sizeof(uint3));
        return;
    }
    const auto& surf = m_impl->tetMesh.surface;
    for(int i = 0; i < n; i++)
    {
        const uint3& f = surf[i];
        out_idx[3 * i + 0] = static_cast<uint32_t>(perm[f.x]);
        out_idx[3 * i + 1] = static_cast<uint32_t>(perm[f.y]);
        out_idx[3 * i + 2] = static_cast<uint32_t>(perm[f.z]);
    }
}

void SimEngine::get_surface_vertex_indices(uint32_t* out_idx, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->tetMesh.surfVerts.size()));
    if(n <= 0) return;

    // [MAS-perm] surfVerts stores engine-order vertex indices; translate to
    // input-order via perm so user-facing indexing is consistent with
    // get_vertices() / get_surface_faces().
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty()
                    && static_cast<int>(perm.size()) >= m_impl->ipc.vertexNum;
    if(!use_perm)
    {
        std::memcpy(out_idx, m_impl->tetMesh.surfVerts.data(), n * sizeof(uint32_t));
        return;
    }
    const auto& sv = m_impl->tetMesh.surfVerts;
    for(int i = 0; i < n; i++)
        out_idx[i] = static_cast<uint32_t>(perm[sv[i]]);
}

// ======================== joint control ========================
int SimEngine::get_num_revolute_joints() const
{
    return static_cast<int>(m_impl->tetMesh.joint_angle_controls.size());
}

int SimEngine::get_num_prismatic_joints() const
{
    return static_cast<int>(m_impl->tetMesh.prismatic_drive_controls.size());
}

JointInfo SimEngine::get_revolute_joint_info(int idx) const
{
    const auto& ctrl = m_impl->tetMesh.joint_angle_controls.at(idx);
    return JointInfo{
        ctrl.joint_name,
        ctrl.lower_limit,
        ctrl.upper_limit,
        ctrl.target_angle,
        ctrl.strength_ratio,
        false
    };
}

JointInfo SimEngine::get_prismatic_joint_info(int idx) const
{
    const auto& ctrl = m_impl->tetMesh.prismatic_drive_controls.at(idx);
    return JointInfo{
        ctrl.joint_name,
        ctrl.lower_limit,
        ctrl.upper_limit,
        ctrl.target_distance,
        ctrl.strength_ratio,
        true
    };
}

void SimEngine::set_revolute_target(int idx, double angle_rad)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).target_angle = angle_rad;
}

void SimEngine::set_revolute_torque(int idx, double torque)
{
    // [force-control] Torque control on a revolute driving joint. Adds the
    // generalized force -tau*dtheta/dq to the driving gradient (no Hessian),
    // independent of the PD term. For PURE torque control, also call
    // set_revolute_strength(idx, 0). Synced to GPU each step via
    // update_revolute_driving_targets. Mirrors libuipc external joint torque.
    m_impl->tetMesh.joint_angle_controls.at(idx).ext_torque = torque;
}

void SimEngine::set_revolute_initial_offset(int idx, double offset_rad)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).initial_angle_offset = offset_rad;
}

void SimEngine::set_prismatic_target(int idx, double distance_m)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance = distance_m;
}

void SimEngine::set_prismatic_force(int idx, double force)
{
    // [force-control] External force (N) along a prismatic driving joint's axis.
    // Pushes child along +axis (parent gets the reaction) via the q_tilde path
    // (no Hessian), independent of the PD term. For PURE force control, also
    // call set_prismatic_strength(idx, 0). Synced to GPU each step via
    // update_prismatic_driving_targets. Mirrors libuipc external prismatic force.
    m_impl->tetMesh.prismatic_drive_controls.at(idx).ext_force = force;
}

void SimEngine::set_revolute_strength(int idx, double strength)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).strength_ratio = strength;
}

void SimEngine::set_prismatic_strength(int idx, double strength)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).strength_ratio = strength;
}

void SimEngine::set_fixed_joint_strength(int idx, double kappa)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system) {
        std::cerr << "[set_fixed_joint_strength] sim not finalized" << std::endl;
        return;
    }
    auto& abd_sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= abd_sys.m_num_joints) {
        std::cerr << "[set_fixed_joint_strength] idx " << idx
                  << " out of range [0," << abd_sys.m_num_joints << ")" << std::endl;
        return;
    }

    // Read current GPU data for this joint, modify kappa, write back.
    JointConstraintGPUData host_jd;
    CUDA_SAFE_CALL(cudaMemcpy(&host_jd,
                              abd_sys.m_joint_data.data() + idx,
                              sizeof(JointConstraintGPUData),
                              cudaMemcpyDeviceToHost));
    host_jd.kappa = static_cast<Float>(kappa);
    CUDA_SAFE_CALL(cudaMemcpy(abd_sys.m_joint_data.data() + idx,
                              &host_jd,
                              sizeof(JointConstraintGPUData),
                              cudaMemcpyHostToDevice));
    std::cout << "[set_fixed_joint_strength] joint #" << idx
              << " kappa = " << kappa << std::endl;
}

void SimEngine::set_body_animated_target(int body_id,
                                         double target_x, double target_y, double target_z,
                                         double strength)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(body_id < 0 || body_id >= num_bodies) return;

    if(impl.d_tetMesh.body_motor_params == nullptr) {
        // Buffer wasn't allocated (no motor_infos at finalize time).  Allocate
        // it here so per-step Animated drive can be set on bodies that
        // weren't pre-registered with body_motor_infos.
        size_t bytes = (size_t)num_bodies * 5 * sizeof(double);
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.body_motor_params, bytes));
        CUDA_SAFE_CALL(cudaMemset(impl.d_tetMesh.body_motor_params, 0, bytes));
    }

    double host_params[5] = {target_x, target_y, target_z, strength, 0.0};
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.body_motor_params + body_id * 5,
                              host_params,
                              5 * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void SimEngine::set_urdf_mesh_override(const std::string& link_name,
                                       const std::string& msh_path,
                                       double young_modulus)
{
    m_impl->pending_urdf_mesh_overrides[link_name] = {msh_path, young_modulus};
}

Eigen::Matrix4d SimEngine::get_urdf_link_transform(const std::string& link_name) const
{
    auto it = m_impl->urdf_link_transforms.find(link_name);
    if(it == m_impl->urdf_link_transforms.end()) {
        std::cerr << "[get_urdf_link_transform] link '" << link_name
                  << "' not found (URDF not loaded or wrong name)" << std::endl;
        return Eigen::Matrix4d::Identity();
    }
    return it->second;
}

void SimEngine::set_abd_body_density(int body_id, double density)
{
    // Override one ABD body's density (mass = density * volume). Must be called
    // AFTER the body is loaded and BEFORE finalize(). The ABDSystem does not
    // exist yet at this point (it is created inside build_gipc_system during
    // finalize), so we stash the override on GIPC; build_gipc_system transfers
    // it to the ABDSystem right before the per-body mass setup runs.
    m_impl->ipc.m_pending_abd_density[body_id] = density;
}

void SimEngine::set_abd_body_mass(int body_id, double mass)
{
    if(body_id < 0)
        throw std::runtime_error("set_abd_body_mass: body_id must be non-negative");
    if(!(mass > 0.0) || !std::isfinite(mass))
        throw std::runtime_error("set_abd_body_mass: mass must be finite and > 0");
    m_impl->ipc.m_pending_abd_mass[body_id] = mass;
}

void SimEngine::set_abd_body_inertia(int body_id, double mass,
                                     const double* com3, const double* inertia9)
{
    // Override one ABD body's inertial properties (mass, COM, 3x3 inertia about
    // the COM) — e.g. from URDF inertial tags via Newton body_mass/body_com/
    // body_inertia — instead of deriving them from the welded collision mesh,
    // whose centroid can be far off for a multi-shape link and skew the joint
    // driving torque. Stashed on GIPC (ABDSystem not built yet); transferred at
    // finalize. com/inertia are in the SAME world/load frame as the mesh verts.
    GIPC::PendingInertia pi;
    pi.mass = mass;
    for(int i = 0; i < 3; i++) pi.com[i] = com3[i];
    for(int i = 0; i < 9; i++) pi.inertia[i] = inertia9[i];
    m_impl->ipc.m_pending_abd_inertia[body_id] = pi;
}

void SimEngine::set_body_external_force(int body_id,
                                       double fx, double fy, double fz)
{
    // [force-control] Set the per-body external LINEAR force (N) on an ABD body.
    // Persistent until changed; pass (0,0,0) to clear. Enters the sim as an
    // acceleration M^{-1}F in q_tilde (cal_q_tilde.cu) — same path as gravity,
    // mirroring libuipc AffineBodyExternalBodyForce. (Prototype: linear only;
    // the buffer is a full 12-DOF wrench, so an affine/torque term can be added
    // by writing components [3:12].)
    auto& impl  = *m_impl;
    int   n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);
    if(body_id < 0 || body_id >= n_abd)
    {
        std::cerr << "[set_body_external_force] body_id " << body_id
                  << " is not an ABD body (n_abd=" << n_abd << ")" << std::endl;
        return;
    }
    if(!impl.ipc.m_abd_sim_data)
    {
        std::cerr << "[set_body_external_force] sim not finalized" << std::endl;
        return;
    }
    auto& f_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_ext_force;
    if(static_cast<int>(f_buf.size()) <= body_id)
    {
        std::cerr << "[set_body_external_force] body_id " << body_id
                  << " out of range (size=" << f_buf.size() << ")" << std::endl;
        return;
    }
    Eigen::Matrix<double, 12, 1> F = Eigen::Matrix<double, 12, 1>::Zero();
    F[0] = fx; F[1] = fy; F[2] = fz;
    CUDA_SAFE_CALL(cudaMemcpy(f_buf.data() + body_id, F.data(),
                              sizeof(double) * 12, cudaMemcpyHostToDevice));
}

void SimEngine::set_body_external_wrench(int body_id, const double* w12)
{
    // [force-control] Set the FULL 12-DOF external generalized force on an ABD
    // body: w[0:3] = linear force, w[3:12] = affine force (row-major vec(F_A)).
    // An affine wrench with w[5]=+omega, w[9]=-omega is a torque about Y (the
    // skew-symmetric part spins the body), mirroring libuipc's body-force test
    // which combines an orbiting linear force with a spinning affine term.
    auto& impl  = *m_impl;
    int   n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);
    if(body_id < 0 || body_id >= n_abd)
    {
        std::cerr << "[set_body_external_wrench] body_id " << body_id
                  << " is not an ABD body (n_abd=" << n_abd << ")" << std::endl;
        return;
    }
    if(!impl.ipc.m_abd_sim_data)
    {
        std::cerr << "[set_body_external_wrench] sim not finalized" << std::endl;
        return;
    }
    auto& f_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_ext_force;
    if(static_cast<int>(f_buf.size()) <= body_id)
    {
        std::cerr << "[set_body_external_wrench] body_id " << body_id
                  << " out of range (size=" << f_buf.size() << ")" << std::endl;
        return;
    }
    CUDA_SAFE_CALL(cudaMemcpy(f_buf.data() + body_id, w12,
                              sizeof(double) * 12, cudaMemcpyHostToDevice));
}

void SimEngine::set_body_apply_gravity(int body_id, bool enabled)
{
    auto& impl = *m_impl;
    int n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);

    if(body_id < n_abd)
    {
        // ABD body: gravity is precomputed at finalize as a 12-DOF
        // body_id_to_abd_gravity[i] vector. apply_gravity[] (per-vertex) is
        // ignored for ABD verts. To toggle, zero/restore the cached vector.
        if(!impl.ipc.m_abd_sim_data) {
            std::cerr << "[set_body_apply_gravity] sim not finalized" << std::endl;
            return;
        }

        using Vec12 = Eigen::Matrix<double, 12, 1>;
        auto& g_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_gravity;
        if(static_cast<int>(g_buf.size()) <= body_id) {
            std::cerr << "[set_body_apply_gravity] ABD body_id " << body_id
                      << " out of range (size=" << g_buf.size() << ")" << std::endl;
            return;
        }

        if(!enabled) {
            // Cache current gravity (so we can restore it).
            Vec12 host_g = Vec12::Zero();
            CUDA_SAFE_CALL(cudaMemcpy(host_g.data(),
                                      g_buf.data() + body_id,
                                      sizeof(Vec12), cudaMemcpyDeviceToHost));
            impl.disabled_abd_gravity_cache[body_id] = host_g;

            Vec12 zero = Vec12::Zero();
            CUDA_SAFE_CALL(cudaMemcpy(g_buf.data() + body_id,
                                      zero.data(),
                                      sizeof(Vec12), cudaMemcpyHostToDevice));
        } else {
            // Restore from cache if we previously disabled it
            auto it = impl.disabled_abd_gravity_cache.find(body_id);
            if(it != impl.disabled_abd_gravity_cache.end()) {
                CUDA_SAFE_CALL(cudaMemcpy(g_buf.data() + body_id,
                                          it->second.data(),
                                          sizeof(Vec12), cudaMemcpyHostToDevice));
                impl.disabled_abd_gravity_cache.erase(it);
            }
        }
        return;
    }

    // FEM body: per-vertex apply_gravity[] flag
    const auto& pt2body = impl.tetMesh.point_id_to_body_id;
    if(pt2body.empty() || impl.d_tetMesh.apply_gravity == nullptr) return;

    int v_start = -1, v_end = -1;
    int N = static_cast<int>(pt2body.size());
    for(int i = 0; i < N; i++)
    {
        if(pt2body[i] == body_id)
        {
            if(v_start < 0) v_start = i;
            v_end = i + 1;
        }
    }
    if(v_start < 0) {
        std::cerr << "[set_body_apply_gravity] body_id " << body_id
                  << " has no vertices" << std::endl;
        return;
    }

    int n = v_end - v_start;
    std::vector<int> host_flags(n, enabled ? 1 : 0);
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.apply_gravity + v_start,
                              host_flags.data(),
                              n * sizeof(int),
                              cudaMemcpyHostToDevice));
    for(int i = v_start; i < v_end; i++)
        impl.tetMesh.apply_gravity[i] = enabled ? 1 : 0;
}

void SimEngine::set_max_revolute_step_per_frame(double rad)
{
    auto& impl = *m_impl;
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.max_revolute_step_per_frame = rad;
    }
}

void SimEngine::set_max_prismatic_step_per_frame(double m)
{
    auto& impl = *m_impl;
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.max_prismatic_step_per_frame = m;
    }
}

double SimEngine::get_revolute_target(int idx) const
{
    return m_impl->tetMesh.joint_angle_controls.at(idx).target_angle;
}

double SimEngine::get_prismatic_target(int idx) const
{
    return m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance;
}

double SimEngine::get_prismatic_drive_force(int idx) const
{
    // [force-control] Current prismatic DRIVING force = K*(target - d), where
    // d = (Cq - Cp).dot(t) is the current opening along the joint axis. Used by
    // force-limited position control: advance the target toward closed until
    // this force reaches the desired F_max, then hold (real-gripper behavior).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
        return 0.0;
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving)
        return 0.0;

    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    Vec12 qp, qc;
    CUDA_SAFE_CALL(cudaMemcpy(&qp, q_buf.data() + drv.parent_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&qc, q_buf.data() + drv.child_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));

    auto worldpt = [](const Vec12& q, const Vector3& xb) -> Vector3 {
        Matrix3x3 A;
        A.row(0) = q.segment<3>(3).transpose();
        A.row(1) = q.segment<3>(6).transpose();
        A.row(2) = q.segment<3>(9).transpose();
        return Vector3(q.segment<3>(0) + A * xb);
    };
    Matrix3x3 Ac;
    Ac.row(0) = qc.segment<3>(3).transpose();
    Ac.row(1) = qc.segment<3>(6).transpose();
    Ac.row(2) = qc.segment<3>(9).transpose();

    Vector3 Cp = worldpt(qp, drv.Cp_bar);
    Vector3 Cq = worldpt(qc, drv.Cq_bar);
    Vector3 t  = Ac * drv.tq_bar;
    double  d  = (Cq - Cp).dot(t);
    return static_cast<double>(drv.stiffness) * (static_cast<double>(drv.target_distance) - d);
}

void SimEngine::get_vertex_contact_force_sum(int vert_offset, int vert_count,
                                             double* out3) const
{
    // [force-control][LEGACY UNITS] Body-body barrier GRADIENT sum over the
    // body's vertices: raw dE/dx = -force*dt^2, NOT Newtons; no ground, no
    // friction. Kept for pre-0.8.4 callers; physical forces in Newtons come
    // from get_vertex_contact_forces. Historical intent (grip reaction) — the
    // correct signal for true force control. Uses the collision pairs from the
    // last step(); call AFTER eng.step().
    out3[0] = out3[1] = out3[2] = 0.0;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || vert_offset < 0 || vert_count <= 0 || vert_offset + vert_count > nv)
        return;
    if(g.m_skip_all_collision)
        return;

    // Re-detect contacts at the CURRENT (post-step) state: the solver clears the
    // DCD pair count after a step, so we rebuild the BVH + collision pairs here
    // before evaluating the barrier (contact) gradient.
    g.buildBVH();
    g.buildCP();
    if(getenv("STIFF_CONTACT_DBG"))
        fprintf(stderr, "[contact_dbg] h_cpNum0=%u h_gpNum=%u Kappa=%g dHat=%g nv=%d\n",
                g.h_cpNum[0], g.h_gpNum.get(), g.Kappa, g.dHat, nv);
    if(g.h_cpNum[0] < 1 && g.h_gpNum < 1)
        return;  // nothing in contact (neither body-body nor ground)

    double3* d_grad = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_grad, nv * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_grad, 0, nv * sizeof(double3)));
    g.zeroBinnedGrad();                       // [4.3] calBarrierGradient scatters to binned buf
    g.calBarrierGradient(d_grad, g.Kappa);    // body-body (DCD) contact force per vertex (→ binned)
    g.combineBinnedGrad(d_grad);              // [4.3] fold binned contact force into d_grad
    g.computeGroundGradient(d_grad, g.Kappa); // ground half-plane contact force per vertex
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<double3> h(vert_count);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_grad + vert_offset,
                              vert_count * sizeof(double3), cudaMemcpyDeviceToHost));
    double sx = 0, sy = 0, sz = 0;
    for(int i = 0; i < vert_count; i++) { sx += h[i].x; sy += h[i].y; sz += h[i].z; }
    out3[0] = sx; out3[1] = sy; out3[2] = sz;
    CUDA_SAFE_CALL(cudaFree(d_grad));
}

// ============================================================================
// Contact-pair "clean export layer"
// ----------------------------------------------------------------------------
// The solver stores collision pairs as MMCVID int4 (sign-packed {type,
// degeneracy} state, inherited from GIPC). That packing is load-bearing inside
// the contact kernels, so we DON'T touch it. Instead, anything OUTSIDE the
// solver (contact sensor, per-pair force, debug viz) goes through this single
// decode into a clean int4 of plain vertex indices ({v0,v1,v2,v3}, -1 padded
// for PP/PE), UIPC-style. The decode runs ON the readback path only — never in
// engine.step() — so it adds zero cost to training / batched solves.
//
// Sign-encoding decoded here (mirrors the barrier-gradient kernels):
//   .x >= 0            -> EE: {.x, .y, .z, (.w>=0? .w : -.w-1)}
//   .x <  0 (v0=-.x-1) -> .z<0: (.y<0 ? {v0,-.y-1,-.z-1,-.w-1} : PP {v0,.y,-1,-1})
//                         .w<0: (.y<0 ? {v0,-.y-1,.z,-.w-1}    : PE {v0,.y,.z,-1})
//                         else: PT {v0,.y,.z,.w}
__host__ __device__ static int4 _decode_pair_clean(int4 m)
{
    int v0, v1, v2, v3;
    v0 = v1 = v2 = v3 = -1;
    if(m.x >= 0)
    {
        v0 = m.x; v1 = m.y; v2 = m.z; v3 = (m.w >= 0) ? m.w : (-m.w - 1);
    }
    else
    {
        int p0 = -m.x - 1;
        if(m.z < 0)
        {
            if(m.y < 0) { v0 = p0; v1 = -m.y - 1; v2 = -m.z - 1; v3 = -m.w - 1; }
            else        { v0 = p0; v1 = m.y; }
        }
        else if(m.w < 0)
        {
            if(m.y < 0) { v0 = p0; v1 = -m.y - 1; v2 = m.z; v3 = -m.w - 1; }
            else        { v0 = p0; v1 = m.y; v2 = m.z; }
        }
        else { v0 = p0; v1 = m.y; v2 = m.z; v3 = m.w; }
    }
    return make_int4(v0, v1, v2, v3);
}

__global__ static void _decodeCleanContactPairs(const int4* mmcvid, int4* clean, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
        return;
    clean[idx] = _decode_pair_clean(mmcvid[idx]);
}

int SimEngine::get_collision_pairs_clean(int* out_flat) const
{
    // Decode the current body-body collision pairs into clean vertex-index
    // 4-tuples (row-major int4 -> out_flat[4*i .. 4*i+3], -1 padded). Returns
    // the pair count. Rebuilds contacts at the current (post-step) state.
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    if(g.m_skip_all_collision)
        return 0;
    g.buildBVH();
    g.buildCP();
    int ncp = static_cast<int>(g.h_cpNum[0]);
    if(ncp < 1)
        return 0;

    int4* d_clean = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_clean, ncp * sizeof(int4)));
    int threads = 256, blocks = (ncp + threads - 1) / threads;
    _decodeCleanContactPairs<<<blocks, threads>>>(g._collisonPairs, d_clean, ncp);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(out_flat != nullptr)
        CUDA_SAFE_CALL(cudaMemcpy(out_flat, d_clean, ncp * sizeof(int4), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_clean));
    return ncp;
}

int SimEngine::get_ccd_pairs_clean(const double* move_flat,
                                   int           move_count,
                                   double        alpha,
                                   int*          out_flat) const
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    if(g.m_skip_all_collision)
        return 0;
    if(move_flat == nullptr || move_count != g.vertexNum)
        throw std::runtime_error(
            "get_ccd_pairs_clean expects one double3 motion per vertex");
    if(!std::isfinite(alpha) || alpha < 0.0)
        throw std::runtime_error(
            "get_ccd_pairs_clean alpha must be finite and non-negative");

    CUDA_SAFE_CALL(cudaMemcpy(g._moveDir,
                              move_flat,
                              static_cast<size_t>(move_count) * sizeof(double3),
                              cudaMemcpyHostToDevice));
    g.buildBVH_FULLCCD(alpha);
    g.buildFullCP(alpha);
    const int ncp = static_cast<int>(g.h_ccd_cpNum.get());
    if(ncp < 1)
        return 0;

    int4* d_clean = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_clean, ncp * sizeof(int4)));
    const int threads = 256;
    const int blocks  = (ncp + threads - 1) / threads;
    _decodeCleanContactPairs<<<blocks, threads>>>(
        g._ccd_collisonPairs, d_clean, ncp);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(out_flat != nullptr)
        CUDA_SAFE_CALL(cudaMemcpy(out_flat,
                                  d_clean,
                                  ncp * sizeof(int4),
                                  cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_clean));
    return ncp;
}

// [Step B] Compute per-contact forces into grow-only device buffers. Returns the
// contact count (h_cpNum + h_gpNum). Buffers are read-only views valid until the
// next call; fetch device pointers via contacts_pair_ptr()/contacts_force_ptr().
// Call AFTER step().
int SimEngine::compute_contacts(bool rebuild)
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    impl.contact_count = 0;
    if(g.m_skip_all_collision || g.vertexNum <= 0)
        return 0;
    // By default REUSE the contact set the solver already built during the last
    // step() (BVH + collision pairs persist in _collisonPairs / _environment_
    // collisionPair with counts h_cpNum/h_gpNum) — like UIPC, where contacts are
    // queryable after world.advance() without a rebuild. Pass rebuild=true only
    // when calling outside the post-step window.
    if(rebuild)
    {
        g.buildBVH();
        g.buildCP();
    }
    int n = (int)g.h_cpNum[0] + (int)g.h_gpNum;
    if(n <= 0)
        return 0;
    if(n > impl.contact_cap)
    {
        if(impl.d_contact_pair)  CUDA_SAFE_CALL(cudaFree(impl.d_contact_pair));
        if(impl.d_contact_force) CUDA_SAFE_CALL(cudaFree(impl.d_contact_force));
        impl.contact_cap = n + n / 2 + 64;  // grow with slack
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_contact_pair, impl.contact_cap * sizeof(int2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_contact_force, impl.contact_cap * sizeof(double3)));
    }
    impl.contact_count = g.exportContacts(impl.d_contact_pair, impl.d_contact_force);
    return impl.contact_count;
}

uintptr_t SimEngine::contacts_pair_ptr() const
{
    return reinterpret_cast<uintptr_t>(m_impl->d_contact_pair);
}

uintptr_t SimEngine::contacts_force_ptr() const
{
    return reinterpret_cast<uintptr_t>(m_impl->d_contact_force);
}

void SimEngine::get_pair_contact_force(int a_off, int a_cnt, int b_off, int b_cnt,
                                       double* out3) const
{
    // Net IPC contact (barrier) force on body A FROM body B: the barrier
    // gradient summed over A's vertices, restricted to collision pairs that
    // connect A's vertex range [a_off, a_off+a_cnt) and B's [b_off, b_off+b_cnt).
    // Used to populate the contact sensor's per-partner force_matrix_w. Call
    // AFTER step(). Returns the raw IP-scaled gradient (apply -1/dt^2 + sign on
    // the Python side, same convention as get_body_contact_force).
    out3[0] = out3[1] = out3[2] = 0.0;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || a_off < 0 || a_cnt <= 0 || b_off < 0 || b_cnt <= 0)
        return;
    if(a_off + a_cnt > nv || b_off + b_cnt > nv)
        return;
    if(g.m_skip_all_collision)
        return;

    g.buildBVH();
    g.buildCP();
    int ncp = static_cast<int>(g.h_cpNum[0]);
    if(ncp < 1)
        return;  // no body-body pairs (ground-only contact has no partner body)

    std::vector<int4> h_pairs(ncp);
    CUDA_SAFE_CALL(cudaMemcpy(h_pairs.data(), g._collisonPairs,
                              ncp * sizeof(int4), cudaMemcpyDeviceToHost));

    // Membership test on the CLEAN decode (plain vertex indices); the gradient
    // re-run below keeps the ORIGINAL MMCVID so its {type,degeneracy} flags are
    // preserved. Clean decode is the single source of truth (_decode_pair_clean).
    std::vector<int4> filtered;
    filtered.reserve(ncp);
    for(int i = 0; i < ncp; i++)
    {
        int4 c = _decode_pair_clean(h_pairs[i]);
        int  verts[4] = {c.x, c.y, c.z, c.w};
        bool inA = false, inB = false;
        for(int k = 0; k < 4; k++)
        {
            int vv = verts[k];
            if(vv < 0) continue;  // -1 padding (PP/PE unused slots)
            if(vv >= a_off && vv < a_off + a_cnt) inA = true;
            if(vv >= b_off && vv < b_off + b_cnt) inB = true;
        }
        if(inA && inB)
            filtered.push_back(h_pairs[i]);
    }
    if(filtered.empty())
        return;

    int4* d_filtered = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_filtered, filtered.size() * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMemcpy(d_filtered, filtered.data(),
                              filtered.size() * sizeof(int4), cudaMemcpyHostToDevice));

    double3* d_grad = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_grad, nv * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_grad, 0, nv * sizeof(double3)));

    // Reuse the existing barrier-gradient kernel on just the filtered pairs by
    // temporarily pointing GIPC at our subset (synchronous readback context).
    int4*    saved_pairs = g._collisonPairs;
    uint32_t saved_cpNum = g.h_cpNum[0];
    g._collisonPairs = d_filtered;
    g.h_cpNum.set(0, static_cast<uint32_t>(filtered.size()));  // [3b] export-time override
    g.zeroBinnedGrad();                      // [4.3] calBarrierGradient scatters to binned buf
    g.calBarrierGradient(d_grad, g.Kappa);
    g.combineBinnedGrad(d_grad);             // [4.3] fold binned contact force into d_grad
    g._collisonPairs = saved_pairs;
    g.h_cpNum.set(0, saved_cpNum);  // [3b] restore
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<double3> hg(a_cnt);
    CUDA_SAFE_CALL(cudaMemcpy(hg.data(), d_grad + a_off,
                              a_cnt * sizeof(double3), cudaMemcpyDeviceToHost));
    double sx = 0, sy = 0, sz = 0;
    for(int i = 0; i < a_cnt; i++) { sx += hg[i].x; sy += hg[i].y; sz += hg[i].z; }
    out3[0] = sx; out3[1] = sy; out3[2] = sz;
    CUDA_SAFE_CALL(cudaFree(d_filtered));
    CUDA_SAFE_CALL(cudaFree(d_grad));
}

// [force-control / GPU gate] On-device MAX stitch-spring stretch over a range of
// stitch springs [start, start+count). stretch_j = |vert[targetInd[j]] -
// vert[paired[j]]|. Single-block shared-memory max reduction; returns one scalar
// — so a force-gated gripper controller can read the grip signal WITHOUT a full
// vertex-array D2H (only 8 bytes come back). All work stays on the GPU.
__global__ static void _stitch_max_stretch_kernel(const double3* verts,
                                                  const uint32_t* targetInd,
                                                  const int* paired,
                                                  int start, int count,
                                                  double* out)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for(int j = tid; j < count; j += blockDim.x)
    {
        int idx = start + j;
        int ai  = paired[idx];
        if(ai < 0) continue;                 // -1 = functor target, not a stitch pair
        uint32_t fi = targetInd[idx];
        double3 a = verts[fi];
        double3 b = verts[ai];
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        double d  = sqrt(dx * dx + dy * dy + dz * dz);
        local = fmax(local, d);
    }
    sdata[tid] = local;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if(tid == 0) out[0] = sdata[0];
}

double SimEngine::get_stitch_max_stretch(int pair_start, int pair_count) const
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   sn   = static_cast<int>(g.softNum);
    if(pair_count <= 0 || pair_start < 0 || pair_start + pair_count > sn)
        return 0.0;
    if(g._vertexes == nullptr || g.targetInd == nullptr
       || g.m_d_stitch_paired_vertex == nullptr)
        return 0.0;
    if(impl.d_stitch_scalar_out == nullptr)
        CUDA_SAFE_CALL(
            cudaMalloc(&impl.d_stitch_scalar_out, sizeof(double)));
    _stitch_max_stretch_kernel<<<1, 256>>>(g._vertexes, g.targetInd,
                                           g.m_d_stitch_paired_vertex,
                                           pair_start, pair_count,
                                           impl.d_stitch_scalar_out);
    double h = 0.0;
    CUDA_SAFE_CALL(cudaMemcpy(&h,
                              impl.d_stitch_scalar_out,
                              sizeof(double),
                              cudaMemcpyDeviceToHost));
    return h;
}

// [force-control / GPU gate — BATCHED] One block PER segment: block s reduces the
// stitch springs [starts[s], starts[s]+counts[s]) to one max stretch -> out[s].
// A "segment" is one finger (and, for multi-env, one finger of one env). Blocks
// are independent — each writes only its own out[s], with NO cross-segment shared
// state or atomics — so different fingers/ENVS cannot interfere with each other,
// and the whole batch is ONE kernel launch (no per-finger launch latency).
__global__ static void _stitch_max_stretch_batched_kernel(const double3* verts,
                                                          const uint32_t* targetInd,
                                                          const int* paired,
                                                          const int* starts,
                                                          const int* counts,
                                                          double* out)
{
    int seg   = blockIdx.x;
    int start = starts[seg];
    int count = counts[seg];
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for(int j = tid; j < count; j += blockDim.x)
    {
        int idx = start + j;
        int ai  = paired[idx];
        if(ai < 0) continue;
        uint32_t fi = targetInd[idx];
        double3 a = verts[fi];
        double3 b = verts[ai];
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        local = fmax(local, sqrt(dx * dx + dy * dy + dz * dz));
    }
    sdata[tid] = local;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if(tid == 0) out[seg] = sdata[0];
}

void SimEngine::get_stitch_max_stretch_batched(const int* h_starts,
                                               const int* h_counts,
                                               int n_seg, double* h_out) const
{
    for(int i = 0; i < n_seg; i++) h_out[i] = 0.0;
    if(n_seg <= 0) return;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    if(g._vertexes == nullptr || g.targetInd == nullptr
       || g.m_d_stitch_paired_vertex == nullptr)
        return;
    // Instance-owned, grow-only buffers. Output reserves three doubles per
    // segment so this storage is shared with the batched contact-force getter.
    if(n_seg > impl.segment_cap)
    {
        int*    starts = nullptr;
        int*    counts = nullptr;
        double* output = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&starts, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&counts, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&output, n_seg * 3 * sizeof(double)));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_starts));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_counts));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_output));
        impl.d_segment_starts = starts;
        impl.d_segment_counts = counts;
        impl.d_segment_output = output;
        impl.segment_cap      = n_seg;
    }
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_segment_starts,
                              h_starts,
                              n_seg * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_segment_counts,
                              h_counts,
                              n_seg * sizeof(int),
                              cudaMemcpyHostToDevice));
    _stitch_max_stretch_batched_kernel<<<n_seg, 256>>>(
        g._vertexes, g.targetInd, g.m_d_stitch_paired_vertex,
        impl.d_segment_starts,
        impl.d_segment_counts,
        impl.d_segment_output);
    CUDA_SAFE_CALL(cudaMemcpy(h_out,
                              impl.d_segment_output,
                              n_seg * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

// [force-control — BATCHED] One block PER segment (= one gripper finger, and for
// multi-env one finger of one env): block s sums the per-vertex contact force over
// [offsets[s], offsets[s]+counts[s]) -> out[s*3 .. s*3+2]. Blocks are independent
// (each writes only its own out), so envs cannot interfere. The expensive contact
// rebuild (buildBVH+buildCP+calBarrierGradient) is done ONCE by the caller, then
// ALL segments are summed in this ONE launch with ONE D2H — vs the single-body
// get_vertex_contact_force_sum which rebuilds contacts on EVERY call (N D2H syncs +
// N full contact rebuilds per frame at scale).
__global__ static void _contact_force_sum_batched_kernel(const double3* grad, int nv,
                                                         const int* offsets,
                                                         const int* counts,
                                                         double* out /* n_seg*3 */)
{
    int seg = blockIdx.x;
    int off = offsets[seg];
    int cnt = counts[seg];
    __shared__ double sx[256];
    __shared__ double sy[256];
    __shared__ double sz[256];
    int tid = threadIdx.x;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    for(int j = tid; j < cnt; j += blockDim.x)
    {
        int vi = off + j;
        if(vi < 0 || vi >= nv) continue;
        double3 v = grad[vi];
        lx += v.x; ly += v.y; lz += v.z;
    }
    sx[tid] = lx; sy[tid] = ly; sz[tid] = lz;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) { sx[tid] += sx[tid + s]; sy[tid] += sy[tid + s]; sz[tid] += sz[tid + s]; }
        __syncthreads();
    }
    if(tid == 0) { out[seg * 3 + 0] = sx[0]; out[seg * 3 + 1] = sy[0]; out[seg * 3 + 2] = sz[0]; }
}

void SimEngine::get_body_contact_force_batched(const int* h_offsets,
                                               const int* h_counts,
                                               int n_seg, double* h_out3) const
{
    // h_out3 holds n_seg 3-vectors (net IPC contact force per segment/finger).
    for(int i = 0; i < n_seg * 3; i++) h_out3[i] = 0.0;
    if(n_seg <= 0) return;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || g.m_skip_all_collision) return;

    // Rebuild contacts ONCE at the current (post-step) state, then sum every
    // segment from the single barrier-gradient buffer.
    g.buildBVH();
    g.buildCP();
    if(g.h_cpNum[0] < 1) return;   // nothing in contact -> all zeros

    if(nv > impl.contact_gradient_cap)
    {
        double3* gradient = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&gradient, nv * sizeof(double3)));
        CUDA_SAFE_CALL(cudaFree(impl.d_contact_gradient));
        impl.d_contact_gradient  = gradient;
        impl.contact_gradient_cap = nv;
    }
    CUDA_SAFE_CALL(
        cudaMemset(impl.d_contact_gradient, 0, nv * sizeof(double3)));
    g.zeroBinnedGrad();                          // [4.3] calBarrierGradient scatters to binned buf
    g.calBarrierGradient(impl.d_contact_gradient, g.Kappa);
    g.combineBinnedGrad(impl.d_contact_gradient);

    if(n_seg > impl.segment_cap)
    {
        int*    offsets = nullptr;
        int*    counts  = nullptr;
        double* output  = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&offsets, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&counts, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&output, n_seg * 3 * sizeof(double)));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_starts));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_counts));
        CUDA_SAFE_CALL(cudaFree(impl.d_segment_output));
        impl.d_segment_starts = offsets;
        impl.d_segment_counts = counts;
        impl.d_segment_output = output;
        impl.segment_cap      = n_seg;
    }
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_segment_starts,
                              h_offsets,
                              n_seg * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_segment_counts,
                              h_counts,
                              n_seg * sizeof(int),
                              cudaMemcpyHostToDevice));
    _contact_force_sum_batched_kernel<<<n_seg, 256>>>(
        impl.d_contact_gradient,
        nv,
        impl.d_segment_starts,
        impl.d_segment_counts,
        impl.d_segment_output);
    CUDA_SAFE_CALL(cudaMemcpy(h_out3,
                              impl.d_segment_output,
                              n_seg * 3 * sizeof(double),
                              cudaMemcpyDeviceToHost));
}

int SimEngine::get_vertex_contact_forces(double* out3, int n, bool include_ground,
                                         int components)
{
    // [contact-force distribution] identical contact rebuild as
    // get_body_contact_force_batched, but returns the UNsummed per-vertex
    // buffer — the image-style force-distribution feedback.
    // components: 0 = normal (barrier [+ground]) only — historic behavior;
    //             1 = friction_lagged only — the friction-potential gradient
    //                 the solver ACTUALLY used this step (positions current,
    //                 lambda/tangent basis lagged one step by construction);
    //             2 = total = normal + friction_lagged.
    // The lagged-friction path reads the immutable lastH set — it must NOT
    // rebuild friction sets (that would perturb the next solve's state).
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    int   nw   = std::min(n, nv);
    if(nw <= 0)
        return 0;
    memset(out3, 0, (size_t)nw * 3 * sizeof(double));
    if(g.m_skip_all_collision)
        return nw;
    const bool want_normal   = (components == 0 || components == 2);
    const bool want_friction = (components == 1 || components == 2);
    const bool have_friction = (g.h_cpNum_last[0] > 0 || g.h_gpNum_last > 0);

    if(want_normal)
    {
        g.buildBVH();
        g.buildCP();
    }
    // [audit] the old early-return must not swallow a friction-only request:
    // lagged friction can exist while the CURRENT normal set is empty.
    if((!want_normal || (g.h_cpNum[0] < 1 && !(include_ground && g.h_gpNum > 0)))
       && !(want_friction && have_friction))
        return nw;   // nothing requested is present -> all zeros

    if(nv > impl.contact_gradient_cap)
    {
        double3* gradient = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&gradient, nv * sizeof(double3)));
        CUDA_SAFE_CALL(cudaFree(impl.d_contact_gradient));
        impl.d_contact_gradient  = gradient;
        impl.contact_gradient_cap = nv;
    }
    CUDA_SAFE_CALL(
        cudaMemset(impl.d_contact_gradient, 0, nv * sizeof(double3)));
    g.zeroBinnedGrad();
    if(want_normal)
    {
        g.calBarrierGradient(impl.d_contact_gradient, g.Kappa);
        if(include_ground)
            g.computeGroundGradient(impl.d_contact_gradient, g.Kappa);
    }
    if(want_friction && have_friction)
        g.calFrictionGradient(impl.d_contact_gradient,
                              impl.d_tetMesh);  // lastH set, read-only
    g.combineBinnedGrad(impl.d_contact_gradient);
    std::vector<double3> engine_order(nv);
    CUDA_SAFE_CALL(cudaMemcpy(engine_order.data(),
                              impl.d_contact_gradient,
                              (size_t)nv * sizeof(double3),
                              cudaMemcpyDeviceToHost));
    // The buffer holds the incremental-potential GRADIENT (dE/dx = -force*dt^2).
    // Physical contact force = -gradient/dt^2 (same convention as the
    // per-contact force magnitude path, GIPC.cu _calBarrierForces). The first
    // release scaled by +1/dt^2, flipping every vector; the resting-cube
    // regression used |Fy| and hid it. Verified post-fix: resting cube net
    // vertical force = +mg (upward support).
    const double neg_inv_dt2 = -1.0 / (g.IPC_dt * g.IPC_dt);
    const auto& perm = impl.tetMesh.vertex_metis_to_input;
    if(perm.empty() || static_cast<int>(perm.size()) < nv)
    {
        for(int i = 0; i < nw; ++i)
        {
            out3[3 * i + 0] = engine_order[i].x * neg_inv_dt2;
            out3[3 * i + 1] = engine_order[i].y * neg_inv_dt2;
            out3[3 * i + 2] = engine_order[i].z * neg_inv_dt2;
        }
    }
    else
    {
        // Match get_vertex_positions(): per-vertex force is user/input ordered.
        for(int engine_index = 0; engine_index < nv; ++engine_index)
        {
            const int input_index = perm[engine_index];
            if(input_index < 0 || input_index >= nw)
                continue;
            out3[3 * input_index + 0] =
                engine_order[engine_index].x * neg_inv_dt2;
            out3[3 * input_index + 1] =
                engine_order[engine_index].y * neg_inv_dt2;
            out3[3 * input_index + 2] =
                engine_order[engine_index].z * neg_inv_dt2;
        }
    }
    return nw;
}

// [FEM stress] configured per-tet constitutive law -> first Piola -> Cauchy
// -> von Mises. Each compile-time model branch must use the same P(F) as its
// solver energy/gradient path; a generic fallback would silently report a
// different material law.
__global__ void _fem_tet_von_mises(const double3* verts, const uint4* tets,
                                   const __GEIGEN__::Matrix3x3d* DmInv,
                                   const double* lengthRate, const double* volumeRate,
                                   double* tet_vm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    const uint4   t  = tets[i];
    const double3 x0 = verts[t.x], x1 = verts[t.y], x2 = verts[t.z], x3 = verts[t.w];
    double Ds[3][3] = {{x1.x - x0.x, x2.x - x0.x, x3.x - x0.x},
                       {x1.y - x0.y, x2.y - x0.y, x3.y - x0.y},
                       {x1.z - x0.z, x2.z - x0.z, x3.z - x0.z}};
    const auto& Di = DmInv[i];
    double F[3][3];
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            F[r][c] = Ds[r][0] * Di.m[0][c] + Ds[r][1] * Di.m[1][c] + Ds[r][2] * Di.m[2][c];
    const double J = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
                     - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
                     + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
    if(!(J > 1e-12) || !isfinite(J))
    {   // [vM constitutive-consistency] inverted/degenerate element: report NaN,
        // NOT 0 — a silent zero hides the WORST element in the body (audit
        // leftover #2). Downstream: filter with isfinite() / treat NaN as
        // "element state invalid for stress readout".
        tet_vm[i] = nan("");
        return;
    }
#ifdef USE_SNK1
    // [vM constitutive-consistency] use the SAME P(F) as the solver's energy
    // (femEnergy.cu __computePEPF_StableNHK3D1_double, active under USE_SNK1):
    //   P = u*F + r*(J - 1 - u/r)*cof(F),  u = lengthRate, r = volumeRate
    // Cauchy sigma = P F^T / J. The previous exporter recovered standard-NHK
    // (mu, lambda) and used log(J) — a DIFFERENT constitutive law from what the
    // solver minimizes, so exported magnitudes were systematically off.
    {
        const double u = lengthRate[i], r_ = volumeRate[i];
        double cof[3][3];
        cof[0][0] = F[1][1]*F[2][2] - F[1][2]*F[2][1];
        cof[0][1] = F[1][2]*F[2][0] - F[1][0]*F[2][2];
        cof[0][2] = F[1][0]*F[2][1] - F[1][1]*F[2][0];
        cof[1][0] = F[2][1]*F[0][2] - F[2][2]*F[0][1];
        cof[1][1] = F[2][2]*F[0][0] - F[2][0]*F[0][2];
        cof[1][2] = F[2][0]*F[0][1] - F[2][1]*F[0][0];
        cof[2][0] = F[0][1]*F[1][2] - F[1][1]*F[0][2];
        cof[2][1] = F[0][2]*F[1][0] - F[0][0]*F[1][2];
        cof[2][2] = F[0][0]*F[1][1] - F[0][1]*F[1][0];
        const double w = r_ * (J - 1.0 - u / r_);
        double P[3][3], s[3][3];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                P[r][c] = u * F[r][c] + w * cof[r][c];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                s[r][c] = (P[r][0]*F[c][0] + P[r][1]*F[c][1] + P[r][2]*F[c][2]) / J;
        const double tr3s = (s[0][0] + s[1][1] + s[2][2]) / 3.0;
        s[0][0] -= tr3s; s[1][1] -= tr3s; s[2][2] -= tr3s;
        double dd2 = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                dd2 += s[r][c] * s[r][c];
        tet_vm[i] = sqrt(1.5 * dd2);
        return;
    }
#elif defined(USE_SNK2)
    // Use the same SNK2 first-Piola law as the energy/gradient path, then
    // convert sigma = P F^T / J. The previous generic standard-NHK fallback
    // reported stress from a different constitutive law.
    {
        __GEIGEN__::Matrix3x3d Fg;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                Fg.m[r][c] = F[r][c];
        const double I2 = __GEIGEN__::__squaredNorm(Fg);
        const auto Pg = __computePEPF_StableNHK3D2_double(
            Fg, I2, J, lengthRate[i], volumeRate[i]);
        double s[3][3];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                s[r][c] = (Pg.m[r][0] * F[c][0]
                           + Pg.m[r][1] * F[c][1]
                           + Pg.m[r][2] * F[c][2]) / J;
        const double tr3 = (s[0][0] + s[1][1] + s[2][2]) / 3.0;
        s[0][0] -= tr3; s[1][1] -= tr3; s[2][2] -= tr3;
        double dd = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                dd += s[r][c] * s[r][c];
        tet_vm[i] = sqrt(1.5 * dd);
        return;
    }
#elif defined(USE_ARAP)
    // ARAP P(F)=lengthRate*(F-R), with R from the same QR-SVD routine used by
    // the solver. Do not reuse a neo-Hookean stress formula here.
    {
        Eigen::Matrix<double, 3, 3> Fm, U, V;
        Eigen::Matrix<double, 3, 1> sigma;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                Fm(r, c) = F[r][c];
        __GEIGEN__::math::qr_svd(Fm, sigma, U, V);
        const auto P = computePEPF_ARAP_double(Fm, U, V, lengthRate[i]);
        double s[3][3];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                s[r][c] = (P(r, 0) * F[c][0]
                           + P(r, 1) * F[c][1]
                           + P(r, 2) * F[c][2]) / J;
        const double tr3 = (s[0][0] + s[1][1] + s[2][2]) / 3.0;
        s[0][0] -= tr3; s[1][1] -= tr3; s[2][2] -= tr3;
        double dd = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                dd += s[r][c] * s[r][c];
        tet_vm[i] = sqrt(1.5 * dd);
        return;
    }
#else
#error "Von Mises export requires a configured tetrahedral constitutive model"
#endif
}

__device__ inline void _se_atomicMaxPosDouble(double* addr, double val)
{
    // positive doubles compare correctly as int64 bit patterns
    atomicMax((unsigned long long*)addr, __double_as_longlong(val));
}

__global__ void _fem_scatter_vm_to_verts(const uint4* tets, const double* tet_vm,
                                         double* vert_vm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    const uint4  t = tets[i];
    const double v = tet_vm[i];
    _se_atomicMaxPosDouble(&vert_vm[t.x], v);
    _se_atomicMaxPosDouble(&vert_vm[t.y], v);
    _se_atomicMaxPosDouble(&vert_vm[t.z], v);
    _se_atomicMaxPosDouble(&vert_vm[t.w], v);
}

int SimEngine::get_fem_von_mises_stress(double* out, int n)
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    int   tet_offset =
        static_cast<int>(g.abd_fem_count_info.fem_tet_offset);
    int   nt = static_cast<int>(g.abd_fem_count_info.fem_tet_num);
    int   nw   = std::min(n, nv);
    if(nw <= 0)
        return 0;
    memset(out, 0, (size_t)nw * sizeof(double));
    if(nt <= 0)
        return nw;
    if(nt > impl.stress_tet_cap)
    {
        double* stress = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&stress, nt * sizeof(double)));
        CUDA_SAFE_CALL(cudaFree(impl.d_stress_tet));
        impl.d_stress_tet   = stress;
        impl.stress_tet_cap = nt;
    }
    if(nv > impl.stress_vertex_cap)
    {
        double* stress = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&stress, nv * sizeof(double)));
        CUDA_SAFE_CALL(cudaFree(impl.d_stress_vertex));
        impl.d_stress_vertex   = stress;
        impl.stress_vertex_cap = nv;
    }
    CUDA_SAFE_CALL(
        cudaMemset(impl.d_stress_vertex, 0, nv * sizeof(double)));
    const int bs = 256;
    _fem_tet_von_mises<<<(nt + bs - 1) / bs, bs>>>(
        impl.d_tetMesh.vertexes,
        impl.d_tetMesh.tetrahedras + tet_offset,
        impl.d_tetMesh.DmInverses + tet_offset,
        impl.d_tetMesh.lengthRate + tet_offset,
        impl.d_tetMesh.volumeRate + tet_offset,
        impl.d_stress_tet,
        nt);
    _fem_scatter_vm_to_verts<<<(nt + bs - 1) / bs, bs>>>(
        impl.d_tetMesh.tetrahedras + tet_offset,
        impl.d_stress_tet,
        impl.d_stress_vertex,
        nt);
    std::vector<double> engine_order(nv);
    CUDA_SAFE_CALL(cudaMemcpy(engine_order.data(),
                              impl.d_stress_vertex,
                              (size_t)nv * sizeof(double),
                              cudaMemcpyDeviceToHost));
    const auto& perm = impl.tetMesh.vertex_metis_to_input;
    if(perm.empty() || static_cast<int>(perm.size()) < nv)
    {
        std::copy_n(engine_order.begin(), nw, out);
    }
    else
    {
        // Match get_vertex_positions(): return values indexed in input order.
        for(int engine_index = 0; engine_index < nv; ++engine_index)
        {
            const int input_index = perm[engine_index];
            if(input_index >= 0 && input_index < nw)
                out[input_index] = engine_order[engine_index];
        }
    }
    return nw;
}

double SimEngine::get_prismatic_current_distance(int idx) const
{
    // [force-control] Current opening d = (Cq - Cp).dot(t) along the joint axis.
    // Lets a controller (or diagnostics) see the actual gripper opening — e.g.
    // to confirm a force-limited grasp does NOT fully close (stops at the object
    // width).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
        return 0.0;
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving)
        return 0.0;

    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    Vec12 qp, qc;
    CUDA_SAFE_CALL(cudaMemcpy(&qp, q_buf.data() + drv.parent_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&qc, q_buf.data() + drv.child_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));

    auto worldpt = [](const Vec12& q, const Vector3& xb) -> Vector3 {
        Matrix3x3 A;
        A.row(0) = q.segment<3>(3).transpose();
        A.row(1) = q.segment<3>(6).transpose();
        A.row(2) = q.segment<3>(9).transpose();
        return Vector3(q.segment<3>(0) + A * xb);
    };
    Matrix3x3 Ac;
    Ac.row(0) = qc.segment<3>(3).transpose();
    Ac.row(1) = qc.segment<3>(6).transpose();
    Ac.row(2) = qc.segment<3>(9).transpose();
    Vector3 Cp = worldpt(qp, drv.Cp_bar);
    Vector3 Cq = worldpt(qc, drv.Cq_bar);
    Vector3 t  = Ac * drv.tq_bar;
    return (Cq - Cp).dot(t);
}

void SimEngine::set_prismatic_limit_barrier(int idx, double cl, double dir,
                                            double dhat, double kappa, int slot)
{
    // [force-control] Arm a one-sided IPC barrier on prismatic joint idx at the
    // CLOSED coordinate cl: the solver then NEVER lets d cross cl regardless of
    // the (force) drive — a hard no-overshoot guarantee while staying pure-force.
    // dir = +1 if the open end is at d>cl, else -1. kappa<=0 disarms. Written
    // directly to the GPU struct (the per-step target sync leaves these fields).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system) {
        std::cerr << "[set_prismatic_limit_barrier] sim not finalized" << std::endl;
        return;
    }
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving) {
        std::cerr << "[set_prismatic_limit_barrier] idx " << idx
                  << " out of range [0," << sys.m_num_prismatic_driving << ")" << std::endl;
        return;
    }
    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));
    if(slot == 0) {                              // slot 0 = closed-end barrier
        drv.limit_cl    = static_cast<Float>(cl);
        drv.limit_dir   = (dir >= 0.0) ? Float(1) : Float(-1);
        drv.limit_dhat  = static_cast<Float>(dhat);
        drv.limit_kappa = static_cast<Float>(kappa);
    } else {                                     // slot 1 = open-end barrier
        drv.limit_cl2    = static_cast<Float>(cl);
        drv.limit_dir2   = (dir >= 0.0) ? Float(1) : Float(-1);
        drv.limit_dhat2  = static_cast<Float>(dhat);
        drv.limit_kappa2 = static_cast<Float>(kappa);
    }
    CUDA_SAFE_CALL(cudaMemcpy(sys.m_prismatic_driving_data.data() + idx, &drv,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyHostToDevice));
}

void SimEngine::get_revolute_current_angles(double* out, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
    {
        for(int i = 0; i < count; i++)
            out[i] = 0.0;
        return;
    }

    int n = std::min(count, static_cast<int>(impl.tetMesh.joint_angle_controls.size()));
    if(n == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0)
    {
        for(int i = 0; i < n; i++)
            out[i] = 0.0;
        return;
    }

    // Read current ABD state vectors (q) from GPU
    // Vector12 = Eigen::Matrix<double, 12, 1>
    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(), q_buf.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    // Read revolute driving GPU data
    int n_drv = impl.ipc.m_abd_system->m_num_revolute_driving;
    if(n_drv == 0)
    {
        for(int i = 0; i < n; i++)
            out[i] = 0.0;
        return;
    }

    std::vector<RevoluteDrivingGPUData> host_drv(n_drv);
    CUDA_SAFE_CALL(cudaMemcpy(host_drv.data(),
                              impl.ipc.m_abd_system->m_revolute_driving_data.data(),
                              n_drv * sizeof(RevoluteDrivingGPUData),
                              cudaMemcpyDeviceToHost));

    for(int i = 0; i < n && i < n_drv; i++)
    {
        auto& drv = host_drv[i];
        auto& q1  = host_q[drv.parent_body_id];
        auto& q2  = host_q[drv.child_body_id];

        // Extract 3x3 rotation from q = [p(3), a1(3), a2(3), a3(3)]
        Eigen::Matrix3d A1, A2;
        A1.row(0) = q1.segment<3>(3).transpose();
        A1.row(1) = q1.segment<3>(6).transpose();
        A1.row(2) = q1.segment<3>(9).transpose();
        A2.row(0) = q2.segment<3>(3).transpose();
        A2.row(1) = q2.segment<3>(6).transpose();
        A2.row(2) = q2.segment<3>(9).transpose();

        // gipc::Vector3 is Eigen::Matrix<double,3,1>, same as Eigen::Vector3d
        Eigen::Vector3d p  = A1 * drv.p_bar;
        Eigen::Vector3d pN = A1 * drv.pN_bar;
        Eigen::Vector3d q_d  = A2 * drv.q_bar;
        Eigen::Vector3d qN = A2 * drv.qN_bar;

        double cos_theta = 0.5 * (p.dot(q_d) + pN.dot(qN));
        double sin_theta = 0.5 * (q_d.dot(pN) - qN.dot(p));

        out[i] = std::atan2(sin_theta, cos_theta);
    }
}

void SimEngine::get_revolute_initial_offsets(double* out, int count) const
{
    auto& ctrls = m_impl->tetMesh.joint_angle_controls;
    int   n     = std::min(count, static_cast<int>(ctrls.size()));
    for(int i = 0; i < n; i++)
        out[i] = ctrls[i].initial_angle_offset;
    for(int i = n; i < count; i++)
        out[i] = 0.0;
}

// ======================== load_mesh_from_data ========================
void SimEngine::load_mesh_from_data(const double*          vertices,
                                    int                    num_verts,
                                    const int*             faces,
                                    int                    num_faces,
                                    int                    verts_per_face,
                                    int                    dimensions,
                                    int                    body_type,
                                    const Eigen::Matrix4d& transform,
                                    double                 young_modulus,
                                    int                    boundary_type)
{
    int prev_verts = m_impl->tetMesh.vertexNum;

    // Write vertices and faces to a temporary file, then load via SimpleSceneImporter.
    // This reuses the existing loading pipeline including tetrahedralization for 3D.
    std::string tmp_dir = "/tmp/stiffgipc_mesh_data/";
    std::filesystem::create_directories(tmp_dir);
    std::string tmp_path;

    if(dimensions == 2 || verts_per_face == 3)
    {
        tmp_path = tmp_dir + "tmp_mesh_" + std::to_string(m_impl->load_records.size()) + ".obj";
        std::ofstream ofs(tmp_path);
        for(int i = 0; i < num_verts; i++)
            ofs << "v " << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << "f";
            for(int j = 0; j < verts_per_face; j++)
                ofs << " " << (faces[i * verts_per_face + j] + 1);
            ofs << "\n";
        }
        ofs.close();
    }
    else
    {
        // For pre-tetrahedralized data (verts_per_face == 4), write as .msh
        tmp_path = tmp_dir + "tmp_mesh_" + std::to_string(m_impl->load_records.size()) + ".msh";
        std::ofstream ofs(tmp_path);
        ofs << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
        ofs << "$Nodes\n" << num_verts << "\n";
        for(int i = 0; i < num_verts; i++)
            ofs << (i + 1) << " " << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << "\n";
        ofs << "$EndNodes\n$Elements\n" << num_faces << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << (i + 1) << " 4 2 0 0";
            for(int j = 0; j < 4; j++)
                ofs << " " << (faces[i * 4 + j] + 1);
            ofs << "\n";
        }
        ofs << "$EndElements\n";
        ofs.close();
    }

    auto bt = (body_type == 0) ? gipc::BodyType::ABD : gipc::BodyType::FEM;
    auto bb = (boundary_type == 1) ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;

    if(bt == gipc::BodyType::ABD && (dimensions == 2 || verts_per_face == 3))
    {
        m_impl->tetMesh.load_surfaceMesh_ABD(tmp_path, transform, young_modulus, bb);
    }
    else
    {
        SimpleSceneImporter imp;
        // Issue 1: pass runtime metis_dir to avoid the OUTPUT_DIR
        // compile-time path leak (only matters when preconditioner_type != 0).
        std::string metis_dir = m_impl->resolved_assets_dir + "sorted_mesh/";
        std::filesystem::create_directories(metis_dir);
        imp.load_geometry(m_impl->tetMesh, dimensions, bt, transform,
                          young_modulus, tmp_path,
                          m_impl->cfg.preconditioner_type, bb, metis_dir);
    }

    m_impl->record_load(body_type, prev_verts);
    m_impl->load_records.back().label = "from_data";

    std::cout << "[SimEngine] Mesh from data loaded (dim=" << dimensions
              << ", " << (body_type == 0 ? "ABD" : "FEM")
              << ", verts=" << num_verts << ", faces=" << num_faces << ")" << std::endl;
}

// ======================== Impl helpers for instanced loading ========================

std::string SimEngine::Impl::write_temp_mesh(
    const double* vertices, int num_verts,
    const int* faces, int num_faces,
    int verts_per_face, int dimensions,
    const std::string& suffix)
{
    std::string tmp_dir = "/tmp/stiffgipc_mesh_data/";
    std::filesystem::create_directories(tmp_dir);

    if(dimensions == 2 || verts_per_face == 3)
    {
        std::string tmp_path = tmp_dir + "tmp_instanced_" + suffix + ".obj";
        std::ofstream ofs(tmp_path);
        for(int i = 0; i < num_verts; i++)
            ofs << "v " << vertices[i*3] << " " << vertices[i*3+1] << " " << vertices[i*3+2] << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << "f";
            for(int j = 0; j < verts_per_face; j++)
                ofs << " " << (faces[i*verts_per_face + j] + 1);
            ofs << "\n";
        }
        ofs.close();
        return tmp_path;
    }
    else
    {
        std::string tmp_path = tmp_dir + "tmp_instanced_" + suffix + ".msh";
        std::ofstream ofs(tmp_path);
        ofs << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
        ofs << "$Nodes\n" << num_verts << "\n";
        for(int i = 0; i < num_verts; i++)
            ofs << (i+1) << " " << vertices[i*3] << " " << vertices[i*3+1] << " " << vertices[i*3+2] << "\n";
        ofs << "$EndNodes\n$Elements\n" << num_faces << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << (i+1) << " 4 2 0 0";
            for(int j = 0; j < 4; j++)
                ofs << " " << (faces[i*4 + j] + 1);
            ofs << "\n";
        }
        ofs << "$EndElements\n";
        ofs.close();
        return tmp_path;
    }
}

void SimEngine::Impl::load_from_temp_file(
    const std::string& tmp_path,
    int dimensions, int body_type, int verts_per_face,
    const Eigen::Matrix4d& transform,
    double young_modulus, int boundary_type)
{
    auto bt = (body_type == 0) ? gipc::BodyType::ABD : gipc::BodyType::FEM;
    auto bb = (boundary_type == 1) ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;

    if(bt == gipc::BodyType::ABD && (dimensions == 2 || verts_per_face == 3))
    {
        tetMesh.load_surfaceMesh_ABD(tmp_path, transform, young_modulus, bb);
    }
    else
    {
        SimpleSceneImporter imp;
        // Issue 1: pass runtime metis_dir so the MAS preconditioner path
        // doesn't trip the OUTPUT_DIR build-time path leak.
        std::string metis_dir = resolved_assets_dir + "sorted_mesh/";
        std::filesystem::create_directories(metis_dir);
        imp.load_geometry(tetMesh, dimensions, bt, transform,
                          young_modulus, tmp_path,
                          cfg.preconditioner_type, bb, metis_dir);
    }
}

// ======================== load_mesh_instanced ========================

InstancedLoadResult SimEngine::load_mesh_instanced(
    const double*                       vertices,
    int                                 num_verts,
    const int*                          faces,
    int                                 num_faces,
    int                                 verts_per_face,
    int                                 dimensions,
    int                                 body_type,
    const std::vector<Eigen::Matrix4d>& transforms,
    double                              young_modulus,
    int                                 boundary_type)
{
    int N = static_cast<int>(transforms.size());
    if(N == 0) return {};

    // 1. Register a MeshAsset (store rest topology once)
    MeshAsset asset;
    asset.asset_id      = static_cast<int>(m_impl->mesh_assets.size());
    asset.num_verts     = num_verts;
    asset.num_faces     = num_faces;
    asset.verts_per_face = verts_per_face;
    asset.dimensions    = dimensions;
    asset.body_type     = body_type;
    asset.young_modulus = young_modulus;
    asset.boundary_type = boundary_type;
    asset.rest_vertices.assign(vertices, vertices + num_verts * 3);
    asset.faces.assign(faces, faces + num_faces * verts_per_face);
    m_impl->mesh_assets.push_back(asset);

    // 2. Write temp mesh file ONCE
    std::string suffix = "asset" + std::to_string(asset.asset_id);
    std::string tmp_path = m_impl->write_temp_mesh(
        vertices, num_verts, faces, num_faces,
        verts_per_face, dimensions, suffix);

    // 3. Load N instances
    InstancedLoadResult result;
    result.asset_id = asset.asset_id;
    result.body_offsets.reserve(N);
    result.vertex_offsets.reserve(N);
    result.vertex_counts.reserve(N);

    for(int i = 0; i < N; i++)
    {
        int prev_verts = m_impl->tetMesh.vertexNum;

        m_impl->load_from_temp_file(tmp_path, dimensions, body_type,
                                    verts_per_face, transforms[i],
                                    young_modulus, boundary_type);

        m_impl->record_load(body_type, prev_verts);
        auto& rec = m_impl->load_records.back();
        rec.label       = "instanced_" + suffix + "_i" + std::to_string(i);
        rec.asset_id    = asset.asset_id;
        rec.instance_id = i;

        result.body_offsets.push_back(rec.body_offset);
        result.vertex_offsets.push_back(rec.vertex_offset);
        result.vertex_counts.push_back(rec.vertex_count);
    }

    std::cout << "[SimEngine] Instanced load: asset=" << asset.asset_id
              << ", N=" << N << ", " << (body_type == 0 ? "ABD" : "FEM")
              << ", verts_per_instance=" << num_verts
              << ", bodies=" << result.body_offsets.front()
              << ".." << result.body_offsets.back() << std::endl;

    return result;
}

// ======================== Mesh asset queries ========================

int SimEngine::get_mesh_asset_count() const
{
    return static_cast<int>(m_impl->mesh_assets.size());
}

const MeshAsset& SimEngine::get_mesh_asset(int asset_id) const
{
    return m_impl->mesh_assets.at(asset_id);
}

// ======================== FEM body count ========================
int SimEngine::get_fem_body_count() const
{
    return static_cast<int>(m_impl->tetMesh.abd_fem_count_info.fem_body_num);
}

// ======================== ABD body state access ========================
void SimEngine::get_abd_body_transforms(const int* body_offsets, double* out_mat4x4, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        double* m = out_mat4x4 + i * 16;
        std::memset(m, 0, 16 * sizeof(double));
        m[15] = 1.0;

        if(bid < 0 || bid >= num_bodies) continue;
        const auto& q = host_q[bid];

        // Translation: q[0:3] = p
        m[3]  = q[0]; m[7]  = q[1]; m[11] = q[2];
        // Rotation: A = [a1 | a2 | a3], mat[:3,:3] = A^T
        // a1 = q[3:6], a2 = q[6:9], a3 = q[9:12]
        // Row-major 4x4: m[row*4+col]
        m[0]  = q[3];  m[1]  = q[6];  m[2]  = q[9];   // row 0
        m[4]  = q[4];  m[5]  = q[7];  m[6]  = q[10];  // row 1
        m[8]  = q[5];  m[9]  = q[8];  m[10] = q[11];  // row 2
    }
}

void SimEngine::set_abd_body_transforms(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& q = host_q[bid];
        // Translation
        q[0] = m[3]; q[1] = m[7]; q[2] = m[11];
        // Rotation columns: a1=col0, a2=col1, a3=col2 of the 3x3 block
        q[3]  = m[0]; q[4]  = m[4]; q[5]  = m[8];   // a1
        q[6]  = m[1]; q[7]  = m[5]; q[8]  = m[9];   // a2
        q[9]  = m[2]; q[10] = m[6]; q[11] = m[10];  // a3
    }

    CUDA_SAFE_CALL(cudaMemcpy(impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              host_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyHostToDevice));
}

void SimEngine::teleport_abd_bodies(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    auto& abd = impl.ipc.m_abd_sim_data->device;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    size_t buf_bytes = num_bodies * sizeof(Vec12);

    // Read current q from GPU
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(), abd.body_id_to_q.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));

    // Apply new transforms
    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& q = host_q[bid];
        q[0] = m[3]; q[1] = m[7]; q[2] = m[11];
        q[3]  = m[0]; q[4]  = m[4]; q[5]  = m[8];
        q[6]  = m[1]; q[7]  = m[5]; q[8]  = m[9];
        q[9]  = m[2]; q[10] = m[6]; q[11] = m[10];
    }

    // Write the same state to q, q_prev, q_tilde, q_temp
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q.data(),       host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_prev.data(),  host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_tilde.data(), host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_temp.data(),  host_q.data(), buf_bytes, cudaMemcpyHostToDevice));

    // Zero velocity and delta-q for teleported bodies
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(), abd.body_id_to_q_v.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));
    std::vector<Vec12> host_dq(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_dq.data(), abd.body_id_to_dq.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;
        host_qv[bid] = Vec12::Zero();
        host_dq[bid] = Vec12::Zero();
    }
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_v.data(), host_qv.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_dq.data(),  host_dq.data(), buf_bytes, cudaMemcpyHostToDevice));

    // [rl-reset] q jumped, but vertex positions track q only through the
    // solve (step_forward writes vert = J*q) — nothing outside the loop
    // re-derived x, so a teleport left _vertexes at the OLD pose until the
    // next solve ran (stale getters, and any pair rebuild would bin the old
    // configuration). Re-derive the whole ABD block: bitwise-neutral for
    // untouched bodies because x ≡ J*q holds at all times (step_forward is
    // recompute-style, same J*q expression). Vertex-space o/xTilta/velocity
    // need no block writes here: xTilta is overwritten every frame start,
    // and ABD previous-state/velocity live in q_prev/q_v (both set above).
    // Then rebuild the frame-entry pair set — same contract as
    // load_checkpoint and teleport_fem_vertices: the first Newton iteration
    // of the next step runs on the inherited pair set, which after a
    // teleport belongs to the pre-teleport configuration.
    // [rl-reset] teleporting a quarantined env's body is an episode reset:
    // clear its quarantine so the scans re-adjudicate from the new pose.
    {
        const auto& groups = impl.tetMesh.body_groups;
        for(int k = 0; !groups.empty() && k < count; ++k)
            for(size_t r = 0; r < impl.load_records.size(); ++r)
            {
                const auto& rec = impl.load_records[r];
                if(rec.body_type == 0 && rec.body_offset == body_offsets[k]
                   && r < groups.size())
                    impl.ipc.reviveEnv(groups[r]);
            }
    }
    {
        GIPC& g = impl.ipc;
        if(impl.ipc.abd_fem_count_info.abd_point_num > 0)
        {
            auto abd_verts =
                muda::BufferView<double3>{impl.d_tetMesh.vertexes,
                                          (size_t)g.vertexNum}
                    .subview(impl.ipc.abd_fem_count_info.abd_point_offset,
                             impl.ipc.abd_fem_count_info.abd_point_num);
            g.m_abd_system->cal_x_from_q(*g.m_abd_sim_data, abd_verts);
        }
        if(!g.m_skip_all_collision)
        {
            g.bvh_f.invalidateRefitTopology();
            g.bvh_e.invalidateRefitTopology();
            g.buildBVH();
            g.buildCP();
        }
    }
}

void SimEngine::get_abd_body_velocities(const int* body_offsets, double* out_mat4x4, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        double* m = out_mat4x4 + i * 16;
        std::memset(m, 0, 16 * sizeof(double));

        if(bid < 0 || bid >= num_bodies) continue;
        const auto& qv = host_qv[bid];

        m[3]  = qv[0]; m[7]  = qv[1]; m[11] = qv[2];
        m[0]  = qv[3];  m[1]  = qv[6];  m[2]  = qv[9];
        m[4]  = qv[4];  m[5]  = qv[7];  m[6]  = qv[10];
        m[8]  = qv[5];  m[9]  = qv[8];  m[10] = qv[11];
    }
}

void SimEngine::set_abd_body_velocities(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& qv = host_qv[bid];
        qv[0] = m[3]; qv[1] = m[7]; qv[2] = m[11];
        qv[3]  = m[0]; qv[4]  = m[4]; qv[5]  = m[8];
        qv[6]  = m[1]; qv[7]  = m[5]; qv[8]  = m[9];
        qv[9]  = m[2]; qv[10] = m[6]; qv[11] = m[10];
    }

    CUDA_SAFE_CALL(cudaMemcpy(impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              host_qv.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyHostToDevice));
}

// ======================== FEM vertex state ========================
void SimEngine::get_vertex_velocities(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] Same convention as get_vertex_positions: output in input order.
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(out_xyz, m_impl->d_tetMesh.velocities,
                                  n * sizeof(double3), cudaMemcpyDeviceToHost));
        return;
    }
    std::vector<double3> tmp(n);
    CUDA_SAFE_CALL(cudaMemcpy(tmp.data(), m_impl->d_tetMesh.velocities,
                              n * sizeof(double3), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n) j = i;
        out_xyz[3*j + 0] = tmp[i].x;
        out_xyz[3*j + 1] = tmp[i].y;
        out_xyz[3*j + 2] = tmp[i].z;
    }
}

void SimEngine::set_vertex_positions_gpu(const double* xyz, int count)
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] xyz is in input order: xyz[3*j] = pos of input vertex j.
    // Engine internal storage is in metis-sort order (or identity).
    // For each engine vertex i, write user input at perm[i]: gpu[i] = xyz[3*perm[i]].
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->ipc._vertexes, xyz,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        return;
    }
    std::vector<double3> tmp(n);
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            tmp[i] = make_double3(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            continue;
        }
        tmp[i] = make_double3(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->ipc._vertexes, tmp.data(),
                              n * sizeof(double3), cudaMemcpyHostToDevice));
}

void SimEngine::set_vertex_velocities_gpu(const double* xyz, int count)
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // Same MAS-perm convention as set_vertex_positions_gpu.
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.velocities, xyz,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        return;
    }
    std::vector<double3> tmp(n);
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            tmp[i] = make_double3(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            continue;
        }
        tmp[i] = make_double3(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.velocities, tmp.data(),
                              n * sizeof(double3), cudaMemcpyHostToDevice));
}
