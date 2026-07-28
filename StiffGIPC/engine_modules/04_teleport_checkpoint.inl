void SimEngine::teleport_fem_vertices(const double* xyz, int count,
                                      const double* velocities)
{
    // FEM vertices live AFTER the ABD vertices in the global _vertexes buffer. The
    // caller passes FEM-only positions (count = FEM vertex count), so we MUST write at
    // the FEM offset, not at index 0. The old code wrote to _vertexes[0..n], which in a
    // multi-body scene (robot ABD verts first, then the FEM block) clobbered the first
    // ABD verts and NEVER moved the block -> the FEM block could not be reset/teleported
    // at all (it only worked in FEM-only scenes where the offset happens to be 0).
    int fem_offset = 0;
    if(!m_impl->fem_body_ranges.empty())
        fem_offset = m_impl->fem_body_ranges[0].vertex_start;
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum) - fem_offset);
    if(n <= 0) return;
    double3* p_base  = m_impl->ipc._vertexes        + fem_offset;
    double3* o_base  = m_impl->d_tetMesh.o_vertexes  + fem_offset;
    double3* xt_base = m_impl->d_tetMesh.xTilta      + fem_offset;
    double3* v_base  = m_impl->d_tetMesh.velocities   + fem_offset;

    // [MAS-perm FIX] get_vertices() returns INPUT order (it unscrambles the
    // metis sort), and callers naturally round-trip those arrays back in here.
    // The old code wrote them RAW into the engine-order buffers — on any
    // metis-sorted body (cloth under the default MAS preconditioner) that
    // SCRAMBLES the mesh: read-back mismatched by ~0.3 m on a 30x30 cloth,
    // and with a velocity field the spaghettified state crashed the next
    // solve with CUDA illegal access (both previously blamed on other
    // causes). Symmetric fix: permute input->engine before writing, exactly
    // inverse to get_vertex_positions(). perm[engine_i] = input_j, both
    // global indices; entries outside [fem_offset, fem_offset+n) fall back
    // to identity per-slot (defensive, mirrors the getter).
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    const bool  use_perm = !perm.empty()
                          && static_cast<int>(perm.size()) >= fem_offset + n;
    std::vector<double> xyz_e;          // engine-order positions
    const double*       xyz_w = xyz;    // what we actually write
    if(use_perm)
    {
        xyz_e.resize(3 * (size_t)n);
        for(int i = 0; i < n; i++)
        {
            int j = perm[fem_offset + i] - fem_offset;   // input slot for engine slot i
            if(j < 0 || j >= n) j = i;                   // defensive identity
            xyz_e[3*i+0] = xyz[3*j+0];
            xyz_e[3*i+1] = xyz[3*j+1];
            xyz_e[3*i+2] = xyz[3*j+2];
        }
        xyz_w = xyz_e.data();
    }
    // Write _vertexes (current) and o_vertexes (committed previous-step).
    CUDA_SAFE_CALL(cudaMemcpy(p_base, xyz_w, n * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(o_base, xyz_w, n * sizeof(double3), cudaMemcpyHostToDevice));

    if(velocities != nullptr)
    {
        // Preserve inertia: write v (engine order) and build
        // xTilta = x + v*dt + g*dt^2 on the host, then upload.
        double dt = m_impl->ipc.IPC_dt;
        double3 g = m_impl->ipc.gravity;
        double dt2 = dt * dt;
        std::vector<double> vel_e(3 * (size_t)n);
        for(int i = 0; i < n; i++)
        {
            int j = i;
            if(use_perm)
            {
                j = perm[fem_offset + i] - fem_offset;
                if(j < 0 || j >= n) j = i;
            }
            vel_e[3*i+0] = velocities[3*j+0];
            vel_e[3*i+1] = velocities[3*j+1];
            vel_e[3*i+2] = velocities[3*j+2];
        }
        CUDA_SAFE_CALL(cudaMemcpy(v_base, vel_e.data(),
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        std::vector<double> xTilta_host(3 * (size_t)n);
        for(int i = 0; i < n; i++)
        {
            xTilta_host[3*i+0] = xyz_w[3*i+0] + vel_e[3*i+0] * dt + g.x * dt2;
            xTilta_host[3*i+1] = xyz_w[3*i+1] + vel_e[3*i+1] * dt + g.y * dt2;
            xTilta_host[3*i+2] = xyz_w[3*i+2] + vel_e[3*i+2] * dt + g.z * dt2;
        }
        CUDA_SAFE_CALL(cudaMemcpy(xt_base, xTilta_host.data(),
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
    }
    else
    {
        // Zero velocity, xTilta = new_pos (caller declines to preserve
        // inertia; matches teleport_abd_bodies semantics).
        CUDA_SAFE_CALL(cudaMemcpy(xt_base, xyz_w,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(v_base, 0,
                                  n * sizeof(double3)));
    }

    // [rl-reset] a teleport touching a quarantined env is an episode reset:
    // clear its quarantine so the scans re-adjudicate from the NEW
    // configuration (reviveEnv is a no-op for healthy envs; a still-broken
    // reset is re-quarantined within one frame).
    {
        const auto& groups = m_impl->tetMesh.body_groups;
        for(size_t r = 0; !groups.empty() && r < m_impl->load_records.size(); ++r)
        {
            const auto& rec = m_impl->load_records[r];
            if(rec.body_type != 1 || r >= groups.size())
                continue;
            const int rel = rec.vertex_offset - fem_offset;
            if(rel < count && rel + rec.vertex_count > 0)
                m_impl->ipc.reviveEnv(groups[r]);
        }
    }

    // [rl-reset] rebuild the frame-entry pair set — same contract as
    // load_checkpoint: the first Newton iteration of the next step runs on
    // the inherited pair set, which after a teleport belongs to the
    // pre-teleport configuration (episode resets move bodies far; a stale
    // set gives the first post-reset step a garbage descent direction).
    if(!m_impl->ipc.m_skip_all_collision)
    {
        m_impl->ipc.buildBVH();
        m_impl->ipc.buildCP();
    }
}

void SimEngine::get_fem_body_vertex_range(int fem_body_idx, int* out_start, int* out_count) const
{
    if(fem_body_idx >= 0 && fem_body_idx < static_cast<int>(m_impl->fem_body_ranges.size()))
    {
        *out_start = m_impl->fem_body_ranges[fem_body_idx].vertex_start;
        *out_count = m_impl->fem_body_ranges[fem_body_idx].vertex_count;
    }
    else
    {
        *out_start = 0;
        *out_count = 0;
    }
}

// ======================== Load record tracking ========================
int SimEngine::get_load_record_count() const
{
    return static_cast<int>(m_impl->load_records.size());
}

const BodyLoadRecord& SimEngine::get_load_record(int idx) const
{
    return m_impl->load_records.at(idx);
}

}  // namespace gipc


// ======================== Per-step counters (perf debugging) ========================
namespace gipc {
int    SimEngine::get_total_newton_iters() const
{ return m_impl->ipc.m_total_newton_iters; }
int    SimEngine::get_ls_exhausted_count() const
{ return m_impl->ipc.m_ls_exhausted_total; }
int    SimEngine::get_ls_nonfinite_count() const
{ return m_impl->ipc.m_ls_nonfinite_total; }
double SimEngine::get_total_pcg_iters() const
{ return m_impl->ipc.m_total_pcg_iters; }
double SimEngine::get_total_collision_pairs() const
{ return m_impl->ipc.m_total_collision_pairs; }
double SimEngine::get_max_collision_pairs() const
{ return m_impl->ipc.m_max_collision_pairs; }
int    SimEngine::get_total_frames_done() const
{ return m_impl->ipc.m_total_frames; }
frame_fsm::FrameStatus SimEngine::get_frame_status() const
{ return m_impl->ipc.get_frame_status(); }
uint64_t SimEngine::get_total_energy_tolerance_accepts() const
{
    return m_impl->ipc.energy_tolerance_accept_count;
}

void SimEngine::save_checkpoint(const std::string& path)
{
    if(!m_impl->finalized)
        throw LifecycleError(
            "save_checkpoint() requires a finalized SimEngine");
    m_impl->ipc.save_checkpoint(m_impl->d_tetMesh, path.c_str());
}

void SimEngine::load_checkpoint(const std::string& path)
{
    if(!m_impl->finalized)
        throw LifecycleError(
            "load_checkpoint() requires a finalized SimEngine");
    m_impl->ipc.load_checkpoint(m_impl->d_tetMesh, path.c_str());
}
}  // namespace gipc

#ifdef GIPC_ENABLE_DIAGNOSTICS
namespace gipc {
std::vector<double> SimEngine::debug_fd_gradient_check(double h, int nprobes, unsigned seed)
{
    auto r = m_impl->ipc.fd_gradient_check(m_impl->d_tetMesh, h, nprobes, seed);
    return {r.max_rel, r.mean_rel, (double)r.n, (double)r.worst_v,
            (double)r.worst_axis, r.sign, r.p50, r.p95, (double)r.n_nonfinite};
}

std::vector<double> SimEngine::debug_fd_hessian_check(double h, int nprobes, unsigned seed)
{
    auto r = m_impl->ipc.fd_hessian_diagonal_check(m_impl->d_tetMesh, h, nprobes, seed);
    return {r.max_rel, r.mean_rel, (double)r.n, (double)r.worst_v,
            (double)r.worst_axis, r.sign, r.p50, r.p95, (double)r.n_nonfinite};
}

std::vector<double> SimEngine::debug_fd_activity()
{
    auto r = m_impl->ipc.fd_activity();
    return {(double)r.fem_tets, (double)r.triangles, (double)r.bending_edges,
            (double)r.soft, (double)r.contact, (double)r.ground,
            (double)r.friction, (double)r.ground_friction};
}
}  // namespace gipc
#endif
