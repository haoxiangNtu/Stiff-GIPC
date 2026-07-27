// ============================================================================
// multienv/isolation.cu — env-isolation machinery bodies (own TU).
// THE ISOLATION CONTRACT lives in multienv/isolation.cuh; policy stays at the
// three mid-frame entry sites. The four kernels here do flag/zero work only —
// no FP arithmetic that could touch the anchor.
// ============================================================================
#include "GIPC.cuh"
#include "gpu_eigen_libs.cuh"
#include "cuda_tools/cuda_tools.h"
#include <cstdio>
#include <cstdlib>

// [iron-law] mark every body of a quarantined env in the ground-skip table so
// ground DETECTION and ground-CCD ALPHA both stop seeing its infeasible
// vertices (both kernel families take _ground_skip_body).
__global__ void _mark_env_ground_skip(int* skip, const int* b2g, int env, int nb)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nb) return;
    if(b2g[b] == env) skip[b] = 1;
}

// [iron-law completion] per-env non-finite-direction scan: flags[g]=1 when any
// vertex of env g has a NaN/Inf PCG direction component — the trigger to
// quarantine a naturally-diverging env BEFORE the CCD chain trips on its NaNs.
__global__ void _scan_dir_nonfinite(const double3* dir, const int* p2g, int* flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    double3 d = dir[i];
    if(isfinite(d.x) && isfinite(d.y) && isfinite(d.z)) return;
    int g = p2g[i];
    if(g >= 0) atomicOr(flags + g, 1);
}

// [iron-law completion] zero the direction of every quarantined env: its
// positions stay frozen (alpha==0 keeps temp verbatim) and a ZERO direction
// gives neutral CCD candidates — the quarantined env becomes fully inert to
// every downstream consumer (CCD alpha fail-fast, global convergence norm).
__global__ void _zero_dir_quarantined(double3* dir, const int* p2g, const int* quar, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g >= 0 && quar[g]) dir[i] = make_double3(0.0, 0.0, 0.0);
}

// [iron-law] FLAG-ONLY infeasibility probe (no pair-list side effects): the
// frame-start quarantine scan must run BEFORE any CCD-alpha kernel of the new
// frame — teleports/drives can make an env infeasible between frames, and the
// collision state is carried over from the previous frame's last build, so the
// CCD fail-fast would otherwise fire before any detection sees the new state.
__global__ void _probe_ground_infeasible(const double3*  vertexes,
                                         const uint32_t* surfVertIds,
                                         const double*   g_offset,
                                         const double3*  g_normal,
                                         int             number,
                                         const int*      _point_body_id,
                                         const int*      _ground_skip_body,
                                         int             _ground_body_count,
                                         int*            _gdCollapse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    int svI = surfVertIds[idx];
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid])
            return;
    }
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[svI]) - *g_offset;
    if(!isfinite(dist) || dist <= 0.0)
        atomicMin(_gdCollapse, -(svI + 1));
}

// The ONE availability predicate for the whole isolation machinery.
bool GIPC::perEnvIsolationLive()
{
    // [C1] reads the finalize-time ModeConfig snapshot (same env sources,
    // captured once) instead of scattered getenv — behavior-neutral.
    return m_active_group_count > 1
        && (env_newton_iter_cap > 0 || m_mode_config.perenv_telem)
        && m_mode_config.perenv_alpha;
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
    if(!perEnvIsolationLive())
        return false;
    if(env < 0 || env >= kEnvAlphaSlots)
        return false;
    if(m_env_quarantined.empty())
        m_env_quarantined.assign(kEnvAlphaSlots, 0);
    if(!m_d_env_quarantined)
    {   // device mirror for the direction-zero kernel
        m_d_env_quarantined.resize_discard(kEnvAlphaSlots);  // [3d-2]
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

__global__ void _unmark_env_ground_skip(int* skip, const int* b2g, int env, int nb)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nb) return;
    if(b2g[b] == env) skip[b] = 0;
}

// [rl-reset] Inverse of quarantineEnv, for episode resets: a teleport that
// touches a quarantined env clears its persistent quarantine state (host+
// device flags, ground-skip marks) so the frame-start scan re-adjudicates
// from the NEW configuration. Revival is self-correcting, never a smuggling
// path: a still-infeasible env is re-quarantined within one frame (ground:
// the iron-law frame-start probe; non-finite direction: the per-iteration
// detector). The ground-skip table has no user-registration path — its only
// writers are quarantineEnv and checkpoint restore — so the unmark is safe.
bool GIPC::reviveEnv(int env)
{
    if(!perEnvIsolationLive())
        return false;
    if(env < 0 || env >= kEnvAlphaSlots)
        return false;
    if(m_env_quarantined.empty() || !m_env_quarantined[env])
        return false;
    m_env_quarantined[env] = 0;
    if(m_d_env_quarantined)
    {
        const int zero = 0;
        CUDA_SAFE_CALL(cudaMemcpy(m_d_env_quarantined + env, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice));
    }
    if(_ground_skip_body && m_d_b2g && _ground_body_count > 0)
    {
        int bs = 128, gs = (_ground_body_count + bs - 1) / bs;
        _unmark_env_ground_skip<<<gs, bs>>>(_ground_skip_body, m_d_b2g, env,
                                            _ground_body_count);
    }
    // m_env_status is the per-solve presentation vector (reassigned each
    // solve); clear it here too so a status query right after the reset —
    // the natural RL-loop pattern — already reflects the revival.
    if(env < (int)m_env_status.size())
        m_env_status[env] = 0;
    fprintf(stderr,
            "[per-env][REVIVE] env %d un-quarantined by reset/teleport; the "
            "frame-start scan re-adjudicates from the new configuration.\n",
            env);
    return true;
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
    if(!(_gdCollapse && m_d_p2g && m_d_b2g) || !perEnvIsolationLive())
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
