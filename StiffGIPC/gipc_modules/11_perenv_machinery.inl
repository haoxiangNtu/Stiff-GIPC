__global__ void _per_env_sqnorm_accum(const int* p2g, const double3* vec,
                                      double* per_env_sq, int* per_env_cnt,
                                      int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    if(per_env_cnt) atomicAdd(&per_env_cnt[g], 1);
    if(vec)
    {
        double3 v = vec[i];
        atomicAdd(&per_env_sq[g], v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

// [decouple] per-env binned reduction for kappa's FEM DOFs:
// gsum_g = Σ_{v∈g} gc·GE, gsnorm_g = Σ_{v∈g} |gc|².
// Callers must pass only the FEM vertex range. ABD collision vertices are not independent
// Cartesian DOFs and are accumulated in the 12-DOF generalized coordinates below.
__global__ void _per_env_kappa_deposit(const int* p2g, const double3* gc, const double3* GE,
                                       double* gsum_bin, double* gsnorm_bin, int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    double3 a = gc[i], b = GE[i];
    _binDepBase(gsum_bin   + (size_t)g * BINNED_K, a.x * b.x + a.y * b.y + a.z * b.z);
    _binDepBase(gsnorm_bin + (size_t)g * BINNED_K, a.x * a.x + a.y * a.y + a.z * a.z);
}

// [decouple] ABD counterpart of _per_env_kappa_deposit. This mirrors initKappa's global
// generalized-DOF reduction exactly: contact = total - non_contact, then accumulate
// contact·non_contact and |contact|² into the owning body's environment.
__global__ void _per_env_abd_kappa_deposit(const int* body_to_group,
                                           const double* total_gradient,
                                           const double* non_contact_gradient,
                                           double* gsum_bin,
                                           double* gsnorm_bin,
                                           int body_count,
                                           int ng)
{
    int dof = blockIdx.x * blockDim.x + threadIdx.x;
    if(dof >= body_count * 12) return;
    int body = dof / 12;
    int g    = body_to_group[body];
    if(g < 0 || g >= ng) return;
    double contact = total_gradient[dof] - non_contact_gradient[dof];
    _binDepBase(gsum_bin + (size_t)g * BINNED_K,
                contact * non_contact_gradient[dof]);
    _binDepBase(gsnorm_bin + (size_t)g * BINNED_K, contact * contact);
}

__global__ void _per_env_kappa_combine(double* gsum_g, double* gsnorm_g,
                                       const double* gsum_bin, const double* gsnorm_bin, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double s = 0.0, q = 0.0;
    for(int k = BINNED_K - 1; k >= 0; --k)   // finest bin first, fixed order
    { s += gsum_bin[(size_t)g * BINNED_K + k]; q += gsnorm_bin[(size_t)g * BINNED_K + k]; }
    gsum_g[g] = s; gsnorm_g[g] = q;
}

// [multi-env P3a] per-env max CFL speed = max over an env's SURFACE verts of
// |moveDir[v]| (same metric as _reduct_max_cfl_to_double). Used by a read-only
// diagnostic to see whether the global alpha_CFL = sqrt(dHat)/maxSpeed*0.5 is
// being dragged down by ONE env (=> per-env line-search would decouple them).
// atomicMax on double via bit-twiddling (positive doubles only -> monotone bits).
__device__ inline void _atomicMaxPosDouble(double* addr, double val)
{
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long  old = *a, assumed;
    do { assumed = old;
         double cur = __longlong_as_double((long long)assumed);
         if(cur >= val) break;
         old = atomicCAS(a, assumed, (unsigned long long)__double_as_longlong(val));
    } while(assumed != old);
}

// Direct CCD alpha buckets are initialized to 1 and monotonically MIN-reduced.
// Values are finite and non-negative; zero is reserved for an invalid candidate
// and is accompanied by m_ccd_alpha_invalid so the host fails loudly.
__device__ inline void _atomicMinNonnegativeDouble(double* addr, double val)
{
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  old  = *bits;
    unsigned long long  assumed;
    do
    {
        assumed   = old;
        double cur = __longlong_as_double(static_cast<long long>(assumed));
        if(cur <= val) break;
        old = atomicCAS(bits, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(val)));
    } while(assumed != old);
}

__global__ void _fill_double(double* values, double value, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < count) values[idx] = value;
}
__global__ void _per_env_max_cfl(const int* p2g, const double3* moveDir,
                                 const uint32_t* mSVI, double* per_env_max,
                                 int n_surf, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_surf) return;
    int v = mSVI[i];
    int g = p2g[v];
    if(g < 0 || g >= ng) return;
    _atomicMaxPosDouble(&per_env_max[g], __GEIGEN__::__norm(moveDir[v]));
}

// [perf] DEVICE-SIDE per-env feasible alpha + freeze — replaces the per-Newton-iter host round-trip
// (cudaDeviceSynchronize + 5x256-double D2H + 256-env host loop + H2D) that made the per-env path
// host-bound. scratch layout: [0*ng)=ground, [1*ng)=narrow-self, [2*ng)=refined-self,
// [3*ng)=surface cfl-max, [4*ng)=all-vertex Newton max-move.
// Writes env_alpha[g] directly (device), atomics n_env/n_frozen into cnt[0:2],
// and returns the effective invalid mask in cnt[2]. Math is bit-identical to
// the host loop (same per-env formulas, no reduction) → preserves strict cross-env bit-identity.
__global__ void _per_env_alpha_compute(double* env_alpha, const double* scratch, int ng,
                                       double sq, double ccd_size, int have_ccd,
                                       double temp_alpha, double alpha_CFL, int decouple,
                                       int no_refine, double thr_cv,
                                       const double* env_bbox2, double ntol_dt, double vtol_dt,
                                       const int* refined_invalid,
                                       const int* ccd_alpha_invalid,
                                       int* cnt,
                                       const uint32_t* d_ccd_count = nullptr)
{
    // [C5] inside the frame graph the swept-pair gate is a device count.
    if(d_ccd_count)
        have_ccd = (*d_ccd_count > 0u) ? 1 : 0;
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(g == 0 && ccd_alpha_invalid)
        atomicOr(&cnt[2], *ccd_alpha_invalid & kCcdInvalidEffectiveMask);
    double hmx = scratch[3 * ng + g];
    double nmx = scratch[4 * ng + g];
    bool present = nmx > 0.0 || (env_bbox2 && env_bbox2[g] > 0.0);
    if(!present) return;   // absent env → leave env_alpha[g] unchanged
    double hg = scratch[0 * ng + g], hs = scratch[1 * ng + g], hr = scratch[2 * ng + g];
    double ta = fmin(hg, hs);
    double a = ta;
    if(have_ccd && hmx > 0.0)
    {
        double acfl     = sq / hmx * 0.5;
        a               = fmin(ta, acfl);
        double gate_lhs = decouple ? ta : temp_alpha;
        double gate_rhs = decouple ? acfl : alpha_CFL;
        if(!no_refine && gate_lhs > 2.0 * gate_rhs)
        {
            if((refined_invalid && (refined_invalid[g] & kCcdRawInvalid))
               || !isfinite(hr) || hr <= 0.0 || hr > 1.0)
                atomicOr(&cnt[2], kCcdInvalidPerEnvRefined);
            a              = fmin(ta, hr * ccd_size);
            a              = fmax(a, acfl);
        }
    }
    // [env-scale] each env freezes against ITS OWN bbox scale (batch-invariant);
    // vtol_dt>0 = physical override; null env_bbox2 = scalar fallback (thr_cv).
    double thr_g = thr_cv;
    if(env_bbox2 && env_bbox2[g] > 0.0)
        thr_g = (vtol_dt > 0.0) ? vtol_dt : (ntol_dt * sqrt(env_bbox2[g]));
    if(decouple && nmx < thr_g) a = 0.0;   // freeze by ALL vertices, not surface-only CFL
    env_alpha[g] = a;
    atomicAdd(&cnt[0], 1);                   // n_env (present)
    if(a == 0.0) atomicAdd(&cnt[1], 1);      // n_frozen
}

// Diagnostic host mode retains its existing telemetry D2Hs. Promote a raw
// refined failure only for an environment whose exact refinement gate fires.
__global__ void _promote_per_env_refined_invalid(const double* scratch,
                                                 int ng,
                                                 int have_ccd,
                                                 double sq,
                                                 double temp_alpha,
                                                 double alpha_CFL,
                                                 int decouple,
                                                 int no_refine,
                                                 const int* refined_invalid,
                                                 int* ccd_alpha_invalid)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng || !have_ccd || no_refine) return;
    const double hmx = scratch[3 * ng + g];
    if(!(hmx > 0.0)) return;
    const double hg = scratch[0 * ng + g];
    const double hs = scratch[1 * ng + g];
    const double hr = scratch[2 * ng + g];
    const double ta = fmin(hg, hs);
    const double acfl = sq / hmx * 0.5;
    const double gate_lhs = decouple ? ta : temp_alpha;
    const double gate_rhs = decouple ? acfl : alpha_CFL;
    if(gate_lhs > 2.0 * gate_rhs
       && ((refined_invalid && (refined_invalid[g] & kCcdRawInvalid))
           || !isfinite(hr) || hr <= 0.0 || hr > 1.0))
        atomicOr(ccd_alpha_invalid, kCcdInvalidPerEnvRefined);
}

// [de-CPU] per-env CCD search-inflation alpha ta_e = min(ground_e, narrowSelf_e),
// computed directly from the alpha-valued S1 scratch regions.
__global__ void _compute_perenv_ta(const double* scratch, double* ta, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double gs = scratch[0 * ng + g];
    double ns = scratch[1 * ng + g];
    ta[g] = fmin(gs, ns);
}

// [S4-dev] device-derived per-env active mask — replaces the S4 host detection (own max-move
// kernel + D2H + host loop + H2D per iter) with ZERO added D2H: the freeze decision is already on
// device in m_env_alpha (set by _per_env_alpha_compute from a REAL solve). active = (alpha != 0).
// A masked env's next moveDir is 0 (RHS zeroed) -> hmx=0 -> _per_env_alpha_compute treats it as
// absent and leaves env_alpha unchanged (stays 0) -> stays masked until the periodic all-active
// recheck (top of loop) re-solves it for bounce-back detection. Deterministic (fixed cadence,
// per-env decision) -> strict/batch-invariance safe.
__global__ void _mask_fill(int* m, int v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) m[i] = v;
}
__global__ void _mask_from_env_alpha(int* env_active, const double* env_alpha, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    env_active[g] = (env_alpha[g] == 0.0) ? 0 : 1;
}

// [perf] DEVICE-SIDE per-group κ doubling (postLineSearch) — replaces the per-Newton host round-trip
// (D2H close flags + 256-env host loop + H2D). Doubles m_kappa_group[g] in place for groups that hit a
// close contact (capped at kappaMax, host scalar), and atomicMax's the envelope into maxK_out (init =
// current Kappa). Bit-identical to the host loop (same double+cap; max is order-free) → strict OK.
__global__ void _per_group_kappa_double(double* kappa_group, const int* close_grp,
                                        const double* env_alpha, int ng,
                                        double kappaMax, double* maxK_out)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    // A frozen environment must freeze its contact parameters as well as its state. Otherwise,
    // extra Newton iterations required by batch-mates keep doubling this group's kappa and can
    // reactivate it at a different optimum, making strict env0 depend on batch size.
    if(close_grp[g] && (!env_alpha || env_alpha[g] != 0.0))
    {
        double k = kappa_group[g] * 2.0;
        if(k > kappaMax) k = kappaMax;
        kappa_group[g] = k;
        _atomicMaxPosDouble(maxK_out, k);
    }
}

// [perf] DEVICE-SIDE per-env initKappa finalize — replaces the per-frame D2H(gsum_g/gsnorm_g) +
// NG-env host loop + H2D(kappa_group). Kg = clamp(max(-gsum/gsnorm, suggested), 0, kmax), where
// suggested/kmax are env-independent host scalars. Bit-identical to the host loop → strict OK.
__global__ void _per_env_kappa_finalize(const double* gsum_g, const double* gsnorm_g,
                                        double* kappa_group, int ng, double suggested, double kmax)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double mk = (gsnorm_g[g] > 0.0) ? (-gsum_g[g] / gsnorm_g[g]) : 0.0;
    if(mk < 0.0) mk = 0.0;
    double Kg = (mk > suggested) ? mk : suggested;
    if(Kg > kmax) Kg = kmax;
    kappa_group[g] = Kg;
}

// [multi-env S4 probe / aa17212] per-env MAX move (the real Newton-exit metric):
// max over an env's verts of |moveDir[i]|. Used by the S4 active-mask detection
// and the P3a-step2 per-env Newton convergence diagnostic. per_env_max pre-zeroed.
__global__ void _per_env_max_move(const int* p2g, const double3* moveDir,
                                  double* per_env_max, int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = p2g[i];
    if(g < 0 || g >= ng) return;
    _atomicMaxPosDouble(&per_env_max[g], __GEIGEN__::__norm(moveDir[i]));
}

// [multi-env S1] per-env direct feasible-alpha reductions. Alpha buckets are
// initialized to 1 and atomically MIN-reduced, exactly matching the merged path.
__global__ void _per_env_groundAlpha_min(const double3* vertexes,
                                         const uint32_t* surfVertIds,
                                         const double* g_offset, const double3* g_normal,
                                         const double3* moveDir, const int* p2g,
                                         double* per_env_alpha, double slackness, int number,
                                         const int* _point_body_id,
                                         const int* _ground_skip_body, int _ground_body_count,
                                         int ng, int* ccd_alpha_invalid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    int svI = surfVertIds[idx];
    int g   = p2g[svI];
    if(g < 0 || g >= ng) return;
    double temp = 1.0;
    bool   skip = false;
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid]) skip = true;
    }
    if(!skip)
    {
        double3 normal = *g_normal;
        double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
        if(!isfinite(coef))
        {
            if(ccd_alpha_invalid)
                atomicOr(ccd_alpha_invalid, kCcdInvalidPerEnvGround);
            temp = 0.0;
        }
        else if(coef > 0.0)
        {
            double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;
            if(!isfinite(dist) || dist <= 0.0)
            {
                if(ccd_alpha_invalid)
                    atomicOr(ccd_alpha_invalid, kCcdInvalidPerEnvGround);
                temp = 0.0;
            }
            else
            {
                const double candidate = slackness * (dist / coef);
                if(candidate > 0.0)
                    temp = fmin(1.0, candidate);
                else
                {
                    if(ccd_alpha_invalid)
                        atomicOr(ccd_alpha_invalid, kCcdInvalidPerEnvGround);
                    temp = 0.0;
                }
            }
        }
    }
    _atomicMinNonnegativeDouble(&per_env_alpha[g], temp);
}

__global__ void _per_env_selfAlpha_min(const double3* vertexes, const int4* pairs,
                                       const double3* moveDir, const int* p2g,
                                       double* per_env_alpha, double slackness, int number, int ng,
                                       const int* vloc, int* ccd_alpha_invalid,
                                       int invalid_bit, int* refined_invalid,
                                       const uint32_t* d_live = nullptr)
{
    // [C5] capacity-grid launch inside the frame graph: the live pair count
    // is read on device; OOB lanes fall through the min-identity.
    if(d_live)
        number = static_cast<int>(*d_live);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    double CCDDistRatio = 1.0 - slackness;
    int4   MMCVIDI      = pairs[idx];
    double temp;
    int    v0;
    if(MMCVIDI.x < 0)
    {
        MMCVIDI.x = -MMCVIDI.x - 1;
        v0        = MMCVIDI.x;
        int p = MMCVIDI.x, t0 = MMCVIDI.y, t1 = MMCVIDI.z, t2 = MMCVIDI.w;
        // [env-det] canonicalize the triangle vertex order by env-local id so mirror PT pairs feed
        // point_triangle_ccd in an identical order ⇒ bit-identical TOI cross-env.
        if(vloc) { int a=t0,b=t1,c=t2;
            if(vloc[b]<vloc[a]){int t=a;a=b;b=t;} if(vloc[c]<vloc[b]){int t=b;b=c;c=t;}
            if(vloc[b]<vloc[a]){int t=a;a=b;b=t;} t0=a; t1=b; t2=c; }
        temp = point_triangle_ccd(vertexes[p], vertexes[t0],
                                  vertexes[t1], vertexes[t2],
                                  __GEIGEN__::__s_vec_multiply(moveDir[p], -1),
                                  __GEIGEN__::__s_vec_multiply(moveDir[t0], -1),
                                  __GEIGEN__::__s_vec_multiply(moveDir[t1], -1),
                                  __GEIGEN__::__s_vec_multiply(moveDir[t2], -1),
                                  CCDDistRatio, 0);
    }
    else
    {
        v0   = MMCVIDI.x;
        int e0a=MMCVIDI.x, e0b=MMCVIDI.y, e1a=MMCVIDI.z, e1b=MMCVIDI.w;
        // [env-det] canonicalize edge endpoints + edge order by env-local id ⇒ mirror EE pairs feed
        // edge_edge_ccd identically ⇒ bit-identical TOI cross-env (the refined-CCD `hr` 1-ULP seed).
        if(vloc) {
            if(vloc[e0b]<vloc[e0a]){int t=e0a;e0a=e0b;e0b=t;}
            if(vloc[e1b]<vloc[e1a]){int t=e1a;e1a=e1b;e1b=t;}
            if(vloc[e1a]<vloc[e0a]){int t=e0a;e0a=e1a;e1a=t; t=e0b;e0b=e1b;e1b=t;}
        }
        temp = edge_edge_ccd(vertexes[e0a], vertexes[e0b],
                             vertexes[e1a], vertexes[e1b],
                             __GEIGEN__::__s_vec_multiply(moveDir[e0a], -1),
                             __GEIGEN__::__s_vec_multiply(moveDir[e0b], -1),
                             __GEIGEN__::__s_vec_multiply(moveDir[e1a], -1),
                             __GEIGEN__::__s_vec_multiply(moveDir[e1b], -1),
                             CCDDistRatio, 0);
    }
    const int g = p2g[v0];
    if(g < 0 || g >= ng) return;
    if(!isfinite(temp) || temp <= 0.0)
    {
        if(refined_invalid)
            atomicOr(&refined_invalid[g], kCcdRawInvalid);
        else if(ccd_alpha_invalid)
            atomicOr(ccd_alpha_invalid, invalid_bit);
        temp = 0.0;
    }
    else
        temp = fmin(1.0, temp);
    _atomicMinNonnegativeDouble(&per_env_alpha[g], temp);
}

// [multi-env P2] per-env CCD: build each env's swept tree on LOCAL verts + full-detect, looped.
void GIPC::buildBVH_and_CP_perenv_CCD(double alpha, const double* alpha_dev)
{
    h_ccd_cpNum.invalidate();  // [3b] swept re-emission ahead
    if(m_skip_all_collision)
    {
        h_ccd_cpNum = 0;
        m_last_ccd_pair_count = 0;
        return;
    }
    double3* sf = bvh_f._vertexes;
    double3* se = bvh_e._vertexes;
    bvh_f._vertexes = _vertexes;
    bvh_e._vertexes = _vertexes;
    // [decouple] PER-ENV CCD search inflation. The global `alpha` (=temp_alpha=min over ALL envs)
    // makes env e's swept-BVH search — and thus its refined-self CCD pair set + hr (refined-self
    // timestep) — depend on the MATES → batch-coupling (the confirmed root: frame0 k=3 hr diverged
    // 5.48 vs 4.93 when global temp_alpha diverged 0.272 vs 0.033). Here each env e searches with its
    // OWN feasible alpha ta_e = min(ground_e, narrowSelf_e) (from direct-alpha m_env_scratch, filled by S1
    // Phase A; depends only on env e → batch-invariant). ta_e ≥ global alpha → conservative superset
    // of pairs → hr is the true most-constraining value → refinement KEPT (unlike STIFF_NO_REFINE,
    // which skipped it and caused excessive backtracking). Gated STIFF_DECOUPLE_THRESH.
    const int KNG = m_active_group_count;
    bool perenv_ta = (m_mode_config.decouple_thresh && m_env_scratch
                      && m_mode_config.perenv_alpha);
    // [de-CPU] ta computed on device (see _compute_perenv_ta) — the per-env launches below read
    // their env's slot via the kernels' alpha_dev param; NO D2H / host loop.
    double*& d_perenv_ta = m_scr_perenv_ta;
    if(perenv_ta)
    {
        if(!d_perenv_ta)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_perenv_ta, kEnvAlphaSlots * sizeof(double)));
        _compute_perenv_ta<<<(KNG + 255) / 256, 256>>>(m_env_scratch, d_perenv_ta, KNG);
    }
    // scalar fallback (kernels use it when alpha_dev == nullptr)
    auto env_alpha_dev = [&](int e) -> const double* {
        return perenv_ta ? d_perenv_ta + e : alpha_dev;
    };
    // [perenv-parallel #2] STIFF_PERENV_PAR: run the per-env SWEPT (CCD) builds+queries concurrently
    // on the K-stream scratch pool — mirrors the DCD loop. The 6-7ms _selfQuery_*_ccd kernels are
    // occupancy-starved at 1-env size (~25 blocks); overlapping K envs fills the GPU.
    bool ccd_par = m_mode_config.perenv_par;
    int  ccd_K   = 1;
    if(ccd_par) { int cap = getenv("STIFF_PERENV_K") ? atoi(getenv("STIFF_PERENV_K")) : 8;
                  ccd_K = (int)h_perenv_active.size(); if(ccd_K > cap) ccd_K = cap; if(ccd_K < 1) ccd_K = 1;
                  allocPerEnvPool(ccd_K); }
    BvhScratch cof{bvh_f._nodes,bvh_f._bvs,bvh_f._MChash,bvh_f._indices,bvh_f._tempLeafBox,bvh_f._flags,bvh_f.m_node_env,bvh_f.m_node_max_element,
                   bvh_f._sort_tmp,bvh_f._sort_tmp_bytes,bvh_f._mch_alt,bvh_f._idx_alt,bvh_f._sort_cap};
    BvhScratch coe{bvh_e._nodes,bvh_e._bvs,bvh_e._MChash,bvh_e._indices,bvh_e._tempLeafBox,bvh_e._flags,bvh_e.m_node_env,bvh_e.m_node_max_element,
                   bvh_e._sort_tmp,bvh_e._sort_tmp_bytes,bvh_e._mch_alt,bvh_e._idx_alt,bvh_e._sort_cap};
    int* saved_f_node_body = bvh_f.m_node_body;
    int* saved_e_node_body = bvh_e.m_node_body;
    // See the DCD per-env path: pool slots have no cache-generation-local
    // body labels yet, so never race on the main tree's arrays.
    bvh_f.m_node_body = nullptr;
    bvh_e.m_node_body = nullptr;
    auto cswapIn = [](lbvh& b, BvhScratch& s){ b._nodes=s.nodes; b._bvs=s.bvs; b._MChash=s.mch;
        b._indices=s.idx; b._tempLeafBox=s.tmp; b._flags=s.flags; b.m_node_env=s.node_env;
        b.m_node_max_element=s.node_max_element;
        b._sort_tmp=s.sort_tmp; b._sort_tmp_bytes=s.sort_bytes;
        b._mch_alt=s.mch_alt; b._idx_alt=s.idx_alt; b._sort_cap=s.sort_cap; };
  ccd_redo:
    CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, sizeof(uint32_t)));
    // memset is on the DEFAULT stream; pool-stream detects atomicAdd _cpNum → make the zero globally
    // visible before any pool-stream work (same fix as the DCD loop).
    if(ccd_par) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    {
        int ci = 0;
        for(int e : h_perenv_active)
        {
            const double* aedev = env_alpha_dev(e);   // [de-CPU] device slot (nullptr -> scalar alpha)
            cudaStream_t st = ccd_par ? m_pool_streams[ci % ccd_K] : (cudaStream_t)0;
            if(h_perenv_face_cnt[e] > 0)
            {
                if(ccd_par) cswapIn(bvh_f, m_pool_f[ci % ccd_K]);
                bvh_f._active_idx        = d_perenv_face_idx + h_perenv_face_off[e];
                bvh_f.face_number_active = h_perenv_face_cnt[e];
                bvh_f.ConstructFullCCD(_moveDir, alpha, st, aedev);
                bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha, st, aedev);
            }
            if(h_perenv_edge_cnt[e] > 0)
            {
                if(ccd_par) cswapIn(bvh_e, m_pool_e[ci % ccd_K]);
                bvh_e._active_idx        = d_perenv_edge_idx + h_perenv_edge_off[e];
                bvh_e.face_number_active = h_perenv_edge_cnt[e];
                bvh_e.ConstructFullCCD(_moveDir, alpha, st, aedev);
                bvh_e.SelfCollitionFullDetect(dHat, _moveDir, alpha, st, aedev);
            }
            ++ci;
        }
    }
    if(ccd_par) { for(int k2 = 0; k2 < ccd_K; ++k2) CUDA_SAFE_CALL(cudaStreamSynchronize(m_pool_streams[k2]));
                  cswapIn(bvh_f, cof); cswapIn(bvh_e, coe); }  // restore original scratch
    CUDA_SAFE_CALL(cudaMemcpy(h_ccd_cpNum.refresh_dst(), _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // [perenv-parallel #1 FIX] the per-env CCD path (like the merged buildFullCP) MUST grow + redo on
    // overflow — else at the grasp h_ccd_cpNum exceeds the cap and the line-search per-env alpha reads
    // _ccd_collisonPairs OOB → illegal access (the N>4 crash). Emits past cap went to the trash slot.
    if((int)h_ccd_cpNum > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_ccd_cpNum + h_ccd_cpNum / 2) + 1;
        printf("[perenv CCD-grow] h_ccd_cpNum=%u > cap=%d -> grow to %d, redo\n",
               h_ccd_cpNum.get(), MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        pair_buffers_grow_ccd(PairBuffers{_collisonPairs, _MatIndex, _ccd_collisonPairs, MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM}, newcap);   // [v0.8.6 2b]
        bvh_f._ccd_collisionPair = bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        goto ccd_redo;
    }
    bvh_f._active_idx = nullptr; bvh_f.face_number_active = 0;
    bvh_e._active_idx = nullptr; bvh_e.face_number_active = 0;
    bvh_f.m_node_body = saved_f_node_body;
    bvh_e.m_node_body = saved_e_node_body;
    bvh_f._vertexes = sf;
    bvh_e._vertexes = se;
    m_last_ccd_pair_count = static_cast<uint32_t>(h_ccd_cpNum);
    if(m_last_ccd_pair_count > m_peak_ccd_pair_count)
        m_peak_ccd_pair_count = m_last_ccd_pair_count;
    if(getenv("STIFF_SWEPT_DIAG"))
        fprintf(stderr, "[swept] perenv ccd_pairs=%u\n", (unsigned)h_ccd_cpNum);
}

void GIPC::buildFullCP(const double& alpha, const double* alpha_dev)
{
    h_ccd_cpNum.invalidate();  // [3b] swept re-emission ahead
    if(m_skip_all_collision)
    {
        h_ccd_cpNum = 0;
        m_last_ccd_pair_count = 0;
        return;
    }

#ifdef STIFF_BVH_COHERENCE_AUDIT_BUILD
    // A swept Verlet proof must track both segment endpoints.  Reusing the
    // DCD cache's start-position-only decision would be incomplete whenever
    // the Newton direction or device alpha changes.
    auditBvhSweptTemporalCoherence(alpha, alpha_dev);
#endif

    // [multi-env P2] per-env CCD path (the swept-BVH equivalent of the per-env DCD path).
    // [C5] m_graph_merged_detect forces the merged swept path (see buildCP).
    if(m_perenv_bvh && m_d_p2g && m_perenv_bvh_groups > 0 && !m_graph_merged_detect)
    {
        buildBVH_and_CP_perenv_CCD(alpha, alpha_dev);
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

    CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, sizeof(uint32_t), 0));
    CUDA_SAFE_CALL(cudaEventRecord(m_aux_reset_event, cudaStreamPerThread));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(m_aux_stream, m_aux_reset_event, 0));

    // Same overlap pattern as buildCP.
    bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha, 0, alpha_dev);
    bvh_e.SelfCollitionFullDetect(
        dHat, _moveDir, alpha, m_aux_stream, alpha_dev);
    CUDA_SAFE_CALL(cudaEventRecord(m_aux_done_event, m_aux_stream));
    CUDA_SAFE_CALL(cudaStreamWaitEvent(
        cudaStreamPerThread, m_aux_done_event, 0));

    // [B3 ccd-defer] merged line of duty: skip the blocking count refresh —
    // the count and the past-capacity overflow signal ride the scalar-chain
    // read; the consumer falls back to this legacy path (defer flag off) on
    // overflow. The mirror stays invalidated (audit-armed closure proof).
    if(m_ccd_defer_counts)
        return;

    CUDA_SAFE_CALL(cudaMemcpy(h_ccd_cpNum.refresh_dst(), _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Overflow → grow CCD pair buffer + redo detection. The swept BVH
    // (ConstructFullCCD) is unchanged, so we only re-run the query into the
    // larger buffer. Emits past the old cap went to the trash slot (no OOB), and
    // consumers (self_largestFeasibleStepSize) run only after this returns, so
    // they always see a fully-populated, in-bounds buffer.
    while((int)h_ccd_cpNum > MAX_CCD_COLLITION_PAIRS_NUM)
    {
        int newcap = (int)(h_ccd_cpNum + h_ccd_cpNum / 2) + 1;
        printf("[CCD-grow] h_ccd_cpNum=%u > cap=%d -> grow to %d, redo detection\n",
               h_ccd_cpNum.get(), MAX_CCD_COLLITION_PAIRS_NUM, newcap);
        pair_buffers_grow_ccd(PairBuffers{_collisonPairs, _MatIndex, _ccd_collisonPairs, MAX_COLLITION_PAIRS_NUM, MAX_CCD_COLLITION_PAIRS_NUM}, newcap);   // [v0.8.6 2b]
        bvh_f._ccd_collisionPair = _ccd_collisonPairs;
        bvh_e._ccd_collisionPair = _ccd_collisonPairs;
        CUDA_SAFE_CALL(cudaMemsetAsync(_cpNum, 0, sizeof(uint32_t), 0));
        // Preserve the v0.8.4.2 grow-redo reset ordering without allocating a
        // temporary event or synchronizing the auxiliary stream on the host.
        CUDA_SAFE_CALL(cudaEventRecord(m_aux_reset_event, cudaStreamPerThread));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(m_aux_stream, m_aux_reset_event, 0));
        bvh_f.SelfCollitionFullDetect(dHat, _moveDir, alpha, 0, alpha_dev);
        bvh_e.SelfCollitionFullDetect(
            dHat, _moveDir, alpha, m_aux_stream, alpha_dev);
        CUDA_SAFE_CALL(cudaEventRecord(m_aux_done_event, m_aux_stream));
        CUDA_SAFE_CALL(cudaStreamWaitEvent(
            cudaStreamPerThread, m_aux_done_event, 0));
        CUDA_SAFE_CALL(cudaMemcpy(h_ccd_cpNum.refresh_dst(), _cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    m_last_ccd_pair_count = static_cast<uint32_t>(h_ccd_cpNum);
    if(m_last_ccd_pair_count > m_peak_ccd_pair_count)
        m_peak_ccd_pair_count = m_last_ccd_pair_count;
    if(getenv("STIFF_SWEPT_DIAG"))
        fprintf(stderr, "[swept] buildFullCP ccd_pairs=%u alpha=%g\n",
                (unsigned)h_ccd_cpNum, alpha);
}


// [multi-env determinism] d_bvh_vertexes = _vertexes + d_env_offset (no-op when offset=0).
__global__ void _addEnvOffset(double3* out, const double3* v, const double3* off, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    out[i].x = v[i].x + off[i].x;
    out[i].y = v[i].y + off[i].y;
    out[i].z = v[i].z + off[i].z;
}

void GIPC::buildBVH()
{
    if(m_skip_all_collision)
        return;
    // [multi-env P2] per-env mode builds trees inside buildCP (per-env loop); skip the merged build.
    // [C5] m_graph_merged_detect forces the merged build (see buildCP).
    if(m_perenv_bvh && m_perenv_bvh_groups > 0 && !m_graph_merged_detect)
        return;
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _addEnvOffset<<<gs, bs>>>(d_bvh_vertexes, _vertexes, d_env_offset, vertexNum); }
    bvh_f.Construct();
    bvh_e.Construct();
}

// [multi-env cross-env DIAGNOSTIC] scatter a per-vertex buffer into [env*maxL + localid] for
// env0/env1, so the host can compare corresponding entries (identical envs ⇒ should be equal).
__global__ void _xenv_scatter(const double3* buf, const int* p2g, const int* lid,
                              double* out, int maxL, int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= n) return;
    int g = p2g[v];
    if(g < 0 || g > 1) return;
    int L = lid[v];
    if(L < 0 || L >= maxL) return;
    out[((size_t)g * maxL + L) * 3 + 0] = buf[v].x;
    out[((size_t)g * maxL + L) * 3 + 1] = buf[v].y;
    out[((size_t)g * maxL + L) * 3 + 2] = buf[v].z;
}
// [xenv] classify each contact pair: intra-env0 / intra-env1 / cross-env / other. Decodes the
// first two vertices of the int4 pair (gv = c>=0 ? c : -c-1) and compares their groups.
__global__ void _xenv_paircount(const int4* pairs, const int* p2g, int n,
                                int* cnt_g0, int* cnt_g1, int* cross, int* other)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int4 p = pairs[i];
    int gv0 = (p.x >= 0) ? p.x : (-p.x - 1);
    int gv1 = (p.y >= 0) ? p.y : (-p.y - 1);
    int g0 = p2g[gv0];
    int g1 = p2g[gv1];
    if(g0 != g1)        atomicAdd(cross, 1);
    else if(g0 == 0)    atomicAdd(cnt_g0, 1);
    else if(g0 == 1)    atomicAdd(cnt_g1, 1);
    else                atomicAdd(other, 1);
}
void GIPC::xenvPairClassify(const int4* pairs, int n, const char* label)
{
    if(!getenv("STIFF_XENV") || !m_d_p2g || n <= 0) return;
    int*& d4 = m_scr_xenv4;
    if(!d4) CUDA_SAFE_CALL(cudaMalloc((void**)&d4, 4 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d4, 0, 4 * sizeof(int)));
    int bs = 256, gs = (n + bs - 1) / bs;
    _xenv_paircount<<<gs, bs>>>(pairs, m_d_p2g, n, d4 + 0, d4 + 1, d4 + 2, d4 + 3);
    int h[4]; CUDA_SAFE_CALL(cudaMemcpy(h, d4, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("[xenv]   %s: intra-env0=%d intra-env1=%d CROSS-env=%d other=%d (total=%d)\n",
           label, h[0], h[1], h[2], h[3], n);
}
// max |env0[k]-env1[k]| over corresponding (local-id) vertices. Builds the local-id map once.
double GIPC::xenvDiff(const double3* buf, const char* label)
{
    if(!getenv("STIFF_XENV") || !m_d_p2g) return -1.0;
    if(!m_xenv_ready)
    {
        std::vector<int> p2g(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(p2g.data(), m_d_p2g, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> lid(vertexNum, -1), cnt(2, 0);
        for(int v = 0; v < vertexNum; v++)
        { int g = p2g[v]; if(g == 0 || g == 1) lid[v] = cnt[g]++; }
        m_xenv_maxlocal = (cnt[0] > cnt[1]) ? cnt[0] : cnt[1];
        printf("[xenv] env0=%d env1=%d verts (maxlocal=%d)\n", cnt[0], cnt[1], m_xenv_maxlocal);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_xenv_lid, vertexNum * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(d_xenv_lid, lid.data(), vertexNum * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_xenv_buf, (size_t)2 * m_xenv_maxlocal * 3 * sizeof(double)));
        m_xenv_ready = true;
    }
    CUDA_SAFE_CALL(cudaMemset(d_xenv_buf, 0, (size_t)2 * m_xenv_maxlocal * 3 * sizeof(double)));
    { int bs = 256, gs = (vertexNum + bs - 1) / bs;
      _xenv_scatter<<<gs, bs>>>(buf, m_d_p2g, d_xenv_lid, d_xenv_buf, m_xenv_maxlocal, vertexNum); }
    std::vector<double> h((size_t)2 * m_xenv_maxlocal * 3);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_xenv_buf, h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double mx = 0.0; int worst = -1;
    for(int L = 0; L < m_xenv_maxlocal; L++)
        for(int c = 0; c < 3; c++)
        { double d = fabs(h[((size_t)0 * m_xenv_maxlocal + L) * 3 + c] - h[((size_t)1 * m_xenv_maxlocal + L) * 3 + c]);
          if(d > mx) { mx = d; worst = L; } }
    // [env-det dbg] identify the worst lid: env0 global vert + btype (boundary/driven vs free).
    // STIFF_XENV_ID: build env0 lid→vert inverse once, report on first nonzero diff.
    if(getenv("STIFF_XENV_ID") && worst >= 0 && mx > 0.0)
    {
        static std::vector<int> inv;
        if(inv.empty())
        {
            inv.assign(m_xenv_maxlocal, -1);
            std::vector<int> hp(vertexNum), hl(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), m_d_p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hl.data(), d_xenv_lid, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            for(int v=0; v<vertexNum; v++) if(hp[v]==0 && hl[v]>=0 && hl[v]<m_xenv_maxlocal) inv[hl[v]]=v;
        }
        printf("[xenv-id] %-20s worst lid=%d -> env0 global vert=%d (vertexNum=%d)\n",
               label, worst, inv[worst], vertexNum);
    }
    printf("[xenv] %-22s maxdiff %.6e  (worst lid=%d)\n", label, mx, worst);
    return mx;
}

// [multi-env P2 / per-env BVH] primitive→env via the first vertex's group.
__global__ void _prim_env_f(const uint3* faces, const int* p2g, int* env, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    env[i] = p2g[faces[i].x];
}
__global__ void _prim_env_e(const uint2* edges, const int* p2g, int* env, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    env[i] = p2g[edges[i].x];
}
// P1: group faces/edges by env (host, one-time; topology static) → env-contiguous _active_idx.
static void _build_perenv_list(const int* d_env, int n, int NG,
                               int*& d_idx, std::vector<int>& off, std::vector<int>& cnt,
                               const std::vector<uint64_t>* key,
                               bool canonical_order)
{
    std::vector<int> h_env(n);
    CUDA_SAFE_CALL(cudaMemcpy(h_env.data(), d_env, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    off.assign(NG, 0);
    cnt.assign(NG, 0);
    int ungrouped = 0;
    for(int i = 0; i < n; i++) { int e = h_env[i]; if(e >= 0 && e < NG) cnt[e]++; else ungrouped++; }
    int acc = 0;
    for(int e = 0; e < NG; e++) { off[e] = acc; acc += cnt[e]; }
    std::vector<int> idx(acc), cur(off);
    for(int i = 0; i < n; i++) { int e = h_env[i]; if(e >= 0 && e < NG) idx[cur[e]++] = i; }
    // [env-det BVH] order each env's active list by an ENV-LOCAL key so ALL envs share an identical
    // local prim ordering (mirror) ⇒ the per-env Construct sees identical inputs ⇒ identical trees.
    // (default builds in ascending global-prim order, which is NOT env-mirror for co-located envs.)
    if(key && canonical_order)
        for(int e = 0; e < NG; e++)
        { int s = off[e], c = cnt[e];
          std::sort(idx.begin() + s, idx.begin() + s + c,
                    [&](int a, int b){ return (*key)[a] < (*key)[b]; }); }
    if(ungrouped) printf("[perenv-bvh] WARNING %d/%d prims ungrouped (env<0) — excluded from per-env BVH\n", ungrouped, n);
    if(d_idx) CUDA_SAFE_CALL(cudaFree(d_idx));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_idx, (size_t)(acc > 0 ? acc : 1) * sizeof(int)));
    if(acc) CUDA_SAFE_CALL(cudaMemcpy(d_idx, idx.data(), (size_t)acc * sizeof(int), cudaMemcpyHostToDevice));
}
// [env-det] enable env-major Morton on the MERGED BVH: compute per-prim env id (p2g of the prim's
// first vertex) once, point the BVHs at it, and turn on the env-major sort key. Co-located identical
// envs then build env-blocked (mirror) trees ⇒ env-symmetric broad-phase enumeration.
void GIPC::enableEnvMajorBVH(const int* p2g)
{
    if(!p2g || d_face_env) return;   // once
    m_d_p2g = p2g;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_env, (size_t)nF * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_env, (size_t)nE * sizeof(int)));
    { int bs=256, gs=(nF+bs-1)/bs; _prim_env_f<<<gs,bs>>>(bvh_f._faces, p2g, d_face_env, nF); }
    { int bs=256, gs=(nE+bs-1)/bs; _prim_env_e<<<gs,bs>>>(bvh_e._edges, p2g, d_edge_env, nE); }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    bvh_f.m_prim_env = d_face_env;
    bvh_e.m_prim_env = d_edge_env;
    // [env-det] Morton low-bits tie-break = a MIRROR key derived from the prim's env-LOCAL VERTEX ids
    // (the env-local vertex id IS mirror across identical envs; the global prim/edge numbering is NOT).
    // env-local vert id = rank of a vertex among its env's verts by ascending global index.
    std::vector<int> hp2(vertexNum);
    CUDA_SAFE_CALL(cudaMemcpy(hp2.data(), p2g, (size_t)vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> vloc(vertexNum, 0); { std::vector<int> ec;
        for(int v=0;v<vertexNum;v++){ int g=hp2[v]; if(g<0) continue; if(g>=(int)ec.size()) ec.resize(g+1,0); vloc[v]=ec[g]++; } }
    std::vector<uint3> hf(nF); std::vector<uint2> he(nE);
    CUDA_SAFE_CALL(cudaMemcpy(hf.data(), bvh_f._faces, (size_t)nF*sizeof(uint3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
    std::vector<int> floc(nF), eloc(nE);
    for(int i=0;i<nE;i++){ int a=vloc[he[i].x], b=vloc[he[i].y]; int lo=a<b?a:b, hi=a<b?b:a;
        eloc[i] = (lo<<13)|hi; }                              // 2×13-bit env-local vert ids (verts/env<8192)
    for(int i=0;i<nF;i++){ int a=vloc[hf[i].x],b=vloc[hf[i].y],c=vloc[hf[i].z];
        if(a>b){int t=a;a=b;b=t;} if(b>c){int t=b;b=c;c=t;} if(a>b){int t=a;a=b;b=t;}
        floc[i] = (int)((((uint64_t)a*9973u + b)*9973u + c) & 0x3FFFFFFu); }  // mirror hash (26-bit)
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_localid,(size_t)nF*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_localid,(size_t)nE*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_face_localid,floc.data(),(size_t)nF*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edge_localid,eloc.data(),(size_t)nE*sizeof(int),cudaMemcpyHostToDevice));
    bvh_f.m_prim_localid = d_face_localid;
    bvh_e.m_prim_localid = d_edge_localid;
    // [env-det] static per-prim first-vertex index so _calcMChash can subtract the LIVE per-env world
    // offset (d_env_offset is populated per-frame, AFTER this once-call) ⇒ Morton computed in the
    // local frame ⇒ mirror trees, while spacing>0 is kept for broad-phase efficiency.
    std::vector<uint32_t> fv0(nF), ev0(nE);
    for(int i=0;i<nF;i++) fv0[i]=hf[i].x;
    for(int i=0;i<nE;i++) ev0[i]=he[i].x;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_face_v0,(size_t)nF*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_edge_v0,(size_t)nE*sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(d_face_v0,fv0.data(),(size_t)nF*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edge_v0,ev0.data(),(size_t)nE*sizeof(uint32_t),cudaMemcpyHostToDevice));
    bvh_f.m_env_offset = d_env_offset; bvh_f.m_prim_v0 = d_face_v0;
    bvh_e.m_env_offset = d_env_offset; bvh_e.m_prim_v0 = d_edge_v0;
    set_bvh_envmajor(1);
    printf("[env-major-bvh] enabled: %d faces, %d edges\n", nF, nE);
}
void GIPC::buildPerEnvBVHIndex(int NG, const int* p2g)
{
    m_perenv_bvh_groups = NG;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    int *d_fenv = nullptr, *d_eenv = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fenv, (size_t)nF * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_eenv, (size_t)nE * sizeof(int)));
    { int bs = 256, gs = (nF + bs - 1) / bs; _prim_env_f<<<gs, bs>>>(bvh_f._faces, p2g, d_fenv, nF); }
    { int bs = 256, gs = (nE + bs - 1) / bs; _prim_env_e<<<gs, bs>>>(bvh_e._edges, p2g, d_eenv, nE); }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // [env-det BVH] per-prim ENV-LOCAL sort key = packed sorted env-local vertex ids. Env-local id =
    // rank of a vertex among its env's vertices by ascending GLOBAL index (mirror across identical
    // envs, proven by xenvDiff call#0==0). Used to canonicalize the per-env active-list ordering.
    std::vector<uint64_t> fkey, ekey;
    if(m_mode_config.bvh_envdet)
    {
        std::vector<int> hp2g(vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(hp2g.data(), p2g, (size_t)vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> vloc(vertexNum, 0); std::vector<int> ec(NG, 0);
        for(int v = 0; v < vertexNum; v++){ int g = hp2g[v]; if(g >= 0 && g < NG) vloc[v] = ec[g]++; }
        auto pack3 = [](uint64_t a, uint64_t b, uint64_t c){
            uint64_t lo=a<b?a:b, hi=a<b?b:a; if(c<lo){uint64_t t=lo;lo=c;c=t;} if(c<hi){uint64_t t=hi;hi=c;c=t;}
            return (lo<<40)|(hi<<20)|c; };
        std::vector<uint3> hf(nF); std::vector<uint2> he(nE);
        CUDA_SAFE_CALL(cudaMemcpy(hf.data(), bvh_f._faces, (size_t)nF*sizeof(uint3), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(he.data(), bvh_e._edges, (size_t)nE*sizeof(uint2), cudaMemcpyDeviceToHost));
        fkey.resize(nF); ekey.resize(nE);
        for(int i=0;i<nF;i++) fkey[i]=pack3(vloc[hf[i].x],vloc[hf[i].y],vloc[hf[i].z]);
        for(int i=0;i<nE;i++){ uint64_t a=vloc[he[i].x],b=vloc[he[i].y]; ekey[i]=(a<b)?((a<<20)|b):((b<<20)|a); }
    }
    _build_perenv_list(d_fenv, nF, NG, d_perenv_face_idx,
                       h_perenv_face_off, h_perenv_face_cnt,
                       fkey.empty() ? nullptr : &fkey, m_mode_config.bvh_envdet);
    _build_perenv_list(d_eenv, nE, NG, d_perenv_edge_idx,
                       h_perenv_edge_off, h_perenv_edge_cnt,
                       ekey.empty() ? nullptr : &ekey, m_mode_config.bvh_envdet);
    CUDA_SAFE_CALL(cudaFree(d_fenv));
    CUDA_SAFE_CALL(cudaFree(d_eenv));
    int tf = 0, te = 0;
    h_perenv_active.clear();
    for(int e = 0; e < NG; e++)
    {
        tf += h_perenv_face_cnt[e]; te += h_perenv_edge_cnt[e];
        if(h_perenv_face_cnt[e] > 0 || h_perenv_edge_cnt[e] > 0) h_perenv_active.push_back(e);
    }
    printf("[perenv-bvh] built per-env index: NG=%d active=%zu faces %d/%d edges %d/%d\n",
           NG, h_perenv_active.size(), tf, nF, te, nE);
}

// P2: per-env Construct+Detect on LOCAL _vertexes (full precision, per-env identical, no cross-
// env candidates) — replaces buildBVH()+buildCP() when m_perenv_bvh. Pairs append to the shared
// _collisonPairs via the atomic _cpNum (env-order-independent, consumers iterate 0..h_cpNum[0]).
// [perenv-parallel #1] allocate K scratch sets (face + edge sized) + K streams (once).
void GIPC::allocPerEnvPool(int K)
{
    if(m_pool_K >= K) return;
    const int old_K = m_pool_K;
    int nF = (int)bvh_f.face_number, nE = (int)bvh_e.edge_number;
    m_pool_f.resize(K); m_pool_e.resize(K);
    // [perenv-parallel #2] per-slot cub sort scratch, pre-sized to the FULL prim count so the
    // in-loop ensure_sort_scratch never reallocates (pointer stability across swapIn/restore).
    auto allocSort = [](GIPC::BvhScratch& s, int cap)
    {
        if(s.sort_tmp) return;
        size_t bytes = 0;
        cub::DeviceRadixSort::SortPairs((void*)nullptr, bytes, (const uint64_t*)nullptr,
                                        (uint64_t*)nullptr, (const uint32_t*)nullptr,
                                        (uint32_t*)nullptr, cap, 0, 64);
        CUDA_SAFE_CALL(cudaMalloc(&s.sort_tmp, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch_alt, (size_t)cap * sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx_alt, (size_t)cap * sizeof(uint32_t)));
        s.sort_bytes = bytes; s.sort_cap = cap;
    };
    for(int k = 0; k < K; ++k)
    {
        if(!m_pool_f[k].nodes) { BvhScratch& s = m_pool_f[k];
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.nodes,(2*nF-1)*sizeof(Node)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.bvs,(2*nF-1)*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch,nF*sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx,nF*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.tmp,nF*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.flags,(nF-1)*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.node_env,(2*nF-1)*sizeof(int)));
            allocSort(s, nF); }
        if(!m_pool_e[k].nodes) { BvhScratch& s = m_pool_e[k];
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.nodes,(2*nE-1)*sizeof(Node)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.bvs,(2*nE-1)*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.mch,nE*sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.idx,nE*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.tmp,nE*sizeof(AABB)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.flags,(nE-1)*sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s.node_env,(2*nE-1)*sizeof(int)));
            if(bvh_e.m_node_max_element)
                CUDA_SAFE_CALL(cudaMalloc((void**)&s.node_max_element,
                                          (2*nE-1)*sizeof(uint32_t)));
            allocSort(s, nE); }
    }
    m_pool_streams.resize(K);
    for(int k = old_K; k < K; ++k)
        CUDA_SAFE_CALL(cudaStreamCreate(&m_pool_streams[k]));
    m_pool_K = K;
}

// ============================================================================
// [C5] isolated-mode whole-frame graph — per-env device decision kernels.
//
// The per-env Newton exit (all_env_frozen), the S3 per-env line-search loop
// control and the per-env ground-trial adjudication move from host readbacks
// into FrameDeviceState + conditional handles, mirroring the merged C4 path.
// ============================================================================

// Per-env Newton convergence: replaces the 12-byte cnt D2H + host
// all_env_frozen. cnt layout: [0]=n_env(present), [1]=n_frozen, [2]=invalid.
__global__ void _perenv_newton_decide(const int* cnt,
                                      frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x || !frame)
        return;
    const int n_env    = cnt[0];
    const int n_frozen = cnt[1];
    const uint32_t invalid =
        static_cast<uint32_t>(cnt[2]) & kCcdInvalidEffectiveMask;
    frame->newton_converged = (n_env > 0 && n_frozen == n_env) ? 1 : 0;
    frame->phase            = frame_fsm::PHASE_NEWTON_DECIDE;
    if(invalid)
    {
        frame_fsm::fsm_record_error(
            frame, frame_fsm::ERR_CCD_INVALID, invalid, -1, -1);
        frame->result = frame_fsm::FRAME_FATAL;
        frame->phase  = frame_fsm::PHASE_ROLLBACK;
    }
}

// S3 round seed: counts[0]=nfail, [1]=tol-accepts, [2]=round, [3]=budget hit.
__global__ void _s3_conditional_seed(int* counts,
                                     frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;
    counts[0] = 1;   // force at least one round
    counts[1] = 0;
    counts[2] = 0;
    counts[3] = 0;
    if(frame)
        frame->phase = frame_fsm::PHASE_LINE_SEARCH;
}

__global__ void _s3_round_begin(int* counts)
{
    if(blockIdx.x || threadIdx.x)
        return;
    counts[0] = 0;
    ++counts[2];
}

// Ground-trial gate for the recorded S3 body: when any env's trial state is
// ground-invalid the energy decision of this round must not consume the
// (garbage) trial energies. _s3_decide takes this as an early-out gate.
__global__ void _s3_tail_conditional(int*        counts,
                                     const int*  ground_invalid,
                                     double*     env_alpha,
                                     const int*  env_ground_invalid,
                                     int         ng,
                                     int         budget,
                                     cudaGraphConditionalHandle handle,
                                     frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x)
        return;
    const int ground = ground_invalid ? *ground_invalid : 0;
    bool      retry  = false;
    if(ground & 2)
    {
        // Ground-collapse class: the host solver breaks out of the per-env
        // loop and falls through to the uniform-alpha line search. Mark the
        // fallback flag; the recorded IF replays that same fallback.
        counts[3] = 1;
    }
    else if(ground)
    {
        // Ground-invalid trial: halve the offending envs and go again
        // (energy decision was gated out this round).
        for(int g = 0; g < ng; ++g)
            if(env_ground_invalid && env_ground_invalid[g])
                env_alpha[g] *= 0.5;
        retry = true;
    }
    else if(counts[0] > 0)
    {
        retry = true;   // _s3_decide already halved the failing envs
    }
    if(retry && counts[2] > budget)
    {
        // Budget exhausted without per-env descent: NOT a frame failure —
        // the host solver falls through to the uniform-alpha line search
        // from the same temp config. Arm the fallback flag instead.
        counts[3] = 1;
        retry     = false;
    }
    const bool healthy =
        !frame || frame->result == frame_fsm::FRAME_OK;
    if(frame)
    {
        frame->ls_decision = retry ? 1 : 0;
        ++frame->ls_trial;
    }
    cudaGraphSetConditional(handle, (healthy && retry) ? 1u : 0u);
}

// [C5] S3 fallback predicate: the per-env search failed to produce a descent
// for every env (budget exhausted, or a ground-collapse trial). The host
// solver falls through to the uniform-alpha line search from the SAME temp
// configuration; the recorded IF replays exactly that.
__global__ void _s3_fallback_predicate(const int* counts,
                                       frame_fsm::FrameDeviceState* frame,
                                       cudaGraphConditionalHandle handle)
{
    if(blockIdx.x || threadIdx.x)
        return;
    const bool healthy =
        !frame || frame->result == frame_fsm::FRAME_OK;
    cudaGraphSetConditional(handle,
                            (healthy && counts[3] != 0) ? 1u : 0u);
}
