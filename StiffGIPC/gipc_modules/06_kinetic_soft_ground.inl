__global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _calKineticEnergy(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx]  = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}

__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double motionRate,
                                                         double rate,
                                                         int    global_offset,
                                                         int global_hessian_fem_offset,
                                                         const int*     stitch_paired_vertex,
                                                         const double3* stitch_rest_offset,
                                                         const int*     stitch_abd_body_id,
                                                         const __GEIGEN__::Vector12* abd_body_q,
                                                         int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // For bilateral stitch springs, compute target dynamically from current ABD vertex.
    // [stitch local-frame fix] target = anchor_world + A_now * local_offset where
    // local_offset is in the ABD body's rest frame.  Per the canonical q layout in
    // abd_jacobi_matrix.inl operator*(ABDJacobi, Vector12):
    //   q[3..5]  = A.row(0),  q[6..8]  = A.row(1),  q[9..11] = A.row(2)
    // So (A * lo).x = q[3]*lo.x + q[4]*lo.y + q[5]*lo.z, etc.
    // Without this, the stitch target only follows ABD translation, not rotation,
    // so FEM mesh visibly fails to track ABD rotation.
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // Previous code used q[3]/q[6]/q[9] for the x-component, which is
            // A.col(0)·lo = (A^T·lo)[0] — wrong for non-symmetric A.  Only
            // worked when A ≈ scale·I (Animated mode where ABD barely rotates).
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            // Fallback: legacy world-frame offset (no rotation tracking).
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
    __GEIGEN__::Matrix3x3d Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx    = atomicAdd(_gpNum, 1);
    //H3x3[pidx]    = Hpg;
    //D1Index[pidx] = vInd;
    vInd += global_hessian_fem_offset;
    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &vInd, Hpg.m, global_offset + idx);
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               const int*      stitch_paired_vertex,
                                               const double3*  stitch_rest_offset,
                                               const int*      stitch_abd_body_id,
                                               const __GEIGEN__::Vector12* abd_body_q,
                                               int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // [stitch local-frame fix] see _computeSoftConstraintGradientAndHessian
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // q[3..5]/q[6..8]/q[9..11] = A.row(0/1/2); see _computeSoftConstraintGradientAndHessian.
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
}

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

__global__ void _GroundCollisionDetect(const double3*  vertexes,
                                       const uint32_t* surfVertIds,
                                       const double*   g_offset,
                                       const double3*  g_normal,
                                       uint32_t* _environment_collisionPair,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       int       number,
                                       const int* _point_body_id,
                                       const int* _ground_skip_body,
                                       int        _ground_body_count,
                                       int*       _gdCollapse)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int svI = surfVertIds[idx];
    if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
    {
        int bid = _point_body_id[svI];
        if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid])
            return;
    }
    double dist = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[svI]) - *g_offset;
    if(!isfinite(dist) || dist <= 0.0)
    {
        // A non-positive distance is outside the logarithmic barrier domain.
        atomicMin(_gdCollapse, -(svI + 1));
        _environment_collisionPair[atomicAdd(_gpNum, 1)] = svI;
        return;
    }
    if(dist * dist > dHat)
        return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = svI;
}

__global__ void _getTotalForce(const double3* _force0, double3* _force, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    _force[idx].x += _force0[idx].x;
    _force[idx].y += _force0[idx].y;
    _force[idx].z += _force0[idx].z;
}


__global__ void _computeGroundGradientAndHessian(const double3* vertexes,
                                                 const double*  g_offset,
                                                 const double3* g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 double3*  gradient,
                                                 uint32_t* _gpNum,
                                                 Eigen::Matrix3d* triplet_values,
                                                 int*   row_ids,
                                                 int*   col_ids,
                                                 double dHat,
                                                 double Kappa_scalar,
                                                 int    global_offset,
                                                 int    number,
                                                 const double* kappa_grp = nullptr,
                                                 const int*    p2g       = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3      normal = *g_normal;
    unsigned int gidx   = _environment_collisionPair[idx];
    // [multi-env per-group κ] ground pair is a single vertex; nullptr → scalar (baseline).
    double Kappa = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa_scalar /* [-1 guard] */;
    double dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;
    // No d~0 clamp here: ground CCD preserves a numerical interior margin and
    // buildCP() rejects any state that nevertheless leaves the strict domain.

    double t   = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2)
                 + 1.0 / (dist2 * dist2) * (t * t);

    //printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        _gfxAdd(gidx, 0, grad.x);
        _gfxAdd(gidx, 1, grad.y);
        _gfxAdd(gidx, 2, grad.z);
    }

    // [PSD clamp] the log-barrier's curvature coefficient goes NEGATIVE in part
    // of its range; upstream commented out the `if(param > 0)` guard, so the
    // ground contact injected an INDEFINITE rank-1 block (Kappa*param*nn^T,
    // param<0) into the global Hessian — Newton directions lose their descent
    // guarantee along that mode. Clamp to 0 instead of restoring the if: the
    // pair stays in the bookkeeping (same counts/indices, no downstream shift),
    // only the negative curvature is projected out (exact PSD projection of a
    // rank-1 term). STIFF_GROUND_HESS_LEGACY=1 restores upstream behavior.
    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    if(param < 0.0 && !g_ground_hess_legacy)
        param = 0.0;
    {
        __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);
        __GEIGEN__::Matrix3x3d Hpg = __GEIGEN__::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        //H3x3[pidx]    = Hpg;
        //D1Index[pidx] = gidx;

        write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, Hpg.m, global_offset + idx);
    }
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundGradient(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double3*  gradient,
                                       uint32_t* _gpNum,
                                       double    dHat,
                                       double    Kappa_scalar,
                                       int       number,
                                       const double* kappa_grp = nullptr,
                                       const int*    p2g       = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    // [multi-env per-group κ] nullptr → scalar (baseline).
    double  Kappa = (kappa_grp && p2g && p2g[gidx] >= 0) ? kappa_grp[p2g[gidx]] : Kappa_scalar /* [-1 guard] */;
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;
    // [d-floor fail-fast] clamp removed; buildCP() throws before d can collapse here.

    double t   = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    //double H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
    double3 grad = __GEIGEN__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        _gfxAdd(gidx, 0, grad.x);
        _gfxAdd(gidx, 1, grad.y);
        _gfxAdd(gidx, 2, grad.z);
    }
}

__global__ void _computeGroundCloseVal(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double    dTol,
                                       uint32_t* _closeConstraintID,
                                       double*   _closeConstraintVal,
                                       uint32_t* _close_gpNum,
                                       int       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const double3* vertexes,
                                     const double*  g_offset,
                                     const double3* g_normal,
                                     int*           _isChange,
                                     uint32_t*      _closeConstraintID,
                                     double*        _closeConstraintVal,
                                     int            number,
                                     int*           _isChange_grp = nullptr,
                                     const int*     p2g           = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _closeConstraintID[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < _closeConstraintVal[idx])
    {
        *_isChange = 1;
        if(_isChange_grp && p2g && p2g[gidx] >= 0) _isChange_grp[p2g[gidx]] = 1;  /* [-1 guard] */   // [per-group κ]
    }
}

__global__ void _reduct_MGroundDist(const double3* vertexes,
                                    const double*  g_offset,
                                    const double3* g_normal,
                                    uint32_t*      _environment_collisionPair,
                                    double2*       _queue,
                                    int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    double2 temp = make_double2(0.0, 0.0);
    if(idx < number)
    {
        double3 normal = *g_normal;
        int     gidx   = _environment_collisionPair[idx];
        double  dist   = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
        double  tempv  = dist * dist;
        temp = make_double2(1.0 / tempv, tempv);
    }

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    double nextTp;
    int    warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
        double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
        temp.x         = std::max(temp.x, tempMin);
        temp.y         = std::max(temp.y, tempMax);
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = (warpTid < warpNum) ? sdata[warpTid] : make_double2(0.0, 0.0);

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
            double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
            temp.x         = std::max(temp.x, tempMin);
            temp.y         = std::max(temp.y, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _queue[blockIdx.x] = temp;
    }
}

__global__ void _computeSelfCloseVal(const double3*  vertexes,
                                     const double*   g_offset,
                                     const double3*  g_normal,
                                     const uint32_t* _environment_collisionPair,
                                     double          dTol,
                                     uint32_t*       _closeConstraintID,
                                     double*         _closeConstraintVal,
                                     uint32_t*       _close_gpNum,
                                     int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist  = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double  dist2 = dist * dist;

    if(dist2 < dTol)
    {
        int tidx                  = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx]  = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}


__global__ void _checkGroundIntersection(const double3* vertexes,
                                         const double*  g_offset,
                                         const double3* g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int* _isIntersect,
                                         int  number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double3 normal = *g_normal;
    int     gidx   = _environment_collisionPair[idx];
    double  dist = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    //printf("%f  %f\n", *g_offset, dist);
    if(!isfinite(dist) || dist <= 0.0)
        *_isIntersect = -1;
}

__global__ void _markGroundTrialInvalid(const double3* vertexes,
                                        const uint32_t* surface_vertices,
                                        const double* g_offset,
                                        const double3* g_normal,
                                        const int* point_body_id,
                                        const int* ground_skip_body,
                                        int ground_body_count,
                                        const int* point_to_group,
                                        int* env_invalid,
                                        int* status,
                                        int group_count,
                                        int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number) return;
    int vertex = surface_vertices[idx];
    if(point_body_id && ground_skip_body && ground_body_count > 0)
    {
        int body = point_body_id[vertex];
        if(body >= 0 && body < ground_body_count && ground_skip_body[body]) return;
    }
    double distance = __GEIGEN__::__v_vec_dot(*g_normal, vertexes[vertex]) - *g_offset;
    if(isfinite(distance) && distance > 0.0) return;

    if(point_to_group && env_invalid)
    {
        int group = point_to_group[vertex];
        if(group >= 0 && group < group_count)
        {
            atomicExch(env_invalid + group, 1);
            atomicOr(status, 1);
            return;
        }
        atomicOr(status, 2);
        return;
    }
    atomicOr(status, 1);
}

__global__ void _halveGroundInvalidEnvAlpha(double* env_alpha,
                                            const int* env_invalid,
                                            int group_count)
{
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if(group >= group_count || env_invalid[group] == 0) return;
    env_alpha[group] *= 0.5;
}

// [multi-env S3] per-env energy accumulation helper. Element's env = group of one
// of its vertices (intra-env after P1). atomicAdd the per-element energy `e` into
// the env bucket. penv==nullptr -> no-op (global-only callers stay byte-identical).
// vid is a GLOBAL vertex id (p2g is the full point_to_group), except the kinetic
// caller passes p2g already offset to the FEM region and vid local.
__device__ inline void _penv_energy_accum(double* penv, const int* p2g, int vid, int ng, double e)
{
    if(!penv || !p2g) return;
    int g = p2g[vid];
    if(g >= 0 && g < ng) atomicAdd(&penv[g], e);
}

