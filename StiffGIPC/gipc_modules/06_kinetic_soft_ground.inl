// ============================================================================
// gipc_modules/06 — post-E1c residue: ground DETECTION, close-val family
// (frozen-adjacent, untouched), MGroundDist reduction, ground trial/intersect
// machinery, per-env ground alpha policy, and the _penv_energy_accum helper
// (defined here BEFORE the energy/ term files that use it — position is
// load-bearing). The kinetic/soft/ground ENERGY G/H kernels moved to
// energy/10_kinetic.inl / 14_soft_constraints.inl / 17_ground.inl (E1c).
// ============================================================================
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


__global__ void _computeGroundCloseVal(const double3* vertexes,
                                       const double*  g_offset,
                                       const double3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       double    dTol,
                                       uint32_t* _closeConstraintID,
                                       double*   _closeConstraintVal,
                                       uint32_t* _close_gpNum,
                                       int       number,
                                       const uint32_t* d_live = nullptr)
{
    // [C4-b] capacity-grid launch inside the frame graph: live ground-pair
    // count read on device.
    if(d_live)
        number = static_cast<int>(*d_live);
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
                                     const int*     p2g           = nullptr,
                                     const uint32_t* d_live       = nullptr)
{
    // [C4-b] the previous iteration's close-set count, read on device.
    if(d_live)
        number = static_cast<int>(*d_live);
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
#include "energy/term_common.cuh"  // [E3] _penv_energy_accum hoisted (shared with term TUs)

