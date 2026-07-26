// ============================================================================
// gipc_modules/07 — alpha/CCD/shared reductions (post-E1b residue).
// The energy-term reductions moved to energy/10..18_*.inl (E1b); what stays:
// ground/self/injective CCD alpha reductions, cfl max, double3 sqn/dot utils,
// and the shared __add_reduction second stage (used by the energy host
// dispatch AND non-energy consumers — shared mechanics, not a term).
// ============================================================================
__global__ void _reduct_min_groundAlpha_to_double(const double3* vertexes,
                                                  const uint32_t* surfVertIds,
                                                  const double*  g_offset,
                                                  const double3* g_normal,
                                                  const double3* moveDir,
                                                  double* minStepSizes,
                                                  double  slackness,
                                                  int     number,
                                                  const int* _point_body_id,
                                                  const int* _ground_skip_body,
                                                  int        _ground_body_count,
                                                  int*       ccd_alpha_invalid)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    // Direct-alpha semantics: 1 is the neutral candidate and every reduction
    // below is MIN. Inactive lanes stay at 1 and still reach __syncthreads().
    double temp = 1.0;
    if(idx < number)
    {
        int  svI  = surfVertIds[idx];
        bool skip = false;
        if(_point_body_id && _ground_skip_body && _ground_body_count > 0)
        {
            int bid = _point_body_id[svI];
            if(bid >= 0 && bid < _ground_body_count && _ground_skip_body[bid])
                skip = true;
        }
        if(!skip)
        {
            double3 normal = *g_normal;
            double  coef   = __GEIGEN__::__v_vec_dot(normal, moveDir[svI]);
            if(!isfinite(coef))
            {
                if(ccd_alpha_invalid)
                    atomicOr(ccd_alpha_invalid, kCcdInvalidGlobalGround);
                temp = 0.0;
            }
            else if(coef > 0.0)
            {
                double dist = __GEIGEN__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;
                if(!isfinite(dist) || dist <= 0.0)
                {
                    if(ccd_alpha_invalid)
                        atomicOr(ccd_alpha_invalid, kCcdInvalidGlobalGround);
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
                            atomicOr(ccd_alpha_invalid, kCcdInvalidGlobalGround);
                        temp = 0.0;
                    }
                }
            }
        }
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    //printf("%f\n", temp);
    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_min_full_to(temp, tep, number, idof, 1.0, minStepSizes + blockIdx.x);
}

__global__ void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes,
                                                        const uint4* tetrahedra,
                                                        const double3* moveDir,
                                                        double* minStepSizes,
                                                        double  slackness,
                                                        double  errorRate,
                                                        int     number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double ratio = 1 - slackness;

    double temp = 0.0;
    if(idx < number)
        temp = 1.0
               / _computeInjectiveStepSize_3d(vertexes,
                                              moveDir,
                                              tetrahedra[idx].x,
                                              tetrahedra[idx].y,
                                              tetrahedra[idx].z,
                                              tetrahedra[idx].w,
                                              ratio,
                                              errorRate);

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_max_tail_to(temp, tep, number, idof, minStepSizes + blockIdx.x);
}

__global__ void _reduct_min_selfAlpha_to_double(const double3* vertexes,
                                                const int4* _ccd_collitionPairs,
                                                const double3* moveDir,
                                                double*        minStepSizes,
                                                double         slackness,
                                                int            number,
                                                int*           ccd_alpha_invalid,
                                                int            invalid_bit)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp         = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    if(idx < number)
    {
        int4 MMCVIDI = _ccd_collitionPairs[idx];
        if(MMCVIDI.x < 0)
        {
            MMCVIDI.x = -MMCVIDI.x - 1;
            temp = point_triangle_ccd(vertexes[MMCVIDI.x],
                                      vertexes[MMCVIDI.y],
                                      vertexes[MMCVIDI.z],
                                      vertexes[MMCVIDI.w],
                                      __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                                      __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                                      __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                                      __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                                      CCDDistRatio,
                                      0);
        }
        else
        {
            temp = edge_edge_ccd(vertexes[MMCVIDI.x],
                                 vertexes[MMCVIDI.y],
                                 vertexes[MMCVIDI.z],
                                 vertexes[MMCVIDI.w],
                                 __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                                 __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                                 __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                                 __GEIGEN__::__s_vec_multiply(moveDir[MMCVIDI.w], -1),
                                 CCDDistRatio,
                                 0);
        }
        if(!isfinite(temp) || temp <= 0.0)
        {
            if(ccd_alpha_invalid) atomicOr(ccd_alpha_invalid, invalid_bit);
            temp = 0.0;
        }
        else
            temp = fmin(1.0, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_min_full_to(temp, tep, number, idof, 1.0, minStepSizes + blockIdx.x);
}

__global__ void _reduct_max_cfl_to_double(const double3* moveDir,
                                          double*        max_double_val,
                                          uint32_t*      mSVI,
                                          int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = idx < number ? __GEIGEN__::__norm(moveDir[mSVI[idx]]) : 0.0;


    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_max_tail_to(temp, tep, number, idof, max_double_val + blockIdx.x);
}

__global__ void _reduct_double3Sqn_to_double(const double3* A, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = idx < number ? __GEIGEN__::__squaredNorm(A[idx]) : 0.0;


    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, number, idof, D + blockIdx.x);
}

__global__ void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = idx < number ? __GEIGEN__::__v_vec_dot(A[idx], B[idx]) : 0.0;


    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, number, idof, D + blockIdx.x);
}


__global__ void __add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = idx < numbers ? mem[idx] : 0.0;

    __threadfence();

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, mem + blockIdx.x);
}

