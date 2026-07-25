__global__ void _getFrictionEnergy_Reduction_3D(double*        squeue,
                                                const double3* vertexes,
                                                const double3* o_vertexes,
                                                const int4*    _collisionPair,
                                                int            cpNum,
                                                double         dt,
                                                const double2* distCoord,
                                                const __GEIGEN__::Matrix3x2d* tanBasis,
                                                const double* lastH,
                                                double        fricDHat,
                                                double        eps,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                                const double* vert_mu = nullptr, double mu_global = 1.0

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        temp = __cal_Friction_energy(
            vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);
    // [per-body friction] the host combine multiplies the GLOBAL mu into this
    // sum (fric = frictionRate * slot); scale each pair's term by mu_pair/mu
    // here so the product lands on mu_pair exactly — zero changes to the four
    // combine paths, and the per-env slots below get the same scaling.
        if(vert_mu)
            temp *= _pair_mu(_collisionPair[idx], vert_mu, mu_global) / mu_global;

        int v0 = _collisionPair[idx].x;
        if(v0 < 0) v0 = -v0 - 1;
        _penv_energy_accum(penv, p2g, v0, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}

__global__ void _getFrictionEnergy_gd_Reduction_3D(double*        squeue,
                                                   const double3* vertexes,
                                                   const double3* o_vertexes,
                                                   const double3* _normal,
                                                   const uint32_t* _collisionPair_gd,
                                                   int           gpNum,
                                                   double        dt,
                                                   const double* lastH,
                                                   double        eps,
                                                   double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                                   const double* vert_mu_gd = nullptr, double mu_global = 1.0

)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = gpNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        temp = __cal_Friction_gd_energy(
            vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);
    // [per-body friction] see _getFrictionEnergy_Reduction_3D: host combine
    // multiplies the GLOBAL gd mu; scale per-vertex here so the product is exact.
        if(vert_mu_gd)
            temp *= vert_mu_gd[_collisionPair_gd[idx]] / mu_global;

        _penv_energy_accum(penv, p2g, _collisionPair_gd[idx], ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}

__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    double temp = 0.0;
    if(idx < number)
    {
        double3 normal = *g_normal;
        int     gidx   = _environment_collisionPair[idx];
        double  dist   = __GEIGEN__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
        double  dist2  = dist * dist;
        // [d-floor fail-fast] clamp removed; buildCP() throws before d can collapse here.
        temp = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);
        _penv_energy_accum(penv, p2g, gidx, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, number, idof, squeue + blockIdx.x);
}

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


__global__ void _getKineticEnergy_Reduction_3D(
    double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number,
    double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp = 0.0;
    if(idx < number)
    {
        temp = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_vertexes[idx], _xTilta[idx]))
               * _masses[idx] * 0.5;
        _penv_energy_accum(penv, p2g, idx, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, number, idof, _energy + blockIdx.x);
}

#ifdef USE_QUADRATIC_BENDING
__global__ void _getQuadBendingEnergy_Reduction(double*        squeue,
                                                const double3* vertexes,
                                                const double3* rest_vertexex,
                                                const uint2*   edges,
                                                const uint2*   edge_adj_vertex,
                                                const Eigen::Matrix4d* quad_bending_Q,
                                                int    edgesNum,
                                                double bendStiff,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        uint2 adj = edge_adj_vertex[idx];
        temp = __cal_quad_bending_energy(
            vertexes, rest_vertexex, edges[idx], adj, quad_bending_Q[idx], bendStiff);
        _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}
#endif

__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff,
                                            double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        uint2   adj     = edge_adj_vertex[idx];
        double3 rest_x0 = rest_vertexex[edges[idx].x];
        double3 rest_x1 = rest_vertexex[edges[idx].y];
        double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
        temp = __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
        _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);
    }
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}


__global__ void _getFEMEnergy_Reduction_3D(double*        squeue,
                                           const double3* vertexes,
                                           const uint4*   tetrahedras,
                                           const __GEIGEN__::Matrix3x3d* DmInverses,
                                           const double* volume,
                                           int           tetrahedraNum,
                                           double*       lenRate,
                                           double*       volRate,
                                           double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
#ifdef USE_SNK1
        temp = __cal_StabbleNHK_energy1_3D(
            vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#elif USE_SNK2
        temp = __cal_StabbleNHK_energy2_3D(
            vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx], volRate[idx]);
#else
        temp = __cal_ARAP_energy_3D(
            vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate[idx]);
#endif
        _penv_energy_accum(penv, p2g, tetrahedras[idx].x, ng, temp);
    }

    //printf("%f    %f\n\n\n", lenRate, volRate);
    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}
__global__ void _computeSoftConstraintEnergy_Reduction(double*        squeue,
                                                       const double3* vertexes,
                                                       const double3* targetVert,
                                                       const uint32_t* targetInd,
                                                       double motionRate,
                                                       double rate,
                                                       const int*     stitch_paired_vertex,
                                                       const double3* stitch_rest_offset,
                                                       int    number,
                                                       double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    double temp = 0.0;
    if(idx < number)
    {
        uint32_t vInd = targetInd[idx];
        double3 target;
        if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
        {
            int abd_idx = stitch_paired_vertex[idx];
            target = make_double3(
                vertexes[abd_idx].x + stitch_rest_offset[idx].x,
                vertexes[abd_idx].y + stitch_rest_offset[idx].y,
                vertexes[abd_idx].z + stitch_rest_offset[idx].z);
        }
        else
        {
            target = targetVert[idx];
        }
        double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(vertexes[vInd], target), rate));
        temp = motionRate * dis * 0.5;
        _penv_energy_accum(penv, p2g, vInd, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, number, idof, squeue + blockIdx.x);
}
__global__ void _get_triangleFEMEnergy_Reduction_3D(double*        squeue,
                                                    const double3* vertexes,
                                                    const uint3*   triangles,
                                                    const __GEIGEN__::Matrix2x2d* triDmInverses,
                                                    const double* area,
                                                    int           trianglesNum,
                                                    double        stretchStiff,
                                                    double        shearStiff,
                                                    double        strainRate,
                                                    double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = trianglesNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        temp = __cal_BaraffWitkinStretch_energy(
            vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff, strainRate);
        _penv_energy_accum(penv, p2g, triangles[idx].x, ng, temp);
    }

    //printf("%f    %f\n\n\n", lenRate, volRate);
    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}
__global__ void _getRestStableNHKEnergy_Reduction_3D(double*       squeue,
                                                     const double* volume,
                                                     int    tetrahedraNum,
                                                     double lenRate,
                                                     double volRate)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = tetrahedraNum;
    double                   temp = 0.0;
    if(idx < numbers)
        temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate)
                 - 0.5 * lenRate * log(4.0)))
                * volume[idx];

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}

__global__ void _getBarrierEnergy_Reduction_3D(double*        squeue,
                                               const double3* vertexes,
                                               const double3* rest_vertexes,
                                               int4*          _collisionPair,
                                               double         _Kappa,
                                               double         _dHat,
                                               int            cpNum,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = cpNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        temp = __cal_Barrier_energy(
            vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);
        int v0 = _collisionPair[idx].x;
        if(v0 < 0) v0 = -v0 - 1;
        _penv_energy_accum(penv, p2g, v0, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}

__global__ void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = vertexNum;
    double                   temp = idx < numbers ? __GEIGEN__::__v_vec_dot(b[idx], dx[idx]) : 0.0;

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
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

