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
                                                int            invalid_bit,
                                                const uint32_t* d_live,
                                                int            type_filter)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    double temp         = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    // [B3 ccd-defer] capacity grid: mask by the live device count (raw atomic
    // total can exceed capacity when emits hit the trash slot). OOB threads
    // keep the min identity 1.0, so the reduction is bitwise the exact-grid one.
    int live = number;
    if(d_live)
    {
        const unsigned raw = *d_live;
        live = raw < (unsigned)number ? (int)raw : number;
    }
    if(idx < live)
    {
        int4 MMCVIDI = _ccd_collitionPairs[idx];
        // [P0 alpha-type-split] a warp mixing PT and EE pairs executes BOTH
        // ACCD bodies serially. Under the split launch each pass keeps every
        // warp uniform; the other pass's lanes hold the min identity 1.0, so
        // each pair is still computed exactly once and the minima are
        // bitwise the single-pass ones (min is order-independent).
        if(type_filter >= 0 && ((MMCVIDI.x < 0) != (type_filter == 1)))
        {
            // filtered out: keep temp == 1.0
        }
        else if(MMCVIDI.x < 0)
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
    // Under a capacity grid every block participates fully (identity-padded);
    // the tail's last-block arithmetic must then see the full grid extent.
    gipc_block_min_full_to(temp,
                           tep,
                           d_live ? (int)(gridDim.x * blockDim.x) : number,
                           idof,
                           1.0,
                           minStepSizes + blockIdx.x);
}

// [P0 alpha-type-split] launch wrapper: two type-uniform passes over the SAME
// pair buffer (no data movement, no order change). Pass 1 writes PT block
// minima to mqueue[0..bn), pass 2 EE minima to mqueue[bn..2bn); the caller's
// cascade then reduces 2*bn values. Small inputs keep the legacy single pass
// (split gains nothing there and the reduce scratch is sized from the input
// count). Returns the number of block minima written.
__global__ void _fill_double(double* values, double value, int count);   // module 11

// [alpha-resize] The masked CCD reduction sweeps the TRAINED pair capacity so
// the recorded grid stays valid across replays; lanes past the live count hold
// the min identity 1.0, so a fully-masked block writes exactly 1.0 into its
// partial slot. That makes narrowing the grid BITWISE NEUTRAL provided the
// partial slots the narrowed grid never writes already hold 1.0 -- hence the
// identity fill below (blockNum doubles, negligible) before the resized
// launch. The host-width cascade that follows then reduces the same values it
// would have reduced at full width.
static bool alpha_resize_enabled()
{
    static int v = -1;
    if(v < 0)
    {
        const char* e = getenv("STIFF_ALPHA_RESIZE");
        v             = e && e[0] ? (atoi(e) != 0 ? 1 : 0) : 0;
    }
    return v != 0;
}

static inline int launch_reduct_min_selfAlpha(const double3*  vertexes,
                                              const int4*     pairs,
                                              const double3*  moveDir,
                                              double*         mqueue,
                                              double          slackness,
                                              int             numbers,
                                              int*            invalid,
                                              int             invalid_bit,
                                              const uint32_t* d_live,
                                              unsigned int    threadNum,
                                              unsigned int    sharedMsize)
{
    const int blockNum = (numbers + (int)threadNum - 1) / (int)threadNum;
    static int s_split = -1;
    if(s_split < 0)
    {
        // Default OFF: measured on the 4-env foldshirt heavy segment the
        // split is 17% SLOWER per logical reduction (1.854 vs 1.586 ms) —
        // the detect kernels emit PT and EE pairs in contiguous slot ranges,
        // so warps are already type-uniform and the second pass only adds
        // header traffic + tail work. Kept as an opt-in experiment.
        const char* e = std::getenv("STIFF_ALPHA_TYPE_SPLIT");
        s_split       = e && e[0] ? (std::atoi(e) != 0 ? 1 : 0) : 0;
    }
    if(!s_split || numbers < 1024)
    {
        int rs_slot = -1;
        if(d_live && alpha_resize_enabled())
        {
            _fill_double<<<(blockNum + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                mqueue, 1.0, blockNum);
            rs_slot = gipc::graph_resize::arm(
                reinterpret_cast<const int*>(d_live), 1, (int)threadNum, blockNum);
        }
        _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
            vertexes, pairs, moveDir, mqueue, slackness, numbers, invalid, invalid_bit, d_live, -1);
        if(rs_slot >= 0)
            gipc::graph_resize::bind_last(rs_slot);
        return blockNum;
    }
    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        vertexes, pairs, moveDir, mqueue, slackness, numbers, invalid, invalid_bit, d_live, 1);
    _reduct_min_selfAlpha_to_double<<<blockNum, threadNum, sharedMsize>>>(
        vertexes, pairs, moveDir, mqueue + blockNum, slackness, numbers, invalid, invalid_bit, d_live, 0);
    return 2 * blockNum;
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

