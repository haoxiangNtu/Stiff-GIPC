// ============================================================================
// energy/17_ground.inl — ground barrier energy reduction
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 4).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
__global__ void _computeGroundEnergy_Reduction(double*        squeue,
                                               const double3* vertexes,
                                               const double*  g_offset,
                                               const double3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               double dHat,
                                               double Kappa,
                                               int    number,
                                               double* penv = nullptr, const int* p2g = nullptr, int ng = 0,
                                               const uint32_t* d_live = nullptr,
                                               const double* kappa_dev = nullptr)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    if(kappa_dev)   // [C-1 ls-graph] live kappa (see barrier variant)
        Kappa = *kappa_dev;
    // [B3 device-count] see _getBarrierEnergy_Reduction_3D — live count read on
    // device in trial mode; idle threads contribute exact 0.0 (bitwise-neutral).
    if(d_live) number = (int)*d_live;
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
    gipc_block_sum_to(temp, tep, d_live ? (int)(gridDim.x * blockDim.x) : number, idof, squeue + blockIdx.x);  // [C-1] capacity-grid tail
}


// ── verbatim from gipc_modules/06 (pre-E1c lines 215..318) ──
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
                                                 const int*    p2g       = nullptr,
                                                 const int*    offset_dev = nullptr,
                                                 const uint32_t* n_dev   = nullptr,
                                                 const double* kappa_dev = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(kappa_dev)    // [C-3] live kappa
        Kappa_scalar = *kappa_dev;
    if(offset_dev)   // [C-2] device ground assembly base
        global_offset = *offset_dev;
    if(n_dev)        // [C-3] live ground count (GH-start snapshot slot 11)
        number = (int)*n_dev;
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


// ── [E2] registry members for type 4 (ground): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_ground() { return m_energy_use_device_counts ? m_energy_bound_gp : (int)h_gpNum; }  // [B3] trial bound
void GIPC::energy_launch_ground(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal,
                _environment_collisionPair, dHat, Kappa, numbers,
                pe, pe ? p2g : nullptr, ng,
                m_energy_use_device_counts ? _cpNum + 5 : nullptr,
                m_ls_recording ? m_d_ls_scalars + 1 : nullptr);  // [C-1] live kappa
}
