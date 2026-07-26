// ============================================================================
// energy/16_friction.inl — lagged friction energy reductions (self + ground)
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 5,6).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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

