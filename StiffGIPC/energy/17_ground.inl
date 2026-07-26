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

