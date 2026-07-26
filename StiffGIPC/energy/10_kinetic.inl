// ============================================================================
// energy/10_kinetic.inl — kinetic energy reduction
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 0).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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

