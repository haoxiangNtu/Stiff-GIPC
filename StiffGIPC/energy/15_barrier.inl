// ============================================================================
// energy/15_barrier.inl — IPC barrier energy reduction (frozen smooth context untouched)
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 2).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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

