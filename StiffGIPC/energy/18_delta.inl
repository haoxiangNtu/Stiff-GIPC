// ============================================================================
// energy/18_delta.inl — line-search delta (directional) energy reduction
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 3).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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

