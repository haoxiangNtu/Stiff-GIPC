// ============================================================================
// energy/14_soft_constraints.inl — soft-constraint energy reduction
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 9).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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
