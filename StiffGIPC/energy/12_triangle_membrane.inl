// ============================================================================
// energy/12_triangle_membrane.inl — cloth in-plane membrane energy reduction
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 8).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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
