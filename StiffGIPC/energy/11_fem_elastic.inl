// ============================================================================
// energy/11_fem_elastic.inl — FEM tet elastic + rest-stable NHK reductions (one constitutive family)
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 1,7).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
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


// ── [E2] registry members for type 1 (fem_elastic): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_fem_elastic() { return abd_fem_count_info.fem_tet_num; }
void GIPC::energy_launch_fem_elastic(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset, TetMesh.volum + tet_offset,
                numbers, TetMesh.lengthRate + tet_offset, TetMesh.volumeRate + tet_offset,
                pe, pe ? p2g : nullptr, ng);
}

// ── [E2] registry members for type 7 (rest_nhk): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_rest_nhk() { return abd_fem_count_info.fem_tet_num; }
void GIPC::energy_launch_rest_nhk(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
}
