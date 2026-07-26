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


// ── verbatim from gipc_modules/06 (pre-E1c lines 1..24) ──
__global__ void _calKineticGradient(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _calKineticEnergy(
    double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    double3 deltaX = __GEIGEN__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx]  = make_double3(
        deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}


// ── [E2] registry members for type 0 (kinetic): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_kinetic() { return abd_fem_count_info.fem_point_num; }
void GIPC::energy_launch_kinetic(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset, TetMesh.xTilta + point_offset,
                queue, TetMesh.masses + point_offset, numbers,
                pe, pe ? p2g + point_offset : nullptr, ng);
}
