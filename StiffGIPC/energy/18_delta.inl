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


// ── [E2] registry members for type 3 (delta): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_delta() { return abd_fem_count_info.fem_point_num; }
void GIPC::energy_launch_delta(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
}
