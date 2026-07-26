// ============================================================================
// energy/13_bending.inl — bending energy reductions (USE_QUADRATIC_BENDING quad variant + angle variant)
// (v0.8.6 energy separation E1b; term table energy/energy_terms.h, types 10).
// Verbatim from gipc_modules/07; compiled in the GIPC.cu composite TU at the
// old module-07 position (before the host dispatch in energy/01).
// ============================================================================
#ifdef USE_QUADRATIC_BENDING
__global__ void _getQuadBendingEnergy_Reduction(double*        squeue,
                                                const double3* vertexes,
                                                const double3* rest_vertexex,
                                                const uint2*   edges,
                                                const uint2*   edge_adj_vertex,
                                                const Eigen::Matrix4d* quad_bending_Q,
                                                int    edgesNum,
                                                double bendStiff,
                                                double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        uint2 adj = edge_adj_vertex[idx];
        temp = __cal_quad_bending_energy(
            vertexes, rest_vertexex, edges[idx], adj, quad_bending_Q[idx], bendStiff);
        _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);
    }

    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}
#endif

__global__ void _getBendingEnergy_Reduction(double*        squeue,
                                            const double3* vertexes,
                                            const double3* rest_vertexex,
                                            const uint2*   edges,
                                            const uint2*   edge_adj_vertex,
                                            int            edgesNum,
                                            double         bendStiff,
                                            double* penv = nullptr, const int* p2g = nullptr, int ng = 0)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    int                      numbers = edgesNum;
    double                   temp = 0.0;
    if(idx < numbers)
    {
        uint2   adj     = edge_adj_vertex[idx];
        double3 rest_x0 = rest_vertexex[edges[idx].x];
        double3 rest_x1 = rest_vertexex[edges[idx].y];
        double  length  = __GEIGEN__::__norm(__GEIGEN__::__minus(rest_x0, rest_x1));
        temp = __cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
        _penv_energy_accum(penv, p2g, edges[idx].x, ng, temp);
    }
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    // [v0.8.6 2a] unified tail — see device_common/reductions.cuh
    gipc_block_sum_to(temp, tep, numbers, idof, squeue + blockIdx.x);
}



// ── [E2] registry members for type 10 (bending): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_bending() { return tri_edge_num; }
void GIPC::energy_launch_bending(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
#ifdef USE_QUADRATIC_BENDING
            _getQuadBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex, TetMesh.quad_bending_Q, numbers, bendStiff,
                pe, pe ? p2g : nullptr, ng);
#else
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex, numbers, bendStiff,
                pe, pe ? p2g : nullptr, ng);
#endif
}
