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

// ── verbatim from gipc_modules/06 (pre-E1c lines 25..166) ──
__global__ void _computeSoftConstraintGradientAndHessian(const double3* vertexes,
                                                         const double3* targetVert,
                                                         const uint32_t* targetInd,
                                                         double3*  gradient,
                                                         uint32_t* _gpNum,
                                                         Eigen::Matrix3d* triplet_values,
                                                         int*   row_ids,
                                                         int*   col_ids,
                                                         double motionRate,
                                                         double rate,
                                                         int    global_offset,
                                                         int global_hessian_fem_offset,
                                                         const int*     stitch_paired_vertex,
                                                         const double3* stitch_rest_offset,
                                                         const int*     stitch_abd_body_id,
                                                         const __GEIGEN__::Vector12* abd_body_q,
                                                         int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // For bilateral stitch springs, compute target dynamically from current ABD vertex.
    // [stitch local-frame fix] target = anchor_world + A_now * local_offset where
    // local_offset is in the ABD body's rest frame.  Per the canonical q layout in
    // abd_jacobi_matrix.inl operator*(ABDJacobi, Vector12):
    //   q[3..5]  = A.row(0),  q[6..8]  = A.row(1),  q[9..11] = A.row(2)
    // So (A * lo).x = q[3]*lo.x + q[4]*lo.y + q[5]*lo.z, etc.
    // Without this, the stitch target only follows ABD translation, not rotation,
    // so FEM mesh visibly fails to track ABD rotation.
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // Previous code used q[3]/q[6]/q[9] for the x-component, which is
            // A.col(0)·lo = (A^T·lo)[0] — wrong for non-symmetric A.  Only
            // worked when A ≈ scale·I (Animated mode where ABD barely rotates).
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            // Fallback: legacy world-frame offset (no rotation tracking).
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
    __GEIGEN__::Matrix3x3d Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx    = atomicAdd(_gpNum, 1);
    //H3x3[pidx]    = Hpg;
    //D1Index[pidx] = vInd;
    vInd += global_hessian_fem_offset;
    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &vInd, Hpg.m, global_offset + idx);
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradient(const double3*  vertexes,
                                               const double3*  targetVert,
                                               const uint32_t* targetInd,
                                               double3*        gradient,
                                               double          motionRate,
                                               double          rate,
                                               const int*      stitch_paired_vertex,
                                               const double3*  stitch_rest_offset,
                                               const int*      stitch_abd_body_id,
                                               const __GEIGEN__::Vector12* abd_body_q,
                                               int             number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    uint32_t vInd = targetInd[idx];
    double   x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z;
    double   a, b, c;
    // [stitch local-frame fix] see _computeSoftConstraintGradientAndHessian
    if(stitch_paired_vertex && stitch_paired_vertex[idx] >= 0)
    {
        int abd_idx = stitch_paired_vertex[idx];
        double3 lo = stitch_rest_offset[idx];
        if(abd_body_q != nullptr && stitch_abd_body_id != nullptr)
        {
            int bid = stitch_abd_body_id[idx];
            const __GEIGEN__::Vector12& q = abd_body_q[bid];
            // q[3..5]/q[6..8]/q[9..11] = A.row(0/1/2); see _computeSoftConstraintGradientAndHessian.
            a = vertexes[abd_idx].x + q.v[3] * lo.x + q.v[4] * lo.y + q.v[5]  * lo.z;
            b = vertexes[abd_idx].y + q.v[6] * lo.x + q.v[7] * lo.y + q.v[8]  * lo.z;
            c = vertexes[abd_idx].z + q.v[9] * lo.x + q.v[10] * lo.y + q.v[11] * lo.z;
        }
        else
        {
            a = vertexes[abd_idx].x + lo.x;
            b = vertexes[abd_idx].y + lo.y;
            c = vertexes[abd_idx].z + lo.z;
        }
    }
    else
    {
        a = targetVert[idx].x;
        b = targetVert[idx].y;
        c = targetVert[idx].z;
    }
    //double dis = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        _gfxAdd(vInd, 0, d * rate * rate * (x - a));
        _gfxAdd(vInd, 1, d * rate * rate * (y - b));
        _gfxAdd(vInd, 2, d * rate * rate * (z - c));
    }
}


// ── [E2] registry members for type 9 (soft_constraints): launcher body VERBATIM from
// the DeviceOut dispatcher switch; size = its sizing-chain entry ──
int GIPC::energy_size_soft_constraints() { return softNum; }
void GIPC::energy_launch_soft_constraints(device_TetraData& TetMesh, double* queue, int numbers,
                                int blockNum, unsigned int threadNum, unsigned int sharedMsize,
                                double* pe, const int* p2g, int ng,
                                int tet_offset, int point_offset, double energy_kappa)
{
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex,
                softMotionRate, animation_fullRate, TetMesh.d_stitch_paired_vertex,
                TetMesh.d_stitch_rest_offset, numbers,
                pe, pe ? p2g : nullptr, ng);
}
