double GIPC::Energy_Add_Reduction_Algorithm(int type, device_TetraData& TetMesh)
{
    int tet_offset   = abd_fem_count_info.fem_tet_offset;
    int tet_count    = abd_fem_count_info.fem_tet_num;
    int point_offset = abd_fem_count_info.fem_point_offset;
    int point_count  = abd_fem_count_info.fem_point_num;

    int numbers = tet_count;

    if(type == 0 || type == 3)
    {
        numbers = point_count;
    }
    else if(type == 2)
    {
        numbers = h_cpNum[0];
    }
    else if(type == 4)
    {
        numbers = h_gpNum;
    }
    else if(type == 5)
    {
        numbers = h_cpNum_last[0];
    }
    else if(type == 6)
    {
        numbers = h_gpNum_last;
    }
    else if(type == 7 || type == 1)
    {
        numbers = tet_count;
    }
    else if(type == 8 || type == 11)
    {
        numbers = triangleNum;
    }
    else if(type == 9)
    {
        numbers = softNum;
    }
    else if(type == 10)
    {
        numbers = tri_edge_num;
    }
    if(numbers == 0)
        return 0;
    // pair-count energy reductions (barrier/friction) need a pair-sized buffer,
    // not the mesh-sized squeue (V2/V3 overflow fix).
    double* queue = ensure_reduce_scratch(numbers);
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset,
                TetMesh.xTilta + point_offset,
                queue,
                TetMesh.masses + point_offset,
                numbers);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset,
                TetMesh.volum + tet_offset,
                numbers,
                TetMesh.lengthRate + tet_offset,
                TetMesh.volumeRate + tet_offset);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs, Kappa, dHat, numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal, _environment_collisionPair, dHat, Kappa, numbers);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _collisonPairs_lastH,
                numbers,
                IPC_dt,
                distCoord,
                tanBasis,
                lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt,
                sqrt(fDhat) * IPC_dt,
                nullptr, nullptr, 0,
                d_vert_mu, frictionRate);  // [per-body friction]
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.o_vertexes,
                _groundNormal,
                _collisonPairs_lastH_gd,
                numbers,
                IPC_dt,
                lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt,
                nullptr, nullptr, 0,
                d_vert_mu_gd, gd_frictionRate);  // [per-body friction]
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.triangles,
                TetMesh.triDmInverses,
                TetMesh.area,
                numbers,
                stretchStiff,
                shearStiff,
                strainRate);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex, softMotionRate, animation_fullRate,
                TetMesh.d_stitch_paired_vertex, TetMesh.d_stitch_rest_offset, numbers);
            break;
        case 10:
#ifdef USE_QUADRATIC_BENDING
            _getQuadBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                TetMesh.quad_bending_Q,
                numbers,
                bendStiff);
#else
            _getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue,
                TetMesh.vertexes,
                TetMesh.rest_vertexes,
                TetMesh.tri_edges,
                TetMesh.tri_edge_adj_vertex,
                numbers,
                bendStiff);
#endif
            break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


// [backport] standalone per-env energy dispatcher (from S3/4a370e5). Modeled on
// Energy_Add_Reduction_Algorithm but writes the global scalar to a caller-provided
// device slot (out_slot, D2D) and, when out_penv != nullptr, buckets each element's
// energy into per-env slots via the kernels' (penv,p2g,ng) params. Used ONLY by
// computeEnergy_perenv (the per-env line-search path); v0.6.4's computeEnergy is
// untouched. NOTE: not the full ②-D2H batched computeEnergy rewrite — isolated here.
void GIPC::Energy_Add_Reduction_Algorithm_DeviceOut(int               type,
                                                     device_TetraData& TetMesh,
                                                     double*           out_slot,
                                                     double*           out_penv,
                                                     double            energy_kappa)
{
    // [multi-env S3] per-env energy bucket for this term (size kEnvAlphaSlots) and
    // the global point_to_group; passed to the kernels when out_penv != nullptr.
    double*    pe  = out_penv;
    const int* p2g = TetMesh.d_point_to_group;
    const int  ng  = TetMesh.h_group_count;
    if(pe) CUDA_SAFE_CALL(cudaMemsetAsync(pe, 0, ng * sizeof(double)));
    int tet_offset   = abd_fem_count_info.fem_tet_offset;
    int tet_count    = abd_fem_count_info.fem_tet_num;
    int point_offset = abd_fem_count_info.fem_point_offset;
    int point_count  = abd_fem_count_info.fem_point_num;

    int numbers = tet_count;
    if(type == 0 || type == 3)      numbers = point_count;
    else if(type == 2)              numbers = h_cpNum[0];
    else if(type == 4)              numbers = h_gpNum;
    else if(type == 5)              numbers = h_cpNum_last[0];
    else if(type == 6)              numbers = h_gpNum_last;
    else if(type == 7 || type == 1) numbers = tet_count;
    else if(type == 8 || type == 11)numbers = triangleNum;
    else if(type == 9)              numbers = softNum;
    else if(type == 10)             numbers = tri_edge_num;

    if(numbers == 0)
    {
        // Match original `return 0;` behavior — pre-zero the slot.
        CUDA_SAFE_CALL(cudaMemsetAsync(out_slot, 0, sizeof(double)));
        return;
    }

    // Pair-count types (2/5) can exceed the mesh-sized squeue block capacity —
    // use the growable pair-capacity scratch (V2/V3 overflow fix), like the
    // blocking variant.
    double*            queue       = ensure_reduce_scratch(numbers);
    const unsigned int threadNum   = 256;
    int                blockNum    = (numbers + threadNum - 1) / threadNum;
    unsigned int       sharedMsize = sizeof(double) * (threadNum >> 5);

    switch(type)
    {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                TetMesh.vertexes + point_offset, TetMesh.xTilta + point_offset,
                queue, TetMesh.masses + point_offset, numbers,
                pe, pe ? p2g + point_offset : nullptr, ng);
            break;
        case 1:
            _getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.tetrahedras + tet_offset,
                TetMesh.DmInverses + tet_offset, TetMesh.volum + tet_offset,
                numbers, TetMesh.lengthRate + tet_offset, TetMesh.volumeRate + tet_offset,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 2:
            _getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.rest_vertexes, _collisonPairs,
                energy_kappa >= 0.0 ? energy_kappa : Kappa, dHat, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.fb + point_offset, _moveDir + point_offset, numbers);
            break;
        case 4:
            _computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, _groundOffset, _groundNormal,
                _environment_collisionPair, dHat, Kappa, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 5:
            _getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.o_vertexes, _collisonPairs_lastH,
                numbers, IPC_dt, distCoord, tanBasis, lambda_lastH_scalar,
                fDhat * IPC_dt * IPC_dt, sqrt(fDhat) * IPC_dt,
                pe, pe ? p2g : nullptr, ng,
                d_vert_mu, frictionRate);  // [per-body friction]
            break;
        case 6:
            _getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.o_vertexes, _groundNormal,
                _collisonPairs_lastH_gd, numbers, IPC_dt, lambda_lastH_scalar_gd,
                sqrt(fDhat) * IPC_dt,
                pe, pe ? p2g : nullptr, ng,
                d_vert_mu_gd, gd_frictionRate);  // [per-body friction]
            break;
        case 7:
            _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.volum + tet_offset, numbers, lengthRate, volumeRate);
            break;
        case 8:
            _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.triangles, TetMesh.triDmInverses,
                TetMesh.area, numbers, stretchStiff, shearStiff, strainRate,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 9:
            _computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, TetMesh.vertexes, TetMesh.targetVert, TetMesh.targetIndex,
                softMotionRate, animation_fullRate, TetMesh.d_stitch_paired_vertex,
                TetMesh.d_stitch_rest_offset, numbers,
                pe, pe ? p2g : nullptr, ng);
            break;
        case 10:
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
            break;
    }

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while(numbers > 1)
    {
        __add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }

    // D2D copy queue[0] into the caller's slot — queued on PTDS, async.
    // Next call's reduction kernel will not start until this D2D completes
    // (stream ordering), so reusing `queue` for the next call is safe.
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_slot, queue, sizeof(double),
                                   cudaMemcpyDeviceToDevice));
}


// Preserve the original host expression's operation order explicitly.  The
// project is compiled with --use_fast_math, so round-mode intrinsics also keep
// the compiler from reassociating a near-boundary line-search comparison.
__global__ void _global_energy_combine(const double* slots,
                                       double        dt2,
                                       double        Kappa,
                                       double        friction_rate,
                                       double        ground_friction_rate,
                                       double*       out)
{
    double e = 0.0;
    e = __dadd_rn(e, slots[0]);   // FEM kinetic
    e = __dadd_rn(e, slots[9]);   // ABD kinetic
    e = __dadd_rn(e, slots[10]);  // ABD shape
    e = __dadd_rn(e, slots[11]);  // ABD joint
    e = __dadd_rn(e, slots[12]);  // ABD revolute driving
    e = __dadd_rn(e, slots[13]);  // ABD prismatic
    e = __dadd_rn(e, slots[14]);  // ABD prismatic driving
    e = __dadd_rn(e, __dmul_rn(dt2, slots[1]));
    e = __dadd_rn(e, __dmul_rn(dt2, slots[2]));
    e = __dadd_rn(e, __dmul_rn(dt2, slots[3]));
    e = __dadd_rn(e, slots[4]);
    e = __dadd_rn(e, slots[5]);
    e = __dadd_rn(e, __dmul_rn(Kappa, slots[6]));
#ifdef USE_FRICTION
    e = __dadd_rn(e, __dmul_rn(friction_rate, slots[7]));
    e = __dadd_rn(e, __dmul_rn(ground_friction_rate, slots[8]));
#else
    (void)friction_rate;
    (void)ground_friction_rate;
#endif
    *out = e;
}

void GIPC::computeEnergy_DeviceOut(device_TetraData& TetMesh, double* out_scalar)
{
    // slots: 0..8 FEM/contact, 9..14 ABD in the exact order consumed above.
    Energy_Add_Reduction_Algorithm_DeviceOut(0,  TetMesh, m_energy_slots + 0);
    Energy_Add_Reduction_Algorithm_DeviceOut(1,  TetMesh, m_energy_slots + 1);
    Energy_Add_Reduction_Algorithm_DeviceOut(8,  TetMesh, m_energy_slots + 2);
    Energy_Add_Reduction_Algorithm_DeviceOut(10, TetMesh, m_energy_slots + 3);
    Energy_Add_Reduction_Algorithm_DeviceOut(9,  TetMesh, m_energy_slots + 4);
    Energy_Add_Reduction_Algorithm_DeviceOut(2,  TetMesh, m_energy_slots + 5);
    Energy_Add_Reduction_Algorithm_DeviceOut(4,  TetMesh, m_energy_slots + 6);
#ifdef USE_FRICTION
    Energy_Add_Reduction_Algorithm_DeviceOut(5,  TetMesh, m_energy_slots + 7);
    Energy_Add_Reduction_Algorithm_DeviceOut(6,  TetMesh, m_energy_slots + 8);
#endif
    m_abd_system->cal_abd_energy_DeviceOut(*m_abd_sim_data, m_energy_slots + 9);

    _global_energy_combine<<<1, 1>>>(m_energy_slots,
                                     IPC_dt * IPC_dt,
                                     Kappa,
                                     frictionRate,
                                     gd_frictionRate,
                                     out_scalar);

    static bool energy_validated = false;
    if(!energy_validated && getenv("STIFF_ENERGY_VALIDATE"))
    {
        double slots[kEnergySlotCount] = {};
        double device_energy = 0.0;
        CUDA_SAFE_CALL(cudaMemcpy(slots,
                                  m_energy_slots,
                                  sizeof(slots),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&device_energy,
                                  out_scalar,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost));
        const double dt2 = IPC_dt * IPC_dt;
        double host_energy = 0.0;
        host_energy += slots[0];
        host_energy += slots[9];
        host_energy += slots[10];
        host_energy += slots[11];
        host_energy += slots[12];
        host_energy += slots[13];
        host_energy += slots[14];
        host_energy += dt2 * slots[1];
        host_energy += dt2 * slots[2];
        host_energy += dt2 * slots[3];
        host_energy += slots[4];
        host_energy += slots[5];
        host_energy += Kappa * slots[6];
#ifdef USE_FRICTION
        host_energy += frictionRate * slots[7];
        host_energy += gd_frictionRate * slots[8];
#endif
        const bool exact = std::memcmp(&host_energy,
                                       &device_energy,
                                       sizeof(double)) == 0;
        printf("[energy-device-validate] exact=%d host=%.17e device=%.17e\n",
               exact ? 1 : 0,
               host_energy,
               device_energy);
        if(!exact)
            throw std::runtime_error(
                "[line-search] device energy combine differs from host order");
        energy_validated = true;
    }
}

double GIPC::computeEnergy(device_TetraData& TetMesh)
{
    // Diagnostics and legacy callers still receive a host scalar, but all 15
    // reductions and their combine now incur only this single D2H. This
    // compatibility scalar is deliberately separate from line-search E0/Etrial.
    computeEnergy_DeviceOut(TetMesh, m_compatibility_energy);
    double energy = 0.0;
    CUDA_SAFE_CALL(cudaMemcpy(&energy,
                              m_compatibility_energy,
                              sizeof(double),
                              cudaMemcpyDeviceToHost));
    return energy;
}

// [multi-env S3] per-env total energy E_g into env_out[kEnvAlphaSlots].
// FEM terms via the per-env-instrumented reductions (each element's energy added
// to its env bucket); ABD terms added per-env (task #2 — currently lumped into a
// validation-only global until per-body ABD energy lands). Returns the GLOBAL
// energy. Built-in correctness gate (gated STIFF_PENV_STATS): Sum_g E_g(FEM) must
// equal the global FEM energy (independent computeEnergy minus ABD).
// [de-CPU S3] device combine of the pe_all slices -> per-env energy E_g. One thread per env;
// accumulation ORDER AND FACTORS replicate the host combine exactly (kinetic, dt2*fem, dt2*tri,
// dt2*bend, constraint, barrier(kappa-scaled), ground(kappa-scaled), friction, ABD) -> E_g is
// bit-identical to the host env_out. Slices: 0=kin 1=fem 2=tri 3=bend 4=cons 5=barrier 6=fricS
// 7=fricGd 8=ground 9=kappa_group 10=abd.
__global__ void _perenv_energy_combine(const double* pe,
                                       double*       out,
                                       double        Kappa,
                                       double        dt2,
                                       double        fr,
                                       double        gfr,
                                       int           perenv_k,
                                       int           stride,
                                       int           ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double kg = perenv_k ? pe[(size_t)9 * stride + g] : Kappa;
    double e  = 0.0;
    e += pe[(size_t)0 * stride + g];
    e += dt2 * pe[(size_t)1 * stride + g];
    e += dt2 * pe[(size_t)2 * stride + g];
    e += dt2 * pe[(size_t)3 * stride + g];
    e += pe[(size_t)4 * stride + g];
    e += (perenv_k ? kg : 1.0) * pe[(size_t)5 * stride + g];
    e += (perenv_k ? kg : Kappa) * pe[(size_t)8 * stride + g];
#ifdef USE_FRICTION
    e += fr * pe[(size_t)6 * stride + g];
    e += gfr * pe[(size_t)7 * stride + g];
#endif
    e += pe[(size_t)10 * stride + g];
    out[g] = e;
}
// [de-CPU S3] per-env backtrack decision ON DEVICE, operating on the TRUE in-frame alpha state
// m_env_alpha (the old host loop read the h_env_alpha MIRROR, which is STALE in the fast S1 path —
// its failure branch would have H2D'd stale alphas over the device state; dormant only because S3
// virtually always accepts at bt=0). Uses the shared configured comparison and halving rule.
__global__ void _s3_decide(const double* Eg0,
                           const double* Eg1,
                           double*       env_alpha,
                           int*          decision_counts,
                           int           ng,
                           double        energy_abs_tol,
                           double        energy_rel_tol)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] <= 0.0) return;               // absent (or frozen) env
    double tol = energy_abs_tol + energy_rel_tol * fabs(Eg0[g]);
    if(Eg1[g] > Eg0[g] + tol)
    {
        env_alpha[g] *= 0.5;
        atomicAdd(decision_counts, 1);
    }
    else if(Eg1[g] > Eg0[g])
        atomicAdd(decision_counts + 1, 1);
}

// Standard (uniform-alpha) line-search decision.  Status: 0=descent,
// 1=retry/exhausted, 2=accepted only by configured roundoff tolerance.
__global__ void _global_ls_decide(const double* energy0,
                                  const double* energy1,
                                  double        c1m,
                                  double        alpha,
                                  double        energy_abs_tol,
                                  double        energy_rel_tol,
                                  int*          status)
{
    const double e0  = *energy0;
    const double e1  = *energy1;
    const double rhs = __dadd_rn(e0, __dmul_rn(c1m, alpha));
    const double tol = __dadd_rn(energy_abs_tol,
                                 __dmul_rn(energy_rel_tol, fabs(e0)));
    // [NaN-quarantine gap fix] any comparison against NaN is false, so a
    // NaN trial energy used to fall through to status 0 = "accepted descent"
    // — the one path where a diverged env's NaN could slip past the per-env
    // quarantine. Non-finite trial ⇒ status 1 (keep backtracking; on budget
    // exhaustion the loud non-descent warning fires instead of silence).
    *status = !isfinite(e1) ? 1
              : (e1 > __dadd_rn(rhs, tol) ? 1 : (e1 > rhs ? 2 : 0));
}
// [de-CPU S3] intersect-safety halving (was: host loop over the stale mirror + H2D).
__global__ void _s3_halve_all(double* env_alpha, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    if(env_alpha[g] > 0.0) env_alpha[g] *= 0.5;
}

// [de-CPU S3] shared term-launcher for the per-env energy: fills the pe_all slice block on device
// (see slice layout below) and reports whether per-env kappa rescale applies. Used by BOTH the
// host-combine (validation) and the device-combine (S3 decision) variants.
double* GIPC::_launch_perenv_energy_terms(device_TetraData& TetMesh, bool& perenv_k_out)
{
    const int NG = TetMesh.h_group_count;
    constexpr int PE_STRIDE = kEnvAlphaSlots;
    // [perf/de-CPU] SINGLE-D2H layout: every term's per-env bucket goes to its own NG-slice of one
    // device block; the κ-group + ABD arrays are staged into slices too; then ONE blocking D2H of
    // the whole block replaces the previous 12 per-term blocking D2H round-trips (the dominant
    // lineSearch host-mixing). The term kernels/launch order/host combine order+factors are
    // UNCHANGED → every env_out[g] is bit-identical to the per-term-D2H version.
    // slices: 0=kinetic 1=fem 2=tri_fem 3=bend 4=constraint 5=barrier 6=fricS 7=fricGd
    //         8=ground 9=kappa_group 10=abd
    constexpr int PE_SLOTS = 11;
    static double* pe_all = nullptr;
    if(!pe_all)
        CUDA_SAFE_CALL(cudaMalloc(
            (void**)&pe_all, (size_t)PE_SLOTS * PE_STRIDE * sizeof(double)));
    // [backport] throwaway sink for DeviceOut's global scalar (ignored here; the
    // per-env path uses the pe slices, and full_sum is assembled on host below).
    static double* g_sink = nullptr;
    if(!g_sink) CUDA_SAFE_CALL(cudaMalloc((void**)&g_sink, sizeof(double)));

    auto slice = [&](int s) { return pe_all + (size_t)s * PE_STRIDE; };
    bool perenv_k = getenv("STIFF_DECOUPLE_THRESH")
                 && m_pergroup_kappa && m_kappa_group;
    Energy_Add_Reduction_Algorithm_DeviceOut(0,  TetMesh, g_sink, slice(0));   // kinetic
    Energy_Add_Reduction_Algorithm_DeviceOut(1,  TetMesh, g_sink, slice(1));   // fem elastic
    Energy_Add_Reduction_Algorithm_DeviceOut(8,  TetMesh, g_sink, slice(2));   // tri_fem
    Energy_Add_Reduction_Algorithm_DeviceOut(10, TetMesh, g_sink, slice(3));   // bend
    Energy_Add_Reduction_Algorithm_DeviceOut(9,  TetMesh, g_sink, slice(4));   // constraint
    // [decouple] barrier + ground energy must use PER-ENV kappa, matching the gradient. Compute
    // barrier's raw unit-kappa energy directly; multiplying by global Kappa and then dividing it
    // back out is only algebraically equivalent and leaks batch-dependent floating-point bits.
    Energy_Add_Reduction_Algorithm_DeviceOut(
        2, TetMesh, g_sink, slice(5), perenv_k ? 1.0 : -1.0);                 // barrier
    Energy_Add_Reduction_Algorithm_DeviceOut(4,  TetMesh, g_sink, slice(8));   // ground (raw)
#ifdef USE_FRICTION
    Energy_Add_Reduction_Algorithm_DeviceOut(5,  TetMesh, g_sink, slice(6));   // friction (self)
    Energy_Add_Reduction_Algorithm_DeviceOut(6,  TetMesh, g_sink, slice(7));   // friction (ground)
#endif
    if(perenv_k)   // κ-group staged into the block (D2D) — read back with the same single D2H
        CUDA_SAFE_CALL(cudaMemcpyAsync(slice(9), m_kappa_group, NG * sizeof(double),
                                       cudaMemcpyDeviceToDevice));
    // [S3] per-env ABD energy (segment-summed by body_to_group in the subsystem) → slice 10.
    CUDA_SAFE_CALL(cudaMemsetAsync(slice(10), 0, NG * sizeof(double)));
    double abd_total = m_abd_system->cal_abd_energy_perenv(
        *m_abd_sim_data,
        TetMesh.d_body_to_group,
        NG,
        slice(10),
        false);  // per-env/device line search does not need the global host scalar

    (void)abd_total;
    perenv_k_out = perenv_k;
    return pe_all;
}

double GIPC::computeEnergy_perenv(device_TetraData& TetMesh, std::vector<double>& env_out)
{
    const int NG = TetMesh.h_group_count;
    constexpr int PE_STRIDE = kEnvAlphaSlots;
    env_out.assign(NG, 0.0);
    if(!TetMesh.d_point_to_group)
        return computeEnergy(TetMesh);  // no groups -> nothing to decompose
    constexpr int PE_SLOTS = 11;
    bool    perenv_k = false;
    double* pe_all   = _launch_perenv_energy_terms(TetMesh, perenv_k);

    // THE one blocking D2H of everything.
    static std::vector<double> hb;
    hb.resize((size_t)PE_SLOTS * PE_STRIDE);
    CUDA_SAFE_CALL(cudaMemcpy(hb.data(), pe_all, (size_t)PE_SLOTS * PE_STRIDE * sizeof(double),
                              cudaMemcpyDeviceToHost));

    // Host combine — SAME order and factors as the per-term version (bit-identical env_out).
    const double dt2 = IPC_dt * IPC_dt;
    auto hs = [&](int s) { return hb.data() + (size_t)s * PE_STRIDE; };
    for(int g = 0; g < NG; ++g) env_out[g] += 1.0 * hs(0)[g];   // kinetic
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(1)[g];   // fem elastic
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(2)[g];   // tri_fem
    for(int g = 0; g < NG; ++g) env_out[g] += dt2 * hs(3)[g];   // bend
    for(int g = 0; g < NG; ++g) env_out[g] += 1.0 * hs(4)[g];   // constraint
    // barrier: per-env launch stores raw unit-kappa energy; scale exactly once by kappa_g.
    for(int g = 0; g < NG; ++g)
        env_out[g] += (perenv_k ? hs(9)[g] : 1.0) * hs(5)[g];
    // ground (h[g] = raw_g, kernel does not apply Kappa): per-env → kappa_g*raw_g
    for(int g = 0; g < NG; ++g)
        env_out[g] += (perenv_k ? hs(9)[g] : Kappa) * hs(8)[g];
#ifdef USE_FRICTION
    for(int g = 0; g < NG; ++g) env_out[g] += frictionRate * hs(6)[g];      // friction (self)
    for(int g = 0; g < NG; ++g) env_out[g] += gd_frictionRate * hs(7)[g];   // friction (ground)
#endif
    double abd_sum = 0.0;
    for(int g = 0; g < NG; ++g) { env_out[g] += hs(10)[g]; abd_sum += hs(10)[g]; }

    double full_sum = 0.0;
    for(int g = 0; g < NG; ++g) full_sum += env_out[g];

    // full_sum == global computeEnergy (validated machine-precision). Only run the
    // independent computeEnergy for the gated validation print (it ~doubles cost,
    // and the per-env backtracking loop calls this repeatedly).
    if(getenv("STIFF_S3_VALIDATE"))
    {
        double E_global = computeEnergy(TetMesh);
        printf("[S3-energy] sum_g E_g=%.9e  global=%.9e  rel=%.2e  (ABD sum_g=%.6e)\n",
               full_sum, E_global,
               fabs(full_sum - E_global) / std::max(fabs(E_global), 1e-30), abd_sum);
        return E_global;
    }
    return full_sum;
}

void GIPC::computeEnergy_perenv_dev(device_TetraData& TetMesh, double* d_Eg)
{
    const int NG = TetMesh.h_group_count;
    bool    perenv_k = false;
    double* pe_all   = _launch_perenv_energy_terms(TetMesh, perenv_k);
    _perenv_energy_combine<<<(NG + 255) / 256, 256>>>(
        pe_all,
        d_Eg,
        Kappa,
        IPC_dt * IPC_dt,
        frictionRate,
        gd_frictionRate,
        perenv_k ? 1 : 0,
        kEnvAlphaSlots,
        NG);
}

// [4.3 debug] FNV-1a hash of a device buffer (host-copy, deterministic) to bisect the residual
// non-atomic non-determinism. STIFF_KSUM=1 prints checksums; run twice + diff to localize.
static void _dbg_ksum(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t acc = 1469598103934665603ULL;
    for(uint64_t v : h) { acc ^= v; acc *= 1099511628211ULL; }
    printf("[ksum] %-14s %016llx\n", name, (unsigned long long)acc);
}
// COMMUTATIVE checksum (order-independent): sum+xor of bit-patterns. Tells value-non-det from
// order-non-det on the raw triplets (which sit at non-deterministic slots).
static void _dbg_ksum_comm(const char* name, const void* dptr, size_t nbytes)
{
    if(!getenv("STIFF_KSUM") || !dptr || nbytes == 0) return;
    std::vector<uint64_t> h((nbytes + 7) / 8, 0);
    cudaMemcpy(h.data(), dptr, nbytes, cudaMemcpyDeviceToHost);
    uint64_t s = 0, x = 0;
    for(uint64_t v : h) { s += v; x ^= v; }
    printf("[ksum] %-14s sum=%016llx xor=%016llx\n", name, (unsigned long long)s, (unsigned long long)x);
}

// [decouple probe] env0-masked COMMUTATIVE (order-free) hash of the RAW Hessian triplet VALUES.
// Masks triplets whose row block is a FEM vertex in env0 (p2g[row]==0). Order-free integer
// bit-sum ⇒ batch-dependent triplet ORDER is irrelevant; only the env0 VALUE multiset + COUNT
// matter. Compare A-vs-B: ntrip differs⇒contact-pair SET differs; count same+hash differs⇒a
// per-pair value differs; both same⇒merge/preconditioner is the seed (values are bit-exact).
static void _dbg_hess_env0(const void* vals9, const int* rows, const int* cols, long ntrip, const int* p2g_dev, int vN,
                           long b_end = -1, long f_end = -1, long g_end = -1)
{
    std::vector<double> hv((size_t)ntrip * 9);
    std::vector<int>    hr(ntrip), hc(ntrip), hp(vN);
    cudaMemcpy(hv.data(), vals9, (size_t)ntrip * 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), rows, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hc.data(), cols, (size_t)ntrip * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hp.data(), p2g_dev, (size_t)vN * sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t src_sum[4] = {0,0,0,0}; long src_cnt[4] = {0,0,0,0};  // 0=BARRIER 1=FRICTION 2=GROUND 3=FEM
    // split: INTRA = row∈env0 AND col∈env0 ; CROSS = row∈env0 AND col∉env0 (cross-env leak)
    uint64_t si = 0, xi = 0, sc = 0, xc = 0; long ci = 0, cc = 0;
    long col_neg1 = 0, col_oob = 0, col_mate = 0;  // CROSS col classification
    long mate_min = (long)ntrip + 1, mate_max = -1;  // triplet-index range of cross-env-mate triplets
    for(long t = 0; t < ntrip; ++t)
    {
        int r = hr[t], c = hc[t];
        bool re = (r >= 0 && r < vN && hp[r] == 0);
        if(!re) continue;
        bool ce = (c >= 0 && c < vN && hp[c] == 0);
        uint64_t hh = 0, hx = 0;
        for(int e = 0; e < 9; ++e) { uint64_t b; memcpy(&b, &hv[(size_t)t * 9 + e], 8); hh += b; hx ^= b; }
        if(ce) { ++ci; si += hh; xi ^= hx;
                 if(b_end >= 0) { int s = (t < b_end) ? 0 : (t < f_end) ? 1 : (t < g_end) ? 2 : 3;
                                  src_sum[s] += hh; ++src_cnt[s]; } }
        else
        {
            ++cc; sc += hh; xc ^= hx;
            if(c < 0 || c >= vN) ++col_oob;        // col out-of-range (ABD body dof? ground?)
            else if(hp[c] < 0) ++col_neg1;          // col ungrouped (-1): env0 gripper/static
            else { ++col_mate;                       // col in a DIFFERENT env (1/2/3): TRUE cross-env
                   if(t < mate_min) mate_min = t; if(t > mate_max) mate_max = t; }
        }
    }
    printf("[ksum-env0] INTRA: ntrip=%ld sum=%016llx | CROSS: ntrip=%ld | by-src BARRIER(n=%ld,s=%016llx) FRICTION(n=%ld,s=%016llx) GROUND(n=%ld,s=%016llx) FEM(n=%ld,s=%016llx)\n",
           ci, (unsigned long long)si, cc,
           src_cnt[0], (unsigned long long)src_sum[0], src_cnt[1], (unsigned long long)src_sum[1],
           src_cnt[2], (unsigned long long)src_sum[2], src_cnt[3], (unsigned long long)src_sum[3]);
}

int GIPC::calculateMovingDirection(device_TetraData& TetMesh, int cpNum, int preconditioner_type)
{
    gipc::Timer timer{"solve_linear_system"};
    auto        iter = 0;

    if(getenv("STIFF_HESS_ENV0") && TetMesh.d_point_to_group
       && g_dec_frame == (getenv("STIFF_DUMP_FRAME") ? atoi(getenv("STIFF_DUMP_FRAME")) : -1)
       && g_dec_k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
    {
        cudaDeviceSynchronize();
        { extern unsigned long long get_xskip(); printf("[xskip] cross-env pairs skipped so far = %llu\n", get_xskip());
          long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          printf("[tri-bounds] BARRIER=[0,%ld) FRICTION=[%ld,%ld) GROUND=[%ld,%ld) FEM=[%ld,end) total=%lld\n",
                 Bb, Bb, Bb+Ff, Bb+Ff, Bb+Ff+(long)h_gpNum, Bb+Ff+(long)h_gpNum, (long long)gipc_global_triplet.global_triplet_offset); }
        // [corrected] mask by d_dof_to_group (the BLOCK index space the triplet row/col live in:
        // ABD blocks first, then FEM vertex blocks), NOT d_point_to_group (per-vertex, no ABD prefix).
        { long Bb = (long)h_cpNum[4]*M12_Off + (long)h_cpNum[3]*M9_Off + (long)h_cpNum[2]*M6_Off;
          long Ff = (long)h_cpNum_last[4]*M12_Off + (long)h_cpNum_last[3]*M9_Off + (long)h_cpNum_last[2]*M6_Off + (long)h_gpNum_last;
          _dbg_hess_env0(gipc_global_triplet.block_values(), gipc_global_triplet.block_row_indices(),
                       gipc_global_triplet.block_col_indices(),
                       (long)gipc_global_triplet.global_triplet_offset, TetMesh.d_dof_to_group, TetMesh.dof_block_count,
                       Bb, Bb+Ff, Bb+Ff+(long)h_gpNum); }
    }

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        _dbg_ksum("fb_in",      TetMesh.fb,          (size_t)vertexNum * sizeof(double3));
        _dbg_ksum("shapegrad_in", TetMesh.shape_grads, (size_t)vertexNum * sizeof(double3));
        // COMMUTATIVE hash of the RAW triplets (block values + row/col), order-independent →
        // tells if the Hessian VALUE-multiset is deterministic (vs just non-det slot order).
        _dbg_ksum_comm("rawHess_comm", gipc_global_triplet.block_values(),
                       (size_t)gipc_global_triplet.global_triplet_offset * 9 * sizeof(double));
        _dbg_ksum_comm("rawRow_comm", gipc_global_triplet.block_row_indices(),
                       (size_t)gipc_global_triplet.global_triplet_offset * sizeof(int));
        printf("[ksum] cpNum=%d Kappa=%.17g toff=%lld\n",
               h_cpNum[0], Kappa, (long long)gipc_global_triplet.global_triplet_offset);
    }

    // [xenv] gradient RHS asymmetry — is fb already env-asymmetric BEFORE the solve?
    // (fb is assembled by computeGradientAndHessian from identical co-located verts.)
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(TetMesh.fb, "fb(grad) pre-solve");

    iter = m_global_linear_system->solve_linear_system();

    // [xenv] search-direction asymmetry — did the SOLVE (SpMV/dot/precond) introduce it?
    if(getenv("STIFF_XENV") && m_d_p2g) xenvDiff(_moveDir, "moveDir post-solve");

    if(getenv("STIFF_KSUM"))
    {
        cudaDeviceSynchronize();
        // merged matrix (converter output, what the spmv used) + the solve result
        _dbg_ksum("matrix_merged", gipc_global_triplet.block_values(),
                  (size_t)gipc_global_triplet.h_unique_key_number * 9 * sizeof(double));
        printf("[ksum] nuniq_merged=%d\n", gipc_global_triplet.h_unique_key_number);
        _dbg_ksum("moveDir_out", _moveDir, (size_t)vertexNum * sizeof(double3));
    }


    auto& json = gipc::Statistics::instance().at_current_frame();
    json["newton"].back()["pcg"]["iterations"] = iter;
    return iter;
}


bool edgeTriIntersectionQuery(const int*     _bodyId,
                              const int*     _btype,
                              const double3* _vertexes,
                              const uint2*   _edges,
                              const uint3*   _faces,
                              const AABB*    _edge_bvs,
                              const Node*    _edge_nodes,
                              double         dHat,
                              int            number,
                              const int*     _collision_skip_matrix,
                              int            _collision_body_count,
                              const int*     _body_id_to_is_fem)
{
    int numbers = number;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;
    int*               _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery<<<blockNum, threadNum>>>(
        _bodyId, _btype, _vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers,
        _collision_skip_matrix, _collision_body_count, _body_id_to_is_fem);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(device_TetraData& TetMesh)
{
    return edgeTriIntersectionQuery(bvh_e._bodyId,
                                    bvh_e._btype,
                                    TetMesh.vertexes,
                                    bvh_e._edges,
                                    bvh_f._faces,
                                    bvh_e._bvs,
                                    bvh_e._nodes,
                                    dHat,
                                    bvh_f.face_number,
                                    bvh_e._collision_skip_matrix,
                                    bvh_e._collision_body_count,
                                    _body_id_to_is_fem);
}

bool GIPC::checkGroundIntersection()
{
    int numbers = h_gpNum;
    if(numbers <= 0)
        return false;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        _vertexes, _groundOffset, _groundNormal, _environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if(h_isITST < 0)
    {
        return true;
    }
    return false;
}

int GIPC::groundTrialStatus(const int* point_to_group, int group_count)
{
    if(m_skip_all_collision || surf_vertexNum == 0 || !m_ground_trial_invalid)
        return 0;
    const bool per_env = point_to_group && m_env_ground_trial_invalid && group_count > 0;
    CUDA_SAFE_CALL(cudaMemsetAsync(m_ground_trial_invalid, 0, sizeof(int)));
    if(per_env)
        CUDA_SAFE_CALL(cudaMemsetAsync(
            m_env_ground_trial_invalid, 0, group_count * sizeof(int)));
    const int threads = 256;
    _markGroundTrialInvalid<<<(surf_vertexNum + threads - 1) / threads, threads>>>(
        _vertexes,
        _surfVerts,
        _groundOffset,
        _groundNormal,
        _point_body_id,
        _ground_skip_body,
        _ground_body_count,
        per_env ? point_to_group : nullptr,
        per_env ? m_env_ground_trial_invalid : nullptr,
        m_ground_trial_invalid,
        group_count,
        surf_vertexNum);
    int status = 0;
    CUDA_SAFE_CALL(cudaMemcpy(
        &status, m_ground_trial_invalid, sizeof(int), cudaMemcpyDeviceToHost));
    return status;
}

void GIPC::halveGroundInvalidEnvAlpha(int group_count)
{
    if(group_count <= 0 || !m_env_alpha || !m_env_ground_trial_invalid)
        return;
    const int threads = 256;
    _halveGroundInvalidEnvAlpha<<<(group_count + threads - 1) / threads, threads>>>(
        m_env_alpha, m_env_ground_trial_invalid, group_count);
}

bool GIPC::isIntersected(device_TetraData& TetMesh)
{
    if(m_skip_all_collision)
        return false;

    // CCD line-search already constrains alpha to a non-intersecting step.
    // The line-search-tail isIntersected() check is a paranoid second pass
    // that re-runs _edgeTriIntersectionQuery (42% of GPU time in case39)
    // for every line-search alpha bisection.  On smooth-contact scenes it
    // never fires (verified across 9/9 paired runs on case39).
    //
    // Default: SKIP the recheck (was opt-in via STIFF_SKIP_CCD_SANITY=1 in
    // dc11e10).  Set GIPC_FORCE_CCD_SANITY=1 to restore the v0.6-and-earlier
    // behavior of running the check.  STIFF_SKIP_CCD_SANITY=0 also restored
    // (back-compat); any other value or unset = skip.
    // [fail-fast][2026-07-08] We tried default-ON: on a plain scene (single FEM
    // bunny dropped on the ground) the edge-tri recheck reports intersections
    // every bisection ("type 0 intersection happened" x260k) and the alpha loop
    // never exits -> hang. The checker is not usable as a default in its current
    // state (false positives / self-intersection sensitivity), which is the real
    // reason it was disabled — document this instead of hiding it. Default stays
    // OFF; GIPC_FORCE_CCD_SANITY=1 opts in for debugging. The ground-collapse
    // invariant is enforced separately by the buildCP d-floor throw.
    static const bool keep_sanity = []{
        const char* v_force = std::getenv("GIPC_FORCE_CCD_SANITY");
        if(v_force && v_force[0] && v_force[0] != '0') return true;
        return false;
    }();
    if(!keep_sanity) return false;

    if(checkGroundIntersection())
    {
        return true;
    }

    if(checkEdgeTriIntersectionIfAny(TetMesh))
    {
        std::cout << "is edge triangle\n";
        return true;
    }
    return false;
}




void GIPC::tempMalloc_closeConstraint()
{
    // [0be8da3-port, grow-only] buffers persist across sub-iterations/frames;
    // (re)allocate only on growth. Kernels touch [0, h_gpNum)/[0, h_cpNum[0]).
    if((size_t)h_gpNum > m_close_gp_cap)
    {
        if(_closeConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeConstraintVal));
        }
        size_t n = (size_t)h_gpNum + h_gpNum / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintID, n * sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeConstraintVal, n * sizeof(double)));
        m_close_gp_cap = n;
    }
    if((size_t)h_cpNum[0] > m_close_cp_cap)
    {
        if(_closeMConstraintID)
        {
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintID));
            CUDA_SAFE_CALL(cudaFree(_closeMConstraintVal));
        }
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintID, n * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_closeMConstraintVal, n * sizeof(double)));
        m_close_cp_cap = n;
    }
}

void GIPC::tempFree_closeConstraint()
{
    // [0be8da3-port] no-op — buffers persist; freed in FREE_DEVICE_MEM.
}

void GIPC::ensure_frictionBuffers()
{
    // [0be8da3-port, grow-only] replaces the per-frame/per-sub-iter free+malloc
    // of the 7 friction lastH buffers. distCoord's live range [0, h_cpNum[0])
    // is re-zeroed on EVERY call (a fresh cudaMalloc'd buffer was memset the
    // same way), keeping the [4.3] frame-0 lag fix value-identical.
    if((size_t)h_cpNum[0] > m_fric_cp_cap)
    {
        if(lambda_lastH_scalar)
        {
            CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
            CUDA_SAFE_CALL(cudaFree(distCoord));
            CUDA_SAFE_CALL(cudaFree(tanBasis));
            CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
        }
        size_t n = (size_t)h_cpNum[0] + h_cpNum[0] / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, n * sizeof(double2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, n * sizeof(__GEIGEN__::Matrix3x2d)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, n * sizeof(int4)));
        m_fric_cp_cap = n;
    }
    if((size_t)h_gpNum > m_fric_gd_cap)
    {
        if(lambda_lastH_scalar_gd)
        {
            CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
            CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
        }
        size_t n = (size_t)h_gpNum + h_gpNum / 4;
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, n * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, n * sizeof(uint32_t)));
        m_fric_gd_cap = n;
    }
    if(h_cpNum[0])
        CUDA_SAFE_CALL(cudaMemset(distCoord, 0, h_cpNum[0] * sizeof(double2)));  // [4.3] frame-0 lag uninit
}

void GIPC::updateVelocities(device_TetraData& TetMesh)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.o_vertexes, TetMesh.velocities, TetMesh.BoundaryType, IPC_dt, numbers);

    m_abd_system->update_velocity(*m_abd_sim_data);
}

void GIPC::updateBoundary(device_TetraData& TetMesh, double alpha)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundary<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(device_TetraData& TetMesh, double alpha, int fid)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _updateBoundaryMoveDir<<<blockNum, threadNum>>>(
        TetMesh.vertexes, TetMesh.BoundaryType, _moveDir, IPC_dt, FEM::PI, alpha, numbers, fid);
}


void GIPC::computeXTilta(device_TetraData& TetMesh, const double& rate)
{
    int numbers = vertexNum;
    if(numbers <= 0)
        return;
    const unsigned int threadNum = default_threads;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(TetMesh.BoundaryType,
                                            TetMesh.velocities,
                                            TetMesh.o_vertexes,
                                            TetMesh.xTilta,
                                            TetMesh.apply_gravity,
                                            IPC_dt,
                                            rate,
                                            gravity,
                                            numbers);

    m_abd_system->cal_q_tilde(*m_abd_sim_data);
}

extern int total_Frames;   // file-scope frame counter (defined below); drives the stitch/soft target
                           // (update_soft_constraint_target_position(total_Frames+1)) → MUST be in the
                           // checkpoint or the restart's stitch target is for the wrong frame.

// [decouple debug] full-state checkpoint. Persistent cross-frame state only (friction/contact is
// ephemeral, rebuilt each step from positions): FEM vertexes/o_vertexes/velocities/xTilta +
// ABD q/q_prev/q_v + Kappa + total_Frames. Binary: [magic u32][vN u32][nb u32][4*vN double3 FEM]
// [3*nb Vector12 ABD][Kappa double][total_Frames i32]. Load restores them → next step() bit-identical.
void GIPC::save_checkpoint(device_TetraData& tm, const char* path)
{
    const int vN = (int)vertexNum;
    const int nb = (int)abd_fem_count_info.abd_body_num;
    std::vector<double3> hv(vN), ho(vN), hvel(vN), hxt(vN);
    CUDA_SAFE_CALL(cudaMemcpy(hv.data(),   tm.vertexes,   vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(ho.data(),   tm.o_vertexes, vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(hvel.data(), tm.velocities, vN*sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(hxt.data(),  tm.xTilta,     vN*sizeof(double3), cudaMemcpyDeviceToHost));
    std::vector<double> hq(12*nb), hqp(12*nb), hqv(12*nb);
    if(nb > 0 && m_abd_sim_data)
    {
        auto& d = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(hq.data(),  reinterpret_cast<const double*>(d.body_id_to_q.data()),      12*nb*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hqp.data(), reinterpret_cast<const double*>(d.body_id_to_q_prev.data()), 12*nb*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(hqv.data(), reinterpret_cast<const double*>(d.body_id_to_q_v.data()),    12*nb*sizeof(double), cudaMemcpyDeviceToHost));
    }
    FILE* f = fopen(path, "wb");
    if(!f) { printf("[ckpt] cannot open %s for write\n", path); return; }
    uint32_t magic = 0x53544B50u, uvN = (uint32_t)vN, unb = (uint32_t)nb;
    fwrite(&magic,1,4,f); fwrite(&uvN,1,4,f); fwrite(&unb,1,4,f);
    fwrite(hv.data(),sizeof(double3),vN,f); fwrite(ho.data(),sizeof(double3),vN,f);
    fwrite(hvel.data(),sizeof(double3),vN,f); fwrite(hxt.data(),sizeof(double3),vN,f);
    fwrite(hq.data(),sizeof(double),12*nb,f); fwrite(hqp.data(),sizeof(double),12*nb,f);
    fwrite(hqv.data(),sizeof(double),12*nb,f);
    fwrite(&Kappa,sizeof(double),1,f);
    fwrite(&total_Frames,sizeof(int),1,f);
    fclose(f);
    printf("[ckpt] saved %s (vN=%d nb=%d Kappa=%.6e total_Frames=%d)\n", path, vN, nb, Kappa, total_Frames);
}

void GIPC::load_checkpoint(device_TetraData& tm, const char* path)
{
    FILE* f = fopen(path, "rb");
    if(!f) { printf("[ckpt] cannot open %s for read\n", path); return; }
    uint32_t magic=0, uvN=0, unb=0;
    size_t rd = fread(&magic,1,4,f); rd += fread(&uvN,1,4,f); rd += fread(&unb,1,4,f);
    if(magic != 0x53544B50u || (int)uvN != (int)vertexNum || (int)unb != (int)abd_fem_count_info.abd_body_num)
    { printf("[ckpt] MISMATCH magic=%x vN=%u(exp %u) nb=%u(exp %u)\n", magic, uvN, (uint32_t)vertexNum, unb, (uint32_t)abd_fem_count_info.abd_body_num); fclose(f); return; }
    const int vN = (int)uvN, nb = (int)unb;
    std::vector<double3> hv(vN), ho(vN), hvel(vN), hxt(vN);
    std::vector<double> hq(12*nb), hqp(12*nb), hqv(12*nb); double kap=0;
    rd += fread(hv.data(),sizeof(double3),vN,f); rd += fread(ho.data(),sizeof(double3),vN,f);
    rd += fread(hvel.data(),sizeof(double3),vN,f); rd += fread(hxt.data(),sizeof(double3),vN,f);
    rd += fread(hq.data(),sizeof(double),12*nb,f); rd += fread(hqp.data(),sizeof(double),12*nb,f);
    rd += fread(hqv.data(),sizeof(double),12*nb,f);
    rd += fread(&kap,sizeof(double),1,f);
    int tf = 0; rd += fread(&tf,sizeof(int),1,f); fclose(f); (void)rd;
    CUDA_SAFE_CALL(cudaMemcpy(tm.vertexes,   hv.data(),   vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.o_vertexes, ho.data(),   vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.velocities, hvel.data(), vN*sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(tm.xTilta,     hxt.data(),  vN*sizeof(double3), cudaMemcpyHostToDevice));
    if(nb > 0 && m_abd_sim_data)
    {
        auto& d = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q.data()),      hq.data(),  12*nb*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q_prev.data()), hqp.data(), 12*nb*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<double*>(d.body_id_to_q_v.data()),    hqv.data(), 12*nb*sizeof(double), cudaMemcpyHostToDevice));
    }
    Kappa = kap;
    total_Frames = tf;
    printf("[ckpt] loaded %s (vN=%d nb=%d Kappa=%.6e total_Frames=%d)\n", path, vN, nb, Kappa, total_Frames);
}


