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



// [phase-time] lineSearch inner split (per frame): energy evals vs buildBVH+intersect vs buildCP vs step.
static double g_ls_e_ms = 0.0, g_ls_bvh_ms = 0.0, g_ls_cp_ms = 0.0, g_ls_step_ms = 0.0;
// gated event-pair stopwatch (STIFF_PHASE_TIME only; events are stream-ordered, no extra syncs —
// callers place it around ops that already end host-synchronous).
struct _LsTimer {
    bool on; cudaEvent_t a, b; double* acc;
    _LsTimer(double* accum) : on(getenv("STIFF_PHASE_TIME") != nullptr), acc(accum)
    { if(on){ cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a); } }
    void stop()
    { if(on){ cudaEventRecord(b); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b);
              *acc += m; cudaEventDestroy(a); cudaEventDestroy(b); on=false; } }
};

bool GIPC::lineSearch(device_TetraData& TetMesh, double& alpha, const double& cfl_alpha)
{
    bool   stopped       = false;
    const char* device_ls_env = getenv("STIFF_DEVICE_LINESEARCH");
    const bool device_ls = !device_ls_env || !device_ls_env[0]
                        || device_ls_env[0] != '0';
    // The preceding CCD/buildFullCP path already joins its auxiliary stream and
    // reads its counts on the host.  All energy work below is ordered on PTDS,
    // so the former full-device wait here was redundant.
    double lastEnergyVal = 0.0;
    if(device_ls)
        computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 0);
    else
        lastEnergyVal = computeEnergy(TetMesh);
    bool perenv_try = (m_env_alpha_valid && m_env_alpha && TetMesh.d_point_to_group
                       && TetMesh.h_groups_present
                       && abd_fem_count_info.fem_point_num > 0
                       && getenv("STIFF_PERENV_ALPHA"));

    // [multi-env S3] validate per-env energy decomposition (Sum_g E_g(FEM) ==
    // global FEM). Read-only; gated. Run a couple times then it's confirmed.
    if(getenv("STIFF_S3_VALIDATE") && TetMesh.d_point_to_group)
    {
        std::vector<double> eg;
        computeEnergy_perenv(TetMesh, eg);  // prints [S3-energy] when STIFF_PENV_STATS
    }

    double c1m         = 0.0;
    double armijoParam = 0;
    if(armijoParam > 0.0)
    {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, TetMesh);
    }

    CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                              TetMesh.vertexes,
                              vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    m_abd_system->copy_q_to_q_temp(*m_abd_sim_data);


    double alpha_SL = alpha;
    const int line_search_budget =
        line_search_max_iter > 0 ? line_search_max_iter : 64;

    // [multi-env S2/S3] RIGOROUS per-env line search. Step each env by its own
    // CCD-feasible alpha (m_env_alpha, S1-validated safe), then enforce PER-ENV
    // energy DESCENT: any env whose own energy E_g rose halves its alpha_g and
    // re-steps (independent per-env backtracking) — NOT the global-energy heuristic
    // (which can mask one env's increase). Per-env energy is validated exact
    // (Sum_g E_g == global, machine precision). Accept when every env satisfies
    // E_g(alpha_g) <= E_g(0) with no intersection. Falls back to the standard
    // uniform search only if an env can't descend within maxBT halvings. Gated.
    if(perenv_try)   // (perenv_try hoisted to the top — see the lazy lastEnergyVal comment)
    {
        const int NG = TetMesh.h_group_count;
        const int abdN = (int)abd_fem_count_info.abd_body_num;
        const int maxBT = 8;
        // [de-CPU S3] energies + decision fully DEVICE-resident: computeEnergy_perenv_dev fills
        // d_Eg0/d_Eg1 (no 22KB D2H), _s3_decide halves m_env_alpha IN PLACE (the true state; the
        // old host loop operated on the stale h_env_alpha mirror). Host reads ONE int per round —
        // required: it decides whether to re-run the step/rebuild/energy round (host loop control).
        static double* d_Eg0 = nullptr; static double* d_Eg1 = nullptr;
        static int* d_decision_counts = nullptr;
        if(!d_Eg0)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg0, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_Eg1, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_decision_counts, 2 * sizeof(int)));
        }
        _LsTimer _t0(&g_ls_e_ms);
        computeEnergy_perenv_dev(TetMesh, d_Eg0);   // per-env energy at START (temp) config
        _t0.stop();
        bool accepted = false;
        for(int bt = 0; bt <= maxBT; ++bt)
        {
            if(abdN > 0 && TetMesh.d_body_to_group)   // refresh per-body ABD alpha
            {
                if(!m_abd_body_alpha)
                    CUDA_SAFE_CALL(cudaMalloc((void**)&m_abd_body_alpha, abdN * sizeof(double)));
                int tn = 256, bn = (abdN + tn - 1) / tn;
                _gather_abd_body_alpha<<<bn, tn>>>(TetMesh.d_body_to_group, m_env_alpha,
                                                   m_abd_body_alpha, abdN, NG);
            }
            _LsTimer _ts(&g_ls_step_ms);
            m_perenv_apply = true;
            step_forward(TetMesh, alpha, false);   // per-env step from temp
            m_perenv_apply = false;
            _ts.stop();
            _LsTimer _tb(&g_ls_bvh_ms);
            buildBVH();
            int ground_trial_status = groundTrialStatus(TetMesh.d_point_to_group, NG);
            if(ground_trial_status != 0)
            {
                _tb.stop();
                if((ground_trial_status & 2) == 0)
                {
                    halveGroundInvalidEnvAlpha(NG);
                    continue;
                }
                break;
            }
            bool _isect = isIntersected(TetMesh);
            _tb.stop();
            if(_isect)   // CCD-safe alpha should prevent this; safety net
            {
                _s3_halve_all<<<(NG + 255) / 256, 256>>>(m_env_alpha, NG);
                continue;
            }
            _LsTimer _tc(&g_ls_cp_ms);
            buildCP();
            _tc.stop();
            // [backport] buildCP() already D2H-syncs h_cpNum/h_gpNum and the energy
            // kernels consume unsorted pairs directly, so the perf branch's
            // sync_cpNum()/sort_collision_pairs_by_type() are unneeded in this path.
            _LsTimer _t1(&g_ls_e_ms);
            computeEnergy_perenv_dev(TetMesh, d_Eg1);   // per-env energy AFTER step
            _t1.stop();
            CUDA_SAFE_CALL(cudaMemsetAsync(d_decision_counts, 0, 2 * sizeof(int)));
            _s3_decide<<<(NG + 255) / 256, 256>>>(
                d_Eg0, d_Eg1, m_env_alpha, d_decision_counts, NG,
                energy_abs_tol, energy_rel_tol);
            int decision_counts[2] = {0, 0};
            CUDA_SAFE_CALL(cudaMemcpy(decision_counts, d_decision_counts,
                                      2 * sizeof(int), cudaMemcpyDeviceToHost));
            int nfail = decision_counts[0];
            if(nfail == 0)
            {
                energy_tolerance_accept_count +=
                    static_cast<uint64_t>(decision_counts[1]);
                accepted = true;
                if(getenv("STIFF_PENV_STATS")) printf("[S3-accept] per-env descent ok (bt=%d)\n", bt);
                break;
            }
            // (m_env_alpha already halved in place on device — no H2D)
            if(getenv("STIFF_PENV_STATS")) printf("[S3-bt] bt=%d halved %d envs\n", bt, nfail);
        }
        if(accepted) return false;   // accepted; collision state already rebuilt
        if(getenv("STIFF_PENV_STATS")) printf("[S3-fallback] per-env descent not reached -> global\n");
        // fall through to standard uniform-alpha line search (re-steps from temp); the Armijo
        // reference lastEnergyVal is the eager entry computeEnergy (bit-exact original semantics).
    }

    step_forward(TetMesh, alpha, false);

    bool rehash = true;

    buildBVH();

    int ground_trial_backtracks = 0;
    int ground_trial_status = groundTrialStatus(nullptr, 0);
    while(ground_trial_status != 0 && ground_trial_backtracks < line_search_budget)
    {
        alpha *= 0.5;
        ++ground_trial_backtracks;
        step_forward(TetMesh, alpha, false);
        buildBVH();
        ground_trial_status = groundTrialStatus(nullptr, 0);
    }
    if(ground_trial_status != 0)
        throw std::runtime_error(
            "[StiffGIPC] ground trial step remained outside the strict barrier "
            "domain after line-search backtracking");

    int numOfIntersect = 0;
    int insectNum      = 0;

    bool checkInterset = true;

    while(checkInterset && isIntersected(TetMesh))
    {
        if(numOfIntersect >= line_search_budget)
        {
            // After `line_search_budget` halvings the trial state is numerically
            // the start state: the Newton step BEGAN intersecting (IPC feasibility
            // invariant already broken upstream) — no step size can fix that.
            // Fail fast like the ground trial above instead of spinning forever.
            throw std::runtime_error(
                "[StiffGIPC] mesh intersection persists after line-search "
                "backtracking (start state likely already intersecting; IPC "
                "feasibility invariant broken in an earlier step)");
        }
        printf("type 0 intersection happened 0:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = std::min(cfl_alpha, alpha);
        step_forward(TetMesh, alpha, false);
        buildBVH();
        //break;
    }

    buildCP();

    double testingE = 0.0;

    auto evaluate_trial_energy = [&](double trial_alpha) {
        if(device_ls)
        {
            computeEnergy_DeviceOut(TetMesh, m_line_search_energy + 1);
            _global_ls_decide<<<1, 1>>>(m_line_search_energy + 0,
                                        m_line_search_energy + 1,
                                        c1m,
                                        trial_alpha,
                                        energy_abs_tol,
                                        energy_rel_tol,
                                        m_line_search_decision);
            int decision = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&decision,
                                      m_line_search_decision,
                                      sizeof(int),
                                      cudaMemcpyDeviceToHost));
            if(getenv("STIFF_DEVICE_LINESEARCH_VALIDATE"))
            {
                double h_energy[2] = {0.0, 0.0};
                CUDA_SAFE_CALL(cudaMemcpy(h_energy,
                                          m_line_search_energy,
                                          sizeof(h_energy),
                                          cudaMemcpyDeviceToHost));
                const double rhs = h_energy[0] + c1m * trial_alpha;
                const double tol = energy_abs_tol
                                 + energy_rel_tol * fabs(h_energy[0]);
                const int host_decision =
                    !std::isfinite(h_energy[1])   // mirror the kernel's NaN guard
                        ? 1
                        : (h_energy[1] > rhs + tol ? 1
                                                   : (h_energy[1] > rhs ? 2 : 0));
                if(host_decision != decision)
                    throw std::runtime_error(
                        "[line-search] device and host decisions differ");
                static bool first_match = true;
                if(first_match)
                {
                    printf("[line-search-device-validate] first decision matched (%d)\n",
                           decision);
                    first_match = false;
                }
            }
            return decision;
        }

        testingE = computeEnergy(TetMesh);
        const double rhs = lastEnergyVal + c1m * trial_alpha;
        const double tol = energy_abs_tol + energy_rel_tol * fabs(lastEnergyVal);
        // [NaN-quarantine gap fix] see _global_ls_decide: NaN must read as
        // "not a descent" (backtrack), never as silent acceptance.
        if(!std::isfinite(testingE))
            return 1;
        return testingE > rhs + tol ? 1 : (testingE > rhs ? 2 : 0);
    };
    int energy_decision = evaluate_trial_energy(alpha);

    int    numOfLineSearch = 0;
    double LFStepSize      = alpha;

    std::cout.precision(18);
    // A larger configurable budget prevents the former hard-coded eight-step
    // limit from silently accepting a non-descent step in difficult contact.
    // Exhaustion remains loud because the current engine policy accepts the
    // final candidate so callers can decide whether to abort the simulation.
    while(energy_decision == 1 && numOfLineSearch < line_search_budget)
    {
        //std::cout << "[" << numOfLineSearch << "]   testE:    " << testingE
        //          << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        step_forward(TetMesh, alpha, false);
        buildBVH();
        buildCP();
        energy_decision = evaluate_trial_energy(alpha);
    }
    const bool line_search_exhausted = energy_decision == 1;
    if(energy_decision == 2)
        ++energy_tolerance_accept_count;
    if(line_search_exhausted)
    {
        if(device_ls)
        {
            double h_energy[2] = {0.0, 0.0};
            CUDA_SAFE_CALL(cudaMemcpy(h_energy,
                                      m_line_search_energy,
                                      2 * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            lastEnergyVal = h_energy[0];
            testingE      = h_energy[1];
        }
        // [T1] Monotonicity NOT achieved within budget: the accepted step raises
        // the incremental potential. Loud and unconditional — a silent
        // non-descent step corrupts contact state downstream.
        fprintf(stderr,
                "[line-search][WARN] budget exhausted (%d halvings, alpha=%.3e): "
                "energy did NOT decrease (E=%.9e > E0=%.9e). Step accepted anyway "
                "-- POTENTIAL SOLVER ERROR: expect contact drift / collapsed "
                "barrier distances / iteration blow-up in later frames. Raise "
                "Config.line_search_max_iter, reduce dt, or soften the drive.\n",
                numOfLineSearch, alpha, testingE, lastEnergyVal);
    }


    if(alpha < LFStepSize)
    {
        bool needRecomputeCS = false;
        while(checkInterset && isIntersected(TetMesh))
        {
            if(numOfIntersect >= line_search_budget)
            {
                throw std::runtime_error(
                    "[StiffGIPC] mesh intersection persists after energy "
                    "line-search backtracking (start state likely already "
                    "intersecting; IPC feasibility invariant broken in an "
                    "earlier step)");
            }
            printf("type 1 intersection happened 1:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = std::min(cfl_alpha, alpha);

            step_forward(TetMesh, alpha, false);
            buildBVH();
            needRecomputeCS = true;
        }
        if(needRecomputeCS)
        {
            buildCP();
        }
    }

    return stopped;
}


void GIPC::postLineSearch(device_TetraData& TetMesh, double alpha)
{
    if(Kappa == 0.0)
    {
        initKappa(TetMesh);
    }
    else
    {
        if(m_pergroup_kappa && m_kappa_group && m_d_close_grp)
        {
            // [multi-env per-group κ] run BOTH close-val checks (no short-circuit) to populate the
            // per-group flags, then double ONLY the groups that hit a close contact. This decouples
            // the cross-env bifurcation (a global κ doubling was the dominant cross-env coupling).
            int NG = m_active_group_count;
            CUDA_SAFE_CALL(cudaMemset(m_d_close_grp, 0, NG * sizeof(int)));
            (void)checkCloseGroundVal();   // populates m_d_close_grp (global bool ignored)
            (void)checkSelfCloseVal();
            // [perf/device-residence] double the close groups' κ ON DEVICE (in m_kappa_group), taking the
            // envelope via atomicMax — replaces the per-Newton D2H(close flags) + 256-env host loop +
            // H2D(κ). kappaMax is a host scalar (env-independent), the doubling+cap+max are all order-free
            // → strict bit-identical. h_kappa_group is NOT touched here (it is re-seeded by initKappa each
            // frame and read nowhere else during the frame; device m_kappa_group is the in-frame truth).
            double kappaMax = 1e300;
            upperBoundKappa(kappaMax);     // kappaMax = env-independent cap (was recomputed per group)
            static double* d_maxK = nullptr;
            if(!d_maxK) CUDA_SAFE_CALL(cudaMalloc(&d_maxK, sizeof(double)));
            CUDA_SAFE_CALL(cudaMemcpy(d_maxK, &Kappa, sizeof(double), cudaMemcpyHostToDevice));  // envelope init
            {
                int bs = 256;
                const double* frozen_alpha =
                    (getenv("STIFF_DECOUPLE_THRESH") && m_env_alpha_valid)
                        ? m_env_alpha
                        : nullptr;
                _per_group_kappa_double<<<(NG + bs - 1) / bs, bs>>>(
                    m_kappa_group, m_d_close_grp, frozen_alpha, NG, kappaMax, d_maxK);
            }
            CUDA_SAFE_CALL(cudaMemcpy(&Kappa, d_maxK, sizeof(double), cudaMemcpyDeviceToHost));  // scalar envelope
            tempFree_closeConstraint();
            tempMalloc_closeConstraint();
            CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
            CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));
            computeCloseGroundVal();
            computeSelfCloseVal();
            return;
        }

        bool updateKappa = checkCloseGroundVal();
        if(!updateKappa)
        {
            updateKappa = checkSelfCloseVal();
        }
        if(updateKappa)
        {
            Kappa *= 2.0;
            upperBoundKappa(Kappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
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
double maxCOllisionPairNum = 0;
double totalCollisionPairs = 0;
double total_Cg_count      = 0;
double timemakePd          = 0;
#include <vector>
#include <fstream>
std::vector<int> iterV;

// [phase-time] time3 sub-split (per frame): S1 per-env-alpha block vs lineSearch proper.
static double g_t3_s1_ms = 0.0, g_t3_ls_ms = 0.0;

int              GIPC::solve_subIP(device_TetraData& TetMesh,
                      double&           time0,
                      double&           time1,
                      double&           time2,
                      double&           time3,
                      double&           time4)
{
    auto& stats_at_current_frame = gipc::Statistics::instance().at_current_frame();
    if(g_gipc_log_level >= 1)
        std::cout << "solve_subIP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                  << std::endl;

    stats_at_current_frame["newton"] = gipc::Json::array();
    g_t3_s1_ms = 0.0; g_t3_ls_ms = 0.0;   // [phase-time] time3 sub-split, per frame
    g_ls_e_ms = 0.0; g_ls_bvh_ms = 0.0; g_ls_cp_ms = 0.0; g_ls_step_ms = 0.0;
    const int active_group_count = TetMesh.h_group_count;
    m_active_group_count = active_group_count;
    // [env-det] capture p2g at entry so the canon env-local-id tiebreak (g_vloc) is built BEFORE the
    // first buildCP of this frame (buildCP runs before computeGradientAndHessian sets m_d_p2g).
    if(getenv("STIFF_EE_CANON") && TetMesh.d_point_to_group) m_d_p2g = TetMesh.d_point_to_group;

    int iterCap = newton_iter_cap, k = 0;
    double semi_beta = 1.0;
    // [per-env productization] reset per-env telemetry for this solve
    m_env_frozen_iter.assign(kEnvAlphaSlots, -1);
    m_env_status.assign(kEnvAlphaSlots, 0);
    // [semi-implicit per-env] beta_g for the decoupled/host path: each env
    // accumulates its OWN line-search progress (candidate alpha_g from S1)
    // and exits by freezing ITSELF — the per-env analogue of Alg.1 that the
    // global beta cannot provide without re-coupling the batch.
    std::vector<double> semi_beta_env(kEnvAlphaSlots, 1.0);
    // [drive-substep] uipc-style animation substepping for joint driving: with
    // STIFF_DRIVE_SUBSTEP=S (>1), the per-frame driving target ramps linearly
    // over the first S Newton iterations (ratio=(k+1)/S) instead of dumping the
    // whole frame's driving energy into iteration 0 — that lump budget is what
    // the global monotone line search can "spend" on flipping a light resting
    // body (the kick). Convergence exits are suppressed until the ramp
    // completes (uipc: animation_reach_target).
    static const int s_drive_substep =
        getenv("STIFF_DRIVE_SUBSTEP") ? atoi(getenv("STIFF_DRIVE_SUBSTEP")) : 0;
    const bool drive_substep_on = s_drive_substep > 1 && m_drive_substep_mesh;
    double     drive_ratio      = drive_substep_on ? 0.0 : 1.0;

    CUDA_SAFE_CALL(cudaMemset(_moveDir, 0, vertexNum * sizeof(double3)));
    double totalTimeStep = 0;

    // [multi-env S4] inject per-env mask into the linear system: solve_linear_system
    // zeros masked envs' RHS, spmv skips their triplets. Reset all-active at frame
    // start (detection updates m_env_active each iter); cleared after the loop.
    const bool s4_mask_on = (m_env_active && TetMesh.d_dof_to_group
                             && TetMesh.d_point_to_group
                             && TetMesh.h_groups_present
                             && (getenv("STIFF_PERENV_MASK") || getenv("STIFF_PERENV_MASK_DEV")));
    // [S4-dev] device-derived mask (from m_env_alpha, zero D2H). Requires the per-env alpha
    // machinery (m_env_alpha filled by the S1 line-search block each iter).
    const bool s4_dev_mask = s4_mask_on && m_env_alpha && getenv("STIFF_PERENV_MASK_DEV")
                             && getenv("STIFF_PERENV_ALPHA");
    if(s4_mask_on)
    {
        std::fill_n(h_env_active.begin(), active_group_count, 1);
        CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                  active_group_count * sizeof(int), cudaMemcpyHostToDevice));
        m_global_linear_system->set_env_mask(
            m_env_active, TetMesh.d_dof_to_group, active_group_count);
    }
    // [multi-env P3] register the DOF→group map for the SEGMENTED block-diagonal PCG even when
    // masking is off (the PCG reads m_s4_dof_to_group/m_s4_ng). active=nullptr ⇒ no RHS masking.
    else if(getenv("STIFF_SEGMENTED_PCG") && TetMesh.d_dof_to_group
            && TetMesh.d_point_to_group && TetMesh.h_groups_present)
        m_global_linear_system->set_env_mask(
            nullptr, TetMesh.d_dof_to_group, active_group_count);
    else
        m_global_linear_system->set_env_mask(nullptr, nullptr, 0);

    int& s_dec_frame = g_dec_frame;   // [decouple probe] per-solve_subIP-call counter (~frame)
    s_dec_frame++;

    // [decouple] per-env convergence latch for the loop-exit override. The global gradVanish can
    // fire while a per-env env is still UNDER-converged (its mates converged fast → loop ends → that
    // env is cut off at a batch-dependent iter → drift). When DECOUPLE_THRESH, the loop exits only
    // once ALL present envs are per-env frozen (each reached ITS OWN convergence), so an env's final
    // state is independent of the mates / loop length. Updated at the end of S1 Phase B each iter.
    bool all_env_frozen = false;
    // Merged/global Newton convergence must be evaluated on the direction
    // solved at the CURRENT state. The legacy placement checked the previous
    // iteration's direction before the PCG solve, so a newly converged tiny
    // direction still entered line search and could spend 64 halvings fighting
    // energy-reduction roundoff. Per-env modes retain their existing freeze
    // pipeline for now and will be handled separately.
    const bool current_global_exit = (getenv("STIFF_DECOUPLE_THRESH") == nullptr);
    // Seven per-iteration events are diagnostic-only. Production must not
    // create/destroy them or force a device-wide synchronization.
    const bool phase_time = (getenv("STIFF_PHASE_TIME") != nullptr);

    for(; k < iterCap; ++k)
    {
        if(g_gipc_log_level >= 1 && k > 0 && k % 10 == 0)
            printf("  Newton iter %d ...\n", k);
        stats_at_current_frame["newton"].push_back(gipc::Json::object());

        // [drive-substep] ramp the joint driving targets across the solve.
        // theta_prev/d_prev derive from q_prev (frame-start state, constant
        // within the solve), so re-invoking per iteration is deterministic.
        if(drive_substep_on && drive_ratio < 1.0)
        {
            drive_ratio = std::min(1.0, double(k + 1) / s_drive_substep);
            update_joint_angle_targets_from_mesh(*m_drive_substep_mesh, drive_ratio);
        }

        // [S4-dev] periodic all-active recheck (bounce-back detection): every RECHECK iters, unmask
        // ALL envs so masked (frozen) envs get one REAL solve — if a κ doubling / friction update
        // moved a frozen env off its optimum, its hmx exceeds thr and _per_env_alpha_compute
        // un-freezes it (mask follows at end of this iter). Placed BEFORE the solve so the recheck
        // iter solves the full system. Fixed cadence → deterministic.
        if(s4_dev_mask && (k % 4 == 0))
            _mask_fill<<<(active_group_count + 255) / 256, 256>>>(
                m_env_active, 1, active_group_count);

        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum =
            (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        cudaEvent_t start = nullptr, end0 = nullptr, end1 = nullptr, end2 = nullptr;
        cudaEvent_t end3 = nullptr, end4 = nullptr, e2b = nullptr;
        if(phase_time)
        {
            CUDA_SAFE_CALL(cudaEventCreate(&start));
            CUDA_SAFE_CALL(cudaEventCreate(&end0));
            CUDA_SAFE_CALL(cudaEventCreate(&end1));
            CUDA_SAFE_CALL(cudaEventCreate(&end2));
            CUDA_SAFE_CALL(cudaEventCreate(&end3));
            CUDA_SAFE_CALL(cudaEventCreate(&end4));
            CUDA_SAFE_CALL(cudaEventCreate(&e2b));
        }
        auto destroy_iteration_events = [&]()
        {
            if(!phase_time) return;
            CUDA_SAFE_CALL(cudaEventDestroy(start));
            CUDA_SAFE_CALL(cudaEventDestroy(end0));
            CUDA_SAFE_CALL(cudaEventDestroy(end1));
            CUDA_SAFE_CALL(cudaEventDestroy(end2));
            CUDA_SAFE_CALL(cudaEventDestroy(end3));
            CUDA_SAFE_CALL(cudaEventDestroy(end4));
            CUDA_SAFE_CALL(cudaEventDestroy(e2b));
        };

        //printf("\n\n\ncollision num  %d\n\n\n", h_cpNum[0]+h_gpNum);

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(start));
        g_dec_k = (int)k;   // [decouple probe] expose k to computeGradientAndHessian's stage dumps
        timemakePd += computeGradientAndHessian(TetMesh);

        // [decouple probe] PRE-SOLVE gradient dump (shape_grads + fb hold the CLEAN gradient here,
        // before calculateMovingDirection clobbers shape_grads as scratch). frame STIFF_DUMP_FRAME,
        // any k if STIFF_PROBE_K unset → use k==0. Python compares env0 across batches.
        if(getenv("STIFF_GRAD_PRE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0))
        {
            std::vector<double3> hsh(vertexNum), hfb(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hfb.data(), TetMesh.fb, vertexNum*sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum*sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PRE");
            FILE* a=fopen((std::string(fn)+".shape").c_str(),"wb"); if(a){fwrite(hsh.data(),sizeof(double3),vertexNum,a);fclose(a);}
            FILE* b=fopen((std::string(fn)+".fb").c_str(),"wb");    if(b){fwrite(hfb.data(),sizeof(double3),vertexNum,b);fclose(b);}
            FILE* c=fopen((std::string(fn)+".grp").c_str(),"wb");   if(c){fwrite(hp.data(),sizeof(int),vertexNum,c);fclose(c);}
            printf("[grad-pre] dumped shape+fb+grp @frame %d k=%d\n", s_dec_frame, (int)k);
        }

        const char* merged_diag_frame = getenv("STIFF_MERGED_DIAG_FRAME");
        const bool merged_diag_sample = merged_diag_frame
                                     && s_dec_frame == atoi(merged_diag_frame)
                                     && ((int)k < 20 || ((int)k % 10) == 0);
        double distToOpt_PN = DBL_MAX;

        // The merged path keeps its historical BVH-scene scale. Merely declaring body groups must
        // not change merged-mode convergence; group-local scales belong to the decoupled path only.
        double thr_bbox2 = bboxDiagSize2;   // legacy: whole scene (== the env for ungrouped scenes)
        // The scalar fallback used by the decoupled path is the average complete-environment scale;
        // normal freeze sites use each environment's own bbox. A physical velocity tolerance, when
        // configured, overrides both relative scales.
        if(getenv("STIFF_DECOUPLE_THRESH")
           && TetMesh.h_groups_present && m_avg_env_bbox2 > 0.0)
            thr_bbox2 = m_avg_env_bbox2;

        // [uipc-style opt-in] newton_velocity_tol>0: physical exit (max step displacement
        // <= v_tol*dt), scene-size/env-count independent, relative_dhat fully inert.
        double _newton_thr = (newton_velocity_tol > 0.0)
                                 ? (newton_velocity_tol * IPC_dt)
                                 : sqrt(Newton_solver_threshold * Newton_solver_threshold
                                        * thr_bbox2 * IPC_dt * IPC_dt);
        auto device_newton_converged = [&](bool retain_movement) {
            calcMinMovement_DeviceOut(_moveDir, pcg_data.squeue, vertexNum);
            _newton_convergence_decide<<<1, 1>>>(pcg_data.squeue,
                                                 _newton_thr,
                                                 m_newton_convergence_decision);
            int converged = 0;
            CUDA_SAFE_CALL(cudaMemcpy(&converged,
                                      m_newton_convergence_decision,
                                      sizeof(int),
                                      cudaMemcpyDeviceToHost));
            if(retain_movement)
                CUDA_SAFE_CALL(cudaMemcpy(&distToOpt_PN,
                                          pcg_data.squeue,
                                          sizeof(double),
                                          cudaMemcpyDeviceToHost));
            return converged != 0;
        };
        bool gradVanish = current_global_exit
                              ? false
                              : device_newton_converged(merged_diag_sample);

        // [multi-env P3a step2] per-env Newton convergence tracking (precursor to
        // mask early-exit). The merged Newton loop currently breaks on the GLOBAL
        // move norm = the HARDEST env. Here we measure per-env move RMS each Newton
        // iter so we can see envs converge at DIFFERENT k (the early-exit premise,
        // on real merged-run data). Read-only diagnostic, gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && TetMesh.h_groups_present && getenv("STIFF_PENV_STATS"))
        {
            const int NG = active_group_count;
            static double* d_sq = nullptr; static int* d_cnt = nullptr;
            if(!d_sq) { cudaMalloc((void**)&d_sq, kEnvAlphaSlots*sizeof(double));
                        cudaMalloc((void**)&d_cnt, kEnvAlphaSlots*sizeof(int)); }
            cudaMemset(d_sq, 0, NG*sizeof(double)); cudaMemset(d_cnt, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                              d_sq, d_cnt, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hs(NG); std::vector<int> hc(NG);
            cudaMemcpy(hs.data(), d_sq, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.data(), d_cnt, NG*sizeof(int), cudaMemcpyDeviceToHost);
            double thr = sqrt(Newton_solver_threshold * Newton_solver_threshold
                              * bboxDiagSize2 * IPC_dt * IPC_dt);
            // [S4 probe] per-env MAX move (the real Newton-exit metric), not RMS.
            static double* d_mxm = nullptr;
            if(!d_mxm) cudaMalloc((void**)&d_mxm, kEnvAlphaSlots*sizeof(double));
            cudaMemset(d_mxm, 0, NG*sizeof(double));
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mxm, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG);
            cudaMemcpy(hmm.data(), d_mxm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            printf("[P3a-newton] k=%d thr=%.3e per-env maxMove:", k, thr);
            int n_conv = 0, n_present = 0;
            for(int g = 0; g < NG; ++g) if(hc[g] > 0) {
                ++n_present;
                bool conv = (k && hmm[g] < thr);   // matches gradVanish (max move)
                if(conv) ++n_conv;
                printf(" g%d=%.2e%s", g, hmm[g], conv ? "*" : "");
            }
            printf("  (%d/%d done-by-maxmove)\n", n_conv, n_present);
        }

        // [multi-env S4] per-env active-mask DETECTION (foundation; the assembly/
        // PCG/SpMV skips read m_env_active). Mask env once its Newton max-move <
        // thr*margin; every RECHECK iters unmask ALL present envs + re-check
        // (catches non-monotonic bounce-back). Self-contained; gated STIFF_PERENV_MASK.
        // Skip detection on early Newton iters: nothing converges before ~k=MINK
        // (measured), so the per-iter D2H+sync overhead there is pure waste.
        if(m_env_active && TetMesh.d_point_to_group && getenv("STIFF_PERENV_MASK")
           && !s4_dev_mask && k >= 4)   // [S4-dev] device-derived mask supersedes host detection
        {
            const int NG = active_group_count;
            const int RECHECK = 4;
            const double margin = 0.5;
            double thr = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));   // [decouple] batch-invariant; velocity_tol opt-in
            static double* d_mm = nullptr; static int* d_ct = nullptr;
            if(!d_mm) { cudaMalloc((void**)&d_mm, kEnvAlphaSlots*sizeof(double));
                        cudaMalloc((void**)&d_ct, kEnvAlphaSlots*sizeof(int)); }
            cudaMemset(d_mm, 0, NG*sizeof(double)); cudaMemset(d_ct, 0, NG*sizeof(int));
            int bs = 256, gs = (vertexNum + bs - 1) / bs;
            _per_env_max_move<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir, d_mm, vertexNum, NG);
            _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, nullptr, nullptr, d_ct, vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmm(NG); std::vector<int> hct(NG);
            cudaMemcpy(hmm.data(), d_mm, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hct.data(), d_ct, NG*sizeof(int), cudaMemcpyDeviceToHost);
            const bool recheck = (k % RECHECK == 0);
            int n_active = 0, n_present = 0;
            for(int g = 0; g < NG; ++g)
            {
                if(hct[g] <= 0) { h_env_active[g] = 0; continue; }  // absent env
                ++n_present;
                if(recheck || k == 0) h_env_active[g] = 1;          // periodic full re-check
                else if(h_env_active[g]
                        && hmm[g] < ((newton_velocity_tol > 0.0 || h_env_bbox2.empty() || h_env_bbox2[g] <= 0.0)
                                         ? thr
                                         : Newton_solver_threshold * IPC_dt * sqrt(h_env_bbox2[g]))
                                        * margin)   // [env-scale] own-bbox; deeply converged -> mask
                    h_env_active[g] = 0;
                if(h_env_active[g]) ++n_active;
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_active, h_env_active.data(),
                                      NG * sizeof(int), cudaMemcpyHostToDevice));
            if(getenv("STIFF_PENV_STATS"))
                printf("[S4-mask] k=%d active=%d/%d%s\n", k, n_active, n_present,
                       recheck ? " (recheck)" : "");
        }

        //double distToOpt_PN = calcMinMovement(TetMesh.totalForce, pcg_data.squeue, vertexNum);
        //printf("disToopt:  %f        %f\n",
        //       distToOpt_PN,
        //       2 * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //           * IPC_dt * IPC_dt);

        //bool gradVanish =
        //    (distToOpt_PN < 1
        //                        * sqrt(Newton_solver_threshold * Newton_solver_threshold * bboxDiagSize2)
        //                        * IPC_dt * IPC_dt);

        // [decouple] DECOUPLE_THRESH: exit only when ALL envs are per-env frozen (each converged),
        // NOT on the global gradVanish (which can cut off an under-converged env when its mates
        // finish first → batch-dependent final state). all_env_frozen is from the prev iter's S1
        // Phase B. Baseline (off) keeps the global gradVanish exit.
        // [robustness] the frozen-exit needs the S1 per-env-alpha machinery to actually run
        // (m_env_alpha_valid, set by the prev iter's S1). With DECOUPLE_THRESH but WITHOUT
        // STIFF_PERENV_ALPHA, all_env_frozen stays false forever → the loop ran to iterCap every
        // frame (pathological, found by the flag ablation). Fall back to gradVanish in that case.
        bool do_break = !current_global_exit
                     && ((getenv("STIFF_DECOUPLE_THRESH") && m_env_alpha_valid)
                             ? (k && all_env_frozen)
                             : (k && gradVanish));
        // [drive-substep] no convergence exit until the driving ramp completes
        // (uipc: animation_reach_target gates convergence_check).
        do_break = do_break && drive_ratio >= 1.0;
        if(do_break)
        {
            destroy_iteration_events();
            break;
        }
        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end0));

        auto cg_count = calculateMovingDirection(TetMesh, h_cpNum[0], pcg_data.P_type);
        //std::cout << "[" << k << "]"
        //          << "cg_count = " << cg_count << std::endl;
        total_Cg_count += cg_count;

        // [iron-law completion] make quarantined envs fully INERT before any
        // downstream consumer of this iteration's direction. The CCD alpha
        // kernels flag !isfinite(dot(n, dir)) into the shared invalid mask and
        // the device-chain validation would THROW — killing healthy envs.
        // (1) scan the fresh PCG direction per env: a naturally-diverging env
        //     is quarantined HERE, before the CCD chain trips on its NaNs
        //     (the NaN max-move quarantine in the freeze loop runs AFTER the
        //     CCD chain — too late for the throw);
        // (2) zero every quarantined env's direction: positions stay frozen
        //     (alpha==0 keeps temp verbatim), CCD candidates go neutral, the
        //     global convergence norm no longer sees the dead env.
        // Same gate as the NaN/timeout quarantine (host telemetry path);
        // the pure-device fast path is the documented contract: isolation
        // promises require per_env_exit / STIFF_PERENV_TELEM.
        {
            const bool quar_gate = m_d_p2g && perEnvIsolationLive();
            if(quar_gate)
            {
                const int NGq = m_active_group_count;
                if(!m_d_env_dirnan)
                    CUDA_SAFE_CALL(cudaMalloc((void**)&m_d_env_dirnan,
                                              kEnvAlphaSlots * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemsetAsync(m_d_env_dirnan, 0, NGq * sizeof(int), 0));
                {
                    int bs = 256, gs = ((int)vertexNum + bs - 1) / bs;
                    _scan_dir_nonfinite<<<gs, bs>>>(_moveDir, m_d_p2g,
                                                    m_d_env_dirnan, (int)vertexNum);
                }
                std::vector<int> hnan(NGq);
                CUDA_SAFE_CALL(cudaMemcpy(hnan.data(), m_d_env_dirnan,
                                          NGq * sizeof(int), cudaMemcpyDeviceToHost));
                for(int g = 0; g < NGq; ++g)
                    if(hnan[g] && (m_env_quarantined.empty() || !m_env_quarantined[g]))
                        quarantineEnv(g, -1, 0.0);
                bool any_quar = false;
                if(!m_env_quarantined.empty())
                    for(int g = 0; g < NGq; ++g)
                        any_quar |= (m_env_quarantined[g] != 0);
                if(any_quar && m_d_env_quarantined)
                {
                    int bs = 256, gs = ((int)vertexNum + bs - 1) / bs;
                    _zero_dir_quarantined<<<gs, bs>>>(_moveDir, m_d_p2g,
                                                      m_d_env_quarantined, (int)vertexNum);
                }
            }
        }
        if(current_global_exit)
        {
            gradVanish = device_newton_converged(merged_diag_sample);
            if(k && gradVanish && drive_ratio >= 1.0)
            {
                destroy_iteration_events();
                break;
            }
        }
        // [decouple probe] full-precision moveDir dump (engine order) + d_point_to_group (engine
        // order, aligned). At frame STIFF_DUMP_FRAME, Newton iter STIFF_PROBE_K. Python masks
        // verts[grp==g] and compares matesA vs matesB to find the fine per-step coupling seed.
        if(getenv("STIFF_GRAD_PROBE") && TetMesh.d_point_to_group
           && s_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))
           && (int)k == (getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 1))
        {
            std::vector<double3> hmd(vertexNum), hgr(vertexNum), hsh(vertexNum);
            std::vector<int>     hp(vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(hmd.data(), _moveDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hgr.data(), TetMesh.fb, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // contact+ground grad
            CUDA_SAFE_CALL(cudaMemcpy(hsh.data(), TetMesh.shape_grads, vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));  // kinetic+elastic
            CUDA_SAFE_CALL(cudaMemcpy(hp.data(), TetMesh.d_point_to_group, vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
            const char* fn = getenv("STIFF_GRAD_PROBE");
            FILE* f = fopen(fn, "wb"); if(f){ fwrite(hmd.data(), sizeof(double3), vertexNum, f); fclose(f); }
            std::string gradn = std::string(fn) + ".grad";
            FILE* fg = fopen(gradn.c_str(), "wb"); if(fg){ fwrite(hgr.data(), sizeof(double3), vertexNum, fg); fclose(fg); }
            std::string shn = std::string(fn) + ".shape";
            FILE* fs = fopen(shn.c_str(), "wb"); if(fs){ fwrite(hsh.data(), sizeof(double3), vertexNum, fs); fclose(fs); }
            std::string gn = std::string(fn) + ".grp";
            FILE* g = fopen(gn.c_str(), "wb"); if(g){ fwrite(hp.data(), sizeof(int), vertexNum, g); fclose(g); }
            printf("[grad-probe] dumped moveDir+grad+grp @frame %d k=%d (%d verts) -> %s\n",
                   s_dec_frame, (int)k, vertexNum, fn);
            // [decouple] ABD body pose dump (verify the gripper/arm pose drifts across batches → the
            // FEM-pin seed). m_d_abd_body_q = Vector12 (12 doubles) per body; d_body_to_group = env id.
            if(getenv("STIFF_ABD_DUMP") && m_abd_sim_data && TetMesh.d_body_to_group)
            {
                int nb = (int)abd_fem_count_info.abd_body_num;
                const double* qptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_q.data());
                const double* dqptr = reinterpret_cast<const double*>(m_abd_sim_data->device.body_id_to_dq.data());
                std::vector<double> hq(12 * nb), hdq(12 * nb); std::vector<int> hbg(nb);
                CUDA_SAFE_CALL(cudaMemcpy(hq.data(), qptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_SAFE_CALL(cudaMemcpy(hdq.data(), dqptr, 12 * nb * sizeof(double), cudaMemcpyDeviceToHost));
                FILE* dq=fopen((std::string(fn)+".abddq").c_str(),"wb"); if(dq){fwrite(hdq.data(),sizeof(double),12*nb,dq);fclose(dq);}
                CUDA_SAFE_CALL(cudaMemcpy(hbg.data(), TetMesh.d_body_to_group, nb * sizeof(int), cudaMemcpyDeviceToHost));
                FILE* q=fopen((std::string(fn)+".abdq").c_str(),"wb"); if(q){fwrite(hq.data(),sizeof(double),12*nb,q);fclose(q);}
                FILE* b=fopen((std::string(fn)+".abdg").c_str(),"wb"); if(b){fwrite(hbg.data(),sizeof(int),nb,b);fclose(b);}
                printf("[abd-dump] %d bodies @frame %d k=%d\n", nb, s_dec_frame, (int)k);
            }
        }
        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end1));
        double alpha = 1.0, slackness_a = 0.9, slackness_m = 0.8;
        double diag_ground_alpha = 1.0;
        double diag_narrow_alpha = 1.0;
        double diag_refined_alpha = 1.0;
        int    diag_narrow_pairs = h_cpNum[0];
        bool   diag_refine_used = false;

        // Keep the complete scalar CCD chain on device. The first two
        // reductions feed temp_alpha directly into swept BVH construction.
        const bool g_did = !m_skip_all_collision && surf_vertexNum >= 1;
        // Gate/count from the stable DCD snapshot, not the live CCD buffer.
        const bool s_did = !m_skip_all_collision && m_dcd_snap_count >= 1;
        CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_alpha_invalid, 0, sizeof(int)));
        CUDA_SAFE_CALL(cudaMemsetAsync(
            m_ccd_refined_invalid, 0, (1 + kEnvAlphaSlots) * sizeof(int)));
        if(g_did)
            ground_largestFeasibleStepSize_DeviceOut(
                slackness_a, pcg_data.squeue, m_ccd_alpha_slots + 0);
        if(s_did)
            self_largestFeasibleStepSize_DeviceOut(
                slackness_m,
                ensure_reduce_scratch(m_dcd_snap_count),
                m_dcd_snap_count,
                m_ccd_alpha_slots + 1);
        _ccd_initial_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                             g_did ? 1 : 0,
                                             s_did ? 1 : 0,
                                             m_ccd_alpha_invalid);
        //alpha = std::min(alpha, InjectiveStepSize(0.2, 1e-6, pcg_data.squeue, TetMesh.tetrahedras));
        double temp_alpha = 1.0;
        double alpha_CFL  = 1.0;

        double ccd_size = 1.0;
        //#ifdef USE_FRICTION
        //        ccd_size = 0.6;
        //#endif

        // [multi-env S1 Phase A] per-env temp_alpha terms (ground + narrow-self),
        // computed HERE because buildFullCP below overwrites _ccd_collisonPairs.
        // Mirrors the engine's temp_alpha reductions (lines above) but per-env.
        // Regions: m_env_scratch[0*NG]=ground alpha, [1*NG]=narrow-self alpha.
        m_env_alpha_valid = false;  // reset each Newton iter; S1 sets true below
        const bool s1_on = (m_env_scratch && TetMesh.d_point_to_group
                            // [N=1 guard] all -1 p2g = zero env coverage: S1 would flag itself
                            // valid, all_env_frozen unreachable -> Newton pegs at iterCap.
                            && TetMesh.h_groups_present
                            && surf_vertexNum >= 1 && !m_skip_all_collision
                            && getenv("STIFF_PERENV_ALPHA"));
        if(s1_on)
        {
            const int NG = active_group_count, bs = 256;
            _fill_double<<<(2 * NG + bs - 1) / bs, bs>>>(
                m_env_scratch, 1.0, 2 * NG);
            _per_env_groundAlpha_min<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                _vertexes, _surfVerts, _groundOffset, _groundNormal, _moveDir,
                TetMesh.d_point_to_group, m_env_scratch + 0*NG, slackness_a,
                surf_vertexNum, _point_body_id, _ground_skip_body, _ground_body_count, NG,
                m_ccd_alpha_invalid);
            // [narrow-self snapshot] sweep the FULL DCD-time snapshot (stable
            // content, env-balanced by construction) — never the live CCD buffer,
            // whose prefix is a race-ordered slice of last iteration's swept
            // emission (the strict cross-env asymmetry root cause).
            if(m_dcd_snap_count >= 1)
                _per_env_selfAlpha_min<<<(m_dcd_snap_count+bs-1)/bs, bs>>>(
                    _vertexes, _dcd_ccd_snapshot, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 1*NG, slackness_m, m_dcd_snap_count, NG,
                    getenv("STIFF_CCD_CANON") ? m_d_vloc : nullptr,
                    m_ccd_alpha_invalid,
                    kCcdInvalidPerEnvNarrow,
                    nullptr);
        }

        buildBVH_FULLCCD(1.0, m_ccd_alpha_slots + 2);
        buildFullCP(1.0, m_ccd_alpha_slots + 2);
        if(h_ccd_cpNum > 0)
        {
            cfl_largestSpeed_DeviceOut(pcg_data.squeue, m_ccd_alpha_slots + 3);
            // Launch the refined reduction unconditionally. Its raw invalid
            // status becomes effective only if the exact refinement gate fires.
            self_full_largestFeasibleStepSize_DeviceOut(
                slackness_m,
                ensure_reduce_scratch(h_ccd_cpNum),
                h_ccd_cpNum,
                m_ccd_alpha_slots + 4);
        }
        _ccd_final_alpha_combine<<<1, 1>>>(m_ccd_alpha_slots,
                                           h_ccd_cpNum > 0 ? 1 : 0,
                                           dHat,
                                           ccd_size,
                                           m_ccd_alpha_invalid,
                                           m_ccd_refined_invalid);

        // One scalar-chain D2H after every global decision and validation bit.
        double h_ccd_state[8] = {1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};
        CUDA_SAFE_CALL(cudaMemcpy(h_ccd_state,
                                  m_ccd_alpha_slots,
                                  sizeof(h_ccd_state),
                                  cudaMemcpyDeviceToHost));
        validateFinalCcdStateOrThrow(h_ccd_state, "device CCD chain");
        if(getenv("STIFF_CCD_VALIDATE"))
        {
            const double host_temp = h_ccd_state[0] < h_ccd_state[1]
                                         ? h_ccd_state[0]
                                         : h_ccd_state[1];
            double host_cfl   = host_temp;
            double host_alpha = host_temp;
            if(h_ccd_cpNum > 0)
            {
                host_cfl = sqrt(dHat) / h_ccd_state[3] * 0.5;
                host_alpha = host_temp < host_cfl ? host_temp : host_cfl;
                if(host_temp > 2.0 * host_cfl)
                {
                    const double refined = h_ccd_state[4] * ccd_size;
                    host_alpha = host_temp < refined ? host_temp : refined;
                    host_alpha = host_alpha > host_cfl ? host_alpha : host_cfl;
                }
            }
            const bool temp_exact = std::memcmp(
                &host_temp, &h_ccd_state[2], sizeof(double)) == 0;
            const bool cfl_exact = std::memcmp(
                &host_cfl, &h_ccd_state[6], sizeof(double)) == 0;
            const bool alpha_exact = std::memcmp(
                &host_alpha, &h_ccd_state[5], sizeof(double)) == 0;
            if(!temp_exact || !cfl_exact || !alpha_exact)
                throw std::runtime_error(
                    "[CCD] device alpha chain differs from host formula");
            static bool first_ccd_match = true;
            if(first_ccd_match)
            {
                printf("[ccd-device-validate] temp/CFL/final alpha exact\n");
                first_ccd_match = false;
            }
        }
        diag_ground_alpha  = h_ccd_state[0];
        diag_narrow_alpha  = h_ccd_state[1];
        temp_alpha         = h_ccd_state[2];
        diag_refined_alpha = h_ccd_state[4];
        alpha              = h_ccd_state[5];
        alpha_CFL          = h_ccd_state[6];
        diag_refine_used   = h_ccd_cpNum > 0 && temp_alpha > 2.0 * alpha_CFL;

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end2));
        //printf("alpha:  %f\n", alpha);

        // [multi-env P3a] read-only: per-env alpha_CFL spread. The global alpha
        // above is ONE scalar (= min over ALL envs of CCD/CFL feasible step). If
        // per-env maxSpeed differs a lot, the global alpha is dragged by the
        // fastest env -> slower envs forced to over-small steps -> per-env
        // line-search would decouple them. Decides whether per-env alpha is worth
        // building. Gated STIFF_PENV_STATS.
        if(TetMesh.d_point_to_group && TetMesh.h_groups_present
           && surf_vertexNum >= 1 && getenv("STIFF_PENV_STATS"))
        {
            const int NG = active_group_count;
            static double* d_mx = nullptr;
            if(!d_mx) cudaMalloc((void**)&d_mx, kEnvAlphaSlots * sizeof(double));
            cudaMemset(d_mx, 0, NG * sizeof(double));
            int bs = 256, gs = (surf_vertexNum + bs - 1) / bs;
            _per_env_max_cfl<<<gs, bs>>>(TetMesh.d_point_to_group, _moveDir,
                                         _surfVerts, d_mx, surf_vertexNum, NG);
            cudaDeviceSynchronize();
            std::vector<double> hmx(NG);
            cudaMemcpy(hmx.data(), d_mx, NG * sizeof(double), cudaMemcpyDeviceToHost);
            double sq = sqrt(dHat);
            printf("[P3a-cfl] k=%d global_alpha=%.3e per-env alpha_CFL=", k, alpha);
            for(int g = 0; g < NG; ++g) if(hmx[g] > 0.0)
                printf(" g%d=%.3e", g, sq / hmx[g] * 0.5);
            printf("\n");
        }

        // [multi-env S1 Phase B] per-env feasible-alpha substrate (PHYSICS-NEUTRAL).
        // Phase A filled ground+narrow-self; here add refined-self (over the NEW
        // _ccd_collisonPairs) + CFL, then combine per the engine's exact logic:
        //   temp_alpha_env = min(1, ground_env, narrowSelf_env)
        //   if ccd pairs: alpha_env = min(temp_alpha_env, alpha_CFL_env);
        //                 if temp_alpha_env > 2*alpha_CFL_env:
        //                     alpha_env = max(min(temp_alpha_env, refinedSelf_env*ccd_size), alpha_CFL_env)
        // Each CCD term is a direct per-env MIN(alpha), matching the global path.
        // This per-env substrate does not alter the global scalar `alpha`;
        // it only fills the values S2 consumes. Invariant
        // (validated): min_g(m_env_alpha) == global feasibility alpha; N=1 -> one
        // group -> m_env_alpha[g0] == global alpha.
        if(s1_on)
        {
            const int NG = active_group_count, bs = 256;
            _fill_double<<<(NG + bs - 1) / bs, bs>>>(
                m_env_scratch + 2*NG, 1.0, NG);  // refined-alpha region
            cudaMemset(m_env_scratch + 3*NG, 0, 2 * NG * sizeof(double));  // cfl + Newton max regions
            if(h_ccd_cpNum > 0)  // refined-self over NEW _ccd_collisonPairs[0..h_ccd_cpNum)
                _per_env_selfAlpha_min<<<(h_ccd_cpNum+bs-1)/bs, bs>>>(
                    _vertexes, _ccd_collisonPairs, _moveDir, TetMesh.d_point_to_group,
                    m_env_scratch + 2*NG, slackness_m, h_ccd_cpNum, NG,
                    getenv("STIFF_CCD_CANON") ? m_d_vloc : nullptr,
                    m_ccd_alpha_invalid,
                    kCcdInvalidPerEnvRefined,
                    m_ccd_refined_invalid + 1);
            _per_env_max_cfl<<<(surf_vertexNum+bs-1)/bs, bs>>>(
                TetMesh.d_point_to_group, _moveDir, _surfVerts, m_env_scratch + 3*NG,
                surf_vertexNum, NG);
            _per_env_max_move<<<(vertexNum+bs-1)/bs, bs>>>(
                TetMesh.d_point_to_group, _moveDir, m_env_scratch + 4*NG,
                vertexNum, NG);
            // [perf] DEVICE-SIDE per-env alpha + freeze (no cudaDeviceSynchronize, no 5x256 D2H, no
            // host loop, no H2D) — writes m_env_alpha directly + one 3-int D2H
            // (two freeze counters and the already-needed CCD status word).
            // Bit-identical to the host loop (same per-env formulas). Host path kept only under a
            // diagnostic flag.
            const double _sq_    = sqrt(dHat);
            const double _thrcv_ = getenv("STIFF_DECOUPLE_THRESH") ? ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt)) : 0.0;
            const bool _s1diag_ = getenv("STIFF_PENV_STATS") || getenv("STIFF_A0_DUMP")
                               || getenv("STIFF_S1_DEBUG") || getenv("STIFF_ALPHA_DBG")
                               // [per-env productization] telemetry (freeze iters/
                               // status), the per-env iter budget and the NaN
                               // quarantine live in the host loop — route there
                               // when any of them is requested.
                               || env_newton_iter_cap > 0
                               || (getenv("STIFF_PERENV_TELEM")
                                   && getenv("STIFF_PERENV_TELEM")[0] != '0');
            if(!_s1diag_)
            {
                static int* d_env_cnt = nullptr;
                if(!d_env_cnt)
                    CUDA_SAFE_CALL(cudaMalloc((void**)&d_env_cnt, 3 * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemsetAsync(d_env_cnt, 0, 3 * sizeof(int)));
                _per_env_alpha_compute<<<(NG + bs - 1) / bs, bs>>>(
                    m_env_alpha, m_env_scratch, NG, _sq_, 1.0, (h_ccd_cpNum > 0) ? 1 : 0,
                    temp_alpha, alpha_CFL, getenv("STIFF_DECOUPLE_THRESH") ? 1 : 0,
                    getenv("STIFF_NO_REFINE") ? 1 : 0, _thrcv_,
                    d_env_bbox2, Newton_solver_threshold * IPC_dt, newton_velocity_tol * IPC_dt,
                    m_ccd_refined_invalid + 1,
                    m_ccd_alpha_invalid,
                    d_env_cnt);
                int _hc_[3];
                CUDA_SAFE_CALL(cudaMemcpy(
                    _hc_, d_env_cnt, 3 * sizeof(int), cudaMemcpyDeviceToHost));
                throwForInvalidCcdMask(_hc_[2], "per-env CCD reduction");
                m_env_alpha_valid = true;
                all_env_frozen    = (_hc_[0] > 0 && _hc_[1] == _hc_[0]);
            }
            else
            {
            _promote_per_env_refined_invalid<<<(NG + bs - 1) / bs, bs>>>(
                m_env_scratch,
                NG,
                (h_ccd_cpNum > 0) ? 1 : 0,
                _sq_,
                temp_alpha,
                alpha_CFL,
                getenv("STIFF_DECOUPLE_THRESH") ? 1 : 0,
                getenv("STIFF_NO_REFINE") ? 1 : 0,
                m_ccd_refined_invalid + 1,
                m_ccd_alpha_invalid);
            cudaDeviceSynchronize();
            throwIfInvalidCcdAlpha("per-env CCD reduction");
            std::vector<double> hg(NG), hs(NG), hr(NG), hmx(NG), hnm(NG);
            // [semi-implicit beta timing fix] update beta_g from the PREVIOUS
            // iteration's ACCEPTED per-env alpha (m_env_alpha still holds it —
            // this S1 pass overwrites it further below), not from this
            // iteration's pre-line-search candidate. Candidate-based decay
            // overestimated progress whenever backtracking halved the step
            // (audit leftover #5). Threshold crossing only arms a freeze flag
            // consumed in the loop below (freeze-next semantics).
            static std::vector<char> semi_freeze_next;
            if(semi_implicit_enabled && (int)k >= semi_implicit_min_iter + 1)
            {
                std::vector<double> h_acc(NG);
                cudaMemcpy(h_acc.data(), m_env_alpha, NG*sizeof(double), cudaMemcpyDeviceToHost);
                if((int)semi_freeze_next.size() < NG) semi_freeze_next.assign(NG, 0);
                for(int g = 0; g < NG; ++g)
                {
                    if(m_env_status[g] != 0) continue;         // already frozen/diverged
                    semi_beta_env[g] *= std::max(0.0, 1.0 - h_acc[g]);
                    if(semi_beta_env[g] <= semi_implicit_beta_tol)
                        semi_freeze_next[g] = 1;
                }
            }
            else if((int)semi_freeze_next.size() < NG)
                semi_freeze_next.assign(NG, 0);
            if((int)k == 0) std::fill(semi_freeze_next.begin(), semi_freeze_next.end(), 0);
            cudaMemcpy(hg.data(),  m_env_scratch + 0*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(),  m_env_scratch + 1*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hr.data(),  m_env_scratch + 2*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hmx.data(), m_env_scratch + 3*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hnm.data(), m_env_scratch + 4*NG, NG*sizeof(double), cudaMemcpyDeviceToHost);
            const double sq       = sqrt(dHat);
            const double ccd_size = 1.0;
            const bool   have_ccd = (h_ccd_cpNum > 0);
            double       min_env  = 1e30, max_env = 0.0;
            int          n_env    = 0, n_frozen = 0;
            for(int g = 0; g < NG; ++g)
            {
                bool present = hnm[g] > 0.0
                            || (!h_env_bbox2.empty() && h_env_bbox2[g] > 0.0);
                if(!present) continue;
                double ta = std::min(hg[g], hs[g]);  // direct temp_alpha_env
                double a = ta;
                if(have_ccd && hmx[g] > 0.0)
                {
                    double acfl = sq / hmx[g] * 0.5;
                    a = std::min(ta, acfl);
                    // The refinement GATE: the engine gates on the GLOBAL temp_alpha/alpha_CFL,
                    // which makes env_0's branch decision (enter refined CCD or not) depend on the
                    // MATES (global temp_alpha = min over all envs) → BREAKS batch-invariance: at a
                    // frame where global temp_alpha diverges across batches, env_0 enters refinement
                    // in one batch but not the other → env_0's feasible alpha differs → drift →
                    // chaos amplifies. STIFF_DECOUPLE_THRESH gates per-env (ta_g vs acfl_g) so env_0's
                    // branch depends only on env_0 (batch-invariant). Off → exact engine behavior.
                    // [decouple TEST] STIFF_NO_REFINE: skip refinement entirely (env0's a=min(ta,acfl)
                    // → fully per-env/batch-invariant; hr — built from global-temp_alpha CCD pairs — is
                    // the confirmed last leak). isIntersected safety net in lineSearch catches any
                    // resulting penetration. Used to verify hr is the only remaining batch-coupling.
                    double gate_lhs = getenv("STIFF_DECOUPLE_THRESH") ? ta   : temp_alpha;
                    double gate_rhs = getenv("STIFF_DECOUPLE_THRESH") ? acfl : alpha_CFL;
                    if(!getenv("STIFF_NO_REFINE") && gate_lhs > 2.0 * gate_rhs)
                    {
                        a = std::min(ta, hr[g] * ccd_size);
                        a = std::max(a, acfl);
                    }
                }
                h_env_alpha[g] = a;
                // [decouple] FREEZE env g once IT has converged (per-env max-move < the per-env Newton
                // threshold), so env g stops stepping at ITS OWN convergence iter — NOT the global loop
                // count. The merged Newton loop runs a BATCH-DEPENDENT number of iters (harder mates →
                // more iters: measured A=13 vs B=16 at frame0); during the extra iters an already-
                // converged env's ABD (dq small but nonzero) keeps stepping → its gripper/arm pose
                // drifts batch-dependently → FEM-pin seed. Freezing at the env's own convergence makes
                // env g's total steps batch-invariant. Block-diagonal per-env solve ⇒ monotonic ⇒ no
                // re-activation needed. Gated STIFF_DECOUPLE_THRESH.
                if(getenv("STIFF_DECOUPLE_THRESH"))
                {
                    double thr_cv = (newton_velocity_tol > 0.0)
                        ? (newton_velocity_tol * IPC_dt)
                        : ((!h_env_bbox2.empty() && h_env_bbox2[g] > 0.0)
                               ? Newton_solver_threshold * IPC_dt * sqrt(h_env_bbox2[g])   // [env-scale] own bbox
                               : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    if(hnm[g] < thr_cv) h_env_alpha[g] = 0.0;
                }
                // [per-env productization] quarantine + budget, evaluated per iter
                // (no latch needed: NaN persists, k only grows).
                if(!m_env_quarantined.empty() && g < (int)m_env_quarantined.size()
                   && m_env_quarantined[g])
                {   // [iron-law] persistently quarantined env: pin it in EVERY
                    // iteration of EVERY later solve. Checked OUTSIDE the
                    // alpha!=0 gate — the direction-zero pass makes a
                    // quarantined env's max-move 0, so the convergence freeze
                    // above zeroes its alpha first and would otherwise mislabel
                    // it status 1 (converged); status 3 must win the telemetry.
                    h_env_alpha[g] = 0.0;
                    if(m_env_status[g] != 3)
                    {
                        m_env_status[g] = 3;
                        if(m_env_frozen_iter[g] < 0)
                            m_env_frozen_iter[g] = (int)k;
                    }
                }
                else if(h_env_alpha[g] != 0.0)
                {
                    if(std::isnan(hnm[g]) || std::isinf(hnm[g]))
                    {   // diverged env: freeze it so its NaN cannot poison the batch
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] < 2)
                        {
                            m_env_status[g]      = 3;
                            m_env_frozen_iter[g] = (int)k;
                            printf("  [per-env] env %d DIVERGED (NaN/inf max-move) at iter %d -> frozen\n",
                                   g, (int)k);
                        }
                    }
                    else if(env_newton_iter_cap > 0 && (int)k + 1 >= env_newton_iter_cap)
                    {   // per-env iteration budget: give up on THIS env only
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] < 2)
                        {
                            m_env_status[g]      = 2;
                            m_env_frozen_iter[g] = (int)k;
                            printf("  [per-env] env %d TIMEOUT (cap=%d) at iter %d -> frozen\n",
                                   g, env_newton_iter_cap, (int)k);
                        }
                    }
                    else if(semi_implicit_enabled && semi_freeze_next[g])
                    {   // [semi-implicit per-env, beta timing fix] the freeze flag
                        // was armed from the PREVIOUS iteration's accepted alpha
                        // (see the beta update above the loop); consume it here.
                        h_env_alpha[g] = 0.0;
                        if(m_env_status[g] == 0)
                        {
                            m_env_status[g]      = 1;   // converged (semi-implicit accept)
                            m_env_frozen_iter[g] = (int)k;
                        }
                    }
                }
                if(h_env_alpha[g] == 0.0) ++n_frozen;   // [decouple] per-env converged (frozen)
                if(h_env_alpha[g] == 0.0 && m_env_frozen_iter[g] < 0)
                {   // [per-env productization] first freeze = convergence iter
                    m_env_frozen_iter[g] = (int)k;
                    if(m_env_status[g] == 0) m_env_status[g] = 1;
                }
                min_env = std::min(min_env, a);
                max_env = std::max(max_env, a);
                ++n_env;
                if(getenv("STIFF_A0_DUMP") && g == 0
                   && (!getenv("STIFF_DUMP_FRAME") || g_dec_frame == atoi(getenv("STIFF_DUMP_FRAME"))))
                {   // [decouple] env0 per-iter: applied alpha (h_env_alpha[g], post-freeze) + hmx (max-move) + frozen?
                    double thr_cv = ((newton_velocity_tol > 0.0) ? (newton_velocity_tol * IPC_dt) : sqrt(Newton_solver_threshold * Newton_solver_threshold * thr_bbox2 * IPC_dt * IPC_dt));
                    printf("[a0] frame=%d k=%d a_applied=%.17e a_feasible=%.17e "
                           "newtonMax=%.17e cflMax=%.17e thr_cv=%.6e frozen=%d\n",
                           g_dec_frame, (int)k, h_env_alpha[g], a, hnm[g], hmx[g],
                           thr_cv, (int)(h_env_alpha[g] == 0.0));
                }
                if(getenv("STIFF_S1_DEBUG") && alpha > 0.99 && a < 0.9)
                    printf("  [S1-dbg] g%d a=%.4e ta=%.4e ground=%.4e narrow=%.4e "
                           "refined=%.4e acfl=%.4e ccdN=%d alphaCFLglob=%.4e (global=%.4e)\n",
                           g, a, ta, hg[g], hs[g], hr[g], have_ccd?sq/hmx[g]*0.5:9.99,
                           (int)h_ccd_cpNum, alpha_CFL, alpha);
            }
            // [env-det dbg] cross-env alpha mismatch (STIFF_ALPHA_DBG): pin the component that differs.
            if(getenv("STIFF_ALPHA_DBG") && NG >= 2 && hmx[0] > 0.0 && hmx[1] > 0.0
               && (h_env_alpha[0] != h_env_alpha[1] || hg[0] != hg[1] || hs[0] != hs[1]
                   || hr[0] != hr[1] || hmx[0] != hmx[1]))
            {
                static int _ad = 0;
                if(_ad++ < 12)
                    printf("[alpha-dbg] a0=%.17e a1=%.17e | hg %.17e/%.17e hs %.17e/%.17e "
                           "hr %.17e/%.17e hmx %.17e/%.17e ccdN=%d\n",
                           h_env_alpha[0], h_env_alpha[1], hg[0], hg[1], hs[0], hs[1],
                           hr[0], hr[1], hmx[0], hmx[1], (int)h_ccd_cpNum);
            }
            CUDA_SAFE_CALL(cudaMemcpy(m_env_alpha, h_env_alpha.data(),
                                      NG * sizeof(double), cudaMemcpyHostToDevice));
            m_env_alpha_valid = true;  // m_env_alpha fresh -> lineSearch may use it
            // [decouple] all present envs per-env frozen (converged)? → drives the loop-exit override
            // so the loop runs until env_0 (and every env) reaches ITS OWN convergence, batch-independent.
            all_env_frozen = (n_env > 0 && n_frozen == n_env);
            if(getenv("STIFF_PENV_STATS"))
                printf("[S1-envalpha] k=%d global_alpha=%.6e min_env=%.6e max_env=%.6e "
                       "n_env=%d rel=%.2e (neutral-check; headroom=max_env/global)\n",
                       k, alpha, min_env, max_env, n_env,
                       fabs(alpha - min_env) / std::max(alpha, 1e-30));
            }   // [perf] end diagnostic host path
        }

        // The current move direction has just been tested against every environment's Newton
        // threshold. Exit immediately, exactly like the merged path, rather than performing a
        // full zero-active assembly/solve on the next iteration.
        if(getenv("STIFF_DECOUPLE_THRESH") && m_env_alpha_valid
           && all_env_frozen && k && drive_ratio >= 1.0)
        {
            destroy_iteration_events();
            break;
        }

        // [S4-dev] derive next iter's active mask from the freeze decision already on device
        // (m_env_alpha == 0 ⇔ env converged this iter). Zero D2H — replaces the S4 host detection.
        // Frozen set is final here: the S3 per-env backtrack only halves nonzero alphas (never → 0).
        if(s4_dev_mask && m_env_alpha_valid)
            _mask_from_env_alpha<<<(active_group_count + 255) / 256, 256>>>(
                m_env_active, m_env_alpha, active_group_count);

        if(phase_time)
            CUDA_SAFE_CALL(cudaEventRecord(e2b));  // end S1 per-env-alpha / start lineSearch
        double alpha_before_line_search = alpha;
        if(merged_diag_sample)
        {
            std::vector<double3> positions(vertexNum), directions(vertexNum);
            std::vector<int> body_ids(vertexNum);
            std::vector<uint32_t> surface_ids(surf_vertexNum);
            CUDA_SAFE_CALL(cudaMemcpy(positions.data(), _vertexes,
                                      vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(directions.data(), _moveDir,
                                      vertexNum * sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(body_ids.data(), _point_body_id,
                                      vertexNum * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(surface_ids.data(), _surfVerts,
                                      surf_vertexNum * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            double3 normal;
            double offset;
            CUDA_SAFE_CALL(cudaMemcpy(
                &normal, _groundNormal, sizeof(double3), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(
                &offset, _groundOffset, sizeof(double), cudaMemcpyDeviceToHost));

            int max_vertex = -1;
            double max_component = -1.0;
            for(int vertex = 0; vertex < static_cast<int>(vertexNum); ++vertex)
            {
                const double3& direction = directions[vertex];
                const double component = std::max(
                    std::max(fabs(direction.x), fabs(direction.y)), fabs(direction.z));
                if(component > max_component)
                {
                    max_component = component;
                    max_vertex = vertex;
                }
            }

            int limiting_vertex = -1;
            double limiting_alpha = 1.0;
            double limiting_distance = 0.0;
            double limiting_coefficient = 0.0;
            for(uint32_t vertex : surface_ids)
            {
                const double3& position = positions[vertex];
                const double3& direction = directions[vertex];
                const double distance = normal.x * position.x + normal.y * position.y
                                      + normal.z * position.z - offset;
                const double coefficient = normal.x * direction.x + normal.y * direction.y
                                         + normal.z * direction.z;
                if(distance > 0.0 && coefficient > 0.0)
                {
                    const double candidate = std::min(
                        1.0, slackness_a * (distance / coefficient));
                    if(candidate < limiting_alpha)
                    {
                        limiting_alpha = candidate;
                        limiting_vertex = static_cast<int>(vertex);
                        limiting_distance = distance;
                        limiting_coefficient = coefficient;
                    }
                }
            }
            const double3 max_direction = max_vertex >= 0
                                              ? directions[max_vertex]
                                              : make_double3(0.0, 0.0, 0.0);
            printf("[merged-alpha-detail] frame=%d k=%d maxV=%d maxB=%d "
                   "maxDir=(%.17e,%.17e,%.17e) groundV=%d groundB=%d "
                   "groundAlpha=%.17e groundDist=%.17e groundCoef=%.17e\n",
                   s_dec_frame,
                   (int)k,
                   max_vertex,
                   max_vertex >= 0 ? body_ids[max_vertex] : -1,
                   max_direction.x,
                   max_direction.y,
                   max_direction.z,
                   limiting_vertex,
                   limiting_vertex >= 0 ? body_ids[limiting_vertex] : -1,
                   limiting_alpha,
                   limiting_distance,
                   limiting_coefficient);
        }
        lineSearch(TetMesh, alpha, alpha_CFL);

        if(merged_diag_sample)
            printf("[merged-alpha] frame=%d k=%d move=%.17e thr=%.17e "
                   "ground=%.17e narrow=%.17e narrowN=%d temp=%.17e "
                   "ccdN=%d cfl=%.17e refined=%.17e refine=%d "
                   "preLS=%.17e postLS=%.17e lsRatio=%.17e cp=%d gp=%d\n",
                   s_dec_frame, (int)k, distToOpt_PN, _newton_thr,
                   diag_ground_alpha, diag_narrow_alpha, diag_narrow_pairs, temp_alpha,
                   (int)h_ccd_cpNum, alpha_CFL, diag_refined_alpha, (int)diag_refine_used,
                   alpha_before_line_search, alpha,
                   alpha_before_line_search > 0.0 ? alpha / alpha_before_line_search : 0.0,
                   (int)h_cpNum[0], (int)h_gpNum);

        if(phase_time) CUDA_SAFE_CALL(cudaEventRecord(end3));
        postLineSearch(TetMesh, alpha);
        //computeGradientAndHessian(TetMesh);
        if(phase_time)
        {
            CUDA_SAFE_CALL(cudaEventRecord(end4));
            // Waiting for the final timing event is sufficient; avoid stalling
            // unrelated streams even in diagnostic mode.
            CUDA_SAFE_CALL(cudaEventSynchronize(end4));
            float time00 = 0, time11 = 0, time22 = 0, time33 = 0, time44 = 0;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time00, start, end0));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time11, end0, end1));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time22, end1, end2));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time33, end2, end3));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&time44, end3, end4));
            {   // time3 sub-split: S1 per-env alpha vs lineSearch
                float t3a = 0, t3b = 0;
                CUDA_SAFE_CALL(cudaEventElapsedTime(&t3a, end2, e2b));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&t3b, e2b, end3));
                g_t3_s1_ms += t3a;
                g_t3_ls_ms += t3b;
            }
            time0 += time00;
            time1 += time11;
            time2 += time22;
            time3 += time33;
            time4 += time44;
            destroy_iteration_events();
        }
        totalTimeStep += alpha;

        // Semi-implicit early exit (ref: arXiv 2512.12151, Algorithm 1)
        // beta tracks cumulative line-search progress; when alpha≈1 (good step),
        // beta decays fast -> early exit.  When alpha is small, beta stays large.
        // [semi-implicit x multi-env] beta/alpha are GLOBAL. Under the per-env
        // decoupled exit (DECOUPLE_THRESH) a global early break would re-couple the
        // batch: one env's good steps cut off still-unconverged mates at a
        // batch-dependent iter — the exact drift DECOUPLE_THRESH exists to prevent.
        // The decoupled path therefore ignores the semi-implicit exit (its per-env
        // frozen check already exits as soon as every env converged). A true per-env
        // beta belongs with the per-env productization work.
        const bool semi_decoupled = getenv("STIFF_DECOUPLE_THRESH") && m_env_alpha_valid;
        if(semi_implicit_enabled && semi_decoupled)
        {
            static bool noted = false;
            if(!noted)
            {
                printf("  [semi-implicit] NOTE: per-env decoupled exit active -> "
                       "beta runs PER-ENV (each env freezes at its own beta<=tol); "
                       "the global early exit is disabled.\n");
                noted = true;
            }
        }
        if(semi_implicit_enabled && k >= semi_implicit_min_iter && !semi_decoupled
           && drive_ratio >= 1.0)  // [drive-substep] ramp not done -> no early exit
        {
            if(TetMesh.h_groups_present)  // [N=1 guard] p2g is allocated (all -1) even single-env
            {   // multi-env scene, merged exit: legal but batch-coupled — say so once.
                static bool warned = false;
                if(!warned)
                {
                    printf("  [semi-implicit] WARNING: multi-env scene with a GLOBAL "
                           "semi-implicit exit — every env stops at the same Newton "
                           "iter, so a fast env can terminate a still-unconverged "
                           "mate (batch-coupled result). For per-env convergence "
                           "run with STIFF_DECOUPLE_THRESH=1 STIFF_PERENV_ALPHA=1.\n");
                    warned = true;
                }
            }
            semi_beta *= fmax(0.0, 1.0 - alpha);  // clamp: alpha>1 must not flip beta's sign
            if(semi_beta <= semi_implicit_beta_tol)
            {
                printf("  [semi-implicit] early exit at Newton iter %d (beta=%.6e, tol=%.6e)\n",
                       k, semi_beta, semi_implicit_beta_tol);
                k++;
                break;
            }
        }
    }
    // [multi-env S4] clear the linear-system mask so later/other solves are unmasked
    if(s4_mask_on) m_global_linear_system->set_env_mask(nullptr, nullptr, 0);
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for(int ii = 0; ii < iterV.size(); ii++)
    //{
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    if(g_gipc_log_level >= 1)
        printf("\n\n      Kappa: %f                               iteration k:  %d\n", Kappa, k);
    // [phase-time] cumulative GPU ms per phase (across frames). time0=Hessian/grad assembly,
    // time1=PCG linear solve, time2=CCD-BVH build, time3=line-search(+per-env alpha), time4=κ update.
    // Compare modes to localize the isolated/strict slowdown. Also reports Newton iters this frame.
    if(getenv("STIFF_PHASE_TIME"))
        printf("[phase-time cum-ms] Hess=%.0f PCG=%.0f ccdBVH=%.0f lineSearch=%.0f kappaUpd=%.0f | ls-split[s1=%.0f ls=%.0f] ls-inner[e=%.0f bvh=%.0f cp=%.0f step=%.0f] | this-frame-newton-iters=%d cum-pcg-iters=%lld\n",
               time0, time1, time2, time3, time4, g_t3_s1_ms, g_t3_ls_ms,
               g_ls_e_ms, g_ls_bvh_ms, g_ls_cp_ms, g_ls_step_ms, k, (long long)total_Cg_count);
    return k;
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


int    totalNT          = 0;
double totalTime        = 0;
int    total_Frames     = 0;
double ttime0           = 0;
double ttime1           = 0;
double ttime2           = 0;
double ttime3           = 0;
double ttime4           = 0;
bool   isUpdateBoundary = false;
void   GIPC::IPC_Solver(device_TetraData& TetMesh)
{
    //double animation_fullRate = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);
    //    if(isRotate&&total_Frames*IPC_dt>=2.2){
    //        isRotate = false;
    //        updateBoundary2(TetMesh);
    //    }
    if(isUpdateBoundary)
    {
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        buildBVH_FULLCCD(alpha);
        buildFullCP(alpha);
        if(h_ccd_cpNum > 0)
        {
            double slackness_m = 0.8;
            CUDA_SAFE_CALL(cudaMemsetAsync(m_ccd_alpha_invalid, 0, sizeof(int)));
            alpha              = std::min(alpha,
                             self_largestFeasibleStepSize(slackness_m, ensure_reduce_scratch(h_ccd_cpNum), h_ccd_cpNum));
        }
        //updateBoundary(TetMesh, alpha);

        CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem,
                                  TetMesh.vertexes,
                                  vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToDevice));
        updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
        stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);
        //step_forward(TetMesh, 1, true);

        bool rehash = true;

        buildBVH();
        int       numOfIntersect        = 0;
        const int boundary_isect_budget = line_search_max_iter > 0 ? line_search_max_iter : 64;
        while(isIntersected(TetMesh))
        {
            if(numOfIntersect >= boundary_isect_budget)
            {
                throw std::runtime_error(
                    "[StiffGIPC] boundary-move intersection persists after "
                    "backtracking (pre-move state likely already intersecting)");
            }
            printf("type 6 intersection happened:    %f\n", alpha);
            alpha /= 2.0;
            updateBoundaryMoveDir(TetMesh, alpha, total_Frames);
            numOfIntersect++;
            stepForward(TetMesh.vertexes,
                        TetMesh.temp_double3Mem,
                        _moveDir,
                        TetMesh.BoundaryType,
                        1,
                        true,
                        vertexNum);
            //step_forward(TetMesh, 1, true);
            buildBVH();
        }

        buildCP();
        printf("boundary alpha: %f\n  finished a step\n", alpha);
    }

    TetMesh.update_soft_constraint_target_position(total_Frames + 1, IPC_dt);
    //suggestKappa(Kappa);
    upperBoundKappa(Kappa);
    if(Kappa < 1e-16)
    {
        suggestKappa(Kappa);
    }
    initKappa(TetMesh);
    //Kappa = 1e4;
#ifdef USE_FRICTION
    ensure_frictionBuffers();  // [0be8da3-port] grow-only, no per-frame malloc
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    int    k           = 0;
    double time0       = 0;
    double time1       = 0;
    double time2       = 0;
    double time3       = 0;
    double time4       = 0;
    // [iron-law] frame-start quarantine scan — must precede every CCD-alpha
    // kernel of this frame (see quarantineGroundInfeasibleAtFrameStart).
    quarantineGroundInfeasibleAtFrameStart();

    while(true)
    {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(TetMesh, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = std::min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = std::max(minMaxDist1.y, minMaxDist2.y);


        bool finishMotion = animation_fullRate > 0.99 ? true : false;

        if(finishMotion)
        {
            tempFree_closeConstraint();
            break;
            //}
        }
        else
        {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        //updateVelocities(TetMesh);

        //computeXTilta(TetMesh, 1);
#ifdef USE_FRICTION
        ensure_frictionBuffers();  // [0be8da3-port] grow-only, no sub-iter realloc
        buildFrictionSets();
#endif
    }

#ifdef USE_FRICTION
    // [0be8da3-port] friction buffers persist across frames; freed in FREE_DEVICE_MEM.
#endif

    updateVelocities(TetMesh);

    computeXTilta(TetMesh, 1);
    cudaEventRecord(end0);
    // Engine.step remains synchronous, but only waits for this PTDS chain.
    CUDA_SAFE_CALL(cudaEventSynchronize(end0));
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    cudaEventDestroy(start);
    cudaEventDestroy(end0);
    totalTime += tttime;
    total_Frames++;
    if(g_gipc_log_level >= 1)
        printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);

    // [multi-env P3a] validate the segmented per-env reduction primitive on real
    // data: per-env vertex count (must match the substrate, e.g. 11433/env) and a
    // real per-env quantity (velocity norm). Read-only diagnostic, gated.
    if(getenv("STIFF_PENV_STATS") && TetMesh.d_point_to_group && TetMesh.h_groups_present)
    {
        const int NG = TetMesh.h_group_count;
        static double* d_sq = nullptr;
        static int*    d_cnt = nullptr;
        if(!d_sq)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_sq, kEnvAlphaSlots * sizeof(double)));
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_cnt, kEnvAlphaSlots * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemset(d_sq, 0, NG * sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(d_cnt, 0, NG * sizeof(int)));
        int bs = 256, gs = (vertexNum + bs - 1) / bs;
        _per_env_sqnorm_accum<<<gs, bs>>>(TetMesh.d_point_to_group, TetMesh.velocities,
                                          d_sq, d_cnt, vertexNum, NG);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::vector<double> h_sq(NG);
        std::vector<int> h_cnt(NG);
        CUDA_SAFE_CALL(cudaMemcpy(h_sq.data(), d_sq, NG * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_cnt.data(), d_cnt, NG * sizeof(int), cudaMemcpyDeviceToHost));
        printf("[P3a-reduce] frame %d per-env:", total_Frames);
        for(int g = 0; g < NG; ++g)
            if(h_cnt[g] > 0) printf(" g%d[n=%d |v|=%.4e]", g, h_cnt[g], sqrt(h_sq[g]));
        printf("\n");
    }

    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;


    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << ttime0 / 1000.0 << std::endl;
    outTime << "time1: " << ttime1 / 1000.0 << std::endl;
    outTime << "time2: " << ttime2 / 1000.0 << std::endl;
    outTime << "time3: " << ttime3 / 1000.0 << std::endl;
    outTime << "time4: " << ttime4 / 1000.0 << std::endl;
    outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;

    outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "averageCollision: " << totalCollisionPairs / totalNT << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_count << std::endl;
    outTime.close();


    auto& stats = gipc::Statistics::instance();

    stats.at_current_frame()["timer"] =
        gipc::GlobalTimer::current()->report_merged_as_json();
    if(g_gipc_log_level >= 1)
        gipc::GlobalTimer::current()->print_merged_timings();
    gipc::GlobalTimer::current()->clear();
    stats.write_to_file(std::string{gipc::output_dir()} + "/stats.json");

    auto f = stats.frame();
    stats.frame(f + 1);
}
