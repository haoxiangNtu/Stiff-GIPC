// ============================================================================
// energy/01_energy_host_dispatch.inl — the energy-term HOST DISPATCH layer
// (v0.8.6 energy separation, phase E1a; blueprint docs/ENERGY_SEPARATION_PLAN.md).
//
// Owns: Energy_Add_Reduction_Algorithm(+_DeviceOut) — the type-switch that
// sizes and launches every energy-term reduction kernel — the global/per-env
// energy combiners, and the computeEnergy family. The term table (type <->
// size source <-> kernel <-> G/H owner) is energy/energy_terms.h; change the
// switch and the table in the SAME commit.
//
// Compiled inside the GIPC.cu composite TU, included immediately BEFORE
// gipc_modules/14 (whose line-search decide kernels _s3_decide /
// _global_ls_decide / _s3_halve_all deliberately STAY there: they are solver
// policy, not energy terms). Bodies below are verbatim moves from the head
// of gipc_modules/14; per-term kernel bodies (gipc_modules/07) and G/H
// assembly move in later E1 slices.
// ============================================================================
// ── verbatim from gipc_modules/14 (pre-E1a lines 1..485) ──
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
            (m_fric_anchor_on && fric_anchor)    ? fric_anchor    : nullptr,
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
            (m_fric_anchor_on && fric_anchor_gd) ? fric_anchor_gd : nullptr,
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


// [strict-LS N-invariance] fixed-order combine of the BINNED_K-wide bins block into
// the plain per-env slice (see _penv_energy_accum in term_common.cuh).
__global__ void _penv_bins_combine(const double* bins, double* pe, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    pe[g] = binned_combine(bins + (size_t)g * BINNED_K);
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
    int point_offset = abd_fem_count_info.fem_point_offset;

    // [E2] registry-driven sizing: each term owns its size in ITS file; the
    // X-macro table (energy/energy_terms.h) is the single list. Unknown type
    // (incl. the removed vestigial 11) sizes to 0 -> zeroed slot, no launch.
    int numbers = 0;
#define GIPC_ENERGY_SIZE_CASE(id, name) case id: numbers = energy_size_##name(); break;
    switch(type) { GIPC_ENERGY_TERMS(GIPC_ENERGY_SIZE_CASE) default: break; }
#undef GIPC_ENERGY_SIZE_CASE

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

    // [strict-LS N-invariance] kernels deposit into a BINNED_K-wide bins block
    // (order-independent under g_det_reduce), combined into pe after the switch.
    // pe itself stays a plain per-env slice — every downstream consumer unchanged.
    double* pe_bins = nullptr;
    if(pe)
    {
        static double* s_pe_bins = nullptr;
        static int     s_pe_cap  = 0;
        if(s_pe_cap < ng)
        {
            if(s_pe_bins) CUDA_SAFE_CALL(cudaFree(s_pe_bins));
            CUDA_SAFE_CALL(cudaMalloc((void**)&s_pe_bins,
                                      (size_t)ng * BINNED_K * sizeof(double)));
            s_pe_cap = ng;
        }
        pe_bins = s_pe_bins;
        CUDA_SAFE_CALL(
            cudaMemsetAsync(pe_bins, 0, (size_t)ng * BINNED_K * sizeof(double)));
    }

    // [E2] registry-driven launch: adding a constitutive term = its file
    // (energy_size_/energy_launch_ members) + ONE row in GIPC_ENERGY_TERMS.
#define GIPC_ENERGY_LAUNCH_CASE(id, name)                                      \
    case id:                                                                   \
        energy_launch_##name(TetMesh, queue, numbers, blockNum, threadNum,     \
                             sharedMsize, pe_bins, p2g, ng, tet_offset,        \
                             point_offset, energy_kappa);                      \
        break;
    switch(type) { GIPC_ENERGY_TERMS(GIPC_ENERGY_LAUNCH_CASE) default: break; }
#undef GIPC_ENERGY_LAUNCH_CASE

    // [strict-LS N-invariance] fixed-order combine of the bins block into the slice.
    if(pe_bins)
        _penv_bins_combine<<<(ng + 255) / 256, 256>>>(pe_bins, pe, ng);

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
        static const char* kSlotNames[15] = {
            "FEM_kinetic", "FEM_elastic(dt2)", "membrane(dt2)", "bending(dt2)",
            "soft", "ground", "barrier(xKappa)", "friction(xmu)", "gfriction(xmu)",
            "ABD_kinetic", "ABD_shape", "ABD_joint", "ABD_rev_drive",
            "ABD_prismatic", "ABD_pri_drive"};
        for(int si = 0; si < kEnergySlotCount; ++si)
            printf("[energy-slot] %-16s = %.17e%s\n", kSlotNames[si], slots[si],
                   std::isnan(slots[si]) ? "   <-- NaN" : "");
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

// ── verbatim from gipc_modules/14 (pre-E1a lines 541..670) ──
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
    // [descriptor phase-0b] instance-owned (was function-static, shared across
    // engines in one process); PE_STRIDE is a compile-time constant so the
    // size is engine-independent — freed in FREE_DEVICE_MEM.
    if(!m_pe_all)
        CUDA_SAFE_CALL(cudaMalloc(
            (void**)&m_pe_all, (size_t)PE_SLOTS * PE_STRIDE * sizeof(double)));
    // [backport] throwaway sink for DeviceOut's global scalar (ignored here; the
    // per-env path uses the pe slices, and full_sum is assembled on host below).
    if(!m_energy_sink)
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_energy_sink, sizeof(double)));
    double* pe_all = m_pe_all;
    double* g_sink = m_energy_sink;

    auto slice = [&](int s) { return pe_all + (size_t)s * PE_STRIDE; };
    bool perenv_k = m_mode_config.decouple_thresh
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
