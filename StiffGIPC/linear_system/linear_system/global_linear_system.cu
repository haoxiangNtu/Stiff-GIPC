#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

extern std::vector<double> g_audit_abd_P, g_audit_abd_M;
extern int g_audit_abd_nb;
namespace gipc
{
bool GlobalLinearSystem::build_linear_system()
{
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);
    // PCG's initial guess must start finite; ::resize() leaves new elements
    // uninitialized and the IterativeSolver reads x on the very first spmv.
    CUDA_SAFE_CALL(cudaMemset(m_x.view().data(), 0,
                              total_rhs_count * sizeof(Float)));

    // [P0-mem pre-grow → towel-strict root fix] grow the triplet storage before
    // the solve. The original [P0-mem] version used ensure_capacity_discard here
    // on the assumption that the buffer held "last iteration's dead triplets" —
    // WRONG: GIPC assembles the full triplet stream in computeGradientAndHessian
    // BEFORE solve_linear_system(), so this point sits AFTER assembly and the
    // buffer holds the live matrix. The discarding grow destroyed it the first
    // time 2*length crossed the current allocation (towel strict frame 26:
    // 2*168469=336938 > oldcap 336199 → matrix row/col replaced by stale pages,
    // all (0,0) → unique-key reduction collapses to 32 → SpMV reads garbage rows
    // like 105631 ≫ dof=961 → illegal address). Now grows PRESERVING [0:length)
    // with the same absolute
    // margin cap; the converter scratch [length:2length) needs no preservation.
    // Hash scratch stays discard-resized (update_hash_value/convert rewrite it).
    {
        auto*           gt     = gipc_global_triplet;
        const long long length = gt->global_triplet_offset;
        gt->ensure_capacity_preserve((size_t)length, (size_t)(2LL * length));
        if(gt->global_external_max_capcity < length)
        {
            long long hm = length * 3 / 10;
            long long hcap_bytes = 512ll * 1024 * 1024 / (long long)sizeof(uint64_t);
            if(hm > hcap_bytes) hm = hcap_bytes;
            gt->resize_collision_hash_size((size_t)(length + hm));
            gt->global_external_max_capcity = (int)(length + hm);
        }
    }

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();

    // [audit 2026-10-09] STIFF_AUDIT_ABDPRE=1: compare each ABD body's preconditioner block (before
    // inversion) with the same 12x12 diagonal block of the assembled matrix; print the difference
    // relative to the body's mass matrix (a difference equal to M means the mass is counted twice).
    if(getenv("STIFF_AUDIT_ABDPRE") && g_audit_abd_nb > 0)
    {
        static int prints = 0;
        if(prints < 12)
        {
            ++prints;
            const int nb = g_audit_abd_nb, nu = gipc_global_triplet->h_unique_key_number;
            std::vector<int>    hr(nu), hc(nu);
            std::vector<double> hv((size_t)nu * 9);
            CUDA_SAFE_CALL(cudaMemcpy(hr.data(), gipc_global_triplet->block_row_indices(), nu * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hc.data(), gipc_global_triplet->block_col_indices(), nu * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(hv.data(), gipc_global_triplet->block_values(), (size_t)nu * 9 * sizeof(double), cudaMemcpyDeviceToHost));
            std::vector<double> T((size_t)nb * 144, 0.0);
            for(int t = 0; t < nu; ++t)
            {
                int r = hr[t], c = hc[t];
                if(r / 4 != c / 4 || r / 4 >= nb) continue;
                double* B = &T[(size_t)(r / 4) * 144];
                const double* H = &hv[(size_t)t * 9];   // Eigen column-major 3x3
                for(int a = 0; a < 3; ++a)
                    for(int b = 0; b < 3; ++b)
                    {
                        double h = H[b * 3 + a];
                        B[((r % 4) * 3 + a) * 12 + (c % 4) * 3 + b] += h;
                        if(r != c) B[((c % 4) * 3 + b) * 12 + (r % 4) * 3 + a] += h;
                    }
            }
            double worst_dm = 0.0, worst_d = 0.0, mean_dm = 0.0;
            for(int b = 0; b < nb; ++b)
            {
                double nm = 0, nd = 0, ndm = 0, na = 0;
                for(int q = 0; q < 144; ++q)
                {
                    // P and M are Eigen 12x12 column-major; T is row-major -> transpose index
                    int i = q / 12, j = q % 12;
                    double p = g_audit_abd_P[(size_t)b * 144 + j * 12 + i];
                    double m = g_audit_abd_M[(size_t)b * 144 + j * 12 + i];
                    double a = T[(size_t)b * 144 + q];
                    double d = p - a;
                    nm += m * m; nd += d * d; ndm += (d - m) * (d - m); na += a * a;
                }
                nm = std::sqrt(nm); nd = std::sqrt(nd); ndm = std::sqrt(ndm); na = std::sqrt(na);
                worst_d  = std::max(worst_d, nd / std::max(nm, 1e-300));
                worst_dm = std::max(worst_dm, ndm / std::max(nm, 1e-300));
                mean_dm += ndm / std::max(nm, 1e-300) / nb;
                if(prints <= 2)
                    printf("[audit-abdpre] body %2d: |A_bb| %.3e |M| %.3e | |P - A_bb|/|A_bb| %.3e | (P - A_bb) vs M: |(P-A)-M|/|M| %.3e, |(P-A)-M|/|A_bb| %.3e\n",
                           b, na, nm, nd / std::max(na, 1e-300), ndm / std::max(nm, 1e-300), ndm / std::max(na, 1e-300));
            }
            printf("[audit-abdpre] %d bodies: max |P - A_bb|/|M| %.3e; |(P - A_bb) - M|/|M| max %.3e mean %.3e\n",
                   nb, worst_d, worst_dm, mean_dm);
        }
    }

    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);

    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    // All subsystem retrieval kernels and the subsequent GIPC line-search work
    // use the per-thread default stream (the project is compiled with
    // --default-stream=per-thread). Stream ordering is sufficient here; the
    // former wait_device() stalled every device stream once per Newton step.
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    MUDA_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

// [multi-env S4] zero the RHS for masked-env DOFs. DOF i -> block i/3 -> group
// dof_to_group[i/3]; if that env is masked, b[i]=0. Makes masked envs produce 0
// in the PCG (block-diagonal after P1 -> no pollution of the shared alpha/beta).
__global__ void _s4_zero_masked_rhs(double* b, const int* active, const int* d2g,
                                    int n, int ng)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = d2g[i / 3];
    if(g >= 0 && g < ng && active[g] == 0) b[i] = 0.0;
}

// [live-prefix PCG] first index in the (row, col)-sorted unique triplets whose row >= key.
__global__ void _live_lower_bound(const int* rows, int n, int key, int* out)
{
    int lo = 0, hi = n;
    while(lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if(rows[mid] < key) lo = mid + 1;
        else hi = mid;
    }
    *out = lo;
}

// [live-prefix PCG] verification: count unique blocks that couple a live block to a
// dead one with any nonzero entry, and dead RHS entries that are nonzero.
__global__ void _live_coupling_check(const Eigen::Matrix3d* vals, const int* rows, const int* cols,
                                     int n_trip, int live_blocks, const double* b, int n_full,
                                     int n_live, unsigned long long* out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t < n_trip)
    {
        int  i = rows[t], j = cols[t];
        bool li = i < live_blocks, lj = j < live_blocks;
        if(li != lj)
        {
            const double* m = vals[t].data();
            bool nz = false;
            for(int q = 0; q < 9; ++q) nz |= (m[q] != 0.0);
            if(nz) atomicAdd(&out[0], 1ull);
            atomicAdd(&out[1], 1ull);
        }
    }
    int d = n_live + t;
    if(d < n_full && b[d] != 0.0) atomicAdd(&out[2], 1ull);
}

SizeT GlobalLinearSystem::live_prefix_for_solve(SizeT n_full)
{
    static int s_env = -1;
    if(s_env < 0)
    {
        const char* e = getenv("STIFF_PCG_LIVE");
        s_env = e ? atoi(e) : 1;
    }
    const SizeT n = m_live_dof_request;
    if(!s_env || n == 0 || n >= n_full || (n % 3) != 0)
        return 0;
    if(m_s4_active)  // multi-env masked path keeps the full-length solve
        return 0;
    if(getenv("STIFF_SEGMENTED_PCG") && m_s4_dof_to_group && m_s4_ng > 0)
        return 0;  // seg_pcg: not prefix-aware
    if(!m_global_preconditioner || !m_global_preconditioner->prefix_capable())
        return 0;
    for(auto& p : m_local_preconditioners)
    {
        const SizeT off = (SizeT)p->m_subsystem->dof_offset()[0];
        const SizeT cnt = p->m_subsystem->right_hand_side_dof();
        if(off + cnt > n)
            return 0;
    }
    return n;
}

void GlobalLinearSystem::begin_live_solve(SizeT n_live)
{
    const int live_blocks = (int)(n_live / 3);
    const int n_trip      = gipc_global_triplet->h_unique_key_number;
    static int* d_lb      = nullptr;
    if(!d_lb)
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_lb, sizeof(int)));
    _live_lower_bound<<<1, 1>>>(gipc_global_triplet->block_row_indices(), n_trip, live_blocks, d_lb);
    int cut = n_trip;
    CUDA_SAFE_CALL(cudaMemcpy(&cut, d_lb, sizeof(int), cudaMemcpyDeviceToHost));
    // Round up to a whole warp: the SpMV's warp-segmented row reduce then sees exactly
    // the lanes of the full-length launch for every live row (lanes past the cut hold
    // dead rows, which the kernel reduces but never deposits).
    long long cut32 = ((long long)cut + 31) / 32 * 32;
    if(cut32 > n_trip) cut32 = n_trip;
    m_spmv_triplets    = (int)cut32;
    m_spmv_live_blocks = live_blocks;
    m_solver->m_live_n = n_live;
    static const bool fulldot = getenv("STIFF_PCG_LIVE_FULLDOT") != nullptr;
    m_solver->m_dot_n = fulldot ? m_b.size() : 0;
}

void GlobalLinearSystem::end_live_solve()
{
    m_spmv_triplets    = -1;
    m_spmv_live_blocks = 0;
    m_solver->m_live_n = 0;
    m_solver->m_dot_n  = 0;
}

// STIFF_PCG_LIVE_VERIFY=1: solve every system full-length AND on the live prefix and
// compare (=2 adds a second full-length solve: the run-to-run noise floor). Keeps the
// live-prefix result, so the run follows the optimized trajectory.
SizeT GlobalLinearSystem::verify_live_solve(SizeT n_live, SizeT n_full)
{
    const int live_blocks = (int)(n_live / 3);
    // structural check: no nonzero block may couple a live and a dead block
    static unsigned long long* d_cnt = nullptr;
    if(!d_cnt)
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_cnt, 3 * sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemset(d_cnt, 0, 3 * sizeof(unsigned long long)));
    {
        int n_trip = gipc_global_triplet->h_unique_key_number;
        int n_dead = (int)(n_full - n_live);
        int n      = std::max(n_trip, n_dead);
        _live_coupling_check<<<(n + 255) / 256, 256>>>(gipc_global_triplet->block_values(),
                                                       gipc_global_triplet->block_row_indices(),
                                                       gipc_global_triplet->block_col_indices(),
                                                       n_trip, live_blocks, m_b.view().data(),
                                                       (int)n_full, (int)n_live, d_cnt);
    }
    unsigned long long cnt[3] = {0, 0, 0};
    CUDA_SAFE_CALL(cudaMemcpy(cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));

    static int verify = atoi(getenv("STIFF_PCG_LIVE_VERIFY"));
    auto fetch = [&](std::vector<double>& h)
    {
        h.resize(n_full);
        CUDA_SAFE_CALL(cudaMemcpy(h.data(), m_x.view().data(), n_full * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    };
    std::vector<double> xa, xb, xc;
    SizeT ka = m_solver->solve(m_x, m_b);
    fetch(xa);
    SizeT kb = 0;
    if(verify >= 2)
    {
        kb = m_solver->solve(m_x, m_b);
        fetch(xb);
    }
    begin_live_solve(n_live);
    SizeT kc = m_solver->solve(m_x, m_b);
    end_live_solve();
    fetch(xc);

    auto cmp = [&](const std::vector<double>& u, const std::vector<double>& v,
                   double& maxabs, long long& ndiff)
    {
        maxabs = 0.0;
        ndiff  = 0;
        for(SizeT q = 0; q < n_full; ++q)
            if(u[q] != v[q])
            {
                ++ndiff;
                maxabs = std::max(maxabs, std::fabs(u[q] - v[q]));
            }
    };
    double    xmax = 0.0;
    long long tail_nz = 0;
    for(SizeT q = 0; q < n_full; ++q)
    {
        xmax = std::max(xmax, std::fabs(xa[q]));
        if(q >= n_live && xa[q] != 0.0) ++tail_nz;
    }
    double    d_ac = 0.0, d_ab = 0.0;
    long long n_ac = 0, n_ab = 0;
    cmp(xa, xc, d_ac, n_ac);
    if(verify >= 2) cmp(xa, xb, d_ab, n_ab);
    static int call = 0;
    char kb_s[32] = "", ab_s[96] = "";
    if(verify >= 2)
    {
        snprintf(kb_s, sizeof(kb_s), " / %zu", (size_t)kb);
        snprintf(ab_s, sizeof(ab_s), " | full-vs-full: %lld differ, max %.3e (rel %.3e)", n_ab, d_ab,
                 xmax > 0 ? d_ab / xmax : 0.0);
    }
    printf("[live-verify] #%d dof %zu live %zu (%.1f%%) iters full %zu%s live %zu | "
           "full-vs-live: %lld differ, max %.3e (rel %.3e)%s | full tail nonzero %lld | "
           "live/dead coupling blocks %llu (nonzero %llu) | dead rhs nonzero %llu\n",
           call++, (size_t)n_full, (size_t)n_live, 100.0 * n_live / n_full, (size_t)ka, kb_s,
           (size_t)kc, n_ac, d_ac, xmax > 0 ? d_ac / xmax : 0.0, ab_s, tail_nz, cnt[1], cnt[0],
           cnt[2]);
    return kc;
}

}  // namespace gipc
// [audit 2026-10-09] PCG phase split (STIFF_PCG_SPLIT=1; synchronising event timers, per frame,
// reset and printed by GIPC::solve_subIP under STIFF_AUDIT_STATS).
double    g_pcg_build_ms = 0.0, g_pcg_solve_ms = 0.0, g_pcg_setup_ms = 0.0, g_pcg_iter_ms = 0.0;
long long g_pcg_solves = 0;
namespace gipc
{
static bool pcg_split_on() { static const bool on = getenv("STIFF_PCG_SPLIT") != nullptr; return on; }

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    cudaEvent_t sp_a = nullptr, sp_b = nullptr, sp_c = nullptr;
    if(pcg_split_on())
    {
        cudaEventCreate(&sp_a); cudaEventCreate(&sp_b); cudaEventCreate(&sp_c);
        cudaEventRecord(sp_a, cudaStreamPerThread);
    }
    bool success = build_linear_system();
    if(sp_a) cudaEventRecord(sp_b, cudaStreamPerThread);
    if(!success)
    {
        if(sp_a) { cudaEventDestroy(sp_a); cudaEventDestroy(sp_b); cudaEventDestroy(sp_c); }
        return 0;
    }
    MUDA_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    // [S4] RHS-zero for masked envs (correctness of per-env masking)
    if(m_s4_active && m_s4_dof_to_group && m_b.size() > 0)
    {
        int n = (int)m_b.size(), bs = 256, gn = (n + bs - 1) / bs;
        _s4_zero_masked_rhs<<<gn, bs>>>(m_b.view().data(), m_s4_active,
                                        m_s4_dof_to_group, n, m_s4_ng);
    }
    const SizeT n_full = m_b.size();
    const SizeT n_live = live_prefix_for_solve(n_full);
    SizeT       iter   = 0;
    if(n_live && getenv("STIFF_PCG_LIVE_VERIFY"))
    {
        // the verification keeps the live-prefix result; its iteration count is
        // the one reported (printed alongside the full-length count)
        iter = verify_live_solve(n_live, n_full);
    }
    else if(n_live)
    {
        begin_live_solve(n_live);
        iter = m_solver->solve(m_x, m_b);
        end_live_solve();
    }
    else
        iter = m_solver->solve(m_x, m_b);
    if(sp_a)
    {
        cudaEventRecord(sp_c, cudaStreamPerThread);
        cudaEventSynchronize(sp_c);
        float ab = 0, bc = 0;
        cudaEventElapsedTime(&ab, sp_a, sp_b);
        cudaEventElapsedTime(&bc, sp_b, sp_c);
        g_pcg_build_ms += ab;
        g_pcg_solve_ms += bc;
        g_pcg_solves += 1;
        cudaEventDestroy(sp_a); cudaEventDestroy(sp_b); cudaEventDestroy(sp_c);
    }
    distribute_solution();
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                              muda::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
    {
        // BufferView::copy_from() creates a temporary BufferLaunch and calls
        // wait(), which is illegal during CUDA Graph capture. Keep the identity
        // stage ordered on the project's per-thread default stream without a
        // host synchronization point.
        MUDA_ASSERT(z.size() == r.size(),
                    "Identity preconditioner input/output size mismatch.");
        CUDA_SAFE_CALL(cudaMemcpyAsync(z.data(),
                                       r.data(),
                                       r.size() * sizeof(Float),
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
    }

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



void GlobalLinearSystem::convert_new()
{
    // [P1-dyn / 9c2da9c] safety net for the dynamic (non-hybrid) triplet buffer:
    // ensure it holds the converter's [length:2*length) scratch/output region and the
    // hash scratch holds `length`. length here is EXACT (= assembled
    // global_triplet_offset), so this guarantees the converter never overflows.
    // [audit lens-A fix] guarded preserve API instead of the bare
    // resize+reserve pair: on a growth where length itself already exceeded
    // the capacity, resize_triplets() would have silently DESTROYED the live
    // matrix (free->malloc) — reachable in hybrid mode, which skips the
    // [P1-dyn] frame-start bound grow. The guarded call throws loudly instead.
    // No-op when the buffer is already large -> byte-for-byte unchanged.
    {
        auto*           gt     = gipc_global_triplet;
        const long long length = gt->global_triplet_offset;
        if(length >= 1)
        {
            gt->ensure_capacity_preserve((size_t)length, (size_t)(2LL * length));
            if(gt->global_external_max_capcity < length)
            {
                gt->resize_collision_hash_size((size_t)((long long)length * 13 / 10));
                gt->global_external_max_capcity = (int)((long long)length * 13 / 10);
            }
        }
    }
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
//#ifndef SymGH
//    m_converter.ge2sym(*gipc_global_triplet);
//#endif
}



void GlobalLinearSystem::spmv(Float                         a,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
{

    // [live-prefix PCG] during a live solve only the warp-rounded prefix of the sorted
    // unique triplets can touch a live row; the kernel zeroes any dead-row/column term.
    const int n_trip = (m_spmv_triplets >= 0) ? m_spmv_triplets
                                              : gipc_global_triplet->h_unique_key_number;
    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                n_trip,
                                x,
                                b,
                                y,
                                m_s4_active,        // [S4] skip masked envs' triplets
                                m_s4_dof_to_group,
                                m_s4_ng,
                                m_spmv_live_blocks);
}
}  // namespace gipc
