#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>

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

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
        return 0;
    MUDA_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    // [S4] RHS-zero for masked envs (correctness of per-env masking)
    if(m_s4_active && m_s4_dof_to_group && m_b.size() > 0)
    {
        int n = (int)m_b.size(), bs = 256, gn = (n + bs - 1) / bs;
        _s4_zero_masked_rhs<<<gn, bs>>>(m_b.view().data(), m_s4_active,
                                        m_s4_dof_to_group, n, m_s4_ng);
    }
    auto iter = m_solver->solve(m_x, m_b);
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

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y,
                                m_s4_active,        // [S4] skip masked envs' triplets
                                m_s4_dof_to_group,
                                m_s4_ng);
}
}  // namespace gipc
