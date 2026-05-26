#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;

    double temp = a[idx] * b[idx];

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    //double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}



__global__ void add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    extern __shared__ double tep[];
    if(idx >= numbers)
        return;
    double temp = mem[idx];
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}

// === Device-scalar variants for PCG D2H elimination audit ===

__global__ void update_vector_dx_r_dev(
    double* dx, double* r, const double* c, const double* q, const double* d_alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers) return;
    double a = *d_alpha;
    dx[idx] = dx[idx] + a * c[idx];
    r[idx]  = r[idx] - a * q[idx];
}

__global__ void update_vector_c_dev(
    double* c, const double* s, const double* d_beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers) return;
    c[idx] = s[idx] + (*d_beta) * c[idx];
}

// === Step E: fused PCG kernels ===
// Each thread re-derives alpha = *d_rz / *d_dot_res. Saves 1 launch per
// PCG iter (the separate compute_alpha_kernel<<<1,1>>>). Cost: every
// thread does the divide instead of just one — negligible for 25K-thread
// kernels (one div is ~30ns on Ada FP64; launch overhead is ~500ns).
// Soundness: if dot_res <= 0 || !isfinite, thread 0 sets d_break (same
// guard the original compute_alpha_kernel had); axpy still runs with
// alpha = inf/nan but won't propagate further because d_break causes the
// host-side break check to fire on next K-stride sync.
__global__ void update_vector_dx_r_fused(
    double* dx, double* r, const double* c, const double* q,
    const double* d_rz, const double* d_dot_res, int* d_break, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers) return;
    double dot = *d_dot_res;
    // Soundness: same exit criteria as original compute_alpha_kernel —
    // dot_res<=0 or non-finite means PCG has effectively converged (PD A
    // implies p^T A p > 0 unless p = 0). Set break flag and SKIP axpy
    // on every thread (otherwise alpha=inf/nan would propagate into x/r).
    if(!isfinite(dot) || dot <= 0.0)
    {
        if(idx == 0) *d_break = 1;
        return;
    }
    double a = (*d_rz) / dot;
    dx[idx] = dx[idx] + a * c[idx];
    r[idx]  = r[idx] - a * q[idx];
}

// Same trick for beta + axpy on p. Swap (d_rz = d_rz_new) and convergence
// check are deferred to a tiny <<<1,1>>> post kernel below.
__global__ void update_vector_c_fused(
    double* c, const double* s, const double* d_rz_new, const double* d_rz_old, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers) return;
    double rz_old = *d_rz_old;
    // Soundness: rz_old should be > 0 in a healthy PCG (it's |r|_M from
    // previous iter). If it's 0/non-finite, original compute_beta_and_swap
    // would have produced inf and propagated. Skip axpy under the same
    // condition the original would have failed on.
    if(!isfinite(rz_old) || rz_old == 0.0) return;
    double b = (*d_rz_new) / rz_old;
    c[idx] = s[idx] + b * c[idx];
}

// Combines the swap (d_rz = d_rz_new) with the convergence check into one
// <<<1,1>>> kernel. Replaces the separate compute_beta_and_swap_kernel +
// check_convergence_kernel pair (saves 1 launch per iter).
__global__ void post_iter_swap_and_check(
    double* d_rz, const double* d_rz_new, const double* d_rz0,
    double tol_rate, int* d_break)
{
    double new_v = *d_rz_new;
    *d_rz = new_v;
    if(fabs(new_v) <= tol_rate * (*d_rz0))
        *d_break = 1;
}

// alpha = rz / dot_res; if dot_res <= 0 or non-finite, set break flag.
__global__ void compute_alpha_kernel(const double* d_rz,
                                     const double* d_dot_res,
                                     double*       d_alpha,
                                     int*          d_break)
{
    double dot = *d_dot_res;
    if(!isfinite(dot) || dot <= 0.0)
    {
        *d_break = 1;
        *d_alpha = 0.0;
        return;
    }
    *d_alpha = (*d_rz) / dot;
}

// beta = rz_new / rz; copy rz_new -> rz for next iter.
__global__ void compute_beta_and_swap_kernel(double* d_rz,
                                             const double* d_rz_new,
                                             double* d_beta)
{
    double new_v = *d_rz_new;
    *d_beta = new_v / (*d_rz);
    *d_rz   = new_v;
}

// converged if |rz| <= tol * rz0; OR-into d_break.
__global__ void check_convergence_kernel(const double* d_rz,
                                         const double* d_rz0,
                                         double        tol_rate,
                                         int*          d_break)
{
    if(fabs(*d_rz) <= tol_rate * (*d_rz0))
        *d_break = 1;
}

// d_rz0 = d_rz (init only).
__global__ void copy_scalar_kernel(double* dst, const double* src)
{
    *dst = *src;
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

// === Step E: cub-based fused dot product ===
// Uses cub::DeviceReduce::Sum + a TransformInputIterator that fuses the
// elementwise multiply (a[i] * b[i]) with the tree reduction, into a
// single kernel launch (instead of the 2-3 launches of the manual
// PCG_vdv_Reduction + add_reduction loop).
struct DotProductOp
{
    const double* a;
    const double* b;
    __host__ __device__ __forceinline__
    double operator()(int i) const { return a[i] * b[i]; }
};

// Cached cub temp storage (lives in PCGSolver, passed in by ref).
void Cub_PCG_DotReduction(double* A, double* B, int n, double* d_out,
                          void** cub_temp_ptr, size_t* cub_temp_bytes)
{
    if(n < 1) {
        cudaMemset(d_out, 0, sizeof(double));
        return;
    }

    cub::CountingInputIterator<int>                       counter(0);
    cub::TransformInputIterator<double, DotProductOp,
                                cub::CountingInputIterator<int>> input(
        counter, DotProductOp{A, B});

    // Query temp size on first call; alloc lazily.
    if(*cub_temp_ptr == nullptr)
    {
        cub::DeviceReduce::Sum(nullptr, *cub_temp_bytes, input, d_out, n);
        cudaMalloc(cub_temp_ptr, *cub_temp_bytes);
    }
    cub::DeviceReduce::Sum(*cub_temp_ptr, *cub_temp_bytes, input, d_out, n);
}

// Device-output variant: leaves the reduced scalar in *d_out (which can be
// `temp` itself or any device address). Skips the final cudaMemcpy/D2H.
void My_PCG_General_v_v_Reduction_DeviceOut(double* temp, double* A, double* B,
                                            int vertexNum, double* d_out)
{
    int numbers = vertexNum;
    if(numbers < 1) {
        cudaMemset(d_out, 0, sizeof(double));
        return;
    }
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }

    if(d_out != temp)
        cudaMemcpyAsync(d_out, temp, sizeof(double), cudaMemcpyDeviceToDevice);
}

extern void My_PCG_General_v_v_Reduction_DeviceOut(double* temp, double* A, double* B,
                                                   int vertexNum, double* d_out);

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}

PCGSolver::~PCGSolver()
{
    if(d_scalars_alloced)
    {
        cudaFree(d_rz);
        cudaFree(d_rz0);
        cudaFree(d_rz_new);
        cudaFree(d_dot_res);
        cudaFree(d_alpha);
        cudaFree(d_beta);
        cudaFree(d_break);
    }
    if(cub_temp_ptr) cudaFree(cub_temp_ptr);
    // ③ Graph PoC cleanup
    if(m_pcg_graph_exec) cudaGraphExecDestroy(m_pcg_graph_exec);
    if(m_pcg_graph)      cudaGraphDestroy(m_pcg_graph);
}

SizeT PCGSolver::solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());

    if(!d_scalars_alloced)
    {
        cudaMalloc(&d_rz,      sizeof(Float));
        cudaMalloc(&d_rz0,     sizeof(Float));
        cudaMalloc(&d_rz_new,  sizeof(Float));
        cudaMalloc(&d_dot_res, sizeof(Float));
        cudaMalloc(&d_alpha,   sizeof(Float));
        cudaMalloc(&d_beta,    sizeof(Float));
        cudaMalloc(&d_break,   sizeof(int));
        d_scalars_alloced = true;
    }

    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    return iter;
}


SizeT PCGSolver::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    // ③ Invalidate captured graph from PREVIOUS solve — the sparse matrix is
    // rebuilt each Newton iteration (different block_values pointer + new
    // h_unique_key_number triplet_count baked into the captured spmv kernel).
    // Re-capture per solve; instantiate cost (~100us) << per-iter launch savings
    // across ~25 PCG iters. This was the source of the ~7e-5 checksum drift.
    if(m_pcg_graph_exec) { cudaGraphExecDestroy(m_pcg_graph_exec); m_pcg_graph_exec = nullptr; }
    m_graph_dof = 0;

    r.buffer_view().copy_from(b.buffer_view());

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    // Step E: cub-based fused dot for initial rz = dot(r, z).
    Cub_PCG_DotReduction(r.buffer_view().data(),
                         z.buffer_view().data(),
                         z.size(),
                         d_rz,
                         &cub_temp_ptr, &cub_temp_bytes);
    copy_scalar_kernel<<<1, 1>>>(d_rz0, d_rz);
    cudaMemsetAsync(d_break, 0, sizeof(int));

    p = z;

    const SizeT K = 8;

    // ③ CUDA Graph PoC: unlocked by adding capture-guards in muda's
    // wait_stream/wait_device. The iteration body is split into two halves
    // around the mid-iter h_break check (matching the original control flow
    // exactly — keeps physics bit-identical, unlike the single-graph version).
    //   first_half  : spmv, dot(p,Ap), fused axpy (x,r)         [sets d_break on breakdown]
    //   mid-iter check on d_break (every K iters)
    //   second_half : apply_precond, dot(r,z), fused axpy(p), post_iter_swap_and_check
    auto first_half = [&]() {
        spmv(p.cview(), Ap.view());
        Cub_PCG_DotReduction(p.buffer_view().data(),
                             Ap.buffer_view().data(),
                             z.size(),
                             d_dot_res,
                             &cub_temp_ptr, &cub_temp_bytes);
        LaunchCudaKernal_default(z.size(),
                                 256, 0, update_vector_dx_r_fused,
                                 x.buffer_view().data(),
                                 r.buffer_view().data(),
                                 (const double*)p.buffer_view().data(),
                                 (const double*)Ap.buffer_view().data(),
                                 (const double*)d_rz,
                                 (const double*)d_dot_res,
                                 d_break,
                                 (int)z.size());
    };
    auto second_half = [&]() {
        apply_preconditioner(z, r);
        Cub_PCG_DotReduction(r.buffer_view().data(),
                             z.buffer_view().data(),
                             z.size(),
                             d_rz_new,
                             &cub_temp_ptr, &cub_temp_bytes);
        LaunchCudaKernal_default(z.size(),
                                 256, 0, update_vector_c_fused,
                                 p.buffer_view().data(),
                                 (const double*)z.buffer_view().data(),
                                 (const double*)d_rz_new,
                                 (const double*)d_rz,
                                 (int)z.size());
        post_iter_swap_and_check<<<1, 1>>>(d_rz, d_rz_new, d_rz0,
                                           m_config.global_tol_rate, d_break);
    };

    // Helper: try to capture a sub-body as a graph (re-used for both halves).
    auto try_capture = [&](auto&& body, cudaGraphExec_t& exec_out) -> bool {
        if(cudaStreamBeginCapture(cudaStreamPerThread,
                                  cudaStreamCaptureModeThreadLocal) != cudaSuccess)
            return false;
        body();
        cudaGraph_t g = nullptr;
        if(cudaStreamEndCapture(cudaStreamPerThread, &g) != cudaSuccess)
        {
            if(g) cudaGraphDestroy(g);
            return false;
        }
        cudaGraphExec_t exec = nullptr;
        if(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0) != cudaSuccess)
        {
            cudaGraphDestroy(g);
            return false;
        }
        cudaGraphDestroy(g);  // graph descriptor not needed after instantiate
        exec_out = exec;
        return true;
    };

    // exec_half_b is per-solve (not cached across solves yet); first_half exec
    // (m_pcg_graph_exec) is a class member so it caches across solves with same dof.
    cudaGraphExec_t exec_half_b = nullptr;
    bool tried_capture = false;
    for(k = 1; k < max_iter; ++k)
    {
        // ─── Capture-once block ─── at iter 2, after iter 1 ran normally and
        // lazily allocated cub_temp. Capture is RECORD-ONLY (no execution); we
        // launch the captured graphs below for actual work.
        if(m_graph_enabled && k == 2 && !tried_capture && cub_temp_ptr != nullptr)
        {
            tried_capture = true;
            if(m_pcg_graph_exec) { cudaGraphExecDestroy(m_pcg_graph_exec); m_pcg_graph_exec = nullptr; }
            cudaGraphExec_t exec_a = nullptr;
            if(try_capture(first_half, exec_a))
            {
                m_pcg_graph_exec = exec_a;
                m_graph_dof      = z.size();
                cudaGraphExec_t exec_b = nullptr;
                if(try_capture(second_half, exec_b))
                    exec_half_b = exec_b;
                // If second_half capture failed, exec_half_b stays nullptr -> fallback for B.
            }
            // If first_half capture failed, m_pcg_graph_exec stays nullptr -> fallback for both.
        }

        // ─── First half ── spmv + dot(p,Ap) + axpy(x,r); may set d_break on breakdown.
        if(m_graph_enabled && m_pcg_graph_exec && m_graph_dof == z.size())
            cudaGraphLaunch(m_pcg_graph_exec, cudaStreamPerThread);
        else
            first_half();

        // ─── Mid-iter h_break check (SAME logical position as original) ───
        if(k % K == 0)
        {
            cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_break) break;
        }

        // ─── Second half ── precond + dot(r,z) + axpy(p) + swap_and_check.
        if(m_graph_enabled && exec_half_b)
            cudaGraphLaunch(exec_half_b, cudaStreamPerThread);
        else
            second_half();
    }

    // Cleanup local per-solve second-half exec.
    if(exec_half_b) cudaGraphExecDestroy(exec_half_b);

    // Final sync of break flag (in case loop exited on max_iter).
    cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
    return k;
}

}  // namespace gipc