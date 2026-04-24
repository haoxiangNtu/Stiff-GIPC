#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>



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


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum, cudaStream_t stream = 0)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize, stream>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize, stream>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpyAsync(&result, temp, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return result;
}

// Device-output variant: leaves the reduced scalar in *d_out (which can be
// `temp` itself or any device address). Skips the final cudaMemcpy/D2H.
void My_PCG_General_v_v_Reduction_DeviceOut(double* temp, double* A, double* B,
                                            int vertexNum, double* d_out,
                                            cudaStream_t stream = 0)
{
    int numbers = vertexNum;
    if(numbers < 1) {
        cudaMemsetAsync(d_out, 0, sizeof(double), stream);
        return;
    }
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize, stream>>>(temp, A, B, numbers);

    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize, stream>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }

    if(d_out != temp)
        cudaMemcpyAsync(d_out, temp, sizeof(double), cudaMemcpyDeviceToDevice, stream);
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
    if(pcg_stream)
        cudaStreamDestroy(pcg_stream);
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
    // Step C STATUS: infeasible without engine-wide stream refactor.
    // Stream plumbing is correct (passing pcg_stream=0 works fine), but
    // running PCG kernels on a non-default stream causes
    // cudaErrorIllegalAddress. Root cause: other engine subsystems
    // (ABDPreconditioner, MAS internal state, etc.) implicitly depend on
    // default-stream ordering for state visible during PCG iteration,
    // and are not stream-aware. Making PCG run on a custom stream
    // requires plumbing stream through ABD body H/grad assembly,
    // ABDPreconditioner setup, MAS preconditioner setup chain, and any
    // other subsystem whose output is read by PCG. That's weeks of
    // engine-wide refactoring beyond reasonable audit scope.
    // Keeping pcg_stream=0 (default stream) — equivalent in behavior to
    // RC1 with the addition of stream-parameter plumbing in solver code
    // (no functional change vs RC1 baseline).
    pcg_stream = nullptr;

    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    return iter;
}


SizeT PCGSolver::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    apply_preconditioner(z, r, pcg_stream);

    // Initial rz = dot(r, z), reduced into d_rz on device.
    My_PCG_General_v_v_Reduction_DeviceOut(p.buffer_view().data(),
                                           r.buffer_view().data(),
                                           z.buffer_view().data(),
                                           z.size(),
                                           d_rz, pcg_stream);
    copy_scalar_kernel<<<1, 1, 0, pcg_stream>>>(d_rz0, d_rz);
    cudaMemsetAsync(d_break, 0, sizeof(int), pcg_stream);

    p = z;

    const SizeT K = 8;

    // Step C inner-iter lambda that records all PCG kernel launches on
    // pcg_stream. Used twice: once for live execution, once for graph
    // capture. Captured graph replays K iters in a single launch.
    auto run_one_pcg_iter = [&]() {
        spmv(p.cview(), Ap.view(), pcg_stream);

        My_PCG_General_v_v_Reduction_DeviceOut(z.buffer_view().data(),
                                               p.buffer_view().data(),
                                               Ap.buffer_view().data(),
                                               z.size(),
                                               d_dot_res, pcg_stream);
        compute_alpha_kernel<<<1, 1, 0, pcg_stream>>>(d_rz, d_dot_res, d_alpha, d_break);

        LaunchCudaKernal_default_stream(z.size(), 256, 0, pcg_stream,
                                 update_vector_dx_r_dev,
                                 x.buffer_view().data(),
                                 r.buffer_view().data(),
                                 (const double*)p.buffer_view().data(),
                                 (const double*)Ap.buffer_view().data(),
                                 (const double*)d_alpha,
                                 (int)z.size());

        check_convergence_kernel<<<1, 1, 0, pcg_stream>>>(d_rz, d_rz0, m_config.global_tol_rate, d_break);

        apply_preconditioner(z, r, pcg_stream);

        My_PCG_General_v_v_Reduction_DeviceOut(Ap.buffer_view().data(),
                                               r.buffer_view().data(),
                                               z.buffer_view().data(),
                                               z.size(),
                                               d_rz_new, pcg_stream);

        compute_beta_and_swap_kernel<<<1, 1, 0, pcg_stream>>>(d_rz, d_rz_new, d_beta);

        LaunchCudaKernal_default_stream(z.size(), 256, 0, pcg_stream,
                                 update_vector_c_dev,
                                 p.buffer_view().data(),
                                 (const double*)z.buffer_view().data(),
                                 (const double*)d_beta,
                                 (int)z.size());
    };

    // Run first K iters non-captured (warmup any lazy allocations).
    for(k = 1; k < max_iter && k <= K; ++k)
    {
        run_one_pcg_iter();
        if(k % K == 0)
        {
            cudaMemcpyAsync(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost, pcg_stream);
            cudaStreamSynchronize(pcg_stream);
            if(h_break) break;
        }
    }

    // After warmup, run remaining iters non-captured (graph-capture disabled
    // for soundness debugging).
    for(; k < max_iter && !h_break; ++k)
    {
        run_one_pcg_iter();
        if(k % K == 0)
        {
            cudaMemcpyAsync(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost, pcg_stream);
            cudaStreamSynchronize(pcg_stream);
            if(h_break) break;
        }
    }

    cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(pcg_stream);
    return k;
}

}  // namespace gipc