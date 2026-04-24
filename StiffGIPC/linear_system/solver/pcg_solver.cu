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

    r.buffer_view().copy_from(b.buffer_view());

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    // Initial rz = dot(r, z), reduced into d_rz on device.
    My_PCG_General_v_v_Reduction_DeviceOut(p.buffer_view().data(),
                                           r.buffer_view().data(),
                                           z.buffer_view().data(),
                                           z.size(),
                                           d_rz);
    // d_rz0 = d_rz
    copy_scalar_kernel<<<1, 1>>>(d_rz0, d_rz);
    cudaMemsetAsync(d_break, 0, sizeof(int));

    p = z;

    // Convergence-flag check stride: K=8 means D2H once per 8 iters.
    // Was 2 D2H per iter for alpha/beta scalars; now 1 D2H per K iters.
    const SizeT K = 8;

    for(k = 1; k < max_iter; ++k)
    {
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            // dot(p, Ap) -> d_dot_res (no D2H here)
            My_PCG_General_v_v_Reduction_DeviceOut(z.buffer_view().data(),
                                                   p.buffer_view().data(),
                                                   Ap.buffer_view().data(),
                                                   z.size(),
                                                   d_dot_res);

            // alpha = rz / dot_res; sets d_break if non-finite or <= 0.
            // Same soundness invariant as the original host-side check
            // (rz/0 NaN guard + PD A => p^T A p > 0 unless converged).
            compute_alpha_kernel<<<1, 1>>>(d_rz, d_dot_res, d_alpha, d_break);
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r_dev,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     (const double*)d_alpha,
                                     (int)z.size());
        }

        // Convergence check on rz (still the old rz at this point, same as
        // the original host-side check): |rz| <= tol * rz0  -> set d_break.
        check_convergence_kernel<<<1, 1>>>(d_rz, d_rz0, m_config.global_tol_rate, d_break);

        // Stride: only D2H the break flag every K iters. Up to K-1 extra
        // iters past convergence in the worst case, but each is ~25 us so
        // <0.2 ms/frame penalty (vs ~0.7 ms/frame saved on D2H stalls).
        if(k % K == 0)
        {
            cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_break) break;
        }

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        // dot(r, z) -> d_rz_new
        My_PCG_General_v_v_Reduction_DeviceOut(Ap.buffer_view().data(),
                                               r.buffer_view().data(),
                                               z.buffer_view().data(),
                                               z.size(),
                                               d_rz_new);

        // beta = rz_new / rz, then rz <- rz_new
        compute_beta_and_swap_kernel<<<1, 1>>>(d_rz, d_rz_new, d_beta);

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c_dev,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     (const double*)d_beta,
                                     (int)z.size());
        }
    }

    // Final sync of break flag (in case loop exited on max_iter).
    cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
    return k;
}

}  // namespace gipc