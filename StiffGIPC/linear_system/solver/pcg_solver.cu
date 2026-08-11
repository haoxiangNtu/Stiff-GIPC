#include <linear_system/solver/pcg_solver.h>
#include <linear_system/linear_system/global_linear_system.h>   // [P3] system_ptr()->m_s4_*
#include <linear_system/utils/binned_reduce.cuh>                // [P3] exact per-env dot
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <vector>
#include <cstdlib>

// [decouple probe] global (non-namespaced) frame/Newton-iter counters defined in GIPC.cu — used to
// gate the STIFF_H_DUMP rz0 dump to a specific frame/iteration.
extern int g_dec_frame;
extern int g_dec_k;

// ===================== [multi-env P3] segmented (block-diagonal) PCG kernels =====================
// Per-env dot via binned deposit (exact, order-independent ⇒ per-env deterministic AND, for
// identical envs, cross-env symmetric). DOF i → block i/3 → group d2g[i/3].
// [P3 perf] block-level shared-memory binning, then one flush per block per bin. Cuts the global
// atomic contention from ~(DOFs/env) to ~(#blocks) per bin (~64x here) and stays EXACT: each block's
// bin slice is a multiple of that bin's ULP, and there are <2^30 blocks ⇒ the flush atomicAdd never
// rounds ⇒ order-independent. Shared = ng*BINNED_K doubles (8KB at ng=256).
// [multienv-mode] binned (order-free) per-env dot = strict-only determinism. merged/isolated use the
// fast plain shared-atomic into bin 0 (correct, non-deterministic order). Combine sums K bins → works
// for both (non-binned leaves bins 1..K-1 = 0). Set once from STIFF_FAST_GRAD (same gate as g_binned_on).
__device__ int g_seg_binned = 1;
static void set_seg_binned(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_seg_binned, &v, sizeof(int))); }
// [warp-reduce] per-warp fixed-tree pre-sum before the shared-bin deposit: one deposit per WARP
// instead of one per thread (32x fewer same-address shared atomics — ncu shows this kernel at
// healthy 62-79% occupancy yet 18.5% of the strict frame: pure atomic-replay serialization).
// Determinism: the lane→DOF mapping is fixed by the launch config, the shuffle tree order is fixed,
// and env k's absolute DOF range doesn't depend on N ⇒ run-to-run bit-identity AND batch-invariance
// hold. (The pre-sum rounds before binning, so values differ from the per-lane version — a wheel
// version change, same as any kernel edit. Identical envs may now differ from EACH OTHER in final
// ulps because their DOF ranges sit at different warp phases; each env stays deterministic.)
// Warps straddling an env boundary (~1 per 1000 with ~32k DOFs/env) fall back to per-lane deposits.
__device__ int g_seg_warp = 1;
static void set_seg_warp(int v){ CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_seg_warp, &v, sizeof(int))); }
__global__ void _seg_dot_deposit(const double* a, const double* b, const int* d2g,
                                 double* segbin, int ng, int n)
{
    extern __shared__ double sbin[];
    for(int j = threadIdx.x; j < ng * BINNED_K; j += blockDim.x) sbin[j] = 0.0;
    __syncthreads();
    int    i = blockIdx.x * blockDim.x + threadIdx.x;
    int    g = -1;
    double v = 0.0;
    if(i < n)
    {
        int gg = d2g[i / 3];
        if(gg >= 0 && gg < ng) { g = gg; v = a[i] * b[i]; }
    }
    if(g_seg_warp)
    {
        const unsigned full = 0xffffffffu;   // no early returns above ⇒ all 32 lanes present
        unsigned       has  = __ballot_sync(full, g >= 0);
        if(has)
        {
            int lg = __shfl_sync(full, g, __ffs(has) - 1);
            if(__all_sync(full, g < 0 || g == lg))
            {   // whole warp in one env segment (dominant case): tree-sum, lane 0 deposits
                double s = v;
#pragma unroll
                for(int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(full, s, o);
                if((threadIdx.x & 31) == 0)
                {
                    if(g_seg_binned) binned_deposit(sbin + (size_t)lg * BINNED_K, s);
                    else atomicAdd(&sbin[(size_t)lg * BINNED_K], s);
                }
            }
            else if(g >= 0)
            {   // env-boundary warp: per-lane (original path)
                if(g_seg_binned) binned_deposit(sbin + (size_t)g * BINNED_K, v);
                else atomicAdd(&sbin[(size_t)g * BINNED_K], v);
            }
        }
    }
    else if(g >= 0)
    {
        if(g_seg_binned) binned_deposit(sbin + (size_t)g * BINNED_K, v);
        else atomicAdd(&sbin[(size_t)g * BINNED_K], v);   // fast: plain shared-atomic → bin 0
    }
    __syncthreads();
    for(int j = threadIdx.x; j < ng * BINNED_K; j += blockDim.x)
    {
        double vb = sbin[j];
        if(vb != 0.0) atomicAdd(&segbin[j], vb);   // exact: block slices share the bin's exponent
    }
}
__global__ void _seg_dot_combine(double* out_g, double* segbin, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double* b = segbin + (size_t)g * BINNED_K;
    out_g[g]  = binned_combine(b);
#pragma unroll
    for(int kk = 0; kk < BINNED_K; ++kk) b[kk] = 0.0;   // re-zero for the next binned dot
}
// [opt B] non-binned SINGLE-kernel seg dot: block-local plain sum → direct atomicAdd to out_g[g].
// Skips the d_segbin + separate combine kernel (2 launches → 1) for merged/isolated (not bit-identical).
// out_g must be pre-zeroed. shared = ng doubles.
__global__ void _seg_dot_fused(const double* a, const double* b, const int* d2g,
                               double* out_g, int ng, int n)
{
    extern __shared__ double ssum[];
    for(int j = threadIdx.x; j < ng; j += blockDim.x) ssum[j] = 0.0;
    __syncthreads();
    int    i = blockIdx.x * blockDim.x + threadIdx.x;
    int    g = -1;
    double v = 0.0;
    if(i < n)
    {
        int gg = d2g[i / 3];
        if(gg >= 0 && gg < ng) { g = gg; v = a[i] * b[i]; }
    }
    if(g_seg_warp)
    {   // [warp-reduce] same pre-sum as _seg_dot_deposit (no bit-identity claim on this path)
        const unsigned full = 0xffffffffu;
        unsigned       has  = __ballot_sync(full, g >= 0);
        if(has)
        {
            int lg = __shfl_sync(full, g, __ffs(has) - 1);
            if(__all_sync(full, g < 0 || g == lg))
            {
                double s = v;
#pragma unroll
                for(int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(full, s, o);
                if((threadIdx.x & 31) == 0) atomicAdd(&ssum[lg], s);
            }
            else if(g >= 0) atomicAdd(&ssum[g], v);
        }
    }
    else if(g >= 0) atomicAdd(&ssum[g], v);
    __syncthreads();
    for(int j = threadIdx.x; j < ng; j += blockDim.x)
        if(ssum[j] != 0.0) atomicAdd(&out_g[j], ssum[j]);
}
// x += α_g c; r -= α_g Ap; α_g = rz_g/dot_g. Converged envs (brk_g) and bad dot are frozen.
__global__ void _seg_axpy_xr(double* dx, double* r, const double* c, const double* q,
                             const double* rz_g, const double* dot_g, const int* brk_g,
                             const int* d2g, int ng, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = d2g[i / 3];
    if(g < 0 || g >= ng || brk_g[g]) return;
    double dot = dot_g[g];
    if(!isfinite(dot) || dot <= 0.0) return;
    double a = rz_g[g] / dot;
    dx[i] += a * c[i];
    r[i]  -= a * q[i];
}
// p = z + β_g p; β_g = rzn_g/rz_g.
__global__ void _seg_axpy_p(double* c, const double* z, const double* rzn_g, const double* rz_g,
                            const int* brk_g, const int* d2g, int ng, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = d2g[i / 3];
    if(g < 0 || g >= ng || brk_g[g]) return;
    double rzo = rz_g[g];
    if(!isfinite(rzo) || rzo == 0.0) return;
    double bb = rzn_g[g] / rzo;
    c[i] = z[i] + bb * c[i];
}
// per-env swap (rz_g=rzn_g) + convergence (brk_g=1 when |rzn_g|<=tol*rz0_g).
__global__ void _seg_swap_check(double* rz_g, const double* rzn_g, const double* rz0_g,
                                double tol, int* brk_g, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double v  = rzn_g[g];
    rz_g[g]   = v;
    if(fabs(v) <= tol * rz0_g[g]) brk_g[g] = 1;
}
// [seg-fused dot] combine the spmv-fused per-env dot partials (ng x PSTRIDE, block-hashed) into
// out_g AND re-zero the slots for the next iteration (no separate memset). Launch: ng blocks x 256.
__global__ void _seg_partial_combine(double* out_g, double* partials, int ng)
{
    __shared__ double sh[256];
    int g = blockIdx.x;
    if(g >= ng) return;
    double v = partials[(size_t)g * 256 + threadIdx.x];
    partials[(size_t)g * 256 + threadIdx.x] = 0.0;   // re-zero for the next spmv
    sh[threadIdx.x] = v;
    __syncthreads();
    for(int s = 128; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) out_g[g] = sh[0];
}
// [micro-fusion (3)] partial-combine WITH the per-env convergence check folded in (replaces the
// separate _seg_swap_check launch; the rz swap itself becomes a host-side POINTER ping-pong).
// tol2_g == nullptr -> scalar tol. Bit-identical values to combine-then-check.
__global__ void _seg_partial_combine_ck(double* out_g, double* partials, const double* rz0_g,
                                        const double* tol2_g, double tol, int* brk_g, int ng)
{
    __shared__ double sh[256];
    int g = blockIdx.x;
    if(g >= ng) return;
    double v = partials[(size_t)g * 256 + threadIdx.x];
    partials[(size_t)g * 256 + threadIdx.x] = 0.0;
    sh[threadIdx.x] = v;
    __syncthreads();
    for(int st = 128; st > 0; st >>= 1)
    {
        if(threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        double rzn = sh[0];
        out_g[g]   = rzn;
        double t   = tol2_g ? tol2_g[g] : tol;
        if(fabs(rzn) <= t * rz0_g[g]) brk_g[g] = 1;
    }
}
// [micro-fusion (3)] binned-dot combine WITH the check folded in (strict path equivalent).
__global__ void _seg_dot_combine_ck(double* out_g, double* segbin, const double* rz0_g,
                                    const double* tol2_g, double tol, int* brk_g, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double* b   = segbin + (size_t)g * BINNED_K;
    double  rzn = binned_combine(b);
#pragma unroll
    for(int kk = 0; kk < BINNED_K; ++kk) b[kk] = 0.0;
    out_g[g] = rzn;
    double t = tol2_g ? tol2_g[g] : tol;
    if(fabs(rzn) <= t * rz0_g[g]) brk_g[g] = 1;
}
// [micro-fusion (3)] check-only variant (fast-path fallback when the precond-fused dot is not
// armed): brk from an already-combined rzn.
__global__ void _seg_check_only(const double* rzn_g, const double* rz0_g, const double* tol2_g,
                                double tol, int* brk_g, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double t = tol2_g ? tol2_g[g] : tol;
    if(fabs(rzn_g[g]) <= t * rz0_g[g]) brk_g[g] = 1;
}
// [warm-start (1)] r = b - Ap (Ap = A*x0 from the extra spmv).
__global__ void _warm_residual(double* r, const double* b, const double* Ap, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) r[i] = b[i] - Ap[i];
}
// [warm-start (1)] per-env decision: keep x0 iff its residual is no worse than the zero start
// (||r0_g||^2 <= ||b_g||^2). Uses ONLY env g's own data -> batch-invariant by construction.
__global__ void _warm_select(const double* rr_g, const double* bb_g, int* use_warm, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double rr = rr_g[g], bb = bb_g[g];
    use_warm[g] = (isfinite(rr) && rr <= bb) ? 1 : 0;
}
// [warm-start (1)] envs that rejected the warm start reset to the zero start (x=0, r=b).
__global__ void _warm_fixup(double* x, double* r, const double* b, const int* d2g,
                            const int* use_warm, int ng, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    int g = d2g[i / 3];
    if(g < 0 || g >= ng || use_warm[g]) return;
    x[i] = 0.0;
    r[i] = b[i];
}
// [E-W (2)] per-env Eisenstat-Walker forcing term (choice 2, alpha=2): the residual-REDUCTION
// target for THIS solve is eta_g = gamma * (||g_k||/||g_{k-1}||)^2 = gamma * rz0_k/rz0_{k-1}
// (rz0 = b.M^-1 b per env = squared-norm proxy), so the rz-space tolerance is tol2 = eta^2.
// Clamped to [tol_floor2 (the configured tol), tol_max2]. prev is then updated to rz0_k.
// Depends ONLY on env g's own history -> mate-independent -> batch-invariant.
__global__ void _ew_eta(const double* rz0_g, double* prev_g, double* tol2_g,
                        double gamma, double tol_max2, double tol_floor2, int ng)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if(g >= ng) return;
    double rz0 = rz0_g[g];
    double pr  = prev_g[g];
    double t2  = tol_max2;   // first solve (no history) -> loosest
    if(pr > 0.0 && rz0 > 0.0 && isfinite(pr) && isfinite(rz0))
    {
        double eta = gamma * (rz0 / pr);
        t2         = eta * eta;
    }
    if(!(t2 > tol_floor2)) t2 = tol_floor2;   // never looser than... (tighter floor; NaN-safe)
    if(t2 > tol_max2) t2 = tol_max2;
    tol2_g[g]  = t2;
    prev_g[g]  = rz0;
}



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

// A device-launchable graph executes K ordinary PCG iterations and ends in
// this kernel.  If another full K-batch fits and convergence has not been
// reached, enqueue the same executable graph as a tail launch.  Tail launches
// are serialized after the current graph, so PCG's iteration order is exactly
// the same as the host K-stride loop while the convergence decision stays on
// the GPU.  Device graph launch is available before conditional graph nodes
// (CUDA 12.0 vs 12.3), which keeps this path usable on the A800's R535 driver.
__global__ void pcg_graph_state_init(unsigned long long* state,
                                     unsigned long long  first_iteration)
{
    state[0] = first_iteration;
    state[1] = 0;
}

__global__ void pcg_graph_tail_relaunch(unsigned long long* state,
                                        const int*          d_break,
                                        unsigned long long  max_iter,
                                        unsigned long long  check_k)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const unsigned long long next = state[0] + check_k;
    state[0] = next;
    state[1] = static_cast<unsigned long long>(*d_break != 0);
    if(*d_break == 0 && next + check_k <= max_iter)
        cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

__global__ void pcg_seg_graph_tail_relaunch(unsigned long long* state,
                                            const int*          d_break_g,
                                            int                 ng,
                                            unsigned long long  max_iter,
                                            unsigned long long  check_k)
{
    if(threadIdx.x != 0 || blockIdx.x != 0)
        return;

    bool all_converged = true;
    for(int g = 0; g < ng; ++g)
        if(d_break_g[g] == 0)
        {
            all_converged = false;
            break;
        }

    const unsigned long long next = state[0] + check_k;
    state[0] = next;
    state[1] = static_cast<unsigned long long>(all_converged);
    if(!all_converged && next + check_k <= max_iter)
        cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
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

// [seg-trace-fk] frame-keyed arm (set by the Newton loop via GIPC.cu).
int g_seg_trace_arm = 0;

PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}

PCGSolver::~PCGSolver()
{
    // Executables retain kernel arguments pointing at the scalar/segmented
    // buffers, so release them before the referenced allocations.
    if(m_device_loop_exec)
        cudaGraphExecDestroy(m_device_loop_exec);
    if(m_seg_device_loop_exec)
        cudaGraphExecDestroy(m_seg_device_loop_exec);
    if(d_scalars_alloced)
    {
        cudaFree(d_rz);
        cudaFree(d_rz0);
        cudaFree(d_rz_new);
        cudaFree(d_dot_res);
        cudaFree(d_alpha);
        cudaFree(d_beta);
        cudaFree(d_break);
        cudaFree(d_graph_state);
    }
    if(cub_temp_ptr) cudaFree(cub_temp_ptr);
    if(m_seg_alloced)
    { cudaFree(d_rz_g); cudaFree(d_rz0_g); cudaFree(d_rzn_g); cudaFree(d_dot_g);
      cudaFree(d_segbin); cudaFree(d_break_g); cudaFree(d_dot_partials);
      cudaFree(d_rr_g); cudaFree(d_bb_g); cudaFree(d_use_warm);
      cudaFree(d_ew_prev); cudaFree(d_tol2_g); }
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
        cudaMalloc(&d_graph_state, 2 * sizeof(unsigned long long));
        d_scalars_alloced = true;
    }

    // [multi-env P3] segmented block-diagonal PCG when enabled + env map present. Each env is a
    // mathematically independent solve (per-env α/β/convergence) → cross-env bit-identical for
    // identical envs. ng is the ACTIVE group count, not the fixed slot capacity. ng=1 still uses
    // segmented kernels so changing batch count never silently changes the numerical algorithm.
    // Falls back to scalar PCG only when segmentation is disabled or no groups were declared.
    const int* d2g = nullptr;
    int        ng  = 0;
    if(getenv("STIFF_SEGMENTED_PCG") && system_ptr())
    {
        d2g = system_ptr()->m_s4_dof_to_group;
        ng  = system_ptr()->m_s4_ng;
    }
    auto iter = (d2g && ng > 0)
                    ? seg_pcg(x, b, m_config.max_iter_ratio * b.size(), d2g, ng)
                    : pcg(x, b, m_config.max_iter_ratio * b.size());

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

    // Step E: cub-based fused dot for initial rz = dot(r, z).
    Cub_PCG_DotReduction(r.buffer_view().data(),
                         z.buffer_view().data(),
                         z.size(),
                         d_rz,
                         &cub_temp_ptr, &cub_temp_bytes);
    copy_scalar_kernel<<<1, 1>>>(d_rz0, d_rz);
    cudaMemsetAsync(d_break, 0, sizeof(int));

    p = z;

    // [perf] convergence-check period: each check is a BLOCKING D2H (host loop needs the flag to
    // break) → the dominant host-bound cost (nsys: cudaMemcpy 68% of API). Larger K = fewer syncs at
    // the cost of ≤K extra iters after convergence. Fixed value → deterministic iter count (strict OK).
    const SizeT K = getenv("STIFF_PCG_CHECK_K") ? (SizeT)atoi(getenv("STIFF_PCG_CHECK_K")) : 8;
    const double pcg_tol = getenv("STIFF_PCG_TOL") ? atof(getenv("STIFF_PCG_TOL"))
                                                   : m_config.global_tol_rate;

    // [pcg-graph] one FULL iteration of the (shape-constant-within-a-solve) inner loop. The break
    // check moved to ITERATION BOUNDARIES (same K cadence): the second half-iteration only touches
    // p/rz/brk — x and r are bit-identical to the old mid-iteration-check loop at every exit point.
    auto body = [&]()
    {
        // Ap = A * p
        spmv(p.cview(), Ap.view());
        // Step E: cub fused dot(p, Ap) -> d_dot_res (1 launch instead of 2-3).
        Cub_PCG_DotReduction(p.buffer_view().data(), Ap.buffer_view().data(), z.size(),
                             d_dot_res, &cub_temp_ptr, &cub_temp_bytes);
        // Step E: fused axpy re-deriving alpha = *d_rz / *d_dot_res per-thread.
        LaunchCudaKernal_default(z.size(), 256, 0, update_vector_dx_r_fused,
                                 x.buffer_view().data(), r.buffer_view().data(),
                                 (const double*)p.buffer_view().data(),
                                 (const double*)Ap.buffer_view().data(),
                                 (const double*)d_rz, (const double*)d_dot_res,
                                 d_break, (int)z.size());
        apply_preconditioner(z, r);
        // Step E: cub fused dot(r, z) -> d_rz_new.
        Cub_PCG_DotReduction(r.buffer_view().data(), z.buffer_view().data(), z.size(),
                             d_rz_new, &cub_temp_ptr, &cub_temp_bytes);
        // Step E: fused axpy on p re-deriving beta = *d_rz_new / *d_rz per-thread.
        LaunchCudaKernal_default(z.size(), 256, 0, update_vector_c_fused,
                                 p.buffer_view().data(), (const double*)z.buffer_view().data(),
                                 (const double*)d_rz_new, (const double*)d_rz, (int)z.size());
        // Step E: combined swap (d_rz = d_rz_new) + convergence check.
        post_iter_swap_and_check<<<1, 1>>>(d_rz, d_rz_new, d_rz0, pcg_tol, d_break);
    };

    // [pcg-graph] capture K iterations into a CUDA graph and replay between break checks — removes
    // ~8 launch gaps per iteration. PTDS build ⇒ cudaStreamPerThread capture records all default-
    // stream work (kernels are RECORDED, not executed). First K iterations run plain so lazy
    // allocations (cub temp) precede the capture (cudaMalloc is illegal during capture). Gated to
    // the fast path (STIFF_FAST_GRAD, i.e. merged/isolated) — strict keeps the plain loop.
    // STIFF_PCG_GRAPH=0 kills it; any capture failure falls back to the plain loop.
    static int s_graph_env = -1;
    if(s_graph_env < 0)
    { const char* e = getenv("STIFF_PCG_GRAPH"); s_graph_env = e ? atoi(e) : 1; }
    bool use_graph = s_graph_env && !getenv("STIFF_SPMV_DET")   // [det-gating] graph for ALL non-strict modes (was: required STIFF_FAST_GRAD)
                     // STIFF_KSUM intentionally synchronizes inside MAS diagnostics.
                     && !getenv("STIFF_KSUM")
                     && !getenv("STIFF_MAS_FUSE_VALIDATE")
                     && !getenv("STIFF_MAS_DUMP")
                     && system_ptr() && system_ptr()->precond_graph_capturable();

    // [persistent PCG graph] Capture one K-iteration batch as a device-launchable
    // graph.  Its final kernel evaluates d_break and tail-launches the same graph,
    // eliminating all intermediate D2H convergence checks.  This is deliberately
    // attempted before the legacy host-replayed graph.  Any capture/instantiate
    // failure is non-fatal and falls through to that well-tested path.
    static int s_device_loop_env = -1;
    if(s_device_loop_env < 0)
    {
        const char* e = getenv("STIFF_PCG_DEVICE_LOOP");
        s_device_loop_env = e ? atoi(e) : 1;
    }
    static int s_driver_version = -1;
    if(s_driver_version < 0)
        cudaDriverGetVersion(&s_driver_version);
    const bool use_device_loop = use_graph && s_device_loop_env
                              && s_driver_version >= 12000
                              && K > 0 && 1 + K <= max_iter;

    if(use_device_loop)
    {
        cudaGraph_t     dg  = nullptr;
        bool            ready = false;
        bool            rebuilt = false;

        cudaError_t capture_status = cudaStreamBeginCapture(
            cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if(capture_status == cudaSuccess)
        {
            for(SizeT i = 0; i < K; ++i)
                body();
            pcg_graph_tail_relaunch<<<1, 1>>>(
                d_graph_state,
                d_break,
                static_cast<unsigned long long>(max_iter),
                static_cast<unsigned long long>(K));
            capture_status = cudaStreamEndCapture(cudaStreamPerThread, &dg);
            if(capture_status == cudaSuccess && dg)
            {
                if(m_device_loop_exec)
                {
                    cudaGraphExecUpdateResultInfo update_info{};
                    capture_status = cudaGraphExecUpdate(
                        m_device_loop_exec, dg, &update_info);
                    ready = capture_status == cudaSuccess
                         && update_info.result == cudaGraphExecUpdateSuccess;
                    if(!ready)
                    {
                        cudaGraphExecDestroy(m_device_loop_exec);
                        m_device_loop_exec = nullptr;
                    }
                }
                if(!m_device_loop_exec)
                {
                    capture_status = cudaGraphInstantiateWithFlags(
                        &m_device_loop_exec,
                        dg,
                        cudaGraphInstantiateFlagDeviceLaunch);
                    ready   = capture_status == cudaSuccess && m_device_loop_exec;
                    rebuilt = ready;
                }
            }
        }
        else
        {
            // A failed BeginCapture does not put the stream into capture mode.
            cudaGetLastError();
        }

        if(ready)
        {
            // Device-graph updates must be uploaded again before device launch.
            // Upload is stream ordered, so no host synchronization is introduced.
            CUDA_SAFE_CALL(cudaGraphUpload(m_device_loop_exec, cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaGraphDestroy(dg));
            dg = nullptr;
            pcg_graph_state_init<<<1, 1>>>(d_graph_state, 1ULL);
            CUDA_SAFE_CALL(cudaGraphLaunch(m_device_loop_exec, cudaStreamPerThread));

            unsigned long long graph_state[2] = {1ULL, 0ULL};
            CUDA_SAFE_CALL(cudaMemcpy(graph_state,
                                      d_graph_state,
                                      sizeof(graph_state),
                                      cudaMemcpyDeviceToHost));
            k       = static_cast<SizeT>(graph_state[0]);
            h_break = static_cast<int>(graph_state[1]);

            static bool once = false;
            if(!once)
            {
                once = true;
                printf("[pcg-device-loop] self-tail graph active (K=%d, driver=%d)\n",
                       (int)K,
                       s_driver_version);
            }
            if(rebuilt && getenv("STIFF_PCG_GRAPH_DIAG"))
                printf("[pcg-device-loop] executable rebuilt\n");

            // The device graph stops before a partial final batch.  Preserve the
            // legacy max-iteration semantics with at most K-1 plain tail steps.
            if(!h_break)
                for(; k < max_iter; ++k)
                    body();
            return k;
        }

        if(dg)  cudaGraphDestroy(dg);
        cudaGetLastError();
        static bool oncef = false;
        if(!oncef)
        {
            oncef = true;
            printf("[pcg-device-loop] unavailable (%s) -> host graph fallback\n",
                   cudaGetErrorString(capture_status));
        }
    }

    cudaGraph_t     pg  = nullptr;
    cudaGraphExec_t pge = nullptr;
    bool            captured = false;
    k = 1;
    while(k < max_iter)
    {
        if(captured && k + K <= (SizeT)max_iter)
        {
            cudaGraphLaunch(pge, cudaStreamPerThread);   // = K iterations
            k += K;
        }
        else
        {
            SizeT stop = (k + K < max_iter) ? k + K : max_iter;
            for(; k < stop; ++k) body();
            if(use_graph && !captured && k + K <= (SizeT)max_iter)
            {
                if(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)
                   == cudaSuccess)
                {
                    for(SizeT i = 0; i < K; ++i) body();   // recorded, not executed
                    if(cudaStreamEndCapture(cudaStreamPerThread, &pg) == cudaSuccess && pg
                       && cudaGraphInstantiate(&pge, pg, nullptr, nullptr, 0) == cudaSuccess)
                    {
                        captured = true;
                        static bool once = false;
                        if(!once) { once = true; printf("[pcg-graph] merged loop captured (K=%d)\n", (int)K); }
                    }
                }
                if(!captured)
                {
                    use_graph = false;   // permanent fallback for this solve
                    if(pg) { cudaGraphDestroy(pg); pg = nullptr; }
                    cudaGetLastError();  // clear sticky capture error
                    static bool oncef = false;
                    if(oncef == false) { oncef = true; printf("[pcg-graph] merged capture FAILED -> plain loop\n"); }
                }
            }
        }
        cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
        if(h_break) break;
    }
    if(pge) cudaGraphExecDestroy(pge);
    if(pg) cudaGraphDestroy(pg);

    // Final sync of break flag (in case loop exited on max_iter).
    cudaMemcpy(&h_break, d_break, sizeof(int), cudaMemcpyDeviceToHost);
    if(getenv("STIFF_SEG_DIAG"))
    { static int _c = 0; if(_c++ < 4) printf("[seg-diag] SCALAR solve#%d: k=%zu (max=%zu)\n",
                                             _c, (size_t)k, (size_t)max_iter); }
    return k;
}

// [multi-env P3] segmented binned dot: out_g[g] = Σ_{i:d2g[i/3]==g} a[i]*b[i] (exact, per-env).
static int s_seg_binned_host = 1;   // [opt B] host mirror of g_seg_binned (set in seg_pcg)
void PCGSolver::seg_dot(const Float* a, const Float* b, const int* d2g, int ng, int n, Float* out_g)
{
    int bs = 256;
    if(!s_seg_binned_host)
    {   // [opt B] fast fused path (merged/isolated): 1 kernel instead of deposit+combine
        cudaMemsetAsync(out_g, 0, (size_t)ng * sizeof(double));
        _seg_dot_fused<<<(n + bs - 1) / bs, bs, (size_t)ng * sizeof(double)>>>(a, b, d2g, out_g, ng, n);
        return;
    }
    // [strict-perf] no per-call memset: _seg_dot_combine re-zeroes each bin after reading
    // (d_segbin zeroed once at alloc). Bit-identical by construction.
    size_t shmem = (size_t)ng * BINNED_K * sizeof(double);   // block-local bins (8KB at ng=256)
    _seg_dot_deposit<<<(n + bs - 1) / bs, bs, shmem>>>(a, b, d2g, d_segbin, ng, n);
    _seg_dot_combine<<<(ng + bs - 1) / bs, bs>>>(out_g, d_segbin, ng);
}

// [multi-env P3] block-diagonal PCG: per-env α/β/convergence. Each env is an INDEPENDENT solve
// (matrix block-diagonal after P1, preconditioner intra-env), so the per-env binned dots are the
// only change vs the scalar PCG. For identical envs every env's iteration is bit-identical ⇒
// cross-env bit-identical. Frozen (converged) envs are skipped in the axpy.
SizeT PCGSolver::seg_pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b,
                         SizeT max_iter, const int* d2g, int ng)
{
    int n = (int)b.size();
    static bool s_seg_binned_set = false;   // [multienv-mode] binned seg-dot only for strict (bit-identity)
    if(!s_seg_binned_set) {
        // [det-gating] binned seg-dot = strict-only (positive gate); STIFF_SEG_BINNED=1 forces.
        int on = (getenv("STIFF_SPMV_DET") || getenv("STIFF_SEG_BINNED")) ? 1 : 0;
        set_seg_binned(on); s_seg_binned_host = on; s_seg_binned_set = true;   // device + host mirror
        // [warp-reduce] DETERMINISM FIX (was: default ON for all modes — WRONG). The per-warp float
        // pre-sum is NOT binned-exact, AND the warp grouping depends on the GLOBAL DOF layout (env k
        // starts at offset Σ_{j<k} N_j, which is not 32-aligned), so env0 and env1 sum DIFFERENT lane
        // groups → ~ULP cross-env/batch divergence that chaotic contact amplifies to macroscopic.
        // It preserves run-to-run (layout fixed within a run) but breaks cross-env AND batch. So:
        // strict (binned, requires bit-identity) MUST take the plain per-lane binned deposit;
        // merged/isolated keep the 7.9× warp pre-sum (they don't require bit-identity). Override:
        // STIFF_SEG_WARP=1 forces it on even for strict (A/B only, breaks determinism).
        const char* w = getenv("STIFF_SEG_WARP");
        set_seg_warp(w ? atoi(w) : (on ? 0 : 1));
        // [warp-reduce diag] STIFF_SEG_WARP=2: audit d2g at 32-DOF (warp) granularity. Mixed warps
        // silently degrade the pre-sum to the per-lane fallback — this measures how often.
        if(w && atoi(w) == 2)
        {
            int nb = (int)b.size() / 3;
            std::vector<int> hd(nb);
            cudaMemcpy(hd.data(), d2g, (size_t)nb * sizeof(int), cudaMemcpyDeviceToHost);
            long nn = (long)b.size(), uni = 0, mix = 0;
            for(long w0 = 0; w0 < nn; w0 += 32)
            {
                int lg = -2; bool u = true;
                for(long i = w0; i < w0 + 32 && i < nn; i++)
                {
                    int gg = hd[i / 3];
                    if(gg < 0) continue;
                    if(lg == -2) lg = gg;
                    else if(gg != lg) { u = false; break; }
                }
                (u ? uni : mix)++;
            }
            long tot = uni + mix; if(tot < 1) tot = 1;
            printf("[seg-warp] d2g audit: n=%ld nb=%d warps uni=%ld mix=%ld (%.1f%% mixed)\n",
                   nn, nb, uni, mix, 100.0 * mix / tot);
            printf("[seg-warp] d2g[0..47]:");
            for(int j = 0; j < 48 && j < nb; j++) printf(" %d", hd[j]);
            printf("\n[seg-warp] d2g[mid..mid+23]:");
            for(int j = nb / 2; j < nb / 2 + 24 && j < nb; j++) printf(" %d", hd[j]);
            printf("\n");
            fflush(stdout);
        }
    }
    if(!m_seg_alloced || m_seg_ng < ng)
    {
        if(m_seg_alloced)
        {
          if(m_seg_device_loop_exec)
          { cudaGraphExecDestroy(m_seg_device_loop_exec); m_seg_device_loop_exec = nullptr; }
          cudaFree(d_rz_g); cudaFree(d_rz0_g); cudaFree(d_rzn_g); cudaFree(d_dot_g);
          cudaFree(d_segbin); cudaFree(d_break_g); cudaFree(d_dot_partials);
          cudaFree(d_rr_g); cudaFree(d_bb_g); cudaFree(d_use_warm);
          cudaFree(d_ew_prev); cudaFree(d_tol2_g); }
        cudaMalloc(&d_rz_g,   ng * sizeof(Float));
        cudaMalloc(&d_rz0_g,  ng * sizeof(Float));
        cudaMalloc(&d_rzn_g,  ng * sizeof(Float));
        cudaMalloc(&d_dot_g,  ng * sizeof(Float));
        cudaMalloc(&d_segbin, (size_t)ng * BINNED_K * sizeof(double));
        cudaMemset(d_segbin, 0, (size_t)ng * BINNED_K * sizeof(double));   // combine re-zeroes after
        cudaMalloc(&d_break_g, ng * sizeof(int));
        // [seg-fused dot] spmv-fused per-env dot partials (block-hashed strided slots); zeroed once
        // here, then the combine kernel re-zeroes after each read.
        cudaMalloc(&d_dot_partials, (size_t)ng * 256 * sizeof(double));
        cudaMemset(d_dot_partials, 0, (size_t)ng * 256 * sizeof(double));
        // [warm-start (1)] + [E-W (2)] per-env state (all ng-sized)
        cudaMalloc(&d_rr_g,     ng * sizeof(Float));
        cudaMalloc(&d_bb_g,     ng * sizeof(Float));
        cudaMalloc(&d_use_warm, ng * sizeof(int));
        cudaMalloc(&d_ew_prev,  ng * sizeof(Float));
        cudaMemset(d_ew_prev, 0, ng * sizeof(Float));   // 0 = "no history" -> loosest eta first solve
        cudaMalloc(&d_tol2_g,   ng * sizeof(Float));
        m_seg_ng = ng;
        m_seg_alloced = true;
    }
    int bs = 256, gn = (n + bs - 1) / bs;

    SizeT k = 0;
    // [batch-invariance] x0 = 0 (matches scalar pcg() line ~455). seg_pcg sets r=b below, which
    // ASSUMES x0=0; without this fill x was the stale previous m_x (warm-start) → result was
    // x_stale + H⁻¹b (a latent correctness bug) AND batch-dependent (the stale m_x carries the
    // mates' influence from the prior solve). Zeroing makes the solve a pure function of (H,b).
    // [warm-start (1)] STIFF_PCG_WARM=1 (default off, non-binned/isolated path first): x0 = the
    // previous solve's solution (m_x persists in GlobalLinearSystem across Newton iters), with the
    // CORRECT residual r0 = b - A*x0 (the old removed warm start set r=b — that was the bug) and a
    // PER-ENV SAFEGUARD: env g keeps x0 only if ||r0_g||^2 <= ||b_g||^2, else it resets to the zero
    // start. Both the decision and the history are per-env-local -> batch-invariant by construction.
    // The convergence reference rz0_g stays the b-based one (b . M^-1 b), so the absolute target is
    // unchanged vs the zero-start solver.
    static int s_warm = -1;
    if(s_warm < 0) { const char* e = getenv("STIFF_PCG_WARM"); s_warm = e ? atoi(e) : 0; }
    const bool warm = s_warm && !s_seg_binned_host;
    static int s_ew = -1;
    if(s_ew < 0) { const char* e = getenv("STIFF_PCG_EW"); s_ew = e ? atoi(e) : 0; }
    const bool ew = s_ew && !s_seg_binned_host;

    if(warm)
    {
        // b-based convergence reference FIRST (z is scratch here): rz0_g = b . M^-1 b per env.
        apply_preconditioner(z, b);
        seg_dot(b.buffer_view().data(), z.buffer_view().data(), d2g, ng, n, d_rz0_g);
        // r0 = b - A*x0 (x = previous solution, NOT zeroed), then the per-env safeguard.
        spmv(x.as_const(), Ap.view());
        _warm_residual<<<gn, bs>>>(r.buffer_view().data(), b.buffer_view().data(),
                                   (const double*)Ap.buffer_view().data(), n);
        seg_dot(r.buffer_view().data(), r.buffer_view().data(), d2g, ng, n, d_rr_g);
        seg_dot(b.buffer_view().data(), b.buffer_view().data(), d2g, ng, n, d_bb_g);
        _warm_select<<<(ng + bs - 1) / bs, bs>>>(d_rr_g, d_bb_g, d_use_warm, ng);
        _warm_fixup<<<gn, bs>>>(x.buffer_view().data(), r.buffer_view().data(),
                                b.buffer_view().data(), d2g, d_use_warm, ng, n);
        apply_preconditioner(z, r);
        seg_dot(r.buffer_view().data(), z.buffer_view().data(), d2g, ng, n, d_rz_g);
    }
    else
    {
        // [batch-invariance] x0 = 0 (matches scalar pcg()). The OLD unconditional warm start was
        // removed because it set r=b (wrong residual: result = x_stale + H^-1 b) AND the stale
        // merged m_x carried the mates' influence. The gated warm path above fixes both.
        x.buffer_view().fill(0);
        r.buffer_view().copy_from(b.buffer_view());
        apply_preconditioner(z, r);
        seg_dot(r.buffer_view().data(), z.buffer_view().data(), d2g, ng, n, d_rz_g);
    }
    if(getenv("STIFF_SEG_DIAG"))   // [P3 debug] is the per-env dot a correct partition of the global?
    {
        static bool _once = false;
        if(!_once)
        {
            _once = true;
            std::vector<double> hrz(ng);
            cudaMemcpy(hrz.data(), d_rz_g, ng * sizeof(double), cudaMemcpyDeviceToHost);
            double sumg = 0; for(int g = 0; g < ng; g++) sumg += hrz[g];
            Cub_PCG_DotReduction(r.buffer_view().data(), z.buffer_view().data(), z.size(),
                                 d_rz, &cub_temp_ptr, &cub_temp_bytes);
            double fullrz = 0; cudaMemcpy(&fullrz, d_rz, sizeof(double), cudaMemcpyDeviceToHost);
            printf("[seg-diag] n=%d ng=%d  full_rz=%.10e  sum_rz_g=%.10e  diff=%.3e (nonzero ⇒ ungrouped DOFs!)\n",
                   n, ng, fullrz, sumg, fullrz - sumg);
            printf("[seg-diag] rz_g[0..7]:");
            for(int g = 0; g < ng && g < 8; g++) printf(" %.6e", hrz[g]);
            printf("  (identical envs ⇒ should be equal)\n");
        }
    }
    if(!warm)   // warm path already set the b-based rz0_g
        cudaMemcpyAsync(d_rz0_g, d_rz_g, ng * sizeof(Float), cudaMemcpyDeviceToDevice);
    cudaMemsetAsync(d_break_g, 0, ng * sizeof(int));
    p = z;
    // [E-W (2)] per-env Eisenstat-Walker forcing term for THIS solve. Semantics: the existing
    // check |rzn| <= T * rz0 operates on rz = ||r||^2_{M^-1} (a NORM-SQUARED), so a norm-reduction
    // target eta corresponds to T = eta^2. EW choice-2 with alpha=2 on norms gives
    // eta_g = gamma * (||g_k||/||g_{k-1}||)^2 = gamma * rz0_g / prev_g  (rz0 IS the norm^2 proxy).
    // T_g = eta_g^2 clamped to [configured tol (tightest), eta_max^2 (loosest)]; prev <- rz0.
    // Per-env-local history -> batch-invariant. Constant within a solve -> graph-safe (baked ptr).
    double ew_gamma = 0.9, ew_max2 = 0.25;   // eta_max=0.5 -> T_max=0.25
    if(ew)
    {
        static double s_g = -1.0, s_m2 = -1.0;
        if(s_g < 0)
        {
            const char* g1 = getenv("STIFF_PCG_EW_GAMMA");
            const char* g2 = getenv("STIFF_PCG_EW_ETAMAX");
            s_g            = g1 ? atof(g1) : 0.9;
            double em      = g2 ? atof(g2) : 0.5;
            s_m2           = em * em;
        }
        ew_gamma = s_g;
        ew_max2  = s_m2;
    }

    // [decouple probe] rz0 = b·M⁻¹b per env. b (gradient) is bit-exact across batches; if rz0_g[0]
    // differs A-vs-B, H's DIAGONAL (preconditioner) differs ⇒ Hessian H_0 is batch-dependent.
    if(getenv("STIFF_H_DUMP"))
    {
        int wf = getenv("STIFF_DUMP_FRAME") ? atoi(getenv("STIFF_DUMP_FRAME")) : -1;
        int wk = getenv("STIFF_PROBE_K") ? atoi(getenv("STIFF_PROBE_K")) : 0;
        if(::g_dec_frame == wf && ::g_dec_k == wk)
        {
            std::vector<double> h0(ng);
            cudaDeviceSynchronize();
            cudaMemcpy(h0.data(), d_rz0_g, ng * sizeof(Float), cudaMemcpyDeviceToHost);
            printf("[h-dump] frame=%d k=%d rz0_g[0..3]= %.17e %.17e %.17e %.17e\n",
                   g_dec_frame, g_dec_k, h0[0], h0[1], h0[2], h0[3]);
            // [localize] dump z = M⁻¹b (full vector) + d2g. z differs across batches ONLY where the
            // H DIAGONAL differs (b is bit-exact) → the differing DOF indices localize the seed
            // (FEM-cloth dofs vs ABD-body dofs are contiguous blocks at known offsets).
            if(getenv("STIFF_Z_DUMP"))
            {
                int nb = n / 3;   // d2g is per vertex/body BLOCK (indexed by dof/3)
                std::vector<double> hz(n), hb(n); std::vector<int> hd(nb);
                cudaMemcpy(hz.data(), z.buffer_view().data(), (size_t)n * sizeof(Float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hb.data(), r.buffer_view().data(), (size_t)n * sizeof(Float), cudaMemcpyDeviceToHost);  // r == b here
                cudaMemcpy(hd.data(), d2g, (size_t)nb * sizeof(int), cudaMemcpyDeviceToHost);
                const char* zf = getenv("STIFF_Z_DUMP");
                FILE* f = fopen(zf, "wb"); if(f){ fwrite(hz.data(), sizeof(double), n, f); fclose(f); }
                FILE* fb = fopen((std::string(zf) + ".b").c_str(), "wb"); if(fb){ fwrite(hb.data(), sizeof(double), n, fb); fclose(fb); }
                FILE* d = fopen((std::string(zf) + ".d2g").c_str(), "wb"); if(d){ fwrite(hd.data(), sizeof(int), nb, d); fclose(d); }
                printf("[z-dump] wrote %d dofs (z + .b RHS) -> %s\n", n, zf);
            }
        }
    }

    // [perf] convergence-check period: each check is a BLOCKING D2H (host loop needs the flag to
    // break) → the dominant host-bound cost (nsys: cudaMemcpy 68% of API). Larger K = fewer syncs at
    // the cost of ≤K extra iters after convergence. Fixed value → deterministic iter count (strict OK).
    const SizeT K = getenv("STIFF_PCG_CHECK_K") ? (SizeT)atoi(getenv("STIFF_PCG_CHECK_K")) : 8;
    std::vector<int> h_brk(ng);
    // [seg-trace] batch-invariance forensics: STIFF_SEG_TRACE=<solve#> arms ONE
    // solve (0-based, counted across the process) for per-PCG-iteration
    // per-env rz / p·Ap dumps at 17 digits. Run with STIFF_PCG_GRAPH=0 so the
    // plain loop executes (bit-identical arithmetic to the graph paths).
    // [seg-trace-fk] frame-keyed arm set by the Newton loop (exact targeting).
    extern int g_seg_trace_arm;
    static long long s_seg_solve_idx = -1;
    ++s_seg_solve_idx;
    static int s_seg_trace_target = -2;
    if(s_seg_trace_target == -2)
    { const char* e = getenv("STIFF_SEG_TRACE"); s_seg_trace_target = e ? atoi(e) : -1; }
    const bool seg_trace = (g_seg_trace_arm != 0)
                        || (s_seg_trace_target >= 0 && s_seg_solve_idx == (long long)s_seg_trace_target);
    double tol = getenv("STIFF_PCG_TOL") ? atof(getenv("STIFF_PCG_TOL")) : m_config.global_tol_rate;
    if(ew)   // [E-W (2)] per-env tolerance for this solve (floor = the configured tol)
        _ew_eta<<<(ng + bs - 1) / bs, bs>>>(d_rz0_g, d_ew_prev, d_tol2_g, ew_gamma, ew_max2, tol, ng);
    const double* tol2_arg = ew ? d_tol2_g : nullptr;

    // [seg-fused dot] fast path: the spmv itself accumulates the per-env p·Ap (strided partials),
    // the combine kernel folds them into d_dot_g (and re-zeroes) — replaces the standalone
    // seg_dot(p,Ap) full-vector pass + its memset. strict (binned) keeps the standalone exact dot.
    GlobalLinearSystem* fsys     = system_ptr();
    const bool          fuse_dot = (!s_seg_binned_host && fsys != nullptr);

    // [pcg-graph] one FULL seg iteration (break check at iteration boundaries — the second half
    // only touches p/rz/brk, so x and r are bit-identical at every exit point).
    // [micro-fusion (3)] the convergence check is FOLDED into the rz combine kernels and the rz
    // swap became a host-side POINTER PING-PONG (rz_cur/rz_next) — removes the swap kernel. Values
    // are bit-identical (same numbers, different storage). NOTE: freshly-converged envs now skip
    // their final p-update (axpy_p reads brk set in the same iteration) — p is never read again
    // for a broken env, so x/r remain bit-identical. K is even and each check boundary runs whole
    // K-blocks, so the ping-pong parity at capture/replay boundaries is stable.
    double* rz_cur  = d_rz_g;
    double* rz_next = d_rzn_g;
    auto body = [&]()
    {
        if(fuse_dot) fsys->set_seg_dot_accum(d_dot_partials, ng);   // baked into kernel args on capture
        spmv(p.cview(), Ap.view());
        if(fuse_dot)
            _seg_partial_combine<<<ng, 256>>>(d_dot_g, d_dot_partials, ng);
        else
            seg_dot(p.buffer_view().data(), Ap.buffer_view().data(), d2g, ng, n, d_dot_g);
        _seg_axpy_xr<<<gn, bs>>>(x.buffer_view().data(), r.buffer_view().data(),
                                 (const double*)p.buffer_view().data(),
                                 (const double*)Ap.buffer_view().data(),
                                 rz_cur, d_dot_g, d_break_g, d2g, ng, n);
        // [seg-fused dot] preconditioners accumulate per-env r·z alongside their z-write
        // (final-z exact via the ABD correction); falls back if any preconditioner isn't capable.
        bool rz_armed = fuse_dot && fsys->arm_precond_seg_dot(d_dot_partials, d2g, ng);
        apply_preconditioner(z, r);
        if(rz_armed)
            _seg_partial_combine_ck<<<ng, 256>>>(rz_next, d_dot_partials, d_rz0_g,
                                                 tol2_arg, tol, d_break_g, ng);
        else if(!s_seg_binned_host)
        {   // fast path without armed preconditioners (e.g. MAS): standalone fast dot + check
            seg_dot(r.buffer_view().data(), z.buffer_view().data(), d2g, ng, n, rz_next);
            _seg_check_only<<<(ng + bs - 1) / bs, bs>>>(rz_next, d_rz0_g, tol2_arg, tol, d_break_g, ng);
        }
        else
        {   // binned (strict) path: deposit + combine-with-check
            size_t shmem = (size_t)ng * BINNED_K * sizeof(double);
            _seg_dot_deposit<<<gn, bs, shmem>>>(r.buffer_view().data(), z.buffer_view().data(),
                                                d2g, d_segbin, ng, n);
            _seg_dot_combine_ck<<<(ng + bs - 1) / bs, bs>>>(rz_next, d_segbin, d_rz0_g,
                                                            tol2_arg, tol, d_break_g, ng);
        }
        _seg_axpy_p<<<gn, bs>>>(p.buffer_view().data(), (const double*)z.buffer_view().data(),
                                rz_next, rz_cur, d_break_g, d2g, ng, n);
        std::swap(rz_cur, rz_next);   // pointer ping-pong (replaces the swap kernel)
    };

    // [pcg-graph] capture K iterations, replay between break checks (see pcg() — same design).
    // Gated to the fast (non-binned) path — strict keeps the plain loop byte-for-byte.
    static int s_graph_env_seg = -1;
    if(s_graph_env_seg < 0)
    { const char* e = getenv("STIFF_PCG_GRAPH"); s_graph_env_seg = e ? atoi(e) : 1; }
    // [strict-perf] graphs are arithmetic-neutral (same kernels/order/args) — enable for the
    // binned (strict) path too; verified by the full determinism battery.
    bool use_graph = (s_graph_env_seg != 0)
                     && !getenv("STIFF_KSUM")
                     && !getenv("STIFF_MAS_FUSE_VALIDATE")
                     && !getenv("STIFF_MAS_DUMP")
                     && system_ptr() && system_ptr()->precond_graph_capturable();

    // The segmented solver uses two rz buffers with host-side pointer
    // ping-pong.  An even K returns to the same orientation at graph boundaries,
    // making the captured batch safely repeatable.  The final graph kernel
    // checks all per-environment break flags and self-tail-launches on device.
    static int s_device_loop_env_seg = -1;
    if(s_device_loop_env_seg < 0)
    {
        const char* e = getenv("STIFF_PCG_DEVICE_LOOP");
        s_device_loop_env_seg = e ? atoi(e) : 1;
    }
    static int s_driver_version_seg = -1;
    if(s_driver_version_seg < 0)
        cudaDriverGetVersion(&s_driver_version_seg);
    const bool use_device_loop = use_graph && s_device_loop_env_seg
                              && s_driver_version_seg >= 12000
                              && K > 0 && (K & 1) == 0
                              && 1 + K <= max_iter;

    if(use_device_loop)
    {
        // The deterministic SpMV owns a lazily-sized binned accumulator. Its
        // first call performs cudaMalloc/cudaMemset, which is illegal inside a
        // stream capture and used to invalidate the real strict+grouped graph
        // before the first PCG iteration. Prime only Ap here (no solver state
        // changes); the captured body overwrites Ap before consuming it. Work
        // is ordered on the same PTDS stream, so no host synchronization is
        // needed. merged/isolated use the non-binned SpMV and skip this cost.
        if(getenv("STIFF_SPMV_DET"))
            spmv(p.cview(), Ap.view());

        cudaGraph_t dg = nullptr;
        bool ready = false;
        bool rebuilt = false;
        cudaError_t capture_status = cudaStreamBeginCapture(
            cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if(capture_status == cudaSuccess)
        {
            for(SizeT i = 0; i < K; ++i)
                body();
            pcg_seg_graph_tail_relaunch<<<1, 1>>>(
                d_graph_state,
                d_break_g,
                ng,
                static_cast<unsigned long long>(max_iter),
                static_cast<unsigned long long>(K));
            capture_status = cudaStreamEndCapture(cudaStreamPerThread, &dg);
            if(capture_status == cudaSuccess && dg)
            {
                if(m_seg_device_loop_exec)
                {
                    cudaGraphExecUpdateResultInfo update_info{};
                    capture_status = cudaGraphExecUpdate(
                        m_seg_device_loop_exec, dg, &update_info);
                    ready = capture_status == cudaSuccess
                         && update_info.result == cudaGraphExecUpdateSuccess;
                    if(!ready)
                    {
                        cudaGraphExecDestroy(m_seg_device_loop_exec);
                        m_seg_device_loop_exec = nullptr;
                    }
                }
                if(!m_seg_device_loop_exec)
                {
                    capture_status = cudaGraphInstantiateWithFlags(
                        &m_seg_device_loop_exec,
                        dg,
                        cudaGraphInstantiateFlagDeviceLaunch);
                    ready   = capture_status == cudaSuccess && m_seg_device_loop_exec;
                    rebuilt = ready;
                }
            }
        }
        else
        {
            cudaGetLastError();
        }

        if(ready)
        {
            CUDA_SAFE_CALL(cudaGraphUpload(m_seg_device_loop_exec,
                                           cudaStreamPerThread));
            CUDA_SAFE_CALL(cudaGraphDestroy(dg));
            dg = nullptr;
            pcg_graph_state_init<<<1, 1>>>(d_graph_state, 1ULL);
            CUDA_SAFE_CALL(cudaGraphLaunch(m_seg_device_loop_exec,
                                           cudaStreamPerThread));

            unsigned long long graph_state[2] = {1ULL, 0ULL};
            CUDA_SAFE_CALL(cudaMemcpy(graph_state,
                                      d_graph_state,
                                      sizeof(graph_state),
                                      cudaMemcpyDeviceToHost));
            k = static_cast<SizeT>(graph_state[0]);
            const bool all_converged = graph_state[1] != 0;

            static bool once = false;
            if(!once)
            {
                once = true;
                printf("[pcg-device-loop] segmented self-tail graph active "
                       "(K=%d, driver=%d)\n",
                       (int)K,
                       s_driver_version_seg);
            }
            if(rebuilt && getenv("STIFF_PCG_GRAPH_DIAG"))
                printf("[pcg-device-loop] segmented executable rebuilt\n");

            if(!all_converged)
                for(; k < max_iter; ++k)
                    body();
            return k;
        }

        if(dg) cudaGraphDestroy(dg);
        cudaGetLastError();
        static bool oncef = false;
        if(!oncef)
        {
            oncef = true;
            printf("[pcg-device-loop] segmented unavailable (%s) "
                   "-> host graph fallback\n",
                   cudaGetErrorString(capture_status));
        }
    }

    cudaGraph_t     pg  = nullptr;
    cudaGraphExec_t pge = nullptr;
    bool            captured = false;
    k = 1;
    while(k < max_iter)
    {
        if(captured && k + K <= max_iter)
        {
            cudaGraphLaunch(pge, cudaStreamPerThread);   // = K iterations
            k += K;
        }
        else
        {
            SizeT stop = (k + K < max_iter) ? k + K : max_iter;
            for(; k < stop; ++k)
            {
                body();
                if(seg_trace)
                {
                    // rz_cur points at the freshest rz after body()'s swap.
                    double hrz[2] = {0, 0}, hdt[2] = {0, 0};
                    cudaMemcpy(hrz, rz_cur, 2 * sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(hdt, d_dot_g, 2 * sizeof(double), cudaMemcpyDeviceToHost);
                    printf("[seg-trace] it=%zu rz0=%.17e rz1=%.17e pAp0=%.17e pAp1=%.17e\n",
                           (size_t)k, hrz[0], hrz[1], hdt[0], hdt[1]);
                }
            }
            if(use_graph && !captured && k + K <= max_iter)
            {
                if(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)
                   == cudaSuccess)
                {
                    for(SizeT i = 0; i < K; ++i) body();   // recorded, not executed
                    if(cudaStreamEndCapture(cudaStreamPerThread, &pg) == cudaSuccess && pg
                       && cudaGraphInstantiate(&pge, pg, nullptr, nullptr, 0) == cudaSuccess)
                    {
                        captured = true;
                        static bool once = false;
                        if(!once) { once = true; printf("[pcg-graph] seg loop captured (K=%d)\n", (int)K); }
                    }
                }
                if(!captured)
                {
                    use_graph = false;
                    if(pg) { cudaGraphDestroy(pg); pg = nullptr; }
                    cudaGetLastError();
                    static bool oncef = false;
                    if(oncef == false) { oncef = true; printf("[pcg-graph] seg capture FAILED -> plain loop\n"); }
                }
            }
        }
        cudaMemcpy(h_brk.data(), d_break_g, ng * sizeof(int), cudaMemcpyDeviceToHost);
        bool all = true;
        for(int g = 0; g < ng; g++) if(!h_brk[g]) { all = false; break; }
        if(all) break;
    }
    if(pge) cudaGraphExecDestroy(pge);
    if(pg) cudaGraphDestroy(pg);
    if(getenv("STIFF_SEG_DIAG"))   // [P3 debug] converged or ran to max_iter?
    {
        static int _c = 0;
        if(_c++ < 4)
        {
            std::vector<double> hr(ng), h0(ng);
            cudaMemcpy(hr.data(), d_rz_g,  ng * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h0.data(), d_rz0_g, ng * sizeof(double), cudaMemcpyDeviceToHost);
            printf("[seg-diag] seg solve#%d: k=%zu (max=%zu) tol=%.2e  env0 rz/rz0=%.3e/%.3e ratio=%.3e\n",
                   _c, (size_t)k, (size_t)max_iter, tol, hr[0], h0[0], h0[0]!=0?hr[0]/h0[0]:0.0);
        }
    }
    return k;
}

}  // namespace gipc
