// GPU tactile depth rasteriser for StiffGIPC.
//
// The tactile "camera" is an orthographic z-buffer over the gel's coat surface
// (see stiff_tactile/observation.py for why): Taccel casts one ray per pixel
// from local z=+ray_start along -z, so the first hit is the largest-z surface
// point and the depth image is exactly the coat surface's local height field.
// This rasterises that directly from the engine's device vertex buffer, so no
// vertex data crosses the PCIe bus.
//
// One thread per coat triangle, atomicMax into a monotonic uint32 key buffer
// (~2.6k triangles x ~60 covered pixels for the fine gel = ~155k atomics, versus
// 160k pixels x 2.6k triangles if you parallelise over pixels instead).
//
// IMPORTANT: `tri` must hold ENGINE-order vertex indices. The raw device buffer
// is permuted by the MAS preconditioner; map input indices through
// Engine.get_vertex_input_to_metis() before uploading them here.

#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

// Order-preserving float <-> uint32 mapping so atomicMax on the key is a max on
// the float. 0u is the smallest key, so a zeroed buffer means "no hit".
__device__ __forceinline__ unsigned int f2key(float f)
{
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float key2f(unsigned int k)
{
    unsigned int u = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
    return __uint_as_float(u);
}

__global__ void raster_kernel(const double3* __restrict__ verts,
                              const int* __restrict__ tri,
                              int n_tri,
                              // world -> sensor-local: local = Rt * (world - t)
                              double rt0, double rt1, double rt2,
                              double rt3, double rt4, double rt5,
                              double rt6, double rt7, double rt8,
                              double tx, double ty, double tz,
                              int H, int W, double inv_p,
                              double half_h, double half_w,
                              unsigned int* __restrict__ zkey)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= n_tri)
        return;

    double lx[3], ly[3], lz[3];
    for(int k = 0; k < 3; k++)
    {
        const double3 v = verts[tri[t * 3 + k]];
        const double  dx = v.x - tx, dy = v.y - ty, dz = v.z - tz;
        lx[k] = rt0 * dx + rt1 * dy + rt2 * dz;
        ly[k] = rt3 * dx + rt4 * dy + rt5 * dz;
        lz[k] = rt6 * dx + rt7 * dy + rt8 * dz;
    }

    // Pixel-index bbox. Pixel (i, j) centre is ((i - half_h) * p, (j - half_w) * p).
    double xmin = fmin(lx[0], fmin(lx[1], lx[2])) * inv_p + half_h;
    double xmax = fmax(lx[0], fmax(lx[1], lx[2])) * inv_p + half_h;
    double ymin = fmin(ly[0], fmin(ly[1], ly[2])) * inv_p + half_w;
    double ymax = fmax(ly[0], fmax(ly[1], ly[2])) * inv_p + half_w;

    int i0 = max((int)ceil(xmin), 0), i1 = min((int)floor(xmax), H - 1);
    int j0 = max((int)ceil(ymin), 0), j1 = min((int)floor(ymax), W - 1);
    if(i0 > i1 || j0 > j1)
        return;

    const double e0x = lx[1] - lx[0], e0y = ly[1] - ly[0];
    const double e1x = lx[2] - lx[0], e1y = ly[2] - ly[0];
    const double det = e0x * e1y - e1x * e0y;
    if(fabs(det) < 1e-18)
        return;  // degenerate under projection
    const double inv_det = 1.0 / det;

    const double p = 1.0 / inv_p;
    for(int i = i0; i <= i1; i++)
    {
        const double px = (i - half_h) * p;
        for(int j = j0; j <= j1; j++)
        {
            const double py = (j - half_w) * p;
            const double qx = px - lx[0], qy = py - ly[0];
            const double w1 = (qx * e1y - e1x * qy) * inv_det;
            const double w2 = (e0x * qy - qx * e0y) * inv_det;
            const double w0 = 1.0 - w1 - w2;
            if(w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9)
                continue;
            const float z = (float)(w0 * lz[0] + w1 * lz[1] + w2 * lz[2]);
            atomicMax(&zkey[i * W + j], f2key(z));
        }
    }
}

// Area-weighted vertex normals of the DEFORMED coat surface, accumulated in the
// sensor-local frame. Phong-interpolating these removes the triangulation from
// the normal field, which is what the RGB MLP sees -- refining the mesh instead
// is hopeless: coat edge length scales as V^(1/3), so going from ~5 px to
// sub-pixel triangles would need ~175x more tets (~8e7) on a gel whose usable
// budget is ~1e4.
__global__ void vertex_normal_kernel(const double3* __restrict__ verts,
                                     const int* __restrict__ tri, int n_tri,
                                     double rt0, double rt1, double rt2,
                                     double rt3, double rt4, double rt5,
                                     double rt6, double rt7, double rt8,
                                     float* __restrict__ vn)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= n_tri)
        return;

    double p[3][3];
    int    id[3];
    for(int k = 0; k < 3; k++)
    {
        id[k]           = tri[t * 3 + k];
        const double3 v = verts[id[k]];
        p[k][0]         = rt0 * v.x + rt1 * v.y + rt2 * v.z;
        p[k][1]         = rt3 * v.x + rt4 * v.y + rt5 * v.z;
        p[k][2]         = rt6 * v.x + rt7 * v.y + rt8 * v.z;
    }
    // Un-normalised cross product == 2 * area * unit normal, so summing it is
    // already the area weighting.
    const double ax = p[1][0] - p[0][0], ay = p[1][1] - p[0][1], az = p[1][2] - p[0][2];
    const double bx = p[2][0] - p[0][0], by = p[2][1] - p[0][1], bz = p[2][2] - p[0][2];
    float nx = (float)(ay * bz - az * by);
    float ny = (float)(az * bx - ax * bz);
    float nz = (float)(ax * by - ay * bx);
    if(nz < 0.0f)  // orient toward the camera (+z in sensor local)
    {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }
    for(int k = 0; k < 3; k++)
    {
        atomicAdd(&vn[id[k] * 3 + 0], nx);
        atomicAdd(&vn[id[k] * 3 + 1], ny);
        atomicAdd(&vn[id[k] * 3 + 2], nz);
    }
}

// Second pass: re-rasterise and write the winning triangle's Phong-interpolated
// normal. The winner is identified by matching the depth pass's z-buffer.
__global__ void normal_kernel(const double3* __restrict__ verts,
                              const int* __restrict__ tri, int n_tri,
                              double rt0, double rt1, double rt2,
                              double rt3, double rt4, double rt5,
                              double rt6, double rt7, double rt8,
                              double tx, double ty, double tz,
                              int H, int W, double inv_p,
                              double half_h, double half_w,
                              const unsigned int* __restrict__ zkey,
                              const float* __restrict__ vn,
                              float* __restrict__ normal)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= n_tri)
        return;

    double lx[3], ly[3], lz[3];
    int    id[3];
    for(int k = 0; k < 3; k++)
    {
        id[k]            = tri[t * 3 + k];
        const double3 v  = verts[id[k]];
        const double  dx = v.x - tx, dy = v.y - ty, dz = v.z - tz;
        lx[k] = rt0 * dx + rt1 * dy + rt2 * dz;
        ly[k] = rt3 * dx + rt4 * dy + rt5 * dz;
        lz[k] = rt6 * dx + rt7 * dy + rt8 * dz;
    }

    double xmin = fmin(lx[0], fmin(lx[1], lx[2])) * inv_p + half_h;
    double xmax = fmax(lx[0], fmax(lx[1], lx[2])) * inv_p + half_h;
    double ymin = fmin(ly[0], fmin(ly[1], ly[2])) * inv_p + half_w;
    double ymax = fmax(ly[0], fmax(ly[1], ly[2])) * inv_p + half_w;
    int i0 = max((int)ceil(xmin), 0), i1 = min((int)floor(xmax), H - 1);
    int j0 = max((int)ceil(ymin), 0), j1 = min((int)floor(ymax), W - 1);
    if(i0 > i1 || j0 > j1)
        return;

    const double e0x = lx[1] - lx[0], e0y = ly[1] - ly[0];
    const double e1x = lx[2] - lx[0], e1y = ly[2] - ly[0];
    const double det = e0x * e1y - e1x * e0y;
    if(fabs(det) < 1e-18)
        return;
    const double inv_det = 1.0 / det;
    const double p       = 1.0 / inv_p;

    for(int i = i0; i <= i1; i++)
    {
        const double px = (i - half_h) * p;
        for(int j = j0; j <= j1; j++)
        {
            const double py = (j - half_w) * p;
            const double qx = px - lx[0], qy = py - ly[0];
            const double w1 = (qx * e1y - e1x * qy) * inv_det;
            const double w2 = (e0x * qy - qx * e0y) * inv_det;
            const double w0 = 1.0 - w1 - w2;
            if(w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9)
                continue;
            const float z   = (float)(w0 * lz[0] + w1 * lz[1] + w2 * lz[2]);
            const int   idx = i * W + j;
            if(f2key(z) != zkey[idx])
                continue;  // another triangle is in front here
            float nx = 0.f, ny = 0.f, nz = 0.f;
            const double w[3] = {w0, w1, w2};
            for(int k = 0; k < 3; k++)
            {
                nx += (float)w[k] * vn[id[k] * 3 + 0];
                ny += (float)w[k] * vn[id[k] * 3 + 1];
                nz += (float)w[k] * vn[id[k] * 3 + 2];
            }
            const float inv_len = rsqrtf(fmaxf(nx * nx + ny * ny + nz * nz, 1e-30f));
            normal[idx * 3 + 0] = nx * inv_len;
            normal[idx * 3 + 1] = ny * inv_len;
            normal[idx * 3 + 2] = nz * inv_len;
        }
    }
}

__global__ void finalize_kernel(const unsigned int* __restrict__ zkey,
                                float* __restrict__ depth,
                                int n_px, float rest_level, float max_depth)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= n_px)
        return;
    const unsigned int key = zkey[k];
    float d = (key == 0u) ? 0.0f : (key2f(key) - rest_level);
    depth[k] = fminf(fmaxf(d, 0.0f), max_depth);
}

}  // namespace

// verts_ptr: device double3* (engine order), e.g. Engine.get_vertices_device_ptr()
// tri:       (T, 3) int32 CUDA tensor of ENGINE-order vertex indices
// rt:        (3, 3) float64 CPU tensor, world->local rotation (= R^T)
// t:         (3,)   float64 CPU tensor, sensor origin in world
torch::Tensor raster_depth(int64_t verts_ptr, torch::Tensor tri, torch::Tensor rt,
                           torch::Tensor t, int64_t H, int64_t W, double pixel_size_m,
                           double rest_level, double max_depth)
{
    TORCH_CHECK(tri.is_cuda() && tri.scalar_type() == torch::kInt32 && tri.dim() == 2
                    && tri.size(1) == 3,
                "tri must be a (T,3) int32 CUDA tensor");
    TORCH_CHECK(tri.is_contiguous(), "tri must be contiguous");
    auto rtc = rt.to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto tc  = t.to(torch::kCPU).to(torch::kFloat64).contiguous();
    TORCH_CHECK(rtc.numel() == 9 && tc.numel() == 3, "rt must have 9 and t 3 elements");
    const double* r = rtc.data_ptr<double>();
    const double* o = tc.data_ptr<double>();

    auto opts  = torch::TensorOptions().device(tri.device());
    auto zkey  = torch::zeros({H, W}, opts.dtype(torch::kInt32));
    auto depth = torch::empty({H, W}, opts.dtype(torch::kFloat32));

    const int n_tri = (int)tri.size(0);
    const int n_px  = (int)(H * W);
    if(n_tri > 0)
    {
        const int threads = 128;
        raster_kernel<<<(n_tri + threads - 1) / threads, threads>>>(
            reinterpret_cast<const double3*>(verts_ptr), tri.data_ptr<int>(), n_tri,
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], o[0], o[1], o[2],
            (int)H, (int)W, 1.0 / pixel_size_m, (H - 1) * 0.5, (W - 1) * 0.5,
            reinterpret_cast<unsigned int*>(zkey.data_ptr<int>()));
    }
    {
        const int threads = 256;
        finalize_kernel<<<(n_px + threads - 1) / threads, threads>>>(
            reinterpret_cast<const unsigned int*>(zkey.data_ptr<int>()),
            depth.data_ptr<float>(), n_px, (float)rest_level, (float)max_depth);
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return depth;
}

// Same as raster_depth, plus a Phong-interpolated normal map from area-weighted
// deformed-coat vertex normals. `n_verts_total` sizes the vertex-normal scratch
// (index space of the device vertex buffer). Pixels with no hit get +z.
std::vector<torch::Tensor> raster_depth_normal(int64_t verts_ptr, torch::Tensor tri,
                                               torch::Tensor rt, torch::Tensor t,
                                               int64_t H, int64_t W, double pixel_size_m,
                                               double rest_level, double max_depth,
                                               int64_t n_verts_total)
{
    TORCH_CHECK(tri.is_cuda() && tri.scalar_type() == torch::kInt32 && tri.dim() == 2
                    && tri.size(1) == 3 && tri.is_contiguous(),
                "tri must be a contiguous (T,3) int32 CUDA tensor");
    auto rtc = rt.to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto tc  = t.to(torch::kCPU).to(torch::kFloat64).contiguous();
    const double* r = rtc.data_ptr<double>();
    const double* o = tc.data_ptr<double>();

    auto opts   = torch::TensorOptions().device(tri.device());
    auto zkey   = torch::zeros({H, W}, opts.dtype(torch::kInt32));
    auto depth  = torch::empty({H, W}, opts.dtype(torch::kFloat32));
    auto normal = torch::zeros({H, W, 3}, opts.dtype(torch::kFloat32));
    auto vn     = torch::zeros({n_verts_total, 3}, opts.dtype(torch::kFloat32));

    const int n_tri = (int)tri.size(0);
    const int n_px  = (int)(H * W);
    const auto* vp  = reinterpret_cast<const double3*>(verts_ptr);
    auto* zk        = reinterpret_cast<unsigned int*>(zkey.data_ptr<int>());

    if(n_tri > 0)
    {
        const int threads = 128, blocks = (n_tri + threads - 1) / threads;
        raster_kernel<<<blocks, threads>>>(
            vp, tri.data_ptr<int>(), n_tri, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
            r[7], r[8], o[0], o[1], o[2], (int)H, (int)W, 1.0 / pixel_size_m,
            (H - 1) * 0.5, (W - 1) * 0.5, zk);
        vertex_normal_kernel<<<blocks, threads>>>(
            vp, tri.data_ptr<int>(), n_tri, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
            r[7], r[8], vn.data_ptr<float>());
        normal_kernel<<<blocks, threads>>>(
            vp, tri.data_ptr<int>(), n_tri, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
            r[7], r[8], o[0], o[1], o[2], (int)H, (int)W, 1.0 / pixel_size_m,
            (H - 1) * 0.5, (W - 1) * 0.5, zk, vn.data_ptr<float>(),
            normal.data_ptr<float>());
    }
    {
        const int threads = 256;
        finalize_kernel<<<(n_px + threads - 1) / threads, threads>>>(
            zk, depth.data_ptr<float>(), n_px, (float)rest_level, (float)max_depth);
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return {depth, normal};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("raster_depth", &raster_depth,
          "Orthographic z-buffer of a gel's coat surface, straight off the engine's "
          "device vertex buffer. Returns (H, W) float32 depth in metres.");
    m.def("raster_depth_normal", &raster_depth_normal,
          "As raster_depth, plus a Phong-interpolated (H, W, 3) normal map from "
          "area-weighted deformed vertex normals.");
}
