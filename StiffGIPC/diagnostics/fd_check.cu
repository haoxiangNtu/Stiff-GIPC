// ============================================================================
// diagnostics/fd_check.cu — finite-difference gradient consistency check
// (release/tag FD gate, 2026-07-27). Own TU; TEST-ONLY code path:
// never touched by production stepping — a gate script calls it explicitly
// between frames on small scenes.
//
// Method: after computeGradientAndHessian fills the per-vertex analytic
// gradient (free FEM vertex i: fb(i) + shape_grads(i), exactly the
// FEMLinearSubsystem::assemble combination — dt^2/Kappa scalings live inside
// both the gradient assembly and computeEnergy's combine, so the two sides
// share one scale), probe random free FEM vertex coordinates: central
// difference (E(x+h)-E(x-h))/2h vs the analytic component. Pair lists,
// lagged friction state and xTilta are all frozen during the +-h probes, so
// E and G describe the same frozen configuration.
//
// SCOPE: FEM-side vertices only (ABD surface vertices derive from q — their
// consistency is the affine system's own contract). Sign convention is
// auto-detected on the first probe and must stay consistent for the rest.
// ============================================================================
#include "GIPC.cuh"
#include "cuda_tools/cuda_tools.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

FdCheckResult GIPC::fd_gradient_check(device_TetraData& TetMesh, double h, int nprobes, unsigned seed)
{
    FdCheckResult r{};
    const int off = abd_fem_count_info.fem_point_offset;
    const int n   = abd_fem_count_info.fem_point_num;
    if(n < 1 || nprobes < 1)
        return r;

    // analytic gradient for the CURRENT state
    computeGradientAndHessian(TetMesh);
    std::vector<double3> fb(n), sg(n), x0(n);
    std::vector<int>     bt(n);
    CUDA_SAFE_CALL(cudaMemcpy(fb.data(), TetMesh.fb + off, n * sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(sg.data(), TetMesh.shape_grads + off, n * sizeof(double3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(bt.data(), TetMesh.BoundaryType + off, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(x0.data(), TetMesh.vertexes + off, n * sizeof(double3), cudaMemcpyDeviceToHost));

    std::vector<int> free_verts;
    for(int i = 0; i < n; ++i)
        if(bt[i] == 0)
            free_verts.push_back(i);
    if(free_verts.empty())
        return r;

    unsigned s = seed ? seed : 12345u;
    auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return s; };

    double sign = 0.0;  // auto-detected on first finite probe
    std::vector<double> rels;
    for(int p = 0; p < nprobes; ++p)
    {
        const int    v  = free_verts[rnd() % free_verts.size()];
        const int    a  = rnd() % 3;
        double*      xa = reinterpret_cast<double*>(TetMesh.vertexes + off + v) + a;
        const double x  = reinterpret_cast<const double*>(&x0[v])[a];

        double xp = x + h, xm = x - h;
        CUDA_SAFE_CALL(cudaMemcpy(xa, &xp, sizeof(double), cudaMemcpyHostToDevice));
        const double ep = computeEnergy(TetMesh);
        CUDA_SAFE_CALL(cudaMemcpy(xa, &xm, sizeof(double), cudaMemcpyHostToDevice));
        const double em = computeEnergy(TetMesh);
        CUDA_SAFE_CALL(cudaMemcpy(xa, &x, sizeof(double), cudaMemcpyHostToDevice));

        const double fd = (ep - em) / (2.0 * h);
        const double an = reinterpret_cast<const double*>(&fb[v])[a]
                        + reinterpret_cast<const double*>(&sg[v])[a];
        if(!std::isfinite(fd) || !std::isfinite(an))
        {   // a NaN energy/gradient probe is a HARD failure, never a skip
            ++r.n_nonfinite;
            r.max_rel = std::numeric_limits<double>::infinity();
            if(r.worst_v < 0) { r.worst_v = v; r.worst_axis = a; r.worst_fd = fd; r.worst_an = an; }
            ++r.n;
            continue;
        }
        if(std::getenv("FD_DEBUG"))
            printf("[fd-probe] v=%d axis=%d fd=%.6e an=%.6e (fb=%.3e sg=%.3e)\n", v, a, fd, an,
                   reinterpret_cast<const double*>(&fb[v])[a],
                   reinterpret_cast<const double*>(&sg[v])[a]);
        // central-difference cancellation floor: (E+ - E-) carries ~eps*|E|
        // of roundoff, so fd below this bound is indistinguishable from zero.
        const double noise = 32.0 * 2.220446049250313e-16
                           * std::max(std::abs(ep), std::abs(em)) / (2.0 * h);
        if(std::max(std::abs(fd), std::abs(an)) < std::max(noise, 1e-12))
        {   // both sides zero at the FD noise floor -> consistent
            rels.push_back(0.0);
            ++r.n;
            continue;
        }
        if(sign == 0.0)
            sign = (std::abs(fd - an) <= std::abs(fd + an)) ? 1.0 : -1.0;
        const double diff  = std::abs(fd - sign * an);
        const double scale = std::max({noise, std::abs(fd), std::abs(an)});
        const double rel   = diff / scale;
        rels.push_back(rel);
        r.mean_rel += rel;
        if(rel > r.max_rel)
        {
            r.max_rel   = rel;
            r.worst_v   = v;
            r.worst_axis= a;
            r.worst_fd  = fd;
            r.worst_an  = sign * an;
        }
        ++r.n;
    }
    if(!rels.empty())
    {
        r.mean_rel /= (double)rels.size();
        std::sort(rels.begin(), rels.end());
        r.p50 = rels[rels.size() / 2];
        r.p95 = rels[(size_t)((double)(rels.size() - 1) * 0.95)];
    }
    r.sign = sign;
    return r;
}

FdActivityResult GIPC::fd_activity()
{
    FdActivityResult r{};
    r.fem_tets        = static_cast<uint32_t>(abd_fem_count_info.fem_tet_num);
    r.triangles       = triangleNum;
    r.bending_edges   = tri_edge_num;
    r.soft            = softNum;
    r.contact         = h_cpNum[0];
    r.ground          = h_gpNum.get();
    r.friction        = h_cpNum_last[0];
    r.ground_friction = h_gpNum_last.get();
    return r;
}

FdHessianResult GIPC::fd_hessian_diagonal_check(device_TetraData& TetMesh,
                                                double h,
                                                int nprobes,
                                                unsigned seed)
{
    FdHessianResult r{};
    // The raw triplet row/column ids are directly vertex-indexed only for a
    // pure-FEM scene. ABD q-space and coupling checks require their own map and
    // are deliberately rejected rather than silently checking the wrong block.
    if(abd_fem_count_info.abd_body_num != 0 || nprobes < 1 || h <= 0.0)
        return r;

    const int off = static_cast<int>(abd_fem_count_info.fem_point_offset);
    const int n   = static_cast<int>(abd_fem_count_info.fem_point_num);
    if(off != 0 || n < 1)
        return r;

    computeGradientAndHessian(TetMesh);
    const size_t live = static_cast<size_t>(gipc_global_triplet.global_triplet_offset);
    if(live == 0)
        return r;

    std::vector<Eigen::Matrix3d> blocks(live);
    std::vector<int> rows(live), cols(live);
    CUDA_SAFE_CALL(cudaMemcpy(blocks.data(),
                              gipc_global_triplet.block_values(),
                              live * sizeof(Eigen::Matrix3d),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(rows.data(),
                              gipc_global_triplet.block_row_indices(),
                              live * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(cols.data(),
                              gipc_global_triplet.block_col_indices(),
                              live * sizeof(int),
                              cudaMemcpyDeviceToHost));

    std::vector<int> boundary(n);
    std::vector<double3> x0(n);
    CUDA_SAFE_CALL(cudaMemcpy(boundary.data(),
                              TetMesh.BoundaryType,
                              n * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(x0.data(),
                              TetMesh.vertexes,
                              n * sizeof(double3),
                              cudaMemcpyDeviceToHost));
    std::vector<int> free_verts;
    for(int i = 0; i < n; ++i)
        if(boundary[i] == 0)
            free_verts.push_back(i);
    if(free_verts.empty())
        return r;

    auto hdiag = [&](int v, int axis)
    {
        double out = 0.0;
        for(size_t i = 0; i < live; ++i)
            if(rows[i] == v && cols[i] == v)
                out += blocks[i](axis, axis);
        return out;
    };
    auto gradient_component = [&](int v, int axis)
    {
        double3 fb{}, sg{};
        CUDA_SAFE_CALL(cudaMemcpy(&fb,
                                  TetMesh.fb + v,
                                  sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&sg,
                                  TetMesh.shape_grads + v,
                                  sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        return reinterpret_cast<double*>(&fb)[axis]
             + reinterpret_cast<double*>(&sg)[axis];
    };

    unsigned s = seed ? seed : 54321u;
    auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return s; };
    std::vector<double> rels;
    for(int p = 0; p < nprobes; ++p)
    {
        const int v = free_verts[rnd() % free_verts.size()];
        const int a = rnd() % 3;
        double* xa = reinterpret_cast<double*>(TetMesh.vertexes + v) + a;
        const double x = reinterpret_cast<const double*>(&x0[v])[a];

        const double xp = x + h;
        CUDA_SAFE_CALL(cudaMemcpy(xa, &xp, sizeof(double), cudaMemcpyHostToDevice));
        computeGradientAndHessian(TetMesh);
        const double gp = gradient_component(v, a);

        const double xm = x - h;
        CUDA_SAFE_CALL(cudaMemcpy(xa, &xm, sizeof(double), cudaMemcpyHostToDevice));
        computeGradientAndHessian(TetMesh);
        const double gm = gradient_component(v, a);

        CUDA_SAFE_CALL(cudaMemcpy(xa, &x, sizeof(double), cudaMemcpyHostToDevice));
        const double dg = (gp - gm) / (2.0 * h);
        const double hd = hdiag(v, a);
        if(!std::isfinite(dg) || !std::isfinite(hd))
        {
            ++r.n_nonfinite;
            r.max_rel = std::numeric_limits<double>::infinity();
            if(r.worst_v < 0)
            {
                r.worst_v = v;
                r.worst_axis = a;
            }
            ++r.n;
            continue;
        }
        if(std::max(std::abs(dg), std::abs(hd)) < 1e-9)
        {
            rels.push_back(0.0);
            ++r.n;
            continue;
        }
        if(r.sign == 0.0)
            r.sign = (std::abs(dg - hd) <= std::abs(dg + hd)) ? 1.0 : -1.0;
        const double rel = std::abs(dg - r.sign * hd)
                         / std::max({1e-9, std::abs(dg), std::abs(hd)});
        rels.push_back(rel);
        r.mean_rel += rel;
        if(rel > r.max_rel)
        {
            r.max_rel = rel;
            r.worst_v = v;
            r.worst_axis = a;
        }
        ++r.n;
    }

    // Leave the engine in the unperturbed, fully assembled state.
    computeGradientAndHessian(TetMesh);
    if(!rels.empty())
    {
        r.mean_rel /= static_cast<double>(rels.size());
        std::sort(rels.begin(), rels.end());
        r.p50 = rels[rels.size() / 2];
        r.p95 = rels[static_cast<size_t>((rels.size() - 1) * 0.95)];
    }
    return r;
}
