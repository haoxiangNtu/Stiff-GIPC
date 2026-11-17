// Well-balanced cut-cell operator for the MPM Lite water (LIQ_WB), 2026-10-23.
//
// Still water is an exact stationary point of the discrete energy only if the pressure area vectors, the within-cell
// hydrostatic pressure variation and the nodal weights all come from the SAME fluid sub-volume of each cut cell:
//     a_q    = oint_{non-wall boundary} N_q n dA          pressure area vectors (volume-change form: wall faces act on the wall)
//     b_q    = int_{fluid} (g.(x - x_c)) grad N_q dV     within-cell hydrostatic variation (the whole pressure near the surface)
//     beta_q = oint_{wall} (g.(x - x_c)) N_q n dA        its wall part (handed to the wall too)
//     w_q    = rho int_{fluid} N_q dV                    consistent nodal weight (replaces the kernel-deposited mass for GRAVITY only)
// nodal force F_q = sum_c [p_c a_q + rho (b_q - beta_q)] balances rho w_q g exactly for a linear pressure field, for any wall shape
// that is represented consistently between neighbouring cells. Design + proofs: LIQUID_CUTCELL_DESIGN_2026-10-23.md
// (13:40, 14:05, 14:40); CPU reference of THIS algorithm (bit-comparable): cutcell_mt_proto_2026-10-23.py.
//
// Geometry: the wall is the zero set of a nodal signed distance phi (fluid where phi >= 0, from the particles' wall points:
// min over nearby wall points of n.(x - X)), the free surface the plane up.x = h_c of the cell (from its deposited volume; only
// air-adjacent / under-filled cells are cut by it). A cell is split into 6 Kuhn tetrahedra (every cube face is split along the
// same diagonal from both sides), each tet is cut by the linear phi, each piece by the surface plane; vertices carry a bitmask
// of the boundary planes they lie on, a sub-tet face whose three vertices share a bit is a boundary face of that kind, other
// faces are internal. Degree-3 (tet) and degree-4 (triangle) rules integrate the polynomial integrands exactly.
#pragma once

namespace gipc_liquid_wb {

constexpr int WB_X0 = 1, WB_X1 = 2, WB_Y0 = 4, WB_Y1 = 8, WB_Z0 = 16, WB_Z1 = 32, WB_WALL = 64, WB_SURF = 128, WB_FACEBITS = 63;

struct WbV { double3 p; int m; };
struct WbT { WbV v[4]; };

__device__ __forceinline__ double  wb_dot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ double3 wb_sub(double3 a, double3 b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ double3 wb_add(double3 a, double3 b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ double3 wb_scale(double3 a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ double3 wb_cross(double3 a, double3 b) { return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ void    wb_axpy(double3& acc, double s, double3 v) { acc.x += s * v.x; acc.y += s * v.y; acc.z += s * v.z; }

__device__ __forceinline__ int wb_corner_mask(int q)   // q = 4 a + 2 b + c
{
    int m = (q & 4) ? WB_X1 : WB_X0;
    m |= (q & 2) ? WB_Y1 : WB_Y0;
    m |= (q & 1) ? WB_Z1 : WB_Z0;
    return m;
}
__device__ __forceinline__ double wb_N(int q, double3 s)
{
    return ((q & 4) ? s.x : 1.0 - s.x) * ((q & 2) ? s.y : 1.0 - s.y) * ((q & 1) ? s.z : 1.0 - s.z);
}
__device__ __forceinline__ double3 wb_gradN(int q, double3 s, double dx)
{
    const double fx = (q & 4) ? s.x : 1.0 - s.x, fy = (q & 2) ? s.y : 1.0 - s.y, fz = (q & 1) ? s.z : 1.0 - s.z;
    const double gx = ((q & 4) ? 1.0 : -1.0) / dx, gy = ((q & 2) ? 1.0 : -1.0) / dx, gz = ((q & 1) ? 1.0 : -1.0) / dx;
    return make_double3(gx * fy * fz, fx * gy * fz, fx * fy * gz);
}

// edge point between kept vertex i and cut vertex j (canonical: evaluated from the kept side, same from both cells)
__device__ __forceinline__ WbV wb_edge(const WbT& t, const double f[4], int i, int j, int bit)
{
    const double tt = f[i] / (f[i] - f[j]);
    WbV r;
    r.p = make_double3(t.v[i].p.x + tt * (t.v[j].p.x - t.v[i].p.x), t.v[i].p.y + tt * (t.v[j].p.y - t.v[i].p.y), t.v[i].p.z + tt * (t.v[j].p.z - t.v[i].p.z));
    r.m = (t.v[i].m & t.v[j].m) | bit;
    return r;
}

// keep f >= 0: 0..3 output tets. |f| < eps is snapped to 0 = ON the cut plane: the vertex is kept and gets the plane's bit
// (a wall through a lattice node, the surface at a cell top). Without this a face spanned by on-wall corners would be taken
// for a pressure face, and the zero-volume sliver pieces would carry coincident faces of different kinds that cannot cancel.
__device__ __forceinline__ int wb_cut(const WbT& tin, const double fin[4], int bit, double eps, WbT out[3])
{
    WbT    t = tin;
    double f[4];
    int    P[4], Nn[4], np = 0, nn = 0;
    for(int i = 0; i < 4; ++i)
    {
        f[i] = (fabs(fin[i]) < eps) ? 0.0 : fin[i];
        if(f[i] == 0.0) t.v[i].m |= bit;
        if(f[i] >= 0.0) P[np++] = i;
        else Nn[nn++] = i;
    }
    if(np == 0) return 0;
    if(np == 4) { out[0] = t; return 1; }
    if(np == 1)
    {
        const int i = P[0];
        out[0].v[0] = t.v[i]; out[0].v[1] = wb_edge(t, f, i, Nn[0], bit); out[0].v[2] = wb_edge(t, f, i, Nn[1], bit); out[0].v[3] = wb_edge(t, f, i, Nn[2], bit);
        return 1;
    }
    if(np == 3)
    {   // prism (a, b, c | ea, eb, ec): (0,1,2,3), (1,2,3,4), (2,3,4,5)
        const int j = Nn[0], a = P[0], b = P[1], c = P[2];
        const WbV ea = wb_edge(t, f, a, j, bit), eb = wb_edge(t, f, b, j, bit), ec = wb_edge(t, f, c, j, bit);
        out[0].v[0] = t.v[a]; out[0].v[1] = t.v[b]; out[0].v[2] = t.v[c]; out[0].v[3] = ea;
        out[1].v[0] = t.v[b]; out[1].v[1] = t.v[c]; out[1].v[2] = ea;     out[1].v[3] = eb;
        out[2].v[0] = t.v[c]; out[2].v[1] = ea;     out[2].v[2] = eb;     out[2].v[3] = ec;
        return 3;
    }
    // np == 2: prism with triangles (a, eac, ead) and (b, ebc, ebd)
    const int a = P[0], b = P[1], c = Nn[0], d = Nn[1];
    const WbV eac = wb_edge(t, f, a, c, bit), ead = wb_edge(t, f, a, d, bit), ebc = wb_edge(t, f, b, c, bit), ebd = wb_edge(t, f, b, d, bit);
    out[0].v[0] = t.v[a]; out[0].v[1] = eac;    out[0].v[2] = ead;    out[0].v[3] = t.v[b];
    out[1].v[0] = eac;    out[1].v[1] = ead;    out[1].v[2] = t.v[b]; out[1].v[3] = ebc;
    out[2].v[0] = ead;    out[2].v[1] = t.v[b]; out[2].v[2] = ebc;    out[2].v[3] = ebd;
    return 3;
}

// Kuhn tetrahedron #perm (0..5) of the cell whose min corner is lattice node (i, j, k): corners computed from GLOBAL node
// indices so that a shared corner is bitwise the same from every cell
__device__ __forceinline__ void wb_kuhn(int perm, int i, int j, int k, double3 org, double dx, const double phi[8], WbT& t, double f[4])
{
    const int ax0 = perm / 2, ax1 = (perm & 1) ? ((ax0 + 2) % 3) : ((ax0 + 1) % 3), ax2 = 3 - ax0 - ax1;
    int a = 0, b = 0, c = 0;
    const int axes[3] = {ax0, ax1, ax2};
    for(int n = 0; n < 4; ++n)
    {
        if(n > 0)
        {
            const int ax = axes[n - 1];
            if(ax == 0) a = 1; else if(ax == 1) b = 1; else c = 1;
        }
        const int q = 4 * a + 2 * b + c;
        t.v[n].p = make_double3(org.x + (double)(i + a) * dx, org.y + (double)(j + b) * dx, org.z + (double)(k + c) * dx);
        t.v[n].m = wb_corner_mask(q);
        f[n]     = phi[q];
    }
}

struct WbAcc
{
    double  V;
    double3 M;        // sum vol * centroid
    double3 G[8];     // int grad N_q dV
    double3 A[8];     // int (g.x) grad N_q dV
    double  W[8];     // int N_q dV
    double3 a[8];     // oint_{non-wall} N_q n dA
    double3 om[8];    // oint_{wall} N_q n dA
    double3 omg[8];   // oint_{wall} (g.x) N_q n dA
    double3 Awn;      // oint_{wall} n dA
    double  S[36];    // int_{surface} N_q N_r dA_h (horizontal projection), upper triangle q<=r: free-surface gravity stiffness rho g S
};

__device__ __forceinline__ void wb_acc_init(WbAcc& A)
{
    A.V = 0.0; A.M = make_double3(0, 0, 0); A.Awn = make_double3(0, 0, 0);
    for(int q = 0; q < 8; ++q)
    {
        A.G[q] = A.A[q] = A.a[q] = A.om[q] = A.omg[q] = make_double3(0, 0, 0);
        A.W[q] = 0.0;
        for(int t = 0; t < 36; ++t) A.S[t] = 0.0;
}
}

__device__ __forceinline__ double wb_tet_volume(const WbT& T)
{
    const double3 e1 = wb_sub(T.v[1].p, T.v[0].p), e2 = wb_sub(T.v[2].p, T.v[0].p), e3 = wb_sub(T.v[3].p, T.v[0].p);
    return fabs(wb_dot(wb_cross(e1, e2), e3)) / 6.0;
}

// full accumulation of one final sub-tet: volume rule (degree 3, 5 points) + its boundary faces (degree 4, 6 points)
__device__ __forceinline__ void wb_tet_accumulate(const WbT& T, double3 o, double dx, double3 g, WbAcc& A, double3* aT = nullptr, double* VT = nullptr)
{
    const double vol = wb_tet_volume(T);
    if(vol <= 1e-14 * dx * dx * dx) return;   // flat pieces (a cut through a vertex) carry nothing; their faces belong to their neighbours
    A.V += vol;
    if(VT) *VT += vol;
    const double3 cen = wb_scale(wb_add(wb_add(T.v[0].p, T.v[1].p), wb_add(T.v[2].p, T.v[3].p)), 0.25);
    wb_axpy(A.M, vol, cen);
    // 5-point rule: centroid weight -4/5, four points (1/2, 1/6, 1/6, 1/6) weight 9/20
    for(int r = 0; r < 5; ++r)
    {
        double bc[4] = {0.25, 0.25, 0.25, 0.25}, wt = -0.8;
        if(r > 0)
        {
            bc[0] = bc[1] = bc[2] = bc[3] = 1.0 / 6.0; bc[r - 1] = 0.5; wt = 0.45;
        }
        wt *= vol;
        double3 x = make_double3(0, 0, 0);
        for(int n = 0; n < 4; ++n) wb_axpy(x, bc[n], T.v[n].p);
        const double3 s  = make_double3((x.x - o.x) / dx, (x.y - o.y) / dx, (x.z - o.z) / dx);
        const double  gx = wb_dot(g, x);
        for(int q = 0; q < 8; ++q)
        {
            const double  Nq = wb_N(q, s);
            const double3 gN = wb_gradN(q, s, dx);
            A.W[q] += wt * Nq;
            wb_axpy(A.G[q], wt, gN);
            wb_axpy(A.A[q], wt * gx, gN);
        }
    }
    // faces
    for(int kf = 0; kf < 4; ++kf)
    {
        const int i0 = (kf + 1) & 3, i1 = (kf + 2) & 3, i2 = (kf + 3) & 3;
        const int mask = T.v[i0].m & T.v[i1].m & T.v[i2].m;
        int kind;   // 1 wall, 2 surface, 0 cube face, 3 internal (between pieces of the cell: cancels in the cell sum, needed per piece)
        if(mask & WB_WALL) kind = 1;
        else if(mask & WB_SURF) kind = 2;
        else if(mask & WB_FACEBITS) kind = 0;
        else kind = 3;
        if(kind == 3 && !aT) continue;
        const double3 Pa = T.v[i0].p, Pb = T.v[i1].p, Pc = T.v[i2].p;
        double3 nrm = wb_cross(wb_sub(Pb, Pa), wb_sub(Pc, Pa));
        const double nn = sqrt(wb_dot(nrm, nrm));
        if(nn < 1e-300) continue;
        const double area = 0.5 * nn;
        nrm = wb_scale(nrm, 1.0 / nn);
        if(wb_dot(nrm, wb_sub(T.v[kf].p, Pa)) > 0.0) nrm = wb_scale(nrm, -1.0);   // outward: away from the fourth vertex
        // Dunavant degree-4 rule, 6 points
        const double wA = 0.223381589678011, aA = 0.445948490915965, wB = 0.109951743655322, aB = 0.091576213509771;
        for(int r = 0; r < 6; ++r)
        {
            const double w  = (r < 3) ? wA : wB, al = (r < 3) ? aA : aB, be = 1.0 - 2.0 * al;
            double bc[3] = {al, al, al}; bc[r % 3] = be;
            const double3 x  = wb_add(wb_add(wb_scale(Pa, bc[0]), wb_scale(Pb, bc[1])), wb_scale(Pc, bc[2]));
            const double  wt = w * area;
            const double3 s  = make_double3((x.x - o.x) / dx, (x.y - o.y) / dx, (x.z - o.z) / dx);
            const double  gx = wb_dot(g, x);
            if(kind == 1) wb_axpy(A.Awn, wt, nrm);
            if(kind == 2)
            {   // free-surface face: horizontal-projected area measure for the gravity stiffness 1/2 rho g int (u.up)^2 dA_h
                const double gm = sqrt(wb_dot(g, g));
                const double ca = gm > 0.0 ? fmax(0.0, -wb_dot(nrm, g) / gm) : 1.0;   // nrm points out of the fluid (up-ish)
                const double wh = wt * ca;
                double Nv[8];
                for(int q = 0; q < 8; ++q) Nv[q] = wb_N(q, s);
                int t = 0;
                for(int q = 0; q < 8; ++q)
                    for(int r = q; r < 8; ++r, ++t) A.S[t] += wh * Nv[q] * Nv[r];
            }
            for(int q = 0; q < 8; ++q)
            {
                const double Nq = wb_N(q, s);
                if(kind == 1)
                {
                    wb_axpy(A.om[q], wt * Nq, nrm);
                    wb_axpy(A.omg[q], wt * Nq * gx, nrm);
                }
                else
                {
                    if(kind != 3) wb_axpy(A.a[q], wt * Nq, nrm);
                    if(aT) wb_axpy(aT[q], wt * Nq, nrm);   // the piece's own non-wall boundary (internal faces included)
                }
            }
        }
    }
}

// the region {phi >= 0} ∩ {up.x <= h} of cell (i, j, k): volume only (full = false) or the full accumulation
__device__ __forceinline__ double wb_region(int i, int j, int k, double3 org, double dx, const double phi[8], double3 up, double h, double3 g,
                                            bool full, WbAcc* A, double* hg = nullptr)
{   // hg (LIQ_WB_HG): per Kuhn tet T = 0..5 the fluid piece's volume V_T and non-wall boundary vectors a^T_q (25 doubles each):
    // sub-cell volumetric strains delta_T = sum_q a^T_q.u_q / V_T for the hourglass control of the one-point cut-cell energy
    double V = 0.0;
    const double3 o = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    const double eps = 1e-9 * dx;   // on-plane snap (see wb_cut)
    for(int perm = 0; perm < 6; ++perm)
    {
        double3 aT[8];
        double  VT = 0.0;
        if(hg) for(int q = 0; q < 8; ++q) aT[q] = make_double3(0, 0, 0);
        WbT    t0; double f0[4];
        wb_kuhn(perm, i, j, k, org, dx, phi, t0, f0);
        WbT t1[3];
        const int n1 = wb_cut(t0, f0, WB_WALL, eps, t1);
        for(int u = 0; u < n1; ++u)
        {
            double fs[4];
            for(int n = 0; n < 4; ++n) fs[n] = h - wb_dot(up, t1[u].v[n].p);
            WbT t2[3];
            const int n2 = wb_cut(t1[u], fs, WB_SURF, eps, t2);
            for(int w = 0; w < n2; ++w)
            {
                if(full) wb_tet_accumulate(t2[w], o, dx, g, *A, hg ? aT : nullptr, hg ? &VT : nullptr);
                else V += wb_tet_volume(t2[w]);
            }
        }
        if(hg)
        {
            hg[25 * perm] = VT;
            for(int q = 0; q < 8; ++q) { hg[25 * perm + 1 + 3 * q] = aT[q].x; hg[25 * perm + 2 + 3 * q] = aT[q].y; hg[25 * perm + 3 + 3 * q] = aT[q].z; }
        }
    }
    return full ? A->V : V;
}

// nodal signed distance to the wall: min over the wall points of the particles in the 4 x 4 x 4 cells around the node of
// n.(x - X) (n points from the wall into the fluid). A node with particles nearby but no wall point is deep in the fluid
// (phi = +big); a node with NO particles within two cells is outside the fluid altogether (behind a wall or far above the
// surface: phi = -big) -- the kernel-spread deposits make such cells active, and they must not be mistaken for fluid.
// closest point on triangle ABC to P (Ericson, RTCD 5.1.5)
__device__ __forceinline__ double3 wb_pt_tri_cp(const double3 p, const double3 a, const double3 b, const double3 c)
{
    const double3 ab = wb_sub(b, a), ac = wb_sub(c, a), ap = wb_sub(p, a);
    const double d1 = wb_dot(ab, ap), d2 = wb_dot(ac, ap);
    double3 q;
    if(d1 <= 0.0 && d2 <= 0.0) q = a;
    else
    {
        const double3 bp = wb_sub(p, b);
        const double d3 = wb_dot(ab, bp), d4 = wb_dot(ac, bp);
        if(d3 >= 0.0 && d4 <= d3) q = b;
        else
        {
            const double vc = d1 * d4 - d3 * d2;
            if(vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) q = wb_add(a, wb_scale(ab, d1 / (d1 - d3)));
            else
            {
                const double3 cp = wb_sub(p, c);
                const double d5 = wb_dot(ab, cp), d6 = wb_dot(ac, cp);
                if(d6 >= 0.0 && d5 <= d6) q = c;
                else
                {
                    const double vb = d5 * d2 - d1 * d6;
                    if(vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) q = wb_add(a, wb_scale(ac, d2 / (d2 - d6)));
                    else
                    {
                        const double va = d3 * d6 - d5 * d4;
                        if(va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) q = wb_add(b, wb_scale(wb_sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
                        else
                        {
                            const double den = 1.0 / (va + vb + vc);
                            const double v = vb * den, w2 = vc * den;
                            q = wb_add(a, wb_add(wb_scale(ab, v), wb_scale(ac, w2)));
                        }
                    }
                }
            }
        }
    }
    return q;
}
// squared distance from P to triangle ABC
__device__ __forceinline__ double wb_pt_tri_d2(const double3 p, const double3 a, const double3 b, const double3 c)
{
    const double3 d = wb_sub(p, wb_pt_tri_cp(p, a, b, c));
    return wb_dot(d, d);
}
// does the segment x -> q cross a solid triangle (binned per cell)? crossings within ttol_abs of x are the node lying on the surface: ignored.
// Returns the distance from x to the crossing nearest q (the exposed surface between the node and the water), or -1.
__device__ __forceinline__ double wb_seg_hit(double3 x, double3 q, double3 org, double dx, int3 dims, const int* tri_start, const int* tri_list,
                                            const uint3* faces, const double3* svert, double ttol_abs)
{
    const double3 sd = wb_sub(q, x);
    const double  sl = sqrt(wb_dot(sd, sd));
    if(!(sl > 1e-12 * dx)) return -1.0;
    int c0[3], c1[3];
    const double lo3[3] = {fmin(x.x, q.x), fmin(x.y, q.y), fmin(x.z, q.z)}, hi3[3] = {fmax(x.x, q.x), fmax(x.y, q.y), fmax(x.z, q.z)};
    const double o3[3] = {org.x, org.y, org.z};
    const int    n3[3] = {dims.x, dims.y, dims.z};
    for(int ax = 0; ax < 3; ++ax)
    {
        c0[ax] = max(0, min(n3[ax] - 1, (int)floor((lo3[ax] - o3[ax]) / dx - 1e-9)));
        c1[ax] = max(0, min(n3[ax] - 1, (int)floor((hi3[ax] - o3[ax]) / dx + 1e-9)));
    }
    const double ttol = ttol_abs / sl;
    double best = -1.0;
    for(int ci = c0[0]; ci <= c1[0]; ++ci)
        for(int cj = c0[1]; cj <= c1[1]; ++cj)
            for(int ck = c0[2]; ck <= c1[2]; ++ck)
            {
                const int cl = cell_lin(ci, cj, ck, dims);
                for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                {
                    const uint3   F  = faces[tri_list[e]];
                    const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                    const double3 e1 = wb_sub(B, A), e2 = wb_sub(C, A);
                    const double3 pv = wb_cross(sd, e2);
                    const double  det = wb_dot(e1, pv);
                    if(fabs(det) < 1e-300) continue;
                    const double  inv = 1.0 / det;
                    const double3 tv = wb_sub(x, A);
                    const double  u  = wb_dot(tv, pv) * inv;
                    if(u < 0.0 || u > 1.0) continue;
                    const double3 qv = wb_cross(tv, e1);
                    const double  v  = wb_dot(sd, qv) * inv;
                    if(v < 0.0 || u + v > 1.0) continue;
                    const double  tt = wb_dot(e2, qv) * inv;
                    if(tt > ttol && tt <= 1.0) best = fmax(best, tt * sl);
                }
            }
    return best;
}
// ---- LIQ_WB_PHI_SIGN=9: connectivity of the open space (which water a node belongs to) ----
// segment x -> q crosses a solid strictly between its ends (either end may lie on a surface)
__device__ __forceinline__ bool wb_seg_cross(double3 x, double3 q, double3 org, double dx, int3 dims, const int* tri_start, const int* tri_list,
                                             const uint3* faces, const double3* svert, double ttol_abs)
{
    const double3 sd = wb_sub(q, x);
    const double  sl = sqrt(wb_dot(sd, sd));
    if(!(sl > 1e-12 * dx)) return false;
    int c0[3], c1[3];
    const double lo3[3] = {fmin(x.x, q.x), fmin(x.y, q.y), fmin(x.z, q.z)}, hi3[3] = {fmax(x.x, q.x), fmax(x.y, q.y), fmax(x.z, q.z)};
    const double o3[3] = {org.x, org.y, org.z};
    const int    n3[3] = {dims.x, dims.y, dims.z};
    for(int ax = 0; ax < 3; ++ax)
    {
        c0[ax] = max(0, min(n3[ax] - 1, (int)floor((lo3[ax] - o3[ax]) / dx - 1e-9)));
        c1[ax] = max(0, min(n3[ax] - 1, (int)floor((hi3[ax] - o3[ax]) / dx + 1e-9)));
    }
    const double ttol = ttol_abs / sl;
    for(int ci = c0[0]; ci <= c1[0]; ++ci)
        for(int cj = c0[1]; cj <= c1[1]; ++cj)
            for(int ck = c0[2]; ck <= c1[2]; ++ck)
            {
                const int cl = cell_lin(ci, cj, ck, dims);
                for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                {
                    const uint3   F  = faces[tri_list[e]];
                    const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                    const double3 e1 = wb_sub(B, A), e2 = wb_sub(C, A);
                    const double3 pv = wb_cross(sd, e2);
                    const double  det = wb_dot(e1, pv);
                    if(fabs(det) < 1e-300) continue;
                    const double  inv = 1.0 / det;
                    const double3 tv = wb_sub(x, A);
                    const double  u  = wb_dot(tv, pv) * inv;
                    if(u < 0.0 || u > 1.0) continue;
                    const double3 qv = wb_cross(tv, e1);
                    const double  v  = wb_dot(sd, qv) * inv;
                    if(v < 0.0 || u + v > 1.0) continue;
                    const double  tt = wb_dot(e2, qv) * inv;
                    if(tt > ttol && tt < 1.0 - ttol) return true;
                }
            }
    return false;
}
// LIQ_WB_PHI_SIGN=11: is the wall at this wall point thin? Ray from just inside the surface along -n (into the solid): an exit face within
// T = thin x dx -> thin. Only a wall thinner than about a cell can have fluid corners of ONE cell on both of its sides (the cross-talk the
// rule-3 dead zone prevents); a thicker wall (the 10 mm floor slab around an orifice, the box walls) separates the cells by itself.
__global__ void k_wb_thin(int np, const int* wp_valid, const int* wp_near, const double3* wp_X, const double3* wp_n, const int* wp_tri, double3 org,
                          double dx, int3 dims, const int* tri_start, const int* tri_list, const uint3* faces, const double3* svert, double T, int* thin)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= np) return;
    thin[p] = 0;
    if(!(wp_valid[p] || wp_near[p]) || wp_tri[p] < 0) return;   // no wall point, or the ground plane (a thick floor)
    const double3 n  = wp_n[p];
    const double3 x0 = wb_sub(wp_X[p], wb_scale(n, 1e-7 * dx));
    const double3 x1 = wb_sub(wp_X[p], wb_scale(n, T));
    thin[p] = wb_seg_cross(x0, x1, org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx) ? 1 : 0;
}
// open-space nodes: the node sees its nearest particle (no solid in between) -> lab = its own id, else -1 (inside a solid / no water near)
__global__ void k_wb_open(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const int* cstart, const int* cend, const int* pidx,
                          const double3* xp, const int* tri_start, const int* tri_list, const uint3* faces, const double3* svert, int* lab)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 x = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    int pnear = -1; double pnd2 = 1e300;
    for(int a = -2; a <= 1; ++a)
        for(int b = -2; b <= 1; ++b)
            for(int c = -2; c <= 1; ++c)
            {
                const int ci = i + a, cj = j + b, ck = k + c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                const int cl = cell_lin(ci, cj, ck, dims);
                const int st = cstart[cl];
                if(st < 0) continue;
                for(int e = st; e < cend[cl]; ++e)
                {
                    const int     p  = pidx[e];
                    const double3 rp = wb_sub(xp[p], x);
                    const double  dp = wb_dot(rp, rp);
                    if(dp < pnd2) { pnd2 = dp; pnear = p; }
                }
            }
    if(pnear < 0) { lab[n] = -1; return; }
    lab[n] = (wb_seg_hit(x, xp[pnear], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx) < 0.0) ? n : -1;
}
// grid edges between open nodes that cross a solid: bit a of blk[n] = the edge n -> n + e_a is blocked
__global__ void k_wb_edges(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const int* lab, const int* tri_start, const int* tri_list,
                           const uint3* faces, const double3* svert, unsigned char* blk)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    blk[n] = 0;
    if(lab[n] < 0) return;
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 x = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    unsigned char b = 0;
    for(int a = 0; a < 3; ++a)
    {
        const int ii = i + (a == 0), jj = j + (a == 1), kk = k + (a == 2);
        if(ii > dims.x || jj > dims.y || kk > dims.z) continue;
        const int m = node_lin(ii, jj, kk, dims);
        if(lab[m] < 0) continue;
        const double3 y = make_double3(org.x + (double)ii * dx, org.y + (double)jj * dx, org.z + (double)kk * dx);
        if(wb_seg_cross(x, y, org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx)) b |= (unsigned char)(1 << a);
    }
    blk[n] = b;
}
// one sweep of min-label propagation over unblocked edges (+ pointer jumping); *changed = 1 if any label dropped
__global__ void k_wb_cc_iter(int nact, const int* active_nodes, int3 dims, const unsigned char* blk, int* lab, int* changed)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    const int L = lab[n];
    if(L < 0) return;
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    int m = L;
    const int LL = lab[L];
    if(LL >= 0 && LL < m) m = LL;
    for(int a = 0; a < 3; ++a)
    {
        const int ip = i + (a == 0), jp = j + (a == 1), kp = k + (a == 2);
        if(ip <= dims.x && jp <= dims.y && kp <= dims.z && !((blk[n] >> a) & 1))
        {
            const int l = lab[node_lin(ip, jp, kp, dims)];
            if(l >= 0 && l < m) m = l;
        }
        const int im = i - (a == 0), jm = j - (a == 1), km = k - (a == 2);
        if(im >= 0 && jm >= 0 && km >= 0)
        {
            const int nb = node_lin(im, jm, km, dims);
            const int l  = lab[nb];
            if(l >= 0 && !((blk[nb] >> a) & 1) && l < m) m = l;
        }
    }
    if(m < L) { atomicMin(&lab[n], m); *changed = 1; }
}
// the water a particle belongs to: the label of the nearest open corner of its cell that it sees (-1: none)
__global__ void k_wb_pcomp(int np, const double3* xp, double3 org, double dx, int3 dims, const int* lab, const int* tri_start, const int* tri_list,
                           const uint3* faces, const double3* svert, int* pcomp)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= np) return;
    const double3 x = xp[p];
    const int i = max(0, min(dims.x - 1, (int)floor((x.x - org.x) / dx))), j = max(0, min(dims.y - 1, (int)floor((x.y - org.y) / dx))),
              k = max(0, min(dims.z - 1, (int)floor((x.z - org.z) / dx)));
    int best = -1; double bd = 1e300;
    for(int q = 0; q < 8; ++q)
    {
        const int ii = i + ((q >> 2) & 1), jj = j + ((q >> 1) & 1), kk = k + (q & 1);
        const int l = lab[node_lin(ii, jj, kk, dims)];
        if(l < 0) continue;
        const double3 y = make_double3(org.x + (double)ii * dx, org.y + (double)jj * dx, org.z + (double)kk * dx);
        const double3 r = wb_sub(y, x);
        const double  d2 = wb_dot(r, r);
        if(d2 >= bd) continue;
        if(wb_seg_cross(x, y, org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx)) continue;
        bd = d2; best = l;
    }
    pcomp[p] = best;
}
__global__ void k_wb_phi(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const int* cstart, const int* cend, const int* pidx,
                         const int* wp_valid, const int* wp_near, const double3* wp_X, const double3* wp_n, double lat2, double* phi_node, double3* nrm_node, int* pmin_node,
                         int euclid = 0, double hole_lat2 = 0.0, const int* wp_tri = nullptr, const uint3* faces = nullptr, const double3* svert = nullptr,
                         int sign_mode = 1, int tri_dist = 0, double dtol = 0.0, int* frz = nullptr, int freeze_now = 0,
                         const double3* xp = nullptr, const int* tri_start = nullptr, const int* tri_list = nullptr,
                         double ghost = 0.0,   // LIQ_WB_PHI_GHOST (x dx, mode 7): the zero level sits ghost inside the solid
                         int ghost_side = 0,   // LIQ_WB_PHI_GHOST_SIDE: 0 every node, 1 fluid-side nodes only (behind nodes keep -depth)
                         const int* lab = nullptr, const int* pcomp = nullptr,   // LIQ_WB_PHI_SIGN=9: open-space labels (nodes), water labels (particles)
                         const int* wthin = nullptr,   // LIQ_WB_PHI_SIGN=11: per wall point, the wall is thin (k_wb_thin)
                         double vis_r = 0.0)           // LIQ_WB_VIS_R (x dx, modes 8+): the visibility test only where the nearest wall point is farther
{   // frz / freeze_now (LIQ_WB_PHI_FREEZE, diagnostic for static containers): a node frozen at the freeze step keeps its phi / normal from then on   // dtol (LIQ_WB_PHI_DTOL x dx): a node within dtol of a wall point's plane is ON the wall, not behind it. Without it the side of a node that
    // lies exactly on a lattice-aligned wall (org + i dx vs the projected wall point) is decided by 1-ulp roundoff, and a node "behind" gets
    // |phi| = the distance to the nearest (sparse) wall point -- up to ~0.3 dx on a floor-wall edge: the corner cut cell lost 22 % of its volume
    // and the area vector of its inner corner (step-0 hydrostatic probe: 5.7 node weights toward the wall), flickering as particles move.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    if(frz && frz[n]) return;
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 x = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    double  phi = 1e30;
    double3 nrm = make_double3(0, 1, 0);
    int     pmin = -1;
    bool    any = false;
    double  ebest = 1e30; double3 ebn = make_double3(0, 1, 0); int ebp = -1; bool ebeh = false; double ebd = 0.0;   // euclid mode
    double  cbest = 1e30; bool cneg = false;   // LIQ_WB_PHI_SIGN=2: nearest wall point whose triangle contains the node's projection
    int     pnear = -1; double pnd2 = 1e300;     // LIQ_WB_PHI_SIGN=5: nearest particle
    double  pdep = 0.0;                          // LIQ_WB_PHI_SIGN=7: deepest behind-plane depth
    for(int a = -2; a <= 1; ++a)
        for(int b = -2; b <= 1; ++b)
            for(int c = -2; c <= 1; ++c)
            {
                const int ci = i + a, cj = j + b, ck = k + c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                const int cl = cell_lin(ci, cj, ck, dims);
                const int st = cstart[cl];
                if(st < 0) continue;
                const int en = cend[cl];
                if(en > st) any = true;
                for(int e = st; e < en; ++e)
                {
                    const int p = pidx[e];
                    if((sign_mode == 5 || sign_mode >= 7) && xp)
                    {   // nearest particle (any) for the visibility rule
                        const double3 rp = wb_sub(xp[p], x);
                        const double  dp = wb_dot(rp, rp);
                        if(dp < pnd2) { pnd2 = dp; pnear = p; }
                    }
                    if(!(wp_valid[p] || wp_near[p])) continue;
                    const double3 r = wb_sub(x, wp_X[p]);
                    double        d = wb_dot(wp_n[p], r);
                    if(euclid)
                    {   // LIQ_WB_PHI_EUCLID: |phi| = Euclidean distance to the NEAREST wall point (not to its plane): a node over a hole or past a
                        // wall's edge is as far as the nearest wall point really is. Sign: behind the wall iff ANY nearby wall point has the node
                        // on its negative side within lateral hole_lat of it (a node under a hole is open space; a node under the floor next to a
                        // side wall is still behind the floor).
                        double        rn = sqrt(wb_dot(r, r));
                        if(tri_dist && wp_tri && faces && svert && wp_tri[p] >= 0)
                        {   // LIQ_WB_PHI_TRI: true distance to the wall point's triangle (a wall point is a sparse particle-projection sample:
                            // a node ON a flat wall is 0 away from the wall, not the in-plane offset to the nearest sample -- up to h/2)
                            const uint3 F = faces[wp_tri[p]];
                            rn = sqrt(wb_pt_tri_d2(x, svert[F.x], svert[F.y], svert[F.z]));
                        }
                        const double3 tl = wb_sub(r, wb_scale(wp_n[p], d));
                        if(hole_lat2 > 0.0 && d < -dtol && wb_dot(tl, tl) <= hole_lat2) ebeh = true;
                        if(sign_mode == 0) { if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; } continue; }
                        if(sign_mode == 3 || sign_mode == 8) { if(d < -dtol) ebeh = true; if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; } continue; }   // sti48: ANY wall point's plane
                        if(sign_mode == 11)
                        {   // LIQ_WB_PHI_SIGN=11: rule 3's any-plane dead zone only for THIN walls; a thick wall's plane never puts a node behind
                            // (a node inside a thick solid cannot see its nearest particle: the visibility test after the loop)
                            if(d < -dtol && wthin && wthin[p]) ebeh = true;
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 10)
                        {   // LIQ_WB_PHI_SIGN=10: a wall point's plane puts the node behind only if the node projects INSIDE that wall point's triangle
                            // (the face really covers it): a thin plate's far face covers the nodes next to it (the dead zone stays); an orifice's
                            // floor-top faces do not cover the nodes in the opening; an L-shaped tank's extended planes cover nothing. The outside
                            // of a thin-walled beaker's convex corner (covered by no inner face) is caught by the visibility test after the loop.
                            if(d < -dtol && wp_tri && faces && svert && wp_tri[p] >= 0)
                            {
                                const uint3   F = faces[wp_tri[p]];
                                const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                                const double3 ab = wb_sub(B, A), ac = wb_sub(C, A), ap = wb_sub(x, A);
                                const double  d00 = wb_dot(ab, ab), d01 = wb_dot(ab, ac), d11 = wb_dot(ac, ac), d20 = wb_dot(ap, ab), d21 = wb_dot(ap, ac);
                                const double  den = d00 * d11 - d01 * d01;
                                if(den > 1e-30)
                                {
                                    const double v = (d11 * d20 - d01 * d21) / den, ww = (d00 * d21 - d01 * d20) / den, u = 1.0 - v - ww;
                                    if(u >= -1e-6 && v >= -1e-6 && ww >= -1e-6) ebeh = true;
                                }
                            }
                            else if(d < -dtol && !(wp_tri && wp_tri[p] >= 0)) ebeh = true;   // the ground plane (no triangle): rule 3
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 9)
                        {   // a wall point's plane puts the node behind only if its particle is other water than the node's (or the node is no open
                            // space): a thin plate's far side (the dead zone) is other water; the floor around an orifice, the walls of an L-shaped
                            // tank, the bore of a dropper are the node's own water (connected through the opening)
                            if(d < -dtol && (!lab || !pcomp || lab[n] < 0 || pcomp[p] < 0 || pcomp[p] != lab[n])) ebeh = true;
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 7)
                        {   // LIQ_WB_PHI_SIGN=7: mode 6 behind-planes (thin-plate dead zone) + their depth for the magnitude; the visibility
                            // rule and the exact mesh distance are added after the loop
                            if(d < -dtol && xp && tri_start && tri_list && faces && svert)
                                if(wb_seg_hit(x, xp[p], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx) >= 0.0) { ebeh = true; pdep = fmax(pdep, -d); }
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 6)
                        {   // LIQ_WB_PHI_SIGN=6: rule 3, but a wall point puts the node behind its plane only if the node cannot see that wall
                            // point's particle (the segment node -> particle crosses a solid). A thin plate keeps both neighbouring node layers
                            // behind (the far face's particles are seen through the plate: the two-cell dead zone that stops cross-talk);
                            // a node in an orifice sees the floor's particles through the opening (open water); an L-shaped container's
                            // extended planes no longer cut into its own water.
                            if(!ebeh && d < -dtol && xp && tri_start && tri_list && faces && svert)
                                if(wb_seg_hit(x, xp[p], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx) >= 0.0) ebeh = true;
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 5) { if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; } continue; }   // sign from visibility (below)
                        if(sign_mode == 4)
                        {   // LIQ_WB_PHI_SIGN=4: rule 3 with a lateral gate on the BEHIND decision only (the distance still comes from every wall point):
                            // a wall point can put the node behind its plane only if the node projects within hole_lat of it. A node over an
                            // orifice projects onto the floor plane far from every floor wall point (they sit around the hole) -> open water;
                            // a node outside a thin wall projects onto the wetted face right where the wall points are -> behind, as before.
                            if(d < -dtol && wb_dot(tl, tl) <= hole_lat2) ebeh = true;
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(sign_mode == 2)
                        {   // LIQ_WB_PHI_SIGN=2: the sign comes from the NEAREST (plane distance) wall point whose triangle contains the node's
                            // projection; none -> open space (under a hole, past an outer edge). A thin wall with water on both sides: each side's
                            // nodes are nearest to their own face (positive), nodes inside the plate are behind both faces (negative). Rule 1
                            // ("any containing triangle with the node behind it") marked BOTH sides of a 1 mm plate as behind (m9 regression).
                            bool inside = true;
                            if(wp_tri && faces && svert && wp_tri[p] >= 0)
                            {
                                const uint3   F = faces[wp_tri[p]];
                                const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                                const double3 ab = wb_sub(B, A), ac = wb_sub(C, A), ap = wb_sub(x, A);
                                const double  d00 = wb_dot(ab, ab), d01 = wb_dot(ab, ac), d11 = wb_dot(ac, ac), d20 = wb_dot(ap, ab), d21 = wb_dot(ap, ac);
                                const double  den = d00 * d11 - d01 * d01;
                                if(den > 1e-30)
                                {
                                    const double v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den, u = 1.0 - v - w;
                                    inside = (u >= -0.05 && v >= -0.05 && w >= -0.05);
                                }
                            }
                            if(inside && fabs(d) < cbest) { cbest = fabs(d); cneg = d < -dtol; }
                            if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                            continue;
                        }
                        if(hole_lat2 <= 0.0 && d < -dtol && !ebeh)
                        {   // exact rule, over ALL nearby wall points: behind iff the node projects inside some wall point's triangle on its
                            // negative side (a hole has no triangle: open space; an outer corner is behind the floor OR the side wall)
                            bool inside = true;
                            if(wp_tri && faces && svert && wp_tri[p] >= 0)
                            {
                                const uint3   F = faces[wp_tri[p]];
                                const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                                const double3 ab = wb_sub(B, A), ac = wb_sub(C, A), ap = wb_sub(x, A);
                                const double  d00 = wb_dot(ab, ab), d01 = wb_dot(ab, ac), d11 = wb_dot(ac, ac), d20 = wb_dot(ap, ab), d21 = wb_dot(ap, ac);
                                const double  den = d00 * d11 - d01 * d01;
                                if(den > 1e-30)
                                {
                                    const double v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den, u = 1.0 - v - w;
                                    inside = (u >= -0.05 && v >= -0.05 && w >= -0.05);
                                }
                            }
                            if(inside) ebeh = true;
                        }
                        if(rn < ebest) { ebest = rn; ebn = wp_n[p]; ebp = p; ebd = d; }
                        continue;
                    }
                    if(lat2 > 0.0)
                    {   // lateral gate: the node must lie over this wall point (within lat of its foot); a node in a hole is not "on the wall"
                        const double3 tl = wb_sub(r, wb_scale(wp_n[p], d));
                        if(wb_dot(tl, tl) > lat2) continue;
                    }
                    if(d < phi) { phi = d; nrm = wp_n[p]; pmin = p; }
                }
            }
    if(euclid && sign_mode == 2) ebeh = (cbest < 1e29) && cneg;
    double hitd = -1.0;
    if(euclid && sign_mode == 5 && pnear >= 0 && tri_start && tri_list && faces && svert)
    {   // LIQ_WB_PHI_SIGN=5 (visibility): behind iff the segment from the node to its nearest particle crosses a solid triangle. Open space
        // under an orifice sees the water through the hole; the dry side of a thin wall, the inside of a thick body (a piston) and the
        // floor under overlapping wall boxes see it only through a solid. Orientation-free, no watertightness needed. A crossing within
        // 1e-9 dx of the node is the node lying on the surface: not behind (like LIQ_WB_PHI_DTOL).
        const double3 q  = xp[pnear];
        const double3 sd = wb_sub(q, x);
        const double  sl = sqrt(wb_dot(sd, sd));
        if(sl > 1e-12 * dx)
        {
            int c0[3], c1[3];
            const double lo3[3] = {fmin(x.x, q.x), fmin(x.y, q.y), fmin(x.z, q.z)}, hi3[3] = {fmax(x.x, q.x), fmax(x.y, q.y), fmax(x.z, q.z)};
            const double o3[3] = {org.x, org.y, org.z};
            const int    n3[3] = {dims.x, dims.y, dims.z};
            for(int ax = 0; ax < 3; ++ax)
            {
                c0[ax] = max(0, min(n3[ax] - 1, (int)floor((lo3[ax] - o3[ax]) / dx - 1e-9)));
                c1[ax] = max(0, min(n3[ax] - 1, (int)floor((hi3[ax] - o3[ax]) / dx + 1e-9)));
            }
            const double ttol = 1e-9 * dx / sl;
            for(int ci = c0[0]; ci <= c1[0] && hitd < 0.0; ++ci)
                for(int cj = c0[1]; cj <= c1[1] && hitd < 0.0; ++cj)
                    for(int ck = c0[2]; ck <= c1[2] && hitd < 0.0; ++ck)
                    {
                        const int cl = cell_lin(ci, cj, ck, dims);
                        for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                        {
                            const uint3   F  = faces[tri_list[e]];
                            const double3 A = svert[F.x], B = svert[F.y], C = svert[F.z];
                            const double3 e1 = wb_sub(B, A), e2 = wb_sub(C, A);
                            const double3 pv = wb_cross(sd, e2);
                            const double  det = wb_dot(e1, pv);
                            if(fabs(det) < 1e-300) continue;
                            const double  inv = 1.0 / det;
                            const double3 tv = wb_sub(x, A);
                            const double  u  = wb_dot(tv, pv) * inv;
                            if(u < 0.0 || u > 1.0) continue;
                            const double3 qv = wb_cross(tv, e1);
                            const double  v  = wb_dot(sd, qv) * inv;
                            if(v < 0.0 || u + v > 1.0) continue;
                            const double  tt = wb_dot(e2, qv) * inv;
                            if(tt > ttol && tt <= 1.0) { hitd = tt * sl; break; }
                        }
                    }
        }
        ebeh = hitd >= 0.0;
        if(ebeh && !(ebest < 1e29)) { phi = -fmax(hitd, 1e-9 * dx); }   // behind with no wall point in reach: depth >= the distance to the crossing
    }
    if(euclid && sign_mode == 0) ebeh = (ebest < 1e29) && ebd < -dtol;   // LIQ_WB_PHI_SIGN=0 (sti48): sign of the nearest wall point's plane
    if(euclid && sign_mode >= 8 && tri_start && tri_list && faces && svert)
    {   // (mode 9: the same magnitudes; "unseen" = not open space, the plane test above filtered by the water labels)   // LIQ_WB_PHI_SIGN=8: rule 3 where its particle sampling is adequate, the mesh where it is not. Rule 3's distance to the nearest wall point
        // is the true distance plus the lateral offset to the nearest sample (<= h/sqrt2 = 0.354 dx on the natural lattice): an accidental ghost
        // sliver behind every wall that balances the on-wall nodes and keeps corners representable -- kept as is. Where no particle is within the
        // wall-point radius of the facing wall (flask axis-extreme columns: node 0.5 mm inside the glass read +1.44 dx, a free node with 4.47 W0),
        // (a) a node that cannot see its nearest particle (the segment crosses a solid) is behind, |phi| = its exact depth; (b) a fluid-side node's
        // distance is capped at exact + ghost (ghost = LIQ_WB_PHI_GHOST, the natural-lattice offset), which leaves well-sampled walls unchanged.
        double hit = -1.0;
        // LIQ_WB_VIS_R: visibility only where rule 3 lacks samples (the nearest wall point is more than vis_r dx away: the flask axis columns
        // read 1.44 dx); next to a well-sampled wall the visibility test only adds flicker (a segment that grazes a corner as particles move)
        if(!ebeh && pnear >= 0 && (vis_r <= 0.0 || !(ebest < vis_r * dx)))
        {
            if(sign_mode == 9 && lab) hit = (lab[n] < 0) ? fmax(wb_seg_hit(x, xp[pnear], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx), 0.0) : -1.0;
            else hit = wb_seg_hit(x, xp[pnear], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx);
        }
        double td2 = 1e300;
        for(int a = -2; a <= 1; ++a)
            for(int b = -2; b <= 1; ++b)
                for(int c = -2; c <= 1; ++c)
                {
                    const int ci = i + a, cj = j + b, ck = k + c;
                    if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                    const int cl = cell_lin(ci, cj, ck, dims);
                    for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                    {
                        const uint3 F = faces[tri_list[e]];
                        td2 = fmin(td2, wb_pt_tri_d2(x, svert[F.x], svert[F.y], svert[F.z]));
                    }
                }
        const double td = (td2 < 1e299) ? sqrt(td2) : -1.0;
        if(ebeh) { if(ebest < 1e29) phi = -ebest; }                                    // rule 3 behind: unchanged
        else if(hit >= 0.0) phi = -fmax((td >= 0.0) ? td : hit, 1e-9 * dx);            // (a) unseen: behind, exact depth
        else if(ebest < 1e29) phi = (td >= 0.0) ? fmin(ebest, td + ghost * dx) : ebest;  // (b) capped rule-3 distance
        else if(td >= 0.0) phi = td + ghost * dx;                                       // no wall point in reach, a wall is: its distance
        if(ebest < 1e29) { nrm = ebn; pmin = ebp; }
        phi_node[n] = any ? phi : -1e30;
        nrm_node[n] = nrm;
        pmin_node[n] = pmin;
        if(frz && freeze_now && any) frz[n] = 1;
        return;
    }
    if(euclid && sign_mode == 7 && tri_start && tri_list && faces && svert)
    {   // LIQ_WB_PHI_SIGN=7: behind = visibility (the node cannot see its nearest particle: inside a solid even where no wall point
        // reaches, e.g. glass where the seeding left a > 2.5 mm gap) OR the mode-6 shadow (thin-plate dead zone). Magnitude from the
        // mesh itself (no dependence on how close the particles happen to be): fluid side = exact distance to the nearest triangle;
        // behind = max(that distance, the deepest behind-plane), so a hidden face under an overlapping box cannot read 0.
        double hit5 = -1.0;
        if(pnear >= 0) hit5 = wb_seg_hit(x, xp[pnear], org, dx, dims, tri_start, tri_list, faces, svert, 1e-9 * dx);
        const bool beh = ebeh || hit5 >= 0.0;
        double td2 = 1e300;
        for(int a = -2; a <= 1; ++a)
            for(int b = -2; b <= 1; ++b)
                for(int c = -2; c <= 1; ++c)
                {
                    const int ci = i + a, cj = j + b, ck = k + c;
                    if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                    const int cl = cell_lin(ci, cj, ck, dims);
                    for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                    {
                        const uint3 F = faces[tri_list[e]];
                        td2 = fmin(td2, wb_pt_tri_d2(x, svert[F.x], svert[F.y], svert[F.z]));
                    }
                }
        const double td = (td2 < 1e299) ? sqrt(td2) : -1.0;
        if(!beh)
        {
            if(td >= 0.0) phi = td;
            else if(ebest < 1e29) phi = ebest;
        }
        else
        {
            double dep = fmax(fmax(td, 0.0), pdep);
            if(hit5 >= 0.0) dep = fmax(dep, (td >= 0.0) ? td : hit5);
            phi = -fmax(dep, 1e-9 * dx);
        }
        // LIQ_WB_PHI_GHOST: the exact field puts every on-wall node at phi = 0 -- (a) the cells behind a wall have no fluid, so the wall
        // nodes carry the one-sided pressure on the band penalty (aligned box floor: -11.5 W0, 2 um, ~80 Pa of J error), and (b) a node on
        // the floor masks an oblique side wall along the floor-level edges (min-distance field: the floor face of the corner cut cell
        // collapses; yawed box, step 0: 2.9-5.7 W0 on free nodes one layer up). Rule 3's particle-sampled distance hides both by accident
        // (on-wall nodes read the lateral offset to the nearest wall point, 0.354 dx on the natural lattice). Here: the fluid region dilated
        // by ghost x dx into the solids, uniformly (phi + ghost): behind-wall slivers carry the wall pressure, corners round off.
        // LIQ_WB_PHI_GHOST_SIDE=1: only the fluid side moves up -- a thin plate's dead zone (behind nodes on both sides) keeps its depth, so
        // the two dilated sides cannot meet inside a 1 mm plate (m9 divider, ghost on every node: 96 crossings).
        if(ghost > 0.0 && fabs(phi) < 1e29 && (ghost_side == 0 || phi >= 0.0)) phi += ghost * dx;
        if(ebest < 1e29) { nrm = ebn; pmin = ebp; }
        phi_node[n] = any ? phi : -1e30;
        nrm_node[n] = nrm;
        pmin_node[n] = pmin;
        if(frz && freeze_now && any) frz[n] = 1;
        return;
    }
    if(euclid && ebest < 1e29) { phi = ebeh ? -ebest : ebest; nrm = ebn; pmin = ebp; }
    phi_node[n] = any ? phi : -1e30;
    nrm_node[n] = nrm;
    pmin_node[n] = pmin;
    if(frz && freeze_now && any) frz[n] = 1;
}

// LIQ_WB_NRM_EXACT: the wall normal of a band node (|phi| <= r) from the mesh itself -- the direction from the closest point on the
// solid surface to the node (a node on the surface: that triangle's normal). The nearest wall point's normal is the normal of whichever
// facet that particle projected onto: next to a polygonal curved wall (flask axis columns: the facing facet carries no wall point,
// the neighbours sit at +-16 deg) the band's "normal" penalty then takes a tangential share of the one-sided pressure. Only the line
// matters (the penalty and the velocity projection are even in n).
__global__ void k_wb_nrm_exact(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const double* phi_node, double r,
                               const int* tri_start, const int* tri_list, const uint3* faces, const double3* svert, double3* nrm_node, int* counter)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    const double ph = phi_node[n];
    if(!(fabs(ph) <= r)) return;
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 x = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    double  best = 1e300;
    double3 bcp = x;
    int     bf = -1;
    for(int a = -2; a <= 1; ++a)
        for(int b = -2; b <= 1; ++b)
            for(int c = -2; c <= 1; ++c)
            {
                const int ci = i + a, cj = j + b, ck = k + c;
                if(ci < 0 || cj < 0 || ck < 0 || ci >= dims.x || cj >= dims.y || ck >= dims.z) continue;
                const int cl = cell_lin(ci, cj, ck, dims);
                for(int e = tri_start[cl]; e < tri_start[cl + 1]; ++e)
                {
                    const uint3   F  = faces[tri_list[e]];
                    const double3 cp = wb_pt_tri_cp(x, svert[F.x], svert[F.y], svert[F.z]);
                    const double3 d  = wb_sub(x, cp);
                    const double  d2 = wb_dot(d, d);
                    if(d2 < best) { best = d2; bcp = cp; bf = tri_list[e]; }
                }
            }
    if(bf < 0) return;
    double3 nn;
    const double L = sqrt(best);
    if(L > 1e-6 * dx) nn = wb_scale(wb_sub(x, bcp), 1.0 / L);
    else
    {
        const uint3   F = faces[bf];
        const double3 cr = wb_cross(wb_sub(svert[F.y], svert[F.x]), wb_sub(svert[F.z], svert[F.x]));
        const double  cl = sqrt(wb_dot(cr, cr));
        if(!(cl > 0.0)) return;
        nn = wb_scale(cr, 1.0 / cl);
    }
    if(wb_dot(nn, nrm_node[n]) < 0.0) nn = wb_scale(nn, -1.0);   // keep the fluid-facing orientation of the wall-point normal
    nrm_node[n] = nn;
    if(counter) atomicAdd(counter, 1);
}

// slip wall BC for the nodes on / behind the wall: normal displacement tied to the wall (static container: its own position),
// implicit Euler under BDF2 (bh_wall). The volume-change operator alone gives these nodes no normal stiffness (their area vectors
// are tangential), so without this the fluid drifts through the wall.
// moving containers: the wall point that set the node's phi moves within the step from its position on the step-start solid
// (o_vertexes) to the current one (a teleported body jumps there at the step start, a dynamic one moves during the solve);
// the node's target follows that displacement so the wall layer moves with the wall instead of pinning it to the lattice.
// Classification: phi < -thr (behind the wall) -> ghost_mode (slip / fixed); |phi| <= thr (on the wall) -> slip; thr < phi <= near*dx
// with a geometric fluid share below wmin (a sliver corner: it barely represents fluid, yet it is free and nearly massless -- the
// cut-cell "small cell problem") -> fixed. Everything else is free fluid.
__global__ void k_wb_slip_mark(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const double* phi_node, const double3* nrm_node, double thr, double thr_mov, double mov_eps2,
                               int ghost_mode, const int* pmin_node, const int* wp_tri, const double3* wp_b, const uint3* faces, const double3* vert,
                               const double3* overt, const double3* svel, double dt, const double* node_w, double wmin, double near, double mfloor, double* node_mfl,
                               const int* agroot, const int* agtype, int* wall_t, double3* wall_n, double3* wall_xt, int* bh_wall, int* counter,
                               int stiff_static = 0,   // LIQ_WALL_STATIC_MASS: bit 4 = a static wall's node (stiffer penalty in k_wall_assemble)
                               int mark_static = 0)    // LIQ_WALL_TANG < 1: bit 8 = static wall, bit 16 = behind the wall (tangential multiplier)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    const double ph = phi_node[n];
    if(ph < -1e29) return;   // -big = no particles anywhere near: not a wall node, just empty
    if(agroot && agroot[n] >= 0 && agtype[n] == 0) { if(bh_wall) bh_wall[n] = 1; return; }   // fully aggregated: tied to its root cell, not to the wall (implicit Euler like a wall node)
    // the wall this node belongs to: its displacement over the step (moving container) decides the band thickness --
    // a lattice-fixed band can only follow a wall that stands still; a moving wall keeps the thin on-wall layer (thr_mov)
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    double3 dX = make_double3(0, 0, 0);
    const int p = pmin_node[n];
    if(p >= 0 && overt && wp_tri[p] >= 0)
    {
        const uint3   F = faces[wp_tri[p]];
        const double3 b = wp_b[p];
        dX = wb_add(wb_add(wb_scale(wb_sub(vert[F.x], overt[F.x]), b.x), wb_scale(wb_sub(vert[F.y], overt[F.y]), b.y)), wb_scale(wb_sub(vert[F.z], overt[F.z]), b.z));
        if(svel)   // a dynamic body moves during the solve: predicted displacement dt v (a teleported body has v = 0 and the jump above)
        {
            const double3 vw = wb_add(wb_add(wb_scale(svel[F.x], b.x), wb_scale(svel[F.y], b.y)), wb_scale(svel[F.z], b.z));
            dX = wb_add(dX, wb_scale(vw, dt));
        }
    }
    const bool moving = wb_dot(dX, dX) > mov_eps2;   // a static ABD container still jitters by ~1e-9 m/step under the water: needs a physical threshold
    if(moving && thr_mov >= 0.0) thr = thr_mov;
    const bool sliver = ph > thr && ph <= near && node_w && node_w[n] < wmin;
    if(ph > thr && ph <= near && mfloor > 0.0 && node_mfl) node_mfl[n] = mfloor;   // free near-wall node: extra inertia (LIQ_WB_MFLOOR)
    if(ph > thr && !sliver) return;
    wall_t[s]  = ((ghost_mode == 3 || sliver) ? 2 : ((ph < -thr) ? ghost_mode : 1))   // behind the wall: slip or fixed; on the wall: slip; sliver: fixed (mode 3: everything fixed)
                 | ((stiff_static && !moving && (stiff_static == 1 || ph < 0.0)) ? 4 : 0)    // 2: only nodes behind the wall (no fluid on their far side)
                 | ((mark_static && !moving) ? 8 : 0) | ((mark_static && ph < 0.0) ? 16 : 0);
    wall_n[s]  = nrm_node[n];
    wall_xt[s] = make_double3(org.x + (double)i * dx + dX.x, org.y + (double)j * dx + dX.y, org.z + (double)k * dx + dX.z);
    if(bh_wall) bh_wall[n] = 1;
    atomicAdd(counter, 1);
}
__global__ void k_wb_slip_project(int nact, int base, const int* wall_t, const double3* wall_n, double3* vel, int tang_free = 0)
{   // tang_free (LIQ_WALL_TANG < 1): a static wall's fluid-side band node keeps its tangential velocity (the penalty decides how much it slips)
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact || (wall_t[s] & 3) == 0) return;
    const bool tf = tang_free && (wall_t[s] & 8) && !(wall_t[s] & 16);
    if((wall_t[s] & 3) == 2 && !tf) { vel[base + s] = make_double3(0, 0, 0); return; }   // fixed ghost (static container)
    const double3 n = wall_n[s];
    double3       v = vel[base + s];
    const double  vn = wb_dot(v, n);
    vel[base + s] = make_double3(v.x - vn * n.x, v.y - vn * n.y, v.z - vn * n.z);
}

// ---- per-cell geometry pass: fluid-side flag and geometric fluid volume (no surface cut), dense per cell ----
// flag 0 = inactive, 1 = has fluid-side corners, 2 = every corner outside (behind a wall / far from any particle)
__global__ void k_wb_cellgeo(int nact, const int* active_cells, int3 dims, double3 org, double dx, const double* phi_node, double3 up, double3 g,
                             int* cflag, double* cvgeo)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    const int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    double phi[8];
    int    nneg = 0;
    for(int q = 0; q < 8; ++q)
    {
        phi[q] = phi_node[node_lin(i + ((q >> 2) & 1), j + ((q >> 1) & 1), k + (q & 1), dims)];
        if(phi[q] < 0.0) ++nneg;
    }
    if(nneg == 8) { cflag[c] = 2; cvgeo[c] = 0.0; return; }
    cflag[c] = 1;
    cvgeo[c] = (nneg > 0) ? wb_region(i, j, k, org, dx, phi, up, 1e300, g, false, nullptr) : dx * dx * dx;
}

// ---- per-column pass: the water body standing on the floor of each lattice column (along the axis closest to "up") ----
// Scan from the bottom: the first active cell starts the body; the body is the run of contiguous active cells; its deposited
// volume D (every cell of the run, including the kernel spill into the floor and above the surface) fills its geometric fluid
// volume from the bottom: h_col is where the cumulative geometric volume reaches D (bisection inside the cell that contains
// it). Cells of the run above h_col are kernel spill (dead); cells above the run (after a gap) are separate water
// (a pouring stream, a droplet) and keep the current operator with their deposited weight. h_col is continuous in the
// particle positions: no threshold can flip a cell's state.
__global__ void k_wb_column(int ncol, int3 dims, double3 org, double dx, int U, int sgn, const int* cflag, const double* cvgeo, const double* Vc,
                            const double* phi_node, double3 up, double3 g, const double3* xp, const int* cstart, const int* cend, const int* pidx, double hs,
                            int topk, double* hcol, int* body_lo, int* body_hi,
                            int hmode = 0, double hwin = 0.5,   // LIQ_WB_HMODE: 0 = mean of the top-k, 1 = top-K with K = the column's own particles per layer, 2 = top-layer window
                            double colmin = 0.0,                // LIQ_WB_COLMIN: a column whose deposit is below colmin x its geometric volume under h_col is not a body
                            double* hwt = nullptr,              // LIQ_WB_HSMOOTH_W: per-column smoothing weight = the column's particles per layer
                            double* colD = nullptr, double* colG = nullptr,   // LIQ_WB_COLMIN_POOL: store deposit / geometric volume, guard in k_wb_colguard
                            int hfreeze = 0,                    // LIQ_WB_HCOL_FREEZE (diagnostic): keep each column's previous raw height
                            int hwmode = 1,                     // LIQ_WB_HSMOOTH_W: 1 = particles per layer, 2 = the column's particle count
                            double capw = 0.0,                  // LIQ_WB_CAP: a column whose water touches a wall above it has no free surface
                            int* ovh = nullptr, double ovh_r = 0.0, double ovh_c = 0.1,   // LIQ_WB_OVH: overhang-limited column flag
                            const int* tri_start = nullptr, const int* tri_list = nullptr, const uint3* faces = nullptr, const double3* svert = nullptr,
                            double trim = 0.0)                  // LIQ_WB_TRIM: bottom cells of the run filled below trim x their geometric volume are not the body
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= ncol) return;
    // decode the column: the two axes other than U
    int n0, n1, nU;
    if(U == 0) { n0 = dims.y; n1 = dims.z; nU = dims.x; }
    else if(U == 1) { n0 = dims.x; n1 = dims.z; nU = dims.y; }
    else { n0 = dims.x; n1 = dims.y; nU = dims.z; }
    const int a = col / n1, b = col % n1;
    auto cell_of = [&](int t, int& i, int& j, int& k) {
        const int u = (sgn > 0) ? t : (nU - 1 - t);
        if(U == 0) { i = u; j = a; k = b; }
        else if(U == 1) { i = a; j = u; k = b; }
        else { i = a; j = b; k = u; }
    };
    const double hprev = hcol[col];
    hcol[col] = -1e300; body_lo[col] = -1; body_hi[col] = -1;
    if(ovh) ovh[col] = 0;
    if(hwt) hwt[col] = 0.0;
    if(colD) { colD[col] = 0.0; colG[col] = 0.0; }
    // runs of contiguous active cells; the body is the run with the most deposited volume (a stray run of kernel spill
    // below the floor, or a droplet above, must not be taken for the body)
    int    lo = -1, hi = -1, rlo = -1;
    double D = 0.0, rD = 0.0;
    for(int t = 0; t <= nU; ++t)
    {
        bool active = false;
        if(t < nU)
        {
            int i, j, k;
            cell_of(t, i, j, k);
            const int c = cell_lin(i, j, k, dims);
            active = cflag[c] != 0 && Vc[c] > 0.0;
            if(active)
            {
                if(rlo < 0) { rlo = t; rD = 0.0; }
                rD += Vc[c];
            }
        }
        if(!active && rlo >= 0)
        {   // a run ended at t - 1
            if(rD > D) { D = rD; lo = rlo; hi = t - 1; }
            rlo = -1;
        }
    }
    if(lo < 0) return;
    if(trim > 0.0)
    {   // LIQ_WB_TRIM: a hydrostatic body stands on something. The run of an orifice column continues through the opening into the jet
        // below (cells 5-7 % filled, geometric volume a full cell): counted as the body's bottom, the jet becomes a full water column
        // (weight x15, hydrostatic pressure on its particles). Trim the run's bottom while a cell is filled below trim x its geometric
        // fluid volume -- a cell wholly behind a wall (geometric volume 0: the kernel spill into a container floor) stops the trim, so a
        // resting body keeps its bottom; the trimmed cells are separate water (current operator, deposited weight).
        while(lo < hi)
        {
            int i, j, k;
            cell_of(lo, i, j, k);
            const int c = cell_lin(i, j, k, dims);
            if(!(cvgeo[c] > 0.0 && Vc[c] < trim * cvgeo[c])) break;
            ++lo;
        }
    }
    body_lo[col] = lo; body_hi[col] = hi;
    // the free surface is where the particles end: the mean height of the top-k particles of the body's cells (the top
    // layer: k = particles per layer per column) + half a particle spacing. (The deposited volume is biased low next to
    // walls -- particles stay ~h/2 off the wall and the kernel spills part of their volume across it -- so a volume-based
    // height would step down along every wall; the single topmost particle jitters.) Capped at the body's top.
    double top[8];
    int    ntop = 0;
    const int K = (hmode != 0) ? 8 : ((topk < 1) ? 1 : (topk > 8 ? 8 : topk));   // modes 1/2 keep the 8 highest, then choose
    double htop = -1e300;
    int    npart = 0;
    double hbot  = 1e300;
    for(int t = lo; t <= hi; ++t)
    {
        int i, j, k;
        cell_of(t, i, j, k);
        const int c = cell_lin(i, j, k, dims);
        for(int q = 0; q < 8; ++q)
        {
            const double3 pq = make_double3(org.x + (double)(i + ((q >> 2) & 1)) * dx, org.y + (double)(j + ((q >> 1) & 1)) * dx, org.z + (double)(k + (q & 1)) * dx);
            htop = fmax(htop, wb_dot(up, pq));
        }
        const int st = cstart[c];
        if(st < 0) continue;
        for(int e = st; e < cend[c]; ++e)
        {   // keep the K largest heights (insertion into a tiny sorted array)
            const double hp = wb_dot(up, xp[pidx[e]]);
            ++npart; hbot = fmin(hbot, hp);
            if(ntop < K) { int m = ntop++; while(m > 0 && top[m - 1] < hp) { top[m] = top[m - 1]; --m; } top[m] = hp; }
            else if(hp > top[K - 1]) { int m = K - 1; while(m > 0 && top[m - 1] < hp) { top[m] = top[m - 1]; --m; } top[m] = hp; }
        }
    }
    double hmean = 0.0;
    if(hmode == 1 && ntop > 0)
    {   // the column's own particles per layer (a wall column holds fewer than an interior one: the fixed top-k then reaches into the
        // second layer and reads the surface up to h/2 low) -- fractional K, continuous in the particle heights
        const double nl = (top[0] - hbot) / hs + 1.0;
        const double Ke = fmin(fmax((double)npart / nl, 1.0), (double)ntop);
        const int    f  = (int)floor(Ke);
        const double fr = Ke - (double)f;
        for(int m = 0; m < f; ++m) hmean += top[m];
        if(f < ntop) hmean += fr * top[f];
        hmean /= Ke;
    }
    else if(hmode == 2 && ntop > 0)
    {   // top-layer window: every particle within hwin * h of the column's topmost
        int nw = 0;
        for(int m = 0; m < ntop; ++m) if(top[m] > top[0] - hwin * hs) { hmean += top[m]; ++nw; }
        hmean /= (double)nw;
    }
    else if(hmode == 3 && ntop > 0)
    {   // soft top-layer window: weight 1 - (y_top - y) / (hwin h), clamped at 0 -- continuous in every particle height (the hard window
        // jumps when a particle crosses its edge); exact on a lattice for hwin <= 1 (the second layer, h below, gets weight 0)
        double ws = 0.0;
        for(int m = 0; m < ntop; ++m)
        {
            const double wq = 1.0 - (top[0] - top[m]) / (hwin * hs);
            if(wq > 0.0) { hmean += wq * top[m]; ws += wq; }
        }
        hmean /= ws;
    }
    else
    {
        for(int m = 0; m < ntop; ++m) hmean += top[m];
        if(ntop > 0) hmean /= ntop;
    }
    hcol[col] = (ntop > 0) ? fmin(hmean + 0.5 * hs, htop) : -1e300;
    if(hfreeze && hprev > -1e299 && ntop > 0) hcol[col] = hprev;
    if(capw > 0.0 && ntop > 0)
    {   // LIQ_WB_CAP: under an overhanging wall (the cone of an Erlenmeyer flask) the water of a column is bounded by the wall, not by a
        // free surface. The particle-top estimate then puts a zero-pressure surface just under the glass, sloping down towards the wall, where
        // the true pressure is the hydrostatic one of the real free surface elsewhere -> a permanent circulation. Wall height above the top
        // particle: zero crossing of the nodal phi along the column's four vertical edges (up to 3 cells up); every fluid-side edge (at least
        // 2) meets the wall within capw * h of the top particle -> capped: no free surface in this column (h = +inf; the wall cut cells bound it).
        const double ytop = top[0];
        const int    bitU = (U == 0) ? 2 : (U == 1) ? 1 : 0;   // corner bit of the U axis in q
        const double xU   = (sgn > 0) ? ytop : -ytop;
        const double oU   = (U == 0) ? org.x : (U == 1) ? org.y : org.z;
        int idx = (int)floor((xU - oU) / dx);
        idx = max(0, min(nU - 1, idx));
        const int t0 = (sgn > 0) ? idx : (nU - 1 - idx);
        // an edge counts only if it is on the fluid side at the water level (phi >= 0 at its last node at or below the top particle):
        // an edge beside / behind a vertical wall is not "under" it. Capped iff >= 2 such edges and every one meets the wall above in reach.
        double hw[4]; int found[4] = {0, 0, 0, 0}, valid[4] = {0, 0, 0, 0}, seen[4] = {0, 0, 0, 0};
        for(int t = max(t0 - 1, 0); t <= min(t0 + 3, nU - 1); ++t)
        {
            int i, j, k;
            cell_of(t, i, j, k);
            for(int e = 0; e < 4; ++e)
            {
                if(found[e]) continue;
                int qb = 0, qt = 0, m = 0;
                for(int bit = 2; bit >= 0; --bit)
                {
                    if(bit == bitU) continue;
                    const int v = (e >> m) & 1; ++m;
                    qb |= v << bit; qt |= v << bit;
                }
                const int bb = (sgn > 0) ? 0 : 1;
                qb |= bb << bitU; qt |= (1 - bb) << bitU;
                const int nb = node_lin(i + ((qb >> 2) & 1), j + ((qb >> 1) & 1), k + (qb & 1), dims);
                const int nt = node_lin(i + ((qt >> 2) & 1), j + ((qt >> 1) & 1), k + (qt & 1), dims);
                const double3 pb = make_double3(org.x + (double)(i + ((qb >> 2) & 1)) * dx, org.y + (double)(j + ((qb >> 1) & 1)) * dx, org.z + (double)(k + (qb & 1)) * dx);
                const double3 pt = make_double3(org.x + (double)(i + ((qt >> 2) & 1)) * dx, org.y + (double)(j + ((qt >> 1) & 1)) * dx, org.z + (double)(k + (qt & 1)) * dx);
                const double hb = wb_dot(up, pb), ht = wb_dot(up, pt);
                const double fb = phi_node[nb], ft = phi_node[nt];
                if(fb < -1e29 || ft < -1e29) continue;
                if(hb <= ytop && ht > ytop) { seen[e] = 1; valid[e] = (fb >= 0.0); }   // the segment that holds the water level
                if(!valid[e] || ht <= ytop) continue;
                if(ft < 0.0)
                {
                    const double hz = (fb >= 0.0) ? hb + (ht - hb) * fb / (fb - ft) : hb;
                    hw[e] = fmax(hz, ytop); found[e] = 1;
                }
            }
        }
        int nv = 0, nf = 0; double hmax = -1e300;
        for(int e = 0; e < 4; ++e)
        {
            if(!seen[e] || !valid[e]) continue;
            ++nv;
            if(found[e]) { ++nf; hmax = fmax(hmax, hw[e]); }
        }
        if(nv >= 2 && nf == nv && hmax - ytop < capw * hs) hcol[col] = 1e300;
    }
    if(ovh && ntop > 0 && ovh_r > 0.0 && tri_start && tri_list && faces && svert)
    {   // LIQ_WB_OVH: the column's top is bounded by a wall above it (the cone of an Erlenmeyer flask, a bottle shoulder, a piston), not by a free
        // surface: its particle top (the highest lattice layer that keeps h/2 off the wall) is no surface height -- flask natural lattice: 84 of
        // 208 columns 2.5-27.5 mm low, and the steep cone (17 deg off vertical) puts the wall 4-5 h above the top particle along the column's
        // edges, out of reach of a vertical cap test (LIQ_WB_CAP: 0 of 84 at 3 h). Orientation-free: some solid triangle within ovh_r h of a
        // top-layer particle has its closest point above the particle ((x_p - q).up < -ovh_c |x_p - q|). The flagged column takes its height
        // from its unflagged neighbours (k_wb_hsmooth / k_wb_ovh_fill), never lower than its own estimate.
        const double R = ovh_r * hs;
        int flag = 0;
        for(int t = hi; t >= lo && !flag; --t)
        {
            int i, j, k;
            cell_of(t, i, j, k);
            const int c = cell_lin(i, j, k, dims);
            const double3 pc = make_double3(org.x + ((double)i + 0.5) * dx, org.y + ((double)j + 0.5) * dx, org.z + ((double)k + 0.5) * dx);
            if(wb_dot(up, pc) + 0.87 * dx < top[0] - hs) break;   // the cell lies entirely below the top layer
            const int st = cstart[c];
            if(st < 0) continue;
            for(int e = st; e < cend[c] && !flag; ++e)
            {
                const double3 xq = xp[pidx[e]];
                if(wb_dot(up, xq) <= top[0] - hs) continue;   // not in the top layer
                int c0[3], c1[3];
                const double xq3[3] = {xq.x, xq.y, xq.z}, o3[3] = {org.x, org.y, org.z};
                const int    n3[3] = {dims.x, dims.y, dims.z};
                for(int ax = 0; ax < 3; ++ax)
                {
                    c0[ax] = max(0, min(n3[ax] - 1, (int)floor((xq3[ax] - R - o3[ax]) / dx)));
                    c1[ax] = max(0, min(n3[ax] - 1, (int)floor((xq3[ax] + R - o3[ax]) / dx)));
                }
                for(int ci = c0[0]; ci <= c1[0] && !flag; ++ci)
                    for(int cj = c0[1]; cj <= c1[1] && !flag; ++cj)
                        for(int ck = c0[2]; ck <= c1[2] && !flag; ++ck)
                        {
                            const int cl = cell_lin(ci, cj, ck, dims);
                            for(int f = tri_start[cl]; f < tri_start[cl + 1]; ++f)
                            {
                                const uint3   F  = faces[tri_list[f]];
                                const double3 dq = wb_sub(xq, wb_pt_tri_cp(xq, svert[F.x], svert[F.y], svert[F.z]));
                                const double  L  = sqrt(wb_dot(dq, dq));
                                if(L < R && L > 1e-12 * dx && wb_dot(dq, up) < -ovh_c * L) { flag = 1; break; }
                            }
                        }
            }
        }
        ovh[col] = flag;
    }
    if(hwt && ntop > 0)
        hwt[col] = (hwmode == 2) ? (double)npart                                       // particle count: a sliver column behind an oblique wall holding
                                 : (double)npart / ((top[0] - hbot) / hs + 1.0);       // one stray particle (h 55 mm low) must not pull its neighbours down
    if(colmin > 0.0 && ntop > 0)
    {   // the column model assumes water fills the column's fluid cross-section from the floor up to h_col. A thin film, a puddle of scattered
        // particles, a stream or a splash does not: its geometric volume under h_col (and its weight) is several times its deposit (pour
        // probe at 76 deg: columns with deposit / geometric < 0.3 carried 52 % of the weight, 516 vs 54 dx^3). Such a column is no body:
        // all its cells become separate water (current operator, deposited weight).
        const double h = hcol[col];
        double G = 0.0;
        for(int t = lo; t <= hi; ++t)
        {
            int i, j, k;
            cell_of(t, i, j, k);
            const int c = cell_lin(i, j, k, dims);
            double clo = 1e300, chi = -1e300;
            for(int q = 0; q < 8; ++q)
            {
                const double3 pq = make_double3(org.x + (double)(i + ((q >> 2) & 1)) * dx, org.y + (double)(j + ((q >> 1) & 1)) * dx, org.z + (double)(k + (q & 1)) * dx);
                const double  uq = wb_dot(up, pq);
                clo = fmin(clo, uq); chi = fmax(chi, uq);
            }
            const double f = (chi > clo) ? fmin(fmax((h - clo) / (chi - clo), 0.0), 1.0) : 0.0;
            G += cvgeo[c] * f;
        }
        if(colD) { colD[col] = D; colG[col] = G; }                       // pooled guard: decided over the 3 x 3 neighbourhood
        else if(D < colmin * G) { hcol[col] = -1e300; body_lo[col] = -1; body_hi[col] = -1; }
    }
}

// LIQ_WB_COLMIN_POOL: the column-density guard over the 3 x 3 neighbourhood of body columns. A sliver column along a curved wall (small
// geometric cross-section, particles kept off the wall) sits next to dense columns and stays in the body (flask: 15 such columns were cut out
// one by one, their neighbours got ~3 node weights of imbalance); a puddle, a film or a stream is sparse over its whole neighbourhood.
__global__ void k_wb_colguard(int ncol, int3 dims, int U, const double* colD, const double* colG, double colmin, double* hraw, int* body_lo, int* body_hi)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= ncol) return;
    if(body_lo[col] < 0 || !(colG[col] > 0.0)) return;
    const int n1 = (U == 2) ? dims.y : dims.z, n0 = (U == 0) ? dims.y : dims.x;
    const int a = col / n1, b = col % n1;
    double sd = 0.0, sg = 0.0;
    for(int da = -1; da <= 1; ++da)
        for(int db = -1; db <= 1; ++db)
        {
            const int aa = a + da, bb = b + db;
            if(aa < 0 || bb < 0 || aa >= n0 || bb >= n1) continue;
            const int cc = aa * n1 + bb;
            if(colG[cc] > 0.0) { sd += colD[cc]; sg += colG[cc]; }
        }
    if(sd < colmin * sg) { hraw[col] = -1e300; body_lo[col] = -1; body_hi[col] = -1; }
}

// 3 x 3 column average of the raw heights (columns with a body only): the surface geometry must not follow single-particle jitter
__global__ void k_wb_hsmooth(int ncol, int3 dims, int U, int rad, const double* hraw, const int* body_lo, double* hout, const double* hwt = nullptr,
                             int capped = 0,   // LIQ_WB_HSMOOTH_W=3: weight min(1, count / (half the neighbourhood's largest count))
                             const int* ovh = nullptr, int* ovr = nullptr)   // LIQ_WB_OVH: overhang-limited columns get no say; ovr = resolved
{   // hwt (LIQ_WB_HSMOOTH_W): weight each column by its particles per layer -- a sparse column (cut by a curved wall) averages 1-2 particles and
    // is noisy; any positive weights keep a flat surface flat, so this adds no bias at rest
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= ncol) return;
    int n1 = (U == 2) ? dims.y : dims.z, n0 = (U == 0) ? dims.y : dims.x;
    const int a = col / n1, b = col % n1;
    if(ovr) ovr[col] = 1;
    if(body_lo[col] < 0) { hout[col] = -1e300; return; }
    if(hraw[col] > 1e299) { hout[col] = 1e300; return; }   // LIQ_WB_CAP: capped column, no free surface
    double sum = 0.0, wsum = 0.0; int cnt = 0;
    double wref = 0.0;
    if(hwt && capped)
    {   // capped weights: every column with at least half the neighbourhood's largest particle count counts fully (a wall column of a curved
        // beaker keeps its full say in the sloshing surface); only a sparse sliver (one stray particle behind an oblique wall) is discounted
        for(int da = -rad; da <= rad; ++da)
            for(int db = -rad; db <= rad; ++db)
            {
                const int aa = a + da, bb = b + db;
                if(aa < 0 || bb < 0 || aa >= n0 || bb >= n1) continue;
                const int cc = aa * n1 + bb;
                if(body_lo[cc] < 0 || hraw[cc] < -1e299 || hraw[cc] > 1e299) continue;
                wref = fmax(wref, hwt[cc]);
            }
        wref *= 0.5;
    }
    for(int da = -rad; da <= rad; ++da)
        for(int db = -rad; db <= rad; ++db)
        {
            const int aa = a + da, bb = b + db;
            if(aa < 0 || bb < 0 || aa >= n0 || bb >= n1) continue;
            const int cc = aa * n1 + bb;
            if(body_lo[cc] < 0 || hraw[cc] < -1e299 || hraw[cc] > 1e299) continue;
            if(ovh && ovh[cc]) continue;   // LIQ_WB_OVH: a wall-limited top is no surface height
            const double wq = hwt ? ((capped && wref > 0.0) ? fmin(1.0, fmax(hwt[cc], 1e-6) / wref) : fmax(hwt[cc], 1e-6)) : 1.0;
            sum += wq * hraw[cc]; wsum += wq; ++cnt;
        }
    hout[col] = cnt > 0 ? sum / wsum : hraw[col];
    if(ovh && ovh[col])
    {   // overhang-limited: the neighbours' surface, never below the column's own estimate; none in reach -> resolved by k_wb_ovh_fill
        if(cnt > 0) hout[col] = fmax(hout[col], hraw[col]);
        else if(ovr) ovr[col] = 0;
    }
}
// LIQ_WB_OVH: propagate the surface height into overhang-limited columns with no unflagged neighbour (a ring under a cone is 2-3 columns wide).
// One Jacobi pass: an unresolved flagged column takes the mean of its resolved neighbours (never below its own estimate). Double-buffered.
__global__ void k_wb_ovh_fill(int ncol, int3 dims, int U, const double* hraw, const int* body_lo, const int* ovh, const double* hin, const int* rin,
                              double* hout2, int* rout)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= ncol) return;
    hout2[col] = hin[col]; rout[col] = rin[col];
    if(rin[col] || !ovh[col] || body_lo[col] < 0 || hraw[col] < -1e299 || hraw[col] > 1e299) return;
    const int n1 = (U == 2) ? dims.y : dims.z, n0 = (U == 0) ? dims.y : dims.x;
    const int a = col / n1, b = col % n1;
    double sum = 0.0; int cnt = 0;
    for(int da = -1; da <= 1; ++da)
        for(int db = -1; db <= 1; ++db)
        {
            const int aa = a + da, bb = b + db;
            if((da == 0 && db == 0) || aa < 0 || bb < 0 || aa >= n0 || bb >= n1) continue;
            const int cc = aa * n1 + bb;
            if(!rin[cc] || body_lo[cc] < 0 || hin[cc] < -1e299 || hin[cc] > 1e299) continue;
            sum += hin[cc]; ++cnt;
        }
    if(cnt > 0) { hout2[col] = fmax(sum / cnt, hraw[col]); rout[col] = 1; }
}

// ---- per active cell (= voxel element in the non-split path): the cut-cell operator, or the full-cell weights ----
__global__ void k_wb_geom(int nact, const int* active_cells, int3 dims, double3 org, double dx, int U, int sgn, const double* phi_node, const double* Vc,
                          const int* cflag, const double* hcol, const int* body_lo, const int* body_hi, double3 g, double rho, int* vox_cc, double3* vox_a,
                          double* vox_vg, double3* vox_f, double* vox_w, double* node_w, int* counters, int dump = 0, int surf_mode = 1, int* dead_cell = nullptr,
                          double* hg = nullptr, double* vox_s = nullptr, double snap = 0.0, int hg_wallonly = 0, int topface = 0, int* vox_wc = nullptr,
                          double* vox_yc = nullptr, int fullf = 0)
{   // fullf (LIQ_WB_FULLF): full cells also carry the exact hydrostatic term rho int g.(x - x_c) grad N_q dV that cut cells carry. The one-point
    // rule misses it (horizontal +-rho g dx^3 / 24 per corner, zero vertical): it cancels between two full cells, but not between a full and a
    // cut cell, nor at the free surface next to a cut cell -- oblique / curved walls (yawed tank, step 0: exactly 1/12 node weight on the
    // contact-line nodes; flask: everywhere along the sloped wall, where the cut pattern changes with height)
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    if(hg) for(int T = 0; T < 6; ++T) hg[150 * v + 25 * T] = 0.0;   // no sub-cell record unless the cut branch fills it
    if(vox_wc) vox_wc[v] = 0;
    if(vox_yc) vox_yc[v] = -1e300;   // LIQ_WB_YBAR: no correction unless a full / cut branch sets the fluid centroid
    if(vox_s) for(int t = 0; t < 36; ++t) vox_s[36 * v + t] = 0.0;
    const int c = active_cells[v];
    int i, j, k;
    cell_ijk(c, dims, i, j, k);
    int nid[8];
    for(int q = 0; q < 8; ++q) nid[q] = node_lin(i + ((q >> 2) & 1), j + ((q >> 1) & 1), k + (q & 1), dims);
    auto dead = [&](int which) {
        if(dead_cell) dead_cell[c] = 1;
        vox_cc[v] = 1; vox_vg[v] = 0.0;
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = 0.0; vox_a[8 * v + q] = make_double3(0, 0, 0); vox_f[8 * v + q] = make_double3(0, 0, 0); }
        atomicAdd(counters + which, 1);
    };
    if(cflag[c] == 2)
    {   // every corner outside: only kernel spill from across a wall. Dead element (its deposit was credited to its column's body)
        dead(2);
        if(dump && v < dump) printf("[wb-dump] cell %d (%d,%d,%d) all corners outside, Vdep %.3e -> dead\n", c, i, j, k, Vc[c]);
        return;
    }
    // column bookkeeping
    const int uidx = (U == 0) ? i : (U == 1) ? j : k;
    const int nU   = (U == 0) ? dims.x : (U == 1) ? dims.y : dims.z;
    const int t    = (sgn > 0) ? uidx : (nU - 1 - uidx);
    const int col  = (U == 0) ? (j * dims.z + k) : (U == 1) ? (i * dims.z + k) : (i * dims.y + j);
    const int lo = body_lo[col], hi = body_hi[col];
    const double gm = sqrt(wb_dot(g, g));
    const double3 up = gm > 0.0 ? wb_scale(g, -1.0 / gm) : make_double3(0, 0, 0);
    double phi[8];
    int    nneg = 0;
    for(int q = 0; q < 8; ++q) { phi[q] = phi_node[nid[q]]; if(phi[q] < 0.0) ++nneg; }
    const double dx3 = dx * dx * dx;
    if(gm <= 0.0 || lo < 0 || t < lo || t > hi)
    {   // no gravity, or separate water above the column's body (a stream, a droplet): current operator, deposited weight
        vox_cc[v] = 0;
        const double w8 = rho * Vc[c] / 8.0;
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); if(fullf) vox_f[8 * v + q] = make_double3(0, 0, 0); }
        atomicAdd(counters + 4, 1);
        if(dump && v < dump) printf("[wb-dump] cell %d (%d,%d,%d) separate water: t %d body [%d,%d] hcol %.5f Vdep %.3e cflag %d\n", c, i, j, k, t, lo, hi, hcol[col], Vc[c], cflag[c]);
        return;
    }
    double hlo = 1e300, hhi = -1e300;
    for(int q = 0; q < 8; ++q)
    {
        const double3 pq = make_double3(org.x + (double)(i + ((q >> 2) & 1)) * dx, org.y + (double)(j + ((q >> 1) & 1)) * dx, org.z + (double)(k + (q & 1)) * dx);
        const double  uq = wb_dot(up, pq);
        hlo = fmin(hlo, uq); hhi = fmax(hhi, uq);
    }
    double h = hcol[col];
    if(snap > 0.0)
    {   // a surface within snap*dx of a node plane sits ON it: no sliver surface cells (the small-cell problem), no flicker of the top cell's fill
        const double hc = org.x * up.x + org.y * up.y + org.z * up.z;   // lattice offset along up (axis-aligned up assumed for the snap)
        const double kf = (h - hc) / dx, kr = round(kf);
        if(fabs(kf - kr) < snap) h = hc + kr * dx;
    }
    if(surf_mode == 0 && h < hhi - 1e-12 * dx && nneg == 0)
    {   // LIQ_WB_SURF=0 (diagnostic): no surface plane -- an uncut cell at or above the surface keeps the current operator,
        // weight = its deposited volume (the pre-WB treatment of the free surface)
        vox_cc[v] = 0;
        const double w8 = rho * Vc[c] / 8.0;
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); if(fullf) vox_f[8 * v + q] = make_double3(0, 0, 0); }
        atomicAdd(counters + 4, 1);
        return;
    }
    if(h <= hlo + 1e-12 * dx)
    {   // above the column's water: kernel spill, dead
        dead(3);
        if(dump && v < dump) printf("[wb-dump] cell %d (%d,%d,%d) above the column surface (h %.5f <= cell bottom %.5f), Vdep %.3e -> dead\n", c, i, j, k, h, hlo, Vc[c]);
        return;
    }
    const bool surf = (surf_mode != 0) && (h < hhi - 1e-12 * dx || (topface && h <= hhi + 1e-9 * dx));   // LIQ_WB_TOPFACE: a surface exactly on the top face is registered as a surface plane (worse at rest, 10-23)
    if(nneg == 0 && !surf)
    {   // full cell: the current one-point operator is exact for it; consistent weights rho dx^3 / 8 per corner
        vox_cc[v] = 0;
        if(vox_yc) vox_yc[v] = wb_dot(up, make_double3(org.x + (i + 0.5) * dx, org.y + (j + 0.5) * dx, org.z + (k + 0.5) * dx));
        const double w8 = rho * dx3 / 8.0;
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); }
        if(fullf)
        {   // f_q = rho int g.(x - x_c) grad N_q dV over the full cube (trilinear N_q, s = +-1 per corner bit): rho dx^3 / 24 x
            // ( s_x (g_y s_y + g_z s_z), s_y (g_x s_x + g_z s_z), s_z (g_x s_x + g_y s_y) ) -- what wb_tet_accumulate gives for an uncut cell
            const double c24 = rho * dx3 / 24.0;
            for(int q = 0; q < 8; ++q)
            {
                const double sx = ((q >> 2) & 1) ? 1.0 : -1.0, sy = ((q >> 1) & 1) ? 1.0 : -1.0, sz = (q & 1) ? 1.0 : -1.0;
                vox_f[8 * v + q] = make_double3(c24 * sx * (g.y * sy + g.z * sz), c24 * sy * (g.x * sx + g.z * sz), c24 * sz * (g.x * sx + g.y * sy));
            }
        }
        return;
    }
    if(dump && v < dump)
        printf("[wb-dump] cell %d (%d,%d,%d) phi %.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g | Vdep %.3e surf %d h %.5f (hlo %.5f hhi %.5f) nneg %d\n",
               c, i, j, k, phi[0], phi[1], phi[2], phi[3], phi[4], phi[5], phi[6], phi[7], Vc[c], (int)surf, h, hlo, hhi, nneg);
    WbAcc A;
    wb_acc_init(A);
    wb_region(i, j, k, org, dx, phi, up, surf ? h : hhi, g, true, &A, hg ? hg + 150 * v : nullptr);
    if(!(A.V > 1e-9 * dx3)) { if(hg) for(int T = 0; T < 6; ++T) hg[150 * v + 25 * T] = 0.0; dead(3); return; }
    const double3 xc  = wb_scale(A.M, 1.0 / A.V);
    if(vox_yc) vox_yc[v] = wb_dot(up, xc);
    const double  gxc = wb_dot(g, xc);
    vox_cc[v] = 1;
    vox_vg[v] = A.V;
    for(int q = 0; q < 8; ++q)
    {
        vox_a[8 * v + q] = A.a[q];
        double3 f = wb_sub(wb_sub(A.A[q], wb_scale(A.G[q], gxc)), wb_sub(A.omg[q], wb_scale(A.om[q], gxc)));   // rho (b - beta)
        vox_f[8 * v + q] = wb_scale(f, rho);
        const double w = rho * A.W[q];
        vox_w[8 * v + q] = w;
        atomicAdd(&node_w[nid[q]], w);
    }
    if(vox_s) for(int t = 0; t < 36; ++t) vox_s[36 * v + t] = A.S[t];
    if(vox_wc) vox_wc[v] = (nneg > 0) ? 1 : 0;   // wall-cut element (some corner behind the wall)
    if(hg && hg_wallonly && nneg == 0) for(int T = 0; T < 6; ++T) hg[150 * v + 25 * T] = 0.0;   // LIQ_WB_HG_WALLONLY: no sub-cell term on surface-only cut cells
    atomicAdd(counters, 1);
    if(surf) atomicAdd(counters + 1, 1);
    if(dump && v < dump)
    {
        printf("[wb-op] cell %d o %.17g %.17g %.17g dx %.17g up %.17g %.17g %.17g h %.17g g %.17g %.17g %.17g V %.17g phi %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
               c, org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx, dx, up.x, up.y, up.z, surf ? h : hhi, g.x, g.y, g.z, A.V,
               phi[0], phi[1], phi[2], phi[3], phi[4], phi[5], phi[6], phi[7]);
        for(int q = 0; q < 8; ++q)
            printf("[wb-op] cell %d q %d a %.17g %.17g %.17g f %.17g %.17g %.17g w %.17g\n", c, q, vox_a[8 * v + q].x, vox_a[8 * v + q].y, vox_a[8 * v + q].z,
                   vox_f[8 * v + q].x, vox_f[8 * v + q].y, vox_f[8 * v + q].z, vox_w[8 * v + q]);
    }
}


// ---- aggregation of poorly supported nodes (Coombs, Bird, Pretti: Aggregated MPM; Badia et al.: AgFEM) ----
// A node whose geometric fluid share is below tau rho dx^3 (a sliver corner near an oblique wall, a corner above a nearly empty
// surface cell) barely represents fluid, yet it is a free, nearly massless degree of freedom: it jitters and leaks. Instead of fixing it
// (an artificial no-slip layer) its displacement is tied to the trilinear EXTRAPOLATION of a well-supported root cell: u_s = sum_j N_j^R(X_s) u_j.
// Root = among the cells at Chebyshev distance d = 1.. that do not contain the node and whose 8 corners are all well supported, the one
// with the most fluid (ties: the one most along the wall normal); smallest d first. Nodes ON the wall (|phi| <= thr) keep the slip
// condition for the normal and are slaved only tangentially (projector I - n n^T): the tangential hourglass motion of the wall layer is
// what destabilises a slip wall without ghost penalty. Nodes behind the wall stay with the GHOST mode (fixed) unless aggr_ghost = 1:
// an extrapolated ghost feeds amplified interior noise back to the particles through the 27-node stencil (aligned tank 127 mm/s).
// Penalty form, K in inertia units like the wall BC. At the hydrostatic stationary point every node force vanishes, so the constraint
// carries no force: the rest state is preserved; in motion it is a kinematic tie, no dissipation, no new parameter besides tau.
__global__ void k_wb_aggr_mark(int nact, const int* active_nodes, int3 dims, double dx, const double* phi_node, const double3* nrm_node, double thr,
                               const double* node_w, double tau_w, int depth, int aggr_wall, int aggr_ghost, const int* cell_flag, int* agroot, int* agtype, int* count)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    agroot[n] = -1;
    agtype[n] = 0;
    const double ph = phi_node[n];
    if(ph < -1e29) return;                                   // no fluid anywhere near
    const bool onwall = fabs(ph) <= thr, behind = ph < -thr;
    if(behind && !aggr_ghost) return;                        // ghost: GHOST mode (fixed)
    if(onwall && !aggr_wall) return;
    if(!onwall && node_w[n] >= tau_w) return;                // well supported interior node: free
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 nr = nrm_node[n];
    for(int d = 1; d <= depth; ++d)
    {
        int    best = -1;
        double bestw = -1.0, bests = -2.0;
        for(int ex = -1; ex <= 1; ++ex)
            for(int ey = -1; ey <= 1; ++ey)
                for(int ez = -1; ez <= 1; ++ez)
                {
                    if(ex == 0 && ey == 0 && ez == 0) continue;
                    // cells at Chebyshev distance d along (ex,ey,ez); a zero component takes both sides
                    for(int sub = 0; sub < 8; ++sub)
                    {
                        if((ex != 0 && (sub & 4)) || (ey != 0 && (sub & 2)) || (ez != 0 && (sub & 1))) continue;
                        const int ai = ex > 0 ? i + d : (ex < 0 ? i - d - 1 : i - 1 + ((sub >> 2) & 1));
                        const int aj = ey > 0 ? j + d : (ey < 0 ? j - d - 1 : j - 1 + ((sub >> 1) & 1));
                        const int ak = ez > 0 ? k + d : (ez < 0 ? k - d - 1 : k - 1 + (sub & 1));
                        if(ai < 0 || aj < 0 || ak < 0 || ai >= dims.x || aj >= dims.y || ak >= dims.z) continue;
                        const int c = cell_lin(ai, aj, ak, dims);
                        if(cell_flag[c] != 1) continue;      // not a fluid-side active cell
                        double wsum = 0.0;
                        bool   ok   = true;
                        for(int r = 0; r < 8 && ok; ++r)
                        {
                            const int m = node_lin(ai + ((r >> 2) & 1), aj + ((r >> 1) & 1), ak + (r & 1), dims);
                            const double wm = node_w[m];
                            if(wm < tau_w) ok = false;
                            wsum += wm;
                        }
                        if(!ok) continue;
                        const double sc = nr.x * ex + nr.y * ey + nr.z * ez;
                        if(wsum > bestw + 1e-12 || (fabs(wsum - bestw) <= 1e-12 && sc > bests)) { best = c; bestw = wsum; bests = sc; }
                    }
                }
        if(best >= 0)
        {
            agroot[n] = best;
            agtype[n] = onwall ? 1 : 0;
            atomicAdd(count, 1);
            return;
        }
    }
}
// compact: constraint t -> (node, root cell, type, normal, 8 extrapolation coefficients)
__global__ void k_wb_aggr_compact(int nact, const int* active_nodes, const int* agroot, const int* agtype, const double3* nrm_node, const int* scan, int3 dims,
                                  int* agn, int* agr, int* agt, double3* agnrm, double* agc)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s], c = agroot[n];
    if(c < 0) return;
    const int t = scan[s];
    agn[t] = n; agr[t] = c; agt[t] = agtype[n]; agnrm[t] = nrm_node[n];
    int i, j, k, ci, cj, ck;
    node_ijk(n, dims, i, j, k);
    cell_ijk(c, dims, ci, cj, ck);
    const double3 xi = make_double3((double)(i - ci), (double)(j - cj), (double)(k - ck));   // node in the root cell's unit coordinates
    for(int q = 0; q < 8; ++q) agc[8 * t + q] = wb_N(q, xi);
}
__global__ void k_wb_aggr_flag(int nact, const int* active_nodes, const int* agroot, int* flag)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    flag[s] = agroot[active_nodes[s]] >= 0 ? 1 : 0;
}
__device__ __forceinline__ double3 wb_proj(int type, double3 nr, double3 r)
{   // type 1: tangential part only
    if(type != 1) return r;
    const double rn = wb_dot(nr, r);
    return wb_sub(r, wb_scale(nr, rn));
}
// gradient (inertia units, into the binned gradient) and 45 upper Hessian blocks per constraint
__global__ void k_wb_aggr_assemble(int nag, const int* agn, const int* agr, const int* agt, const double3* agnrm, const double* agc, int3 dims, double3 org, double dx,
                                   const int* node_slot, int base, int fem_off, const double3* vert, double K, int off, Eigen::Matrix3d* vals, int* rows, int* cols)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= nag) return;
    const int n = agn[t], c = agr[t];
    int i, j, k, ci, cj, ck;
    node_ijk(n, dims, i, j, k);
    cell_ijk(c, dims, ci, cj, ck);
    int vid[9];
    double coef[9];
    bool ok = node_slot[n] >= 0;
    vid[0] = base + node_slot[n]; coef[0] = 1.0;
    double3 r = wb_sub(vert[vid[0]], make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx));   // u_s
    for(int q = 0; q < 8; ++q)
    {
        const int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
        const int m = node_lin(ci + a, cj + b, ck + cc, dims);
        if(node_slot[m] < 0) ok = false;
        vid[q + 1]  = base + node_slot[m];
        coef[q + 1] = -agc[8 * t + q];
        const double3 u = wb_sub(vert[ok ? vid[q + 1] : vid[0]], make_double3(org.x + (ci + a) * dx, org.y + (cj + b) * dx, org.z + (ck + cc) * dx));
        r = wb_add(r, wb_scale(u, coef[q + 1]));
    }
    if(!ok)
    {   // a root corner without a slot (should not happen): zero blocks on valid indices, no force
        const int v0 = base + (node_slot[n] >= 0 ? node_slot[n] : 0);
        for(int kk = 0; kk < 45; ++kk) { rows[off + 45 * t + kk] = v0 + fem_off; cols[off + 45 * t + kk] = v0 + fem_off; vals[off + 45 * t + kk].setZero(); }
        return;
    }
    const int     ty = agt[t];
    const double3 nr = agnrm[t];
    r = wb_proj(ty, nr, r);
    Eigen::Matrix3d Pm = Eigen::Matrix3d::Identity();
    if(ty == 1)
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 3; ++b) Pm(a, b) -= (a == 0 ? nr.x : (a == 1 ? nr.y : nr.z)) * (b == 0 ? nr.x : (b == 1 ? nr.y : nr.z));
    for(int p = 0; p < 9; ++p)
    {
        const double f[3] = {K * coef[p] * r.x, K * coef[p] * r.y, K * coef[p] * r.z};
        for(int a = 0; a < 3; ++a) binned_deposit(g_gbin + ((size_t)vid[p] * 3 + a) * BINNED_K, f[a]);
    }
    int kk = 0;
    for(int p = 0; p < 9; ++p)
        for(int q2 = p; q2 < 9; ++q2)
        {
            const int idx = off + 45 * t + kk;
            int row = vid[p] + fem_off, col = vid[q2] + fem_off;
            Eigen::Matrix3d B = Pm * (K * coef[p] * coef[q2]);
            if(row <= col) { rows[idx] = row; cols[idx] = col; }
            else { rows[idx] = col; cols[idx] = row; }
            vals[idx] = B;
            ++kk;
        }
}
__global__ void k_wb_aggr_energy(int nag, const int* agn, const int* agr, const int* agt, const double3* agnrm, const double* agc, int3 dims, double3 org, double dx,
                                 const int* node_slot, int base, const double3* vert, double Kp, double* part)
{
    extern __shared__ double sh[];
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(t < nag)
    {
        const int n = agn[t], c = agr[t];
        int i, j, k, ci, cj, ck;
        node_ijk(n, dims, i, j, k);
        cell_ijk(c, dims, ci, cj, ck);
        bool ok = node_slot[n] >= 0;
        double3 r = wb_sub(vert[base + (ok ? node_slot[n] : 0)], make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx));
        for(int q = 0; q < 8; ++q)
        {
            const int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
            const int m = node_lin(ci + a, cj + b, ck + cc, dims);
            if(node_slot[m] < 0) { ok = false; break; }
            const double3 u = wb_sub(vert[base + node_slot[m]], make_double3(org.x + (ci + a) * dx, org.y + (cj + b) * dx, org.z + (ck + cc) * dx));
            r = wb_sub(r, wb_scale(u, agc[8 * t + q]));
        }
        r = wb_proj(agt[t], agnrm[t], r);
        e = ok ? 0.5 * Kp * wb_dot(r, r) : 0.0;
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int st = blockDim.x / 2; st > 0; st >>= 1)
    {
        if(threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if(threadIdx.x == 0) part[blockIdx.x] = sh[0];
}


// debug (LIQ_WB_AGGR_DEBUG): constraint residual after the solve, |P(u_s - sum c_j u_j)| per constraint -> max, sum, by type
__global__ void k_wb_aggr_residual(int nag, const int* agn, const int* agr, const int* agt, const double3* agnrm, const double* agc, int3 dims, double3 org, double dx,
                                   const int* node_slot, int base, const double3* vert, double* out /* [0] max, [1] sum, [2] max type1, [3] max |u_s| */)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if(t >= nag) return;
    const int n = agn[t], c = agr[t];
    int i, j, k, ci, cj, ck;
    node_ijk(n, dims, i, j, k);
    cell_ijk(c, dims, ci, cj, ck);
    if(node_slot[n] < 0) return;
    const double3 us = wb_sub(vert[base + node_slot[n]], make_double3(org.x + i * dx, org.y + j * dx, org.z + k * dx));
    double3 r = us;
    for(int q = 0; q < 8; ++q)
    {
        const int a = (q >> 2) & 1, b = (q >> 1) & 1, cc = q & 1;
        const int m = node_lin(ci + a, cj + b, ck + cc, dims);
        if(node_slot[m] < 0) return;
        const double3 u = wb_sub(vert[base + node_slot[m]], make_double3(org.x + (ci + a) * dx, org.y + (cj + b) * dx, org.z + (ck + cc) * dx));
        r = wb_sub(r, wb_scale(u, agc[8 * t + q]));
    }
    r = wb_proj(agt[t], agnrm[t], r);
    const double rn = sqrt(wb_dot(r, r)), un = sqrt(wb_dot(us, us));
    atomicMax((unsigned long long*)&out[0], __double_as_longlong(rn));
    atomicAdd(&out[1], rn);
    if(agt[t] == 1) atomicMax((unsigned long long*)&out[2], __double_as_longlong(rn));
    atomicMax((unsigned long long*)&out[3], __double_as_longlong(un));
}

}   // namespace gipc_liquid_wb
