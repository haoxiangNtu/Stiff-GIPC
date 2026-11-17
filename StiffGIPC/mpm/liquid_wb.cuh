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
__global__ void k_wb_phi(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const int* cstart, const int* cend, const int* pidx,
                         const int* wp_valid, const int* wp_near, const double3* wp_X, const double3* wp_n, double lat2, double* phi_node, double3* nrm_node, int* pmin_node)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    int i, j, k;
    node_ijk(n, dims, i, j, k);
    const double3 x = make_double3(org.x + (double)i * dx, org.y + (double)j * dx, org.z + (double)k * dx);
    double  phi = 1e30;
    double3 nrm = make_double3(0, 1, 0);
    int     pmin = -1;
    bool    any = false;
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
                    if(!(wp_valid[p] || wp_near[p])) continue;
                    const double3 r = wb_sub(x, wp_X[p]);
                    const double  d = wb_dot(wp_n[p], r);
                    if(lat2 > 0.0)
                    {   // lateral gate: the node must lie over this wall point (within lat of its foot); a node in a hole is not "on the wall"
                        const double3 tl = wb_sub(r, wb_scale(wp_n[p], d));
                        if(wb_dot(tl, tl) > lat2) continue;
                    }
                    if(d < phi) { phi = d; nrm = wp_n[p]; pmin = p; }
                }
            }
    phi_node[n] = any ? phi : -1e30;
    nrm_node[n] = nrm;
    pmin_node[n] = pmin;
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
__global__ void k_wb_slip_mark(int nact, const int* active_nodes, int3 dims, double3 org, double dx, const double* phi_node, const double3* nrm_node, double thr,
                               int ghost_mode, const int* pmin_node, const int* wp_tri, const double3* wp_b, const uint3* faces, const double3* vert,
                               const double3* overt, const double3* svel, double dt, const double* node_w, double wmin, double near, double mfloor, double* node_mfl,
                               const int* agroot, const int* agtype, int* wall_t, double3* wall_n, double3* wall_xt, int* bh_wall, int* counter)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact) return;
    const int n = active_nodes[s];
    const double ph = phi_node[n];
    if(ph < -1e29) return;   // -big = no particles anywhere near: not a wall node, just empty
    if(agroot && agroot[n] >= 0 && agtype[n] == 0) { if(bh_wall) bh_wall[n] = 1; return; }   // fully aggregated: tied to its root cell, not to the wall (implicit Euler like a wall node)
    const bool sliver = ph > thr && ph <= near && node_w && node_w[n] < wmin;
    if(ph > thr && ph <= near && mfloor > 0.0 && node_mfl) node_mfl[n] = mfloor;   // free near-wall node: extra inertia (LIQ_WB_MFLOOR)
    if(ph > thr && !sliver) return;
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
    wall_t[s]  = (ghost_mode == 3 || sliver) ? 2 : ((ph < -thr) ? ghost_mode : 1);   // behind the wall: slip or fixed; on the wall: slip; sliver: fixed (mode 3: everything fixed)
    wall_n[s]  = nrm_node[n];
    wall_xt[s] = make_double3(org.x + (double)i * dx + dX.x, org.y + (double)j * dx + dX.y, org.z + (double)k * dx + dX.z);
    if(bh_wall) bh_wall[n] = 1;
    atomicAdd(counter, 1);
}
__global__ void k_wb_slip_project(int nact, int base, const int* wall_t, const double3* wall_n, double3* vel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if(s >= nact || wall_t[s] == 0) return;
    if(wall_t[s] == 2) { vel[base + s] = make_double3(0, 0, 0); return; }   // fixed ghost (static container)
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
                            int topk, double* hcol, int* body_lo, int* body_hi)
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
    hcol[col] = -1e300; body_lo[col] = -1; body_hi[col] = -1;
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
    body_lo[col] = lo; body_hi[col] = hi;
    // the free surface is where the particles end: the mean height of the top-k particles of the body's cells (the top
    // layer: k = particles per layer per column) + half a particle spacing. (The deposited volume is biased low next to
    // walls -- particles stay ~h/2 off the wall and the kernel spills part of their volume across it -- so a volume-based
    // height would step down along every wall; the single topmost particle jitters.) Capped at the body's top.
    double top[8];
    int    ntop = 0;
    const int K = (topk < 1) ? 1 : (topk > 8 ? 8 : topk);
    double htop = -1e300;
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
            if(ntop < K) { int m = ntop++; while(m > 0 && top[m - 1] < hp) { top[m] = top[m - 1]; --m; } top[m] = hp; }
            else if(hp > top[K - 1]) { int m = K - 1; while(m > 0 && top[m - 1] < hp) { top[m] = top[m - 1]; --m; } top[m] = hp; }
        }
    }
    double hmean = 0.0;
    for(int m = 0; m < ntop; ++m) hmean += top[m];
    hcol[col] = (ntop > 0) ? fmin(hmean / ntop + 0.5 * hs, htop) : -1e300;
    (void)D;
}

// 3 x 3 column average of the raw heights (columns with a body only): the surface geometry must not follow single-particle jitter
__global__ void k_wb_hsmooth(int ncol, int3 dims, int U, int rad, const double* hraw, const int* body_lo, double* hout)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= ncol) return;
    int n1 = (U == 2) ? dims.y : dims.z, n0 = (U == 0) ? dims.y : dims.x;
    const int a = col / n1, b = col % n1;
    if(body_lo[col] < 0) { hout[col] = -1e300; return; }
    double sum = 0.0; int cnt = 0;
    for(int da = -rad; da <= rad; ++da)
        for(int db = -rad; db <= rad; ++db)
        {
            const int aa = a + da, bb = b + db;
            if(aa < 0 || bb < 0 || aa >= n0 || bb >= n1) continue;
            const int cc = aa * n1 + bb;
            if(body_lo[cc] < 0 || hraw[cc] < -1e299) continue;
            sum += hraw[cc]; ++cnt;
        }
    hout[col] = cnt > 0 ? sum / cnt : hraw[col];
}

// ---- per active cell (= voxel element in the non-split path): the cut-cell operator, or the full-cell weights ----
__global__ void k_wb_geom(int nact, const int* active_cells, int3 dims, double3 org, double dx, int U, int sgn, const double* phi_node, const double* Vc,
                          const int* cflag, const double* hcol, const int* body_lo, const int* body_hi, double3 g, double rho, int* vox_cc, double3* vox_a,
                          double* vox_vg, double3* vox_f, double* vox_w, double* node_w, int* counters, int dump = 0, int surf_mode = 1, int* dead_cell = nullptr,
                          double* hg = nullptr, double* vox_s = nullptr, double snap = 0.0, int hg_wallonly = 0, int topface = 0, int* vox_wc = nullptr)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= nact) return;
    if(hg) for(int T = 0; T < 6; ++T) hg[150 * v + 25 * T] = 0.0;   // no sub-cell record unless the cut branch fills it
    if(vox_wc) vox_wc[v] = 0;
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
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); }
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
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); }
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
        const double w8 = rho * dx3 / 8.0;
        for(int q = 0; q < 8; ++q) { vox_w[8 * v + q] = w8; atomicAdd(&node_w[nid[q]], w8); }
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
