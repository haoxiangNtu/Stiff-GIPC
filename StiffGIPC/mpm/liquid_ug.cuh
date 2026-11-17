// LIQ_UG: the particle-quadrature MPM liquid of the recovered Unified_GIPC engine, inside this engine's liquid system.
// Included by liquid_system.cu (inside namespace gipc_liquid, after the voxel helpers).
//
//   * every particle is a quadrature point with the trilinear 8-node stencil of its cell (nodes at the cell corners), or,
//     with LIQ_UG_BSPLINE=1, the quadratic B-spline 27-node stencil (nodes b..b+2 per axis, b = floor(x/dx - 1/2));
//   * trial deformation A_p = I + sum_k u_k grad w_k^T, J_p = J_p^n det(A_p)   (J^n tracked on the particle);
//   * energy  V_p^0 [ kappa/2 (J-1)^2  +  barrier(J) ]  (no dt^2 here: the engine's energy combine scales by dt^2);
//     barrier(J) = k_b (s-1)^2 ln^2 s, s = J / J_act, active below the activation ratio J_act (recovered PQ volume barrier);
//   * gradient / Hessian w.r.t. the node displacements, the 9x9 Hessian in A projected to PSD;
//   * the particle embedding x_p = x_p^n + sum_k w_k u_k (also the contact embedding, via node_stencil with g_liq_tri = 1 / 2);
//   * commit: dx = sum w u, D = sum u grad w^T, x += dx, v = dx/dt, C = D/dt, J *= det(I + D).
// No grid wall constraint, ghost penalty, rest drag, push-apart or density reset: water meets solids only through the
// particle IPC barrier + CCD (as in the recovered engine).
//
// The kernels are templated on NA = nodes per axis (2 trilinear, 3 quadratic B-spline): NS = NA^3 stencil nodes per particle
// (8 / 27), NB = NS (NS + 1) / 2 upper-triangle 3x3 Hessian blocks per particle (36 / 378). The per-particle arrays
// (nodes, weights, gradients) have stride NS; the triplet range of a particle starts at global_offset + NB * p.

// lattice (rest) position of the grid-node vertex v: the node's displacement is x[v] - ug_lattice(v)
// (the engine's o_vertexes are the Newton iterate at the line-search start, not the lattice)
struct UgGrid
{
    const int* active;   // slot -> lattice node id
    int        base;     // vertex index of slot 0
    int3       dims;
    double3    org;
    double     dx;
};
__device__ __forceinline__ double3 ug_lattice(const UgGrid& G, int v)
{
    int i, j, k;
    node_ijk(G.active[v - G.base], G.dims, i, j, k);
    return make_double3(G.org.x + i * G.dx, G.org.y + j * G.dx, G.org.z + k * G.dx);
}

struct UgState
{
    double A[3][3];
    double detA;
    double J;
};

template<int NA> struct UgN
{
    static constexpr int NS = NA * NA * NA;
    static constexpr int NB = NS * (NS + 1) / 2;
};

// trilinear stencil of a particle: base node i0 = floor((x-org)/dx) (clamped so that i0+1 stays inside)
__device__ __forceinline__ void ug_weights(double3 x, double3 org, double dx, int3 dims, int i0[3], double w[3][3], double dw[3][3])
{
    double r[3] = {(x.x - org.x) / dx, (x.y - org.y) / dx, (x.z - org.z) / dx};
    int    n[3] = {dims.x, dims.y, dims.z};   // cells per axis; nodes 0..n
    for(int a = 0; a < 3; ++a)
    {
        int    b = (int)floor(r[a]);
        double f = r[a] - b;
        if(b < 0) { b = 0; f = 0.0; }
        if(b > n[a] - 1) { b = n[a] - 1; f = 1.0; }
        i0[a]    = b;
        w[a][0]  = 1.0 - f;
        w[a][1]  = f;
        w[a][2]  = 0.0;
        dw[a][0] = -1.0 / dx;
        dw[a][1] = 1.0 / dx;
        dw[a][2] = 0.0;
    }
}

// quadratic B-spline stencil: base node b = floor((x-org)/dx - 1/2), nodes b, b+1, b+2 (clamped so that b+2 stays inside);
// with f = (x-org)/dx - 1/2 - b in [0, 1) the particle sits at node coordinate b + 1/2 + f (the engine's centre-stencil
// convention, so node_stencil() with g_liq_tri = 2 reproduces exactly these weights for the contact embedding):
//   w = ( (1-f)^2 / 2,  3/4 - (f-1/2)^2,  f^2 / 2 ),  dw/dx = ( -(1-f),  1 - 2f,  f ) / dx
__device__ __forceinline__ void ug_weights_q(double3 x, double3 org, double dx, int3 dims, int i0[3], double w[3][3], double dw[3][3])
{
    double r[3] = {(x.x - org.x) / dx - 0.5, (x.y - org.y) / dx - 0.5, (x.z - org.z) / dx - 0.5};
    int    n[3] = {dims.x, dims.y, dims.z};
    for(int a = 0; a < 3; ++a)
    {
        int    b = (int)floor(r[a]);
        double f = r[a] - b;
        if(b < 0) { b = 0; f = 0.0; }
        if(b > n[a] - 2) { b = n[a] - 2; f = 1.0; }
        i0[a]    = b;
        w[a][0]  = 0.5 * (1.0 - f) * (1.0 - f);
        w[a][1]  = 0.75 - (f - 0.5) * (f - 0.5);
        w[a][2]  = 0.5 * f * f;
        dw[a][0] = -(1.0 - f) / dx;
        dw[a][1] = (1.0 - 2.0 * f) / dx;
        dw[a][2] = f / dx;
    }
}

template<int NA>
__device__ __forceinline__ void ug_stencil_weights(double3 x, double3 org, double dx, int3 dims, int i0[3], double w[3][3], double dw[3][3])
{
    if(NA == 2) ug_weights(x, org, dx, dims, i0, w, dw);
    else ug_weights_q(x, org, dx, dims, i0, w, dw);
}

// P2G (APIC) to the stencil nodes: node mass and momentum, plus the "referenced" flag of nodes with weight > 0
template<int NA>
__global__ void k_ug_p2g(int n, const double3* xp, const double3* vp, const double* Cp, const double* mp, double3 org, double dx, int3 dims,
                         int apic, double* mi, double3* mvi, int* ref)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 x = xp[p], v = vp[p];
    double  m = mp[p];
    int     i0[3];
    double  w[3][3], dw[3][3];
    ug_stencil_weights<NA>(x, org, dx, dims, i0, w, dw);
    for(int a = 0; a < NA; ++a)
        for(int b = 0; b < NA; ++b)
            for(int c = 0; c < NA; ++c)
            {
                double wq = w[0][a] * w[1][b] * w[2][c];
                if(!(wq > 0.0)) continue;
                int     nid = node_lin(i0[0] + a, i0[1] + b, i0[2] + c, dims);
                double3 d   = make_double3(org.x + (i0[0] + a) * dx - x.x, org.y + (i0[1] + b) * dx - x.y, org.z + (i0[2] + c) * dx - x.z);
                double3 va  = v;
                if(apic)
                {
                    const double* C = Cp + 9 * p;
                    va.x += C[0] * d.x + C[1] * d.y + C[2] * d.z;
                    va.y += C[3] * d.x + C[4] * d.y + C[5] * d.z;
                    va.z += C[6] * d.x + C[7] * d.y + C[8] * d.z;
                }
                double wm = wq * m;
                ref[nid]  = 1;
                atomicAdd(&mi[nid], wm);
                atomicAdd(&mvi[nid].x, wm * va.x);
                atomicAdd(&mvi[nid].y, wm * va.y);
                atomicAdd(&mvi[nid].z, wm * va.z);
            }
}

// per-particle stencil: NS vertex ids (-1 = node inactive, i.e. weight 0), weights and weight gradients
template<int NA>
__global__ void k_ug_stencil(int n, const double3* xp, double3 org, double dx, int3 dims, const int* node_slot, int base, int* nodes, double* wts,
                             double3* grad, int* bad)
{
    constexpr int NS = UgN<NA>::NS;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    int    i0[3];
    double w[3][3], dw[3][3];
    ug_stencil_weights<NA>(xp[p], org, dx, dims, i0, w, dw);
    for(int a = 0; a < NA; ++a)
        for(int b = 0; b < NA; ++b)
            for(int c = 0; c < NA; ++c)
            {
                int    k  = (a * NA + b) * NA + c;
                double wq = w[0][a] * w[1][b] * w[2][c];
                int    sl = node_slot[node_lin(i0[0] + a, i0[1] + b, i0[2] + c, dims)];
                if(sl < 0 && wq > 0.0) atomicAdd(bad, 1);
                nodes[NS * p + k] = sl < 0 ? -1 : base + sl;
                wts[NS * p + k]   = sl < 0 ? 0.0 : wq;
                grad[NS * p + k]  = sl < 0 ? make_double3(0, 0, 0)
                                           : make_double3(dw[0][a] * w[1][b] * w[2][c], w[0][a] * dw[1][b] * w[2][c], w[0][a] * w[1][b] * dw[2][c]);
            }
}

template<int NS>
__device__ __forceinline__ void ug_state(const int* nodes, const double3* g, const double3* x, const UgGrid& G, double Jn, UgState& st)
{
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c) st.A[r][c] = (r == c) ? 1.0 : 0.0;
    for(int q = 0; q < NS; ++q)
    {
        int v = nodes[q];
        if(v < 0) continue;
        double3 u    = d3sub(x[v], ug_lattice(G, v));
        double  uu[3] = {u.x, u.y, u.z}, gg[3] = {g[q].x, g[q].y, g[q].z};
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) st.A[r][c] += uu[r] * gg[c];
    }
    double C[3][3];
    cofactor3(st.A, C);
    st.detA = st.A[0][0] * C[0][0] + st.A[0][1] * C[0][1] + st.A[0][2] * C[0][2];
    st.J    = Jn * st.detA;
}

// LIQ_UG_HG hourglass viscosity (trilinear stencil only): the fluid energy constrains just the 8 particle J's of a cell, so the
// cell's 4 x 3 hourglass modes (Flanagan-Belytschko base vectors xi*eta, eta*zeta, zeta*xi, xi*eta*zeta on the 8 corners,
// k = (a*2+b)*2+c) carry no stiffness and let the particles cluster inside the cell under gravity (sub-cell null-space creep).
// E_hg = 1/2 (eta V_p dt / dx^2) sum_m |Gamma_m . u|^2 on the step displacement u (dt^2-scaled potential) -> a viscous force
// eta V_p / dx^2 Gamma (Gamma . v) on those modes only (smooth / linear fields are untouched). NOTE (11-06): the hourglass
// modes also carry the physical sub-cell response of the particle quadrature (a half-cell load is a hourglass mode), so
// any eta strong enough to stop the null-space creep also stiffens resolved dynamics; LIQ_UG_REP (pair repulsion) is the
// stabiliser that acts only on actual particle clustering.
__device__ __forceinline__ void ug_hg_gamma(int k, double G4[4])
{
    const double sx = ((k >> 2) & 1) ? 1.0 : -1.0, sy = ((k >> 1) & 1) ? 1.0 : -1.0, sz = (k & 1) ? 1.0 : -1.0;
    G4[0] = sx * sy; G4[1] = sy * sz; G4[2] = sz * sx; G4[3] = sx * sy * sz;
}
template<int NS>
__device__ __forceinline__ void ug_hg_modes(const int* nodes, const double3* x, const UgGrid& G, double3 hm[4])
{
    for(int m = 0; m < 4; ++m) hm[m] = make_double3(0, 0, 0);
    for(int q = 0; q < NS; ++q)
    {
        int v = nodes[q];
        if(v < 0) continue;
        double3 u = d3sub(x[v], ug_lattice(G, v));
        double  G4[4];
        ug_hg_gamma(q, G4);
        for(int m = 0; m < 4; ++m) hm[m] = d3add(hm[m], d3scale(u, G4[m]));
    }
}

// psi(J), psi'(J), psi''(J) per unit rest volume: pressure (tension branch scaled by tr) + volume barrier below J_act
__device__ __forceinline__ void ug_psi(double J, double kappa, double tr, double jact, double kb, double& e, double& d1, double& d2)
{
    double k = (J < 1.0) ? kappa : kappa * tr;
    e  = 0.5 * k * (J - 1.0) * (J - 1.0);
    d1 = k * (J - 1.0);
    d2 = k;
    if(kb > 0.0 && jact > 0.0 && J < jact && J > 0.0)
    {
        double s = J / jact, L = log(s), g = s - 1.0;
        e  += kb * g * g * L * L;
        d1 += kb * (2.0 * g * L * L + 2.0 * g * g * L / s) / jact;
        d2 += kb * (2.0 * L * L + 8.0 * g * L / s + 2.0 * g * g * (1.0 - L) / (s * s)) / (jact * jact);
    }
}

template<int NA>
__global__ void k_ug_assemble(int n, const int* ug_nodes, const double3* ug_grad, const double* Vp0, const double* Jp, const double3* vert, UgGrid G,
                              double kappa, double tr, double jact, double kb, double dt, int fem_off, int global_offset, Eigen::Matrix3d* vals, int* rows, int* cols,
                              double hg_eta = 0.0)
{
    constexpr int NS = UgN<NA>::NS, NB = UgN<NA>::NB;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    const int*     nodes = ug_nodes + NS * p;
    const double3* g     = ug_grad + NS * p;
    int            toff  = global_offset + NB * p;
    int            fallback = -1;
    for(int q = 0; q < NS; ++q)
        if(nodes[q] >= 0) { fallback = nodes[q]; break; }
    // the NB upper-triangle block slots of the NS x NS node block matrix, zero first (stable layout)
    {
        int kk = 0;
        for(int q = 0; q < NS; ++q)
            for(int r = q; r < NS; ++r, ++kk)
            {
                int a = nodes[q] >= 0 ? nodes[q] : fallback, b = nodes[r] >= 0 ? nodes[r] : fallback;
                if(a < 0) a = b = 0;   // particle without any active node (cannot happen after k_ug_stencil passes)
                int row = a + fem_off, col = b + fem_off;
                rows[toff + kk] = row <= col ? row : col;
                cols[toff + kk] = row <= col ? col : row;
                vals[toff + kk].setZero();
            }
    }
    if(fallback < 0) return;
    const double V = Vp0[p], Jn = Jp[p];
    UgState      st;
    ug_state<NS>(nodes, g, vert, G, Jn, st);
    if(!(V > 0.0) || !isfinite(st.J) || !(st.detA > 0.0)) return;
    double e, psi1, psi2;
    ug_psi(st.J, kappa, tr, jact, kb, e, psi1, psi2);
    double C[3][3];
    cofactor3(st.A, C);
    const double dt2 = dt * dt;
    // gradient: dE/du_q = dt^2 V psi'(J) Jn cof(A) g_q
    const double s0 = dt2 * V * psi1 * Jn;
    for(int q = 0; q < NS; ++q)
    {
        if(nodes[q] < 0) continue;
        double gq[3] = {g[q].x, g[q].y, g[q].z};
        for(int a = 0; a < 3; ++a)
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + a) * BINNED_K, s0 * (C[a][0] * gq[0] + C[a][1] * gq[1] + C[a][2] * gq[2]));
    }
    // LIQ_UG_HG: hourglass viscosity gradient (trilinear stencil only); its Hessian is added to the node blocks below.
    // Viscous force eta V_p / dx^2 Gamma (Gamma . v) with v = u / dt -> in the dt^2-scaled incremental potential the
    // coefficient is eta V_p dt / dx^2 (NOT 1/dt: that was 1/dt^2 = 40000x too strong in the first sweep).
    const double chg = (NA == 2 && hg_eta > 0.0) ? hg_eta * V * dt / (G.dx * G.dx) : 0.0;
    if(chg > 0.0)
    {
        double3 hm[4];
        ug_hg_modes<NS>(nodes, vert, G, hm);
        for(int q = 0; q < NS; ++q)
        {
            if(nodes[q] < 0) continue;
            double  G4[4];
            ug_hg_gamma(q, G4);
            double3 f = make_double3(0, 0, 0);
            for(int m = 0; m < 4; ++m) f = d3add(f, d3scale(hm[m], G4[m]));
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 0) * BINNED_K, chg * f.x);
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 1) * BINNED_K, chg * f.y);
            binned_deposit(g_gbin + ((size_t)nodes[q] * 3 + 2) * BINNED_K, chg * f.z);
        }
    }
    // Hessian w.r.t. A (9x9, index 3a+b), projected to PSD
    Eigen::Matrix<double, 9, 9> HA;
    for(int a = 0; a < 3; ++a)
        for(int b = 0; b < 3; ++b)
            for(int c = 0; c < 3; ++c)
                for(int d = 0; d < 3; ++d)
                {
                    double dcof = 0.0;
                    for(int m = 0; m < 3; ++m)
                        for(int q = 0; q < 3; ++q)
                        {
                            double e1 = levi(a, c, m);
                            if(e1 == 0.0) continue;
                            double e2 = levi(b, d, q);
                            if(e2 == 0.0) continue;
                            dcof += e1 * e2 * st.A[m][q];
                        }
                    HA(3 * a + b, 3 * c + d) = V * (psi2 * Jn * Jn * C[a][b] * C[c][d] + psi1 * Jn * dcof);
                }
    HA = 0.5 * (HA + HA.transpose().eval());
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> es(HA);
        if(es.eigenvalues()[0] < 0.0)
        {
            Eigen::Matrix<double, 9, 9> D = Eigen::Matrix<double, 9, 9>::Zero();
            for(int i = 0; i < 9; ++i)
                if(es.eigenvalues()[i] > 0.0) D(i, i) = es.eigenvalues()[i];
            HA = es.eigenvectors() * D * es.eigenvectors().transpose();
        }
    }
    HA *= dt2;
    int kk = 0;
    for(int q = 0; q < NS; ++q)
        for(int r = q; r < NS; ++r, ++kk)
        {
            if(nodes[q] < 0 || nodes[r] < 0) continue;
            double          gq[3] = {g[q].x, g[q].y, g[q].z}, gr[3] = {g[r].x, g[r].y, g[r].z};
            Eigen::Matrix3d B;
            for(int a = 0; a < 3; ++a)
                for(int c = 0; c < 3; ++c)
                {
                    double acc = 0.0;
                    for(int b = 0; b < 3; ++b)
                        for(int d = 0; d < 3; ++d) acc += gq[b] * HA(3 * a + b, 3 * c + d) * gr[d];
                    B(a, c) = acc;
                }
            if(chg > 0.0)
            {   // hourglass viscosity Hessian: chg * sum_m Gamma_m(q) Gamma_m(r) * I
                double Gq[4], Gr[4], s = 0.0;
                ug_hg_gamma(q, Gq);
                ug_hg_gamma(r, Gr);
                for(int m = 0; m < 4; ++m) s += Gq[m] * Gr[m];
                B += chg * s * Eigen::Matrix3d::Identity();
            }
            int row = nodes[q] + fem_off, col = nodes[r] + fem_off;
            if(q == r) B = 0.5 * (B + B.transpose().eval());
            vals[toff + kk] = (row <= col) ? B : Eigen::Matrix3d(B.transpose());
        }
}

template<int NA>
__global__ void k_ug_energy(int n, const int* ug_nodes, const double3* ug_grad, const double* Vp0, const double* Jp, const double3* vert, UgGrid G,
                            double kappa, double tr, double jact, double kb, double* partial, double hg_eta = 0.0, double dt = 1.0)
{
    constexpr int NS = UgN<NA>::NS;
    extern __shared__ double sh[];
    int    p = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if(p < n)
    {
        UgState st;
        ug_state<NS>(ug_nodes + NS * p, ug_grad + NS * p, vert, G, Jp[p], st);
        double V = Vp0[p];
        if(V > 0.0)
        {
            if(!(st.detA > 0.0) || !isfinite(st.J) || (kb > 0.0 && jact > 0.0 && !(st.J > 0.0)))
                e = 1e300;   // inverted / collapsed particle: reject the step (feasible-step role of the recovered engine)
            else
            {
                double d1, d2;
                ug_psi(st.J, kappa, tr, jact, kb, e, d1, d2);
                e *= V;
                if(NA == 2 && hg_eta > 0.0)
                {   // LIQ_UG_HG hourglass viscosity (see ug_hg_gamma)
                    double3 hm[4];
                    ug_hg_modes<NS>(ug_nodes + NS * p, vert, G, hm);
                    double s = 0.0;
                    for(int m = 0; m < 4; ++m) s += hm[m].x * hm[m].x + hm[m].y * hm[m].y + hm[m].z * hm[m].z;
                    e += 0.5 * hg_eta * V * dt / (G.dx * G.dx) * s;
                }
            }
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

// commit after the Newton solve: x += sum w u, v = dx/dt, C = D/dt (D = sum u grad w^T), J *= det(I + D).
// LIQ_BDF2 (vi_vel != nullptr): v = sum w v_i with the nodes' BDF2 velocity (k_read_node_vel_bdf2, per lattice node id);
// bh_vp / bh_Cp keep this step's incoming particle velocity / affine velocity as next step's history.
template<int NA>
__global__ void k_ug_commit(int n, const int* ug_nodes, const double* ug_w, const double3* ug_grad, const double3* vert, UgGrid G, double dt, int apic,
                            double3 lo, double3 hi, double3* xp, double3* vp, double* Cp, double* Jp, int* bad,
                            const double3* vi_vel = nullptr, double3* bh_vp = nullptr, double* bh_Cp = nullptr)
{
    constexpr int NS = UgN<NA>::NS;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    double3 dxp = make_double3(0, 0, 0), vb = make_double3(0, 0, 0);
    double  D[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for(int q = 0; q < NS; ++q)
    {
        int v = ug_nodes[NS * p + q];
        if(v < 0) continue;
        double3 u = d3sub(vert[v], ug_lattice(G, v));
        double  w = ug_w[NS * p + q];
        dxp       = d3add(dxp, d3scale(u, w));
        if(vi_vel) vb = d3add(vb, d3scale(vi_vel[G.active[v - G.base]], w));
        double3 g = ug_grad[NS * p + q];
        double  uu[3] = {u.x, u.y, u.z}, gg[3] = {g.x, g.y, g.z};
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) D[r][c] += uu[r] * gg[c];
    }
    double A[3][3], Cf[3][3];
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c) A[r][c] = D[r][c] + (r == c ? 1.0 : 0.0);
    cofactor3(A, Cf);
    double det = A[0][0] * Cf[0][0] + A[0][1] * Cf[0][1] + A[0][2] * Cf[0][2];
    double Jn  = Jp[p] * det;
    if(!isfinite(Jn) || !(Jn > 0.0) || !d3finite(dxp)) { atomicAdd(bad, 1); return; }
    Jp[p] = Jn;
    double3 x = d3add(xp[p], dxp);
    x.x = fmin(fmax(x.x, lo.x), hi.x);
    x.y = fmin(fmax(x.y, lo.y), hi.y);
    x.z = fmin(fmax(x.z, lo.z), hi.z);
    xp[p] = x;
    if(bh_vp)
    {
        bh_vp[p] = vp[p];
        if(bh_Cp) for(int k = 0; k < 9; ++k) bh_Cp[9 * p + k] = Cp[9 * p + k];
    }
    vp[p] = vi_vel ? vb : d3scale(dxp, 1.0 / dt);
    if(apic)
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) Cp[9 * p + 3 * r + c] = D[r][c] / dt;
}

// LIQ_UG_REANCHOR: tent-smeared cell masses from the particles (centre stencil, as the main path's P2G) for the
// density re-anchor k_reanchor_J (j_mode 3 of the main path): interior particles take J = rho0 dx^3 / m_cell, so that
// particle-level compaction the tracked J cannot see (smooth-kernel clumping) builds pressure again.
__global__ void k_ug_cellmass(int n, const double3* xp, const double* mp, double3 org, double dx, int3 dims, double* mc, int* cflag)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p >= n) return;
    CentreStencil s = centre_stencil(xp[p], org, dx, dims);
    const double m = mp[p];
    for(int a = 0; a < 2; ++a)
        for(int b = 0; b < 2; ++b)
            for(int c = 0; c < 2; ++c)
            {
                double w = s.w[0][a] * s.w[1][b] * s.w[2][c];
                if(!(w > 0.0)) continue;
                int cid = cell_lin(s.b[0] + a, s.b[1] + b, s.b[2] + c, dims);
                atomicAdd(&mc[cid], w * m);
                cflag[cid] = 1;
            }
}

// LIQ_UG_FEASIBLE: the recovered engine's "parent volume feasible step". Along the Newton step x <- x - alpha d the trial
// map of particle p is A(alpha) = A0 - alpha B with A0 = I + sum_q u_q grad w_q^T and B = sum_q d_q grad w_q^T, so
// det A(alpha) = c0 - c1 alpha + c2 alpha^2 - c3 alpha^3 (c0 = det A0 > 0, c1 = cof(A0):B, c2 = cof(B):A0, c3 = det B).
// The block minimum over particles of safety x (first root in (0, 1]) bounds the line search before it starts, so the
// volume energy stays finite and smooth along the search (the infinite-energy rejection remains only as a never-hit guard).
__device__ __forceinline__ double ug_cubic(double c0, double c1, double c2, double c3, double a)
{
    return c0 + a * (-c1 + a * (c2 - a * c3));
}
template<int NA>
__global__ void k_ug_feasible(int n, const int* ug_nodes, const double3* ug_grad, const double3* vert, const double3* dir, UgGrid G, double safety, double* partial)
{
    constexpr int NS = UgN<NA>::NS;
    extern __shared__ double sh[];
    int    p     = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha = 1.0;
    if(p < n)
    {
        const int*     nodes = ug_nodes + NS * p;
        const double3* g     = ug_grad + NS * p;
        double A[3][3], B[3][3];
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) { A[r][c] = (r == c) ? 1.0 : 0.0; B[r][c] = 0.0; }
        for(int q = 0; q < NS; ++q)
        {
            int v = nodes[q];
            if(v < 0) continue;
            double3 u = d3sub(vert[v], ug_lattice(G, v)), d = dir[v];
            double  uu[3] = {u.x, u.y, u.z}, dd[3] = {d.x, d.y, d.z}, gg[3] = {g[q].x, g[q].y, g[q].z};
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c) { A[r][c] += uu[r] * gg[c]; B[r][c] += dd[r] * gg[c]; }
        }
        double CA[3][3], CB[3][3];
        cofactor3(A, CA);
        cofactor3(B, CB);
        double c0 = A[0][0] * CA[0][0] + A[0][1] * CA[0][1] + A[0][2] * CA[0][2];
        double c3 = B[0][0] * CB[0][0] + B[0][1] * CB[0][1] + B[0][2] * CB[0][2];
        double c1 = 0.0, c2 = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c) { c1 += CA[r][c] * B[r][c]; c2 += CB[r][c] * A[r][c]; }
        if(!(c0 > 0.0)) alpha = 0.0;                       // already inverted (never, after a feasible previous step): no step
        else if(isfinite(c1) && isfinite(c2) && isfinite(c3))
        {
            // the cubic is monotonic between its critical points: the first root lies in the first interval
            // [cand_k-1, cand_k] (candidates = critical points in (0,1) and the end point 1) whose right end has f <= 0
            double cand[3] = {1.0, 1.0, 1.0};
            int    nc      = 1;
            double qa = -3.0 * c3, qb = 2.0 * c2, qc = -c1;   // f'(a) = qa a^2 + qb a + qc
            if(fabs(qa) > 1e-300)
            {
                double disc = qb * qb - 4.0 * qa * qc;
                if(disc >= 0.0)
                {
                    double sq = sqrt(disc), r1 = (-qb - sq) / (2.0 * qa), r2 = (-qb + sq) / (2.0 * qa);
                    if(r1 > 0.0 && r1 < 1.0) cand[nc++] = r1;
                    if(r2 > 0.0 && r2 < 1.0) cand[nc++] = r2;
                }
            }
            else if(fabs(qb) > 1e-300)
            {
                double r1 = -qc / qb;
                if(r1 > 0.0 && r1 < 1.0) cand[nc++] = r1;
            }
            for(int i = 1; i < nc; ++i)                       // insertion sort (<= 3 values)
                for(int j = i; j > 0 && cand[j] < cand[j - 1]; --j) { double t = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = t; }
            double lo = 0.0, hi = -1.0;
            for(int k = 0; k < nc; ++k)
            {
                if(ug_cubic(c0, c1, c2, c3, cand[k]) <= 0.0) { hi = cand[k]; break; }
                lo = cand[k];
            }
            if(hi > 0.0)
            {
                for(int it = 0; it < 40; ++it)
                {
                    double m = 0.5 * (lo + hi);
                    if(ug_cubic(c0, c1, c2, c3, m) > 0.0) lo = m; else hi = m;
                }
                alpha = fmax(0.0, safety * lo);
            }
        }
    }
    sh[threadIdx.x] = alpha;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sh[threadIdx.x] = fmin(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}
