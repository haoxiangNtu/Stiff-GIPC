__global__ void _calSelfCloseVal(const double3* _vertexes,
                                 const int4*    _collisionPair,
                                 int4*          _close_collisionPair,
                                 double*        _close_collisionVal,
                                 uint32_t*      _close_cpNum,
                                 double         dTol,
                                 int            number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < dTol)
    {
        int tidx                   = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx]  = dist2;
    }
}

__global__ void _checkSelfCloseVal(const double3* _vertexes,
                                   int*           _isChange,
                                   int4*          _close_collisionPair,
                                   double*        _close_collisionVal,
                                   int            number,
                                   int*           _isChange_grp = nullptr,
                                   const int*     p2g           = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI = _close_collisionPair[idx];
    double dist2   = _selfConstraintVal(_vertexes, MMCVIDI);
    if(dist2 < _close_collisionVal[idx])
    {
        *_isChange = 1;
        // [multi-env per-group κ] flag only THIS pair's env (intra-env after P1).
        if(_isChange_grp && p2g)
        { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) _isChange_grp[_gg] = 1; }  /* [-1 guard] */
    }
}


__global__ void _reduct_MSelfDist(const double3* _vertexes,
                                  int4*          _collisionPairs,
                                  double2*       _queue,
                                  int            number)
{
    int                       idof = blockIdx.x * blockDim.x;
    int                       idx  = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    double2 temp = make_double2(0.0, 0.0);
    if(idx < number)
    {
        int4   MMCVIDI = _collisionPairs[idx];
        double tempv   = _selfConstraintVal(_vertexes, MMCVIDI);
        temp = make_double2(1.0 / tempv, tempv);
    }
    int     warpTid = threadIdx.x % 32;
    int     warpId  = (threadIdx.x >> 5);
    double  nextTp;
    int     warpNum;
    //int tidNum = 32;
    if(blockIdx.x == gridDim.x - 1)
    {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
        double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
        temp.x         = std::max(temp.x, tempMin);
        temp.y         = std::max(temp.y, tempMax);
    }
    if(warpTid == 0)
    {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if(warpId != 0)
        return;
    if(warpNum > 1)
    {
        //	tidNum = warpNum;
        temp = (warpTid < warpNum) ? sdata[warpTid] : make_double2(0.0, 0.0);

        //	warpNum = ((tidNum + 31) >> 5);
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            double tempMin = __shfl_down_sync(0xffffffff, temp.x, i);
            double tempMax = __shfl_down_sync(0xffffffff, temp.y, i);
            temp.x         = std::max(temp.x, tempMin);
            temp.y         = std::max(temp.y, tempMax);
        }
    }
    if(threadIdx.x == 0)
    {
        _queue[blockIdx.x] = temp;
    }
}

__global__ void _calFrictionGradient_gd(const double3* _vertexes,
                                        const double3* _o_vertexes,
                                        const double3* _normal,
                                        const const uint32_t* _last_collisionPair_gd,
                                        double3* _gradient,
                                        int      number,
                                        double   dt,
                                        double   eps2,
                                        double*  lastH,
                                        double   coef,
                                        const double* vert_mu_gd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double   eps    = sqrt(eps2);
    double3  normal = *_normal;
    uint32_t gidx   = _last_collisionPair_gd[idx];
    double3  Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3  VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);
    if(VProjMag2 > eps2)
    {
        double3 gdf =
            __GEIGEN__::__s_vec_multiply(VProj, (vert_mu_gd ? vert_mu_gd[gidx] : coef) * lastH[idx] / sqrt(VProjMag2));
        /*_gfxAdd(gidx, 0, gdf.x);
        _gfxAdd(gidx, 1, gdf.y);
        _gfxAdd(gidx, 2, gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
    else
    {
        double3 gdf = __GEIGEN__::__s_vec_multiply(VProj, (vert_mu_gd ? vert_mu_gd[gidx] : coef) * lastH[idx] / eps);
        /*_gfxAdd(gidx, 0, gdf.x);
        _gfxAdd(gidx, 1, gdf.y);
        _gfxAdd(gidx, 2, gdf.z);*/
        _gradient[gidx] = __GEIGEN__::__add(_gradient[gidx], gdf);
    }
}

__global__ void _calFrictionGradient(const double3*    _vertexes,
                                     const double3*    _o_vertexes,
                                     const const int4* _last_collisionPair,
                                     double3*          _gradient,
                                     int               number,
                                     double            dt,
                                     double2*          distCoord,
                                     __GEIGEN__::Matrix3x2d* tanBasis,
                                     double                  eps2,
                                     double*                 lastH,
                                     double                  coef,
                                     const double*           vert_mu)
{
    double eps = std::sqrt(eps2);
    int    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    const double mu = _pair_mu(MMCVIDI, vert_mu, coef);  // [per-body friction]
    double3 relDX3D;
    if(MMCVIDI.x >= 0)
    {
        Friction::computeRelDX_EE(
            __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x,
            distCoord[idx].y,
            relDX3D);

        __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
        double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
        double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
        if(relDXSqNorm > eps2)
        {
            relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __GEIGEN__::Vector12 TTTDX;
        Friction::liftRelDXTanToMesh_EE(
            relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * mu);
        {
            _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
            _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
            _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
            _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
            _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
            _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
            _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
            _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            _gfxAdd(MMCVIDI.w, 0, TTTDX.v[9]);
            _gfxAdd(MMCVIDI.w, 1, TTTDX.v[10]);
            _gfxAdd(MMCVIDI.w, 2, TTTDX.v[11]);
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            MMCVIDI.x = v0I;

            Friction::computeRelDX_PP(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __GEIGEN__::Vector6 TTTDX;
            Friction::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __GEIGEN__::__s_vec6_multiply(TTTDX, lastH[idx] * mu);
            {
                _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
                _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
                _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
                _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
                _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
                _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PE(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector9 TTTDX;
            Friction::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __GEIGEN__::__s_vec9_multiply(TTTDX, lastH[idx] * mu);
            {
                _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
                _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
                _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
                _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
                _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
                _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
                _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
                _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
                _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            }
        }
        else
        {
            MMCVIDI.x = v0I;
            Friction::computeRelDX_PT(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x,
                distCoord[idx].y,
                relDX3D);

            __GEIGEN__::Matrix2x3d tB_T = __GEIGEN__::__Transpose3x2(tanBasis[idx]);
            double2 relDX = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            if(relDXSqNorm > eps2)
            {
                relDX = __GEIGEN__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __GEIGEN__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __GEIGEN__::Vector12 TTTDX;
            Friction::liftRelDXTanToMesh_PT(
                relDX, tanBasis[idx], distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = __GEIGEN__::__s_vec12_multiply(TTTDX, lastH[idx] * mu);

            _gfxAdd(MMCVIDI.x, 0, TTTDX.v[0]);
            _gfxAdd(MMCVIDI.x, 1, TTTDX.v[1]);
            _gfxAdd(MMCVIDI.x, 2, TTTDX.v[2]);
            _gfxAdd(MMCVIDI.y, 0, TTTDX.v[3]);
            _gfxAdd(MMCVIDI.y, 1, TTTDX.v[4]);
            _gfxAdd(MMCVIDI.y, 2, TTTDX.v[5]);
            _gfxAdd(MMCVIDI.z, 0, TTTDX.v[6]);
            _gfxAdd(MMCVIDI.z, 1, TTTDX.v[7]);
            _gfxAdd(MMCVIDI.z, 2, TTTDX.v[8]);
            _gfxAdd(MMCVIDI.w, 0, TTTDX.v[9]);
            _gfxAdd(MMCVIDI.w, 1, TTTDX.v[10]);
            _gfxAdd(MMCVIDI.w, 2, TTTDX.v[11]);
        }
    }
}


// [Step B] optional per-contact export hook. When _ec_out_pair != nullptr, each
// contact also writes (bodyA,bodyB) and the physical contact force on bodyA
// (N) = -(sum of this contact's gradient over bodyA's verts)/dt^2. bodyA = body
// of the first vertex; same-body (self) contacts get bodyB==bodyA (consumer
// skips). nullptr -> no-op (solve path unchanged).
__device__ inline void _ec_emit(int idx, int2* op, double3* of, const int* pbid,
                                double inv_dt2, int va, int vb, int vc, int vd,
                                int nv, const double* g)
{
    if(!op)
        return;
    int    vs[4] = {va, vb, vc, vd};
    int    bA = pbid[va], bB = pbid[va];
    double fx = 0, fy = 0, fz = 0;
    for(int k = 0; k < nv; k++)
    {
        int b = pbid[vs[k]];
        if(b == bA)
        {
            fx += g[3 * k]; fy += g[3 * k + 1]; fz += g[3 * k + 2];
        }
        else
            bB = b;
    }
    op[idx] = make_int2(bA, bB);
    of[idx] = make_double3(-fx * inv_dt2, -fy * inv_dt2, -fz * inv_dt2);
}

__global__ void _calBarrierGradient(const double3*    _vertexes,
                                    const double3*    _rest_vertexes,
                                    const const int4* _collisionPair,
                                    double3*          _gradient,
                                    double            dHat,
                                    double            Kappa_scalar,
                                    int               number,
                                    const double*     kappa_grp = nullptr,
                                    const int*        p2g       = nullptr,
                                    int2*             _ec_out_pair  = nullptr,
                                    double3*          _ec_out_force = nullptr,
                                    const int*        _ec_pbid      = nullptr,
                                    double            _ec_inv_dt2   = 0.0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4   MMCVIDI   = _collisionPair[idx];
    // [multi-env per-group κ] nullptr → scalar (baseline). Before MMCVIDI mutation.
    double Kappa = Kappa_scalar;
    if(kappa_grp && p2g)
    { int _gv = (MMCVIDI.x >= 0) ? MMCVIDI.x : (-MMCVIDI.x - 1); int _gg = p2g[_gv]; if(_gg >= 0) Kappa = kappa_grp[_gg]; }  /* [-1 guard] wildcard -> scalar */
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w >= 0)
        {
#ifdef NEWF
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else

            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                v0, __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double  dis    = __GEIGEN__::__v_vec_dot(v1, normal);
            if(dis < 0)
            {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis    = -dis;
            }

            double3 pos2 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.z],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 =
                __GEIGEN__::__add(_vertexes[MMCVIDI.w],
                                  __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = __GEIGEN__::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = __GEIGEN__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
            }
        }
        else
        {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 =
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
            double I1 = c * c;
            if(I1 == 0)
                return;
            double dis;
            _d_EE(_vertexes[MMCVIDI.x],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            double I2 = dis / dHat;
            dis       = sqrt(dis);

            __GEIGEN__::Matrix3x3d F;
            __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                        _rest_vertexes[MMCVIDI.y],
                                        _rest_vertexes[MMCVIDI.z],
                                        _rest_vertexes[MMCVIDI.w]);

            __GEIGEN__::Matrix3x3d g1, g2;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
            __GEIGEN__::__M_Mat_multiply(F, nn, g1);
            nn = __GEIGEN__::__v_vec_toMat(n2, n2);
            __GEIGEN__::__M_Mat_multiply(F, nn, g2);

            __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
            __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

            __GEIGEN__::Matrix12x9d PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     dHat_sqrt,
                     PFPx);


#if (RANK == 1)
            double p1 = Kappa * 2
                        * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = Kappa * 2
                        * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                           * (I2 + 2 * I2 * log(I2) - 1))
                        / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                        / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2
                        * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                           * (I2 - 1) * (I2 - 1))
                        / (eps_x * eps_x);
            double p2 = -Kappa * 2
                        * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                           * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                        / (I2 * (eps_x * eps_x));
#endif


            __GEIGEN__::Vector9 flatten_pk1 =
                __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                   __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
            }
        }
    }
    else
    {
        int v0I = -MMCVIDI.x - 1;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.z],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.w]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);
#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis                            = sqrt(dis);
                double              d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Vector6 PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5  = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                double flatten_pk1 =
                    fnn * 2 * Kappa
                    * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif (RANK == 2)
                //double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                double flatten_pk1 = fnn * 2
                                     * (2 * Kappa * dHat * dHat * log(I5)
                                        * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                                     / I5;
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5;

#elif (RANK == 3)
                double flatten_pk1 = fnn * -2
                                     * (Kappa * dHat * dHat * log(I5) * log(I5)
                                        * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3))
                                     / I5;
#elif (RANK == 4)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5;
#elif (RANK == 5)
                double flatten_pk1 =
                    fnn * -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5;
#elif (RANK == 6)
                double flatten_pk1 =
                    fnn
                    * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5;
#endif

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds  = v0;
                double  dis = __GEIGEN__::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal =
                    __GEIGEN__::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec    = __GEIGEN__::__v_vec_cross(vec_normal, target);
                double  cos    = __GEIGEN__::__v_vec_dot(vec_normal, target);
                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __GEIGEN__::Vector6 PDmPx;
                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0    = uv1 - uv0;
                double Dm    = u0;  //PFPx
                double DmInv = 1 / u0;

                double3 F  = __GEIGEN__::__s_vec_multiply(Ds, DmInv);
                double  I5 = __GEIGEN__::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = __GEIGEN__::__s_vec_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif
                __GEIGEN__::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                __GEIGEN__::Vector6 gradient_vec =
                    __GEIGEN__::__M6x3_v3_multiply(__GEIGEN__::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, v0I, MMCVIDI.y, 0, 0, 2, gradient_vec.v);
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 =
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = __GEIGEN__::__norm(__GEIGEN__::__v_vec_cross(v0, v1)) /*/ __GEIGEN__::__norm(v0)*/;
                double I1 = c * c;
                if(I1 == 0)
                    return;
                double dis;
                _d_PE(_vertexes[MMCVIDI.x],
                      _vertexes[MMCVIDI.y],
                      _vertexes[MMCVIDI.z],
                      dis);
                double I2 = dis / dHat;
                dis       = sqrt(dis);

                __GEIGEN__::Matrix3x3d F;
                __GEIGEN__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = _compute_epx(_rest_vertexes[MMCVIDI.x],
                                            _rest_vertexes[MMCVIDI.w],
                                            _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z]);

                __GEIGEN__::Matrix3x3d g1, g2;

                __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(n1, n1);
                __GEIGEN__::__M_Mat_multiply(F, nn, g1);
                nn = __GEIGEN__::__v_vec_toMat(n2, n2);
                __GEIGEN__::__M_Mat_multiply(F, nn, g2);

                __GEIGEN__::Vector9 flatten_g1 = __GEIGEN__::__Mat3x3_to_vec9_double(g1);
                __GEIGEN__::Vector9 flatten_g2 = __GEIGEN__::__Mat3x3_to_vec9_double(g2);

                __GEIGEN__::Matrix12x9d PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x],
                         _vertexes[MMCVIDI.y],
                         _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w],
                         dHat_sqrt,
                         PFPx);

#if (RANK == 1)
                double p1 =
                    Kappa * 2
                    * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                    / (eps_x * eps_x);
                double p2 = Kappa * 2
                            * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1)
                               * (I2 + 2 * I2 * log(I2) - 1))
                            / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * log(I2) * log(I2)
                               * (I1 - eps_x) * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (I2 + I2 * log(I2) - 1))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2))
                            / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2
                            * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x)
                               * (I2 - 1) * (I2 - 1))
                            / (eps_x * eps_x);
                double p2 = -Kappa * 2
                            * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x)
                               * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3))
                            / (I2 * (eps_x * eps_x));
#endif
                __GEIGEN__::Vector9 flatten_pk1 =
                    __GEIGEN__::__add9(__GEIGEN__::__s_vec9_multiply(flatten_g1, p1),
                                       __GEIGEN__::__s_vec9_multiply(flatten_g2, p2));
                __GEIGEN__::Vector12 gradient_vec =
                    __GEIGEN__::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    _gfxAdd(MMCVIDI.x, 0, gradient_vec.v[0]);
                    _gfxAdd(MMCVIDI.x, 1, gradient_vec.v[1]);
                    _gfxAdd(MMCVIDI.x, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
                    _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
                    _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
                }
            }
            else
            {
#ifdef NEWF
                double dis;
                _d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis                               = sqrt(dis);
                double                 d_hat_sqrt = sqrt(dHat);
                __GEIGEN__::Matrix9x4d PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double              I5 = pow(dis / d_hat_sqrt, 2);
                __GEIGEN__::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if (RANK == 1)


                double judge = (2 * dHat * dHat
                                * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5
                                   - 6 * I5 * I5 * log(I5) + 1))
                               / I5;
                double judge2 =
                    2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                    / I5 * dis / d_hat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
                //__GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                double judge =
                    -(4 * dHat * dHat
                      * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                         + 6 * I5 * log(I5) - 2 * I5 * I5
                         + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                    / I5;
                double judge2 =
                    2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                    / I5 * dis / dHat_sqrt;
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
                //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = __GEIGEN__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector4 flatten_pk1 = __GEIGEN__::__s_vec4_multiply(
                    fnn,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                __GEIGEN__::Matrix3x2d Ds;
                __GEIGEN__::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal =
                    __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = __GEIGEN__::__v_vec_cross(triangle_normal, target);
                double cos = __GEIGEN__::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                double dis = __GEIGEN__::__v_vec_dot(
                    __GEIGEN__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __GEIGEN__::Matrix3x3d rotation;
                __GEIGEN__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __GEIGEN__::Matrix9x4d PDmPx;

                if(cos + 1 == 0)
                {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else
                {
                    __GEIGEN__::Matrix3x3d cross_vec;
                    __GEIGEN__::__set_Mat_val(
                        cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = __GEIGEN__::__Mat_add(
                        rotation,
                        __GEIGEN__::__Mat_add(cross_vec,
                                              __GEIGEN__::__S_Mat_multiply(
                                                  __GEIGEN__::__M_Mat_multiply(cross_vec, cross_vec),
                                                  1.0 / (1 + cos))));
                }

                double3 pos0 = __GEIGEN__::__add(
                    _vertexes[v0I],
                    __GEIGEN__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = __GEIGEN__::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 =
                    __GEIGEN__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = __GEIGEN__::__M_v_multiply(rotation, edge_normal);

                double2 uv0    = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1    = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2    = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = __GEIGEN__::__minus_v2(uv1, uv0);
                double2 u1 = __GEIGEN__::__minus_v2(uv2, uv0);

                __GEIGEN__::Matrix2x2d Dm;

                __GEIGEN__::__set_Mat2x2_val_column(Dm, u0, u1);

                __GEIGEN__::Matrix2x2d DmInv;
                __GEIGEN__::__Inverse2x2(Dm, DmInv);

                __GEIGEN__::Matrix3x2d F = __GEIGEN__::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = __GEIGEN__::__M3x2_v2_multiply(F, normal);
                double  I5  = __GEIGEN__::__squaredNorm(FxN);

                __GEIGEN__::Matrix3x2d fnn;

                __GEIGEN__::Matrix2x2d nn = __GEIGEN__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __GEIGEN__::__M3x2_M2x2_Multiply(F, nn);

                __GEIGEN__::Vector6 tmp = __GEIGEN__::__Mat3x2_to_vec6_double(fnn);


#if (RANK == 1)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                        / I5);
#elif (RANK == 3)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                           * (3 * I5 + 2 * I5 * log(I5) - 3))
                        / I5);
#elif (RANK == 4)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                        / I5);
#elif (RANK == 5)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    -2
                        * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                           * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                        / I5);
#elif (RANK == 6)
                __GEIGEN__::Vector6 flatten_pk1 = __GEIGEN__::__s_vec6_multiply(
                    tmp,
                    (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                     * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                        / I5);
#endif

                __GEIGEN__::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

                __GEIGEN__::Vector9 gradient_vec =
                    __GEIGEN__::__M9x6_v6_multiply(__GEIGEN__::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    _gfxAdd(v0I, 0, gradient_vec.v[0]);
                    _gfxAdd(v0I, 1, gradient_vec.v[1]);
                    _gfxAdd(v0I, 2, gradient_vec.v[2]);
                    _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
                    _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
                    _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
                    _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
                    _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
                    _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
                    _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, v0I, MMCVIDI.y, MMCVIDI.z, 0, 3, gradient_vec.v);
                }
            }
        }
        else
        {
#ifdef NEWF
            double dis;
            _d_PT(_vertexes[v0I],
                  _vertexes[MMCVIDI.y],
                  _vertexes[MMCVIDI.z],
                  _vertexes[MMCVIDI.w],
                  dis);
            dis                                = sqrt(dis);
            double                  d_hat_sqrt = sqrt(dHat);
            __GEIGEN__::Matrix12x9d PFPxT;
            pFpx_pt2(_vertexes[v0I],
                     _vertexes[MMCVIDI.y],
                     _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w],
                     d_hat_sqrt,
                     PFPxT);
            double              I5 = pow(dis / d_hat_sqrt, 2);
            __GEIGEN__::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] =
                tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8]                = dis / d_hat_sqrt;
#else
            double3 v0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __GEIGEN__::Matrix3x3d Ds;
            __GEIGEN__::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = __GEIGEN__::__normalized(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __GEIGEN__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double  dis    = __GEIGEN__::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            __GEIGEN__::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if(dis > 0)
            {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else
            {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = __GEIGEN__::__add(
                _vertexes[v0I], __GEIGEN__::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = __GEIGEN__::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = __GEIGEN__::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = __GEIGEN__::__minus(_vertexes[MMCVIDI.w], pos0);

            __GEIGEN__::Matrix3x3d Dm, DmInv;
            __GEIGEN__::__set_Mat_val_column(Dm, u0, u1, u2);

            __GEIGEN__::__Inverse(Dm, DmInv);

            __GEIGEN__::Matrix3x3d F;  //, Ftest;
            __GEIGEN__::__M_Mat_multiply(Ds, DmInv, F);
            //__GEIGEN__::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = __GEIGEN__::__M_v_multiply(F, normal);
            double  I5  = __GEIGEN__::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            __GEIGEN__::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            __GEIGEN__::Matrix3x3d fnn;

            __GEIGEN__::Matrix3x3d nn = __GEIGEN__::__v_vec_toMat(normal, normal);

            __GEIGEN__::__M_Mat_multiply(F, nn, fnn);

            __GEIGEN__::Vector9 tmp = __GEIGEN__::__Mat3x3_to_vec9_double(fnn);
#endif


#if (RANK == 1)


            double judge =
                (2 * dHat * dHat
                 * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1))
                / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1))
                            / I5 * dis / d_hat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
            //__GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat
                             * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5)
                                + 6 * I5 * log(I5) - 2 * I5 * I5
                                + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2))
                           / I5;
            double judge2 =
                2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1))
                / I5 * dis / dHat_sqrt;
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
            //flatten_pk1 = __GEIGEN__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1)
                       * (3 * I5 + 2 * I5 * log(I5) - 3))
                    / I5);
#elif (RANK == 4)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                 * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2))
                    / I5);
#elif (RANK == 5)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                -2
                    * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5)
                       * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5))
                    / I5);
#elif (RANK == 6)
            __GEIGEN__::Vector9 flatten_pk1 = __GEIGEN__::__s_vec9_multiply(
                tmp,
                (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5)
                 * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3))
                    / I5);
#endif

#ifdef NEWF
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __GEIGEN__::Vector12 gradient_vec =
                __GEIGEN__::__M12x9_v9_multiply(__GEIGEN__::__Transpose9x12(PFPx), flatten_pk1);
#endif

            _gfxAdd(v0I, 0, gradient_vec.v[0]);
            _gfxAdd(v0I, 1, gradient_vec.v[1]);
            _gfxAdd(v0I, 2, gradient_vec.v[2]);
            _gfxAdd(MMCVIDI.y, 0, gradient_vec.v[3]);
            _gfxAdd(MMCVIDI.y, 1, gradient_vec.v[4]);
            _gfxAdd(MMCVIDI.y, 2, gradient_vec.v[5]);
            _gfxAdd(MMCVIDI.z, 0, gradient_vec.v[6]);
            _gfxAdd(MMCVIDI.z, 1, gradient_vec.v[7]);
            _gfxAdd(MMCVIDI.z, 2, gradient_vec.v[8]);
            _gfxAdd(MMCVIDI.w, 0, gradient_vec.v[9]);
            _gfxAdd(MMCVIDI.w, 1, gradient_vec.v[10]);
            _gfxAdd(MMCVIDI.w, 2, gradient_vec.v[11]);
            _ec_emit(idx, _ec_out_pair, _ec_out_force, _ec_pbid, _ec_inv_dt2, MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w, 4, gradient_vec.v);
        }
    }
}

