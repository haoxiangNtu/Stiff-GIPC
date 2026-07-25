__device__ void _d_PP(const double3& v0, const double3& v1, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__ void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b    = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1),
                                          __GEIGEN__::__minus(v3, v1));
    double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);  //(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d)
{
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)))
        / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__ void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0),
                                          __GEIGEN__::__minus(v3, v2));  //(v1 - v0).cross(v3 - v2);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);  //(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}


__device__ void _d_EEParallel(const double3& v0,
                              const double3& v1,
                              const double3& v2,
                              const double3& v3,
                              double&        d)
{
    double3 b = __GEIGEN__::__v_vec_cross(
        __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)),
        __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);  //(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__ double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))
           * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__ double _compute_epx_cp(const double3& v0,
                                  const double3& v1,
                                  const double3& v2,
                                  const double3& v3)
{
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))
           * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

__device__ int _dType_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
    __GEIGEN__::Matrix3x3d D, D1, D2;

    __GEIGEN__::__set_Mat_val(D,
                              basis0.x,
                              basis1.x,
                              nVec.x,
                              basis0.y,
                              basis1.y,
                              nVec.y,
                              basis0.z,
                              basis1.z,
                              nVec.z);
    __GEIGEN__::__set_Mat_val(D1,
                              basis2.x,
                              basis1.x,
                              nVec.x,
                              basis2.y,
                              basis1.y,
                              nVec.y,
                              basis2.z,
                              basis1.z,
                              nVec.z);
    __GEIGEN__::__set_Mat_val(D2,
                              basis0.x,
                              basis2.x,
                              nVec.x,
                              basis0.y,
                              basis2.y,
                              nVec.y,
                              basis0.z,
                              basis2.z,
                              nVec.z);

    double2 param[3];
    param[0].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
    param[0].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

    if(param[0].x > 0 && param[0].x < 1 && param[0].y >= 0)
    {
        return 3;  // PE v1v2
    }
    else
    {
        basis0 = __GEIGEN__::__minus(v3, v2);
        basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
        basis2 = __GEIGEN__::__minus(v0, v2);

        __GEIGEN__::__set_Mat_val(D,
                                  basis0.x,
                                  basis1.x,
                                  nVec.x,
                                  basis0.y,
                                  basis1.y,
                                  nVec.y,
                                  basis0.z,
                                  basis1.z,
                                  nVec.z);
        __GEIGEN__::__set_Mat_val(D1,
                                  basis2.x,
                                  basis1.x,
                                  nVec.x,
                                  basis2.y,
                                  basis1.y,
                                  nVec.y,
                                  basis2.z,
                                  basis1.z,
                                  nVec.z);
        __GEIGEN__::__set_Mat_val(D2,
                                  basis0.x,
                                  basis2.x,
                                  nVec.x,
                                  basis0.y,
                                  basis2.y,
                                  nVec.y,
                                  basis0.z,
                                  basis2.z,
                                  nVec.z);

        param[1].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
        param[1].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

        if(param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0)
        {
            return 4;  // PE v2v3
        }
        else
        {
            basis0 = __GEIGEN__::__minus(v1, v3);
            basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
            basis2 = __GEIGEN__::__minus(v0, v3);

            __GEIGEN__::__set_Mat_val(D,
                                      basis0.x,
                                      basis1.x,
                                      nVec.x,
                                      basis0.y,
                                      basis1.y,
                                      nVec.y,
                                      basis0.z,
                                      basis1.z,
                                      nVec.z);
            __GEIGEN__::__set_Mat_val(D1,
                                      basis2.x,
                                      basis1.x,
                                      nVec.x,
                                      basis2.y,
                                      basis1.y,
                                      nVec.y,
                                      basis2.z,
                                      basis1.z,
                                      nVec.z);
            __GEIGEN__::__set_Mat_val(D2,
                                      basis0.x,
                                      basis2.x,
                                      nVec.x,
                                      basis0.y,
                                      basis2.y,
                                      nVec.y,
                                      basis0.z,
                                      basis2.z,
                                      nVec.z);

            param[2].x = __GEIGEN__::__Determiant(D1) / __GEIGEN__::__Determiant(D);
            param[2].y = __GEIGEN__::__Determiant(D2) / __GEIGEN__::__Determiant(D);

            if(param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0)
            {
                return 5;  // PE v3v1
            }
            else
            {
                if(param[0].x <= 0.0 && param[2].x >= 1.0)
                {
                    return 0;  // PP v1
                }
                else if(param[1].x <= 0.0 && param[0].x >= 1.0)
                {
                    return 1;  // PP v2
                }
                else if(param[2].x <= 0.0 && param[1].x >= 1.0)
                {
                    return 2;  // PP v3
                }
                else
                {
                    return 6;  // PT
                }
            }
        }
    }
}

__device__ int _dType_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 u = __GEIGEN__::__minus(v1, v0);
    double3 v = __GEIGEN__::__minus(v3, v2);
    double3 w = __GEIGEN__::__minus(v0, v2);

    double a = __GEIGEN__::__squaredNorm(u);
    double b = __GEIGEN__::__v_vec_dot(u, v);
    double c = __GEIGEN__::__squaredNorm(v);
    double d = __GEIGEN__::__v_vec_dot(u, w);
    double e = __GEIGEN__::__v_vec_dot(v, w);

    double D  = a * c - b * b;  // always >= 0
    double tD = D;              // tc = tN / tD, default tD = D >= 0
    double sN, tN;
    int    defaultCase = 8;
    sN                 = (b * e - c * d);
    if(sN <= 0.0)
    {  // sc < 0 => the s=0 edge is visible
        tN          = e;
        tD          = c;
        defaultCase = 2;
    }
    else if(sN >= D)
    {  // sc > 1  => the s=1 edge is visible
        tN          = e + b;
        tD          = c;
        defaultCase = 5;
    }
    else
    {
        tN = (a * e - b * d);
        if(tN > 0.0 && tN < tD
           && (__GEIGEN__::__v_vec_dot(w, __GEIGEN__::__v_vec_cross(u, v)) == 0.0
               || __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(u, v)) < 1.0e-20 * a * c))
        {
            if(sN < D / 2)
            {
                tN          = e;
                tD          = c;
                defaultCase = 2;
            }
            else
            {
                tN          = e + b;
                tD          = c;
                defaultCase = 5;
            }
        }
    }

    if(tN <= 0.0)
    {
        if(-d <= 0.0)
        {
            return 0;
        }
        else if(-d >= a)
        {
            return 3;
        }
        else
        {
            return 6;
        }
    }
    else if(tN >= tD)
    {
        if((-d + b) <= 0.0)
        {
            return 1;
        }
        else if((-d + b) >= a)
        {
            return 4;
        }
        else
        {
            return 7;
        }
    }

    return defaultCase;
}


