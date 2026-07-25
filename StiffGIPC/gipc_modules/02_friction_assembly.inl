__global__ void _calFrictionHessian_gd(const double3*   _vertexes,
                                       const double3*   _o_vertexes,
                                       const double3*   _normal,
                                       const uint32_t*  _last_collisionPair_gd,
                                       Eigen::Matrix3d* triplet_values,
                                       int*             row_ids,
                                       int*             col_ids,
                                       int              number,
                                       double           dt,
                                       double           eps2,
                                       double*          lastH,
                                       int              global_offset,
                                       double           coef,
                                       const double*    vert_mu_gd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    double                 eps           = sqrt(eps2);
    unsigned int           gidx          = _last_collisionPair_gd[idx];
    double                 multiplier_vI =
        (vert_mu_gd ? vert_mu_gd[gidx] : coef) * lastH[idx];  // [per-body friction]
    __GEIGEN__::Matrix3x3d H_vI;

    double3 Vdiff  = __GEIGEN__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 normal = *_normal;
    double3 VProj  = __GEIGEN__::__minus(
        Vdiff, __GEIGEN__::__s_vec_multiply(normal, __GEIGEN__::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = __GEIGEN__::__squaredNorm(VProj);

    // Build tangent basis from ground normal (works for any orientation).
    double3 t1, t2;
    {
        double3 ref = make_double3(1.0, 0.0, 0.0);
        if(fabs(normal.x) > 0.9)
            ref = make_double3(0.0, 1.0, 0.0);
        t1 = __GEIGEN__::__v_vec_cross(normal, ref);
        double t1_len = sqrt(__GEIGEN__::__squaredNorm(t1));
        t1 = __GEIGEN__::__s_vec_multiply(t1, 1.0 / t1_len);
        t2 = __GEIGEN__::__v_vec_cross(normal, t1);
    }

    double2 relDX = make_double2(__GEIGEN__::__v_vec_dot(t1, VProj),
                                 __GEIGEN__::__v_vec_dot(t2, VProj));
    double  relDXSqNorm = relDX.x * relDX.x + relDX.y * relDX.y;

    __GEIGEN__::Matrix2x2d projH;

    if(relDXSqNorm > eps2)
    {
        double relDXNorm = sqrt(relDXSqNorm);

        __GEIGEN__::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double  eigenValues[2];
        int     eigenNum = 0;
        double2 eigenVecs[2];
        __GEIGEN__::__makePD2x2(
            relDX.x * relDX.x * -multiplier_vI / relDXSqNorm / relDXNorm
                + (multiplier_vI / relDXNorm),
            relDX.x * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm,
            relDX.x * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm,
            relDX.y * relDX.y * -multiplier_vI / relDXSqNorm / relDXNorm
                + (multiplier_vI / relDXNorm),
            eigenValues,
            eigenNum,
            eigenVecs);
        for(int i = 0; i < eigenNum; i++)
        {
            if(eigenValues[i] > 0)
            {
                __GEIGEN__::Matrix2x2d eigenMatrix =
                    __GEIGEN__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix =
                    __GEIGEN__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __GEIGEN__::__Mat2x2_add(projH, eigenMatrix);
            }
        }
    }
    else
    {
        __GEIGEN__::__set_Mat2x2_val_column(projH,
            make_double2(multiplier_vI / eps, 0),
            make_double2(0, multiplier_vI / eps));
    }

    // Map 2x2 tangent-space Hessian back to 3x3: H = T * projH * T^T
    // where T = [t1 | t2] is a 3x2 matrix.
    // H_ij = sum_ab t_i^a * projH_ab * t_j^b
    double t1a[3] = {t1.x, t1.y, t1.z};
    double t2a[3] = {t2.x, t2.y, t2.z};
    double h[3][3];
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            h[i][j] = t1a[i] * projH.m[0][0] * t1a[j]
                     + t1a[i] * projH.m[0][1] * t2a[j]
                     + t2a[i] * projH.m[1][0] * t1a[j]
                     + t2a[i] * projH.m[1][1] * t2a[j];
        }
    }
    __GEIGEN__::__set_Mat_val(H_vI,
                              h[0][0], h[0][1], h[0][2],
                              h[1][0], h[1][1], h[1][2],
                              h[2][0], h[2][1], h[2][2]);

    write_triplet<3, 3>(triplet_values, row_ids, col_ids, &gidx, H_vI.m, global_offset + idx);
}

__global__ void _calFrictionHessian(const double3*          _vertexes,
                                    const double3*          _o_vertexes,
                                    const int4*             _last_collisionPair,
                                    Eigen::Matrix3d*        triplet_values,
                                    int*                    row_ids,
                                    int*                    col_ids,
                                    uint32_t*               _cpNum,
                                    int                     number,
                                    double                  dt,
                                    double2*                distCoord,
                                    __GEIGEN__::Matrix3x2d* tanBasis,
                                    double                  eps2,
                                    double*                 lastH,
                                    double                  coef,
                                    const double*           vert_mu,
                                    int                     cd_offset4,
                                    int                     cd_offset3,
                                    int                     cd_offset2,
                                    int                     f_offset4,
                                    int                     f_offset3,
                                    int                     f_offset2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4    MMCVIDI = _last_collisionPair[idx];
    const double mu = _pair_mu(MMCVIDI, vert_mu, coef);  // [per-body friction]
    double  eps     = sqrt(eps2);
    double3 relDX3D;
    int global_offset = cd_offset4 * M12_Off + cd_offset3 * M9_Off + cd_offset2 * M6_Off;
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
        double  relDXNorm   = sqrt(relDXSqNorm);
        __GEIGEN__::Matrix12x2d T;
        Friction::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __GEIGEN__::Matrix2x2d M2;
        if(relDXSqNorm > eps2)
        {
            __GEIGEN__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __GEIGEN__::__Mat2x2_minus(
                M2,
                __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                1 / (relDXSqNorm * relDXNorm)));
        }
        else
        {
            double f1_div_relDXNorm;
            Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            Friction::f2_SF(relDXSqNorm, eps, f2);
            if(f2 != f1_div_relDXNorm && relDXSqNorm)
            {

                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __GEIGEN__::Matrix2x2d projH;

        Matrix2d F_mat2;
        F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
        makePDGeneral<double, 2>(F_mat2);
        projH.m[0][0] = F_mat2(0, 0);
        projH.m[0][1] = F_mat2(0, 1);
        projH.m[1][0] = F_mat2(1, 0);
        projH.m[1][1] = F_mat2(1, 1);


        __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

        __GEIGEN__::Matrix12x12d HessianBlock =
            __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                            mu * lastH[idx]);
        int Hidx   = atomicAdd(_cpNum + 4, 1);
        int offset = global_offset + Hidx * M12_Off;
        //Hidx += cd_offset4;
        //H12x12[Hidx]  = HessianBlock;
        uint4 global_index = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        //D4Index[Hidx] = global_index;

        write_triplet<12, 12>(
            triplet_values, row_ids, col_ids, &(global_index.x), HessianBlock.m, offset);
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
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix6x2d T;
            Friction::computeT_PP(tanBasis[idx], T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix6x2d TM2 = __GEIGEN__::__M6x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix6x6d HessianBlock =
                __GEIGEN__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                                              mu * lastH[idx]);

            int Hidx   = atomicAdd(_cpNum + 2, 1);
            int offset = global_offset + f_offset4 * M12_Off
                         + f_offset3 * M9_Off + Hidx * M6_Off;
            //Hidx += cd_offset2;
            //H6x6[Hidx]    = HessianBlock;
            uint2 global_index = make_uint2(MMCVIDI.x, MMCVIDI.y);
            //D2Index[Hidx]      = global_index;


            write_triplet<6, 6>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
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
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix9x2d T;
            Friction::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix9x2d TM2 = __GEIGEN__::__M9x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix9x9d HessianBlock =
                __GEIGEN__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                                              mu * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 3, 1);
            int offset = global_offset + f_offset4 * M12_Off + Hidx * M9_Off;
            //Hidx += cd_offset3;
            //H9x9[Hidx]    = HessianBlock;
            uint3 global_index = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            //D3Index[Hidx]      = global_index;


            write_triplet<9, 9>(triplet_values,
                                row_ids,
                                col_ids,
                                &(global_index.x),
                                HessianBlock.m,
                                offset);
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
            double2 relDX       = __GEIGEN__::__M2x3_v3_multiply(tB_T, relDX3D);
            double  relDXSqNorm = __GEIGEN__::__squaredNorm(relDX);
            double  relDXNorm   = sqrt(relDXSqNorm);
            __GEIGEN__::Matrix12x2d T;
            Friction::computeT_PT(
                tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __GEIGEN__::Matrix2x2d M2;
            if(relDXSqNorm > eps2)
            {
                __GEIGEN__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __GEIGEN__::__Mat2x2_minus(
                    M2,
                    __GEIGEN__::__s_Mat2x2_multiply(__GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                                                    1 / (relDXSqNorm * relDXNorm)));
            }
            else
            {
                double f1_div_relDXNorm;
                Friction::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                Friction::f2_SF(relDXSqNorm, eps, f2);
                if(f2 != f1_div_relDXNorm && relDXSqNorm)
                {

                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __GEIGEN__::__Mat2x2_minus(
                        M2,
                        __GEIGEN__::__s_Mat2x2_multiply(
                            __GEIGEN__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else
                {
                    __GEIGEN__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __GEIGEN__::Matrix2x2d projH;
            Matrix2d               F_mat2;
            F_mat2 << M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1];
            makePDGeneral<double, 2>(F_mat2);
            projH.m[0][0] = F_mat2(0, 0);
            projH.m[0][1] = F_mat2(0, 1);
            projH.m[1][0] = F_mat2(1, 0);
            projH.m[1][1] = F_mat2(1, 1);

            __GEIGEN__::Matrix12x2d TM2 = __GEIGEN__::__M12x2_M2x2_Multiply(T, projH);

            __GEIGEN__::Matrix12x12d HessianBlock =
                __GEIGEN__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                                                mu * lastH[idx]);
            int Hidx   = atomicAdd(_cpNum + 4, 1);
            int offset = global_offset + Hidx * M12_Off;
            //Hidx += cd_offset4;
            //H12x12[Hidx]  = HessianBlock;
            uint4 global_index = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            //D4Index[Hidx] = global_index;


            write_triplet<12, 12>(triplet_values,
                                  row_ids,
                                  col_ids,
                                  &(global_index.x),
                                  HessianBlock.m,
                                  offset);
        }
    }
}

template <typename T>
__global__ inline void moveMemory_1(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

