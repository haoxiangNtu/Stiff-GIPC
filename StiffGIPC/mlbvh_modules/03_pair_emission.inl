__device__ inline bool _checkPTintersection(const double3*  _vertexes,
                                            const uint32_t& id0,
                                            const uint32_t& id1,
                                            const uint32_t& id2,
                                            const uint32_t& id3,
                                            const double&   dHat,
                                            uint32_t*       _cpNum,
                                            int*            _mInx,
                                            int4*           _collisionPair,
                                            int4* _ccd_collisionPair) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];

    int dtype = _dType_PT(v0, v1, v2, v3);

    double d = 100;
    switch(dtype)
    {
        case 0: {
            _d_PP(v0, v1, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 1: {
            _d_PP(v0, v2, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 2: {
            _d_PP(v0, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        case 3: {
            _d_PE(v0, v1, v2, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 4: {
            _d_PE(v0, v2, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 5: {
            _d_PE(v0, v3, v1, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, id1, -1);
                _mInx[cdp_idx]          = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        case 6: {
            _d_PT(v0, v1, v2, v3, d);
            if(d < dHat)
            {
                //printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                //printf("ccbcbcbcbbcbcbbcbcb  %d  %d  %d  %d\n", -id0 - 1, id1, id2, id3);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            break;
        }

        default:
            break;
    }
}


__device__ inline bool _checkEEintersection(const double3*  _vertexes,
                                            const double3*  _rest_vertexes,
                                            uint32_t        id0,
                                            uint32_t        id1,
                                            uint32_t        id2,
                                            uint32_t        id3,
                                            const uint32_t& obj_idx,
                                            const double&   dHat,
                                            uint32_t*       _cpNum,
                                            int*            MatIndex,
                                            int4*           _collisionPair,
                                            int4*           _ccd_collisionPair,
                                            int             edgeNum) noexcept
{
    double3 v0 = _vertexes[id0];
    double3 v1 = _vertexes[id1];
    double3 v2 = _vertexes[id2];
    double3 v3 = _vertexes[id3];
    if(g_ee_trace) {
        double ssd = _seg_seg_d(v0,v1,v2,v3);
        if(ssd < 1.5*sqrt(dHat)) printf("EE %u %u %u %u %.17e\n", id0,id1,id2,id3, ssd);
    }

    // [xenv pin/fix] canonicalize each edge's endpoint order by POSITION (env-invariant) so
    // _dType_EE is order-independent → identical envs classify identically.
    if(g_ee_canon)
    {
        // (1) internal endpoint order within each edge — TOTAL order (position, env-local id)
        if(_vless(v1, id1, v0, id0)) { double3 t=v0; v0=v1; v1=t; uint32_t s=id0; id0=id1; id1=s; }
        if(_vless(v3, id3, v2, id2)) { double3 t=v2; v2=v3; v3=t; uint32_t s=id2; id2=id3; id3=s; }
        // (2) edge-pair order (which edge is self vs obj) — by smaller endpoint, TOTAL order
        if(_vless(v2, id2, v0, id0))
        {
            double3 t0=v0,t1=v1; v0=v2; v1=v3; v2=t0; v3=t1;
            uint32_t s0=id0,s1=id1; id0=id2; id1=id3; id2=s0; id3=s1;
        }
    }

    int    dtype  = _dType_EE(v0, v1, v2, v3);
    if(g_ee_trace && (id0==(uint32_t)g_ee_tgt0||id1==(uint32_t)g_ee_tgt0||id2==(uint32_t)g_ee_tgt0||id3==(uint32_t)g_ee_tgt0
                    ||id0==(uint32_t)g_ee_tgt1||id1==(uint32_t)g_ee_tgt1||id2==(uint32_t)g_ee_tgt1||id3==(uint32_t)g_ee_tgt1))
        printf("DT ids %u %u %u %u dtype %d pos %.15e %.15e %.15e | %.15e %.15e %.15e | %.15e %.15e %.15e | %.15e %.15e %.15e\n",
               id0,id1,id2,id3,dtype, v0.x,v0.y,v0.z, v1.x,v1.y,v1.z, v2.x,v2.y,v2.z, v3.x,v3.y,v3.z);
    // [env-det] order-invariant true seg-seg squared distance for the deterministic emit gate.
    double dsg2_ = 0.0;
    if(g_ee_detgate) { double sd_ = _seg_seg_d(v0,v1,v2,v3); dsg2_ = sd_*sd_; }
    int    add_e  = -1;
    double d      = 100.0;
    bool   smooth = false;
    switch(dtype)
    {
        case 0: {
            _d_PP(v0, v2, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id2 - 1, -id1 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);

                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 1: {
            _d_PP(v0, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id3 - 1, -id1 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 2: {
            _d_PE(v0, v2, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id0 - 1, -id2 - 1, id3, -id1 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 3: {
            _d_PP(v1, v2, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id2 - 1, -id0 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 4: {
            _d_PP(v1, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id3 - 1, -id0 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 5: {
            _d_PE(v1, v2, v3, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                               _rest_vertexes[id1],
                                               _rest_vertexes[id2],
                                               _rest_vertexes[id3]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id1 - 1, -id2 - 1, id3, -id0 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 6: {
            _d_PE(v2, v0, v1, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v2, v3), __GEIGEN__::__minus(v0, v1))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id2],
                                               _rest_vertexes[id3],
                                               _rest_vertexes[id0],
                                               _rest_vertexes[id1]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id2 - 1, -id0 - 1, id1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 7: {
            _d_PE(v3, v0, v1, d);
            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {

                double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                    __GEIGEN__::__minus(v2, v3), __GEIGEN__::__minus(v0, v1))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3))*/;
                double eps_x = _compute_epx_cp(_rest_vertexes[id2],
                                               _rest_vertexes[id3],
                                               _rest_vertexes[id0],
                                               _rest_vertexes[id1]);
                add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);


                if(add_e <= -2)
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] =
                            make_int4(-id3 - 1, -id0 - 1, id1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
                else
                {
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 8: {
            _d_EE(v0, v1, v2, v3, d);

            double eeSqureNCross = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(
                __GEIGEN__::__minus(v0, v1), __GEIGEN__::__minus(v2, v3))) /* / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1))*/;
            double eps_x = _compute_epx_cp(_rest_vertexes[id0],
                                           _rest_vertexes[id1],
                                           _rest_vertexes[id2],
                                           _rest_vertexes[id3]);
            add_e        = g_ee_nomollify ? -1 : ((eeSqureNCross < eps_x) ? -obj_idx - 2 : -1);

            if((g_ee_detgate ? dsg2_ : d) < dHat)
            {
                if(add_e <= -2)
                {
                    //printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nxxxxxxxxxxx\n");
                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    MatIndex[cdp_idx]           = atomicAdd(_cpNum + 4, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if(smooth)
                    {
                        _collisionPair[cdp_idx] = make_int4(id0, id1, id2, -id3 - 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                }
                else
                {

                    int cdp_idx                 = (int)_emit_slot(_cpNum, g_dcd_cp_cap);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx]     = make_int4(id0, id1, id2, id3);
                    MatIndex[cdp_idx]           = atomicAdd(_cpNum + 4, 1);
                }
            }
            break;
        }

        default:
            break;
    }
}

