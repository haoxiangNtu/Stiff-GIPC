__global__ void _stepForward(double3* _vertexes,
                             double3* _vertexesTemp,
                             double3* _moveDir,
                             int*     bType,
                             double   alpha,
                             bool     moveBoundary,
                             int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        // [audit lens-D fix] alpha==0 must mean "do not move" LITERALLY: with a
        // NaN/Inf moveDir (diverged env), temp - dir*0 = NaN — the freeze
        // decision failed to stop the very write it was made for. Keep the
        // last-accepted position instead. Bit-neutral for alpha != 0.
        if(alpha == 0.0)
            _vertexes[idx] = _vertexesTemp[idx];
        else
            _vertexes[idx] =
                __GEIGEN__::__minus(_vertexesTemp[idx],
                                    __GEIGEN__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

// [multi-env S2] per-env FEM step: vertex idx moves by env_alpha[p2g[idx]] instead
// of a global scalar (env g's verts step uniformly with env g's ABD bodies).
// p2g and env_alpha are indexed in the SAME local frame as _vertexes here (caller
// passes p2g already offset to the FEM region). Falls back to `alpha` if group<0.
__global__ void _stepForward_perenv(double3*       _vertexes,
                                    const double3* _vertexesTemp,
                                    const double3* _moveDir,
                                    const int*     bType,
                                    const int*     p2g,
                                    const double*  env_alpha,
                                    double         alpha,
                                    bool           moveBoundary,
                                    int            numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers) return;
    if(abs(bType[idx]) == 0 || moveBoundary)
    {
        int    g = p2g[idx];
        double a = (g >= 0 && env_alpha[g] >= 0.0) ? env_alpha[g] : alpha;
        // [audit lens-D fix] a frozen env (a==0) must KEEP its last-accepted
        // position: with a NaN/Inf moveDir, temp - dir*0 = NaN — the freeze
        // failed exactly when it mattered. Bit-neutral for a != 0.
        if(a == 0.0)
            _vertexes[idx] = _vertexesTemp[idx];
        else
            _vertexes[idx] =
                __GEIGEN__::__minus(_vertexesTemp[idx],
                                    __GEIGEN__::__s_vec_multiply(_moveDir[idx], a));
    }
}

// [multi-env S2] gather per-ABD-body alpha: body b (0..abd_body_num) belongs to
// collision body b -> group body_to_group[b] -> env_alpha[group]. -1 if ungrouped.
__global__ void _gather_abd_body_alpha(const int* body_to_group, const double* env_alpha,
                                       double* abd_body_alpha, int abd_body_num, int ng)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= abd_body_num) return;
    int g = body_to_group[b];
    abd_body_alpha[b] = (g >= 0 && g < ng) ? env_alpha[g] : -1.0;
}

__global__ void _updateVelocities(double3* _vertexes,
                                  double3* _o_vertexes,
                                  double3* _velocities,
                                  int*     btype,
                                  double   ipc_dt,
                                  int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(btype[idx] == 0)
    {
        _velocities[idx] = __GEIGEN__::__s_vec_multiply(
            __GEIGEN__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        //_velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else
    {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == -1 || (_btype[idx]) == 1)
    {
        _vertexes[idx] = __GEIGEN__::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__ void _updateBoundary2(int* _btype, __GEIGEN__::Matrix3x3d* _constraints, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    if((_btype[idx]) == 1)
    {
        _btype[idx] = 0;
        __GEIGEN__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}


__global__ void _updateBoundaryMoveDir(double3* _vertexes,
                                       int*     _btype,
                                       double3* _moveDir,
                                       double   ipc_dt,
                                       double   PI,
                                       double   alpha,
                                       int      numbers,
                                       int      frameid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double                 massSum = 0;
    double                 angleX  = PI / 2.5 * ipc_dt * alpha;
    __GEIGEN__::Matrix3x3d rotationL, rotationR;
    __GEIGEN__::__set_Mat_val(
        rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    __GEIGEN__::__set_Mat_val(
        rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    //_moveDir[idx] = make_double3(0, 0, 0);
    double mvl = -0.3 * ipc_dt * alpha;
    //if((_btype[idx]) == 1)
    //{
    //    _moveDir[idx] = make_double3(mvl, 0, 0);  //__GEIGEN__::__minus(__GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    //}
    if((_btype[idx]) > 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
        }
    }
    if((_btype[idx]) < 0)
    {
        if(frameid < 32)
        {
            if(_vertexes[idx].y > 0.01)
            {
                _moveDir[idx] = make_double3(0, -mvl, 0);
            }
            else if(_vertexes[idx].y < -0.01)
            {
                _moveDir[idx] = make_double3(0, mvl, 0);
            }
        }
        else
        {
            _moveDir[idx] = __GEIGEN__::__minus(
                __GEIGEN__::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
        }
    }
}

__global__ void _computeXTilta(int*     _btype,
                               double3* _velocities,
                               double3* _o_vertexes,
                               double3* _xTilta,
                               int*     _apply_gravity,
                               double   ipc_dt,
                               double   rate,
                               double3  gravity_vec,
                               int      numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    double3 gravityDtSq = make_double3(0, 0, 0);
    if(_btype[idx] == 0 && _apply_gravity[idx])
    {
        gravityDtSq = __GEIGEN__::__s_vec_multiply(gravity_vec, ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __GEIGEN__::__add(
        _o_vertexes[idx],
        __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply(_velocities[idx], ipc_dt),
                          gravityDtSq));
}

__global__ void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_faces[idx].x < _offset_num)
    {
        _faces[idx].x = sortIndex[_faces[idx].x];
    }
    else
    {
        _faces[idx].x = _faces[idx].x;
    }
    if(_faces[idx].y < _offset_num)
    {
        _faces[idx].y = sortIndex[_faces[idx].y];
    }
    else
    {
        _faces[idx].y = _faces[idx].y;
    }
    if(_faces[idx].z < _offset_num)
    {
        _faces[idx].z = sortIndex[_faces[idx].z];
    }
    else
    {
        _faces[idx].z = _faces[idx].z;
    }
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__ void _updateNeighborNum(unsigned int*   _neighborNumInit,
                                   unsigned int*   _neighborNum,
                                   const uint32_t* sortMapVertIndex,
                                   int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__ void _updateNeighborList(unsigned int*   _neighborListInit,
                                    unsigned int*   _neighborList,
                                    unsigned int*   _neighborNum,
                                    unsigned int*   _neighborStart,
                                    unsigned int*   _neighborStartTemp,
                                    const uint32_t* sortIndex,
                                    const uint32_t* sortMapVertIndex,
                                    int             numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int startId   = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum    = _neighborNum[idx];
    for(int i = 0; i < neiNum; i++)
    {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
    //_neighborStart[sortMapVertIndex[idx]] = startId;
    //_neighborNum[idx] = _neighborNum[sortMapVertIndex[idx]];
}

__global__ void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }
}

__global__ void _updateTriEdges_adjVerts(
    uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_edges[idx].x < _offset_num)
    {
        _edges[idx].x = sortIndex[_edges[idx].x];
    }
    else
    {
        _edges[idx].x = _edges[idx].x;
    }
    if(_edges[idx].y < _offset_num)
    {
        _edges[idx].y = sortIndex[_edges[idx].y];
    }
    else
    {
        _edges[idx].y = _edges[idx].y;
    }


    if(_adj_verts[idx].x < _offset_num)
    {
        _adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
    }
    else
    {
        _adj_verts[idx].x = _adj_verts[idx].x;
    }
    if(_adj_verts[idx].y < _offset_num)
    {
        _adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
    }
    else
    {
        _adj_verts[idx].y = _adj_verts[idx].y;
    }
}

__global__ void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;
    if(_sVerts[idx] < _offset_num)
    {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
    else
    {
        _sVerts[idx] = _sVerts[idx];
    }
}

// Check if collision between bodyA and bodyB should be skipped
// according to the collision exclusion matrix.
//
// [multi-FEM-bodyid] Previously mapped body_id == -1 (legacy FEM sentinel)
// to the last matrix slot. Now every body has its real body_id and indexes
// the matrix directly.
__device__ inline bool _is_collision_excluded_gipc(int bodyA, int bodyB,
                                                   const int* _collision_skip_matrix,
                                                   int _collision_body_count)
{
    if(_collision_skip_matrix == nullptr || _collision_body_count <= 0)
        return false;
    if(bodyA < 0 || bodyB < 0 || bodyA >= _collision_body_count || bodyB >= _collision_body_count)
        return false;
    return _collision_skip_matrix[bodyA * _collision_body_count + bodyB] != 0;
}

// [multi-FEM-bodyid] Skip same-body filter for sanity-check kernels
// (segment-triangle intersection). Mirrors mlbvh.cu's _should_check_pair
// but inverted: returns TRUE if the pair should be SKIPPED (same ABD body,
// no point checking).
__device__ inline bool _skip_same_abd_body(int bodyA, int bodyB,
                                           const int* _body_id_to_is_fem)
{
    if(bodyA != bodyB) return false;        // different bodies -> not skipped here
    if(bodyA < 0) return false;             // unassigned -> defensive (don't skip)
    if(_body_id_to_is_fem == nullptr) return true;  // legacy fallback: skip same body
    return _body_id_to_is_fem[bodyA] == 0;  // skip if same ABD body
}

__global__ void _edgeTriIntersectionQuery(const int*     _bodyId,
                                          const int*     _btype,
                                          const double3* _vertexes,
                                          const uint2*   _edges,
                                          const uint3*   _faces,
                                          const AABB*    _edge_bvs,
                                          const Node*    _edge_nodes,
                                          int*           _isIntesect,
                                          double         dHat,
                                          int            number,
                                          const int*     _collision_skip_matrix,
                                          int            _collision_body_count,
                                          const int*     _body_id_to_is_fem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++        = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = 0;  //sqrt(dHat);
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx   = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx   = _edge_nodes[node_id].right_idx;

        if(_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    // [multi-FEM-bodyid] Skip if face and edge belong to the same
                    // ABD body. FEM body self-intersection sanity-check is still
                    // run (allowed) since the FEM mesh might fold onto itself.
                    if(_skip_same_abd_body(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                           _body_id_to_is_fem))
                    {
                        // same ABD body, skip
                    }
                    // Skip if bodies are in the collision exclusion list
                    else if(_is_collision_excluded_gipc(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                                       _collision_skip_matrix, _collision_body_count))
                    {
                        // excluded body pair, skip
                    }
                    else if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            printf("[INTERSECT-L] tri(%d,%d,%d) body=%d  edge(%d,%d) body=%d\n",
                                   face.x, face.y, face.z, _bodyId[face.x],
                                   _edges[obj_idx].x, _edges[obj_idx].y, _bodyId[_edges[obj_idx].x]);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y
                     || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y
                     || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y))
                {
                    // [multi-FEM-bodyid] Skip if face and edge belong to the same
                    // ABD body. FEM body self-intersection sanity-check is still
                    // run (allowed) since the FEM mesh might fold onto itself.
                    if(_skip_same_abd_body(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                           _body_id_to_is_fem))
                    {
                        // same ABD body, skip
                    }
                    // Skip if bodies are in the collision exclusion list
                    else if(_is_collision_excluded_gipc(_bodyId[face.x], _bodyId[_edges[obj_idx].x],
                                                       _collision_skip_matrix, _collision_body_count))
                    {
                        // excluded body pair, skip
                    }
                    else if(!(_btype[face.x] >= 2 && _btype[face.y] >= 2
                         && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2
                         && _btype[_edges[obj_idx].y] >= 2))
                        if(segTriIntersect(_vertexes[_edges[obj_idx].x],
                                           _vertexes[_edges[obj_idx].y],
                                           _vertexes[face.x],
                                           _vertexes[face.y],
                                           _vertexes[face.z]))
                        {
                            *_isIntesect = -1;
                            printf("[INTERSECT-R] tri(%d,%d,%d) body=%d  edge(%d,%d) body=%d\n",
                                   face.x, face.y, face.z, _bodyId[face.x],
                                   _edges[obj_idx].x, _edges[obj_idx].y, _bodyId[_edges[obj_idx].x]);
                            return;
                        }
                }
            }
            else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while(stack < stack_ptr);
}

