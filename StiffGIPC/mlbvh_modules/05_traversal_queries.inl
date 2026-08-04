__global__ void _selfQuery_vf(const int*      _bodyID,
                              const int*      _btype,
                              const double3*  _vertexes,
                              const uint3*    _faces,
                              const uint32_t* _surfVerts,
                              const AABB*     _bvs,
                              const Node*     _nodes,
                              int4*           _collisionPair,
                              int4*           _ccd_collisionPair,
                              uint32_t*       _cpNum,
                              int*            MatIndex,
                              double          dHat,
                              int             number,
                              const int*      _collision_skip_matrix,
                              int             _collision_body_count,
                              const int*      _body_id_to_is_fem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);

    AABB _bv;
    idx       = _surfVerts[idx];

    // BVH-skip: query vertex's body has no possible collisions → exit early.
    // (audit/perf-bvh-skip-isolated: diag[B][B]==1 marks isolated body)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[idx];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhVfDcd);

    _bv.upper = _vertexes[idx];
    _bv.lower = _vertexes[idx];
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = BVH_TRAVERSAL_MARGIN(sqrt(dHat));  // audit-only expanded broad phase
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if(overlap(_bv, _bvs[L_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                       && idx != _faces[obj_idx].z)
                    {
                        if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                             && _btype[_faces[obj_idx].y] >= 2
                             && _btype[_faces[obj_idx].z] >= 2))
                        {
                            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                            _checkPTintersection(_vertexes,
                                                 idx,
                                                 _faces[obj_idx].x,
                                                 _faces[obj_idx].y,
                                                 _faces[obj_idx].z,
                                                 dHat,
                                                 _cpNum,
                                                 MatIndex,
                                                 _collisionPair,
                                                 _ccd_collisionPair);
                        }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(L_idx);
            }
        }
        if(overlap(_bv, _bvs[R_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                       && idx != _faces[obj_idx].z)
                    {
                        if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                             && _btype[_faces[obj_idx].y] >= 2
                             && _btype[_faces[obj_idx].z] >= 2))
                        {
                            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                            _checkPTintersection(_vertexes,
                                                 idx,
                                                 _faces[obj_idx].x,
                                                 _faces[obj_idx].y,
                                                 _faces[obj_idx].z,
                                                 dHat,
                                                 _cpNum,
                                                 MatIndex,
                                                 _collisionPair,
                                                 _ccd_collisionPair);
                        }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(R_idx);
            }
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

__global__ void _selfQuery_vf_wide8(const int*      _bodyID,
                                    const int*      _btype,
                                    const double3*  _vertexes,
                                    const uint3*    _faces,
                                    const uint32_t* _surfVerts,
                                    const AABB*     _bvs,
                                    const Node*     _nodes,
                                    int4*           _collisionPair,
                                    int4*           _ccd_collisionPair,
                                    uint32_t*       _cpNum,
                                    int*            MatIndex,
                                    double          dHat,
                                    int             number,
                                    const int*      _collision_skip_matrix,
                                    int             _collision_body_count,
                                    const int*      _body_id_to_is_fem,
                                    const uint32_t* wide_children)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);
    idx = _surfVerts[idx];
    if(_collision_skip_matrix && _collision_body_count > 0)
    {
        const int body = _bodyID[idx];
        if(body >= 0 && body < _collision_body_count
           && _collision_skip_matrix[body * _collision_body_count + body] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhVfDcd);
    AABB query;
    query.upper = _vertexes[idx];
    query.lower = _vertexes[idx];
    const double gap = BVH_TRAVERSAL_MARGIN(sqrt(dHat));
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit)
        {
            const int depth = static_cast<int>(stack_ptr - stack);
            atomicMax(&g_max_stack, depth);
        }
        const uint32_t* children = wide_children + 8 * node_id;
#pragma unroll
        for(int slot = 0; slot < 8; ++slot)
        {
            const uint32_t child = children[slot];
            if(child == 0xFFFFFFFFu)
                break;
            if(!overlap(query, _bvs[child], gap))
                continue;
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const uint32_t obj_idx = _nodes[child].element_idx;
            if(obj_idx == 0xFFFFFFFFu)
            {
                BVH_STACK_PUSH(child);
                continue;
            }
            const uint3 face = _faces[obj_idx];
            if(!_should_check_pair(
                   _bodyID[idx], _bodyID[face.x], _body_id_to_is_fem)
               || _is_collision_excluded(_bodyID[idx],
                                          _bodyID[face.x],
                                          _collision_skip_matrix,
                                          _collision_body_count)
               || _cross_env_skip(idx, face.x) || !_same_env(idx, face.x)
               || idx == face.x || idx == face.y || idx == face.z
               || (_btype[idx] >= 2 && _btype[face.x] >= 2
                   && _btype[face.y] >= 2 && _btype[face.z] >= 2))
                continue;
            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
            _checkPTintersection(_vertexes,
                                 idx,
                                 face.x,
                                 face.y,
                                 face.z,
                                 dHat,
                                 _cpNum,
                                 MatIndex,
                                 _collisionPair,
                                 _ccd_collisionPair);
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

__global__ void _selfQuery_vf_ccd(const int*      _bodyID,
                                  const int*      _btype,
                                  const double3*  _vertexes,
                                  const double3*  moveDir,
                                  double          alpha,
                                  const uint3*    _faces,
                                  const uint32_t* _surfVerts,
                                  const AABB*     _bvs,
                                  const Node*     _nodes,
                                  int4*           _ccd_collisionPair,
                                  uint32_t*       _cpNum,
                                  double          dHat,
                                  int             number,
                                  const int*      _collision_skip_matrix,
                                  int             _collision_body_count,
                                  const int*      _body_id_to_is_fem,
                                  const double*   alpha_dev = nullptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev) alpha = *alpha_dev;   // [de-CPU] per-env CCD search alpha read on device

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);

    AABB _bv;
    idx                    = _surfVerts[idx];

    // BVH-skip (audit/perf-bvh-skip-isolated)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[idx];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhVfCcd);

    double3 current_vertex = _vertexes[idx];
    double3 mvD            = moveDir[idx];
    _bv.upper              = current_vertex;
    _bv.lower              = current_vertex;
    _bv.combines(current_vertex.x - mvD.x * alpha,
                 current_vertex.y - mvD.y * alpha,
                 current_vertex.z - mvD.z * alpha);
    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = BVH_TRAVERSAL_MARGIN(sqrt(dHat));  // audit-only expanded broad phase
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        if(overlap(_bv, _bvs[L_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {

                    if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                         && _btype[_faces[obj_idx].y] >= 2
                         && _btype[_faces[obj_idx].z] >= 2))
                        if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                           && idx != _faces[obj_idx].z)
                        {
                            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                make_int4(-idx - 1,
                                          _faces[obj_idx].x,
                                          _faces[obj_idx].y,
                                          _faces[obj_idx].z);
                        }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(L_idx);
            }
        }
        if(overlap(_bv, _bvs[R_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(_should_check_pair(_bodyID[idx], _bodyID[_faces[obj_idx].x], _body_id_to_is_fem)
                   && !_is_collision_excluded(_bodyID[idx], _bodyID[_faces[obj_idx].x],
                                             _collision_skip_matrix, _collision_body_count)
                   && !_cross_env_skip(idx, _faces[obj_idx].x)
                   && _same_env(idx, _faces[obj_idx].x))
                {
                    if(!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2
                         && _btype[_faces[obj_idx].y] >= 2
                         && _btype[_faces[obj_idx].z] >= 2))
                        if(idx != _faces[obj_idx].x && idx != _faces[obj_idx].y
                           && idx != _faces[obj_idx].z)
                        {
                            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                make_int4(-idx - 1,
                                          _faces[obj_idx].x,
                                          _faces[obj_idx].y,
                                          _faces[obj_idx].z);
                        }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(R_idx);
            }
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

__global__ void _selfQuery_vf_ccd_wide8(
    const int*      _bodyID,
    const int*      _btype,
    const double3*  _vertexes,
    const double3*  moveDir,
    double          alpha,
    const uint3*    _faces,
    const uint32_t* _surfVerts,
    const AABB*     _bvs,
    const Node*     _nodes,
    int4*           _ccd_collisionPair,
    uint32_t*       _cpNum,
    double          dHat,
    int             number,
    const int*      _collision_skip_matrix,
    int             _collision_body_count,
    const int*      _body_id_to_is_fem,
    const double*   alpha_dev,
    const uint32_t* wide_children)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev)
        alpha = *alpha_dev;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);
    idx = _surfVerts[idx];
    if(_collision_skip_matrix && _collision_body_count > 0)
    {
        const int body = _bodyID[idx];
        if(body >= 0 && body < _collision_body_count
           && _collision_skip_matrix[body * _collision_body_count + body] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhVfCcd);
    const double3 current = _vertexes[idx];
    const double3 move    = moveDir[idx];
    AABB query;
    query.upper = current;
    query.lower = current;
    query.combines(current.x - move.x * alpha,
                   current.y - move.y * alpha,
                   current.z - move.z * alpha);
    const double gap = BVH_TRAVERSAL_MARGIN(sqrt(dHat));
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit)
        {
            const int depth = static_cast<int>(stack_ptr - stack);
            atomicMax(&g_max_stack, depth);
        }
        const uint32_t* children = wide_children + 8 * node_id;
#pragma unroll
        for(int slot = 0; slot < 8; ++slot)
        {
            const uint32_t child = children[slot];
            if(child == 0xFFFFFFFFu)
                break;
            if(!overlap(query, _bvs[child], gap))
                continue;
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const uint32_t obj_idx = _nodes[child].element_idx;
            if(obj_idx == 0xFFFFFFFFu)
            {
                BVH_STACK_PUSH(child);
                continue;
            }
            const uint3 face = _faces[obj_idx];
            if(!_should_check_pair(
                   _bodyID[idx], _bodyID[face.x], _body_id_to_is_fem)
               || _is_collision_excluded(_bodyID[idx],
                                          _bodyID[face.x],
                                          _collision_skip_matrix,
                                          _collision_body_count)
               || _cross_env_skip(idx, face.x) || !_same_env(idx, face.x)
               || idx == face.x || idx == face.y || idx == face.z
               || (_btype[idx] >= 2 && _btype[face.x] >= 2
                   && _btype[face.y] >= 2 && _btype[face.z] >= 2))
                continue;
            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                make_int4(-idx - 1, face.x, face.y, face.z);
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}


// [ee-lb] traversal body shared by the __launch_bounds__ occupancy variants below (same code,
// different register budgets — selected at launch via STIFF_EE_LB).
template <int RangePruneMode>
static __device__ __forceinline__ void _selfQuery_ee_body(const int*     _bodyID,
                              const int*     _btype,
                              const double3* _vertexes,
                              const double3* _rest_vertexes,
                              const uint2*   _edges,
                              const AABB*    _bvs,
                              const Node*    _nodes,
                              int4*          _collisionPair,
                              int4*          _ccd_collisionPair,
                              uint32_t*      _cpNum,
                              int*           MatIndex,
                              double         dHat,
                              int            number,
                              const int*     _collision_skip_matrix,
                              int            _collision_body_count,
                              const int*     _body_id_to_is_fem,
                              const int* node_env,
                              const uint32_t* node_max_element)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);

    const uint32_t self_leaf = (uint32_t)idx;
    idx               = idx + number - 1;
    AABB     _bv      = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;  // [env-part B] query edge env

    // BVH-skip (audit/perf-bvh-skip-isolated): if both edge endpoints' body
    // is isolated, no collision is possible — exit early.
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[_edges[self_eid].x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhEeDcd);

    //double bboxDiagSize2 = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(_bvs[0].upper, _bvs[0].lower));
    //printf("%f\n", bboxDiagSize2);
    double gapl = BVH_TRAVERSAL_MARGIN(sqrt(dHat));  // audit-only expanded broad phase
    //double dHat = gapl * gapl;// *bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        bool own_l = true;
        if constexpr(RangePruneMode == 1)
            if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                own_l = node_max_element[L_idx] >= self_eid;
        if constexpr(RangePruneMode == 2)
            if(node_max_element)
                own_l = node_max_element[L_idx] >= self_leaf;
        if(own_l && (qenv < 0 || node_env[L_idx] < 0 || node_env[L_idx] == qenv) && overlap(_bv, _bvs[L_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {


                        bool duplicate =
                            _edges[self_eid].x == _edges[obj_idx].x
                            || _edges[self_eid].x == _edges[obj_idx].y
                            || _edges[self_eid].y == _edges[obj_idx].x
                            || _edges[self_eid].y == _edges[obj_idx].y;
                        if constexpr(RangePruneMode != 2)
                            duplicate = duplicate
                                || (!g_ee_nodedup
                                    && (g_ee_canon
                                            ? (_edge_lkey(_edges[obj_idx])
                                               < _edge_lkey(_edges[self_eid]))
                                            : (obj_idx < self_eid)));
                        if(!duplicate)
                        {
                            //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                            if(!(_btype[_edges[self_eid].x] >= 2
                                 && _btype[_edges[self_eid].y] >= 2
                                 && _btype[_edges[obj_idx].x] >= 2
                                 && _btype[_edges[obj_idx].y] >= 2))
                            {
                                BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                                _checkEEintersection<(RangePruneMode == 2)>(_vertexes,
                                                     _rest_vertexes,
                                                     _edges[self_eid].x,
                                                     _edges[self_eid].y,
                                                     _edges[obj_idx].x,
                                                     _edges[obj_idx].y,
                                                     obj_idx,
                                                     dHat,
                                                     _cpNum,
                                                     MatIndex,
                                                     _collisionPair,
                                                     _ccd_collisionPair,
                                                     number);
                            }
                        }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(L_idx);
            }
        }
        bool own_r = true;
        if constexpr(RangePruneMode == 1)
            if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                own_r = node_max_element[R_idx] >= self_eid;
        if constexpr(RangePruneMode == 2)
            if(node_max_element)
                own_r = node_max_element[R_idx] >= self_leaf;
        if(own_r && (qenv < 0 || node_env[R_idx] < 0 || node_env[R_idx] == qenv) && overlap(_bv, _bvs[R_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        bool duplicate =
                            _edges[self_eid].x == _edges[obj_idx].x
                            || _edges[self_eid].x == _edges[obj_idx].y
                            || _edges[self_eid].y == _edges[obj_idx].x
                            || _edges[self_eid].y == _edges[obj_idx].y;
                        if constexpr(RangePruneMode != 2)
                            duplicate = duplicate
                                || (!g_ee_nodedup
                                    && (g_ee_canon
                                            ? (_edge_lkey(_edges[obj_idx])
                                               < _edge_lkey(_edges[self_eid]))
                                            : (obj_idx < self_eid)));
                        if(!duplicate)
                        {
                            //printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
                            if(!(_btype[_edges[self_eid].x] >= 2
                                 && _btype[_edges[self_eid].y] >= 2
                                 && _btype[_edges[obj_idx].x] >= 2
                                 && _btype[_edges[obj_idx].y] >= 2))
                            {
                                BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                                _checkEEintersection<(RangePruneMode == 2)>(_vertexes,
                                                     _rest_vertexes,
                                                     _edges[self_eid].x,
                                                     _edges[self_eid].y,
                                                     _edges[obj_idx].x,
                                                     _edges[obj_idx].y,
                                                     obj_idx,
                                                     dHat,
                                                     _cpNum,
                                                     MatIndex,
                                                     _collisionPair,
                                                     _ccd_collisionPair,
                                                     number);
                            }
                        }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(R_idx);
            }
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

// [ee-lb] launch shells. Baseline compiles to ~168 reg → 1 block/SM (8 warps, 16.7% theoretical
// occupancy; 11.8% achieved) while DRAM sits at ~0.3% — latency-bound with nothing in flight to
// hide it. The capped variants trade registers (spills land in an idle L1/L2) for resident warps:
//   lb2: __launch_bounds__(256,2) → ≤128 reg → 16 warps/SM;  lb3: (256,3) → ≤85 reg → 24 warps/SM.
#define _SQEE_PARAMS                                                                               \
    const int *_bodyID, const int *_btype, const double3 *_vertexes,                               \
        const double3 *_rest_vertexes, const uint2 *_edges, const AABB *_bvs,                      \
        const Node *_nodes, int4 *_collisionPair, int4 *_ccd_collisionPair, uint32_t *_cpNum,      \
        int *MatIndex, double dHat, int number, const int *_collision_skip_matrix,                 \
        int _collision_body_count, const int *_body_id_to_is_fem, const int *node_env
#define _SQEE_ARGS                                                                                 \
    _bodyID, _btype, _vertexes, _rest_vertexes, _edges, _bvs, _nodes, _collisionPair,              \
        _ccd_collisionPair, _cpNum, MatIndex, dHat, number, _collision_skip_matrix,                \
        _collision_body_count, _body_id_to_is_fem, node_env
__global__ void _selfQuery_ee(_SQEE_PARAMS)
{
    _selfQuery_ee_body<0>(_SQEE_ARGS, nullptr);
}
__global__ void __launch_bounds__(256, 2) _selfQuery_ee_lb2(_SQEE_PARAMS)
{
    _selfQuery_ee_body<0>(_SQEE_ARGS, nullptr);
}
__global__ void __launch_bounds__(256, 3) _selfQuery_ee_lb3(_SQEE_PARAMS)
{
    _selfQuery_ee_body<0>(_SQEE_ARGS, nullptr);
}
__global__ void _selfQuery_ee_range_prune(_SQEE_PARAMS,
                                          const uint32_t* node_max_element)
{
    _selfQuery_ee_body<1>(_SQEE_ARGS, node_max_element);
}
__global__ void _selfQuery_ee_sorted_prune(_SQEE_PARAMS,
                                           const uint32_t* node_max_element)
{
    _selfQuery_ee_body<2>(_SQEE_ARGS, node_max_element);
}

template <int RangePruneMode>
static __device__ __forceinline__ void _selfQuery_ee_wide8_body(
    _SQEE_PARAMS,
    const uint32_t* wide_children,
    const uint32_t* node_max_element)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);
    const uint32_t self_leaf = static_cast<uint32_t>(idx);
    idx += number - 1;
    const AABB query = _bvs[idx];
    const uint32_t self_eid = _nodes[idx].element_idx;
    const uint2 self_edge = _edges[self_eid];
    const int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;
    if(_collision_skip_matrix && _collision_body_count > 0)
    {
        const int body = _bodyID[self_edge.x];
        if(body >= 0 && body < _collision_body_count
           && _collision_skip_matrix[body * _collision_body_count + body] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhEeDcd);
    const double gap = BVH_TRAVERSAL_MARGIN(sqrt(dHat));
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit)
        {
            const int depth = static_cast<int>(stack_ptr - stack);
            atomicMax(&g_max_stack, depth);
        }
        const uint32_t* children = wide_children + 8 * node_id;
#pragma unroll
        for(int slot = 0; slot < 8; ++slot)
        {
            const uint32_t child = children[slot];
            if(child == 0xFFFFFFFFu)
                break;
            bool owned = true;
            if constexpr(RangePruneMode == 1)
            {
                if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                    owned = node_max_element[child] >= self_eid;
            }
            if constexpr(RangePruneMode == 2)
            {
                if(node_max_element)
                    owned = node_max_element[child] >= self_leaf;
            }
            if(!owned
               || !(qenv < 0 || node_env[child] < 0
                    || node_env[child] == qenv)
               || !overlap(query, _bvs[child], gap))
                continue;
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const uint32_t obj_idx = _nodes[child].element_idx;
            if(obj_idx == 0xFFFFFFFFu)
            {
                BVH_STACK_PUSH(child);
                continue;
            }
            if(self_eid == obj_idx)
                continue;
            const uint2 other_edge = _edges[obj_idx];
            if(!_should_check_pair(_bodyID[self_edge.x],
                                   _bodyID[other_edge.x],
                                   _body_id_to_is_fem)
               || _is_collision_excluded(_bodyID[self_edge.x],
                                          _bodyID[other_edge.x],
                                          _collision_skip_matrix,
                                          _collision_body_count)
               || _cross_env_skip(self_edge.x, other_edge.x)
               || !_same_env(self_edge.x, other_edge.x))
                continue;
            bool duplicate = self_edge.x == other_edge.x
                             || self_edge.x == other_edge.y
                             || self_edge.y == other_edge.x
                             || self_edge.y == other_edge.y;
            if constexpr(RangePruneMode != 2)
            {
                duplicate = duplicate
                            || (!g_ee_nodedup
                                && (g_ee_canon
                                        ? (_edge_lkey(other_edge)
                                           < _edge_lkey(self_edge))
                                        : (obj_idx < self_eid)));
            }
            if(duplicate
               || (_btype[self_edge.x] >= 2 && _btype[self_edge.y] >= 2
                   && _btype[other_edge.x] >= 2
                   && _btype[other_edge.y] >= 2))
                continue;
            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
            _checkEEintersection<(RangePruneMode == 2)>(_vertexes,
                                                         _rest_vertexes,
                                                         self_edge.x,
                                                         self_edge.y,
                                                         other_edge.x,
                                                         other_edge.y,
                                                         obj_idx,
                                                         dHat,
                                                         _cpNum,
                                                         MatIndex,
                                                         _collisionPair,
                                                         _ccd_collisionPair,
                                                         number);
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

__global__ void __launch_bounds__(256, 2) _selfQuery_ee_wide8(
    _SQEE_PARAMS, const uint32_t* wide_children)
{
    _selfQuery_ee_wide8_body<0>(_SQEE_ARGS, wide_children, nullptr);
}

__global__ void __launch_bounds__(256, 2) _selfQuery_ee_range_prune_wide8(
    _SQEE_PARAMS,
    const uint32_t* wide_children,
    const uint32_t* node_max_element)
{
    _selfQuery_ee_wide8_body<1>(
        _SQEE_ARGS, wide_children, node_max_element);
}

__global__ void __launch_bounds__(256, 2) _selfQuery_ee_sorted_prune_wide8(
    _SQEE_PARAMS,
    const uint32_t* wide_children,
    const uint32_t* node_max_element)
{
    _selfQuery_ee_wide8_body<2>(
        _SQEE_ARGS, wide_children, node_max_element);
}

template <int RangePruneMode>
static __device__ __forceinline__ void _selfQuery_ee_ccd_body(
    const int*      _bodyID,
    const int*      _btype,
    const double3*  _vertexes,
    const double3*  moveDir,
    double          alpha,
    const uint2*    _edges,
    const AABB*     _bvs,
    const Node*     _nodes,
    int4*           _ccd_collisionPair,
    uint32_t*       _cpNum,
    double          dHat,
    int             number,
    const int*      _collision_skip_matrix,
    int             _collision_body_count,
    const int*      _body_id_to_is_fem,
    const int*      node_env,
    const double*   alpha_dev,
    const uint32_t* node_max_element)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev) alpha = *alpha_dev;   // [de-CPU] per-env CCD search alpha read on device

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr   = stack;
    BVH_STACK_PUSH(0);
    idx                   = idx + number - 1;
    AABB     _bv          = _bvs[idx];
    uint32_t self_eid     = _nodes[idx].element_idx;
    int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;  // [env-part B] query edge env
    uint2    current_edge = _edges[self_eid];

    // BVH-skip (audit/perf-bvh-skip-isolated)
    if(_collision_skip_matrix && _collision_body_count > 0) {
        int B = _bodyID[current_edge.x];
        if(B >= 0 && B < _collision_body_count
           && _collision_skip_matrix[B * _collision_body_count + B] != 0)
            return;
    }
    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhEeCcd);
    //double3 edge_tvert0 = __GEIGEN__::__minus(_vertexes[current_edge.x], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.x], alpha));
    //double3 edge_tvert1 = __GEIGEN__::__minus(_vertexes[current_edge.y], __GEIGEN__::__s_vec_multiply(moveDir[current_edge.y], alpha));
    //_bv.combines(edge_tvert0.x, edge_tvert0.y, edge_tvert0.z);
    //_bv.combines(edge_tvert1.x, edge_tvert1.y, edge_tvert1.z);
    double gapl = BVH_TRAVERSAL_MARGIN(sqrt(dHat));

    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit) { int _d=(int)(stack_ptr-stack); atomicMax(&g_max_stack,_d); }
        const uint32_t L_idx   = _nodes[node_id].left_idx;
        const uint32_t R_idx   = _nodes[node_id].right_idx;

        bool own_l = true;
        if constexpr(RangePruneMode == 1)
            if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                own_l = node_max_element[L_idx] >= self_eid;
        if(own_l && (qenv < 0 || node_env[L_idx] < 0 || node_env[L_idx] == qenv) && overlap(_bv, _bvs[L_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[L_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        if(!(_btype[_edges[self_eid].x] >= 2
                             && _btype[_edges[self_eid].y] >= 2
                             && _btype[_edges[obj_idx].x] >= 2
                             && _btype[_edges[obj_idx].y] >= 2))
                            if(!(current_edge.x == _edges[obj_idx].x
                                 || current_edge.x == _edges[obj_idx].y
                                 || current_edge.y == _edges[obj_idx].x
                                 || current_edge.y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(current_edge)) : (obj_idx < self_eid)))))
                            {
                                BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                                _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                    make_int4(current_edge.x,
                                              current_edge.y,
                                              _edges[obj_idx].x,
                                              _edges[obj_idx].y);
                            }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(L_idx);
            }
        }
        bool own_r = true;
        if constexpr(RangePruneMode == 1)
            if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                own_r = node_max_element[R_idx] >= self_eid;
        if(own_r && (qenv < 0 || node_env[R_idx] < 0 || node_env[R_idx] == qenv) && overlap(_bv, _bvs[R_idx], gapl))
        {
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const auto obj_idx = _nodes[R_idx].element_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(self_eid != obj_idx)
                {
                    if(_should_check_pair(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x], _body_id_to_is_fem)
                       && !_is_collision_excluded(_bodyID[_edges[self_eid].x], _bodyID[_edges[obj_idx].x],
                                                 _collision_skip_matrix, _collision_body_count)
                       && !_cross_env_skip(_edges[self_eid].x, _edges[obj_idx].x)
                       && _same_env(_edges[self_eid].x, _edges[obj_idx].x))
                    {
                        if(!(_btype[_edges[self_eid].x] >= 2
                             && _btype[_edges[self_eid].y] >= 2
                             && _btype[_edges[obj_idx].x] >= 2
                             && _btype[_edges[obj_idx].y] >= 2))
                            if(!(current_edge.x == _edges[obj_idx].x
                                 || current_edge.x == _edges[obj_idx].y
                                 || current_edge.y == _edges[obj_idx].x
                                 || current_edge.y == _edges[obj_idx].y || (!g_ee_nodedup && (g_ee_canon ? (_edge_lkey(_edges[obj_idx]) < _edge_lkey(current_edge)) : (obj_idx < self_eid)))))
                            {
                                BVH_TRAVERSAL_AUDIT_PRIMITIVE();
                                _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                                    make_int4(current_edge.x,
                                              current_edge.y,
                                              _edges[obj_idx].x,
                                              _edges[obj_idx].y);
                            }
                    }
                }
            }
            else  // the node is not a leaf.
            {
                BVH_STACK_PUSH(R_idx);
            }
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

#define _SQEE_CCD_PARAMS                                                            \
    const int* _bodyID, const int* _btype, const double3* _vertexes,                \
        const double3* moveDir, double alpha, const uint2* _edges, const AABB* _bvs, \
        const Node* _nodes, int4* _ccd_collisionPair, uint32_t* _cpNum,              \
        double dHat, int number, const int* _collision_skip_matrix,                  \
        int _collision_body_count, const int* _body_id_to_is_fem,                    \
        const int* node_env, const double* alpha_dev
#define _SQEE_CCD_ARGS                                                              \
    _bodyID, _btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes,               \
        _ccd_collisionPair, _cpNum, dHat, number, _collision_skip_matrix,            \
        _collision_body_count, _body_id_to_is_fem, node_env, alpha_dev

__global__ void _selfQuery_ee_ccd(_SQEE_CCD_PARAMS)
{
    _selfQuery_ee_ccd_body<0>(_SQEE_CCD_ARGS, nullptr);
}

__global__ void _selfQuery_ee_ccd_range_prune(
    _SQEE_CCD_PARAMS, const uint32_t* node_max_element)
{
    _selfQuery_ee_ccd_body<1>(_SQEE_CCD_ARGS, node_max_element);
}

template <int RangePruneMode>
static __device__ __forceinline__ void _selfQuery_ee_ccd_wide8_body(
    _SQEE_CCD_PARAMS,
    const uint32_t* wide_children,
    const uint32_t* node_max_element)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    if(alpha_dev)
        alpha = *alpha_dev;

    uint32_t  stack[STIFF_BVH_STACK_CAP];
    uint32_t* stack_ptr = stack;
    BVH_STACK_PUSH(0);
    idx += number - 1;
    const AABB query = _bvs[idx];
    const uint32_t self_eid = _nodes[idx].element_idx;
    const uint2 self_edge = _edges[self_eid];
    const int qenv = (g_bvh_envpart && node_env) ? node_env[idx] : -1;
    if(_collision_skip_matrix && _collision_body_count > 0)
    {
        const int body = _bodyID[self_edge.x];
        if(body >= 0 && body < _collision_body_count
           && _collision_skip_matrix[body * _collision_body_count + body] != 0)
            return;
    }

    BVH_TRAVERSAL_AUDIT_BEGIN(kBvhEeCcd);
    const double gap = BVH_TRAVERSAL_MARGIN(sqrt(dHat));
    do
    {
        const uint32_t node_id = *--stack_ptr;
        BVH_TRAVERSAL_AUDIT_POP();
        if(g_bvh_audit)
        {
            const int depth = static_cast<int>(stack_ptr - stack);
            atomicMax(&g_max_stack, depth);
        }
        const uint32_t* children = wide_children + 8 * node_id;
#pragma unroll
        for(int slot = 0; slot < 8; ++slot)
        {
            const uint32_t child = children[slot];
            if(child == 0xFFFFFFFFu)
                break;
            bool owned = true;
            if constexpr(RangePruneMode == 1)
            {
                if(!g_ee_nodedup && !g_ee_canon && node_max_element)
                    owned = node_max_element[child] >= self_eid;
            }
            if(!owned
               || !(qenv < 0 || node_env[child] < 0
                    || node_env[child] == qenv)
               || !overlap(query, _bvs[child], gap))
                continue;
            BVH_TRAVERSAL_AUDIT_OVERLAP();
            const uint32_t obj_idx = _nodes[child].element_idx;
            if(obj_idx == 0xFFFFFFFFu)
            {
                BVH_STACK_PUSH(child);
                continue;
            }
            if(self_eid == obj_idx)
                continue;
            const uint2 other_edge = _edges[obj_idx];
            if(!_should_check_pair(_bodyID[self_edge.x],
                                   _bodyID[other_edge.x],
                                   _body_id_to_is_fem)
               || _is_collision_excluded(_bodyID[self_edge.x],
                                          _bodyID[other_edge.x],
                                          _collision_skip_matrix,
                                          _collision_body_count)
               || _cross_env_skip(self_edge.x, other_edge.x)
               || !_same_env(self_edge.x, other_edge.x)
               || self_edge.x == other_edge.x
               || self_edge.x == other_edge.y
               || self_edge.y == other_edge.x
               || self_edge.y == other_edge.y
               || (!g_ee_nodedup
                   && (g_ee_canon
                           ? (_edge_lkey(other_edge) < _edge_lkey(self_edge))
                           : (obj_idx < self_eid)))
               || (_btype[self_edge.x] >= 2 && _btype[self_edge.y] >= 2
                   && _btype[other_edge.x] >= 2
                   && _btype[other_edge.y] >= 2))
                continue;
            BVH_TRAVERSAL_AUDIT_PRIMITIVE();
            _ccd_collisionPair[_emit_slot(_cpNum, g_ccd_cp_cap)] =
                make_int4(self_edge.x,
                          self_edge.y,
                          other_edge.x,
                          other_edge.y);
        }
    } while(stack < stack_ptr);
    BVH_TRAVERSAL_AUDIT_COMMIT();
}

__global__ void _selfQuery_ee_ccd_wide8(
    _SQEE_CCD_PARAMS, const uint32_t* wide_children)
{
    _selfQuery_ee_ccd_wide8_body<0>(
        _SQEE_CCD_ARGS, wide_children, nullptr);
}

__global__ void _selfQuery_ee_ccd_range_prune_wide8(
    _SQEE_CCD_PARAMS,
    const uint32_t* wide_children,
    const uint32_t* node_max_element)
{
    _selfQuery_ee_ccd_wide8_body<1>(
        _SQEE_CCD_ARGS, wide_children, node_max_element);
}

#undef _SQEE_CCD_ARGS
#undef _SQEE_CCD_PARAMS

///////////////////////////////////////host//////////////////////////////////////////////
