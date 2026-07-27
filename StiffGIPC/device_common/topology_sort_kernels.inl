// Topology remapping and Morton-sort kernels.

uint64_t GIPC::getHashCode(double3 p, uint32_t i)
{
    uint64_t code = hash_code(-1, p.x, p.y, p.z);
    return (code << 32) | i;
}

__global__ void _calcTetMChash(uint64_t*       _MChash,
                               const double3*  _vertexes,
                               uint4*          tets,
                               const AABB*     _MaxBv,
                               const uint32_t* sortMapVertIndex,
                               int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;

    tets[idx].x = sortMapVertIndex[tets[idx].x];
    tets[idx].y = sortMapVertIndex[tets[idx].y];
    tets[idx].z = sortMapVertIndex[tets[idx].z];
    tets[idx].w = sortMapVertIndex[tets[idx].w];

    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP   = __GEIGEN__::__s_vec_multiply(
        __GEIGEN__::__add(
            __GEIGEN__::__add(_vertexes[tets[idx].x], _vertexes[tets[idx].y]),
            __GEIGEN__::__add(_vertexes[tets[idx].z], _vertexes[tets[idx].w])),
        0.25);
    double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);

    int type = 0;
    if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
    {
        type = 0;
    }
    else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
    {
        type = 1;
    }
    else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
    {
        type = 2;
    }
    else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
    {
        type = 3;
    }
    else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
    {
        type = 4;
    }
    else
    {
        type = 5;
    }

    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    _MChash[idx]  = mc64;
}

__global__ void _updateTopology(uint4*          tets,
                                uint3*          tris,
                                const uint32_t* sortMapVertIndex,
                                int             traNumber,
                                int             triNumber)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < traNumber)
    {
        tets[idx].x = sortMapVertIndex[tets[idx].x];
        tets[idx].y = sortMapVertIndex[tets[idx].y];
        tets[idx].z = sortMapVertIndex[tets[idx].z];
        tets[idx].w = sortMapVertIndex[tets[idx].w];
    }
    if(idx < triNumber)
    {
        tris[idx].x = sortMapVertIndex[tris[idx].x];
        tris[idx].y = sortMapVertIndex[tris[idx].y];
        tris[idx].z = sortMapVertIndex[tris[idx].z];
    }
}

__global__ void _updateVertexes(double3*                      o_vertexes,
                                const double3*                _vertexes,
                                double*                       tempM,
                                const double*                 mass,
                                __GEIGEN__::Matrix3x3d*       tempCons,
                                int*                          tempBtype,
                                const __GEIGEN__::Matrix3x3d* cons,
                                const int*                    bType,
                                const uint32_t*               sortIndex,
                                uint32_t*                     sortMapIndex,
                                int                           number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    o_vertexes[idx]              = _vertexes[sortIndex[idx]];
    tempM[idx]                   = mass[sortIndex[idx]];
    tempCons[idx]                = cons[sortIndex[idx]];
    sortMapIndex[sortIndex[idx]] = idx;
    tempBtype[idx]               = bType[sortIndex[idx]];
}

__global__ void _updateTetrahedras(uint4*                        o_tetrahedras,
                                   uint4*                        tetrahedras,
                                   double*                       tempV,
                                   const double*                 volum,
                                   __GEIGEN__::Matrix3x3d*       tempDmInverse,
                                   const __GEIGEN__::Matrix3x3d* dmInverse,
                                   const uint32_t*               sortTetIndex,
                                   const uint32_t*               sortMapVertIndex,
                                   int                           number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    o_tetrahedras[idx] = tetrahedras[sortTetIndex[idx]];
    tempV[idx]         = volum[sortTetIndex[idx]];
    tempDmInverse[idx] = dmInverse[sortTetIndex[idx]];
}

__global__ void _calcVertMChash(uint64_t*       _MChash,
                                const double3*  _vertexes,
                                const AABB*     _MaxBv,
                                int             number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x,
                                     (*_MaxBv).upper.y - (*_MaxBv).lower.y,
                                     (*_MaxBv).upper.z - (*_MaxBv).lower.z);
    double3 centerP = _vertexes[idx];
    double3 offset  = make_double3(centerP.x - (*_MaxBv).lower.x,
                                  centerP.y - (*_MaxBv).lower.y,
                                  centerP.z - (*_MaxBv).lower.z);
    int type = -1;
    if(type >= 0)
    {
        if(SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z)
        {
            type = 0;
        }
        else if(SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y)
        {
            type = 1;
        }
        else if(SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x)
        {
            type = 2;
        }
        else if(SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z)
        {
            type = 3;
        }
        else if(SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y)
        {
            type = 4;
        }
        else
        {
            type = 5;
        }
    }

    uint64_t mc32 = hash_code(type,
                              offset.x / SceneSize.x,
                              offset.y / SceneSize.y,
                              offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << 32) | idx);
    _MChash[idx]  = mc64;
}
