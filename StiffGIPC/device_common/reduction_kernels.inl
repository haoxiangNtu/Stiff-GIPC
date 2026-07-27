// Generic scalar/vector reduction kernels used by GIPC.

__global__ void _reduct_max_double3_to_double(const double3* _double3Dim,
                                              double*        _double1Dim,
                                              int            number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    // Inactive lanes must still participate in both barriers and full-warp
    // shuffles.
    double temp = 0.0;
    if(idx < number)
    {
        double3 tempMove = _double3Dim[idx];
        temp = std::max(std::max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));
    }
    gipc_block_max_full_to(temp, tep, number, idof, 0.0, _double1Dim + blockIdx.x);
}

__global__ void _reduct_min_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    double temp = (idx < number) ? _double1Dim[idx] : DBL_MAX;

    __threadfence();
    gipc_block_min_full_to(temp, tep, number, idof, DBL_MAX, _double1Dim + blockIdx.x);
}

__global__ void _reduct_M_double2(double2* _double2Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    double2 temp = idx < number ? _double2Dim[idx] : make_double2(0.0, 0.0);
    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
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
        temp = (warpTid < warpNum) ? sdata[warpTid] : make_double2(0.0, 0.0);
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
        _double2Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_double(double* _double1Dim, int number)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];
    double temp = (idx < number) ? _double1Dim[idx] : -DBL_MAX;

    __threadfence();
    gipc_block_max_full_to(temp, tep, number, idof, -DBL_MAX, _double1Dim + blockIdx.x);
}
