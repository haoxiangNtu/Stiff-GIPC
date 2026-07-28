__global__ void _schwarzLocalXSym3(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number,
                                   const int2*                      d_lvl,
                                   int                              lvln)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // [B2'-b] live cluster total from the device hierarchy when armed; the
    // host grid still uses the (equal) mirror, so launches stay identical.
    if(d_lvl)
        number = d_lvl[lvln].y * BANKSIZE * 3;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    int r3id = MRid % 3;

    int    lvrid = vrid % BANKSIZE;
    int    lvcid = vcid % BANKSIZE;
    FloatP rdata = 0;

    __shared__ Eigen::Vector3f smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(lvcid >= lvrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index](r3id, 0) * smR[lvcid][0]
                + Pred[Hid].M[index](r3id, 1) * smR[lvcid][1]
                + Pred[Hid].M[index](r3id, 2) * smR[lvcid][2];
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index](0, r3id) * smR[lvcid][0]
                + Pred[Hid].M[index](1, r3id) * smR[lvcid][1]
                + Pred[Hid].M[index](2, r3id) * smR[lvcid][2];
    }
    //__syncthreads();
    int landidx = threadIdx.x % BANKSIZE;
    // Static segment geometry (BANKSIZE-aligned rows inside the warp): the
    // former ballot/brev/clz path computed `mark << 32` on physical lane 31,
    // which is C++ undefined behavior. interval = distance to my row's end.
    bool         bBoundary = (landidx == 0);
    unsigned int interval  = (BANKSIZE - 1) - landidx;

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}


__global__ void _schwarzLocalXSym6(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number,
                                   const int2*                      d_lvl,
                                   int                              lvln)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(d_lvl)
        number = d_lvl[lvln].y * BANKSIZE;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * BANKSIZE);

    int Hid   = idx / hessianSize;
    int lvrid = (idx % hessianSize) / (BANKSIZE);
    int lvcid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + lvrid;
    int vcid = Hid * BANKSIZE + lvcid;

    Eigen::Vector3f rdata;
    //rdata.setZero();

    __shared__ Eigen::Vector3f smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(vcid >= vrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index] * smR[lvcid];
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].transpose() * smR[lvcid];
    }
    //__syncthreads();
    int landidx = threadIdx.x % BANKSIZE;
    // Static segment geometry (BANKSIZE-aligned rows inside the warp): the
    // former ballot/brev/clz path computed `mark << 32` on physical lane 31,
    // which is C++ undefined behavior. interval = distance to my row's end.
    bool         bBoundary = (landidx == 0);
    unsigned int interval  = (BANKSIZE - 1) - landidx;

    int maxSize = std::min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata[0], iter);
        FloatP tmpy = __shfl_down_sync(0xffffffff, rdata[1], iter);
        FloatP tmpz = __shfl_down_sync(0xffffffff, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        // [4.3] binned deposit instead of float atomicAdd → order-independent ⇒ deterministic
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 0) * BINNED_K, (double)rdata[0]);
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 1) * BINNED_K, (double)rdata[1]);
        binned_deposit(g_mZbin + ((size_t)vrid * 3 + 2) * BINNED_K, (double)rdata[2]);
    }
}

// Fused coarse-mR combine + Schwarz apply. Each 16-row matrix bank is split
// into two independent 8-row/128-thread blocks. All threads in a block reach
// the shared-memory barrier; the launch geometry itself is bounded on the host
// by the current hierarchy count and m_clusterCap.
__global__ void _schwarzLocalXSym6_fused8(
    const __GEIGEN__::MasMatrixSymf* Pred,
    const Eigen::Vector3f*           mR,
    const double*                    mRbin,
    double*                          mZbin,
    int                              totalMapNodes,
    int                              clusterCount,
    size_t                           clusterCapacity,
    const int2*                     extent)
{
    if(extent)
        clusterCount = extent->y;
    constexpr int rowsPerBlock = 8;
    const int     matrixId      = blockIdx.x / 2;
    if(matrixId * BANKSIZE >= clusterCount)
        return;
    const int     rowBase       = (blockIdx.x & 1) * rowsPerBlock;
    const int     localRow      = rowBase + threadIdx.x / BANKSIZE;
    const int     localCol      = threadIdx.x % BANKSIZE;
    const int     row           = matrixId * BANKSIZE + localRow;

    // Eigen::Vector3f has a non-trivial constructor, so use a plain shared
    // layout and load each bank once per half-bank block.
    __shared__ float smR[BANKSIZE][3];
    if(threadIdx.x < BANKSIZE)
    {
        const int node  = matrixId * BANKSIZE + threadIdx.x;
        const bool fine = node < totalMapNodes;
        const bool inRange = node >= 0 && node < clusterCount
                             && static_cast<size_t>(node) < clusterCapacity;
        if(inRange && fine)
        {
            smR[threadIdx.x][0] = mR[node][0];
            smR[threadIdx.x][1] = mR[node][1];
            smR[threadIdx.x][2] = mR[node][2];
        }
        else if(inRange)
        {
            smR[threadIdx.x][0] = _mas_comb_component(mRbin, node, 0);
            smR[threadIdx.x][1] = _mas_comb_component(mRbin, node, 1);
            smR[threadIdx.x][2] = _mas_comb_component(mRbin, node, 2);
        }
        else
        {
            smR[threadIdx.x][0] = 0.0f;
            smR[threadIdx.x][1] = 0.0f;
            smR[threadIdx.x][2] = 0.0f;
        }
    }
    __syncthreads();

    Eigen::Vector3f input;
    input[0] = smR[localCol][0];
    input[1] = smR[localCol][1];
    input[2] = smR[localCol][2];
    Eigen::Vector3f rdata;
    if(localCol >= localRow)
    {
        int index = BANKSIZE * localRow - localRow * (localRow + 1) / 2 + localCol;
        rdata     = Pred[matrixId].M[index] * input;
    }
    else
    {
        int index = BANKSIZE * localCol - localCol * (localCol + 1) / 2 + localRow;
        rdata     = Pred[matrixId].M[index].transpose() * input;
    }

    // Same static BANKSIZE-aligned segment geometry as v0.8.4.2's corrected
    // Sym6 kernel. In particular, there is no lane-31 left shift by 32.
    const int          laneInRow = threadIdx.x % BANKSIZE;
    const bool         rowHead   = laneInRow == 0;
    const unsigned int interval  = (BANKSIZE - 1) - laneInRow;
#pragma unroll
    for(int offset = 1; offset < BANKSIZE; offset <<= 1)
    {
        float x = __shfl_down_sync(0xffffffffu, rdata[0], offset);
        float y = __shfl_down_sync(0xffffffffu, rdata[1], offset);
        float z = __shfl_down_sync(0xffffffffu, rdata[2], offset);
        if(interval >= static_cast<unsigned int>(offset))
        {
            rdata[0] += x;
            rdata[1] += y;
            rdata[2] += z;
        }
    }

    if(rowHead && row < clusterCount
       && static_cast<size_t>(row) < clusterCapacity)
    {
        binned_deposit(mZbin + ((size_t)row * 3 + 0) * BINNED_K,
                       static_cast<double>(rdata[0]));
        binned_deposit(mZbin + ((size_t)row * 3 + 1) * BINNED_K,
                       static_cast<double>(rdata[1]));
        binned_deposit(mZbin + ((size_t)row * 3 + 2) * BINNED_K,
                       static_cast<double>(rdata[2]));
    }
}


__device__ void get_index(int& row, int& col, const int& hash, const int& size)
{
    //row = 0;
    for(row = 0; row < size; row++)
    {
        col = hash - size * row + row * (row + 1) / 2;
        if(col >= 0 && col < size)
        {
            if(size * row - row * (row + 1) / 2 + col == hash)
                return;
        }
    }
}


__global__ void _schwarzLocalXSym9(const __GEIGEN__::MasMatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number,
                                   const int2*                      d_lvl,
                                   int                              lvln)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(d_lvl)
        number = (d_lvl[lvln].y / BANKSIZE) * ((1 + BANKSIZE) * BANKSIZE / 2);
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * (1 + BANKSIZE)) / 2;

    int Hid   = idx / hessianSize;
    int index = (idx % hessianSize);
    int lvrid, lvcid;
    get_index(lvrid, lvcid, index, BANKSIZE);

    int vrid = Hid * BANKSIZE + lvrid;
    int vcid = Hid * BANKSIZE + lvcid;

    __shared__ int row_ids[BANKSIZE * BANKSIZE];
    row_ids[threadIdx.x] = vrid;

    __syncthreads();
    int prev_i = -1;
    if(threadIdx.x > 0)
    {
        prev_i = row_ids[threadIdx.x - 1];
    }

    auto block_value = Pred[Hid].M[index];
    Eigen::Vector3f rdata = block_value * mR[vcid];

    if(vrid != vcid)  // process lower triangle
    {
        Eigen::Vector3f vec_ =
            block_value.transpose() * mR[vrid];

        atomicAdd((&(mZ[vcid].x)), vec_[0]);
        atomicAdd((&(mZ[vcid].y)), vec_[1]);
        atomicAdd((&(mZ[vcid].z)), vec_[2]);
    }


    int warpId = threadIdx.x & 0x1f;
    //int lane_id = threadIdx.x % BANKSIZE;

    bool bBoundary = (warpId == 0) || (prev_i != vrid);
    auto mask_val  = __activemask();

    unsigned int mark = __ballot_sync(mask_val, bBoundary);  // a bit-mask
    mark              = __brev(mark);
    // guard the lane-31 case: a 32-bit shift by 32 is C++ UB (this kernel is
    // currently uncalled but kept compilable and UB-free; its segment
    // boundary is data-dependent so the static-geometry rewrite of Sym3/Sym6
    // does not apply here)
    int          clzlen   = (warpId >= 31) ? 32 : __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    mark = interval;
    for(int iter = 1; iter & 0x1f; iter <<= 1)
    {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        if(tmp > mark)
            mark = tmp;
    }
    int maxSize = __shfl_sync(mask_val, mark, 0);
    //__syncthreads();

    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        float tmpx = __shfl_down_sync(mask_val, rdata[0], iter);
        float tmpy = __shfl_down_sync(mask_val, rdata[1], iter);
        float tmpz = __shfl_down_sync(mask_val, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x)), rdata[0]);
        atomicAdd((&(mZ[vrid].y)), rdata[1]);
        atomicAdd((&(mZ[vrid].z)), rdata[2]);
    }
}

