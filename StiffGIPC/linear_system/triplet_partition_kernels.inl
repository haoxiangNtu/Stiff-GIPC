// Collision-triplet partition/reorder kernels.

__global__ void _partition_collision_triplets(const uint64_t* sort_hash,
                                              int*            abd_abd_offset,
                                              int*            abd_fem_offset,
                                              int*            fem_abd_offset,
                                              int*            fem_fem_offset,
                                              int             number)
{
    extern __shared__ int shared_hash[];
    unsigned int          idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int                   self_hash;
    if(idx < number)
    {
        self_hash                    = sort_hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;
        if(idx > 0 && threadIdx.x == 0)
        {
            shared_hash[0] = sort_hash[idx - 1];
        }
    }
    __syncthreads();
    if(idx < number)
    {
        int prior_hash = idx == 0 ? -1 : shared_hash[threadIdx.x];
        if(self_hash != prior_hash)
        {
            if(self_hash == 3)
            {
                *abd_abd_offset = idx;
            }
            else if(self_hash == 1)
            {
                *abd_fem_offset = idx;
            }
            else if(self_hash == 2)
            {
                *fem_abd_offset = idx;
            }
            else if(self_hash == 0)
            {
                *fem_fem_offset = idx;
            }
        }
    }
}

__global__ void _reorder_triplets(int*             row_ids_input,
                                  int*             col_ids_input,
                                  Eigen::Matrix3d* triplet_value_inpuit,
                                  int*             row_ids,
                                  int*             col_ids,
                                  Eigen::Matrix3d* triplet_value,
                                  const uint32_t*  sort_index,
                                  int              number)
{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= number)
        return;
    row_ids[idx]       = row_ids_input[sort_index[idx]];
    col_ids[idx]       = col_ids_input[sort_index[idx]];
    triplet_value[idx] = triplet_value_inpuit[sort_index[idx]];
}

__global__ void _reorder_triplet_segment(
    const int*             row_ids_input,
    const int*             col_ids_input,
    const Eigen::Matrix3d* triplet_value_input,
    int*                   row_ids,
    int*                   col_ids,
    Eigen::Matrix3d*       triplet_value,
    const uint32_t*        sort_index,
    int                    sorted_start,
    int                    output_start,
    int                    number)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    const uint32_t source = sort_index[sorted_start + idx];
    row_ids[output_start + idx]       = row_ids_input[source];
    col_ids[output_start + idx]       = col_ids_input[source];
    triplet_value[output_start + idx] = triplet_value_input[source];
}

__global__ void _compact_triplet_segment(int*             row_ids,
                                         int*             col_ids,
                                         Eigen::Matrix3d* triplet_value,
                                         int              input_start,
                                         int              output_start,
                                         int              number)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    row_ids[output_start + idx]       = row_ids[input_start + idx];
    col_ids[output_start + idx]       = col_ids[input_start + idx];
    triplet_value[output_start + idx] = triplet_value[input_start + idx];
}
