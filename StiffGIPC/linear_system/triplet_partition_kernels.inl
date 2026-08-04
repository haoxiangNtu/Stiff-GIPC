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

__device__ __forceinline__ int _contact_class_raw_start(
    const int* legacy_starts,
    int contact_class)
{
    // legacy storage order: ABD/ABD, ABD/FEM, FEM/ABD, FEM/FEM.
    constexpr int legacy_index[4] = {3, 1, 2, 0};
    return legacy_starts[legacy_index[contact_class]];
}

__device__ __forceinline__ int _contact_class_count(
    const int* legacy_starts,
    int contact_class,
    int payload_count)
{
    const int start =
        _contact_class_raw_start(legacy_starts, contact_class);
    if(start < 0)
        return 0;
    int end = payload_count;
    for(int next = contact_class + 1; next < 4; ++next)
    {
        const int next_start =
            _contact_class_raw_start(legacy_starts, next);
        if(next_start >= 0)
        {
            end = next_start;
            break;
        }
    }
    return end >= start ? end - start : 0;
}

__global__ void _publish_contact_class_counts(
    const int* legacy_starts,
    int payload_count,
    int c0,
    int c1,
    int c2,
    int c3,
    frame_fsm::FrameDeviceState* frame)
{
    if(blockIdx.x || threadIdx.x || !frame)
        return;
    const int capacity[4] = {c0, c1, c2, c3};
    int required = 0;
    bool overflow = false;
    for(int contact_class = 0; contact_class < 4; ++contact_class)
    {
        const int exact = _contact_class_count(
            legacy_starts, contact_class, payload_count);
        frame->contact_class_count[contact_class] = exact;
        if(exact > capacity[contact_class])
        {
            overflow = true;
            required += exact;
        }
        else
        {
            required += capacity[contact_class];
        }
    }
    atomicMax(&frame->hw_triplets, payload_count);
    if(!overflow)
        return;
    atomicMax(&frame->required_triplets, required);
    frame_fsm::fsm_record_error(frame,
                                frame_fsm::ERR_CAPACITY,
                                frame_fsm::OVF_TRIPLETS,
                                -1,
                                -1);
    atomicCAS(&frame->result,
              frame_fsm::FRAME_OK,
              frame_fsm::FRAME_RETRY_REQUIRED);
}

__global__ void _stage_contact_class_segment(
    const int*             row_ids_input,
    const int*             col_ids_input,
    const Eigen::Matrix3d* triplet_value_input,
    int*                   row_ids,
    int*                   col_ids,
    Eigen::Matrix3d*       triplet_value,
    const uint32_t*        sort_index,
    const int*             legacy_starts,
    int                    contact_class,
    int                    payload_count,
    int                    output_start,
    int                    output_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= output_capacity)
        return;
    const int count = _contact_class_count(
        legacy_starts, contact_class, payload_count);
    if(idx >= count)
    {
        // [graph-stale-tail] the downstream class convert consumes the FULL
        // trained width. A lane that merely returns leaves the PREVIOUS
        // iteration's triplet in its slot — a stale, valid-keyed, nonzero
        // contribution that contaminates the merged Hessian on frames whose
        // live count SHRINKS between Newton iterations (fs4: in-graph PCG at
        // 61-163 iters/Newton vs the host's 8-9 against the same tolerance).
        // Writing the zero triplet restores the pad-is-zero invariant the
        // graph layout was designed around: the (0,0) key dedups to one slot
        // and the zero-skip scatter drops the deposit entirely. Gate scenes
        // keep bitwise results (their tails were never-written zeros already).
        row_ids[output_start + idx] = 0;
        col_ids[output_start + idx] = 0;
        triplet_value[output_start + idx].setZero();
        return;
    }
    const int sorted_start =
        _contact_class_raw_start(legacy_starts, contact_class);
    const uint32_t source = sort_index[sorted_start + idx];
    row_ids[output_start + idx]       = row_ids_input[source];
    col_ids[output_start + idx]       = col_ids_input[source];
    triplet_value[output_start + idx] = triplet_value_input[source];
}
