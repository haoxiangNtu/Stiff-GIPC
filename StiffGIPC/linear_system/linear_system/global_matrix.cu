#include "linear_system/linear_system/global_matrix.h"
#include "cuda_tools/cuda_tools.h"
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>


__global__ void _set_hash_value(const int* row_ids,
                                const int* col_ids,
                                uint32_t*  index,
                                uint64_t*  hashValue,
                                int        abd_vert_num,
                                int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    index[idx]     = idx;
    //hashValue[idx] = (((uint64_t)rows[idx]) << 32) | ((uint64_t)cols[idx]);
    uint64_t self_hash;
    if(row_ids[idx] < abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 3;
    }
    else if(row_ids[idx] < abd_vert_num && col_ids[idx] >= abd_vert_num)
    {
        self_hash = 1;
    }
    else if(row_ids[idx] >= abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 2;
    }
    else
    {
        self_hash = 0;
    }
    hashValue[idx] = self_hash;
}



void GIPCTripletMatrix::update_hash_value(int fem_offset)
{
    //reset_zero();
    int threadNum = 256;
    int blockNum = (global_collision_triplet_offset + threadNum - 1) / threadNum;

    if(global_collision_triplet_offset > global_external_max_capcity)
    {
        global_external_max_capcity = global_collision_triplet_offset;
        resize_collision_hash_size(global_collision_triplet_offset);
    }

    LaunchCudaKernal(blockNum,
                     threadNum,
                     0,
                     _set_hash_value,
                     (const int*)m_block_row_indices.data(),
                     (const int*)m_block_col_indices.data(),
                     m_block_index.data(),
                     m_block_hash_value.data(),
                     fem_offset,
                     global_collision_triplet_offset);
}


// ---- [3c] slot-contract audit (STIFF_SLOT_AUDIT=1) ----
namespace
{
inline bool slot_audit_enabled()
{
    static const bool on = (std::getenv("STIFF_SLOT_AUDIT") != nullptr);
    return on;
}
__global__ void _slot_audit_scan(const int* rows, int n, int* out)  // out: {count, first}
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    if(rows[i] == -1)  // 0xFFFFFFFF sentinel survived = reserved, never written
    {
        atomicAdd(out, 1);
        atomicMin(out + 1, i);
    }
}
}  // namespace

void GIPCTripletMatrix::slot_audit_arm()
{
    if(!slot_audit_enabled())
        return;
    CUDA_SAFE_CALL(cudaMemset(m_block_row_indices.data(), 0xFF,
                              m_block_row_indices.capacity() * sizeof(int)));
}

void GIPCTripletMatrix::slot_audit_check_and_restore(const char* context)
{
    if(!slot_audit_enabled())
        return;
    const long long live = global_triplet_offset;
    const size_t    cap  = m_block_row_indices.capacity();
    static int*     d_out = nullptr;  // audit-only scratch, intentionally leaked
    if(!d_out)
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, 2 * sizeof(int)));
    const int init[2] = {0, INT_MAX};
    CUDA_SAFE_CALL(cudaMemcpy(d_out, init, 2 * sizeof(int), cudaMemcpyHostToDevice));
    if(live > 0)
    {
        const int bs = 256;
        _slot_audit_scan<<<(int)((live + bs - 1) / bs), bs>>>(
            m_block_row_indices.data(), (int)live, d_out);
    }
    int host_out[2];
    CUDA_SAFE_CALL(cudaMemcpy(host_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    if(host_out[0] > 0)
        throw std::runtime_error(
            std::string("[slot-audit] ") + std::to_string(host_out[0])
            + " reserved-but-unwritten triplet slot(s) in " + context
            + ", first at block index " + std::to_string(host_out[1])
            + ", live=" + std::to_string(live)
            + " — an assembly kernel reserved offsets it never filled");
    if((long long)cap > live)
        CUDA_SAFE_CALL(cudaMemset(m_block_row_indices.data() + live, 0,
                                  (cap - (size_t)live) * sizeof(int)));
}
