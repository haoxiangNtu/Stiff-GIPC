#pragma once

#include"cuda_tools/cuda_device_buffer.h"
#include"Eigen/Eigen"

#define SymGH   // [symgh-validate] block-level upper-triangular storage (-37.5% triplets); half-finished per author — this branch is the strict validation run
#ifdef SymGH
#define M12_Off 10
#define M9_Off 6
#define M6_Off 3
#else
#define M12_Off 16
#define M9_Off 9
#define M6_Off 4
#endif


class GIPCTripletMatrix
{
  public:
    using BlockMatrix = Eigen::Matrix<double, 3, 3>;
    //using EntryValueType = T;
    //int Dimenstion              = M;
  public:
    cudatool::CudaDeviceBuffer<BlockMatrix> m_block_values;
    cudatool::CudaDeviceBuffer<int>         m_block_row_indices;
    cudatool::CudaDeviceBuffer<int>         m_block_col_indices;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_hash_value;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_sort_hash_value;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_sort_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_temp_buffer;
    int                                     m_block_rows = 0;
    int                                     m_block_cols = 0;

  public:
    GIPCTripletMatrix()                                    = default;
    ~GIPCTripletMatrix() { free_var(); }
    GIPCTripletMatrix(const GIPCTripletMatrix&)            = default;
    GIPCTripletMatrix(GIPCTripletMatrix&&)                 = default;
    GIPCTripletMatrix& operator=(const GIPCTripletMatrix&) = default;
    GIPCTripletMatrix& operator=(GIPCTripletMatrix&&)      = default;

    void reshape(int row, int col)
    {
        m_block_rows = row;
        m_block_cols = col;
    }

    void resize_triplets(size_t nonzero_count)
    {
        m_block_values.resize(nonzero_count);
        m_block_row_indices.resize(nonzero_count);
        m_block_col_indices.resize(nonzero_count);
    }

    // [P0-mem] pre-assembly growth: contents need NOT survive (assembly fully
    // rewrites [0:len) and the converter's scratch [len:2len) is written before read),
    // so grow free->malloc (no copy, no double residency). Margin is CAPPED in
    // absolute bytes (512MB of blocks) instead of a pure ratio: a 30% ratio on a
    // multi-GB buffer over-reserves by GBs, and that overshoot is what tips 24GB
    // cards over the edge at high env counts.
    void ensure_capacity_discard(size_t need)
    {
        if(m_block_values.capacity() >= need)
            return;
        size_t margin_cap = (size_t)(512ull * 1024 * 1024) / sizeof(BlockMatrix);
        size_t margin     = need * 3 / 10;
        if(margin > margin_cap) margin = margin_cap;
        size_t cap = need + margin;
        m_block_values.reserve_discard(cap);
        m_block_row_indices.reserve_discard(cap);
        m_block_col_indices.reserve_discard(cap);
    }

    void reserve_triplets(size_t nonzero_count)
    {
        m_block_values.reserve(nonzero_count);
        m_block_row_indices.reserve(nonzero_count);
        m_block_col_indices.reserve(nonzero_count);
    }

    // [towel-strict root fix] PRESERVING growth for the solve-time capacity
    // guarantee. The triplet stream is assembled in computeGradientAndHessian
    // BEFORE solve_linear_system()/build() runs, so at the pre-solve grow point
    // the buffer holds THIS iteration's LIVE matrix — a discarding grow there
    // destroys it the first time 2*live crosses the current capacity (observed:
    // towel strict, frame 26, 2*168469 > oldcap 336199 → whole matrix replaced
    // by stale pages → 32 phantom unique keys → SpMV OOB). Keeps the
    // [P0-mem] absolute margin cap so the fix does not reintroduce the multi-GB
    // over-reservation this call replaced.
    void ensure_capacity_preserve(size_t live_count, size_t need)
    {
        if(m_block_values.capacity() >= need)
            return;
        size_t margin_cap = (size_t)(512ull * 1024 * 1024) / sizeof(BlockMatrix);
        size_t margin     = need * 3 / 10;
        if(margin > margin_cap) margin = margin_cap;
        size_t cap = need + margin;
        resize_triplets(live_count);   // publish the live size: reserve()'s copy covers [0:live)
        reserve_triplets(cap);
    }

    void resize(int row, int col, size_t nonzero_count)
    {
        reshape(row, col);
        resize_triplets(nonzero_count);
    }

    void resize_collision_hash_size(size_t nonzero_count)
    {
        m_block_hash_value.resize(nonzero_count);
        m_block_sort_hash_value.resize(nonzero_count);
        m_block_index.resize(nonzero_count);
        m_block_sort_index.resize(nonzero_count);
        m_block_temp_buffer.resize(nonzero_count);
    }

    void reset_zero()
    {
        m_block_values.reset_zero();
        m_block_row_indices.reset_zero();
        m_block_col_indices.reset_zero();
    }

    void update_hash_value(int fem_offset);

    auto block_values(int offset = 0) { return m_block_values.data() + offset; }
    auto block_values(int offset = 0) const
    {
        return m_block_values.data() + offset;
    }
    auto block_row_indices(int offset = 0)
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_row_indices(int offset = 0) const
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0)
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0) const
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_hash_value(int offset = 0)
    {
        return m_block_hash_value.data() + offset;
    }
    auto block_hash_value(int offset = 0) const
    {
        return m_block_hash_value.data() + offset;
    }

    auto block_sort_hash_value(int offset = 0)
    {
        return m_block_sort_hash_value.data() + offset;
    }
    auto block_sort_hash_value(int offset = 0) const
    {
        return m_block_sort_hash_value.data() + offset;
    }

    auto block_temp_buffer(int offset = 0)
    {
        return m_block_temp_buffer.data() + offset;
    }
    auto block_temp_buffer(int offset = 0) const
    {
        return m_block_temp_buffer.data() + offset;
    }

    auto block_index(int offset = 0) { return m_block_index.data() + offset; }
    auto block_index(int offset = 0) const
    {
        return m_block_index.data() + offset;
    }

    auto block_sort_index(int offset = 0)
    {
        return m_block_sort_index.data() + offset;
    }
    auto block_sort_index(int offset = 0) const
    {
        return m_block_sort_index.data() + offset;
    }

    auto block_rows() const { return m_block_rows; }
    auto block_cols() const { return m_block_cols; }
    auto triplet_count() const { return m_block_values.size(); }
    auto triplet_capacity() const { return m_block_values.capacity(); }

    void clear()
    {
        m_block_rows = 0;
        m_block_cols = 0;
        m_block_values.clear();
        m_block_row_indices.clear();
        m_block_col_indices.clear();
    }
    int global_triplet_offset           = 0;
    int global_collision_triplet_offset = 0;
    int global_external_max_capcity     = 0;
    int global_internal_capcity         = 0;

    int* d_abd_abd_contact_start_id;
    int* d_abd_fem_contact_start_id;
    int* d_fem_abd_contact_start_id;
    int* d_fem_fem_contact_start_id;
    // [v0.8.5.1] The converter-published unique count is a persistent solver
    // input. It must never alias a scratch counter: the local-preconditioner
    // DeviceSelect count and the pinned-FEM extension counter both legitimately
    // overwrite their scratch after convert — with the old aliased layout
    // (d_contact_start_block + 4) that clobbered the unique count and a stale
    // host mirror could pick the garbage up (towel-strict OOB, 2026-07-24).
    int* d_unique_key_number;
    int* d_assembly_scratch_count = nullptr;

    // ②-D2H: one contiguous [5] block (abd_abd, abd_fem, fem_abd, fem_fem,
    // unique_key) so partitionContactHessian reads the 4 start-ids in a SINGLE
    // blocking D2H instead of 4 separate ones (each drains the GPU). The 4
    // pointers below alias offsets 0..3, so kernels that write them are
    // unchanged and the values are bit-identical.
    int* d_contact_start_block = nullptr;

    void init_var()
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_contact_start_block, 5 * sizeof(int)));
        d_abd_abd_contact_start_id = d_contact_start_block + 0;
        d_abd_fem_contact_start_id = d_contact_start_block + 1;
        d_fem_abd_contact_start_id = d_contact_start_block + 2;
        d_fem_fem_contact_start_id = d_contact_start_block + 3;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique_key_number, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_assembly_scratch_count, sizeof(int)));
    }

    void free_var()
    {
        CUDA_SAFE_CALL(cudaFree(d_contact_start_block));
        CUDA_SAFE_CALL(cudaFree(d_unique_key_number));
        CUDA_SAFE_CALL(cudaFree(d_assembly_scratch_count));
    }

    int h_abd_abd_contact_start_id = -1;
    int h_abd_fem_contact_start_id = -1;
    int h_fem_abd_contact_start_id = -1;
    int h_fem_fem_contact_start_id = -1;
    int h_unique_key_number        = 0;

    uint32_t abd_abd_contact_num = 0;
    uint32_t abd_fem_contact_num = 0;
    uint32_t fem_fem_contact_num = 0;
    uint32_t fem_abd_contact_num = 0;
};
