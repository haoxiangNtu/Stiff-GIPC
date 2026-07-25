// ============================================================================
// device_common/device_buffer.cuh — RAII owner for raw device buffers
// (v0.8.6 Phase 3d).
//
// MECHANICS only: allocation lifetime, free-then-null discipline, capacity
// bookkeeping. POLICY (when to grow, growth factor, what must be preserved)
// stays at call sites — this class deliberately has NO preserve/copy path;
// the global triplet matrix keeps its own ensure_capacity_{preserve,discard}
// machinery (towel-strict contract) and contact/pair_buffers.cuh keeps the
// DCD<=CCD lockstep. Do not fold those in here.
//
// Implicit T* conversion keeps consumer sites textually unchanged: a member
// converted from `double* x` to `DeviceBuffer<double> x` still works as
// kernel<<<...>>>(x, ...), cudaMemset(x, ...), x + offset, if(x).
//
// Invariants enforced by the type (previously call-site disciplines):
//   - a free ALWAYS nulls the pointer (no dangling reuse);
//   - release() is idempotent (double-free cannot compile into existence);
//   - destruction releases (silent leak-shaped bugs like the P5
//     _MatIndex_last ghost buffer can no longer survive unnoticed);
//   - non-copyable (two owners of one device allocation do not compile).
// ============================================================================
#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_tools/cuda_tools.h"

template <typename T>
class DeviceBuffer
{
  public:
    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    // Unchecked free on destruction: the destructor may run during process
    // teardown after the CUDA context is gone; explicit call sites use the
    // checked release() below.
    ~DeviceBuffer()
    {
        if(m_ptr)
            cudaFree(m_ptr);
    }

    // implicit raw views — see header note
    operator T*() { return m_ptr; }
    operator const T*() const { return m_ptr; }
    T*       data() { return m_ptr; }
    const T* data() const { return m_ptr; }
    size_t   capacity() const { return m_cap; }  // elements

    // Discard-grow to exactly n elements; old contents are NOT preserved
    // (that is the point — callers needing preservation are in the wrong
    // class, see header note).
    void resize_discard(size_t n)
    {
        release();
        CUDA_SAFE_CALL(cudaMalloc((void**)&m_ptr, n * sizeof(T)));
        m_cap = n;
    }

    // Checked free + null; idempotent.
    void release()
    {
        if(m_ptr)
        {
            CUDA_SAFE_CALL(cudaFree(m_ptr));
            m_ptr = nullptr;
            m_cap = 0;
        }
    }

  private:
    T*     m_ptr = nullptr;
    size_t m_cap = 0;
};
