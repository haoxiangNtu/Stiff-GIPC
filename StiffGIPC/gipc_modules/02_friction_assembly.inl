// ============================================================================
// gipc_modules/02 — post-E1c residue: generic moveMemory_1 template only.
// The friction Hessian kernels moved to energy/16_friction.inl (E1c) — one
// file per term, energy+G/H together. This template is a generic memory
// utility, not friction; device_common candidate for a later slice.
// ============================================================================
template <typename T>
__global__ inline void moveMemory_1(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

