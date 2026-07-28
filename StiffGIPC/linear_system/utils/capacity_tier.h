#pragma once

#include <cstdlib>
#include <limits>

namespace gipc
{
// Stable CUDA-graph launch shape for sparse assembly. The exact count remains
// device-resident; host code chooses only this allocation-backed upper bound.
inline int assembly_capacity_tier(int count)
{
    if(count <= 0)
        return 0;

    unsigned int blocks =
        (static_cast<unsigned int>(count) + 255u) / 256u;
    unsigned int tier_blocks = 1u;
    while(tier_blocks < blocks && tier_blocks < (1u << 22))
        tier_blocks <<= 1u;

    if(const char* text = std::getenv("STIFF_ASSEMBLY_TIER_SHIFT"))
    {
        int shift = std::atoi(text);
        while(shift-- > 0 && tier_blocks <= (1u << 21))
            tier_blocks <<= 1u;
    }

    const unsigned long long tier =
        static_cast<unsigned long long>(tier_blocks) * 256ull;
    return tier
                   <= static_cast<unsigned long long>(
                       std::numeric_limits<int>::max())
               ? static_cast<int>(tier)
               : count;
}
}  // namespace gipc
