#pragma once

#include <cstdlib>
#include <limits>

namespace gipc
{
// P3a sparse-assembly tiers: 256-item aligned powers of two.  Every kernel and
// CUB primitive in a tier uses this capacity as its shape; the exact item count
// is only a device/guard value.  The shift hook qualifies cross-tier bitwise
// equivalence without changing production defaults.
inline int assembly_capacity_tier(int count)
{
    if(count <= 0)
        return 0;

    unsigned int blocks = (static_cast<unsigned int>(count) + 255u) / 256u;
    unsigned int tier_blocks = 1u;
    while(tier_blocks < blocks && tier_blocks < (1u << 22))
        tier_blocks <<= 1u;

    if(const char* shift_text = std::getenv("STIFF_ASSEMBLY_TIER_SHIFT"))
    {
        int shift = std::atoi(shift_text);
        while(shift-- > 0 && tier_blocks <= (1u << 21))
            tier_blocks <<= 1u;
    }

    const unsigned long long tier = static_cast<unsigned long long>(tier_blocks) * 256ull;
    return tier <= static_cast<unsigned long long>(std::numeric_limits<int>::max())
               ? static_cast<int>(tier)
               : count;
}
}  // namespace gipc
