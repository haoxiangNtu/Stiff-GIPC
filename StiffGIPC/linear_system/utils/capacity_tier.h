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

    // [tier-granularity] The power-of-two ladder rounds a count up by as much
    // as 2x (measured: 971162 live -> 2097152 tier = 2.16x). Padding is a
    // LINEAR cost in this engine -- doubling the headroom multiplier costs
    // +19% and quadrupling it +68% on A800 fs4-4env -- so the ladder's own
    // rounding is worth roughly as much as the entire headroom parameter.
    // STIFF_TIER_STEPS=<n> inserts n-1 geometric rungs between each power of
    // two (n=1 is the legacy ladder, n=2 adds the 1.5x rung, n=4 the
    // 1.25/1.5/1.75 rungs), trading finer widths for more frequent
    // re-records, which measure only 25-28 ms.
    static int s_steps = -1;
    if(s_steps < 0)
    {
        const char* e = std::getenv("STIFF_TIER_STEPS");
        s_steps       = e && e[0] ? std::atoi(e) : 1;
        if(s_steps < 1)
            s_steps = 1;
        if(s_steps > 8)
            s_steps = 8;
    }
    unsigned int tier_blocks = 1u;
    if(s_steps <= 1)
    {
        while(tier_blocks < blocks && tier_blocks < (1u << 22))
            tier_blocks <<= 1u;
    }
    else
    {
        // Walk the geometric ladder base*(1 + k/steps), k = 0..steps-1.
        while(tier_blocks < blocks && tier_blocks < (1u << 22))
        {
            unsigned int next = tier_blocks << 1u;
            bool         hit  = false;
            for(int k = 1; k < s_steps; ++k)
            {
                const unsigned long long rung =
                    static_cast<unsigned long long>(tier_blocks)
                    * (s_steps + k) / s_steps;
                if(rung >= blocks)
                {
                    tier_blocks = static_cast<unsigned int>(rung);
                    hit         = true;
                    break;
                }
            }
            if(hit)
                break;
            tier_blocks = next;
        }
    }

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
