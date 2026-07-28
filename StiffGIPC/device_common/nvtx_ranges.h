#pragma once
// [phase-attribution] STIFF_NVTX=1 wraps solver phases in NVTX ranges so nsys
// can attribute API calls (blocking memcpys, launches, syncs) per phase —
// the measurement instrument behind the GPU-residency campaign (Phase B3).
// Default off: one cached getenv, zero per-call cost. Header-only nvtx3.
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

inline bool gipc_nvtx_on()
{
    static int s_on = -1;
    if(s_on < 0)
    {
        const char* e = std::getenv("STIFF_NVTX");
        s_on = (e && e[0] == '1') ? 1 : 0;
    }
    return s_on == 1;
}
inline void gipc_nvtx_push(const char* name)
{
    if(gipc_nvtx_on())
        nvtxRangePushA(name);
}
inline void gipc_nvtx_pop()
{
    if(gipc_nvtx_on())
        nvtxRangePop();
}
struct GipcNvtxScope
{
    explicit GipcNvtxScope(const char* name) { gipc_nvtx_push(name); }
    ~GipcNvtxScope() { gipc_nvtx_pop(); }
};
