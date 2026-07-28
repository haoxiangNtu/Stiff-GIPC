#pragma once
// [B2'-b] host-side coordination for the cross-solve PCG graph cache.
//
// pcg_grid_capacity_mode(): set (only) while recording the cached device-loop
// graph — launch-grid sizing in spmv / SchwarzLocalXSym switches from live
// counts to capacity bounds so the recorded grids stay valid across solves.
// The kernels mask by device-resident counts either way, so oversized grids
// are semantically identical (extra warps exit on the first bound check).
//
// pcg_buffer_generation(): bumped by ANY realloc that can move a pointer the
// captured graph baked in (triplet storage, MAS output buffers). A mismatch
// against the generation stored with the cached exec forces a re-capture.
// Deliberately one coarse counter: spurious bumps only cost a re-record.
inline int& pcg_grid_capacity_mode()
{
    static int v = 0;
    return v;
}

inline long long& pcg_buffer_generation()
{
    static long long g = 0;
    return g;
}
