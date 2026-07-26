// ============================================================================
// device_common/mirrors.h — HostMirror: host-side mirrors of device truth,
// with an opt-in staleness audit (v0.8.6 Phase 3b).
//
// A "mirror" is a host copy of a device-resident value (pair counts, key
// counts, ...). Its implicit contract — refresh (D2H) after every device-side
// change, BEFORE any host read — used to live in people's heads; the h_close_*
// dead-mirror chain and the pre-snapshot narrow-self bug (GIPC.cuh, [narrow-
// self snapshot] note) are what silent violations look like. These wrappers
// make the contract explicit and machine-checkable:
//
//   writer marks truth changed:   mirror.invalidate();       (build entry)
//   refresh point:                cudaMemcpy(mirror.refresh_dst(), ...);
//                                 memcpy(mirror.refresh_dst(), ...);
//                                 mirror = value;            (assign = fresh)
//   consumer:                     uint32_t n = mirror;  /  mirror[k]
//
// Audit: OFF by default — reads compile to the bare value plus one cached
// boolean test, no behavior change. STIFF_MIRROR_AUDIT=1 (env) arms it:
// reading a mirror whose device truth changed since the last refresh throws
// std::runtime_error naming the mirror. Fail-loud diagnostics, same spirit
// as STIFF_PHASE_TIME / the planned STIFF_SLOT_AUDIT.
//
// Escape hatch: raw_alias() hands out a raw pointer the audit cannot see
// (needed for long-lived consumers like MAS_Preconditioner's uint32_t*).
// Every raw_alias() call site must justify itself in a comment.
//
// Families: (3b) h_cpNum[5]/h_gpNum/h_ccd_cpNum — counts-after-build,
// invalidated at build entries; (3b-2) h_cpNum_last[5]/h_gpNum_last —
// lagged-friction snapshots, value = "as of last snapshot", refresh-only
// lifecycle (no invalidation sites by design, never stale between snapshots).
// ============================================================================
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gipc_mirror_detail
{
inline bool audit_enabled()
{
    static const bool on = (std::getenv("STIFF_MIRROR_AUDIT") != nullptr);
    return on;
}
[[noreturn]] inline void stale(const char* name)
{
    throw std::runtime_error(std::string("[mirror-audit] stale read of ") + name
                             + ": device truth changed after the last refresh "
                               "(missing refresh, or a read between build entry "
                               "and its D2H copy-back)");
}
}  // namespace gipc_mirror_detail

template <typename T>
class HostMirror
{
  public:
    explicit HostMirror(const char* name, T v = T{})
        : m_v(v), m_name(name)
    {
    }

    operator T() const  // audited read
    {
        if(gipc_mirror_detail::audit_enabled() && !m_fresh)
            gipc_mirror_detail::stale(m_name);
        return m_v;
    }
    T get() const { return static_cast<T>(*this); }  // for variadic (printf) sites
    const T* read_ptr() const  // audited address-of read (e.g. H2D upload source)
    {
        if(gipc_mirror_detail::audit_enabled() && !m_fresh)
            gipc_mirror_detail::stale(m_name);
        return &m_v;
    }

    HostMirror& operator=(T v)  // assignment IS a refresh (writer states truth)
    {
        m_v     = v;
        m_fresh = true;
        return *this;
    }
    T* refresh_dst()  // D2H/memcpy target; statement completes the refresh
    {
        m_fresh = true;
        return &m_v;
    }
    void invalidate() { m_fresh = false; }  // call where device truth changes

  private:
    T           m_v;
    bool        m_fresh = true;  // initial value is authoritative (zero state)
    const char* m_name;
};

template <typename T, int N>
class HostMirrorArray
{
  public:
    explicit HostMirrorArray(const char* name) : m_name(name)
    {
        for(int i = 0; i < N; ++i)
            m_v[i] = T{};
    }

    T operator[](int i) const  // audited read (by value: variadic-safe)
    {
        if(gipc_mirror_detail::audit_enabled() && !m_fresh)
            gipc_mirror_detail::stale(m_name);
        return m_v[i];
    }
    void set(int i, T v) { m_v[i] = v; }  // targeted write, freshness unchanged
                                          // (export-time override/restore)
    T* refresh_dst()  // memcpy/cudaMemcpy target for all N; marks fresh
    {
        m_fresh = true;
        return m_v;
    }
    void invalidate() { m_fresh = false; }
    // Audit-invisible long-lived alias (e.g. MAS_Preconditioner keeps a
    // uint32_t* for its lifetime). Justify every call site.
    T* raw_alias() { return m_v; }

  private:
    T           m_v[N];
    bool        m_fresh = true;
    const char* m_name;
};
