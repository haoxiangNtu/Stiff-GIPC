#pragma once
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace gipc_liquid
{
inline void liquid_require(bool ok, const char* message)
{
    if(!ok) throw std::invalid_argument(std::string("[liquid] ") + message);
}

inline bool finite_positive(double x) { return std::isfinite(x) && x > 0.0; }
inline bool finite_nonnegative(double x) { return std::isfinite(x) && x >= 0.0; }

// Check before a floating-point to int conversion or an int product occurs.
inline int checked_cells(double lo, double hi, double dx)
{
    liquid_require(finite_positive(dx), "dx must be finite and positive");
    liquid_require(std::isfinite(lo) && std::isfinite(hi) && hi > lo,
                   "domain bounds must be finite and strictly increasing");
    const double n = std::ceil((hi - lo) / dx);
    liquid_require(std::isfinite(n) && n >= 2 && n < std::numeric_limits<int>::max(),
                   "domain needs 2..INT_MAX-1 cells per axis");
    return static_cast<int>(n);
}

inline int checked_grid_count(int x, int y, int z, int extra = 0)
{
    long long count = 1;
    for(int d : {x, y, z})
    {
        liquid_require(d >= 2 && d < std::numeric_limits<int>::max(), "invalid grid dimensions");
        const long long extent = static_cast<long long>(d) + extra;
        liquid_require(count <= std::numeric_limits<int>::max() / extent,
                       "grid exceeds 32-bit indexing capacity");
        count *= extent;
    }
    return static_cast<int>(count);
}

inline double liquid_env(const char* name, double fallback)
{
    const char* value = std::getenv(name);
    if(!value) return fallback;
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if(end == value || *end != '\0' || errno == ERANGE || !std::isfinite(parsed))
        throw std::invalid_argument(std::string("[liquid] invalid finite number for ") + name);
    return parsed;
}

inline int liquid_env(const char* name, int fallback)
{
    const double value = liquid_env(name, static_cast<double>(fallback));
    liquid_require(value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max()
                       && std::floor(value) == value, "integer environment option is out of range or fractional");
    return static_cast<int>(value);
}
} // namespace gipc_liquid
