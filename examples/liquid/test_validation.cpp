// Host-only tests: no CUDA initialization or simulation.
#include "../../StiffGIPC/mpm/liquid_validation.h"
#include <cassert>
#include <cstdio>
#include <functional>

using namespace gipc_liquid;
int main()
{
    int rejected = 0;
    auto reject = [&](const std::function<void()>& f) {
        bool caught = false;
        try { f(); } catch(const std::invalid_argument&) { caught = true; }
        assert(caught);
        ++rejected;
    };
    assert(checked_cells(0, 1, .005) == 200);
    assert(checked_grid_count(88, 84, 144) == 1064448);
    assert(checked_grid_count(88, 84, 144, 1) == 1096925);
    reject([] { checked_cells(0, 1, 0); });
    reject([] { checked_cells(0, 1, 1e-300); });
    reject([] { checked_cells(1, 0, .005); });
    reject([] { checked_grid_count(100000, 100000, 100000); });
    reject([] { checked_grid_count(2147483646, 2, 2, 1); });
    reject([] { checked_grid_count(1, 2, 2); });
    const char* key = "LIQ_VALIDATION_TEST";
    unsetenv(key);
    assert(liquid_env(key, 6) == 6);
    for(const char* value : {"nan", "inf", "", "abc", "1foo", "1e999"})
    {
        setenv(key, value, 1);
        reject([&] { liquid_env(key, 0.); });
    }
    for(const char* value : {"1.5", "2147483648", "-2147483649"})
    {
        setenv(key, value, 1);
        reject([&] { liquid_env(key, 0); });
    }
    setenv(key, "6", 1);
    assert(liquid_env(key, 0) == 6);
    setenv(key, "0.0005", 1);
    assert(liquid_env(key, 0.) == .0005);
    unsetenv(key);
    std::printf("PASS host validation: %d rejected, valid dimensions and options accepted\n", rejected);
}
