#include <gipc/statistics.h>
#include <cstdlib>
#include <fstream>
namespace gipc
{
Statistics::Statistics() {}

void Statistics::write_to_file(const std::string& filename)
{
    if(std::getenv("GIPC_STATS_ENABLED") == nullptr) return;
    std::ofstream file(filename);
    file << m_json.dump(4);
}
}  // namespace gipc
