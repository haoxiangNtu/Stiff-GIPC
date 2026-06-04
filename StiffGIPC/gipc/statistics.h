#pragma once
#include <gipc/utils/json.h>
#include <cstdlib>
#include <cstddef>
namespace gipc
{
class Statistics
{
  private:
    Json   m_json;
    size_t m_frame = 0;
    Json   m_scratch;
    size_t m_scratch_frame = static_cast<size_t>(-1);
    Statistics();

    static bool enabled()
    {
        static bool flag = (std::getenv("GIPC_STATS_ENABLED") != nullptr);
        return flag;
    }

  public:
    auto&              json() { return m_json; }
    static Statistics& instance()
    {
        thread_local static Statistics instance;
        return instance;
    }
    auto& at_frame(int i) { return m_json["frames"][i]; }
    auto  frame(int i) { m_frame = i; }
    auto  frame() { return m_frame; }
    auto& at_current_frame()
    {
        if(!enabled())
        {
            if(m_scratch_frame != m_frame)
            {
                m_scratch       = Json::object();
                m_scratch_frame = m_frame;
            }
            return m_scratch;
        }
        return m_json["frames"][m_frame];
    }
    void write_to_file(const std::string& filename);
};
}  // namespace gipc
