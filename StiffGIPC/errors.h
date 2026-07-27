#pragma once

#include <stdexcept>
#include <string>

namespace gipc
{
enum class ErrorCode
{
    configuration,
    geometry,
    checkpoint_io,
    checkpoint_format,
    lifecycle,
};

class StiffGIPCError : public std::runtime_error
{
  public:
    StiffGIPCError(ErrorCode code, const std::string& message)
        : std::runtime_error(message)
        , m_code(code)
    {
    }

    ErrorCode code() const noexcept { return m_code; }

  private:
    ErrorCode m_code;
};

class ConfigurationError : public StiffGIPCError
{
  public:
    explicit ConfigurationError(const std::string& message)
        : StiffGIPCError(ErrorCode::configuration, message)
    {
    }
};

class GeometryError : public StiffGIPCError
{
  public:
    explicit GeometryError(const std::string& message)
        : StiffGIPCError(ErrorCode::geometry, message)
    {
    }
};

class CheckpointError : public StiffGIPCError
{
  public:
    CheckpointError(ErrorCode code, const std::string& message)
        : StiffGIPCError(code, message)
    {
        if(code != ErrorCode::checkpoint_io
           && code != ErrorCode::checkpoint_format)
            throw std::invalid_argument(
                "CheckpointError requires a checkpoint error code");
    }
};

class LifecycleError : public StiffGIPCError
{
  public:
    explicit LifecycleError(const std::string& message)
        : StiffGIPCError(ErrorCode::lifecycle, message)
    {
    }
};
}  // namespace gipc
