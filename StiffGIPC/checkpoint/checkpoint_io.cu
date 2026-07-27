#include "GIPC.cuh"

#include "abd_system/abd_sim_data.h"
#include "abd_system/abd_system.h"
#include "cuda_tools/cuda_tools.h"
#include "errors.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{
using ByteBuffer = std::vector<std::uint8_t>;

constexpr char          kMagic[8]          = {'S', 'T', 'I', 'F', 'F', 'C', 'P', '2'};
constexpr std::uint32_t kVersion           = 2;
constexpr std::uint32_t kHeaderSize        = 120;
constexpr std::uint32_t kEndianMarker      = 0x01020304u;
constexpr std::uint32_t kScalarSize        = sizeof(double);
constexpr std::uint32_t kKnownStateFlags   = 0x1fu;
constexpr std::uint32_t kStateGroupKappa   = 1u << 0;
constexpr std::uint32_t kStateEnvActive    = 1u << 1;
constexpr std::uint32_t kStateQuarantined  = 1u << 2;
constexpr std::uint32_t kStateDirectionNaN = 1u << 3;
constexpr std::uint32_t kStateGroundSkip   = 1u << 4;
constexpr std::uint64_t kCrcPolynomial     = 0x42F0E1EBA9EA3693ULL;
#if defined(USE_SNK1)
constexpr std::uint32_t kConstitutiveModel = 1;
#elif defined(USE_SNK2)
constexpr std::uint32_t kConstitutiveModel = 2;
#elif defined(USE_ARAP)
constexpr std::uint32_t kConstitutiveModel = 3;
#else
#error "Checkpoint ABI requires one configured tetrahedral constitutive model"
#endif

std::string system_message(const char* operation, const std::string& path)
{
    return std::string("[checkpoint] ") + operation + " '" + path
         + "': " + std::strerror(errno);
}

[[noreturn]] void io_error(const char* operation, const std::string& path)
{
    throw gipc::CheckpointError(
        gipc::ErrorCode::checkpoint_io, system_message(operation, path));
}

[[noreturn]] void filesystem_error(const char*                  operation,
                                   const std::filesystem::path& path,
                                   const std::error_code&       error)
{
    throw gipc::CheckpointError(
        gipc::ErrorCode::checkpoint_io,
        std::string("[checkpoint] ") + operation + " '" + path.string()
            + "': " + error.message());
}

[[noreturn]] void format_error(const std::string& message)
{
    throw gipc::CheckpointError(
        gipc::ErrorCode::checkpoint_format, "[checkpoint] " + message);
}

void append_u32(ByteBuffer& out, std::uint32_t value)
{
    for(int shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(ByteBuffer& out, std::uint64_t value)
{
    for(int shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_i32(ByteBuffer& out, std::int32_t value)
{
    append_u32(out, static_cast<std::uint32_t>(value));
}

void append_i64(ByteBuffer& out, std::int64_t value)
{
    append_u64(out, static_cast<std::uint64_t>(value));
}

void append_double(ByteBuffer& out, double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(out, bits);
}

std::uint32_t read_u32(const ByteBuffer& input, std::size_t& cursor)
{
    if(cursor + 4 > input.size())
        format_error("truncated uint32 field");
    std::uint32_t value = 0;
    for(int shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(input[cursor++]) << shift;
    return value;
}

std::uint64_t read_u64(const ByteBuffer& input, std::size_t& cursor)
{
    if(cursor + 8 > input.size())
        format_error("truncated uint64 field");
    std::uint64_t value = 0;
    for(int shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(input[cursor++]) << shift;
    return value;
}

std::int32_t read_i32(const ByteBuffer& input, std::size_t& cursor)
{
    return static_cast<std::int32_t>(read_u32(input, cursor));
}

std::int64_t read_i64(const ByteBuffer& input, std::size_t& cursor)
{
    return static_cast<std::int64_t>(read_u64(input, cursor));
}

double read_double(const ByteBuffer& input, std::size_t& cursor)
{
    const std::uint64_t bits = read_u64(input, cursor);
    double              value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint64_t crc64(const void* data, std::size_t size, std::uint64_t crc = 0)
{
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for(std::size_t i = 0; i < size; ++i)
    {
        crc ^= static_cast<std::uint64_t>(bytes[i]) << 56;
        for(int bit = 0; bit < 8; ++bit)
            crc = (crc & (1ULL << 63)) ? (crc << 1) ^ kCrcPolynomial
                                      : (crc << 1);
    }
    return crc;
}

template <typename T>
void crc_value(std::uint64_t& crc, const T& value)
{
    crc = crc64(&value, sizeof(value), crc);
}

template <typename T>
std::vector<T> copy_device(const T* pointer, std::size_t count)
{
    std::vector<T> host(count);
    if(count > 0)
    {
        if(pointer == nullptr)
            format_error("scene contains a null device array");
        CUDA_SAFE_CALL(cudaMemcpy(host.data(),
                                  pointer,
                                  count * sizeof(T),
                                  cudaMemcpyDeviceToHost));
    }
    return host;
}

template <typename T>
void crc_device(std::uint64_t& crc, const T* pointer, std::size_t count)
{
    const auto host = copy_device(pointer, count);
    if(!host.empty())
        crc = crc64(host.data(), host.size() * sizeof(T), crc);
}

template <typename Vector>
void crc_vector3(std::uint64_t& crc, const Vector& value)
{
    for(int axis = 0; axis < 3; ++axis)
        crc_value(crc, value(axis));
}

void crc_joint_constraints(std::uint64_t&                 crc,
                           const gipc::JointConstraintGPUData* pointer,
                           std::size_t                     count)
{
    for(const auto& joint : copy_device(pointer, count))
    {
        crc_value(crc, joint.parent_body_id);
        crc_value(crc, joint.child_body_id);
        crc_value(crc, joint.num_points);
        for(int i = 0; i < gipc::kMaxJointConstraintPoints; ++i)
        {
            crc_vector3(crc, joint.parent_xbar[i]);
            crc_vector3(crc, joint.child_xbar[i]);
            crc_value(crc, joint.point_weight[i]);
        }
        crc_value(crc, joint.kappa);
        crc_value(crc, joint.has_direction_constraint);
        crc_vector3(crc, joint.parent_t_bar);
        crc_vector3(crc, joint.child_t_bar);
        crc_vector3(crc, joint.parent_n_bar);
        crc_vector3(crc, joint.child_n_bar);
        crc_vector3(crc, joint.parent_b_bar);
        crc_vector3(crc, joint.child_b_bar);
    }
}

void crc_revolute_driving(std::uint64_t&                  crc,
                          const gipc::RevoluteDrivingGPUData* pointer,
                          std::size_t                      count)
{
    for(const auto& joint : copy_device(pointer, count))
    {
        crc_value(crc, joint.parent_body_id);
        crc_value(crc, joint.child_body_id);
        crc_vector3(crc, joint.p_bar);
        crc_vector3(crc, joint.pN_bar);
        crc_vector3(crc, joint.q_bar);
        crc_vector3(crc, joint.qN_bar);
        crc_value(crc, joint.initial_angle_offset);
        crc_value(crc, joint.lower_limit);
        crc_value(crc, joint.upper_limit);
        crc_value(crc, joint.limit_stiffness);
    }
}

void crc_prismatic_constraints(std::uint64_t&               crc,
                               const gipc::PrismaticJointGPUData* pointer,
                               std::size_t                    count)
{
    for(const auto& joint : copy_device(pointer, count))
    {
        crc_value(crc, joint.parent_body_id);
        crc_value(crc, joint.child_body_id);
        crc_vector3(crc, joint.Cp_bar);
        crc_vector3(crc, joint.Cq_bar);
        crc_vector3(crc, joint.tp_bar);
        crc_vector3(crc, joint.tq_bar);
        crc_vector3(crc, joint.np_bar);
        crc_vector3(crc, joint.nq_bar);
        crc_vector3(crc, joint.bp_bar);
        crc_vector3(crc, joint.bq_bar);
    }
}

void crc_prismatic_driving(std::uint64_t&                 crc,
                           const gipc::PrismaticDrivingGPUData* pointer,
                           std::size_t                      count)
{
    for(const auto& joint : copy_device(pointer, count))
    {
        crc_value(crc, joint.parent_body_id);
        crc_value(crc, joint.child_body_id);
        crc_vector3(crc, joint.Cp_bar);
        crc_vector3(crc, joint.Cq_bar);
        crc_vector3(crc, joint.tq_bar);
        crc_vector3(crc, joint.tp_bar);
        crc_value(crc, joint.limit_cl);
        crc_value(crc, joint.limit_dir);
        crc_value(crc, joint.limit_dhat);
        crc_value(crc, joint.limit_kappa);
        crc_value(crc, joint.limit_cl2);
        crc_value(crc, joint.limit_dir2);
        crc_value(crc, joint.limit_dhat2);
        crc_value(crc, joint.limit_kappa2);
        crc_value(crc, joint.lower_limit);
        crc_value(crc, joint.upper_limit);
        crc_value(crc, joint.pen_limit_stiffness);
    }
}

std::uint32_t mode_flags(const ModeConfig& mode)
{
    std::uint32_t flags = 0;
    flags |= static_cast<std::uint32_t>(mode.bvh_envdet) << 0;
    flags |= static_cast<std::uint32_t>(mode.perenv_bvh) << 1;
    flags |= static_cast<std::uint32_t>(mode.decouple_thresh) << 2;
    flags |= static_cast<std::uint32_t>(mode.pergroup_kappa) << 3;
    flags |= static_cast<std::uint32_t>(mode.segmented_pcg) << 4;
    flags |= static_cast<std::uint32_t>(mode.perenv_alpha) << 5;
    flags |= static_cast<std::uint32_t>(mode.perenv_par) << 6;
    flags |= static_cast<std::uint32_t>(mode.ee_canon) << 7;
    flags |= static_cast<std::uint32_t>(mode.ee_detgate) << 8;
    flags |= static_cast<std::uint32_t>(mode.ccd_canon) << 9;
    flags |= static_cast<std::uint32_t>(mode.spmv_det) << 10;
    flags |= static_cast<std::uint32_t>(mode.perenv_telem) << 11;
    flags |= static_cast<std::uint32_t>(mode.perenv_mask) << 12;
    flags |= static_cast<std::uint32_t>(mode.perenv_mask_dev) << 13;
    return flags;
}

std::uint64_t scene_signature(const GIPC& engine, const device_TetraData& mesh)
{
    std::uint64_t crc = 0;
    const auto    vN  = static_cast<std::size_t>(engine.vertexNum);
    const auto    tN = static_cast<std::size_t>(engine.tetrahedraNum);
    const auto triN  = static_cast<std::size_t>(engine.triangleNum);
    const auto softN = static_cast<std::size_t>(engine.softNum);

    crc_value(crc, engine.vertexNum);
    crc_value(crc, engine.abd_fem_count_info.abd_body_offset);
    crc_value(crc, engine.abd_fem_count_info.abd_body_num);
    crc_value(crc, engine.abd_fem_count_info.fem_body_offset);
    crc_value(crc, engine.abd_fem_count_info.fem_body_num);
    crc_value(crc, engine.abd_fem_count_info.abd_tet_offset);
    crc_value(crc, engine.abd_fem_count_info.abd_tet_num);
    crc_value(crc, engine.abd_fem_count_info.fem_tet_offset);
    crc_value(crc, engine.abd_fem_count_info.fem_point_num);
    crc_value(crc, engine.abd_fem_count_info.fem_tet_num);
    crc_value(crc, engine.abd_fem_count_info.fem_tri_offset);
    crc_value(crc, engine.abd_fem_count_info.fem_tri_num);
    crc_value(crc, engine.abd_fem_count_info.abd_point_offset);
    crc_value(crc, engine.abd_fem_count_info.abd_point_num);
    crc_value(crc, engine.abd_fem_count_info.fem_point_offset);
    crc_value(crc, engine.tetrahedraNum);
    crc_value(crc, engine.triangleNum);
    crc_value(crc, engine.tri_edge_num);
    crc_value(crc, engine.softNum);
    crc_value(crc, mesh.collision_body_num);
    const auto flags = mode_flags(engine.m_mode_config);
    crc_value(crc, flags);
    crc_value(crc, kConstitutiveModel);
    crc_value(crc, engine.Newton_solver_threshold);
    crc_value(crc, engine.newton_velocity_tol);
    crc_value(crc, engine.pcg_threshold);
    crc_value(crc, engine.newton_iter_cap);
    crc_value(crc, engine.env_newton_iter_cap);
    crc_value(crc, engine.line_search_max_iter);
    crc_value(crc, engine.energy_abs_tol);
    crc_value(crc, engine.energy_rel_tol);
    crc_value(crc, engine.semi_implicit_enabled);
    crc_value(crc, engine.semi_implicit_beta_tol);
    crc_value(crc, engine.semi_implicit_min_iter);
    crc_value(crc, engine.m_skip_all_collision);
    crc_value(crc, engine.pcg_data.P_type);

    const double config_values[] = {
        engine.IPC_dt,
        engine.gravity.x,
        engine.gravity.y,
        engine.gravity.z,
        engine.relative_dhat,
        engine.absolute_dhat,
        engine.ground_normal_cfg.x,
        engine.ground_normal_cfg.y,
        engine.ground_normal_cfg.z,
        engine.ground_offset_cfg,
        engine.density,
        engine.YoungModulus,
        engine.PoissonRate,
        engine.frictionRate,
        engine.gd_frictionRate,
        engine.clothThickness,
        engine.clothYoungModulus,
        engine.bendYoungModulus,
        engine.stretchStiff,
        engine.shearStiff,
        engine.strainRate,
        engine.clothDensity,
        engine.softMotionRate,
    };
    crc = crc64(config_values, sizeof(config_values), crc);

    crc_device(crc, mesh.rest_vertexes, vN);
    crc_device(crc, mesh.tetrahedras, tN);
    crc_device(crc, mesh.triangles, triN);
    crc_device(crc, mesh.tri_edges, engine.tri_edge_num);
    crc_device(crc, mesh.tri_edge_adj_vertex, engine.tri_edge_num);
    crc_device(crc, engine._faces.data(), engine.surface_Num);
    crc_device(crc, engine._edges.data(), engine.edge_Num);
    crc_device(crc, engine._surfVerts.data(), engine.surf_vertexNum);
    crc_device(crc, mesh.BoundaryType, vN);
    crc_device(crc, mesh.point_id_to_body_id, vN);
    crc_device(crc, mesh.tet_id_to_body_id, tN);
    crc_device(crc, mesh.apply_gravity, vN);
    crc_device(crc, mesh.masses, vN);
    crc_device(crc, mesh.lengthRate, tN);
    crc_device(crc, mesh.volumeRate, tN);
    crc_device(crc, mesh.area, triN);
    crc_device(crc, mesh.targetIndex, softN);
    const auto bodyN =
        static_cast<std::size_t>(std::max(mesh.collision_body_num, 0));
    crc_device(crc, mesh.collision_skip_matrix, bodyN * bodyN);
    crc_device(crc, mesh.d_body_to_group, bodyN);
    crc_device(crc, mesh.body_id_to_is_fem, bodyN);
    crc_device(crc, mesh.body_id_to_boundary_type, bodyN);
    crc_device(crc, mesh.body_motor_params, 5 * bodyN);
    crc_value(crc, mesh.n_fem_pins);
    const auto pinN = static_cast<std::size_t>(std::max(mesh.n_fem_pins, 0));
    crc_device(crc, mesh.d_fem_pin_fem_vertex, pinN);
    crc_device(crc, mesh.d_fem_pin_abd_body_id, pinN);
    crc_device(crc, mesh.d_fem_pin_abd_local_pos, pinN);
    const bool has_body_friction =
        engine.d_vert_mu != nullptr || engine.d_vert_mu_gd != nullptr;
    crc_value(crc, has_body_friction);
    if(has_body_friction)
    {
        crc_device(crc, engine.d_vert_mu, vN);
        crc_device(crc, engine.d_vert_mu_gd, vN);
    }
    if(engine.m_abd_system)
    {
        const auto& parms = engine.m_abd_system->parms;
        const double abd_config[] = {
            parms.dt,
            parms.mass_density,
            parms.kappa,
            parms.motor_speed,
            parms.motor_strength,
            parms.joint_strength_ratio,
            parms.revolute_driving_strength_ratio,
            parms.joint_limit_strength_ratio,
            parms.prismatic_strength_ratio,
            parms.prismatic_driving_strength_ratio,
            parms.max_revolute_step_per_frame,
            parms.max_prismatic_step_per_frame,
            parms.velocity_damping,
            parms.gravity[0],
            parms.gravity[1],
            parms.gravity[2],
        };
        crc = crc64(abd_config, sizeof(abd_config), crc);
        crc_value(crc, engine.m_abd_system->m_num_joints);
        crc_value(crc, engine.m_abd_system->m_num_revolute_driving);
        crc_value(crc, engine.m_abd_system->m_num_prismatic);
        crc_value(crc, engine.m_abd_system->m_num_prismatic_driving);
        crc_joint_constraints(
            crc,
            engine.m_abd_system->m_joint_data.data(),
            engine.m_abd_system->m_num_joints);
        crc_revolute_driving(
            crc,
            engine.m_abd_system->m_revolute_driving_data.data(),
            engine.m_abd_system->m_num_revolute_driving);
        crc_prismatic_constraints(
            crc,
            engine.m_abd_system->m_prismatic_data.data(),
            engine.m_abd_system->m_num_prismatic);
        crc_prismatic_driving(
            crc,
            engine.m_abd_system->m_prismatic_driving_data.data(),
            engine.m_abd_system->m_num_prismatic_driving);

        auto crc_sorted_scalar_overrides =
            [&](const std::unordered_map<int, double>& overrides)
        {
            std::vector<int> ids;
            ids.reserve(overrides.size());
            for(const auto& [id, value] : overrides)
            {
                (void)value;
                ids.push_back(id);
            }
            std::sort(ids.begin(), ids.end());
            crc_value(crc, ids.size());
            for(int id : ids)
            {
                crc_value(crc, id);
                crc_value(crc, overrides.at(id));
            }
        };
        crc_sorted_scalar_overrides(
            engine.m_abd_system->m_body_density_override);
        crc_sorted_scalar_overrides(
            engine.m_abd_system->m_body_mass_override);

        std::vector<int> inertia_ids;
        inertia_ids.reserve(engine.m_abd_system->m_body_inertia_override.size());
        for(const auto& [id, value] :
            engine.m_abd_system->m_body_inertia_override)
        {
            (void)value;
            inertia_ids.push_back(id);
        }
        std::sort(inertia_ids.begin(), inertia_ids.end());
        crc_value(crc, inertia_ids.size());
        for(int id : inertia_ids)
        {
            const auto& value =
                engine.m_abd_system->m_body_inertia_override.at(id);
            crc_value(crc, id);
            crc_value(crc, value.mass);
            for(int axis = 0; axis < 3; ++axis)
                crc_value(crc, value.com(axis));
            for(int row = 0; row < 3; ++row)
                for(int col = 0; col < 3; ++col)
                    crc_value(crc, value.inertia(row, col));
        }
    }
    if(softN > 0)
    {
        crc_device(crc, mesh.d_stitch_paired_vertex, softN);
        crc_device(crc, mesh.d_stitch_abd_body_id, softN);
        crc_device(crc, mesh.d_stitch_rest_offset, softN);
    }
    return crc;
}

void append_double3s(ByteBuffer& out, const std::vector<double3>& values)
{
    for(const double3& value : values)
    {
        append_double(out, value.x);
        append_double(out, value.y);
        append_double(out, value.z);
    }
}

std::vector<double3>
read_double3s(const ByteBuffer& input, std::size_t& cursor, std::size_t count)
{
    std::vector<double3> values(count);
    for(double3& value : values)
    {
        value.x = read_double(input, cursor);
        value.y = read_double(input, cursor);
        value.z = read_double(input, cursor);
    }
    return values;
}

void require_finite(const std::vector<double3>& values, const char* label)
{
    for(const double3& value : values)
        if(!std::isfinite(value.x) || !std::isfinite(value.y)
           || !std::isfinite(value.z))
            format_error(std::string(label) + " contains a non-finite value");
}

void require_finite(const std::vector<double>& values, const char* label)
{
    for(double value : values)
        if(!std::isfinite(value))
            format_error(std::string(label) + " contains a non-finite value");
}

void require_binary(const std::vector<int>& values, const char* label)
{
    for(int value : values)
        if(value != 0 && value != 1)
            format_error(std::string(label) + " contains a non-binary value");
}

void write_all(int fd, const ByteBuffer& data, const std::string& path)
{
    std::size_t offset = 0;
    while(offset < data.size())
    {
        const ssize_t wrote =
            ::write(fd, data.data() + offset, data.size() - offset);
        if(wrote < 0)
        {
            if(errno == EINTR)
                continue;
            io_error("write failed for", path);
        }
        if(wrote == 0)
            format_error("zero-byte write for '" + path + "'");
        offset += static_cast<std::size_t>(wrote);
    }
}

void atomic_write(const std::string& destination, const ByteBuffer& data)
{
    if(destination.empty())
        format_error("empty checkpoint path");
    const std::filesystem::path final_path(destination);
    const std::filesystem::path parent =
        final_path.has_parent_path() ? final_path.parent_path() : ".";
    std::error_code status_error;
    const bool parent_is_directory =
        std::filesystem::is_directory(parent, status_error);
    if(status_error)
        filesystem_error("cannot inspect parent directory", parent, status_error);
    if(!parent_is_directory)
        format_error("checkpoint parent is not a directory: '"
                     + parent.string() + "'");

    static std::atomic<unsigned long long> sequence{0};
    std::string temp;
    int         fd = -1;
    for(int attempt = 0; attempt < 16 && fd < 0; ++attempt)
    {
        temp = destination + ".tmp." + std::to_string(::getpid()) + "."
             + std::to_string(sequence.fetch_add(1));
        fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if(fd < 0 && errno != EEXIST)
            io_error("cannot create temporary file for", destination);
    }
    if(fd < 0)
        io_error("cannot reserve temporary file for", destination);

    bool renamed = false;
    try
    {
        write_all(fd, data, temp);
        if(::fsync(fd) != 0)
            io_error("fsync failed for", temp);
        if(::close(fd) != 0)
        {
            fd = -1;
            io_error("close failed for", temp);
        }
        fd = -1;
        if(::rename(temp.c_str(), destination.c_str()) != 0)
            io_error("rename failed for", destination);
        renamed = true;

        const int dir_fd =
            ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if(dir_fd < 0)
            io_error("cannot open parent directory for", destination);
        const int sync_rc  = ::fsync(dir_fd);
        const int close_rc = ::close(dir_fd);
        if(sync_rc != 0 || close_rc != 0)
            io_error("directory fsync failed for", destination);
    }
    catch(...)
    {
        if(fd >= 0)
            ::close(fd);
        if(!renamed)
            ::unlink(temp.c_str());
        throw;
    }
}

ByteBuffer read_exact_file(const std::string& path, std::size_t expected_size)
{
    errno = 0;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if(!input)
    {
        errno = errno ? errno : ENOENT;
        io_error("cannot open for reading", path);
    }
    const std::streamoff end = input.tellg();
    if(end < 0)
        format_error("cannot determine size of '" + path + "'");

    if(static_cast<std::uint64_t>(end) != expected_size)
    {
        input.seekg(0);
        std::uint32_t legacy = 0;
        input.read(reinterpret_cast<char*>(&legacy), sizeof(legacy));
        if(input.gcount() == static_cast<std::streamsize>(sizeof(legacy))
           && legacy == 0x53544B50u)
            format_error(
                "legacy unversioned checkpoint rejected; regenerate it with "
                "the current engine");
        std::ostringstream message;
        message << "file-size mismatch for '" << path << "': got " << end
                << " bytes, expected " << expected_size;
        format_error(message.str());
    }

    ByteBuffer data(expected_size);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if(!input || input.gcount() != static_cast<std::streamsize>(data.size()))
        format_error("short read from '" + path + "'");
    return data;
}

std::uint64_t payload_size(std::uint64_t vertices,
                           std::uint64_t abd_bodies,
                           std::uint64_t soft_constraints,
                           std::uint64_t env_count,
                           std::uint64_t collision_bodies,
                           std::uint32_t state_flags)
{
    constexpr std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max() / sizeof(double);
    const std::uint64_t double_count =
        12 * vertices + 60 * abd_bodies + 3 * soft_constraints + 1
        + ((state_flags & kStateGroupKappa) ? env_count : 0);
    if(double_count > limit)
        format_error("payload size overflow");
    std::uint64_t bytes = double_count * sizeof(double);
    const std::uint64_t int_count =
        ((state_flags & kStateEnvActive) ? env_count : 0)
        + ((state_flags & kStateQuarantined) ? env_count : 0)
        + ((state_flags & kStateDirectionNaN) ? env_count : 0)
        + ((state_flags & kStateGroundSkip) ? collision_bodies : 0);
    if(int_count > (std::numeric_limits<std::uint64_t>::max() - bytes - 16) / 4)
        format_error("payload size overflow");
    return bytes + 4 * int_count + 16;  // recheck counter + frame index
}

void ensure_group_kappa(GIPC& engine,
                        const device_TetraData& mesh,
                        std::size_t env_count)
{
    if(env_count == 0 || engine.m_kappa_group)
        return;
    if(!mesh.d_point_to_group || !mesh.h_groups_present
       || static_cast<std::size_t>(mesh.h_group_count) != env_count)
        format_error("cannot restore per-group kappa into this scene");
    engine.m_pergroup_kappa    = true;
    engine.m_d_p2g             = mesh.d_point_to_group;
    engine.m_active_group_count = static_cast<int>(env_count);
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&engine.m_kappa_group, env_count * sizeof(double)));
    CUDA_SAFE_CALL(
        cudaMalloc((void**)&engine.m_d_close_grp, env_count * sizeof(int)));
    CUDA_SAFE_CALL(
        cudaMemset(engine.m_d_close_grp, 0, env_count * sizeof(int)));
}
}  // namespace

void GIPC::save_checkpoint(device_TetraData& mesh, const char* raw_path)
{
    const std::string path = raw_path ? raw_path : "";
    if(path.empty())
        format_error("empty checkpoint path");
    const std::size_t vN   = static_cast<std::size_t>(vertexNum);
    const std::size_t nb =
        static_cast<std::size_t>(abd_fem_count_info.abd_body_num);
    const std::size_t softN = static_cast<std::size_t>(softNum);

    std::uint32_t state_flags = 0;
    const std::size_t env_count =
        mesh.h_groups_present ? static_cast<std::size_t>(mesh.h_group_count) : 0;
    const std::size_t collision_bodies =
        static_cast<std::size_t>(std::max(mesh.collision_body_num, 0));
    if(env_count > 0 && m_pergroup_kappa && m_kappa_group)
        state_flags |= kStateGroupKappa;
    if(env_count > 0 && m_env_active.capacity() >= env_count)
        state_flags |= kStateEnvActive;
    if(env_count > 0 && m_d_env_quarantined.capacity() >= env_count)
        state_flags |= kStateQuarantined;
    if(env_count > 0 && m_d_env_dirnan.capacity() >= env_count)
        state_flags |= kStateDirectionNaN;
    if(collision_bodies > 0)
        state_flags |= kStateGroundSkip;

    const auto current = copy_device(mesh.vertexes, vN);
    const auto previous = copy_device(mesh.o_vertexes, vN);
    const auto velocity = copy_device(mesh.velocities, vN);
    const auto predictor = copy_device(mesh.xTilta, vN);
    const auto targets = copy_device(mesh.targetVert, softN);
    require_finite(current, "vertex positions");
    require_finite(previous, "previous vertex positions");
    require_finite(velocity, "vertex velocities");
    require_finite(predictor, "vertex predictors");
    require_finite(targets, "soft targets");

    std::vector<double> q(12 * nb), q_prev(12 * nb), q_velocity(12 * nb),
        q_tilde(12 * nb), q_external_force(12 * nb);
    if(nb > 0)
    {
        if(!m_abd_sim_data)
            format_error("ABD state is unavailable");
        auto& device = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(q.data(),
                                  reinterpret_cast<const double*>(
                                      device.body_id_to_q.data()),
                                  q.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(q_prev.data(),
                                  reinterpret_cast<const double*>(
                                      device.body_id_to_q_prev.data()),
                                  q_prev.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(q_velocity.data(),
                                  reinterpret_cast<const double*>(
                                      device.body_id_to_q_v.data()),
                                  q_velocity.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(q_tilde.data(),
                                  reinterpret_cast<const double*>(
                                      device.body_id_to_q_tilde.data()),
                                  q_tilde.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(q_external_force.data(),
                                  reinterpret_cast<const double*>(
                                      device.body_id_to_abd_ext_force.data()),
                                  q_external_force.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    }
    require_finite(q, "ABD q");
    require_finite(q_prev, "ABD q_prev");
    require_finite(q_velocity, "ABD q_velocity");
    require_finite(q_tilde, "ABD q_tilde");
    require_finite(q_external_force, "ABD external force");
    if(!std::isfinite(Kappa) || Kappa < 0.0)
        format_error("Kappa is invalid");

    ByteBuffer payload;
    payload.reserve(static_cast<std::size_t>(
        payload_size(vN, nb, softN, env_count, collision_bodies, state_flags)));
    append_double3s(payload, current);
    append_double3s(payload, previous);
    append_double3s(payload, velocity);
    append_double3s(payload, predictor);
    append_double3s(payload, targets);
    for(double value : q)
        append_double(payload, value);
    for(double value : q_prev)
        append_double(payload, value);
    for(double value : q_velocity)
        append_double(payload, value);
    for(double value : q_tilde)
        append_double(payload, value);
    for(double value : q_external_force)
        append_double(payload, value);
    append_double(payload, Kappa);

    if(state_flags & kStateGroupKappa)
    {
        const auto values = copy_device(m_kappa_group, env_count);
        require_finite(values, "per-group kappa");
        for(double value : values)
            append_double(payload, value);
    }
    if(state_flags & kStateEnvActive)
        for(int value : copy_device(m_env_active.data(), env_count))
            append_i32(payload, value);
    if(state_flags & kStateQuarantined)
        for(int value : copy_device(m_d_env_quarantined.data(), env_count))
            append_i32(payload, value);
    if(state_flags & kStateDirectionNaN)
        for(int value : copy_device(m_d_env_dirnan.data(), env_count))
            append_i32(payload, value);
    if(state_flags & kStateGroundSkip)
    {
        std::vector<int> ground_skip(collision_bodies, 0);
        if(_ground_skip_body)
        {
            if(_ground_body_count != static_cast<int>(collision_bodies))
                format_error("ground-skip table size differs from the scene");
            ground_skip =
                copy_device(_ground_skip_body, collision_bodies);
        }
        require_binary(ground_skip, "ground-skip table");
        for(int value : ground_skip)
            append_i32(payload, value);
    }
    append_i64(payload, m_recheck_counter);
    append_i64(payload, m_total_frames);

    const auto expected_payload =
        payload_size(vN, nb, softN, env_count, collision_bodies, state_flags);
    if(payload.size() != expected_payload)
        format_error("internal payload-size mismatch");

    ByteBuffer header;
    header.insert(header.end(), std::begin(kMagic), std::end(kMagic));
    append_u32(header, kVersion);
    append_u32(header, kHeaderSize);
    append_u32(header, kEndianMarker);
    append_u32(header, kScalarSize);
    append_u32(header, static_cast<std::uint32_t>(m_mode_config.mode));
    append_u32(header, mode_flags(m_mode_config));
    append_u32(header, state_flags);
    append_u32(header, kConstitutiveModel);
    append_u64(header, vN);
    append_u64(header, nb);
    append_u64(header, env_count);
    append_u64(header, tetrahedraNum);
    append_u64(header, triangleNum);
    append_u64(header, tri_edge_num);
    append_u64(header, softN);
    append_u64(header, scene_signature(*this, mesh));
    append_u64(header, payload.size());
    append_u64(header, crc64(payload.data(), payload.size()));
    if(header.size() != kHeaderSize)
        format_error("internal header-size mismatch");

    header.insert(header.end(), payload.begin(), payload.end());
    atomic_write(path, header);
    std::printf(
        "[ckpt] saved v2 %s (vN=%zu nb=%zu env=%zu Kappa=%.6e frame=%d)\n",
        path.c_str(),
        vN,
        nb,
        env_count,
        Kappa,
        m_total_frames);
}

void GIPC::load_checkpoint(device_TetraData& mesh, const char* raw_path)
{
    const std::string path = raw_path ? raw_path : "";
    if(path.empty())
        format_error("empty checkpoint path");
    const std::uint64_t local_vN   = vertexNum;
    const std::uint64_t local_nb   = abd_fem_count_info.abd_body_num;
    const std::uint64_t local_soft = softNum;
    const std::uint64_t local_collision_bodies =
        static_cast<std::uint64_t>(std::max(mesh.collision_body_num, 0));

    // Read the fixed header first to obtain flags while still bounding the
    // eventual allocation by the current scene's known counts.
    errno = 0;
    std::ifstream prefix(path, std::ios::binary);
    if(!prefix)
    {
        errno = errno ? errno : ENOENT;
        io_error("cannot open for reading", path);
    }
    ByteBuffer header(kHeaderSize);
    prefix.read(reinterpret_cast<char*>(header.data()), header.size());
    const auto prefix_read = prefix.gcount();
    if(prefix_read < 4)
        format_error("checkpoint is truncated before its header");
    if(std::memcmp(header.data(), kMagic, sizeof(kMagic)) != 0)
    {
        std::uint32_t legacy = 0;
        std::memcpy(&legacy, header.data(), sizeof(legacy));
        if(legacy == 0x53544B50u)
            format_error(
                "legacy unversioned checkpoint rejected; regenerate it with "
                "the current engine");
        format_error("bad checkpoint magic");
    }
    if(prefix_read != static_cast<std::streamsize>(header.size()))
        format_error("checkpoint header is truncated");

    std::size_t cursor = sizeof(kMagic);
    const auto version = read_u32(header, cursor);
    const auto header_size = read_u32(header, cursor);
    const auto endian = read_u32(header, cursor);
    const auto scalar_size = read_u32(header, cursor);
    const auto saved_mode = read_u32(header, cursor);
    const auto saved_mode_flags = read_u32(header, cursor);
    const auto state_flags = read_u32(header, cursor);
    const auto saved_constitutive_model = read_u32(header, cursor);
    const auto saved_vN = read_u64(header, cursor);
    const auto saved_nb = read_u64(header, cursor);
    const auto env_count = read_u64(header, cursor);
    const auto saved_tets = read_u64(header, cursor);
    const auto saved_triangles = read_u64(header, cursor);
    const auto saved_bending = read_u64(header, cursor);
    const auto saved_soft = read_u64(header, cursor);
    const auto saved_scene_signature = read_u64(header, cursor);
    const auto saved_payload_size = read_u64(header, cursor);
    const auto saved_checksum = read_u64(header, cursor);

    if(version != kVersion || header_size != kHeaderSize)
        format_error("unsupported checkpoint version/header");
    if(endian != kEndianMarker || scalar_size != kScalarSize)
        format_error("checkpoint scalar/endian ABI is incompatible");
    if(state_flags & ~kKnownStateFlags)
        format_error("checkpoint uses unknown state flags");
    if(local_collision_bodies > 0 && !(state_flags & kStateGroundSkip))
        format_error("checkpoint is missing the ground-skip state");
    if(saved_constitutive_model != kConstitutiveModel)
        format_error("tetrahedral constitutive model differs from checkpoint");
    if(saved_mode != static_cast<std::uint32_t>(m_mode_config.mode)
       || saved_mode_flags != mode_flags(m_mode_config))
        format_error("execution-mode configuration differs from checkpoint");
    if(saved_vN != local_vN || saved_nb != local_nb
       || saved_tets != tetrahedraNum
       || saved_triangles != triangleNum || saved_bending != tri_edge_num
       || saved_soft != local_soft)
        format_error("scene topology counts differ from checkpoint");
    if(env_count > 0
       && (!mesh.h_groups_present
           || env_count != static_cast<std::uint64_t>(mesh.h_group_count)))
        format_error("environment layout differs from checkpoint");

    const auto expected_payload =
        payload_size(local_vN,
                     local_nb,
                     local_soft,
                     env_count,
                     local_collision_bodies,
                     state_flags);
    if(saved_payload_size != expected_payload)
        format_error("declared payload size is inconsistent with the header");
    const auto expected_file = static_cast<std::size_t>(
        static_cast<std::uint64_t>(kHeaderSize) + expected_payload);
    ByteBuffer file = read_exact_file(path, expected_file);
    ByteBuffer payload(file.begin() + kHeaderSize, file.end());
    if(crc64(payload.data(), payload.size()) != saved_checksum)
        format_error("payload checksum mismatch");
    if(scene_signature(*this, mesh) != saved_scene_signature)
        format_error("scene/material signature differs from checkpoint");

    cursor = 0;
    auto current = read_double3s(payload, cursor, local_vN);
    auto previous = read_double3s(payload, cursor, local_vN);
    auto velocity = read_double3s(payload, cursor, local_vN);
    auto predictor = read_double3s(payload, cursor, local_vN);
    auto targets = read_double3s(payload, cursor, local_soft);
    std::vector<double> q(12 * local_nb), q_prev(12 * local_nb),
        q_velocity(12 * local_nb), q_tilde(12 * local_nb),
        q_external_force(12 * local_nb);
    for(double& value : q)
        value = read_double(payload, cursor);
    for(double& value : q_prev)
        value = read_double(payload, cursor);
    for(double& value : q_velocity)
        value = read_double(payload, cursor);
    for(double& value : q_tilde)
        value = read_double(payload, cursor);
    for(double& value : q_external_force)
        value = read_double(payload, cursor);
    const double saved_kappa = read_double(payload, cursor);
    std::vector<double> group_kappa;
    if(state_flags & kStateGroupKappa)
    {
        group_kappa.resize(env_count);
        for(double& value : group_kappa)
            value = read_double(payload, cursor);
    }
    std::vector<int> env_active, quarantined, direction_nan, ground_skip;
    auto read_ints = [&](std::vector<int>& values)
    {
        values.resize(env_count);
        for(int& value : values)
            value = read_i32(payload, cursor);
    };
    if(state_flags & kStateEnvActive)
        read_ints(env_active);
    if(state_flags & kStateQuarantined)
        read_ints(quarantined);
    if(state_flags & kStateDirectionNaN)
        read_ints(direction_nan);
    if(state_flags & kStateGroundSkip)
    {
        ground_skip.resize(local_collision_bodies);
        for(int& value : ground_skip)
            value = read_i32(payload, cursor);
    }
    const auto recheck = read_i64(payload, cursor);
    const auto frame = read_i64(payload, cursor);
    if(cursor != payload.size())
        format_error("payload has trailing or unparsed bytes");

    require_finite(current, "vertex positions");
    require_finite(previous, "previous vertex positions");
    require_finite(velocity, "vertex velocities");
    require_finite(predictor, "vertex predictors");
    require_finite(targets, "soft targets");
    require_finite(q, "ABD q");
    require_finite(q_prev, "ABD q_prev");
    require_finite(q_velocity, "ABD q_velocity");
    require_finite(q_tilde, "ABD q_tilde");
    require_finite(q_external_force, "ABD external force");
    require_finite(group_kappa, "per-group kappa");
    require_binary(env_active, "per-env active state");
    require_binary(quarantined, "per-env quarantine state");
    require_binary(direction_nan, "per-env direction-NaN state");
    require_binary(ground_skip, "ground-skip table");
    for(double value : group_kappa)
        if(value < 0.0)
            format_error("per-group kappa contains a negative value");
    if(!std::isfinite(saved_kappa) || saved_kappa < 0.0)
        format_error("saved Kappa is invalid");
    if(frame < 0 || frame > std::numeric_limits<int>::max()
       || recheck < std::numeric_limits<int>::min()
       || recheck > std::numeric_limits<int>::max())
        format_error("saved frame/recheck counter is out of range");
    if((state_flags & kStateGroundSkip) && _ground_skip_body
       && _ground_body_count != static_cast<int>(local_collision_bodies))
        format_error("live ground-skip table size differs from checkpoint");

    // All format, checksum, topology, and finite-value checks complete before
    // the first live device byte is mutated.
    CUDA_SAFE_CALL(cudaMemcpy(mesh.vertexes,
                              current.data(),
                              current.size() * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mesh.o_vertexes,
                              previous.data(),
                              previous.size() * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mesh.velocities,
                              velocity.data(),
                              velocity.size() * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mesh.xTilta,
                              predictor.data(),
                              predictor.size() * sizeof(double3),
                              cudaMemcpyHostToDevice));
    if(!targets.empty())
        CUDA_SAFE_CALL(cudaMemcpy(mesh.targetVert,
                                  targets.data(),
                                  targets.size() * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    if(local_nb > 0)
    {
        if(!m_abd_sim_data)
            format_error("ABD state is unavailable");
        auto& device = m_abd_sim_data->device;
        CUDA_SAFE_CALL(cudaMemcpy(
            reinterpret_cast<double*>(device.body_id_to_q.data()),
            q.data(),
            q.size() * sizeof(double),
            cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(
            reinterpret_cast<double*>(device.body_id_to_q_prev.data()),
            q_prev.data(),
            q_prev.size() * sizeof(double),
            cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(
            reinterpret_cast<double*>(device.body_id_to_q_v.data()),
            q_velocity.data(),
            q_velocity.size() * sizeof(double),
            cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(
            reinterpret_cast<double*>(device.body_id_to_q_tilde.data()),
            q_tilde.data(),
            q_tilde.size() * sizeof(double),
            cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(
            reinterpret_cast<double*>(device.body_id_to_abd_ext_force.data()),
            q_external_force.data(),
            q_external_force.size() * sizeof(double),
            cudaMemcpyHostToDevice));
    }
    if(state_flags & kStateGroupKappa)
    {
        ensure_group_kappa(*this, mesh, env_count);
        CUDA_SAFE_CALL(cudaMemcpy(m_kappa_group,
                                  group_kappa.data(),
                                  group_kappa.size() * sizeof(double),
                                  cudaMemcpyHostToDevice));
        h_kappa_group = group_kappa;
    }
    if(state_flags & kStateEnvActive)
    {
        if(m_env_active.capacity() < env_count)
        {
            m_env_active.resize_discard(kEnvAlphaSlots);
            CUDA_SAFE_CALL(
                cudaMemset(m_env_active.data(), 0, kEnvAlphaSlots * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(m_env_active.data(),
                                  env_active.data(),
                                  env_active.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        h_env_active.assign(kEnvAlphaSlots, 1);
        std::copy(env_active.begin(), env_active.end(), h_env_active.begin());
    }
    if(state_flags & kStateQuarantined)
    {
        if(m_d_env_quarantined.capacity() < env_count)
        {
            m_d_env_quarantined.resize_discard(kEnvAlphaSlots);
            CUDA_SAFE_CALL(cudaMemset(m_d_env_quarantined.data(),
                                      0,
                                      kEnvAlphaSlots * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(m_d_env_quarantined.data(),
                                  quarantined.data(),
                                  quarantined.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        m_env_quarantined.assign(kEnvAlphaSlots, 0);
        std::copy(quarantined.begin(),
                  quarantined.end(),
                  m_env_quarantined.begin());
    }
    else
    {
        m_env_quarantined.clear();
        if(m_d_env_quarantined)
            CUDA_SAFE_CALL(cudaMemset(m_d_env_quarantined.data(),
                                      0,
                                      m_d_env_quarantined.capacity() * sizeof(int)));
    }
    if(state_flags & kStateDirectionNaN)
    {
        if(m_d_env_dirnan.capacity() < env_count)
        {
            m_d_env_dirnan.resize_discard(kEnvAlphaSlots);
            CUDA_SAFE_CALL(cudaMemset(m_d_env_dirnan.data(),
                                      0,
                                      kEnvAlphaSlots * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(m_d_env_dirnan.data(),
                                  direction_nan.data(),
                                  direction_nan.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
    }
    else if(m_d_env_dirnan)
    {
        CUDA_SAFE_CALL(cudaMemset(m_d_env_dirnan.data(),
                                  0,
                                  m_d_env_dirnan.capacity() * sizeof(int)));
    }
    if(state_flags & kStateGroundSkip)
    {
        if(!_ground_skip_body && local_collision_bodies > 0)
        {
            CUDA_SAFE_CALL(cudaMalloc(
                (void**)&_ground_skip_body,
                local_collision_bodies * sizeof(int)));
            m_ground_skip_owned = true;
        }
        if(local_collision_bodies > 0)
            CUDA_SAFE_CALL(cudaMemcpy(
                _ground_skip_body,
                ground_skip.data(),
                ground_skip.size() * sizeof(int),
                cudaMemcpyHostToDevice));
        _ground_body_count = static_cast<int>(local_collision_bodies);
    }
    Kappa             = saved_kappa;
    m_recheck_counter = static_cast<int>(recheck);
    m_total_frames    = static_cast<int>(frame);
    std::printf(
        "[ckpt] loaded v2 %s (vN=%llu nb=%llu env=%llu Kappa=%.6e frame=%d)\n",
        path.c_str(),
        static_cast<unsigned long long>(local_vN),
        static_cast<unsigned long long>(local_nb),
        static_cast<unsigned long long>(env_count),
        Kappa,
        m_total_frames);
}
