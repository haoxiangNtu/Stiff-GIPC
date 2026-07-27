#include "sim_engine.h"

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <cuda_runtime.h>

#include "GIPC.cuh"
#include "errors.h"
#include "mlbvh.cuh"
#include "load_mesh.h"
#include "device_fem_data.cuh"
#include "femEnergy.cuh"
#include "gpu_eigen_libs.cuh"
#include "fem_parameters.h"
#include "gipc_path.h"
#include "eigen_data.h"
#include "cuda_tools/cuda_tools.h"
#include <gipc/type_define.h>
#include <gipc/utils/urdf_scene_importer.h>
#include <gipc/utils/simple_scene_importer.h>
#include "abd_system/abd_system.h"
#include <Eigen/Geometry>

// Defined at GLOBAL scope in GIPC.cu (GIPC class is not in namespace gipc).
#include "device_common/debug_probes.h"  // [A4] g_gipc_log_level decl

namespace
{
// The user-facing per-vertex collision filter is backed by one CUDA device
// symbol in mlbvh, not by an lbvh/Engine field. Make that process-global
// constraint explicit so two same-process Engines cannot silently overwrite
// each other's pointer. Leaked function-local storage avoids static-destruction
// ordering hazards when Python tears the extension module down.
std::mutex& vertex_env_owner_mutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

void*& vertex_env_owner()
{
    static auto* owner = new void*(nullptr);
    return *owner;
}

std::mutex& runtime_owner_mutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

void*& runtime_owner()
{
    static auto* owner = new void*(nullptr);
    return *owner;
}

// The solver still publishes several per-engine buffers through process-global
// CUDA symbols (contact accumulation, MAS scratch and collision controls). Keep
// the lease alive until every other Impl member has been destroyed; declaring
// it first makes its destructor run last.
class RuntimeOwnerLease
{
  public:
    RuntimeOwnerLease() = default;
    RuntimeOwnerLease(const RuntimeOwnerLease&) = delete;
    RuntimeOwnerLease& operator=(const RuntimeOwnerLease&) = delete;

    void acquire(void* candidate)
    {
        std::lock_guard<std::mutex> lock(runtime_owner_mutex());
        if(runtime_owner() != nullptr && runtime_owner() != candidate)
            throw gipc::LifecycleError(
                "only one finalized SimEngine may be active in a process; "
                "reset or destroy the current engine before finalizing another");
        runtime_owner() = candidate;
        m_candidate = candidate;
    }

    void release() noexcept
    {
        std::lock_guard<std::mutex> lock(runtime_owner_mutex());
        if(runtime_owner() == m_candidate)
            runtime_owner() = nullptr;
        m_candidate = nullptr;
    }

    ~RuntimeOwnerLease() { release(); }

  private:
    void* m_candidate = nullptr;
};

class RuntimeOwnerAttempt
{
  public:
    RuntimeOwnerAttempt(RuntimeOwnerLease& lease,
                        void*              candidate,
                        bool&              finalize_failed)
        : m_lease(lease)
        , m_finalize_failed(finalize_failed)
    {
        m_lease.acquire(candidate);
    }

    void commit() noexcept { m_committed = true; }

    ~RuntimeOwnerAttempt()
    {
        if(!m_committed)
        {
            m_finalize_failed = true;
            m_lease.release();
        }
    }

  private:
    RuntimeOwnerLease& m_lease;
    bool&              m_finalize_failed;
    bool               m_committed = false;
};
}  // namespace

namespace gipc
{

struct SimEngine::Impl
{
    RuntimeOwnerLease runtime_owner_lease;
    // GIPC borrows several device_TetraData buffers and owns auxiliary CUDA
    // streams that may still reference them. Members are destroyed in reverse
    // declaration order, so declare the borrowed storage first: GIPC then tears
    // down its streams before device_TetraData releases the backing buffers.
    device_TetraData d_tetMesh;
    GIPC             ipc;
    tetrahedra_obj   tetMesh;
    SimEngineConfig  cfg;
    bool             cuda_initialized = false;
    bool             finalized        = false;
    bool             finalize_failed  = false;
    int              step_count       = 0;
    std::string      resolved_assets_dir;

    // [Step B] grow-only device buffers for per-contact force export
    int2*    d_contact_pair  = nullptr;
    double3* d_contact_force = nullptr;
    int      contact_cap     = 0;
    int      contact_count   = 0;

    // [multi-env subscene] device array of per-vertex env ids for broad-phase
    // cross-env contact isolation (see mlbvh_set_vertex_env_id). Alloc'd on first
    // set_vertex_env_ids(); length = vertexNum. Grow-only, mirrors d_contact_pair.
    int*     d_vertex_env_id   = nullptr;
    int      vertex_env_id_cap = 0;

    // Instance-owned scratch for readback/export APIs. These used to be
    // function-static CUDA allocations, which survived reset/destruction and
    // made two same-process Engines share mutable buffers.
    double*  d_stitch_scalar_out = nullptr;
    int*     d_segment_starts    = nullptr;
    int*     d_segment_counts    = nullptr;
    double*  d_segment_output    = nullptr;  // three doubles per segment
    int      segment_cap         = 0;
    double3* d_contact_gradient  = nullptr;
    int      contact_gradient_cap = 0;
    double*  d_stress_tet        = nullptr;
    double*  d_stress_vertex     = nullptr;
    int      stress_tet_cap      = 0;
    int      stress_vertex_cap   = 0;

    std::vector<BodyLoadRecord> load_records;

    // [per-body friction] load-record index -> (mu, ground_mu). ground_mu < 0
    // = keep the global gd_friction_rate. Expanded to per-vertex device tables
    // at finalize; empty map = feature off (kernels take the scalar path).
    std::unordered_map<int, std::pair<double, double>> pending_body_mu;

    // Per-FEM-body vertex ranges (populated during load)
    struct FEMBodyRange { int vertex_start; int vertex_count; };
    std::vector<FEMBodyRange> fem_body_ranges;

    // Shared mesh assets for instanced loading
    std::vector<MeshAsset> mesh_assets;

    // Cache of original abd-gravity vectors when set_body_apply_gravity(off) is
    // called.  Allows restoring without recomputing tet integrals.
    std::map<int, Eigen::Matrix<double, 12, 1>> disabled_abd_gravity_cache;

    // Pending mesh overrides for the next load_urdf() call.
    // Map link_name -> (msh_path, young_modulus). Cleared after load_urdf.
    std::map<std::string, std::pair<std::string, double>> pending_urdf_mesh_overrides;

    // Cache of URDF link world transforms after load_urdf (link_name -> 4x4).
    // Used by get_urdf_link_transform() for hybrid attachment placement.
    std::map<std::string, Eigen::Matrix4d> urdf_link_transforms;

    void apply_config_to_ipc();
    void do_initFEM();
    void do_setMAS_partition();
    void do_upload_to_gpu();
    void do_init_bvh_and_solver();

    void record_load(int body_type, int prev_verts);

    // Write mesh to temp file once, return the path.
    std::string write_temp_mesh(const double* vertices, int num_verts,
                                const int* faces, int num_faces,
                                int verts_per_face, int dimensions,
                                const std::string& suffix);

    // Load from a temp file with a given transform (core loading step).
    void load_from_temp_file(const std::string& tmp_path,
                             int dimensions, int body_type,
                             int verts_per_face,
                             const Eigen::Matrix4d& transform,
                             double young_modulus, int boundary_type);

    ~Impl()
    {
        // Destructors must not throw. CUDA errors are surfaced by the normal
        // APIs; teardown still attempts every owned allocation.
        if(d_vertex_env_id)
        {
            std::lock_guard<std::mutex> lock(vertex_env_owner_mutex());
            if(vertex_env_owner() == this)
            {
                try
                {
                    mlbvh_set_vertex_env_id(nullptr);
                }
                catch(...)
                {
                }
                vertex_env_owner() = nullptr;
            }
        }
        cudaFree(d_contact_pair);
        cudaFree(d_contact_force);
        cudaFree(d_vertex_env_id);
        cudaFree(d_stitch_scalar_out);
        cudaFree(d_segment_starts);
        cudaFree(d_segment_counts);
        cudaFree(d_segment_output);
        cudaFree(d_contact_gradient);
        cudaFree(d_stress_tet);
        cudaFree(d_stress_vertex);
    }
};

SimEngine::SimEngine()
    : m_impl(new Impl)
{
}

SimEngine::~SimEngine()
{
    delete m_impl;
}

// Null sink to fully silence std::cout when log_level <= 0 (catches every
// std::cout-based engine print in one shot).  printf-based prints are gated
// separately by g_gipc_log_level.  Python print() is unaffected (uses Python
// sys.stdout, not C++ std::cout).
namespace {
struct NullStreambuf : std::streambuf { int overflow(int c) override { return c; } };
NullStreambuf  g_null_streambuf;
std::streambuf* g_saved_cout_buf = nullptr;

void validate_programmatic_joint_bodies(int         body_count,
                                        int         parent_body,
                                        int         child_body,
                                        const char* api_name)
{
    if(parent_body < 0 || parent_body >= body_count
       || child_body < 0 || child_body >= body_count)
    {
        throw std::invalid_argument(
            std::string(api_name) + ": parent_body and child_body must be valid ABD "
            "body IDs in [0, " + std::to_string(body_count) + "); got parent="
            + std::to_string(parent_body) + ", child=" + std::to_string(child_body));
    }
    if(parent_body == child_body)
    {
        throw std::invalid_argument(
            std::string(api_name) + ": parent_body and child_body must be different; got body="
            + std::to_string(parent_body));
    }
    if(parent_body > child_body && ::g_gipc_log_level >= 1)
    {
        static bool reverse_order_notice_emitted = false;
        if(!reverse_order_notice_emitted)
        {
            std::cerr << "[SimEngine] NOTICE: " << api_name << " received parent_body="
                      << parent_body << " > child_body=" << child_body
                      << ". This order is valid; cross-body Hessian storage will be "
                         "canonicalized internally. No caller-side reordering is required."
                      << std::endl;
            reverse_order_notice_emitted = true;
        }
    }
}
}  // namespace

void SimEngine::set_log_level(int level)
{
    ::g_gipc_log_level = level;
    if(level <= 0)
    {
        if(!g_saved_cout_buf)
            g_saved_cout_buf = std::cout.rdbuf(&g_null_streambuf);
    }
    else if(g_saved_cout_buf)
    {
        std::cout.rdbuf(g_saved_cout_buf);
        g_saved_cout_buf = nullptr;
    }
}

void SimEngine::reset()
{
    // Recreate the whole Impl: ~Impl, ~GIPC and ~device_TetraData release all
    // instance-owned GPU buffers; a fresh Impl gives an empty world. Allocate
    // the replacement before destroying the live state so allocation failure
    // leaves this engine intact. Preserve Config and re-initialize CUDA.
    Impl* replacement = new Impl;
    replacement->cfg  = m_impl->cfg;
    delete m_impl;
    m_impl = replacement;
    init_cuda();
}

void SimEngine::set_config(const SimEngineConfig& cfg)
{
    if(m_impl->finalized)
        throw LifecycleError(
            "set_config() cannot mutate an already-finalized SimEngine");
    if(m_impl->cuda_initialized)
        throw LifecycleError(
            "set_config() cannot be called after CUDA initialization");
    if(!std::isfinite(cfg.energy_abs_tol) || cfg.energy_abs_tol < 0.0
       || !std::isfinite(cfg.energy_rel_tol) || cfg.energy_rel_tol < 0.0)
        throw std::invalid_argument(
            "energy_abs_tol and energy_rel_tol must be finite and non-negative");
    m_impl->cfg = cfg;
}

const SimEngineConfig& SimEngine::config() const
{
    return m_impl->cfg;
}

void SimEngine::init_cuda()
{
    // Several legacy CUDA scratch caches are still process-static. A second
    // device would make those pointers belong to the wrong CUDA context, so
    // fail explicitly instead of allowing a delayed invalid-pointer crash.
    static std::mutex process_device_mutex;
    static int        process_device = -1;
    std::lock_guard<std::mutex> lock(process_device_mutex);
    if(process_device >= 0 && process_device != m_impl->cfg.cuda_device)
        throw LifecycleError(
            "StiffGIPC CUDA device selection is process-scoped: initialized "
            "device " + std::to_string(process_device) + ", then requested "
            + std::to_string(m_impl->cfg.cuda_device)
            + ". Use a separate process for another GPU.");

    cudaError_t err = cudaSetDevice(m_impl->cfg.cuda_device);
    if(err != cudaSuccess)
        throw ConfigurationError(
            "cudaSetDevice(" + std::to_string(m_impl->cfg.cuda_device)
            + ") failed: " + cudaGetErrorString(err));
    process_device = m_impl->cfg.cuda_device;
    m_impl->cuda_initialized = true;

    if(m_impl->cfg.assets_dir.empty())
        m_impl->resolved_assets_dir = std::string{gipc::assets_dir()};
    else
        m_impl->resolved_assets_dir = m_impl->cfg.assets_dir;
}

std::string SimEngine::get_assets_dir() const
{
    return m_impl->resolved_assets_dir;
}

void SimEngine::Impl::record_load(int body_type, int prev_verts)
{
    BodyLoadRecord rec;
    rec.body_type     = body_type;
    rec.vertex_offset = prev_verts;
    rec.vertex_count  = tetMesh.vertexNum - prev_verts;
    if(body_type == 0)
        rec.body_offset = static_cast<int>(tetMesh.abd_fem_count_info.abd_body_num) - 1;
    else
        rec.body_offset = static_cast<int>(tetMesh.abd_fem_count_info.fem_body_num) - 1;
    load_records.push_back(rec);

    if(body_type == 1)
        fem_body_ranges.push_back({prev_verts, tetMesh.vertexNum - prev_verts});
}

void SimEngine::load_urdf(const std::string&     urdf_path,
                          const Eigen::Matrix4d& global_transform,
                          bool                   root_fixed,
                          bool                   revolute_as_motor,
                          double                 default_young,
                          const std::map<std::string, double>& initial_joint_angles)
{
    int prev_verts = m_impl->tetMesh.vertexNum;
    int prev_abd   = static_cast<int>(m_impl->tetMesh.abd_fem_count_info.abd_body_num);

    UrdfSceneImporter urdf_importer;
    urdf_importer.set_urdf_path(urdf_path);
    urdf_importer.set_global_transform(global_transform);
    urdf_importer.set_root_fixed(root_fixed);
    urdf_importer.set_default_boundary_type(BodyBoundaryType::Free);
    urdf_importer.set_revolute_as_motor(revolute_as_motor);
    urdf_importer.set_default_young_modulus(default_young);
    if(!initial_joint_angles.empty())
        urdf_importer.set_initial_joint_angles(initial_joint_angles);

    // Apply any pending mesh overrides (set via set_urdf_mesh_override
    // before this load_urdf call).
    for(const auto& [link_name, info] : m_impl->pending_urdf_mesh_overrides)
    {
        UrdfLinkMeshOverride o;
        o.msh_path      = info.first;
        o.young_modulus = info.second;
        urdf_importer.set_mesh_override(link_name, o);
    }
    m_impl->pending_urdf_mesh_overrides.clear();

    bool ok = urdf_importer.import_scene(m_impl->tetMesh, m_impl->cfg.preconditioner_type);
    if(!ok)
    {
        std::cerr << "[SimEngine] URDF import failed: " << urdf_path << std::endl;
        return;
    }

    // Cache each link's world transform for later get_urdf_link_transform()
    for(const auto& [name, info] : urdf_importer.link_infos())
        m_impl->urdf_link_transforms[name] = info.global_transform;

    int new_abd = static_cast<int>(m_impl->tetMesh.abd_fem_count_info.abd_body_num);
    // Derive per-body vertex ranges from point_id_to_body_id (one body id per vertex,
    // appended in body-order during loading — the per-body span is a contiguous range).
    const auto& pt2body = m_impl->tetMesh.point_id_to_body_id;
    std::vector<int> body_start(new_abd, -1);
    std::vector<int> body_end(new_abd, 0);  // exclusive upper bound
    for(int i = prev_verts; i < m_impl->tetMesh.vertexNum; i++)
    {
        int b = pt2body[i];
        if(b < 0 || b >= new_abd) continue;
        if(body_start[b] < 0) body_start[b] = i;
        body_end[b] = i + 1;
    }
    // Map body_id -> URDF link name (importer already parsed these; without
    // this, every body would just record the URDF file path as label and
    // become indistinguishable downstream).
    std::vector<std::string> body_link_name(new_abd);
    for(const auto& [link_name, link_info] : urdf_importer.link_infos())
    {
        if(link_info.body_id >= 0 && link_info.body_id < new_abd)
            body_link_name[link_info.body_id] = link_name;
    }
    for(int b = prev_abd; b < new_abd; b++)
    {
        BodyLoadRecord rec;
        rec.body_type     = 0;
        rec.body_offset   = b;
        rec.vertex_offset = body_start[b] >= 0 ? body_start[b] : prev_verts;
        rec.vertex_count  = body_start[b] >= 0 ? (body_end[b] - body_start[b]) : 0;
        rec.label         = body_link_name[b].empty() ? urdf_path : body_link_name[b];
        m_impl->load_records.push_back(rec);
    }

    std::cout << "[SimEngine] URDF loaded: " << urdf_path << std::endl;
    std::cout << "  ABD bodies: " << m_impl->tetMesh.abd_fem_count_info.abd_body_num << std::endl;
    std::cout << "  Revolute joints: " << m_impl->tetMesh.joint_angle_controls.size() << std::endl;
    std::cout << "  Prismatic joints: " << m_impl->tetMesh.prismatic_drive_controls.size() << std::endl;
}

void SimEngine::add_ground(double /*height*/)
{
    // StiffGIPC has implicit ground at y=0 by default in the solver.
    // This is a no-op placeholder; the ground is always active unless
    // m_skip_all_collision is set.
}

void SimEngine::load_mesh(const std::string&     mesh_path,
                          int                    dimensions,
                          int                    body_type,
                          const Eigen::Matrix4d& transform,
                          double                 young_modulus,
                          int                    boundary_type)
{
    int prev_verts = m_impl->tetMesh.vertexNum;

    std::string resolved = mesh_path;
    if(!std::filesystem::path(resolved).is_absolute())
    {
        std::string candidate = m_impl->resolved_assets_dir + resolved;
        if(std::filesystem::exists(candidate))
            resolved = candidate;
    }

    auto bt = (body_type == 0) ? gipc::BodyType::ABD : gipc::BodyType::FEM;
    auto bb = (boundary_type == 1) ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;

    SimpleSceneImporter imp;
    // Pass runtime metis_dir so the metis_partition library writes its
    // sorted_*/part_* intermediates to a runtime-resolved location instead
    // of the build-time OUTPUT_DIR macro (Issue 1).
    std::string metis_dir = m_impl->resolved_assets_dir + "sorted_mesh/";
    std::filesystem::create_directories(metis_dir);
    imp.load_geometry(m_impl->tetMesh, dimensions, bt, transform,
                      young_modulus, resolved,
                      m_impl->cfg.preconditioner_type, bb, metis_dir);

    m_impl->record_load(body_type, prev_verts);
    m_impl->load_records.back().label = resolved;

    std::cout << "[SimEngine] Mesh loaded: " << resolved
              << " (dim=" << dimensions
              << ", " << (body_type == 0 ? "ABD" : "FEM")
              << ", " << (boundary_type == 1 ? "Fixed" : "Free")
              << ", E=" << young_modulus << ")" << std::endl;
    std::cout << "  Total verts: " << m_impl->tetMesh.vertexNum
              << ", ABD bodies: " << m_impl->tetMesh.abd_fem_count_info.abd_body_num
              << ", FEM bodies: " << m_impl->tetMesh.abd_fem_count_info.fem_body_num
              << std::endl;
}

void SimEngine::add_collision_exclusion(int body_a, int body_b)
{
    m_impl->tetMesh.collision_exclusion_pairs.emplace_back(body_a, body_b);
}

void SimEngine::set_body_groups(const std::vector<int>& groups)
{
    // [multi-env] group id per collision-body (ABD ids first, then FEM). Bodies
    // in different groups (both >=0) are excluded from collision at finalize.
    m_impl->tetMesh.body_groups = groups;
}

void SimEngine::set_env_offsets(const std::vector<double>& per_group_xyz)
{
    auto& tm  = m_impl->tetMesh;
    auto& ipc = m_impl->ipc;
    if(ipc.d_env_offset == nullptr)
    {
        printf("[env-offset] WARNING: call set_env_offsets() AFTER finalize() (d_env_offset not allocated)\n");
        return;
    }
    int          G   = (int)per_group_xyz.size() / 3;
    const auto&  p2b = tm.point_id_to_body_id;          // point -> body
    const auto&  bg  = tm.body_groups;                  // body  -> group
    int          n   = ipc.vertexNum;
    std::vector<double3> off(n, make_double3(0.0, 0.0, 0.0));
    for(int v = 0; v < n && v < (int)p2b.size(); ++v)
    {
        int b = p2b[v];
        int g = (b >= 0 && b < (int)bg.size()) ? bg[b] : -1;
        if(g >= 0 && g < G)
            off[v] = make_double3(per_group_xyz[3 * g + 0],
                                  per_group_xyz[3 * g + 1],
                                  per_group_xyz[3 * g + 2]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(ipc.d_env_offset, off.data(),
                              n * sizeof(double3), cudaMemcpyHostToDevice));
    if(::g_gipc_log_level >= 1)
        printf("[env-offset] uploaded per-vertex offsets: %d groups, %d verts\n", G, n);
}

void SimEngine::get_point_groups(int* out, int count) const
{
    // [decouple] per-vertex env/group id in INPUT order (== get_vertex_positions order):
    // out[v] = body_groups[point_id_to_body_id[v]]. -1 if ungrouped/unknown. Host-side, no GPU.
    const auto& p2b = m_impl->tetMesh.point_id_to_body_id;   // input order, point -> body
    const auto& bg  = m_impl->tetMesh.body_groups;           // body -> group (may be empty)
    int n = std::min(count, static_cast<int>(p2b.size()));
    for(int v = 0; v < n; ++v)
    {
        int b = p2b[v];
        out[v] = (b >= 0 && b < static_cast<int>(bg.size())) ? bg[b] : -1;
    }
    for(int v = n; v < count; ++v) out[v] = -1;
}

void SimEngine::set_vertex_env_ids(const std::vector<int>& env_ids)
{
    // [multi-env subscene] Per-VERTEX env id -> broad-phase skips contact pairs
    // whose two vertices belong to different (>=0) envs. Unlike set_body_groups
    // (per-body), this isolates a FEM body whose particles span many envs, WITHOUT
    // spatial separation. Decoupled from the block-diagonal solve (contact only).
    auto& impl = *m_impl;
    int   n    = static_cast<int>(env_ids.size());
    std::lock_guard<std::mutex> lock(vertex_env_owner_mutex());
    if(n <= 0)
    {
        if(vertex_env_owner() == &impl)
        {
            mlbvh_set_vertex_env_id(nullptr);
            vertex_env_owner() = nullptr;
        }
        return;
    }
    if(!impl.finalized)
        throw LifecycleError(
            "set_vertex_env_ids() requires a finalized SimEngine");
    if(n != impl.ipc.vertexNum)
        throw std::invalid_argument(
            "set_vertex_env_ids() requires exactly one id per vertex: expected "
            + std::to_string(impl.ipc.vertexNum) + ", got " + std::to_string(n));
    if(vertex_env_owner() != nullptr && vertex_env_owner() != &impl)
        throw LifecycleError(
            "set_vertex_env_ids() is process-scoped and is already owned by "
            "another SimEngine; clear it there or use a separate process");
    if(impl.d_vertex_env_id == nullptr || impl.vertex_env_id_cap < n)
    {
        int* replacement = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&replacement, n * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(replacement,
                                  env_ids.data(),
                                  n * sizeof(int),
                                  cudaMemcpyHostToDevice));
        int* previous = impl.d_vertex_env_id;
        impl.d_vertex_env_id = replacement;
        impl.vertex_env_id_cap = n;
        mlbvh_set_vertex_env_id(impl.d_vertex_env_id);
        vertex_env_owner() = &impl;
        CUDA_SAFE_CALL(cudaFree(previous));
        return;
    }
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_vertex_env_id, env_ids.data(),
                              n * sizeof(int), cudaMemcpyHostToDevice));
    mlbvh_set_vertex_env_id(impl.d_vertex_env_id);
    vertex_env_owner() = &impl;
}

void SimEngine::add_ground_collision_skip(int body_id)
{
    m_impl->tetMesh.ground_collision_skip_body_ids.push_back(body_id);
}

void SimEngine::add_stitch_spring(int fem_vertex_global_id,
                                  int abd_anchor_vertex_global_id,
                                  int abd_body_id,
                                  const Eigen::Vector3d& rest_offset_world)
{
    auto& tm  = m_impl->tetMesh;
    auto& dtm = m_impl->d_tetMesh;
    // FEM vertex this spring acts on
    tm.targetIndex.push_back(static_cast<uint32_t>(fem_vertex_global_id));
    // Initial target world position; engine will overwrite each step from
    // ABD body's current vertex world pos. Seed with 0 (or pass current pos
    // if available -- engine reads from ABD anchor vertex anyway).
    tm.targetPos.push_back(make_double3(0.0, 0.0, 0.0));
    // Bilateral stitch-spring info (consumed at finalize -> safe_copy to GPU)
    dtm.stitch_paired_vertex.push_back(abd_anchor_vertex_global_id);
    dtm.stitch_rest_offset.push_back(make_double3(
        rest_offset_world.x(), rest_offset_world.y(), rest_offset_world.z()));
    dtm.stitch_abd_body_id.push_back(abd_body_id);
    tm.softNum = static_cast<int>(tm.targetIndex.size());
}

void SimEngine::set_per_tet_young_for_body(int body_offset,
                                            const std::vector<double>& per_tet_young)
{
    auto& tm = m_impl->tetMesh;
    if(body_offset < 0 || body_offset >= (int)m_impl->load_records.size())
        throw std::runtime_error("set_per_tet_young_for_body: invalid body_offset "
                                 + std::to_string(body_offset));
    const auto& r = m_impl->load_records[body_offset];
    int v_off = r.vertex_offset;
    int v_end = v_off + r.vertex_count;
    // Find tets owned by this body: all 4 verts in [v_off, v_end).  Sequential
    // scan over tetrahedras (called once per body at setup, O(n_tets)).
    std::vector<int> tet_indices;
    tet_indices.reserve(per_tet_young.size());
    for(int t = 0; t < tm.tetrahedraNum; ++t)
    {
        const auto& te = tm.tetrahedras[t];
        if((int)te.x >= v_off && (int)te.x < v_end &&
           (int)te.y >= v_off && (int)te.y < v_end &&
           (int)te.z >= v_off && (int)te.z < v_end &&
           (int)te.w >= v_off && (int)te.w < v_end)
            tet_indices.push_back(t);
    }
    if(per_tet_young.size() != tet_indices.size())
        throw std::runtime_error(
            "set_per_tet_young_for_body: per_tet_young size " +
            std::to_string(per_tet_young.size()) + " != body tet count " +
            std::to_string(tet_indices.size()));
    for(size_t k = 0; k < tet_indices.size(); ++k)
        tm.vert_youngth_modules[tet_indices[k]] = per_tet_young[k];
    if(g_gipc_log_level >= 1)
        printf("[per-tet-young] body %d: %zu tets, young range [%.3g, %.3g]\n",
               body_offset, tet_indices.size(),
               *std::min_element(per_tet_young.begin(), per_tet_young.end()),
               *std::max_element(per_tet_young.begin(), per_tet_young.end()));
}

std::vector<int> SimEngine::get_per_env_newton_iters() const
{
    return m_impl->ipc.m_env_frozen_iter;
}

std::vector<int> SimEngine::get_per_env_status() const
{
    return m_impl->ipc.m_env_status;
}

void SimEngine::set_body_friction(int body_offset, double mu, double ground_mu)
{
    if(body_offset < 0 || body_offset >= (int)m_impl->load_records.size())
        throw std::runtime_error("set_body_friction: invalid body_offset "
                                 + std::to_string(body_offset));
    if(!(mu >= 0.0))
        throw std::runtime_error("set_body_friction: mu must be >= 0");
    // Stash only — expanded to per-vertex tables at finalize (vertex layout is
    // final there). ground_mu < 0 keeps the global gd_friction_rate.
    m_impl->pending_body_mu[body_offset] = {mu, ground_mu};
}

void SimEngine::set_soft_body_density(int body_offset, double density)
{
    auto& tm = m_impl->tetMesh;
    if(body_offset < 0 || body_offset >= (int)m_impl->load_records.size())
        throw std::runtime_error("set_soft_body_density: invalid body_offset "
                                 + std::to_string(body_offset));
    if(!(density > 0.0))
        throw std::runtime_error("set_soft_body_density: density must be > 0");
    const auto& r     = m_impl->load_records[body_offset];
    int         v_off = r.vertex_offset;
    int         v_end = v_off + r.vertex_count;
    // Lazy grow (default <= 0 == "use global"); resize only ever grows, so
    // earlier per-body assignments survive later loads.
    if((int)tm.tet_densities.size() < tm.tetrahedraNum)
        tm.tet_densities.resize(tm.tetrahedraNum, -1.0);
    if(tm.tri_densities.size() < tm.triangles.size())
        tm.tri_densities.resize(tm.triangles.size(), -1.0);
    int n_tet = 0, n_tri = 0;
    for(int t = 0; t < tm.tetrahedraNum; ++t)
    {
        const auto& te = tm.tetrahedras[t];
        if((int)te.x >= v_off && (int)te.x < v_end && (int)te.y >= v_off
           && (int)te.y < v_end && (int)te.z >= v_off && (int)te.z < v_end
           && (int)te.w >= v_off && (int)te.w < v_end)
        {
            tm.tet_densities[t] = density;
            ++n_tet;
        }
    }
    for(size_t t = 0; t < tm.triangles.size(); ++t)
    {
        const auto& tr = tm.triangles[t];
        if((int)tr.x >= v_off && (int)tr.x < v_end && (int)tr.y >= v_off
           && (int)tr.y < v_end && (int)tr.z >= v_off && (int)tr.z < v_end)
        {
            tm.tri_densities[t] = density;
            ++n_tri;
        }
    }
    if(n_tet + n_tri == 0)
        throw std::runtime_error(
            "set_soft_body_density: body " + std::to_string(body_offset)
            + " owns no tets or shell triangles (ABD body? use set_abd_body_density)");
    if(g_gipc_log_level >= 1)
        printf("[per-body-density] body %d: %d tets, %d tris -> rho=%.3g\n",
               body_offset, n_tet, n_tri, density);
}

void SimEngine::add_fem_pin_to_abd(int fem_vertex_global_id,
                                   int abd_anchor_vertex_global_id,
                                   int abd_body_id,
                                   const Eigen::Vector3d& rest_offset_world)
{
    auto& tm = m_impl->tetMesh;
    tm.fem_pin_fem_vertex.push_back(fem_vertex_global_id);
    tm.fem_pin_abd_body_id.push_back(abd_body_id);
    // abd_local_pos placeholder; populated at finalize() once ABD's q is initialized.
    // Stored here as the world rest_offset; finalize transforms it to local pos.
    tm.fem_pin_abd_local_pos.push_back(make_double3(0.0, 0.0, 0.0));
    tm.fem_pin_abd_anchor.push_back(abd_anchor_vertex_global_id);
    tm.fem_pin_rest_offset.push_back(make_double3(
        rest_offset_world.x(), rest_offset_world.y(), rest_offset_world.z()));
}

void SimEngine::add_fem_pins_with_local_pos(
    const std::vector<int>&             fem_vertex_global_ids,
    const std::vector<int>&             abd_body_ids,
    const std::vector<Eigen::Vector3d>& abd_local_positions)
{
    const size_t n = fem_vertex_global_ids.size();
    if(abd_body_ids.size() != n || abd_local_positions.size() != n)
    {
        throw std::invalid_argument(
            "add_fem_pins_with_local_pos: input vectors size mismatch ("
            + std::to_string(fem_vertex_global_ids.size()) + ", "
            + std::to_string(abd_body_ids.size()) + ", "
            + std::to_string(abd_local_positions.size()) + ")");
    }
    auto& tm = m_impl->tetMesh;
    tm.fem_pin_fem_vertex.reserve(tm.fem_pin_fem_vertex.size() + n);
    tm.fem_pin_abd_body_id.reserve(tm.fem_pin_abd_body_id.size() + n);
    tm.fem_pin_abd_local_pos.reserve(tm.fem_pin_abd_local_pos.size() + n);
    tm.fem_pin_abd_anchor.reserve(tm.fem_pin_abd_anchor.size() + n);
    tm.fem_pin_rest_offset.reserve(tm.fem_pin_rest_offset.size() + n);
    for(size_t i = 0; i < n; ++i)
    {
        tm.fem_pin_fem_vertex.push_back(fem_vertex_global_ids[i]);
        tm.fem_pin_abd_body_id.push_back(abd_body_ids[i]);
        // local_pos provided directly — finalize() must skip the world-rest-offset
        // → local-pos transform for these pins.  Sentinel: anchor = -1.
        const auto& lp = abd_local_positions[i];
        tm.fem_pin_abd_local_pos.push_back(make_double3(lp.x(), lp.y(), lp.z()));
        tm.fem_pin_abd_anchor.push_back(-1);
        tm.fem_pin_rest_offset.push_back(make_double3(0.0, 0.0, 0.0));
    }
    printf("[Hybrid] add_fem_pins_with_local_pos: appended %zu pins "
           "(total now %zu)\n", n, tm.fem_pin_fem_vertex.size());
}

bool SimEngine::set_abd_body_face_orient(int body_id, const std::vector<int>& orient)
{
    for(auto& smb : m_impl->tetMesh.surface_mesh_bodies)
    {
        if(smb.body_id != body_id) continue;
        if(!orient.empty() && orient.size() != smb.triangles.size())
        {
            std::cerr << "[set_abd_body_face_orient] body " << body_id
                      << ": orient size (" << orient.size() << ") != face count ("
                      << smb.triangles.size() << "), ignoring." << std::endl;
            return false;
        }
        smb.orient = orient;
        return true;
    }
    std::cerr << "[set_abd_body_face_orient] no surface-mesh body with body_id="
              << body_id << " (FEM body, or invalid index)." << std::endl;
    return false;
}

std::vector<int> SimEngine::get_abd_body_face_orient(int body_id) const
{
    for(auto& smb : m_impl->tetMesh.surface_mesh_bodies)
    {
        if(smb.body_id == body_id)
            return smb.orient;
    }
    return {};
}

std::vector<double> SimEngine::get_abd_surface_body_vertices(int body_id) const
{
    for(auto& smb : m_impl->tetMesh.surface_mesh_bodies)
    {
        if(smb.body_id != body_id) continue;
        std::vector<double> out(smb.vertices.size() * 3);
        for(size_t i = 0; i < smb.vertices.size(); i++)
        {
            out[3 * i + 0] = smb.vertices[i].x();
            out[3 * i + 1] = smb.vertices[i].y();
            out[3 * i + 2] = smb.vertices[i].z();
        }
        return out;
    }
    return {};
}

std::vector<int> SimEngine::get_abd_surface_body_triangles(int body_id) const
{
    for(auto& smb : m_impl->tetMesh.surface_mesh_bodies)
    {
        if(smb.body_id != body_id) continue;
        std::vector<int> out(smb.triangles.size() * 3);
        for(size_t i = 0; i < smb.triangles.size(); i++)
        {
            out[3 * i + 0] = smb.triangles[i].x();
            out[3 * i + 1] = smb.triangles[i].y();
            out[3 * i + 2] = smb.triangles[i].z();
        }
        return out;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Programmatic joint creation (mirrors UrdfSceneImporter logic)
// ---------------------------------------------------------------------------

int SimEngine::add_fixed_joint(int parent_body, int child_body,
                               const Eigen::Vector3d& world_anchor,
                               const Eigen::Vector3d& world_normal,
                               const Eigen::Vector3d& world_bitangent)
{
    validate_programmatic_joint_bodies(
        m_impl->tetMesh.abd_fem_count_info.abd_body_num,
        parent_body,
        child_body,
        "add_fixed_joint");

    JointConstraintHostInfo jc;
    jc.parent_body_id = parent_body;
    jc.child_body_id  = child_body;
    jc.type       = JointConstraintHostInfo::Type::Fixed;
    jc.num_points = 1;
    jc.world_anchor[0] = world_anchor;
    jc.has_direction_constraint = true;
    jc.world_normal    = world_normal.normalized();
    jc.world_bitangent = world_bitangent.normalized();
    // Third affine basis axis (libuipc penalizes all of t,n,b). Derive it as the
    // orthogonal complement so the three directions are an orthonormal frame even if
    // the caller passes only normal/bitangent.
    jc.world_tangent   = jc.world_normal.cross(jc.world_bitangent).normalized();

    int idx = static_cast<int>(m_impl->tetMesh.joint_constraints.size());
    m_impl->tetMesh.joint_constraints.push_back(jc);
    return idx;
}

int SimEngine::add_revolute_joint(int parent_body, int child_body,
                                  const Eigen::Vector3d& world_axis,
                                  const Eigen::Vector3d& joint_pos,
                                  double lower_limit, double upper_limit,
                                  double initial_angle,
                                  const std::string& name,
                                  bool passive)
{
    validate_programmatic_joint_bodies(
        m_impl->tetMesh.abd_fem_count_info.abd_body_num,
        parent_body,
        child_body,
        "add_revolute_joint");

    Eigen::Vector3d axis = world_axis.normalized();
    Eigen::Vector3d half = axis * 0.5;

    JointConstraintHostInfo jc;
    jc.parent_body_id = parent_body;
    jc.child_body_id  = child_body;
    jc.type       = JointConstraintHostInfo::Type::Revolute;
    jc.num_points = 2;
    jc.world_anchor[0] = joint_pos + half;
    jc.world_anchor[1] = joint_pos - half;
    jc.point_weight[0] = 1.0;
    jc.point_weight[1] = 1.0;

    // Stable perpendicular direction for angle measurement
    Eigen::Vector3d n_perp;
    if(std::abs(axis.x()) < 0.9)
        n_perp = axis.cross(Eigen::Vector3d::UnitX()).normalized();
    else
        n_perp = axis.cross(Eigen::Vector3d::UnitY()).normalized();

    JointAngleControlInfo ctrl;
    ctrl.constraint_index = static_cast<int>(m_impl->tetMesh.joint_constraints.size());
    ctrl.axis_dir         = axis;
    ctrl.n_dir            = n_perp;
    ctrl.target_angle     = initial_angle;
    ctrl.lower_limit      = std::max(lower_limit, -JointAngleControlInfo::kSafeAngleLimit);
    ctrl.upper_limit      = std::min(upper_limit,  JointAngleControlInfo::kSafeAngleLimit);
    ctrl.joint_name       = name;
    // [passive joints] a passive hinge has NO position servo: K = sr *
    // strength_ratio * mass == 0. The axis/anchor constraint and the
    // independent joint-limit penalty still act. Without this every manual
    // revolute was silently position-locked at initial_angle (a "passive"
    // pendulum stayed frozen horizontal — regression test_passive_revolute).
    ctrl.strength_ratio = passive ? 0.0 : 1.0;
    m_impl->tetMesh.joint_angle_controls.push_back(ctrl);

    int idx = static_cast<int>(m_impl->tetMesh.joint_constraints.size());
    m_impl->tetMesh.joint_constraints.push_back(jc);
    return idx;
}

int SimEngine::add_prismatic_joint(int parent_body, int child_body,
                                   const Eigen::Vector3d& world_center,
                                   const Eigen::Vector3d& world_axis,
                                   double lower_limit, double upper_limit,
                                   const std::string& name,
                                   bool passive)
{
    validate_programmatic_joint_bodies(
        m_impl->tetMesh.abd_fem_count_info.abd_body_num,
        parent_body,
        child_body,
        "add_prismatic_joint");

    Eigen::Vector3d axis = world_axis.normalized();

    Eigen::Vector3d n_perp;
    if(std::abs(axis.x()) < 0.9)
        n_perp = axis.cross(Eigen::Vector3d::UnitX()).normalized();
    else
        n_perp = axis.cross(Eigen::Vector3d::UnitY()).normalized();
    Eigen::Vector3d b_perp = axis.cross(n_perp).normalized();

    PrismaticJointHostInfo pj;
    pj.parent_body_id  = parent_body;
    pj.child_body_id   = child_body;
    pj.world_center    = world_center;
    pj.world_axis      = axis;
    pj.world_normal    = n_perp;
    pj.world_bitangent = b_perp;

    int idx = static_cast<int>(m_impl->tetMesh.prismatic_constraints.size());

    PrismaticDrivingControlInfo pctrl;
    pctrl.prismatic_constraint_index = idx;
    pctrl.axis_dir        = axis;
    pctrl.target_distance = 0.0;
    pctrl.lower_limit     = lower_limit;
    pctrl.upper_limit     = upper_limit;
    pctrl.joint_name      = name;
    // [passive joints] see add_revolute_joint: zero the position servo.
    pctrl.strength_ratio = passive ? 0.0 : 1.0;

    m_impl->tetMesh.prismatic_constraints.push_back(pj);
    m_impl->tetMesh.prismatic_drive_controls.push_back(pctrl);
    m_impl->tetMesh.collision_exclusion_pairs.push_back({parent_body, child_body});

    return idx;
}

void SimEngine::set_vertex_boundary(int vertex_index, int boundary_type)
{
    // [MAS-perm] vertex_index is INPUT-mesh order (the order the user sees via
    // get_vertex_position_host). boundaryTypies is engine (metis) order, so we
    // must translate input -> engine through the inverse permutation, exactly
    // like get_vertex_position_host does. Without this, set_vertex_boundary
    // flags the WRONG engine vertex whenever the mesh is metis-reordered, so
    // pinned cloth corners are not actually held and the sheet collapses.
    // Without a perm, engine == input (identity).
    const int vnum = m_impl->tetMesh.vertexNum;
    if(vertex_index < 0 || vertex_index >= vnum)
        return;
    int engine_idx = vertex_index;
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;  // engine_idx -> input_idx
    if(!perm.empty() && static_cast<int>(perm.size()) >= vnum)
    {
        for(int e = 0; e < vnum; e++)
            if(perm[e] == vertex_index) { engine_idx = e; break; }
    }
    if(engine_idx >= 0 && engine_idx < static_cast<int>(m_impl->tetMesh.boundaryTypies.size()))
        m_impl->tetMesh.boundaryTypies[engine_idx] = boundary_type;
}

int SimEngine::get_abd_body_count() const
{
    return m_impl->tetMesh.abd_fem_count_info.abd_body_num;
}

int SimEngine::get_vertex_count_host() const
{
    return m_impl->tetMesh.vertexNum;
}

void SimEngine::get_vertex_position_host(int idx, double out_xyz[3]) const
{
    // [MAS-perm] idx here is INPUT-mesh order. Resolve via inverse perm:
    // find engine_idx such that perm[engine_idx] == idx. Without perm,
    // identity (engine_idx == idx).
    if(idx < 0 || idx >= m_impl->tetMesh.vertexNum)
    {
        out_xyz[0] = out_xyz[1] = out_xyz[2] = 0.0;
        return;
    }
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    int engine_idx = idx;
    if(!perm.empty() && static_cast<int>(perm.size()) >= m_impl->tetMesh.vertexNum)
    {
        // Linear search; expected use is sparse (one-off lookups). For a
        // batch use case, prefer get_vertex_positions_host below.
        for(int i = 0; i < m_impl->tetMesh.vertexNum; i++)
        {
            if(perm[i] == idx) { engine_idx = i; break; }
        }
    }
    const auto& v = m_impl->tetMesh.vertexes[engine_idx];
    out_xyz[0] = v.x;
    out_xyz[1] = v.y;
    out_xyz[2] = v.z;
}

void SimEngine::get_vertex_positions_host(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->tetMesh.vertexNum));
    if(n <= 0) return;
    // [MAS-perm] tetMesh.vertexes is engine (metis) order; output in input order.
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty()
                    && static_cast<int>(perm.size()) >= m_impl->tetMesh.vertexNum;
    if(!use_perm)
    {
        for(int i = 0; i < n; i++)
        {
            const auto& v = m_impl->tetMesh.vertexes[i];
            out_xyz[3*i + 0] = v.x; out_xyz[3*i + 1] = v.y; out_xyz[3*i + 2] = v.z;
        }
        return;
    }
    for(int i = 0; i < n; i++)
    {
        const auto& v = m_impl->tetMesh.vertexes[i];
        int j = perm[i];
        if(j < 0 || j >= n) { j = i; }
        out_xyz[3*j + 0] = v.x;
        out_xyz[3*j + 1] = v.y;
        out_xyz[3*j + 2] = v.z;
    }
}

// ---------- apply config to IPC fields ----------
