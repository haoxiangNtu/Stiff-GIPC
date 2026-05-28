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
#include <cuda_runtime.h>

#include "GIPC.cuh"
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
extern int g_gipc_log_level;

namespace gipc
{

struct SimEngine::Impl
{
    GIPC             ipc;
    device_TetraData d_tetMesh;
    tetrahedra_obj   tetMesh;
    SimEngineConfig  cfg;
    bool             cuda_initialized = false;
    bool             finalized        = false;
    int              step_count       = 0;
    std::string      resolved_assets_dir;

    // [Step B] grow-only device buffers for per-contact force export
    int2*    d_contact_pair  = nullptr;
    double3* d_contact_force = nullptr;
    int      contact_cap     = 0;
    int      contact_count   = 0;

    std::vector<BodyLoadRecord> load_records;

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
    // Recreate the whole Impl: ~Impl frees GPU buffers via ~GIPC/~device_TetraData
    // (no leak); a fresh Impl gives an empty world.  Preserve Config + re-init CUDA.
    SimEngineConfig saved_cfg = m_impl->cfg;
    delete m_impl;
    m_impl = new Impl;
    m_impl->cfg = saved_cfg;
    init_cuda();
}

void SimEngine::set_config(const SimEngineConfig& cfg)
{
    m_impl->cfg = cfg;
}

const SimEngineConfig& SimEngine::config() const
{
    return m_impl->cfg;
}

void SimEngine::init_cuda()
{
    cudaError_t err = cudaSetDevice(m_impl->cfg.cuda_device);
    if(err != cudaSuccess)
    {
        std::cerr << "[SimEngine] cudaSetDevice(" << m_impl->cfg.cuda_device
                  << ") failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
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
                                  const std::string& name)
{
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
    m_impl->tetMesh.joint_angle_controls.push_back(ctrl);

    int idx = static_cast<int>(m_impl->tetMesh.joint_constraints.size());
    m_impl->tetMesh.joint_constraints.push_back(jc);
    return idx;
}

int SimEngine::add_prismatic_joint(int parent_body, int child_body,
                                   const Eigen::Vector3d& world_center,
                                   const Eigen::Vector3d& world_axis,
                                   double lower_limit, double upper_limit,
                                   const std::string& name)
{
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
void SimEngine::Impl::apply_config_to_ipc()
{
    ipc.density             = cfg.density;
    ipc.PoissonRate         = cfg.poisson_rate;
    ipc.frictionRate        = cfg.friction_rate;
    ipc.gd_frictionRate     = cfg.gd_friction_rate;
    ipc.clothThickness      = cfg.cloth_thickness;
    ipc.clothYoungModulus   = cfg.cloth_young_modulus;
    ipc.bendYoungModulus    = cfg.bend_young_modulus;
    ipc.clothDensity        = cfg.cloth_density;
    ipc.strainRate          = cfg.strain_rate;
    ipc.softMotionRate      = cfg.soft_motion_rate;
    ipc.IPC_dt              = cfg.dt;
    ipc.gravity             = make_double3(cfg.gravity.x(), cfg.gravity.y(), cfg.gravity.z());
    ipc.ground_normal_cfg   = make_double3(cfg.ground_normal.x(), cfg.ground_normal.y(), cfg.ground_normal.z());
    ipc.ground_offset_cfg   = cfg.ground_offset;
    ipc.pcg_threshold       = cfg.pcg_tol;
    ipc.Newton_solver_threshold = cfg.newton_tol;
    ipc.relative_dhat       = cfg.relative_dhat;
    ipc.absolute_dhat       = cfg.absolute_dhat;
    ipc.YoungModulus        = cfg.young_modulus;
    ipc.pcg_data.P_type     = cfg.preconditioner_type;
    ipc.assets_dir_cfg      = resolved_assets_dir;

    ipc.semi_implicit_enabled  = cfg.semi_implicit_enabled;
    ipc.semi_implicit_beta_tol = cfg.semi_implicit_beta_tol;
    ipc.semi_implicit_min_iter = cfg.semi_implicit_min_iter;
    ipc.newton_iter_cap        = cfg.newton_iter_cap;

    ipc.m_skip_all_collision = cfg.skip_all_collision;
}

// ---------- FEM initialization (from gl_main.cu::initFEM) ----------
void SimEngine::Impl::do_initFEM()
{
    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.bendStiff    = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                       / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;

    double massSum   = 0;
    double volumeSum = 0;

    for(int i = 0; i < tetMesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(tetMesh.vertexes.data(), tetMesh.tetrahedras[i], DM);
        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);
        double vlm = calculateVolum(tetMesh.vertexes.data(), tetMesh.tetrahedras[i]);

        tetMesh.masses[tetMesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        tetMesh.masses[tetMesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        tetMesh.DM_inverse.push_back(DM_inverse);
        tetMesh.volum.push_back(vlm);

        double lrl = tetMesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double vrl = tetMesh.vert_youngth_modules[i] * ipc.PoissonRate
                     / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        tetMesh.lengthRate.push_back(4 * lrl / 3);
        tetMesh.volumeRate.push_back(vrl + 5 * lrl / 6);
    }

    for(size_t i = 0; i < tetMesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d DM;
        __calculateDm2D_double(tetMesh.vertexes.data(), tetMesh.triangles[i], DM);
        __GEIGEN__::Matrix2x2d DM_inverse;
        __GEIGEN__::__Inverse2x2(DM, DM_inverse);
        double area = calculateArea(tetMesh.vertexes.data(), tetMesh.triangles[i]);
        area *= ipc.clothThickness;
        tetMesh.area.push_back(area);

        tetMesh.masses[tetMesh.triangles[i].x] += ipc.clothDensity * area / 3;
        tetMesh.masses[tetMesh.triangles[i].y] += ipc.clothDensity * area / 3;
        tetMesh.masses[tetMesh.triangles[i].z] += ipc.clothDensity * area / 3;

        massSum += area * ipc.clothDensity;
        volumeSum += area;
        tetMesh.tri_DM_inverse.push_back(DM_inverse);
    }

    tetMesh.meanMass  = massSum / tetMesh.vertexNum;
    tetMesh.meanVolum = volumeSum / tetMesh.vertexNum;
}

// ---------- MAS partition (from gl_main.cu::setMAS_partition) ----------
void SimEngine::Impl::do_setMAS_partition()
{
    tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());
    int index = 0;
    for(size_t i = 0; i < tetMesh.partId.size(); i++)
    {
        tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = static_cast<int>(i);
        index++;
        if(i <= tetMesh.partId.size() - 2)
        {
            if(tetMesh.partId[i + 1] != tetMesh.partId[i])
                index = 0;
        }
    }
    index = 0;
    for(size_t i = 0; i < tetMesh.partId_map_real.size(); i++)
    {
        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = static_cast<int>(i);
            index++;
        }
    }
}

// ---------- Upload all host data to GPU ----------
void SimEngine::Impl::do_upload_to_gpu()
{
    cudaSetDevice(cfg.cuda_device);

    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                static_cast<int>(tetMesh.tri_edges.size()),
                                tetMesh.abd_fem_count_info.total_body_num());

    auto safe_copy = [](void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
        if(bytes > 0)
            CUDA_SAFE_CALL(cudaMemcpy(dst, src, bytes, kind));
    };

    safe_copy(d_tetMesh.masses, tetMesh.masses.data(),
              tetMesh.vertexNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.apply_gravity, tetMesh.apply_gravity.data(),
              tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.lengthRate, tetMesh.lengthRate.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.volumeRate, tetMesh.volumeRate.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.volum, tetMesh.volum.data(),
              tetMesh.tetrahedraNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.vertexes, tetMesh.vertexes.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.o_vertexes, tetMesh.vertexes.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tetrahedras, tetMesh.tetrahedras.data(),
              tetMesh.tetrahedraNum * sizeof(uint4), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.DmInverses, tetMesh.DM_inverse.data(),
              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.BoundaryType, tetMesh.boundaryTypies.data(),
              tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.velocities, tetMesh.velocities.data(),
              tetMesh.vertexNum * sizeof(double3), cudaMemcpyHostToDevice);
    // [MAS stitch index fix] When MAS is active (preconditioner_type != 0), FEM
    // bodies are loaded in metis-SORTED order: engine vertex (off+i) holds INPUT
    // vertex (off + sort_index[i]); vertex_metis_to_input[engine] = input.
    // Stitch springs are added by the user in INPUT-vertex order, so without
    // this translation they pull the WRONG engine vertices -> garbage forces ->
    // Newton never converges (the case_40 MAS-on bug). P_type==0 => perm is
    // identity => skipped (no-op).
    if(cfg.preconditioner_type != 0 && tetMesh.softNum > 0
       && !tetMesh.vertex_metis_to_input.empty())
    {
        const auto& m2i = tetMesh.vertex_metis_to_input;  // engine_idx -> input_idx
        std::vector<int> i2m(m2i.size(), -1);             // input_idx -> engine_idx
        for(int e = 0; e < (int)m2i.size(); e++)
            if(m2i[e] >= 0 && m2i[e] < (int)i2m.size())
                i2m[m2i[e]] = e;
        auto to_engine = [&](int input_id) -> int {
            return (input_id >= 0 && input_id < (int)i2m.size() && i2m[input_id] >= 0)
                       ? i2m[input_id] : input_id;
        };
        for(auto& v : tetMesh.targetIndex)
            v = static_cast<uint32_t>(to_engine(static_cast<int>(v)));
        for(auto& v : d_tetMesh.stitch_paired_vertex)  // ABD anchors: identity, safe
            v = to_engine(v);
        if(g_gipc_log_level >= 1)
            printf("[MAS-fix] remapped %d stitch FEM indices input->engine (metis) order\n",
                   tetMesh.softNum);
    }
    safe_copy(d_tetMesh.targetIndex, tetMesh.targetIndex.data(),
              tetMesh.softNum * sizeof(uint32_t), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.targetVert, tetMesh.targetPos.data(),
              tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice);

    d_tetMesh.host_target_indices  = tetMesh.targetIndex;
    d_tetMesh.host_target_vertices = tetMesh.targetPos;

    safe_copy(d_tetMesh.triDmInverses, tetMesh.tri_DM_inverse.data(),
              tetMesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.area, tetMesh.area.data(),
              tetMesh.triangleNum * sizeof(double), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.triangles, tetMesh.triangles.data(),
              tetMesh.triangleNum * sizeof(uint3), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tri_edges, tetMesh.tri_edges.data(),
              tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tri_edge_adj_vertex, tetMesh.tri_edges_adj_points.data(),
              tetMesh.tri_edges.size() * sizeof(uint2), cudaMemcpyHostToDevice);

    safe_copy(d_tetMesh.body_id_to_boundary_type, tetMesh.body_id_to_is_fixed.data(),
              tetMesh.body_id_to_is_fixed.size() * sizeof(int), cudaMemcpyHostToDevice);
    // [multi-FEM-bodyid] upload per-body FEM flag (size = collision_body_num).
    if(d_tetMesh.body_id_to_is_fem != nullptr && !tetMesh.body_id_to_is_fem.empty())
    {
        safe_copy(d_tetMesh.body_id_to_is_fem, tetMesh.body_id_to_is_fem.data(),
                  tetMesh.body_id_to_is_fem.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    safe_copy(d_tetMesh.point_id_to_body_id, tetMesh.point_id_to_body_id.data(),
              tetMesh.point_id_to_body_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    safe_copy(d_tetMesh.tet_id_to_body_id, tetMesh.tet_id_to_body_id.data(),
              tetMesh.tet_id_to_body_id.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Motor params
    if(!tetMesh.body_motor_infos.empty())
    {
        int bodyNum = static_cast<int>(tetMesh.body_motor_infos.size());
        std::vector<double> motor_params(bodyNum * 5, 0.0);
        for(int i = 0; i < bodyNum; i++)
        {
            auto& mi                = tetMesh.body_motor_infos[i];
            motor_params[i * 5 + 0] = mi.axis_x;
            motor_params[i * 5 + 1] = mi.axis_y;
            motor_params[i * 5 + 2] = mi.axis_z;
            motor_params[i * 5 + 3] = mi.speed;
            motor_params[i * 5 + 4] = mi.strength;
        }
        safe_copy(d_tetMesh.body_motor_params, motor_params.data(),
                  bodyNum * 5 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Collision exclusion matrix
    if(::g_gipc_log_level >= 1) printf("[CollisionExclusion] pairs=%d, collision_body_num=%d\n",
           (int)tetMesh.collision_exclusion_pairs.size(), d_tetMesh.collision_body_num);
    const bool have_groups = !tetMesh.body_groups.empty();
    if((!tetMesh.collision_exclusion_pairs.empty() || have_groups)
       && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_matrix(N * N, 0);
        for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
        {
            if(a >= 0 && a < N && b >= 0 && b < N)
            {
                host_matrix[a * N + b] = 1;
                host_matrix[b * N + a] = 1;
            }
        }
        // [multi-env] Exclude all cross-group body pairs (both groups >= 0).
        // Guarantees envs never interact regardless of spatial proximity; the
        // existing _is_collision_excluded reads this matrix in every narrow-phase
        // pair-build path, so cross-env candidates are dropped before buffering.
        int n_xgrp = 0;
        if(have_groups)
        {
            const auto& grp = tetMesh.body_groups;
            for(int i = 0; i < N; ++i)
            {
                int gi = (i < (int)grp.size()) ? grp[i] : -1;
                if(gi < 0) continue;
                for(int j = i + 1; j < N; ++j)
                {
                    int gj = (j < (int)grp.size()) ? grp[j] : -1;
                    if(gj < 0 || gj == gi) continue;
                    host_matrix[i * N + j] = 1;
                    host_matrix[j * N + i] = 1;
                    ++n_xgrp;
                }
            }
        }
        safe_copy(d_tetMesh.collision_skip_matrix, host_matrix.data(),
                  N * N * sizeof(int), cudaMemcpyHostToDevice);
        if(::g_gipc_log_level >= 1) printf("[CollisionExclusion] Uploaded %dx%d exclusion matrix (%d pairs + %d cross-group)\n",
               N, N, (int)tetMesh.collision_exclusion_pairs.size(), n_xgrp);
    }

    // [multi-env P2a] upload the group-id substrate (d_body_to_group +
    // d_point_to_group). Foundation for per-group contact segmentation (P2b) and
    // the per-env block-diagonal solve (P3). Default stays -1 (wildcard) when no
    // groups set -> single-env behaviour unchanged.
    if(have_groups && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> bg(N, -1);
        for(int i = 0; i < N && i < (int)tetMesh.body_groups.size(); ++i)
            bg[i] = tetMesh.body_groups[i];
        safe_copy(d_tetMesh.d_body_to_group, bg.data(),
                  N * sizeof(int), cudaMemcpyHostToDevice);
        const auto& pt2body = tetMesh.point_id_to_body_id;
        std::vector<int> pg(pt2body.size(), -1);
        std::map<int,int> grp_vcount;
        for(size_t v = 0; v < pt2body.size(); ++v)
        {
            int b = pt2body[v];
            int g = (b >= 0 && b < N) ? bg[b] : -1;
            pg[v] = g; grp_vcount[g]++;
        }
        safe_copy(d_tetMesh.d_point_to_group, pg.data(),
                  pg.size() * sizeof(int), cudaMemcpyHostToDevice);
        if(::g_gipc_log_level >= 1)
        {
            printf("[multi-env P2a] group substrate uploaded: %d bodies, %zu verts; per-group vert counts:",
                   N, pt2body.size());
            for(auto& [g, c] : grp_vcount) printf(" g%d=%d", g, c);
            printf("\n");
        }
        // [multi-env P2a] block-level group map (the per-env PCG-reduction key).
        // Block layout: [0, abd_body_num*4) ABD (block b -> body b/4); then
        // fem_point_num FEM blocks (block abd_dofs+j -> vertex fem_point_offset+j).
        const auto& ci = tetMesh.abd_fem_count_info;
        int abd_dofs = (int)ci.abd_body_num * 4;
        int nblk = abd_dofs + (int)ci.fem_point_num;
        std::vector<int> dg(nblk, -1);
        std::map<int,int> grp_bcount;
        for(int b = 0; b < abd_dofs; ++b)
        {
            int body = b / 4;
            int g = (body >= 0 && body < N) ? bg[body] : -1;
            dg[b] = g; grp_bcount[g]++;
        }
        for(int j = 0; j < (int)ci.fem_point_num; ++j)
        {
            int v = (int)ci.fem_point_offset + j;
            int g = (v >= 0 && v < (int)pg.size()) ? pg[v] : -1;
            dg[abd_dofs + j] = g; grp_bcount[g]++;
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_dof_to_group, nblk * sizeof(int)));
        safe_copy(d_tetMesh.d_dof_to_group, dg.data(), nblk * sizeof(int), cudaMemcpyHostToDevice);
        d_tetMesh.dof_block_count = nblk;
        if(::g_gipc_log_level >= 1)
        {
            printf("[multi-env P2a] dof_to_group: %d blocks (%d ABD + %d FEM); per-group block counts:",
                   nblk, abd_dofs, (int)ci.fem_point_num);
            for(auto& [g, c] : grp_bcount) printf(" g%d=%d", g, c);
            printf("\n");
        }
    }

    // Ground skip
    if(!tetMesh.ground_collision_skip_body_ids.empty() && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> host_flags(N, 0);
        for(int bid : tetMesh.ground_collision_skip_body_ids)
        {
            if(bid >= 0 && bid < N)
                host_flags[bid] = 1;
        }
        safe_copy(d_tetMesh.ground_skip_body, host_flags.data(),
                  N * sizeof(int), cudaMemcpyHostToDevice);
        ipc._ground_skip_body  = d_tetMesh.ground_skip_body;
        ipc._ground_body_count = N;
    }

    // BVH-skip optimization (audit/perf-bvh-skip-isolated): mark "isolated"
    // bodies via the diagonal of collision_skip_matrix. Body i is isolated iff
    // ground_skip_body[i]==1 AND all off-diagonal cells in row i are 1.
    // Kernels detect this with a single matrix lookup at thread entry.
    //
    // Toggle: BVHSKIP2=0 disables this (and #3, since #3 piggy-backs on the
    // same isolation flag). Use to measure #1-only baseline.
    const char* bvhskip2_env_local = std::getenv("BVHSKIP2");
    bool bvhskip2_enabled_local = (bvhskip2_env_local == nullptr) || (std::string(bvhskip2_env_local) != "0");
    if(!bvhskip2_enabled_local) {
        if(::g_gipc_log_level >= 1) printf("[BVHSkip#2] DISABLED via BVHSKIP2=0 (kernels see no isolated diag bits)\n");
    }
    if(bvhskip2_enabled_local && d_tetMesh.collision_body_num > 0)
    {
        int N = d_tetMesh.collision_body_num;
        std::vector<int> ground_flags(N, 0);
        for(int bid : tetMesh.ground_collision_skip_body_ids)
            if(bid >= 0 && bid < N) ground_flags[bid] = 1;
        std::vector<int> matrix(N * N, 0);
        for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
            if(a >= 0 && a < N && b >= 0 && b < N) {
                matrix[a*N+b] = 1; matrix[b*N+a] = 1;
            }
        int n_iso = 0;
        for(int i = 0; i < N; ++i) {
            if(!ground_flags[i]) continue;
            bool all_excluded = true;
            for(int j = 0; j < N; ++j) {
                if(i == j) continue;
                if(matrix[i*N+j] == 0) { all_excluded = false; break; }
            }
            if(all_excluded) {
                int one = 1;
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.collision_skip_matrix + i*N + i,
                                          &one, sizeof(int), cudaMemcpyHostToDevice));
                n_iso++;
            }
        }
        if(::g_gipc_log_level >= 1) printf("[BVHSkip] %d/%d bodies fully isolated → diag[i][i]=1 short-circuit set\n",
               n_iso, N);
    }

    // Stitch springs
    if(tetMesh.softNum > 0 && !d_tetMesh.stitch_paired_vertex.empty())
    {
        safe_copy(d_tetMesh.d_stitch_paired_vertex, d_tetMesh.stitch_paired_vertex.data(),
                  tetMesh.softNum * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_stitch_rest_offset, d_tetMesh.stitch_rest_offset.data(),
                  tetMesh.softNum * sizeof(double3), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_stitch_abd_body_id, d_tetMesh.stitch_abd_body_id.data(),
                  tetMesh.softNum * sizeof(int), cudaMemcpyHostToDevice);
        ipc.m_d_stitch_paired_vertex = d_tetMesh.d_stitch_paired_vertex;
        ipc.m_d_stitch_rest_offset   = d_tetMesh.d_stitch_rest_offset;
        ipc.m_d_stitch_abd_body_id   = d_tetMesh.d_stitch_abd_body_id;
    }

    // [FEM-pin / M1 substitution] Hard-constraint pin arrays.
    // Each pin: FEM vertex idx + ABD body id + local position in ABD rest frame.
    // The local position is computed from the world-frame offset given to
    // add_fem_pin_to_abd, transformed back through R_finalize^{-1}.
    int n_pins = static_cast<int>(tetMesh.fem_pin_fem_vertex.size());
    if(n_pins > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_fem_vertex,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_body_id,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_local_pos,
                                  n_pins * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_abd_anchor,
                                  n_pins * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.d_fem_pin_rest_offset,
                                  n_pins * sizeof(double3)));
        d_tetMesh.n_fem_pins = n_pins;

        safe_copy(d_tetMesh.d_fem_pin_fem_vertex, tetMesh.fem_pin_fem_vertex.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_fem_pin_abd_body_id, tetMesh.fem_pin_abd_body_id.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        // (abd_local_pos transform is done later in SimEngine::finalize after ABD q init)
        safe_copy(d_tetMesh.d_fem_pin_abd_anchor, tetMesh.fem_pin_abd_anchor.data(),
                  n_pins * sizeof(int), cudaMemcpyHostToDevice);
        safe_copy(d_tetMesh.d_fem_pin_rest_offset, tetMesh.fem_pin_rest_offset.data(),
                  n_pins * sizeof(double3), cudaMemcpyHostToDevice);
        printf("[FEM-pin] allocated %d hard-constraint pins (local_pos transform deferred to finalize)\n", n_pins);
    }
}

// ---------- BVH + solver init ----------
void SimEngine::Impl::do_init_bvh_and_solver()
{
    cudaSetDevice(cfg.cuda_device);
    cudaDeviceSynchronize();

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = static_cast<uint32_t>(tetMesh.surfVerts.size());
    ipc.surface_Num    = static_cast<uint32_t>(tetMesh.surface.size());
    ipc.edge_Num       = static_cast<uint32_t>(tetMesh.surfEdges.size());
    ipc.tri_edge_num   = static_cast<uint32_t>(tetMesh.tri_edges.size());

    if(ipc.m_skip_all_collision)
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM = 1;
        ipc.MAX_COLLITION_PAIRS_NUM     = 1;
    }
    else
    {
        ipc.MAX_CCD_COLLITION_PAIRS_NUM =
            static_cast<int>(
                1 * cfg.collision_detection_buff_scale
                * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
                   * std::max((ipc.IPC_dt / 0.01), 2.0)));
        ipc.MAX_COLLITION_PAIRS_NUM =
            static_cast<int>(
                (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                * 3 * cfg.collision_detection_buff_scale);
    }

    if(::g_gipc_log_level >= 1) printf("[SimEngine] collision_detection_buff_scale=%.1f  MAX_CCD_PAIRS=%d  MAX_PAIRS=%d\n",
           cfg.collision_detection_buff_scale,
           ipc.MAX_CCD_COLLITION_PAIRS_NUM,
           ipc.MAX_COLLITION_PAIRS_NUM);

    ipc.triangleNum = tetMesh.triangleNum;
    ipc.targetVert  = d_tetMesh.targetVert;
    ipc.targetInd   = d_tetMesh.targetIndex;
    ipc.softNum     = tetMesh.softNum;

    ipc.abd_fem_count_info    = tetMesh.abd_fem_count_info;
    ipc.num_joint_constraints = static_cast<int>(tetMesh.joint_constraints.size());

    std::cout << "[SimEngine] BVH init: verts=" << ipc.vertexNum
              << " surface=" << ipc.surface_Num
              << " edges=" << ipc.edge_Num
              << " surf_verts=" << ipc.surf_vertexNum
              << std::endl;
    std::cout.flush();

    ipc.MALLOC_DEVICE_MEM();

    if(ipc.surface_Num > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._faces, tetMesh.surface.data(),
                                  ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    if(ipc.edge_Num > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._edges, tetMesh.surfEdges.data(),
                                  ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    if(ipc.surf_vertexNum > 0)
        CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts, tetMesh.surfVerts.data(),
                                  ipc.surf_vertexNum * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // [multi-FEM-bodyid] hand the per-body FEM flag table to GIPC so the
    // narrow-phase / sanity-check kernels can distinguish FEM vs ABD
    // without the legacy "_bodyId == -1" sentinel. Must be set BEFORE
    // initBVH() so bvh_f/bvh_e read a non-null pointer.
    ipc._body_id_to_is_fem  = d_tetMesh.body_id_to_is_fem;
    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id,
                d_tetMesh.collision_skip_matrix, d_tetMesh.collision_body_num);
    ipc._point_body_id      = d_tetMesh.point_id_to_body_id;

    // BVH-skip #3 wiring is deferred until after ipc.init() — see "[BVHSkip#3-WIRE]" marker.

    // MAS preconditioner setup (must run even for pure-ABD scenes, matching gl_main.cu)
    if(ipc.pcg_data.P_type)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(
            ipc.vertexNum - tetMesh.abd_vertexOffset,
            tetMesh.abd_vertexOffset,
            neighborListSize,
            ipc._collisonPairs,
            tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;

        if(neighborListSize > 0)
        {
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                      tetMesh.neighborList.data(),
                                      neighborListSize * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                      tetMesh.neighborStart.data(),
                                      (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                      tetMesh.neighborNum.data(),
                                      (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
        }

        if(!tetMesh.partId_map_real.empty())
        {
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                      tetMesh.partId_map_real.data(),
                                      tetMesh.part_offset * BANKSIZE * sizeof(int),
                                      cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                      tetMesh.real_map_partId.data(),
                                      tetMesh.real_map_partId.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
        }

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    // Copy rest vertices
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

#ifdef USE_QUADRATIC_BENDING
    if(!tetMesh.tri_edges.empty())
    {
        std::vector<Eigen::Matrix4d> Q_host(tetMesh.tri_edges.size());
        std::vector<double3> rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  d_tetMesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tetMesh.tri_edges.data(),
                            tetMesh.tri_edges_adj_points.data(),
                            tetMesh.tri_edges.size(),
                            Q_host.data());
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.quad_bending_Q,
                                  Q_host.data(),
                                  tetMesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));
    }
#endif

    ipc.buildBVH();
    ipc.setup_surface_mesh_bodies(tetMesh);
    // [P1 triplet-margin] The internal-triplet margin (cfg.triplet_internal_margin,
    // default 32) only exists to reserve headroom for the M3.5 hybrid FEM-ABD pin
    // chain-rule Hessian expansion, which is gated by n_fem_pins>0 (see GIPC.cu M3.5
    // block). For scenes WITHOUT FEM pins the internal Hessian-triplet count is exact
    // (fixed by mesh topology), so margin=1 is sufficient and safe — and it avoids
    // over-allocating the global triplet buffer by up to 32x. Verified: cp-stats shows
    // "ext used 0 / cap 0" for non-hybrid scenes. Hybrid scenes keep the full margin.
    ipc.m_triplet_internal_margin =
        (d_tetMesh.n_fem_pins > 0) ? cfg.triplet_internal_margin : 1.0;
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer,
             cfg.linear_system_buff_scale);

    // [BVHSkip#3-WIRE] Wire _active_idx now that ipc.init() has captured the
    // full-scene bbox into bboxDiagSize2/dHat. Subsequent buildBVH() in step()
    // uses the indirect (filtered) path. (Wiring before ipc.init() shrinks the
    // scene bbox to active leaves only → dHat too small → cloth self-intersect.)
    //
    // Toggle: BVHSKIP3=0 disables (BVH still uses default path, only #1+#2 active).
    // Note: requires BVHSKIP2=1 — #3 reuses the isolation set from #2; if #2
    // is disabled there are no diag bits to read.
    {
        const char* bvhskip3_env = std::getenv("BVHSKIP3");
        bool bvhskip3_enabled = (bvhskip3_env == nullptr) || (std::string(bvhskip3_env) != "0");
        const char* bvhskip2_env = std::getenv("BVHSKIP2");
        bool bvhskip2_enabled = (bvhskip2_env == nullptr) || (std::string(bvhskip2_env) != "0");
        if(!bvhskip3_enabled) {
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] DISABLED via BVHSKIP3=0\n");
        } else if(!bvhskip2_enabled) {
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] AUTO-DISABLED (BVHSKIP2=0 — #3 requires #2's isolation set)\n");
        } else if(d_tetMesh.collision_body_num > 0
                  && (!tetMesh.collision_exclusion_pairs.empty()
                      || !tetMesh.ground_collision_skip_body_ids.empty()))
        {
            const int N = d_tetMesh.collision_body_num;
            std::vector<int> ground_flags(N, 0);
            for(int bid : tetMesh.ground_collision_skip_body_ids)
                if(bid >= 0 && bid < N) ground_flags[bid] = 1;
            std::vector<int> matrix(N * N, 0);
            for(auto& [a, b] : tetMesh.collision_exclusion_pairs)
                if(a >= 0 && a < N && b >= 0 && b < N) {
                    matrix[a*N+b] = 1; matrix[b*N+a] = 1;
                }
            std::vector<int> isolated(N, 0);
            int n_iso = 0;
            for(int i = 0; i < N; ++i) {
                if(!ground_flags[i]) continue;
                bool all_excl = true;
                for(int j = 0; j < N; ++j) {
                    if(i == j) continue;
                    if(matrix[i*N+j] == 0) { all_excl = false; break; }
                }
                if(all_excl) { isolated[i] = 1; n_iso++; }
            }
            // Drop face only if all 3 vertices belong to an isolated body.
            // Drop edge only if both endpoints belong to an isolated body.
            // Conservative: cross-body or cloth-touching primitives stay active.
            auto is_iso = [&](int B) { return (B >= 0 && B < N && isolated[B] != 0); };

            std::vector<int> active_face;
            active_face.reserve(tetMesh.surface.size());
            for(int f = 0; f < (int)tetMesh.surface.size(); ++f) {
                const auto& t = tetMesh.surface[f];
                int Bx = tetMesh.point_id_to_body_id[t.x];
                int By = tetMesh.point_id_to_body_id[t.y];
                int Bz = tetMesh.point_id_to_body_id[t.z];
                if(is_iso(Bx) && is_iso(By) && is_iso(Bz)) continue;
                active_face.push_back(f);
            }
            std::vector<int> active_edge;
            active_edge.reserve(tetMesh.surfEdges.size());
            for(int e = 0; e < (int)tetMesh.surfEdges.size(); ++e) {
                const auto& ed = tetMesh.surfEdges[e];
                int Bx = tetMesh.point_id_to_body_id[ed.x];
                int By = tetMesh.point_id_to_body_id[ed.y];
                if(is_iso(Bx) && is_iso(By)) continue;
                active_edge.push_back(e);
            }
            const int n_af = (int)active_face.size();
            const int n_ae = (int)active_edge.size();
            if(n_af > 0 && n_af < (int)tetMesh.surface.size()) {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.bvh_active_face_idx,
                                          n_af * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.bvh_active_face_idx,
                                          active_face.data(),
                                          n_af * sizeof(int),
                                          cudaMemcpyHostToDevice));
                d_tetMesh.bvh_active_face_num = n_af;
                ipc.bvh_f._active_idx         = d_tetMesh.bvh_active_face_idx;
                ipc.bvh_f.face_number_active  = n_af;
            }
            if(n_ae > 0 && n_ae < (int)tetMesh.surfEdges.size()) {
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_tetMesh.bvh_active_edge_idx,
                                          n_ae * sizeof(int)));
                CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.bvh_active_edge_idx,
                                          active_edge.data(),
                                          n_ae * sizeof(int),
                                          cudaMemcpyHostToDevice));
                d_tetMesh.bvh_active_edge_num = n_ae;
                ipc.bvh_e._active_idx         = d_tetMesh.bvh_active_edge_idx;
                ipc.bvh_e.face_number_active  = n_ae;
            }
            if(::g_gipc_log_level >= 1) printf("[BVHSkip#3] %d/%d isolated  active faces=%d/%zu  active edges=%d/%zu\n",
                   n_iso, N, n_af, tetMesh.surface.size(),
                   n_ae, tetMesh.surfEdges.size());
            // Re-build BVH so it transitions from default (full) to indirect
            // (active subset). Without this, the next buildCP() launches EE
            // self-query with N=active leaves but BVH leaves are still at the
            // default-path offset → reads stale internal nodes → illegal access.
            ipc.buildBVH();
        }
    }

    // Joint constraints
    if(!tetMesh.joint_constraints.empty() || !tetMesh.prismatic_constraints.empty())
        ipc.init_joint_constraints_from_mesh(tetMesh);

    // Build collision pairs + solver warm-start (mirrors gl_main.cu post-init)
    ipc.buildCP();
    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0;
    ipc.computeXTilta(d_tetMesh, 1);
    ipc.create_LinearSystem(d_tetMesh);
}

// ======================== finalize ========================
void SimEngine::finalize()
{
    auto& impl = *m_impl;
    impl.apply_config_to_ipc();

    // Build ABD system + linear system
    impl.ipc.build_gipc_system(impl.d_tetMesh);

    // ABD system parms
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.joint_strength_ratio            = impl.cfg.joint_strength_ratio;
        impl.ipc.m_abd_system->parms.revolute_driving_strength_ratio = impl.cfg.revolute_driving_strength_ratio;
        impl.ipc.m_abd_system->parms.prismatic_strength_ratio        = impl.cfg.prismatic_strength_ratio;
        impl.ipc.m_abd_system->parms.prismatic_driving_strength_ratio = impl.cfg.prismatic_driving_strength_ratio;
        impl.ipc.m_abd_system->parms.max_revolute_step_per_frame     = impl.cfg.max_revolute_step_per_frame;
        impl.ipc.m_abd_system->parms.max_prismatic_step_per_frame    = impl.cfg.max_prismatic_step_per_frame;
        impl.ipc.m_abd_system->parms.dt = impl.cfg.dt;
        impl.ipc.m_abd_system->parms.gravity = impl.cfg.gravity;
        impl.ipc.m_abd_system->parms.velocity_damping = impl.cfg.velocity_damping;
    }

    // Prepare metis dir
    std::string metis_dir = impl.resolved_assets_dir + "sorted_mesh/";
    std::filesystem::create_directories(metis_dir);

    impl.do_setMAS_partition();
    impl.tetMesh.getSurface();
    impl.do_initFEM();
    impl.do_upload_to_gpu();
    impl.do_init_bvh_and_solver();

    impl.finalized = true;
    std::cout << "[SimEngine] Finalized: "
              << impl.ipc.vertexNum << " verts, "
              << impl.ipc.surface_Num << " surface faces, "
              << impl.ipc.edge_Num << " edges" << std::endl;

    // [stitch sanity] Warn if soft_motion_rate (stitch spring stiffness) is
    // large compared to FEM Young modulus and the scene has many stitch
    // springs. The stitch Hessian diagonal contribution is
    //   H_stitch_total ≈ stitch_count * soft_motion_rate
    // and gets summed with the FEM elasticity Hessian (~Young per tet) +
    // IPC barrier Hessian (~kappa per contact). If H_stitch dominates by
    // 10x or more, the combined matrix's condition number can blow up
    // and PCG produces NaN. (Verified empirically on case_27_softgripper:
    // motionRate=1e6 + 130 stitch + Young=1e6 -> NaN at step ~131.)
    {
        int stitch_count = impl.tetMesh.softNum;
        double rate = impl.cfg.soft_motion_rate;
        // Average per-vertex Young modulus across all FEM vertices — this
        // is the actual elasticity stiffness the stitch is competing with,
        // unlike cfg.cloth_young_modulus which only applies to dim=2 cloth
        // FEM bodies and may be unset for tet (dim=3) FEM scenes.
        double young_avg = 0.0;
        const auto& yvec = impl.tetMesh.vert_youngth_modules;
        if(!yvec.empty())
        {
            double sum = 0.0;
            for(double y : yvec) sum += y;
            young_avg = sum / static_cast<double>(yvec.size());
        }
        if(stitch_count > 0 && rate > 0.0 && young_avg > 0.0)
        {
            double total_stitch_h = stitch_count * rate;
            double ratio = total_stitch_h / young_avg;
            // Empirical thresholds (case_27_softgripper, 130 stitch, FEM
            // young 1e6):
            //   ratio = 130    (motionRate=1e4) -> stable
            //   ratio = 1.3e4  (motionRate=1e6) -> NaN at step ~131
            // Set threshold = 1000 (~middle in log scale).
            if(ratio > 1000.0)
            {
                printf("\n[SimEngine] *** WARNING: stitch system may be too stiff ***\n");
                printf("[SimEngine]   stitch_count=%d  soft_motion_rate=%.1e  avg FEM Young=%.1e\n",
                       stitch_count, rate, young_avg);
                printf("[SimEngine]   stitch_count * soft_motion_rate / avg_young = %.1f (threshold = 1000)\n",
                       ratio);
                printf("[SimEngine]   Empirically, ratio > 1000 risks PCG NaN under aggressive joint trajectories.\n");
                printf("[SimEngine]   If you hit NaN, try reducing soft_motion_rate (e.g. /100) or raising\n");
                printf("[SimEngine]   per-mesh young_modulus, or run with NAN_DIAG=1 to confirm.\n\n");
                fflush(stdout);
            }
        }
    }

    // DIAG: dump q for first 3 bodies after full finalize
    if(getenv("STIFF_ABD_DBG"))
    {
        int nb = impl.ipc.abd_fem_count_info.abd_body_num;
        int n = std::min(nb, 3);
        if(n > 0 && impl.ipc.m_abd_sim_data)
        {
            using Vec12 = Eigen::Matrix<double, 12, 1>;
            std::vector<Vec12> dbg_q(nb);
            CUDA_SAFE_CALL(cudaMemcpy(dbg_q.data(),
                                      impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                                      nb * sizeof(Vec12), cudaMemcpyDeviceToHost));
            for(int b = 0; b < n; b++)
            {
                auto& q = dbg_q[b];
                std::cout << "[DIAG-CPP] After finalize: body=" << b
                          << " p=[" << q[0] << "," << q[1] << "," << q[2] << "]"
                          << " a1=[" << q[3] << "," << q[4] << "," << q[5] << "]"
                          << " a2=[" << q[6] << "," << q[7] << "," << q[8] << "]"
                          << " a3=[" << q[9] << "," << q[10] << "," << q[11] << "]"
                          << std::endl;
            }
        }
    }

    // [stitch local-frame fix] DISABLED — see commits e0990e6 / pre-substitution-method.
    impl.ipc.m_d_abd_body_q = nullptr;

    // [M1 substitution method] FEM pin transform world→local.
    // After ABD q is initialized, transform pinned FEM vertices' world rest
    // offset into the ABD body's REST frame, store as local_pos. Each step
    // the kernel does world_pos = q.t + R(q) * local_pos.
    // Also wire the ABD q pointer for the apply-pins kernel.
    if(impl.ipc.m_abd_sim_data && impl.d_tetMesh.n_fem_pins > 0)
    {
        int n_pins = impl.d_tetMesh.n_fem_pins;
        int nb     = impl.ipc.abd_fem_count_info.abd_body_num;

        // Read q from GPU
        using Vec12 = Eigen::Matrix<double, 12, 1>;
        std::vector<Vec12> host_q(nb);
        CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                                  impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                                  nb * sizeof(Vec12), cudaMemcpyDeviceToHost));

        // Pull anchor + body_id arrays, compute local_pos
        const auto& fem_v_vec   = impl.tetMesh.fem_pin_fem_vertex;
        const auto& bid_vec     = impl.tetMesh.fem_pin_abd_body_id;
        const auto& anchor_vec  = impl.tetMesh.fem_pin_abd_anchor;
        const auto& rest_w_vec  = impl.tetMesh.fem_pin_rest_offset;

        std::vector<double3> local_pos(n_pins);
        std::vector<double3> host_verts(impl.tetMesh.vertexNum);
        cudaMemcpy(host_verts.data(), impl.d_tetMesh.vertexes,
                   impl.tetMesh.vertexNum * sizeof(double3), cudaMemcpyDeviceToHost);

        for(int i = 0; i < n_pins; i++)
        {
            int bid = bid_vec[i];
            int av  = anchor_vec[i];
            // [Hybrid mesh] anchor == -1 sentinel: caller used
            // add_fem_pins_with_local_pos and provided local_pos directly
            // (already in ABD rest frame).  Pass through verbatim.
            if(av == -1)
            {
                local_pos[i] = impl.tetMesh.fem_pin_abd_local_pos[i];
                continue;
            }
            double3 fem_world = host_verts[fem_v_vec[i]];
            // Compute fem's position in ABD body's rest frame:
            //   world = q.t + R(q) * fem_local
            //   fem_local = R(q)^T * (world - q.t)   (assuming R orthogonal)
            const Vec12& q = host_q[bid];
            double3 d = make_double3(fem_world.x - q[0], fem_world.y - q[1], fem_world.z - q[2]);
            double3 lo;
            lo.x = q[3] * d.x + q[4]  * d.y + q[5]  * d.z;  // R^T row 1 = a1 (q[3..5])
            lo.y = q[6] * d.x + q[7]  * d.y + q[8]  * d.z;
            lo.z = q[9] * d.x + q[10] * d.y + q[11] * d.z;
            local_pos[i] = lo;
        }
        // Upload local_pos to GPU
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.d_fem_pin_abd_local_pos,
                                  local_pos.data(),
                                  n_pins * sizeof(double3), cudaMemcpyHostToDevice));

        // Wire ABD q pointer to GIPC for apply_fem_pins kernel
        impl.ipc.m_d_abd_body_q = reinterpret_cast<void*>(
            impl.ipc.m_abd_sim_data->device.body_id_to_q.data());

        // Mark pinned FEM vertices' BoundaryType = Fixed (=2). step_forward
        // kernel uses btype=0 check to update positions from PCG Δx; setting
        // it to 2 means the line-search position update is skipped, which
        // is what we want — apply_fem_pins kernel will write the correct
        // ABD-derived position right after step_forward.
        //
        // **NOT changing mass** — earlier we tried mass=1e30 to make PCG
        // naturally output Δx_pinned ≈ 0, but that made inertia energy
        // E_kin = ½ m v² explode (1e30 × 5mm² = 1e23) and broke line search.
        // The fix is at the IPC matrix level (M2 below): when assembling
        // the FEM elasticity / barrier / inertia Hessian, skip the
        // pinned vertex's row/col entirely so PCG sees them as
        // disconnected DOFs.
        std::vector<int> btype_host(impl.tetMesh.vertexNum);
        cudaMemcpy(btype_host.data(), impl.d_tetMesh.BoundaryType,
                   impl.tetMesh.vertexNum * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < n_pins; i++)
        {
            int v = fem_v_vec[i];
            btype_host[v] = 2;            // Fixed; PCG Δx update skipped
        }
        cudaMemcpy(impl.d_tetMesh.BoundaryType, btype_host.data(),
                   impl.tetMesh.vertexNum * sizeof(int), cudaMemcpyHostToDevice);

        // Build per-vertex pin map for O(1) lookup in elasticity kernels:
        // is_pinned_vertex[v] = 1 if v is a pinned FEM vertex, 0 otherwise.
        // Used in M2 to skip writing pinned row/col to the FEM Hessian.
        std::vector<int> pinned_mask(impl.tetMesh.vertexNum, 0);
        for(int i = 0; i < n_pins; i++)
            pinned_mask[fem_v_vec[i]] = 1;
        if(impl.d_tetMesh.is_pinned_vertex == nullptr)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.is_pinned_vertex,
                                      impl.tetMesh.vertexNum * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.is_pinned_vertex,
                                  pinned_mask.data(),
                                  impl.tetMesh.vertexNum * sizeof(int),
                                  cudaMemcpyHostToDevice));
        // wire the mask into GIPC for kernel access
        impl.ipc.m_d_is_pinned_vertex = impl.d_tetMesh.is_pinned_vertex;

        // [M3.5] Build vertex_to_pin_idx (size = vertexNum) for O(1)
        // lookup of (body_id, lo) given a vertex index.  -1 = not pinned.
        std::vector<int> v2pin_host(impl.tetMesh.vertexNum, -1);
        for(int i = 0; i < n_pins; i++)
            v2pin_host[fem_v_vec[i]] = i;
        if(impl.d_tetMesh.vertex_to_pin_idx == nullptr)
        {
            CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.vertex_to_pin_idx,
                                      impl.tetMesh.vertexNum * sizeof(int)));
        }
        CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.vertex_to_pin_idx,
                                  v2pin_host.data(),
                                  impl.tetMesh.vertexNum * sizeof(int),
                                  cudaMemcpyHostToDevice));

        // [Hybrid mesh] populate d_tet_to_abd_body (per-tet body assignment).
        // A tet whose 4 verts are ALL pinned to the SAME ABD body is rigid-
        // internal: its Green strain is zero for any rigid motion of the
        // body, so its FEM elasticity is structurally redundant w.r.t. the
        // ABD body's own energy.  Marking it here lets Phase 4's elasticity
        // kernel early-exit, saving a co-rotational SVD per such tet per
        // Newton iter.  Computed once at finalize since pin info is static.
        {
            const auto& tets    = impl.tetMesh.tetrahedras;       // vector<uint4>
            const auto& body_id = impl.tetMesh.fem_pin_abd_body_id; // vector<int>
            std::vector<int> tet_to_abd(impl.tetMesh.tetrahedraNum, -1);
            int n_rigid_tets = 0;
            for(int t = 0; t < impl.tetMesh.tetrahedraNum; ++t)
            {
                const uint4 vs = tets[t];
                const int p0 = v2pin_host[vs.x];
                const int p1 = v2pin_host[vs.y];
                const int p2 = v2pin_host[vs.z];
                const int p3 = v2pin_host[vs.w];
                if(p0 < 0 || p1 < 0 || p2 < 0 || p3 < 0) continue;
                const int b0 = body_id[p0];
                if(body_id[p1] != b0 || body_id[p2] != b0 || body_id[p3] != b0)
                    continue;
                tet_to_abd[t] = b0;
                ++n_rigid_tets;
            }
            if(impl.d_tetMesh.d_tet_to_abd_body == nullptr)
            {
                CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.d_tet_to_abd_body,
                                          impl.tetMesh.tetrahedraNum * sizeof(int)));
            }
            CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.d_tet_to_abd_body,
                                      tet_to_abd.data(),
                                      impl.tetMesh.tetrahedraNum * sizeof(int),
                                      cudaMemcpyHostToDevice));
            if(n_rigid_tets > 0)
            {
                printf("[Hybrid] %d / %d tets are rigid-internal "
                       "(all 4 verts pinned to same body) — Phase 4 will "
                       "skip their elasticity\n",
                       n_rigid_tets, impl.tetMesh.tetrahedraNum);
            }
        }

        printf("[M1+M2+M3.5] %d FEM pins: local_pos transformed, btype=Fixed, "
               "is_pinned_vertex mask + vertex_to_pin_idx uploaded\n", n_pins);
    }
}

// ======================== step ========================
// [NAN_DIAG] Per-step diagnostic dump (env-gated). Pulls FEM tet volumes,
// vertex velocities and positions to host; reports min/max + NaN counts.
// Useful for pinning down whether NaN is born from tet inversion,
// stitch-spring blowup, kappa overflow, or PCG numerical breakdown.
//
// Activate: NAN_DIAG=1 ./run examples/...
//
// Overhead: ~ vertexNum * 48 bytes D->H copy per step, only when enabled.
namespace {
struct NanDiagState {
    bool enabled = false;
    bool initialized = false;
    int  step_count = 0;
    bool nan_seen   = false;

    void init() {
        if(initialized) return;
        const char* e = std::getenv("NAN_DIAG");
        enabled = (e != nullptr && std::string(e) != "0");
        initialized = true;
        if(enabled) printf("[NAN_DIAG] enabled (env NAN_DIAG=1)\n");
    }
};
static NanDiagState g_diag;

// [NaN-sentinel] opt-in NaN watchdog. One device int gets atomicCAS'd if
// any vertex is NaN/Inf; first occurrence triggers a human-readable
// warning.
//
// **Default OFF** — measured ~163 ms/step overhead on case_27_softgripper
// (12k verts), which is ~150% of a normal 100ms step. The cost comes
// from the synchronous cudaMemcpy(4B, D->H) breaking GPU pipeline
// overlap with the IPC solver. Activate only when debugging NaN:
//
//     NAN_SENTINEL=1 ./run examples/...
//
// Or set NAN_DIAG=1 (which is even more verbose, also opt-in).
struct NanSentinelState {
    int* d_flag = nullptr;
    bool warned = false;
    bool enabled = false;
    bool initialized = false;
    int  step_count = 0;
    void init() {
        if(initialized) return;
        const char* e = std::getenv("NAN_SENTINEL");
        enabled = (e != nullptr && std::string(e) != "0");
        // NAN_DIAG implies NAN_SENTINEL — the diagnostic dump already
        // pulls vertex data, may as well surface a clear warning too.
        const char* diag = std::getenv("NAN_DIAG");
        if(diag != nullptr && std::string(diag) != "0") enabled = true;
        initialized = true;
        if(enabled) printf("[NaN-SENTINEL] enabled (env NAN_SENTINEL=1 or NAN_DIAG=1)\n");
    }
    void ensure_buffer() {
        if(d_flag == nullptr) {
            cudaMalloc(&d_flag, sizeof(int));
        }
    }
};
static NanSentinelState g_sentinel;
}  // namespace

__global__ static void _nan_sentinel_kernel(const double3* verts,
                                            const double3* velocities,
                                            int n,
                                            int* out_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double3 p = verts[idx];
    double3 v = velocities[idx];
    bool bad = isnan(p.x) || isnan(p.y) || isnan(p.z)
               || isinf(p.x) || isinf(p.y) || isinf(p.z)
               || isnan(v.x) || isnan(v.y) || isnan(v.z)
               || isinf(v.x) || isinf(v.y) || isinf(v.z);
    if(bad) atomicCAS(out_flag, 0, 1);
}

static void check_nan_sentinel_(int n_v,
                                const double3* d_vertexes,
                                const double3* d_velocities)
{
    g_sentinel.init();
    if(!g_sentinel.enabled) return;
    g_sentinel.ensure_buffer();
    if(n_v <= 0 || g_sentinel.d_flag == nullptr) return;
    int sc = g_sentinel.step_count++;

    cudaMemset(g_sentinel.d_flag, 0, sizeof(int));
    int blocks = (n_v + 255) / 256;
    _nan_sentinel_kernel<<<blocks, 256>>>(d_vertexes,
                                          d_velocities,
                                          n_v,
                                          g_sentinel.d_flag);
    int h_flag = 0;
    cudaMemcpy(&h_flag, g_sentinel.d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_flag != 0 && !g_sentinel.warned) {
        g_sentinel.warned = true;
        printf("\n========================================================================\n");
        printf("[NaN-SENTINEL] *** NaN/Inf detected in vertex positions or velocities\n");
        printf("[NaN-SENTINEL] *** at step %d. Physics has DIVERGED — subsequent steps\n", sc);
        printf("[NaN-SENTINEL] *** will be garbage and the engine cannot self-recover.\n");
        printf("[NaN-SENTINEL] *** Most common causes:\n");
        printf("[NaN-SENTINEL] ***   1. soft_motion_rate too high vs FEM Young modulus\n");
        printf("[NaN-SENTINEL] ***      (causes Hessian condition number blowup -> PCG NaN)\n");
        printf("[NaN-SENTINEL] ***   2. dt too large for the prescribed joint speed\n");
        printf("[NaN-SENTINEL] ***   3. FEM tet inverted (collision-driven over-compression)\n");
        printf("[NaN-SENTINEL] *** Re-run with NAN_DIAG=1 to see per-step min(tet_vol),\n");
        printf("[NaN-SENTINEL] *** max|v|, max|p|, NaN count + body_id breakdown.\n");
        printf("========================================================================\n\n");
        fflush(stdout);
    }
}

static void dump_nan_diagnostics_(int n_tet, int n_v,
                                  const double* d_volum,
                                  const double3* d_velocities,
                                  const double3* d_vertexes,
                                  const std::vector<int>& point_id_to_body_id)
{
    g_diag.init();
    if(!g_diag.enabled) return;
    int sc = g_diag.step_count++;

    // ---- min(tet_vol) — H1 tet-inverted detector ----
    double min_vol = 0.0;
    int    n_neg_vol = 0;
    if(n_tet > 0)
    {
        std::vector<double> host_vol(n_tet);
        cudaMemcpy(host_vol.data(), d_volum,
                   n_tet * sizeof(double), cudaMemcpyDeviceToHost);
        min_vol = *std::min_element(host_vol.begin(), host_vol.end());
        for(double v : host_vol) if(v < 0.0) ++n_neg_vol;
    }

    // ---- velocities + positions — H2/H4 detectors + NaN tracker ----
    std::vector<double3> host_v(n_v), host_p(n_v);
    cudaMemcpy(host_v.data(), d_velocities,
               n_v * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_p.data(), d_vertexes,
               n_v * sizeof(double3), cudaMemcpyDeviceToHost);

    double max_v2 = 0.0, max_p2 = 0.0;
    int n_nan_v = 0, n_nan_p = 0;
    int first_nan_v = -1, first_nan_p = -1;
    for(int i = 0; i < n_v; i++)
    {
        const double3& v = host_v[i];
        const double3& p = host_p[i];
        bool vn = (std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z)
                   || std::isinf(v.x) || std::isinf(v.y) || std::isinf(v.z));
        bool pn = (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)
                   || std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z));
        if(vn) { ++n_nan_v; if(first_nan_v < 0) first_nan_v = i; }
        if(pn) { ++n_nan_p; if(first_nan_p < 0) first_nan_p = i; }
        if(!vn) {
            double s = v.x*v.x + v.y*v.y + v.z*v.z;
            if(s > max_v2) max_v2 = s;
        }
        if(!pn) {
            double s = p.x*p.x + p.y*p.y + p.z*p.z;
            if(s > max_p2) max_p2 = s;
        }
    }
    double max_v = std::sqrt(max_v2);
    double max_p = std::sqrt(max_p2);

    bool first_nan_step = (n_nan_p + n_nan_v > 0) && !g_diag.nan_seen;
    if(first_nan_step) g_diag.nan_seen = true;

    printf("[NAN_DIAG] step=%4d  min_tet_vol=%+10.3e  neg_vol=%4d  "
           "max|v|=%9.3e  max|p|=%9.3e  nan_v=%4d  nan_p=%4d%s\n",
           sc, min_vol, n_neg_vol, max_v, max_p, n_nan_v, n_nan_p,
           first_nan_step ? "  <-- FIRST NaN HERE" : "");
    if(first_nan_step) {
        if(first_nan_p >= 0) {
            int bid = (first_nan_p < (int)point_id_to_body_id.size())
                      ? point_id_to_body_id[first_nan_p] : -2;
            printf("[NAN_DIAG]   first NaN position vertex idx=%d body_id=%d\n",
                   first_nan_p, bid);
        }
        if(first_nan_v >= 0) {
            int bid = (first_nan_v < (int)point_id_to_body_id.size())
                      ? point_id_to_body_id[first_nan_v] : -2;
            printf("[NAN_DIAG]   first NaN velocity vertex idx=%d body_id=%d\n",
                   first_nan_v, bid);
        }
    }
    fflush(stdout);
}

void SimEngine::step()
{
    auto& impl = *m_impl;
    cudaSetDevice(impl.cfg.cuda_device);

    if(!impl.tetMesh.joint_angle_controls.empty()
       || !impl.tetMesh.prismatic_drive_controls.empty())
    {
        impl.ipc.update_joint_angle_targets_from_mesh(impl.tetMesh);
    }

    impl.ipc.IPC_Solver(impl.d_tetMesh);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    impl.step_count++;

    // [NaN-sentinel] always-on lightweight NaN watchdog (~1 atomic int +
    // 4-byte D->H copy per step). Prints a one-time warning at first
    // occurrence so users notice silent physics divergence.
    check_nan_sentinel_(impl.tetMesh.vertexNum,
                        impl.d_tetMesh.vertexes,
                        impl.d_tetMesh.velocities);

    // [NAN_DIAG] env-gated — only runs when NAN_DIAG=1.
    dump_nan_diagnostics_(impl.tetMesh.tetrahedraNum,
                          impl.tetMesh.vertexNum,
                          impl.d_tetMesh.volum,
                          impl.d_tetMesh.velocities,
                          impl.d_tetMesh.vertexes,
                          impl.tetMesh.point_id_to_body_id);
}

// ======================== state queries ========================
int SimEngine::get_vertex_count() const
{
    return m_impl->ipc.vertexNum;
}

uintptr_t SimEngine::get_vertices_device_ptr() const
{
    // [gpu-direct] Raw device pointer to the global vertex buffer (double3*,
    // length vertexNum). Lets an external GPU framework (Warp) read FEM vertex
    // positions straight from device memory without a host round-trip. Valid
    // after finalize(); contents update after each step().
    return reinterpret_cast<uintptr_t>(m_impl->ipc._vertexes);
}

int SimEngine::get_surface_face_count() const
{
    return static_cast<int>(m_impl->tetMesh.surface.size());
}

int SimEngine::get_surface_vertex_count() const
{
    return static_cast<int>(m_impl->tetMesh.surfVerts.size());
}

void SimEngine::get_vertex_positions(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] Transparent unscramble. perm[i] = j means engine-internal
    // vertex i corresponds to input-mesh vertex j. We want user-facing
    // output in input order: out[j] = engine_pos[i] for each i.
    // If perm is empty / mismatched, fall back to identity (raw copy).
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(out_xyz, m_impl->ipc._vertexes,
                                  n * sizeof(double3), cudaMemcpyDeviceToHost));
        return;
    }
    // Read engine-order vertices to a temp buffer, then permute into out.
    std::vector<double3> tmp(n);
    CUDA_SAFE_CALL(cudaMemcpy(tmp.data(), m_impl->ipc._vertexes,
                              n * sizeof(double3), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            // Defensive: out-of-range perm entry, fall back to identity for this slot.
            out_xyz[3 * i + 0] = tmp[i].x;
            out_xyz[3 * i + 1] = tmp[i].y;
            out_xyz[3 * i + 2] = tmp[i].z;
            continue;
        }
        out_xyz[3 * j + 0] = tmp[i].x;
        out_xyz[3 * j + 1] = tmp[i].y;
        out_xyz[3 * j + 2] = tmp[i].z;
    }
}

void SimEngine::get_surface_faces(uint32_t* out_idx, int face_count) const
{
    int n = std::min(face_count, static_cast<int>(m_impl->tetMesh.surface.size()));
    if(n <= 0) return;

    // [MAS-perm] Map engine face indices (a, b, c) to input-mesh indices via perm.
    // Output triangle vertices reference vertex_metis_to_input[engine_idx].
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty()
                    && static_cast<int>(perm.size()) >= m_impl->ipc.vertexNum;
    if(!use_perm)
    {
        std::memcpy(out_idx, m_impl->tetMesh.surface.data(), n * sizeof(uint3));
        return;
    }
    const auto& surf = m_impl->tetMesh.surface;
    for(int i = 0; i < n; i++)
    {
        const uint3& f = surf[i];
        out_idx[3 * i + 0] = static_cast<uint32_t>(perm[f.x]);
        out_idx[3 * i + 1] = static_cast<uint32_t>(perm[f.y]);
        out_idx[3 * i + 2] = static_cast<uint32_t>(perm[f.z]);
    }
}

void SimEngine::get_surface_vertex_indices(uint32_t* out_idx, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->tetMesh.surfVerts.size()));
    if(n <= 0) return;

    // [MAS-perm] surfVerts stores engine-order vertex indices; translate to
    // input-order via perm so user-facing indexing is consistent with
    // get_vertices() / get_surface_faces().
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty()
                    && static_cast<int>(perm.size()) >= m_impl->ipc.vertexNum;
    if(!use_perm)
    {
        std::memcpy(out_idx, m_impl->tetMesh.surfVerts.data(), n * sizeof(uint32_t));
        return;
    }
    const auto& sv = m_impl->tetMesh.surfVerts;
    for(int i = 0; i < n; i++)
        out_idx[i] = static_cast<uint32_t>(perm[sv[i]]);
}

// ======================== joint control ========================
int SimEngine::get_num_revolute_joints() const
{
    return static_cast<int>(m_impl->tetMesh.joint_angle_controls.size());
}

int SimEngine::get_num_prismatic_joints() const
{
    return static_cast<int>(m_impl->tetMesh.prismatic_drive_controls.size());
}

JointInfo SimEngine::get_revolute_joint_info(int idx) const
{
    const auto& ctrl = m_impl->tetMesh.joint_angle_controls.at(idx);
    return JointInfo{
        ctrl.joint_name,
        ctrl.lower_limit,
        ctrl.upper_limit,
        ctrl.target_angle,
        ctrl.strength_ratio,
        false
    };
}

JointInfo SimEngine::get_prismatic_joint_info(int idx) const
{
    const auto& ctrl = m_impl->tetMesh.prismatic_drive_controls.at(idx);
    return JointInfo{
        ctrl.joint_name,
        ctrl.lower_limit,
        ctrl.upper_limit,
        ctrl.target_distance,
        ctrl.strength_ratio,
        true
    };
}

void SimEngine::set_revolute_target(int idx, double angle_rad)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).target_angle = angle_rad;
}

void SimEngine::set_revolute_torque(int idx, double torque)
{
    // [force-control] Torque control on a revolute driving joint. Adds the
    // generalized force -tau*dtheta/dq to the driving gradient (no Hessian),
    // independent of the PD term. For PURE torque control, also call
    // set_revolute_strength(idx, 0). Synced to GPU each step via
    // update_revolute_driving_targets. Mirrors libuipc external joint torque.
    m_impl->tetMesh.joint_angle_controls.at(idx).ext_torque = torque;
}

void SimEngine::set_revolute_initial_offset(int idx, double offset_rad)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).initial_angle_offset = offset_rad;
}

void SimEngine::set_prismatic_target(int idx, double distance_m)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance = distance_m;
}

void SimEngine::set_prismatic_force(int idx, double force)
{
    // [force-control] External force (N) along a prismatic driving joint's axis.
    // Pushes child along +axis (parent gets the reaction) via the q_tilde path
    // (no Hessian), independent of the PD term. For PURE force control, also
    // call set_prismatic_strength(idx, 0). Synced to GPU each step via
    // update_prismatic_driving_targets. Mirrors libuipc external prismatic force.
    m_impl->tetMesh.prismatic_drive_controls.at(idx).ext_force = force;
}

void SimEngine::set_revolute_strength(int idx, double strength)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).strength_ratio = strength;
}

void SimEngine::set_prismatic_strength(int idx, double strength)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).strength_ratio = strength;
}

void SimEngine::set_fixed_joint_strength(int idx, double kappa)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system) {
        std::cerr << "[set_fixed_joint_strength] sim not finalized" << std::endl;
        return;
    }
    auto& abd_sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= abd_sys.m_num_joints) {
        std::cerr << "[set_fixed_joint_strength] idx " << idx
                  << " out of range [0," << abd_sys.m_num_joints << ")" << std::endl;
        return;
    }

    // Read current GPU data for this joint, modify kappa, write back.
    JointConstraintGPUData host_jd;
    CUDA_SAFE_CALL(cudaMemcpy(&host_jd,
                              abd_sys.m_joint_data.data() + idx,
                              sizeof(JointConstraintGPUData),
                              cudaMemcpyDeviceToHost));
    host_jd.kappa = static_cast<Float>(kappa);
    CUDA_SAFE_CALL(cudaMemcpy(abd_sys.m_joint_data.data() + idx,
                              &host_jd,
                              sizeof(JointConstraintGPUData),
                              cudaMemcpyHostToDevice));
    std::cout << "[set_fixed_joint_strength] joint #" << idx
              << " kappa = " << kappa << std::endl;
}

void SimEngine::set_body_animated_target(int body_id,
                                         double target_x, double target_y, double target_z,
                                         double strength)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(body_id < 0 || body_id >= num_bodies) return;

    if(impl.d_tetMesh.body_motor_params == nullptr) {
        // Buffer wasn't allocated (no motor_infos at finalize time).  Allocate
        // it here so per-step Animated drive can be set on bodies that
        // weren't pre-registered with body_motor_infos.
        size_t bytes = (size_t)num_bodies * 5 * sizeof(double);
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_tetMesh.body_motor_params, bytes));
        CUDA_SAFE_CALL(cudaMemset(impl.d_tetMesh.body_motor_params, 0, bytes));
    }

    double host_params[5] = {target_x, target_y, target_z, strength, 0.0};
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.body_motor_params + body_id * 5,
                              host_params,
                              5 * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void SimEngine::set_urdf_mesh_override(const std::string& link_name,
                                       const std::string& msh_path,
                                       double young_modulus)
{
    m_impl->pending_urdf_mesh_overrides[link_name] = {msh_path, young_modulus};
}

Eigen::Matrix4d SimEngine::get_urdf_link_transform(const std::string& link_name) const
{
    auto it = m_impl->urdf_link_transforms.find(link_name);
    if(it == m_impl->urdf_link_transforms.end()) {
        std::cerr << "[get_urdf_link_transform] link '" << link_name
                  << "' not found (URDF not loaded or wrong name)" << std::endl;
        return Eigen::Matrix4d::Identity();
    }
    return it->second;
}

void SimEngine::set_abd_body_density(int body_id, double density)
{
    // Override one ABD body's density (mass = density * volume). Must be called
    // AFTER the body is loaded and BEFORE finalize(). The ABDSystem does not
    // exist yet at this point (it is created inside build_gipc_system during
    // finalize), so we stash the override on GIPC; build_gipc_system transfers
    // it to the ABDSystem right before the per-body mass setup runs.
    m_impl->ipc.m_pending_abd_density[body_id] = density;
}

void SimEngine::set_abd_body_inertia(int body_id, double mass,
                                     const double* com3, const double* inertia9)
{
    // Override one ABD body's inertial properties (mass, COM, 3x3 inertia about
    // the COM) — e.g. from URDF inertial tags via Newton body_mass/body_com/
    // body_inertia — instead of deriving them from the welded collision mesh,
    // whose centroid can be far off for a multi-shape link and skew the joint
    // driving torque. Stashed on GIPC (ABDSystem not built yet); transferred at
    // finalize. com/inertia are in the SAME world/load frame as the mesh verts.
    GIPC::PendingInertia pi;
    pi.mass = mass;
    for(int i = 0; i < 3; i++) pi.com[i] = com3[i];
    for(int i = 0; i < 9; i++) pi.inertia[i] = inertia9[i];
    m_impl->ipc.m_pending_abd_inertia[body_id] = pi;
}

void SimEngine::set_body_external_force(int body_id,
                                       double fx, double fy, double fz)
{
    // [force-control] Set the per-body external LINEAR force (N) on an ABD body.
    // Persistent until changed; pass (0,0,0) to clear. Enters the sim as an
    // acceleration M^{-1}F in q_tilde (cal_q_tilde.cu) — same path as gravity,
    // mirroring libuipc AffineBodyExternalBodyForce. (Prototype: linear only;
    // the buffer is a full 12-DOF wrench, so an affine/torque term can be added
    // by writing components [3:12].)
    auto& impl  = *m_impl;
    int   n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);
    if(body_id < 0 || body_id >= n_abd)
    {
        std::cerr << "[set_body_external_force] body_id " << body_id
                  << " is not an ABD body (n_abd=" << n_abd << ")" << std::endl;
        return;
    }
    if(!impl.ipc.m_abd_sim_data)
    {
        std::cerr << "[set_body_external_force] sim not finalized" << std::endl;
        return;
    }
    auto& f_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_ext_force;
    if(static_cast<int>(f_buf.size()) <= body_id)
    {
        std::cerr << "[set_body_external_force] body_id " << body_id
                  << " out of range (size=" << f_buf.size() << ")" << std::endl;
        return;
    }
    Eigen::Matrix<double, 12, 1> F = Eigen::Matrix<double, 12, 1>::Zero();
    F[0] = fx; F[1] = fy; F[2] = fz;
    CUDA_SAFE_CALL(cudaMemcpy(f_buf.data() + body_id, F.data(),
                              sizeof(double) * 12, cudaMemcpyHostToDevice));
}

void SimEngine::set_body_external_wrench(int body_id, const double* w12)
{
    // [force-control] Set the FULL 12-DOF external generalized force on an ABD
    // body: w[0:3] = linear force, w[3:12] = affine force (row-major vec(F_A)).
    // An affine wrench with w[5]=+omega, w[9]=-omega is a torque about Y (the
    // skew-symmetric part spins the body), mirroring libuipc's body-force test
    // which combines an orbiting linear force with a spinning affine term.
    auto& impl  = *m_impl;
    int   n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);
    if(body_id < 0 || body_id >= n_abd)
    {
        std::cerr << "[set_body_external_wrench] body_id " << body_id
                  << " is not an ABD body (n_abd=" << n_abd << ")" << std::endl;
        return;
    }
    if(!impl.ipc.m_abd_sim_data)
    {
        std::cerr << "[set_body_external_wrench] sim not finalized" << std::endl;
        return;
    }
    auto& f_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_ext_force;
    if(static_cast<int>(f_buf.size()) <= body_id)
    {
        std::cerr << "[set_body_external_wrench] body_id " << body_id
                  << " out of range (size=" << f_buf.size() << ")" << std::endl;
        return;
    }
    CUDA_SAFE_CALL(cudaMemcpy(f_buf.data() + body_id, w12,
                              sizeof(double) * 12, cudaMemcpyHostToDevice));
}

void SimEngine::set_body_apply_gravity(int body_id, bool enabled)
{
    auto& impl = *m_impl;
    int n_abd = static_cast<int>(impl.tetMesh.abd_fem_count_info.abd_body_num);

    if(body_id < n_abd)
    {
        // ABD body: gravity is precomputed at finalize as a 12-DOF
        // body_id_to_abd_gravity[i] vector. apply_gravity[] (per-vertex) is
        // ignored for ABD verts. To toggle, zero/restore the cached vector.
        if(!impl.ipc.m_abd_sim_data) {
            std::cerr << "[set_body_apply_gravity] sim not finalized" << std::endl;
            return;
        }

        using Vec12 = Eigen::Matrix<double, 12, 1>;
        auto& g_buf = impl.ipc.m_abd_sim_data->device.body_id_to_abd_gravity;
        if(static_cast<int>(g_buf.size()) <= body_id) {
            std::cerr << "[set_body_apply_gravity] ABD body_id " << body_id
                      << " out of range (size=" << g_buf.size() << ")" << std::endl;
            return;
        }

        if(!enabled) {
            // Cache current gravity (so we can restore it).
            Vec12 host_g = Vec12::Zero();
            CUDA_SAFE_CALL(cudaMemcpy(host_g.data(),
                                      g_buf.data() + body_id,
                                      sizeof(Vec12), cudaMemcpyDeviceToHost));
            impl.disabled_abd_gravity_cache[body_id] = host_g;

            Vec12 zero = Vec12::Zero();
            CUDA_SAFE_CALL(cudaMemcpy(g_buf.data() + body_id,
                                      zero.data(),
                                      sizeof(Vec12), cudaMemcpyHostToDevice));
        } else {
            // Restore from cache if we previously disabled it
            auto it = impl.disabled_abd_gravity_cache.find(body_id);
            if(it != impl.disabled_abd_gravity_cache.end()) {
                CUDA_SAFE_CALL(cudaMemcpy(g_buf.data() + body_id,
                                          it->second.data(),
                                          sizeof(Vec12), cudaMemcpyHostToDevice));
                impl.disabled_abd_gravity_cache.erase(it);
            }
        }
        return;
    }

    // FEM body: per-vertex apply_gravity[] flag
    const auto& pt2body = impl.tetMesh.point_id_to_body_id;
    if(pt2body.empty() || impl.d_tetMesh.apply_gravity == nullptr) return;

    int v_start = -1, v_end = -1;
    int N = static_cast<int>(pt2body.size());
    for(int i = 0; i < N; i++)
    {
        if(pt2body[i] == body_id)
        {
            if(v_start < 0) v_start = i;
            v_end = i + 1;
        }
    }
    if(v_start < 0) {
        std::cerr << "[set_body_apply_gravity] body_id " << body_id
                  << " has no vertices" << std::endl;
        return;
    }

    int n = v_end - v_start;
    std::vector<int> host_flags(n, enabled ? 1 : 0);
    CUDA_SAFE_CALL(cudaMemcpy(impl.d_tetMesh.apply_gravity + v_start,
                              host_flags.data(),
                              n * sizeof(int),
                              cudaMemcpyHostToDevice));
    for(int i = v_start; i < v_end; i++)
        impl.tetMesh.apply_gravity[i] = enabled ? 1 : 0;
}

void SimEngine::set_max_revolute_step_per_frame(double rad)
{
    auto& impl = *m_impl;
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.max_revolute_step_per_frame = rad;
    }
}

void SimEngine::set_max_prismatic_step_per_frame(double m)
{
    auto& impl = *m_impl;
    if(impl.ipc.m_abd_system)
    {
        impl.ipc.m_abd_system->parms.max_prismatic_step_per_frame = m;
    }
}

double SimEngine::get_revolute_target(int idx) const
{
    return m_impl->tetMesh.joint_angle_controls.at(idx).target_angle;
}

double SimEngine::get_prismatic_target(int idx) const
{
    return m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance;
}

double SimEngine::get_prismatic_drive_force(int idx) const
{
    // [force-control] Current prismatic DRIVING force = K*(target - d), where
    // d = (Cq - Cp).dot(t) is the current opening along the joint axis. Used by
    // force-limited position control: advance the target toward closed until
    // this force reaches the desired F_max, then hold (real-gripper behavior).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
        return 0.0;
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving)
        return 0.0;

    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    Vec12 qp, qc;
    CUDA_SAFE_CALL(cudaMemcpy(&qp, q_buf.data() + drv.parent_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&qc, q_buf.data() + drv.child_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));

    auto worldpt = [](const Vec12& q, const Vector3& xb) -> Vector3 {
        Matrix3x3 A;
        A.row(0) = q.segment<3>(3).transpose();
        A.row(1) = q.segment<3>(6).transpose();
        A.row(2) = q.segment<3>(9).transpose();
        return Vector3(q.segment<3>(0) + A * xb);
    };
    Matrix3x3 Ac;
    Ac.row(0) = qc.segment<3>(3).transpose();
    Ac.row(1) = qc.segment<3>(6).transpose();
    Ac.row(2) = qc.segment<3>(9).transpose();

    Vector3 Cp = worldpt(qp, drv.Cp_bar);
    Vector3 Cq = worldpt(qc, drv.Cq_bar);
    Vector3 t  = Ac * drv.tq_bar;
    double  d  = (Cq - Cp).dot(t);
    return static_cast<double>(drv.stiffness) * (static_cast<double>(drv.target_distance) - d);
}

void SimEngine::get_vertex_contact_force_sum(int vert_offset, int vert_count,
                                             double* out3) const
{
    // [force-control] Net IPC contact (barrier) force on a body, = sum over the
    // body's vertices of the per-vertex barrier gradient (the repulsion the body
    // feels from everything it touches). For a gripper finger this is the REAL
    // grip force (cup reaction), regardless of finray compliance / motion — the
    // correct signal for true force control. Uses the collision pairs from the
    // last step(); call AFTER eng.step().
    out3[0] = out3[1] = out3[2] = 0.0;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || vert_offset < 0 || vert_count <= 0 || vert_offset + vert_count > nv)
        return;
    if(g.m_skip_all_collision)
        return;

    // Re-detect contacts at the CURRENT (post-step) state: the solver clears the
    // DCD pair count after a step, so we rebuild the BVH + collision pairs here
    // before evaluating the barrier (contact) gradient.
    g.buildBVH();
    g.buildCP();
    if(getenv("STIFF_CONTACT_DBG"))
        fprintf(stderr, "[contact_dbg] h_cpNum0=%u h_gpNum=%u Kappa=%g dHat=%g nv=%d\n",
                g.h_cpNum[0], g.h_gpNum, g.Kappa, g.dHat, nv);
    if(g.h_cpNum[0] < 1 && g.h_gpNum < 1)
        return;  // nothing in contact (neither body-body nor ground)

    double3* d_grad = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_grad, nv * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_grad, 0, nv * sizeof(double3)));
    g.calBarrierGradient(d_grad, g.Kappa);    // body-body (DCD) contact force per vertex
    g.computeGroundGradient(d_grad, g.Kappa); // ground half-plane contact force per vertex
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<double3> h(vert_count);
    CUDA_SAFE_CALL(cudaMemcpy(h.data(), d_grad + vert_offset,
                              vert_count * sizeof(double3), cudaMemcpyDeviceToHost));
    double sx = 0, sy = 0, sz = 0;
    for(int i = 0; i < vert_count; i++) { sx += h[i].x; sy += h[i].y; sz += h[i].z; }
    out3[0] = sx; out3[1] = sy; out3[2] = sz;
    CUDA_SAFE_CALL(cudaFree(d_grad));
}

// ============================================================================
// Contact-pair "clean export layer"
// ----------------------------------------------------------------------------
// The solver stores collision pairs as MMCVID int4 (sign-packed {type,
// degeneracy} state, inherited from GIPC). That packing is load-bearing inside
// the contact kernels, so we DON'T touch it. Instead, anything OUTSIDE the
// solver (contact sensor, per-pair force, debug viz) goes through this single
// decode into a clean int4 of plain vertex indices ({v0,v1,v2,v3}, -1 padded
// for PP/PE), UIPC-style. The decode runs ON the readback path only — never in
// engine.step() — so it adds zero cost to training / batched solves.
//
// Sign-encoding decoded here (mirrors the barrier-gradient kernels):
//   .x >= 0            -> EE: {.x, .y, .z, (.w>=0? .w : -.w-1)}
//   .x <  0 (v0=-.x-1) -> .z<0: (.y<0 ? {v0,-.y-1,-.z-1,-.w-1} : PP {v0,.y,-1,-1})
//                         .w<0: (.y<0 ? {v0,-.y-1,.z,-.w-1}    : PE {v0,.y,.z,-1})
//                         else: PT {v0,.y,.z,.w}
__host__ __device__ static int4 _decode_pair_clean(int4 m)
{
    int v0, v1, v2, v3;
    v0 = v1 = v2 = v3 = -1;
    if(m.x >= 0)
    {
        v0 = m.x; v1 = m.y; v2 = m.z; v3 = (m.w >= 0) ? m.w : (-m.w - 1);
    }
    else
    {
        int p0 = -m.x - 1;
        if(m.z < 0)
        {
            if(m.y < 0) { v0 = p0; v1 = -m.y - 1; v2 = -m.z - 1; v3 = -m.w - 1; }
            else        { v0 = p0; v1 = m.y; }
        }
        else if(m.w < 0)
        {
            if(m.y < 0) { v0 = p0; v1 = -m.y - 1; v2 = m.z; v3 = -m.w - 1; }
            else        { v0 = p0; v1 = m.y; v2 = m.z; }
        }
        else { v0 = p0; v1 = m.y; v2 = m.z; v3 = m.w; }
    }
    return make_int4(v0, v1, v2, v3);
}

__global__ static void _decodeCleanContactPairs(const int4* mmcvid, int4* clean, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
        return;
    clean[idx] = _decode_pair_clean(mmcvid[idx]);
}

int SimEngine::get_collision_pairs_clean(int* out_flat) const
{
    // Decode the current body-body collision pairs into clean vertex-index
    // 4-tuples (row-major int4 -> out_flat[4*i .. 4*i+3], -1 padded). Returns
    // the pair count. Rebuilds contacts at the current (post-step) state.
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    if(g.m_skip_all_collision)
        return 0;
    g.buildBVH();
    g.buildCP();
    int ncp = static_cast<int>(g.h_cpNum[0]);
    if(ncp < 1)
        return 0;

    int4* d_clean = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_clean, ncp * sizeof(int4)));
    int threads = 256, blocks = (ncp + threads - 1) / threads;
    _decodeCleanContactPairs<<<blocks, threads>>>(g._collisonPairs, d_clean, ncp);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(out_flat != nullptr)
        CUDA_SAFE_CALL(cudaMemcpy(out_flat, d_clean, ncp * sizeof(int4), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_clean));
    return ncp;
}

// [Step B] Compute per-contact forces into grow-only device buffers. Returns the
// contact count (h_cpNum + h_gpNum). Buffers are read-only views valid until the
// next call; fetch device pointers via contacts_pair_ptr()/contacts_force_ptr().
// Call AFTER step().
int SimEngine::compute_contacts(bool rebuild)
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    impl.contact_count = 0;
    if(g.m_skip_all_collision || g.vertexNum <= 0)
        return 0;
    // By default REUSE the contact set the solver already built during the last
    // step() (BVH + collision pairs persist in _collisonPairs / _environment_
    // collisionPair with counts h_cpNum/h_gpNum) — like UIPC, where contacts are
    // queryable after world.advance() without a rebuild. Pass rebuild=true only
    // when calling outside the post-step window.
    if(rebuild)
    {
        g.buildBVH();
        g.buildCP();
    }
    int n = (int)g.h_cpNum[0] + (int)g.h_gpNum;
    if(n <= 0)
        return 0;
    if(n > impl.contact_cap)
    {
        if(impl.d_contact_pair)  CUDA_SAFE_CALL(cudaFree(impl.d_contact_pair));
        if(impl.d_contact_force) CUDA_SAFE_CALL(cudaFree(impl.d_contact_force));
        impl.contact_cap = n + n / 2 + 64;  // grow with slack
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_contact_pair, impl.contact_cap * sizeof(int2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&impl.d_contact_force, impl.contact_cap * sizeof(double3)));
    }
    impl.contact_count = g.exportContacts(impl.d_contact_pair, impl.d_contact_force);
    return impl.contact_count;
}

uintptr_t SimEngine::contacts_pair_ptr() const
{
    return reinterpret_cast<uintptr_t>(m_impl->d_contact_pair);
}

uintptr_t SimEngine::contacts_force_ptr() const
{
    return reinterpret_cast<uintptr_t>(m_impl->d_contact_force);
}

void SimEngine::get_pair_contact_force(int a_off, int a_cnt, int b_off, int b_cnt,
                                       double* out3) const
{
    // Net IPC contact (barrier) force on body A FROM body B: the barrier
    // gradient summed over A's vertices, restricted to collision pairs that
    // connect A's vertex range [a_off, a_off+a_cnt) and B's [b_off, b_off+b_cnt).
    // Used to populate the contact sensor's per-partner force_matrix_w. Call
    // AFTER step(). Returns the raw IP-scaled gradient (apply -1/dt^2 + sign on
    // the Python side, same convention as get_body_contact_force).
    out3[0] = out3[1] = out3[2] = 0.0;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || a_off < 0 || a_cnt <= 0 || b_off < 0 || b_cnt <= 0)
        return;
    if(a_off + a_cnt > nv || b_off + b_cnt > nv)
        return;
    if(g.m_skip_all_collision)
        return;

    g.buildBVH();
    g.buildCP();
    int ncp = static_cast<int>(g.h_cpNum[0]);
    if(ncp < 1)
        return;  // no body-body pairs (ground-only contact has no partner body)

    std::vector<int4> h_pairs(ncp);
    CUDA_SAFE_CALL(cudaMemcpy(h_pairs.data(), g._collisonPairs,
                              ncp * sizeof(int4), cudaMemcpyDeviceToHost));

    // Membership test on the CLEAN decode (plain vertex indices); the gradient
    // re-run below keeps the ORIGINAL MMCVID so its {type,degeneracy} flags are
    // preserved. Clean decode is the single source of truth (_decode_pair_clean).
    std::vector<int4> filtered;
    filtered.reserve(ncp);
    for(int i = 0; i < ncp; i++)
    {
        int4 c = _decode_pair_clean(h_pairs[i]);
        int  verts[4] = {c.x, c.y, c.z, c.w};
        bool inA = false, inB = false;
        for(int k = 0; k < 4; k++)
        {
            int vv = verts[k];
            if(vv < 0) continue;  // -1 padding (PP/PE unused slots)
            if(vv >= a_off && vv < a_off + a_cnt) inA = true;
            if(vv >= b_off && vv < b_off + b_cnt) inB = true;
        }
        if(inA && inB)
            filtered.push_back(h_pairs[i]);
    }
    if(filtered.empty())
        return;

    int4* d_filtered = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_filtered, filtered.size() * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMemcpy(d_filtered, filtered.data(),
                              filtered.size() * sizeof(int4), cudaMemcpyHostToDevice));

    double3* d_grad = nullptr;
    CUDA_SAFE_CALL(cudaMalloc(&d_grad, nv * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(d_grad, 0, nv * sizeof(double3)));

    // Reuse the existing barrier-gradient kernel on just the filtered pairs by
    // temporarily pointing GIPC at our subset (synchronous readback context).
    int4*    saved_pairs = g._collisonPairs;
    uint32_t saved_cpNum = g.h_cpNum[0];
    g._collisonPairs = d_filtered;
    g.h_cpNum[0]     = static_cast<uint32_t>(filtered.size());
    g.calBarrierGradient(d_grad, g.Kappa);
    g._collisonPairs = saved_pairs;
    g.h_cpNum[0]     = saved_cpNum;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<double3> hg(a_cnt);
    CUDA_SAFE_CALL(cudaMemcpy(hg.data(), d_grad + a_off,
                              a_cnt * sizeof(double3), cudaMemcpyDeviceToHost));
    double sx = 0, sy = 0, sz = 0;
    for(int i = 0; i < a_cnt; i++) { sx += hg[i].x; sy += hg[i].y; sz += hg[i].z; }
    out3[0] = sx; out3[1] = sy; out3[2] = sz;
    CUDA_SAFE_CALL(cudaFree(d_filtered));
    CUDA_SAFE_CALL(cudaFree(d_grad));
}

// [force-control / GPU gate] On-device MAX stitch-spring stretch over a range of
// stitch springs [start, start+count). stretch_j = |vert[targetInd[j]] -
// vert[paired[j]]|. Single-block shared-memory max reduction; returns one scalar
// — so a force-gated gripper controller can read the grip signal WITHOUT a full
// vertex-array D2H (only 8 bytes come back). All work stays on the GPU.
__global__ static void _stitch_max_stretch_kernel(const double3* verts,
                                                  const uint32_t* targetInd,
                                                  const int* paired,
                                                  int start, int count,
                                                  double* out)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for(int j = tid; j < count; j += blockDim.x)
    {
        int idx = start + j;
        int ai  = paired[idx];
        if(ai < 0) continue;                 // -1 = functor target, not a stitch pair
        uint32_t fi = targetInd[idx];
        double3 a = verts[fi];
        double3 b = verts[ai];
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        double d  = sqrt(dx * dx + dy * dy + dz * dz);
        local = fmax(local, d);
    }
    sdata[tid] = local;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if(tid == 0) out[0] = sdata[0];
}

double SimEngine::get_stitch_max_stretch(int pair_start, int pair_count) const
{
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   sn   = static_cast<int>(g.softNum);
    if(pair_count <= 0 || pair_start < 0 || pair_start + pair_count > sn)
        return 0.0;
    if(g._vertexes == nullptr || g.targetInd == nullptr
       || g.m_d_stitch_paired_vertex == nullptr)
        return 0.0;
    // tiny persistent device scalar (8 bytes); allocated once, never freed
    static double* s_d_out = nullptr;
    if(s_d_out == nullptr)
        CUDA_SAFE_CALL(cudaMalloc(&s_d_out, sizeof(double)));
    _stitch_max_stretch_kernel<<<1, 256>>>(g._vertexes, g.targetInd,
                                           g.m_d_stitch_paired_vertex,
                                           pair_start, pair_count, s_d_out);
    double h = 0.0;
    CUDA_SAFE_CALL(cudaMemcpy(&h, s_d_out, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

// [force-control / GPU gate — BATCHED] One block PER segment: block s reduces the
// stitch springs [starts[s], starts[s]+counts[s]) to one max stretch -> out[s].
// A "segment" is one finger (and, for multi-env, one finger of one env). Blocks
// are independent — each writes only its own out[s], with NO cross-segment shared
// state or atomics — so different fingers/ENVS cannot interfere with each other,
// and the whole batch is ONE kernel launch (no per-finger launch latency).
__global__ static void _stitch_max_stretch_batched_kernel(const double3* verts,
                                                          const uint32_t* targetInd,
                                                          const int* paired,
                                                          const int* starts,
                                                          const int* counts,
                                                          double* out)
{
    int seg   = blockIdx.x;
    int start = starts[seg];
    int count = counts[seg];
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for(int j = tid; j < count; j += blockDim.x)
    {
        int idx = start + j;
        int ai  = paired[idx];
        if(ai < 0) continue;
        uint32_t fi = targetInd[idx];
        double3 a = verts[fi];
        double3 b = verts[ai];
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        local = fmax(local, sqrt(dx * dx + dy * dy + dz * dz));
    }
    sdata[tid] = local;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if(tid == 0) out[seg] = sdata[0];
}

void SimEngine::get_stitch_max_stretch_batched(const int* h_starts,
                                               const int* h_counts,
                                               int n_seg, double* h_out) const
{
    for(int i = 0; i < n_seg; i++) h_out[i] = 0.0;
    if(n_seg <= 0) return;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    if(g._vertexes == nullptr || g.targetInd == nullptr
       || g.m_d_stitch_paired_vertex == nullptr)
        return;
    // persistent device buffers (grow on demand; never freed). Holding starts/
    // counts/out for ALL segments (= all fingers of all envs).
    static int*    s_d_starts = nullptr;
    static int*    s_d_counts = nullptr;
    static double* s_d_out    = nullptr;
    static int     s_cap      = 0;
    if(n_seg > s_cap)
    {
        if(s_d_starts) cudaFree(s_d_starts);
        if(s_d_counts) cudaFree(s_d_counts);
        if(s_d_out)    cudaFree(s_d_out);
        CUDA_SAFE_CALL(cudaMalloc(&s_d_starts, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&s_d_counts, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&s_d_out,    n_seg * sizeof(double)));
        s_cap = n_seg;
    }
    CUDA_SAFE_CALL(cudaMemcpy(s_d_starts, h_starts, n_seg * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(s_d_counts, h_counts, n_seg * sizeof(int), cudaMemcpyHostToDevice));
    _stitch_max_stretch_batched_kernel<<<n_seg, 256>>>(
        g._vertexes, g.targetInd, g.m_d_stitch_paired_vertex,
        s_d_starts, s_d_counts, s_d_out);
    CUDA_SAFE_CALL(cudaMemcpy(h_out, s_d_out, n_seg * sizeof(double), cudaMemcpyDeviceToHost));
}

// [force-control — BATCHED] One block PER segment (= one gripper finger, and for
// multi-env one finger of one env): block s sums the per-vertex contact force over
// [offsets[s], offsets[s]+counts[s]) -> out[s*3 .. s*3+2]. Blocks are independent
// (each writes only its own out), so envs cannot interfere. The expensive contact
// rebuild (buildBVH+buildCP+calBarrierGradient) is done ONCE by the caller, then
// ALL segments are summed in this ONE launch with ONE D2H — vs the single-body
// get_vertex_contact_force_sum which rebuilds contacts on EVERY call (N D2H syncs +
// N full contact rebuilds per frame at scale).
__global__ static void _contact_force_sum_batched_kernel(const double3* grad, int nv,
                                                         const int* offsets,
                                                         const int* counts,
                                                         double* out /* n_seg*3 */)
{
    int seg = blockIdx.x;
    int off = offsets[seg];
    int cnt = counts[seg];
    __shared__ double sx[256];
    __shared__ double sy[256];
    __shared__ double sz[256];
    int tid = threadIdx.x;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    for(int j = tid; j < cnt; j += blockDim.x)
    {
        int vi = off + j;
        if(vi < 0 || vi >= nv) continue;
        double3 v = grad[vi];
        lx += v.x; ly += v.y; lz += v.z;
    }
    sx[tid] = lx; sy[tid] = ly; sz[tid] = lz;
    __syncthreads();
    for(int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if(tid < s) { sx[tid] += sx[tid + s]; sy[tid] += sy[tid + s]; sz[tid] += sz[tid + s]; }
        __syncthreads();
    }
    if(tid == 0) { out[seg * 3 + 0] = sx[0]; out[seg * 3 + 1] = sy[0]; out[seg * 3 + 2] = sz[0]; }
}

void SimEngine::get_body_contact_force_batched(const int* h_offsets,
                                               const int* h_counts,
                                               int n_seg, double* h_out3) const
{
    // h_out3 holds n_seg 3-vectors (net IPC contact force per segment/finger).
    for(int i = 0; i < n_seg * 3; i++) h_out3[i] = 0.0;
    if(n_seg <= 0) return;
    auto& impl = *m_impl;
    GIPC& g    = impl.ipc;
    int   nv   = static_cast<int>(g.vertexNum);
    if(nv <= 0 || g.m_skip_all_collision) return;

    // Rebuild contacts ONCE at the current (post-step) state, then sum every
    // segment from the single barrier-gradient buffer.
    g.buildBVH();
    g.buildCP();
    if(g.h_cpNum[0] < 1) return;   // nothing in contact -> all zeros

    static double3* s_d_grad = nullptr;
    static int      s_grad_cap = 0;
    if(nv > s_grad_cap)
    {
        if(s_d_grad) cudaFree(s_d_grad);
        CUDA_SAFE_CALL(cudaMalloc(&s_d_grad, nv * sizeof(double3)));
        s_grad_cap = nv;
    }
    CUDA_SAFE_CALL(cudaMemset(s_d_grad, 0, nv * sizeof(double3)));
    g.calBarrierGradient(s_d_grad, g.Kappa);   // atomic-adds contact force per vertex

    static int*    s_d_off = nullptr;
    static int*    s_d_cnt = nullptr;
    static double* s_d_out = nullptr;
    static int     s_cap   = 0;
    if(n_seg > s_cap)
    {
        if(s_d_off) cudaFree(s_d_off);
        if(s_d_cnt) cudaFree(s_d_cnt);
        if(s_d_out) cudaFree(s_d_out);
        CUDA_SAFE_CALL(cudaMalloc(&s_d_off, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&s_d_cnt, n_seg * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&s_d_out, n_seg * 3 * sizeof(double)));
        s_cap = n_seg;
    }
    CUDA_SAFE_CALL(cudaMemcpy(s_d_off, h_offsets, n_seg * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(s_d_cnt, h_counts, n_seg * sizeof(int), cudaMemcpyHostToDevice));
    _contact_force_sum_batched_kernel<<<n_seg, 256>>>(s_d_grad, nv, s_d_off, s_d_cnt, s_d_out);
    CUDA_SAFE_CALL(cudaMemcpy(h_out3, s_d_out, n_seg * 3 * sizeof(double), cudaMemcpyDeviceToHost));
}

double SimEngine::get_prismatic_current_distance(int idx) const
{
    // [force-control] Current opening d = (Cq - Cp).dot(t) along the joint axis.
    // Lets a controller (or diagnostics) see the actual gripper opening — e.g.
    // to confirm a force-limited grasp does NOT fully close (stops at the object
    // width).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
        return 0.0;
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving)
        return 0.0;

    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    Vec12 qp, qc;
    CUDA_SAFE_CALL(cudaMemcpy(&qp, q_buf.data() + drv.parent_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&qc, q_buf.data() + drv.child_body_id,
                              sizeof(Vec12), cudaMemcpyDeviceToHost));

    auto worldpt = [](const Vec12& q, const Vector3& xb) -> Vector3 {
        Matrix3x3 A;
        A.row(0) = q.segment<3>(3).transpose();
        A.row(1) = q.segment<3>(6).transpose();
        A.row(2) = q.segment<3>(9).transpose();
        return Vector3(q.segment<3>(0) + A * xb);
    };
    Matrix3x3 Ac;
    Ac.row(0) = qc.segment<3>(3).transpose();
    Ac.row(1) = qc.segment<3>(6).transpose();
    Ac.row(2) = qc.segment<3>(9).transpose();
    Vector3 Cp = worldpt(qp, drv.Cp_bar);
    Vector3 Cq = worldpt(qc, drv.Cq_bar);
    Vector3 t  = Ac * drv.tq_bar;
    return (Cq - Cp).dot(t);
}

void SimEngine::set_prismatic_limit_barrier(int idx, double cl, double dir,
                                            double dhat, double kappa, int slot)
{
    // [force-control] Arm a one-sided IPC barrier on prismatic joint idx at the
    // CLOSED coordinate cl: the solver then NEVER lets d cross cl regardless of
    // the (force) drive — a hard no-overshoot guarantee while staying pure-force.
    // dir = +1 if the open end is at d>cl, else -1. kappa<=0 disarms. Written
    // directly to the GPU struct (the per-step target sync leaves these fields).
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system) {
        std::cerr << "[set_prismatic_limit_barrier] sim not finalized" << std::endl;
        return;
    }
    auto& sys = *impl.ipc.m_abd_system;
    if(idx < 0 || idx >= sys.m_num_prismatic_driving) {
        std::cerr << "[set_prismatic_limit_barrier] idx " << idx
                  << " out of range [0," << sys.m_num_prismatic_driving << ")" << std::endl;
        return;
    }
    PrismaticDrivingGPUData drv;
    CUDA_SAFE_CALL(cudaMemcpy(&drv, sys.m_prismatic_driving_data.data() + idx,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyDeviceToHost));
    if(slot == 0) {                              // slot 0 = closed-end barrier
        drv.limit_cl    = static_cast<Float>(cl);
        drv.limit_dir   = (dir >= 0.0) ? Float(1) : Float(-1);
        drv.limit_dhat  = static_cast<Float>(dhat);
        drv.limit_kappa = static_cast<Float>(kappa);
    } else {                                     // slot 1 = open-end barrier
        drv.limit_cl2    = static_cast<Float>(cl);
        drv.limit_dir2   = (dir >= 0.0) ? Float(1) : Float(-1);
        drv.limit_dhat2  = static_cast<Float>(dhat);
        drv.limit_kappa2 = static_cast<Float>(kappa);
    }
    CUDA_SAFE_CALL(cudaMemcpy(sys.m_prismatic_driving_data.data() + idx, &drv,
                              sizeof(PrismaticDrivingGPUData), cudaMemcpyHostToDevice));
}

void SimEngine::get_revolute_current_angles(double* out, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_system || !impl.ipc.m_abd_sim_data)
    {
        for(int i = 0; i < count; i++)
            out[i] = 0.0;
        return;
    }

    int n = std::min(count, static_cast<int>(impl.tetMesh.joint_angle_controls.size()));
    if(n == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0)
    {
        for(int i = 0; i < n; i++)
            out[i] = 0.0;
        return;
    }

    // Read current ABD state vectors (q) from GPU
    // Vector12 = Eigen::Matrix<double, 12, 1>
    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    auto& q_buf = impl.ipc.m_abd_sim_data->device.body_id_to_q;
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(), q_buf.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    // Read revolute driving GPU data
    int n_drv = impl.ipc.m_abd_system->m_num_revolute_driving;
    if(n_drv == 0)
    {
        for(int i = 0; i < n; i++)
            out[i] = 0.0;
        return;
    }

    std::vector<RevoluteDrivingGPUData> host_drv(n_drv);
    CUDA_SAFE_CALL(cudaMemcpy(host_drv.data(),
                              impl.ipc.m_abd_system->m_revolute_driving_data.data(),
                              n_drv * sizeof(RevoluteDrivingGPUData),
                              cudaMemcpyDeviceToHost));

    for(int i = 0; i < n && i < n_drv; i++)
    {
        auto& drv = host_drv[i];
        auto& q1  = host_q[drv.parent_body_id];
        auto& q2  = host_q[drv.child_body_id];

        // Extract 3x3 rotation from q = [p(3), a1(3), a2(3), a3(3)]
        Eigen::Matrix3d A1, A2;
        A1.row(0) = q1.segment<3>(3).transpose();
        A1.row(1) = q1.segment<3>(6).transpose();
        A1.row(2) = q1.segment<3>(9).transpose();
        A2.row(0) = q2.segment<3>(3).transpose();
        A2.row(1) = q2.segment<3>(6).transpose();
        A2.row(2) = q2.segment<3>(9).transpose();

        // gipc::Vector3 is Eigen::Matrix<double,3,1>, same as Eigen::Vector3d
        Eigen::Vector3d p  = A1 * drv.p_bar;
        Eigen::Vector3d pN = A1 * drv.pN_bar;
        Eigen::Vector3d q_d  = A2 * drv.q_bar;
        Eigen::Vector3d qN = A2 * drv.qN_bar;

        double cos_theta = 0.5 * (p.dot(q_d) + pN.dot(qN));
        double sin_theta = 0.5 * (q_d.dot(pN) - qN.dot(p));

        out[i] = std::atan2(sin_theta, cos_theta);
    }
}

void SimEngine::get_revolute_initial_offsets(double* out, int count) const
{
    auto& ctrls = m_impl->tetMesh.joint_angle_controls;
    int   n     = std::min(count, static_cast<int>(ctrls.size()));
    for(int i = 0; i < n; i++)
        out[i] = ctrls[i].initial_angle_offset;
    for(int i = n; i < count; i++)
        out[i] = 0.0;
}

// ======================== load_mesh_from_data ========================
void SimEngine::load_mesh_from_data(const double*          vertices,
                                    int                    num_verts,
                                    const int*             faces,
                                    int                    num_faces,
                                    int                    verts_per_face,
                                    int                    dimensions,
                                    int                    body_type,
                                    const Eigen::Matrix4d& transform,
                                    double                 young_modulus,
                                    int                    boundary_type)
{
    int prev_verts = m_impl->tetMesh.vertexNum;

    // Write vertices and faces to a temporary file, then load via SimpleSceneImporter.
    // This reuses the existing loading pipeline including tetrahedralization for 3D.
    std::string tmp_dir = "/tmp/stiffgipc_mesh_data/";
    std::filesystem::create_directories(tmp_dir);
    std::string tmp_path;

    if(dimensions == 2 || verts_per_face == 3)
    {
        tmp_path = tmp_dir + "tmp_mesh_" + std::to_string(m_impl->load_records.size()) + ".obj";
        std::ofstream ofs(tmp_path);
        for(int i = 0; i < num_verts; i++)
            ofs << "v " << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << "f";
            for(int j = 0; j < verts_per_face; j++)
                ofs << " " << (faces[i * verts_per_face + j] + 1);
            ofs << "\n";
        }
        ofs.close();
    }
    else
    {
        // For pre-tetrahedralized data (verts_per_face == 4), write as .msh
        tmp_path = tmp_dir + "tmp_mesh_" + std::to_string(m_impl->load_records.size()) + ".msh";
        std::ofstream ofs(tmp_path);
        ofs << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
        ofs << "$Nodes\n" << num_verts << "\n";
        for(int i = 0; i < num_verts; i++)
            ofs << (i + 1) << " " << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << "\n";
        ofs << "$EndNodes\n$Elements\n" << num_faces << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << (i + 1) << " 4 2 0 0";
            for(int j = 0; j < 4; j++)
                ofs << " " << (faces[i * 4 + j] + 1);
            ofs << "\n";
        }
        ofs << "$EndElements\n";
        ofs.close();
    }

    auto bt = (body_type == 0) ? gipc::BodyType::ABD : gipc::BodyType::FEM;
    auto bb = (boundary_type == 1) ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;

    if(bt == gipc::BodyType::ABD && (dimensions == 2 || verts_per_face == 3))
    {
        m_impl->tetMesh.load_surfaceMesh_ABD(tmp_path, transform, young_modulus, bb);
    }
    else
    {
        SimpleSceneImporter imp;
        // Issue 1: pass runtime metis_dir to avoid the OUTPUT_DIR
        // compile-time path leak (only matters when preconditioner_type != 0).
        std::string metis_dir = m_impl->resolved_assets_dir + "sorted_mesh/";
        std::filesystem::create_directories(metis_dir);
        imp.load_geometry(m_impl->tetMesh, dimensions, bt, transform,
                          young_modulus, tmp_path,
                          m_impl->cfg.preconditioner_type, bb, metis_dir);
    }

    m_impl->record_load(body_type, prev_verts);
    m_impl->load_records.back().label = "from_data";

    std::cout << "[SimEngine] Mesh from data loaded (dim=" << dimensions
              << ", " << (body_type == 0 ? "ABD" : "FEM")
              << ", verts=" << num_verts << ", faces=" << num_faces << ")" << std::endl;
}

// ======================== Impl helpers for instanced loading ========================

std::string SimEngine::Impl::write_temp_mesh(
    const double* vertices, int num_verts,
    const int* faces, int num_faces,
    int verts_per_face, int dimensions,
    const std::string& suffix)
{
    std::string tmp_dir = "/tmp/stiffgipc_mesh_data/";
    std::filesystem::create_directories(tmp_dir);

    if(dimensions == 2 || verts_per_face == 3)
    {
        std::string tmp_path = tmp_dir + "tmp_instanced_" + suffix + ".obj";
        std::ofstream ofs(tmp_path);
        for(int i = 0; i < num_verts; i++)
            ofs << "v " << vertices[i*3] << " " << vertices[i*3+1] << " " << vertices[i*3+2] << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << "f";
            for(int j = 0; j < verts_per_face; j++)
                ofs << " " << (faces[i*verts_per_face + j] + 1);
            ofs << "\n";
        }
        ofs.close();
        return tmp_path;
    }
    else
    {
        std::string tmp_path = tmp_dir + "tmp_instanced_" + suffix + ".msh";
        std::ofstream ofs(tmp_path);
        ofs << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
        ofs << "$Nodes\n" << num_verts << "\n";
        for(int i = 0; i < num_verts; i++)
            ofs << (i+1) << " " << vertices[i*3] << " " << vertices[i*3+1] << " " << vertices[i*3+2] << "\n";
        ofs << "$EndNodes\n$Elements\n" << num_faces << "\n";
        for(int i = 0; i < num_faces; i++)
        {
            ofs << (i+1) << " 4 2 0 0";
            for(int j = 0; j < 4; j++)
                ofs << " " << (faces[i*4 + j] + 1);
            ofs << "\n";
        }
        ofs << "$EndElements\n";
        ofs.close();
        return tmp_path;
    }
}

void SimEngine::Impl::load_from_temp_file(
    const std::string& tmp_path,
    int dimensions, int body_type, int verts_per_face,
    const Eigen::Matrix4d& transform,
    double young_modulus, int boundary_type)
{
    auto bt = (body_type == 0) ? gipc::BodyType::ABD : gipc::BodyType::FEM;
    auto bb = (boundary_type == 1) ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;

    if(bt == gipc::BodyType::ABD && (dimensions == 2 || verts_per_face == 3))
    {
        tetMesh.load_surfaceMesh_ABD(tmp_path, transform, young_modulus, bb);
    }
    else
    {
        SimpleSceneImporter imp;
        // Issue 1: pass runtime metis_dir so the MAS preconditioner path
        // doesn't trip the OUTPUT_DIR build-time path leak.
        std::string metis_dir = resolved_assets_dir + "sorted_mesh/";
        std::filesystem::create_directories(metis_dir);
        imp.load_geometry(tetMesh, dimensions, bt, transform,
                          young_modulus, tmp_path,
                          cfg.preconditioner_type, bb, metis_dir);
    }
}

// ======================== load_mesh_instanced ========================

InstancedLoadResult SimEngine::load_mesh_instanced(
    const double*                       vertices,
    int                                 num_verts,
    const int*                          faces,
    int                                 num_faces,
    int                                 verts_per_face,
    int                                 dimensions,
    int                                 body_type,
    const std::vector<Eigen::Matrix4d>& transforms,
    double                              young_modulus,
    int                                 boundary_type)
{
    int N = static_cast<int>(transforms.size());
    if(N == 0) return {};

    // 1. Register a MeshAsset (store rest topology once)
    MeshAsset asset;
    asset.asset_id      = static_cast<int>(m_impl->mesh_assets.size());
    asset.num_verts     = num_verts;
    asset.num_faces     = num_faces;
    asset.verts_per_face = verts_per_face;
    asset.dimensions    = dimensions;
    asset.body_type     = body_type;
    asset.young_modulus = young_modulus;
    asset.boundary_type = boundary_type;
    asset.rest_vertices.assign(vertices, vertices + num_verts * 3);
    asset.faces.assign(faces, faces + num_faces * verts_per_face);
    m_impl->mesh_assets.push_back(asset);

    // 2. Write temp mesh file ONCE
    std::string suffix = "asset" + std::to_string(asset.asset_id);
    std::string tmp_path = m_impl->write_temp_mesh(
        vertices, num_verts, faces, num_faces,
        verts_per_face, dimensions, suffix);

    // 3. Load N instances
    InstancedLoadResult result;
    result.asset_id = asset.asset_id;
    result.body_offsets.reserve(N);
    result.vertex_offsets.reserve(N);
    result.vertex_counts.reserve(N);

    for(int i = 0; i < N; i++)
    {
        int prev_verts = m_impl->tetMesh.vertexNum;

        m_impl->load_from_temp_file(tmp_path, dimensions, body_type,
                                    verts_per_face, transforms[i],
                                    young_modulus, boundary_type);

        m_impl->record_load(body_type, prev_verts);
        auto& rec = m_impl->load_records.back();
        rec.label       = "instanced_" + suffix + "_i" + std::to_string(i);
        rec.asset_id    = asset.asset_id;
        rec.instance_id = i;

        result.body_offsets.push_back(rec.body_offset);
        result.vertex_offsets.push_back(rec.vertex_offset);
        result.vertex_counts.push_back(rec.vertex_count);
    }

    std::cout << "[SimEngine] Instanced load: asset=" << asset.asset_id
              << ", N=" << N << ", " << (body_type == 0 ? "ABD" : "FEM")
              << ", verts_per_instance=" << num_verts
              << ", bodies=" << result.body_offsets.front()
              << ".." << result.body_offsets.back() << std::endl;

    return result;
}

// ======================== Mesh asset queries ========================

int SimEngine::get_mesh_asset_count() const
{
    return static_cast<int>(m_impl->mesh_assets.size());
}

const MeshAsset& SimEngine::get_mesh_asset(int asset_id) const
{
    return m_impl->mesh_assets.at(asset_id);
}

// ======================== FEM body count ========================
int SimEngine::get_fem_body_count() const
{
    return static_cast<int>(m_impl->tetMesh.abd_fem_count_info.fem_body_num);
}

// ======================== ABD body state access ========================
void SimEngine::get_abd_body_transforms(const int* body_offsets, double* out_mat4x4, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        double* m = out_mat4x4 + i * 16;
        std::memset(m, 0, 16 * sizeof(double));
        m[15] = 1.0;

        if(bid < 0 || bid >= num_bodies) continue;
        const auto& q = host_q[bid];

        // Translation: q[0:3] = p
        m[3]  = q[0]; m[7]  = q[1]; m[11] = q[2];
        // Rotation: A = [a1 | a2 | a3], mat[:3,:3] = A^T
        // a1 = q[3:6], a2 = q[6:9], a3 = q[9:12]
        // Row-major 4x4: m[row*4+col]
        m[0]  = q[3];  m[1]  = q[6];  m[2]  = q[9];   // row 0
        m[4]  = q[4];  m[5]  = q[7];  m[6]  = q[10];  // row 1
        m[8]  = q[5];  m[9]  = q[8];  m[10] = q[11];  // row 2
    }
}

void SimEngine::set_abd_body_transforms(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& q = host_q[bid];
        // Translation
        q[0] = m[3]; q[1] = m[7]; q[2] = m[11];
        // Rotation columns: a1=col0, a2=col1, a3=col2 of the 3x3 block
        q[3]  = m[0]; q[4]  = m[4]; q[5]  = m[8];   // a1
        q[6]  = m[1]; q[7]  = m[5]; q[8]  = m[9];   // a2
        q[9]  = m[2]; q[10] = m[6]; q[11] = m[10];  // a3
    }

    CUDA_SAFE_CALL(cudaMemcpy(impl.ipc.m_abd_sim_data->device.body_id_to_q.data(),
                              host_q.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyHostToDevice));
}

void SimEngine::teleport_abd_bodies(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    auto& abd = impl.ipc.m_abd_sim_data->device;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    size_t buf_bytes = num_bodies * sizeof(Vec12);

    // Read current q from GPU
    std::vector<Vec12> host_q(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_q.data(), abd.body_id_to_q.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));

    // Apply new transforms
    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& q = host_q[bid];
        q[0] = m[3]; q[1] = m[7]; q[2] = m[11];
        q[3]  = m[0]; q[4]  = m[4]; q[5]  = m[8];
        q[6]  = m[1]; q[7]  = m[5]; q[8]  = m[9];
        q[9]  = m[2]; q[10] = m[6]; q[11] = m[10];
    }

    // Write the same state to q, q_prev, q_tilde, q_temp
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q.data(),       host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_prev.data(),  host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_tilde.data(), host_q.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_temp.data(),  host_q.data(), buf_bytes, cudaMemcpyHostToDevice));

    // Zero velocity and delta-q for teleported bodies
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(), abd.body_id_to_q_v.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));
    std::vector<Vec12> host_dq(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_dq.data(), abd.body_id_to_dq.data(),
                              buf_bytes, cudaMemcpyDeviceToHost));
    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;
        host_qv[bid] = Vec12::Zero();
        host_dq[bid] = Vec12::Zero();
    }
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_q_v.data(), host_qv.data(), buf_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(abd.body_id_to_dq.data(),  host_dq.data(), buf_bytes, cudaMemcpyHostToDevice));
}

void SimEngine::get_abd_body_velocities(const int* body_offsets, double* out_mat4x4, int count) const
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        double* m = out_mat4x4 + i * 16;
        std::memset(m, 0, 16 * sizeof(double));

        if(bid < 0 || bid >= num_bodies) continue;
        const auto& qv = host_qv[bid];

        m[3]  = qv[0]; m[7]  = qv[1]; m[11] = qv[2];
        m[0]  = qv[3];  m[1]  = qv[6];  m[2]  = qv[9];
        m[4]  = qv[4];  m[5]  = qv[7];  m[6]  = qv[10];
        m[8]  = qv[5];  m[9]  = qv[8];  m[10] = qv[11];
    }
}

void SimEngine::set_abd_body_velocities(const int* body_offsets, const double* mat4x4, int count)
{
    auto& impl = *m_impl;
    if(!impl.ipc.m_abd_sim_data || count == 0) return;

    int num_bodies = impl.ipc.abd_fem_count_info.abd_body_num;
    if(num_bodies == 0) return;

    using Vec12 = Eigen::Matrix<double, 12, 1>;
    std::vector<Vec12> host_qv(num_bodies);
    CUDA_SAFE_CALL(cudaMemcpy(host_qv.data(),
                              impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyDeviceToHost));

    for(int i = 0; i < count; i++)
    {
        int bid = body_offsets[i];
        if(bid < 0 || bid >= num_bodies) continue;

        const double* m = mat4x4 + i * 16;
        auto& qv = host_qv[bid];
        qv[0] = m[3]; qv[1] = m[7]; qv[2] = m[11];
        qv[3]  = m[0]; qv[4]  = m[4]; qv[5]  = m[8];
        qv[6]  = m[1]; qv[7]  = m[5]; qv[8]  = m[9];
        qv[9]  = m[2]; qv[10] = m[6]; qv[11] = m[10];
    }

    CUDA_SAFE_CALL(cudaMemcpy(impl.ipc.m_abd_sim_data->device.body_id_to_q_v.data(),
                              host_qv.data(),
                              num_bodies * sizeof(Vec12), cudaMemcpyHostToDevice));
}

// ======================== FEM vertex state ========================
void SimEngine::get_vertex_velocities(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] Same convention as get_vertex_positions: output in input order.
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(out_xyz, m_impl->d_tetMesh.velocities,
                                  n * sizeof(double3), cudaMemcpyDeviceToHost));
        return;
    }
    std::vector<double3> tmp(n);
    CUDA_SAFE_CALL(cudaMemcpy(tmp.data(), m_impl->d_tetMesh.velocities,
                              n * sizeof(double3), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n) j = i;
        out_xyz[3*j + 0] = tmp[i].x;
        out_xyz[3*j + 1] = tmp[i].y;
        out_xyz[3*j + 2] = tmp[i].z;
    }
}

void SimEngine::set_vertex_positions_gpu(const double* xyz, int count)
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // [MAS-perm] xyz is in input order: xyz[3*j] = pos of input vertex j.
    // Engine internal storage is in metis-sort order (or identity).
    // For each engine vertex i, write user input at perm[i]: gpu[i] = xyz[3*perm[i]].
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->ipc._vertexes, xyz,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        return;
    }
    std::vector<double3> tmp(n);
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            tmp[i] = make_double3(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            continue;
        }
        tmp[i] = make_double3(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->ipc._vertexes, tmp.data(),
                              n * sizeof(double3), cudaMemcpyHostToDevice));
}

void SimEngine::set_vertex_velocities_gpu(const double* xyz, int count)
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;

    // Same MAS-perm convention as set_vertex_positions_gpu.
    const auto& perm = m_impl->tetMesh.vertex_metis_to_input;
    bool use_perm = !perm.empty() && static_cast<int>(perm.size()) >= n;
    if(!use_perm)
    {
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.velocities, xyz,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        return;
    }
    std::vector<double3> tmp(n);
    for(int i = 0; i < n; i++)
    {
        int j = perm[i];
        if(j < 0 || j >= n)
        {
            tmp[i] = make_double3(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            continue;
        }
        tmp[i] = make_double3(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
    }
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.velocities, tmp.data(),
                              n * sizeof(double3), cudaMemcpyHostToDevice));
}

void SimEngine::teleport_fem_vertices(const double* xyz, int count,
                                      const double* velocities)
{
    int n = std::min(count, static_cast<int>(m_impl->ipc.vertexNum));
    if(n <= 0) return;
    // Write _vertexes (current) and o_vertexes (committed previous-step).
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->ipc._vertexes, xyz,
                              n * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.o_vertexes, xyz,
                              n * sizeof(double3), cudaMemcpyHostToDevice));

    if(velocities != nullptr)
    {
        // Preserve inertia: write v and build xTilta = x + v*dt + g*dt^2
        // on the host, then upload.
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.velocities, velocities,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        double dt = m_impl->ipc.IPC_dt;
        double3 g = m_impl->ipc.gravity;
        double dt2 = dt * dt;
        std::vector<double> xTilta_host(3 * n);
        for(int i = 0; i < n; i++)
        {
            xTilta_host[3*i+0] = xyz[3*i+0] + velocities[3*i+0] * dt + g.x * dt2;
            xTilta_host[3*i+1] = xyz[3*i+1] + velocities[3*i+1] * dt + g.y * dt2;
            xTilta_host[3*i+2] = xyz[3*i+2] + velocities[3*i+2] * dt + g.z * dt2;
        }
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.xTilta, xTilta_host.data(),
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
    }
    else
    {
        // Zero velocity, xTilta = new_pos (caller declines to preserve
        // inertia; matches teleport_abd_bodies semantics).
        CUDA_SAFE_CALL(cudaMemcpy(m_impl->d_tetMesh.xTilta, xyz,
                                  n * sizeof(double3), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(m_impl->d_tetMesh.velocities, 0,
                                  n * sizeof(double3)));
    }
}

void SimEngine::get_fem_body_vertex_range(int fem_body_idx, int* out_start, int* out_count) const
{
    if(fem_body_idx >= 0 && fem_body_idx < static_cast<int>(m_impl->fem_body_ranges.size()))
    {
        *out_start = m_impl->fem_body_ranges[fem_body_idx].vertex_start;
        *out_count = m_impl->fem_body_ranges[fem_body_idx].vertex_count;
    }
    else
    {
        *out_start = 0;
        *out_count = 0;
    }
}

// ======================== Load record tracking ========================
int SimEngine::get_load_record_count() const
{
    return static_cast<int>(m_impl->load_records.size());
}

const BodyLoadRecord& SimEngine::get_load_record(int idx) const
{
    return m_impl->load_records.at(idx);
}

}  // namespace gipc


// ======================== Per-step counters (perf debugging) ========================
extern int    totalNT;
extern double total_Cg_count;
extern double totalCollisionPairs;
extern double maxCOllisionPairNum;
extern int    total_Frames;

namespace gipc {
int    SimEngine::get_total_newton_iters() const     { return totalNT; }
double SimEngine::get_total_pcg_iters() const        { return total_Cg_count; }
double SimEngine::get_total_collision_pairs() const  { return totalCollisionPairs; }
double SimEngine::get_max_collision_pairs() const    { return maxCOllisionPairNum; }
int    SimEngine::get_total_frames_done() const      { return total_Frames; }
}  // namespace gipc
