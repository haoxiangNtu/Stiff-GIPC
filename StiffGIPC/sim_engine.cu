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

    std::vector<BodyLoadRecord> load_records;

    // Per-FEM-body vertex ranges (populated during load)
    struct FEMBodyRange { int vertex_start; int vertex_count; };
    std::vector<FEMBodyRange> fem_body_ranges;

    // Shared mesh assets for instanced loading
    std::vector<MeshAsset> mesh_assets;

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

    bool ok = urdf_importer.import_scene(m_impl->tetMesh, m_impl->cfg.preconditioner_type);
    if(!ok)
    {
        std::cerr << "[SimEngine] URDF import failed: " << urdf_path << std::endl;
        return;
    }

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
    if(vertex_index >= 0 && vertex_index < static_cast<int>(m_impl->tetMesh.boundaryTypies.size()))
        m_impl->tetMesh.boundaryTypies[vertex_index] = boundary_type;
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

void SimEngine::get_vertex_positions_host(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->tetMesh.vertexNum));
    for(int i = 0; i < n; i++)
    {
        const auto& v = m_impl->tetMesh.vertexes[i];
        out_xyz[3*i + 0] = v.x;
        out_xyz[3*i + 1] = v.y;
        out_xyz[3*i + 2] = v.z;
    }
}

void SimEngine::get_vertex_positions_host(double* out_xyz, int count) const
{
    int n = std::min(count, static_cast<int>(m_impl->tetMesh.vertexNum));
    for(int i = 0; i < n; i++)
    {
        const auto& v = m_impl->tetMesh.vertexes[i];
        out_xyz[3*i + 0] = v.x;
        out_xyz[3*i + 1] = v.y;
        out_xyz[3*i + 2] = v.z;
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
    printf("[CollisionExclusion] pairs=%d, collision_body_num=%d\n",
           (int)tetMesh.collision_exclusion_pairs.size(), d_tetMesh.collision_body_num);
    if(!tetMesh.collision_exclusion_pairs.empty() && d_tetMesh.collision_body_num > 0)
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
        safe_copy(d_tetMesh.collision_skip_matrix, host_matrix.data(),
                  N * N * sizeof(int), cudaMemcpyHostToDevice);
        printf("[CollisionExclusion] Uploaded %dx%d exclusion matrix (%d pairs)\n",
               N, N, (int)tetMesh.collision_exclusion_pairs.size());
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
        printf("[BVHSkip#2] DISABLED via BVHSKIP2=0 (kernels see no isolated diag bits)\n");
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
        printf("[BVHSkip] %d/%d bodies fully isolated → diag[i][i]=1 short-circuit set\n",
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

    printf("[SimEngine] collision_detection_buff_scale=%.1f  MAX_CCD_PAIRS=%d  MAX_PAIRS=%d\n",
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

    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id,
                d_tetMesh.collision_skip_matrix, d_tetMesh.collision_body_num);
    ipc._point_body_id = d_tetMesh.point_id_to_body_id;

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
            printf("[BVHSkip#3] DISABLED via BVHSKIP3=0\n");
        } else if(!bvhskip2_enabled) {
            printf("[BVHSkip#3] AUTO-DISABLED (BVHSKIP2=0 — #3 requires #2's isolation set)\n");
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
            printf("[BVHSkip#3] %d/%d isolated  active faces=%d/%zu  active edges=%d/%zu\n",
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

    // DIAG: dump q for first 3 bodies after full finalize
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
}

// ======================== step ========================
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
}

// ======================== state queries ========================
int SimEngine::get_vertex_count() const
{
    return m_impl->ipc.vertexNum;
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

void SimEngine::set_revolute_initial_offset(int idx, double offset_rad)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).initial_angle_offset = offset_rad;
}

void SimEngine::set_prismatic_target(int idx, double distance_m)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance = distance_m;
}

void SimEngine::set_revolute_strength(int idx, double strength)
{
    m_impl->tetMesh.joint_angle_controls.at(idx).strength_ratio = strength;
}

void SimEngine::set_prismatic_strength(int idx, double strength)
{
    m_impl->tetMesh.prismatic_drive_controls.at(idx).strength_ratio = strength;
}

double SimEngine::get_revolute_target(int idx) const
{
    return m_impl->tetMesh.joint_angle_controls.at(idx).target_angle;
}

double SimEngine::get_prismatic_target(int idx) const
{
    return m_impl->tetMesh.prismatic_drive_controls.at(idx).target_distance;
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
