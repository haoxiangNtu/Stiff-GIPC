#pragma once
#ifndef STIFFGIPC_SIM_ENGINE_H
#define STIFFGIPC_SIM_ENGINE_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <Eigen/Core>

struct JointAngleControlInfo;
struct PrismaticDrivingControlInfo;

namespace gipc
{

struct SimEngineConfig
{
    double dt                  = 1e-2;
    double density             = 1e3;
    double young_modulus       = 1e7;
    double poisson_rate        = 0.49;
    double friction_rate       = 0.4;
    double gd_friction_rate    = 0.4;
    double cloth_thickness     = 1e-3;
    double cloth_young_modulus = 1e6;
    double bend_young_modulus  = 1e5;
    double cloth_density       = 2e2;
    double strain_rate         = 100;
    double soft_motion_rate    = 1e0;
    double newton_tol          = 1e-2;
    double pcg_tol             = 1e-4;
    double relative_dhat       = 1e-3;

    double joint_strength_ratio            = 100.0;
    double revolute_driving_strength_ratio = 100.0;
    double prismatic_strength_ratio        = 100.0;
    double prismatic_driving_strength_ratio = 100.0;

    double collision_detection_buff_scale = 1.0;
    double linear_system_buff_scale       = 1.0;
    int    preconditioner_type            = 1;  // 1 = MAS
    int    cuda_device                    = 0;

    bool   semi_implicit_enabled  = false;
    double semi_implicit_beta_tol = 1e-3;
    int    semi_implicit_min_iter = 1;
    int    newton_iter_cap        = 1000;

    bool   skip_all_collision = false;

    double velocity_damping = 0.0;  // per-step velocity scaling: v *= (1 - damping)

    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.8, 0.0);

    Eigen::Vector3d ground_normal = Eigen::Vector3d(0.0, 1.0, 0.0);
    double          ground_offset = -1.0;

    std::string assets_dir;  // override GIPC_ASSETS_DIR if non-empty
};

struct JointInfo
{
    std::string name;
    double      lower_limit;     // radians for revolute, metres for prismatic
    double      upper_limit;
    double      target;          // current target (radians / metres)
    double      strength_ratio;
    bool        is_prismatic;
};

struct BodyLoadRecord
{
    int         body_type;      // 0=ABD, 1=FEM
    int         body_offset;    // index into ABD or FEM body array
    int         vertex_offset;  // first vertex index in global vertex array
    int         vertex_count;   // number of vertices for this load
    int         asset_id  = -1; // link to shared MeshAsset (-1 = none)
    int         instance_id = 0;// which instance of the asset
    std::string label;          // optional identifier (prim path, mesh name, etc.)
};

struct MeshAsset
{
    int                 asset_id;
    std::vector<double> rest_vertices;  // local-space vertices (N*3 flat)
    std::vector<int>    faces;          // face indices (M*vpf flat)
    int                 num_verts;
    int                 num_faces;
    int                 verts_per_face;
    int                 dimensions;
    int                 body_type;      // 0=ABD, 1=FEM
    double              young_modulus;
    int                 boundary_type;
};

struct InstancedLoadResult
{
    std::vector<int> body_offsets;    // N body indices
    std::vector<int> vertex_offsets;  // N vertex start indices
    std::vector<int> vertex_counts;   // N vertex counts (all same for same mesh)
    int              asset_id;        // shared mesh asset reference
};

class SimEngine
{
  public:
    SimEngine();
    ~SimEngine();

    SimEngine(const SimEngine&)            = delete;
    SimEngine& operator=(const SimEngine&) = delete;

    void set_config(const SimEngineConfig& cfg);
    const SimEngineConfig& config() const;

    void init_cuda();

    void load_urdf(const std::string&     urdf_path,
                   const Eigen::Matrix4d& global_transform,
                   bool                   root_fixed                = true,
                   bool                   revolute_as_motor         = false,
                   double                 default_young             = 1e7,
                   const std::map<std::string, double>& initial_joint_angles = {});

    // Load a raw mesh (.msh for 3D tet, .obj for 2D cloth/shell).
    // dimensions: 2 = triangle shell, 3 = tet volume.
    // body_type:  0 = ABD (rigid), 1 = FEM (deformable).
    // boundary_type: 0 = Free, 1 = Fixed.
    // ABD bodies must be loaded before any FEM body.
    void load_mesh(const std::string&     mesh_path,
                   int                    dimensions,
                   int                    body_type,
                   const Eigen::Matrix4d& transform,
                   double                 young_modulus,
                   int                    boundary_type = 0);

    // Load mesh from in-memory vertex/face data (no file I/O).
    // For dim=3 with surface-only data, internally tetrahedralizes.
    void load_mesh_from_data(const double*          vertices,
                             int                    num_verts,
                             const int*             faces,
                             int                    num_faces,
                             int                    verts_per_face,
                             int                    dimensions,
                             int                    body_type,
                             const Eigen::Matrix4d& transform,
                             double                 young_modulus,
                             int                    boundary_type = 0);

    // Load one mesh as N instances with different transforms.
    // The mesh topology is parsed/written to disk once; each instance
    // gets its own body in the engine with an independent transform.
    // Returns body offsets, vertex ranges, and the shared asset_id.
    InstancedLoadResult load_mesh_instanced(
        const double*                          vertices,
        int                                    num_verts,
        const int*                             faces,
        int                                    num_faces,
        int                                    verts_per_face,
        int                                    dimensions,
        int                                    body_type,
        const std::vector<Eigen::Matrix4d>&    transforms,
        double                                 young_modulus,
        int                                    boundary_type = 0);

    // ---- Mesh asset queries ----
    int              get_mesh_asset_count() const;
    const MeshAsset& get_mesh_asset(int asset_id) const;

    void add_ground(double height = 0.0);

    void add_collision_exclusion(int body_a, int body_b);
    void add_ground_collision_skip(int body_id);

    // ---- Programmatic joint creation (must call before finalize()) ----
    // Returns the constraint index.

    int add_fixed_joint(int parent_body, int child_body,
                        const Eigen::Vector3d& world_anchor,
                        const Eigen::Vector3d& world_normal,
                        const Eigen::Vector3d& world_bitangent);

    int add_revolute_joint(int parent_body, int child_body,
                           const Eigen::Vector3d& world_axis,
                           const Eigen::Vector3d& joint_pos,
                           double lower_limit, double upper_limit,
                           double initial_angle = 0.0,
                           const std::string& name = "");

    int add_prismatic_joint(int parent_body, int child_body,
                            const Eigen::Vector3d& world_center,
                            const Eigen::Vector3d& world_axis,
                            double lower_limit, double upper_limit,
                            const std::string& name = "");

    // Per-vertex boundary type (0=Free, 1=Fixed). Must call before finalize().
    void set_vertex_boundary(int vertex_index, int boundary_type);

    int  get_abd_body_count() const;
    int  get_fem_body_count() const;
    int  get_vertex_count_host() const;
    void get_vertex_position_host(int idx, double out_xyz[3]) const;

    void finalize();

    void step();

    // ---- State queries ----
    int      get_vertex_count() const;
    int      get_surface_face_count() const;
    int      get_surface_vertex_count() const;

    void     get_vertex_positions(double* out_xyz, int count) const;
    void     get_surface_faces(uint32_t* out_idx, int face_count) const;
    void     get_surface_vertex_indices(uint32_t* out_idx, int count) const;

    // ---- ABD body state (GPU readback) ----
    // Read 4x4 transform matrices for ABD bodies at given offsets.
    // q = [p, a1, a2, a3] -> mat4[:3,:3] = A^T, mat4[:3,3] = p
    void get_abd_body_transforms(const int* body_offsets, double* out_mat4x4, int count) const;
    void set_abd_body_transforms(const int* body_offsets, const double* mat4x4, int count);

    // Teleport ABD bodies: sets q, q_prev, q_tilde, q_temp to the new state
    // and zeros q_v and dq.  Use this for initialization/reset to avoid
    // phantom velocities from stale q_prev.
    void teleport_abd_bodies(const int* body_offsets, const double* mat4x4, int count);

    // Read/write ABD velocity as 4x4 matrices (velocity of q).
    void get_abd_body_velocities(const int* body_offsets, double* out_mat4x4, int count) const;
    void set_abd_body_velocities(const int* body_offsets, const double* mat4x4, int count);

    // ---- FEM vertex state ----
    void get_vertex_velocities(double* out_xyz, int count) const;
    void set_vertex_positions_gpu(const double* xyz, int count);
    void set_vertex_velocities_gpu(const double* xyz, int count);
    void get_fem_body_vertex_range(int fem_body_idx, int* out_start, int* out_count) const;

    // ---- Load record tracking ----
    int  get_load_record_count() const;
    const BodyLoadRecord& get_load_record(int idx) const;

    // ---- Joint control ----
    int  get_num_revolute_joints() const;
    int  get_num_prismatic_joints() const;

    JointInfo get_revolute_joint_info(int idx) const;
    JointInfo get_prismatic_joint_info(int idx) const;

    void set_revolute_target(int idx, double angle_rad);
    void set_revolute_initial_offset(int idx, double offset_rad);
    void set_prismatic_target(int idx, double distance_m);

    /// Set per-joint driving strength multiplier.
    /// Effective stiffness = Config.revolute_driving_strength_ratio *
    /// strength * (m_parent + m_child). Default 1.0.
    /// Lower values (e.g. 0.1) make the joint "give way" under contact —
    /// useful for gripper fingers that should yield when pressing against
    /// cloth instead of crushing it thin (which triggers barrier-Kappa
    /// cascade and slows Newton). Applied starting next step().
    void set_revolute_strength(int idx, double strength);
    void set_prismatic_strength(int idx, double strength);

    double get_revolute_target(int idx) const;
    double get_prismatic_target(int idx) const;

    void   get_revolute_current_angles(double* out, int count) const;

    std::string get_assets_dir() const;

  private:
    struct Impl;
    Impl* m_impl;
};

}  // namespace gipc

#endif  // STIFFGIPC_SIM_ENGINE_H
