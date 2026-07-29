#pragma once
#ifndef STIFFGIPC_SIM_ENGINE_H
#define STIFFGIPC_SIM_ENGINE_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <Eigen/Core>
#include "frame_fsm/frame_status.cuh"

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
    // [uipc-style, opt-in] physical Newton exit: converged when the max vertex
    // displacement of the step <= newton_velocity_tol * dt (units m/s; uipc
    // default is 0.05). 0 = OFF (legacy newton_tol * length-scale * dt keeps
    // exact current behavior). Scene-size and env-count independent by
    // construction; also makes relative_dhat fully inert for the exit check.
    double newton_velocity_tol = 0.0;
    double pcg_tol             = 1e-4;
    double relative_dhat       = 1e-3;

    double joint_strength_ratio            = 100.0;
    double revolute_driving_strength_ratio = 100.0;
    double prismatic_strength_ratio        = 100.0;
    double prismatic_driving_strength_ratio = 100.0;

    // Per-frame driving target slew limits. These are part of solver semantics:
    // interactive soft-FEM demos often use small limits, while Newton/IsaacLab
    // articulation position targets expect much larger immediate target motion.
    double max_revolute_step_per_frame  = 0.1;    // rad
    double max_prismatic_step_per_frame = 0.002;  // m

    double collision_detection_buff_scale = 1.0;
    double linear_system_buff_scale       = 1.0;
    // Margin multiplier on the INTERNAL (FEM/ABD/joint) Hessian-triplet buffer,
    // reserved for the Strategy-D chain-rule pin-pin expansion range. Hardcoded
    // 32 historically; measured ~0 extension for non-rigid-heavy scenes, so it
    // is the dominant per-env over-reservation and the wall for scaling envs.
    // Lower it (e.g. 4) for multi-env; keep 32 for Strategy-D hybrid meshes
    // that actually expand.
    double triplet_internal_margin        = 32.0;
    // Absolute contact distance (meters). >0 pins dHat to this value instead of
    // deriving it from the merged-scene bbox diagonal (which inflates with env
    // count/spacing -> super-linear contact). Set to the single-env dHat_sqrt
    // for multi-env. 0 = legacy bbox-derived behavior (no change for single-env).
    double absolute_dhat                  = 0.0;
    int    preconditioner_type            = 1;  // 1 = MAS
    int    cuda_device                    = 0;

    bool   semi_implicit_enabled  = false;
    double semi_implicit_beta_tol = 1e-3;
    int    semi_implicit_min_iter = 1;
    // [per-env productization] per-env Newton iteration budget: an env still
    // active at this iter is force-frozen with status TIMEOUT (others continue
    // unaffected). 0 = off. Host per-env path only.
    int    env_newton_iter_cap    = 0;
    // [T1] line-search backtracking budget before the engine reports a
    // non-descent step. 8 was the old hard-coded value (bad steps accepted
    // silently); 64 matches libuipc.
    int    line_search_max_iter   = 64;
    // Optional shared energy-comparison tolerance for merged and per-env line
    // search. Defaults to exact non-increase, matching libuipc's IPC path:
    // E1 <= E0. Nonzero values explicitly allow
    // E1 <= E0 + abs_tol + rel_tol*|E0| (plus Armijo when enabled).
    double energy_abs_tol         = 0.0;
    double energy_rel_tol         = 0.0;
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

    // Per-frame solver log verbosity: 0 = silent (engine plays nice in
    // co-simulation / piped tooling), >=1 = verbose (default).
    void set_log_level(int level);

    // Tear down the whole world (bodies, FEM/ABD, constraints, GPU buffers) and
    // return to a fresh empty state, preserving Config.  Re-run load_*()+finalize().
    void reset();

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
    /// Declare one group id per collision body. Non-negative ids must be dense
    /// [0,N), N<=256. Wildcard -1 is supported only by merged-mode execution;
    /// isolated/strict require every body to be grouped. Validated at finalize().
    void set_body_groups(const std::vector<int>& groups);
    // [multi-env subscene] Per-VERTEX env id (length = engine vertex count). The
    // broad-phase skips contact pairs whose two vertices carry different (>=0) env
    // ids -> cross-env isolation without spatial separation, uniform for FEM+ABD.
    void set_vertex_env_ids(const std::vector<int>& env_ids);
    // [multi-env determinism 4.1] per-GROUP world offset (flat xyz, 3 per group). The engine
    // expands to per-vertex (via point->body->group) and the BVH builds on _vertexes+offset
    // (envs spatially separated) while narrow-phase stays on local _vertexes (deterministic).
    // Call AFTER finalize(). Keep the up-axis component 0 so ground contact stays shared.
    void set_env_offsets(const std::vector<double>& per_group_xyz);
    void add_ground_collision_skip(int body_id);

    /// Stitch a FEM vertex to an ABD body via a soft spring constraint.
    /// At each step, the FEM vertex is pulled toward
    ///   target_world = world_pos(_bodyId-th vertex of abd_body) + rest_offset_world
    /// where rest_offset_world is a world-frame offset (NOT body-local).
    /// For best behavior with ABD rotation, place the FEM vertex coincident
    /// with the chosen ABD anchor vertex at finalize time (rest_offset = 0).
    ///
    /// Must be called before finalize().
    /// Spring stiffness is controlled by Config::soft_motion_rate.
    /// Internally appends to tetMesh.targetIndex / tetMesh.targetPos and
    /// d_tetMesh.stitch_paired_vertex / .stitch_rest_offset / .stitch_abd_body_id,
    /// bumping tetMesh.softNum.
    void add_stitch_spring(int fem_vertex_global_id,
                           int abd_anchor_vertex_global_id,
                           int abd_body_id,
                           const Eigen::Vector3d& rest_offset_world);

    /// [Per-region Young's modulus] Override per-tet Young's modulus for one
    /// FEM body. `body_offset` = index into get_load_records().  The array
    /// length must equal the number of tets owned by that body (a tet is
    /// "owned" if all 4 of its vertices lie inside the body's vertex range).
    /// Tets are matched in their loaded order.
    ///
    /// Must be called BEFORE finalize() — vert_youngth_modules[] is read at
    /// finalize-time to compute per-tet Lame parameters which then go to GPU.
    /// After finalize, changes here have no effect (data is on device).
    void set_per_tet_young_for_body(int body_offset,
                                    const std::vector<double>& per_tet_young);

    /// [per-body density] Override one SOFT body's density (FEM tet volume or
    /// cloth shell). `body_offset` = index into get_load_records(). Applies to
    /// every tet (all 4 verts inside the body's vertex range) and every
    /// triangle shell element (all 3 verts inside) owned by that body; the
    /// mass build at finalize then uses it instead of the global
    /// Config.density / cloth_density. Must be called AFTER the body is loaded
    /// and BEFORE finalize(). Throws if the record owns no soft elements
    /// (e.g. an ABD body — use set_abd_body_density for those).
    void set_soft_body_density(int body_offset, double density);

    /// [per-body friction] Override one body's friction coefficient.
    /// `body_offset` = index into get_load_records(); applies to every vertex
    /// of that body (ABD or FEM/cloth). Self-contact pairs combine the two
    /// sides' mu geometrically (sqrt(mu_a * mu_b)); ground contact uses
    /// `ground_mu` for this body's vertices (< 0 = keep the global
    /// gd_friction_rate). Call AFTER loading the body and BEFORE finalize().
    /// If no body is ever overridden the engine runs the legacy global-mu
    /// path bit-identically.
    void set_body_friction(int body_offset, double mu, double ground_mu = -1.0);

    /// [contact-force distribution] Per-vertex physical contact FORCES (N,3)
    /// in NEWTONS: -gradient/dt^2 (body-body barrier; ground barrier added
    /// when include_ground; friction NOT included). Rebuilds contacts once
    /// (BVH+CP); call between frames. NOTE: units deliberately DIFFER from the
    /// legacy batched API, which returns raw gradients (-force*dt^2) (sum over a
    /// body's vertices = its net contact force). Writes min(n, vertexNum)
    /// triples into out3; returns the number written.
    /// components: 0 = normal only (historic), 1 = friction_lagged only
    /// (the friction-potential gradient the solver actually used this step —
    /// positions current, lambda/tangent basis lagged), 2 = total.
    int get_vertex_contact_forces(double* out3, int n, bool include_ground = true,
                                  int components = 0);

    /// [FEM stress] Per-vertex von Mises stress (Pa) of the current state:
    /// per-tet Cauchy stress from the configured tetrahedral constitutive law
    /// -> von Mises -> per-vertex MAX over incident tets. Vertices not in any
    /// tet (cloth, ABD) get 0. Writes min(n, vertexNum) values.
    int get_fem_von_mises_stress(double* out, int n);

    /// [per-env productization] Newton iter at which each env froze last solve
    /// (converged / timeout / diverged; -1 = ran to loop end or absent).
    /// 256 slots. Host per-env path only (per_env_exit / STIFF_PERENV_ALPHA).
    std::vector<int> get_per_env_newton_iters() const;
    /// [per-env productization] Per-env status of the last solve:
    /// 0 active/absent, 1 converged, 2 timeout, 3 diverged. 256 slots.
    std::vector<int> get_per_env_status() const;

    /// [FEM-pin] Hard-constraint pin: forces a FEM vertex's world position
    /// to follow an ABD anchor vertex by a fixed offset, EXACTLY (no soft
    /// spring error). Acts like a stitch spring with infinite stiffness —
    /// implemented as a direct projection kernel applied right after each
    /// line-search step (so IPC's contact barrier sees the corrected FEM
    /// pose).
    ///
    /// When to use vs add_stitch_spring:
    ///   spring: bilateral force-based coupling, FEM can lag ABD under
    ///           fast motion (line-search retreats), rest length sets
    ///           how slack the connection is.
    ///   pin:    FEM vertex *equals* abd_anchor + offset every step.
    ///           ABD doesn't feel any reaction force from the pin (no
    ///           condition-number blowup), and FEM tracks ABD perfectly.
    ///           Useful for "rigidly-attached" parts of a deformable
    ///           body (e.g. soft gripper finger pad fused to a rigid
    ///           finger backbone).
    ///
    /// Must be called before finalize().
    void add_fem_pin_to_abd(int fem_vertex_global_id,
                            int abd_anchor_vertex_global_id,
                            int abd_body_id,
                            const Eigen::Vector3d& rest_offset_world);

    /// [Hybrid mesh] bulk-add FEM pins with EXPLICIT local positions.
    ///
    /// This is the per-vertex pin API designed for hybrid ABD-FEM mesh
    /// scenarios where the rigid region of a continuous tet mesh is
    /// kinematically driven by an ABD body.  Use this when you already
    /// know each pinned vertex's coordinate in the ABD body's REST frame
    /// (e.g. from tools/build_hybrid_mesh.py output's vertex_local_pos).
    ///
    /// Difference from add_fem_pin_to_abd:
    ///   - No anchor vertex needed (no need to identify a paired ABD vert).
    ///   - local_pos is taken AS-IS, not re-derived in finalize() from a
    ///     world-space rest offset.
    ///   - Fast bulk path: avoids per-pin Python<->C++ round-trips at
    ///     hybrid-mesh scales (1k+ pins).
    ///
    /// The three input vectors must have the same length n_pins.
    /// Must be called before finalize().
    void add_fem_pins_with_local_pos(
        const std::vector<int>&             fem_vertex_global_ids,
        const std::vector<int>&             abd_body_ids,
        const std::vector<Eigen::Vector3d>& abd_local_positions);

    /// libuipc-style per-face orient labels for an ABD surface body.
    /// Pass one int per triangle, in {-1, 0, +1}; non-zero values flip
    /// (or preserve) the face's normal sign at mass/centroid/inertia
    /// integration time WITHOUT mutating face vertex order. Empty vector
    /// = revert to "use winding as-is" (default behavior).
    /// MUST be called before finalize() (after that, surface mesh data
    /// has already been copied to ABDSystem).
    /// Returns false if body_id has no surface mesh body in tetMesh
    /// (e.g. it's a FEM body or out-of-range).
    bool set_abd_body_face_orient(int body_id, const std::vector<int>& orient);

    /// Read current per-face orient labels for an ABD surface body.
    /// Returns empty vector if body has no orient overrides set.
    std::vector<int> get_abd_body_face_orient(int body_id) const;

    /// Pre-finalize accessors for per-body surface mesh data — needed by
    /// Python helpers that compute orient labels before finalize().
    /// Returns flattened (N,3) double for verts, (M,3) int for faces.
    /// Empty if body_id has no surface-mesh body in tetMesh.
    std::vector<double> get_abd_surface_body_vertices(int body_id) const;
    std::vector<int>    get_abd_surface_body_triangles(int body_id) const;

    // ---- Programmatic joint creation (must call before finalize()) ----
    // Parent/child IDs may be in either numeric order. Invalid or identical IDs
    // throw std::invalid_argument. Returns the constraint index.

    int add_fixed_joint(int parent_body, int child_body,
                        const Eigen::Vector3d& world_anchor,
                        const Eigen::Vector3d& world_normal,
                        const Eigen::Vector3d& world_bitangent);

    /// [passive joints] passive=true creates a PURE hinge/slider: the position
    /// servo (strength_ratio) is zeroed so the joint swings/slides freely under
    /// physics; URDF limits still act (independent joint_limit penalty).
    /// passive=false (default) keeps the historic position-servo behavior.
    int add_revolute_joint(int parent_body, int child_body,
                           const Eigen::Vector3d& world_axis,
                           const Eigen::Vector3d& joint_pos,
                           double lower_limit, double upper_limit,
                           double initial_angle = 0.0,
                           const std::string& name = "",
                           bool passive = false);

    int add_prismatic_joint(int parent_body, int child_body,
                            const Eigen::Vector3d& world_center,
                            const Eigen::Vector3d& world_axis,
                            double lower_limit, double upper_limit,
                            const std::string& name = "",
                            bool passive = false);

    // Per-vertex boundary type (0=Free, 1=Fixed). Must call before finalize().
    void set_vertex_boundary(int vertex_index, int boundary_type);

    int  get_abd_body_count() const;
    int  get_fem_body_count() const;
    int  get_vertex_count_host() const;
    void get_vertex_position_host(int idx, double out_xyz[3]) const;
    /// Batch host-side vertex getter (zero device sync, reads from
    /// tetMesh.vertexes directly). Useful PRE-finalize for setting up
    /// stitch springs / joint constraints / etc.
    void get_vertex_positions_host(double* out_xyz, int count) const;

    /// Upload and initialize the scene. Several solver buffers are published
    /// through process-global CUDA symbols, so only one finalized SimEngine
    /// may be active per process; reset or destroy it before finalizing another.
    void finalize();

    void step();

    // ---- Phase D: episode-resident RL execution ----
    //
    // A warm-up step() is required first so all lazy CUDA workspaces have
    // stable addresses. Actions are contiguous `(frames, joints, 3)` arrays:
    //   revolute = {target_angle, strength_ratio, external_torque}
    //   prismatic = {target_distance, strength_ratio, external_force}
    // The complete sequences are uploaded once, then one asynchronous CUDA
    // Graph launch consumes every frame without a host wait.
    void launch_episode_async(
        int frames,
        const double* revolute_actions,
        int revolute_joints,
        const double* prismatic_actions,
        int prismatic_joints);
    bool episode_in_flight() const;
    bool episode_observation_ready(int slot) const;
    void wait_episode_observation(int slot) const;
    int  get_episode_slot_first_frame(int slot) const;
    int  get_episode_slot_frame_count(int slot) const;
    int  get_episode_attempted_frame_count() const;
    void get_episode_observation(
        int slot,
        double* positions,
        double* velocities,
        frame_fsm::FrameStatus* statuses,
        int frame_capacity) const;
    // Wait for the terminal slot, commit successful-frame telemetry, and
    // return the number of successfully committed frames.
    int finish_episode();

    // ---- GPU-native RL device ABI ----
    //
    // prepare_gpu_rl() is a setup boundary and may allocate, capture, upload,
    // and synchronize. After it returns, an external CUDA policy writes the
    // packed float64 action buffers directly, then calls
    // launch_gpu_rl_async() on the same CUDA stream. The steady-state launch
    // contains no H2D/D2H graph nodes and performs no host synchronization.
    // Device outputs use engine-internal vertex order.
    void prepare_gpu_rl();
    void launch_gpu_rl_async(uintptr_t cuda_stream = 0);
    bool gpu_rl_prepared() const;
    bool gpu_rl_ready() const;
    void synchronize_gpu_rl() const;  // Explicit debug/teardown boundary.
    void end_gpu_rl();                 // Synchronizes before releasing buffers.
    uintptr_t get_gpu_rl_revolute_actions_device_ptr() const;
    uintptr_t get_gpu_rl_prismatic_actions_device_ptr() const;
    uintptr_t get_gpu_rl_positions_device_ptr() const;
    uintptr_t get_gpu_rl_velocities_device_ptr() const;
    uintptr_t get_gpu_rl_statuses_device_ptr() const;
    uintptr_t get_gpu_rl_frame_counter_device_ptr() const;
    // [D2] device joint observations ({angle,rate} per revolute driving
    // joint then {disp,rate} per prismatic driving joint, float64), written
    // by the simulation graph itself each step.
    uintptr_t get_gpu_rl_joint_observations_device_ptr() const;
    int       get_gpu_rl_joint_observation_count() const;
    // [D2] in-stream reset: replay the prepare-time state snapshot with
    // device copies on the bound stream — no host synchronization. Episode
    // bookkeeping (frame counter, reward accumulators) is the caller's.
    void      launch_gpu_rl_reset_async(uintptr_t cuda_stream = 0);
    // [D3] device env-partition handles for merged multi-env batching:
    // point-to-group map (int per vertex, -1 = wildcard) and the per-env
    // quarantine flags (int per env; nonzero = poisoned, treat as done).
    uintptr_t get_point_to_group_device_ptr() const;
    uintptr_t get_env_quarantined_device_ptr() const;
    int       get_env_group_count() const;
    int get_gpu_rl_graph_node_count() const;
    int get_gpu_rl_graph_h2d_count() const;
    int get_gpu_rl_graph_d2h_count() const;
    int get_gpu_rl_status_size_bytes() const;

    /// Result, work counters and capacity high-water marks for the latest
    /// frame.  In whole-frame graph mode this is the single terminal D2H
    /// packet; the legacy path publishes the same ABI for uniform tooling.
    frame_fsm::FrameStatus get_frame_status() const;

    // ---- State queries ----
    int      get_vertex_count() const;
    uintptr_t get_vertices_device_ptr() const;  // [gpu-direct] double3* device ptr to vertex buffer
    uintptr_t get_vertex_velocities_device_ptr() const;
    int      get_surface_face_count() const;
    int      get_surface_vertex_count() const;

    void     get_vertex_positions(double* out_xyz, int count) const;
    // [decouple] per-vertex env/group id ALIGNED to get_vertex_positions order (input order):
    // out[v] = body_groups[point_id_to_body_id[v]] (-1 if ungrouped). Lets Python extract a single
    // env's verts (verts[groups==g]) for batch-invariance tests regardless of the type-grouped layout.
    void     get_point_groups(int* out, int count) const;
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

    /// Set per-frame Animated-target for an ABD body that was loaded with
    /// boundary_type=Animated (=3).  Each step the engine adds a soft
    /// quadratic penalty
    ///     E += 0.5 * strength * (||q.t - target||^2 + ||A(q) - I||_F^2)
    /// to the body's energy.  This keeps the body's translation tracking
    /// `target` and its rotation near identity, without removing the body's
    /// 12 DOFs from the PCG system — so M3.5 chain-rule pins to this body
    /// still propagate correctly, and standard joint constraints can attach
    /// to the body's q.  Coordinates are in world space (meters).
    /// Strength <= 0 falls back to the default 1e6.
    void set_body_animated_target(int body_id,
                                  double target_x, double target_y, double target_z,
                                  double strength = 0.0);

    /// Toggle gravity for all vertices of a body, applied at next step.  Use
    /// this for ABD bodies that are kinematically driven by joints to a
    /// Fixed parent — e.g. a gripper hanging off an arm link via revolute
    /// joint.  Without disabling, joint penalty must continuously cancel
    /// gravity each step, leaving residual that accumulates as drift +
    /// destabilizes Newton when combined with chain-rule pins.  body_id is
    /// the GLOBAL body id (ABD bodies first, then FEM).  Must call AFTER
    /// finalize() (writes to GPU buffer directly).
    void set_body_apply_gravity(int body_id, bool enabled);

    /// Override one ABD body's density (mass = density * volume). Call after
    /// loading the body and before finalize().
    void set_abd_body_density(int body_id, double density);

    /// Override one surface-mesh ABD body's total mass in kilograms. Call after
    /// loading the body and before finalize().
    void set_abd_body_mass(int body_id, double mass);

    /// Override one ABD body's inertial props (mass, COM[3], inertia[9] row-major
    /// 3x3 about COM, all in the load/world frame). Call after loading the body
    /// and before finalize(). Use authored URDF inertia instead of welded-mesh.
    void set_abd_body_inertia(int body_id, double mass,
                              const double* com3, const double* inertia9);

    /// [force-control] Set a per-body external LINEAR force (N) on an ABD body
    /// (global body id). Persistent until changed; (0,0,0) clears. Applied as an
    /// acceleration M^{-1}F in the q_tilde prediction, like gravity. Call AFTER
    /// finalize(). Mirrors libuipc AffineBodyExternalBodyForce (linear subset).
    void set_body_external_force(int body_id, double fx, double fy, double fz);
    void set_body_external_wrench(int body_id, const double* w12);  // full 12-DOF (linear+affine)

    /// Returns the 4×4 world transform of a URDF link, computed by the
    /// importer's forward kinematics from the link tree + joint angles.
    /// Available immediately after load_urdf() (no need to finalize).
    /// Useful for placing additional bodies (e.g. hybrid gripper) attached
    /// to a specific link with correct orientation in world space.
    /// Returns identity if link_name not found.
    Eigen::Matrix4d get_urdf_link_transform(const std::string& link_name) const;

    /// Override the mesh used for a URDF link in the next load_urdf() call.
    /// URDF importer normally reads the link's <collision> mesh filename
    /// (.obj/.stl) — but for engine bodies we usually want a tetrahedralized
    /// .msh.  Call this BEFORE load_urdf() for each link you want overridden.
    /// All overrides are consumed (cleared) by the next load_urdf().
    /// Required when the URDF references a mesh file that doesn't exist on
    /// disk — without override, URDF importer skips that link silently.
    void set_urdf_mesh_override(const std::string& link_name,
                                const std::string& msh_path,
                                double young_modulus = 1e7);

    // ---- FEM vertex state ----
    void get_vertex_velocities(double* out_xyz, int count) const;
    void set_vertex_positions_gpu(const double* xyz, int count);
    void set_vertex_velocities_gpu(const double* xyz, int count);
    void get_fem_body_vertex_range(int fem_body_idx, int* out_start, int* out_count) const;

    // Teleport FEM vertices to new positions: writes _vertexes (current),
    // o_vertexes (previous-step committed), and xTilta (predictor) so that
    // the next engine.step() does NOT revert to the stale previous position.
    //
    // If ``velocities`` is non-null, it also writes velocities and extends
    // xTilta to x + v*dt + g*dt^2, preserving inertia across the handoff.
    // Pass nullptr to zero velocities (matches teleport_abd_bodies default
    // semantics).
    void teleport_fem_vertices(const double* xyz, int count,
                               const double* velocities = nullptr);

    // ---- frame-boundary integrator checkpoint ----
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    // ---- Load record tracking ----
    int  get_load_record_count() const;
    const BodyLoadRecord& get_load_record(int idx) const;

    // ---- Joint control ----
    int  get_num_revolute_joints() const;
    int  get_num_prismatic_joints() const;

    JointInfo get_revolute_joint_info(int idx) const;
    JointInfo get_prismatic_joint_info(int idx) const;

    void set_revolute_target(int idx, double angle_rad);

    /// [force-control] Set external torque (N*m) on revolute driving joint idx.
    /// Adds -tau*dtheta/dq to the driving gradient (no Hessian), independent of
    /// the PD term. For pure torque control also set_revolute_strength(idx, 0).
    void set_revolute_torque(int idx, double torque);
    void set_revolute_initial_offset(int idx, double offset_rad);
    void set_prismatic_target(int idx, double distance_m);
    void set_prismatic_force(int idx, double force);  // [force-control] external prismatic force (N)
    void set_prismatic_limit_barrier(int idx, double cl, double dir, double dhat, double kappa, int slot = 0);  // [force-control] one-sided IPC barrier; slot 0=closed end, 1=open end (hard no-overshoot both ways)
    double get_prismatic_drive_force(int idx) const;  // [force-control] current K*(target-d) drive force
    double get_prismatic_current_distance(int idx) const;  // [force-control] current opening d along axis
    void   get_vertex_contact_force_sum(int vert_offset, int vert_count, double* out3) const;  // [force-control] net IPC contact force on a body
    void   get_body_contact_force_batched(const int* offsets, const int* counts, int n_seg, double* out3) const;  // [force-control] BATCHED: rebuild contacts ONCE, sum per segment (finger/env) -> n_seg 3-vectors, ONE D2H

    /// Net IPC contact force on body A FROM body B (barrier gradient on A's
    /// vertices, restricted to pairs connecting ranges A and B). For the
    /// contact sensor's per-partner force_matrix. Call AFTER step().
    void   get_pair_contact_force(int a_off, int a_cnt, int b_off, int b_cnt, double* out3) const;

    /// "Clean export layer": decode the current body-body collision pairs into
    /// clean vertex-index 4-tuples (out_flat[4*i..], -1 padded). Returns the
    /// pair count. Pass out_flat=nullptr to just get the count. Read-only path.
    int    get_collision_pairs_clean(int* out_flat) const;

    /// [Step B] GPU-resident per-contact force export for the Newton ContactSensor.
    /// compute_contacts() fills grow-only device buffers and returns the contact
    /// count; contacts_pair_ptr()/contacts_force_ptr() return raw device pointers
    /// (int2 (bodyA,bodyB); double3 world force on bodyA, N). Call AFTER step();
    /// the buffers are valid until the next compute_contacts() call.
    int       compute_contacts(bool rebuild = false);
    uintptr_t contacts_pair_ptr() const;
    uintptr_t contacts_force_ptr() const;
    double get_stitch_max_stretch(int pair_start, int pair_count) const;  // [force-control] on-GPU max stitch stretch over a spring range (scalar; no full-vertex D2H)
    void   get_stitch_max_stretch_batched(const int* starts, const int* counts, int n_seg, double* out) const;  // [force-control] BATCHED: one block per segment (finger/env), one launch, per-segment maxes — multi-env isolated

    /// Override per-fixed-joint stiffness (kappa).  By default kappa is set
    /// at finalize as `joint_strength_ratio * (m_parent + m_child)` for ALL
    /// joints (including URDF revolute constraint points + manually added
    /// fixed joints).  When a hybrid gripper is welded to a URDF arm hand
    /// via fixed_joint, the default kappa (~8e-3) is far too weak to hold
    /// the gripper rigid against the arm — it lags 8mm+ per cm of hand
    /// motion.  Use this to set the fixed_joint kappa directly (e.g. 1e6
    /// matches Animated PD strength). idx = constraint index returned by
    /// add_fixed_joint().  Call AFTER finalize().
    void set_fixed_joint_strength(int idx, double kappa);

    /// Set per-joint driving strength multiplier.
    /// Effective stiffness = Config.revolute_driving_strength_ratio *
    /// strength * (m_parent + m_child). Default 1.0.
    /// Lower values (e.g. 0.1) make the joint "give way" under contact —
    /// useful for gripper fingers that should yield when pressing against
    /// cloth instead of crushing it thin (which triggers barrier-Kappa
    /// cascade and slows Newton). Applied starting next step().
    void set_revolute_strength(int idx, double strength);
    void set_prismatic_strength(int idx, double strength);

    /// Set the maximum revolute joint angle change per IPC step (in radians).
    /// Default 0.1 rad ≈ 5.7°.  For scenes with fine FEM softpads pinned to
    /// ABD bodies, large per-step rotation triggers FEM mesh self-intersection
    /// (the kinematic teleport of pinned vertices outpaces the elastic
    /// response of free neighbors).  Lower values (e.g. 0.01 rad ≈ 0.6°)
    /// keep softpad self-collision tame.  Applied starting next step().
    void set_max_revolute_step_per_frame(double rad);
    void set_max_prismatic_step_per_frame(double m);

    double get_revolute_target(int idx) const;
    double get_prismatic_target(int idx) const;

    void   get_revolute_current_angles(double* out, int count) const;

    /// Read each revolute joint's `initial_angle_offset` — the URDF-frame
    /// angle the joint was at when `load_urdf(initial_joint_angles=...)`
    /// applied FK at load time.  Used to convert the relative angle
    /// returned by `get_revolute_current_angles` to absolute URDF angle:
    ///   absolute_urdf = relative + initial_offset
    /// Returns 0.0 per joint when no `initial_joint_angles` was passed.
    void   get_revolute_initial_offsets(double* out, int count) const;

    std::string get_assets_dir() const;

    // ---- Per-step counters (perf debugging; cumulative since process start) ----
    // Read directly from GIPC.cu file-scope globals. Caller computes per-step delta.
    int    get_total_newton_iters() const;
    double get_total_pcg_iters() const;
    // [rl-reset] step-health telemetry (per-instance, monotone): line-search
    // budget exhaustions and the subset with non-finite incremental
    // potential. Diff across step() to detect a degraded step; merged mode
    // WARNs and continues by contract, so this is the RL loop's discard
    // signal for poisoned episodes.
    int    get_ls_exhausted_count() const;
    int    get_ls_nonfinite_count() const;
    double get_total_collision_pairs() const;
    double get_max_collision_pairs() const;
    int    get_total_frames_done() const;
    uint64_t get_total_energy_tolerance_accepts() const;
#ifdef GIPC_ENABLE_DIAGNOSTICS
    // Intrusive test-only diagnostics; omitted from production builds.
    std::vector<double> debug_fd_gradient_check(double h, int nprobes, unsigned seed);
    std::vector<double> debug_fd_hessian_check(double h, int nprobes, unsigned seed);
    std::vector<double> debug_fd_activity();
#endif


  private:
    struct Impl;
    Impl* m_impl;
};

}  // namespace gipc

#endif  // STIFFGIPC_SIM_ENGINE_H
