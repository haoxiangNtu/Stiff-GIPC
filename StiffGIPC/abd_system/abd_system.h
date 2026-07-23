#pragma once
#include <abd_system/abd_sim_data.h>
#include <abd_system/abd_joint_constraint.h>
#include <abd_system/abd_driving_joint.h>
#include <joint_constraint_host_info.h>
#include <joint_angle_control.h>
#include <muda/buffer/device_var.h>
#include <muda/ext/linear_system.h>
#include <abd_system/abd_system_parms.h>
#include <linear_system/utils/converter.h>
#include "linear_system/linear_system/global_matrix.h"
#include <Eigen/Dense>
#include <unordered_map>
namespace gipc
{

/// Host-side surface mesh data for ABD bodies loaded from triangle meshes.
/// Used by ABDSystem to compute mass/volume/gravity via surface integrals.
struct ABDSurfaceMeshBody
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    /// Per-face orientation override (libuipc-style), values in {-1, 0, +1}.
    /// Empty (the default) means "use face winding as-is" — preserves
    /// pre-orient-feature behavior for assets that already have correct
    /// winding. Set per-face to -1 to flip a single face's normal at
    /// integration time without mutating the topology, so collision /
    /// BVH / render code still see the original face order.
    std::vector<int>             orient;
    int body_id     = -1;
    int point_start = -1;  // first unique_point_id belonging to this body
    int point_count = 0;   // number of unique points in this body
};
class ABDSystem
{
  private:
    muda::DeviceBuffer<Vector12>  m_temp_q;
    muda::DeviceBuffer<Vector12>  m_temp_q_v;
    muda::DeviceBuffer<Vector12>  m_temp_q_prev;
    muda::DeviceBuffer<Vector12>  m_temp_q_tilde;
    muda::DeviceBuffer<int>       m_temp_is_fixed;
    muda::DeviceBuffer<ABDJacobi> m_temp_jacobi;
    muda::DeviceBuffer<Vector12>  m_temp_abd_gravity_force;
    muda::DeviceVar<Float>        m_kinetic_energy;
    muda::DeviceBuffer<Float>     m_kinetic_energy_per_affine_body;

    muda::DeviceVar<Float>      m_shape_energy;
    muda::DeviceBuffer<Float>   m_shape_energy_per_affine_body;
    muda::DeviceBuffer<Vector3> m_body_centered_positions;

    Float                     m_suggest_max_tolerance = 0.0f;
    muda::DeviceBuffer<Float> m_local_tolerance;
    muda::DeviceVar<Float>    m_local_tolerance_max;

  public:  // public just for convenience
    int*                                  fem_boundary_type;
    muda::DeviceBuffer<Vector12>          abd_gradient;  // just for legacy code
    muda::DeviceBuffer<Matrix12x12>       abd_body_hessian;
    GIPCTripletMatrix*   global_triplet;
    muda::DeviceDenseMatrix<double>       dense_system_hessian;
    muda::DeviceCSRMatrix<double>         csr_system_hessian;
    muda::DeviceDenseVector<double>       system_gradient;
    muda::DeviceDenseVector<double>       temp_system_gradient;
    muda::DeviceDoubletVector<double, 12> doublet_system_gradient;
    muda::DeviceBuffer<Matrix12x12>       abd_system_diag_preconditioner;

    // [multi-env determinism 4.3] binned accumulators for the ABD assembly (system_gradient,
    // body_hessian, wrench) — replace muda atomic_add with order-independent binned deposits
    // so the every-frame arm solve is deterministic. Raw device ptrs (lazily grown); the
    // assembly kernels reach them via g_abd_* device globals (set by memcpyToSymbol).
    double* m_abd_sysbin    = nullptr; size_t m_abd_sysbin_cap    = 0;
    double* m_abd_hessbin   = nullptr; size_t m_abd_hessbin_cap   = 0;
    double* m_abd_wrenchbin = nullptr; size_t m_abd_wrenchbin_cap = 0;
    // [4.3] reused scratch for the SETUP-time mass binning (unique_point_mass / body_mass /
    // mass_center / dyadic / volume — the ABD mass matrix M root). Sized per-quantity.
    double* m_massbin       = nullptr; size_t m_massbin_cap       = 0;
    void _massbin_prep(size_t total_components);   // grow+zero+bind g_massbin

    muda::DeviceBuffer<Vector3> body_mass_center;
    muda::DeviceBuffer<Float>   body_mass;
    muda::DeviceBuffer<int>     body_unique_point_count;
    // Vector6: [x0,x1,x2,x3,x4,x5]
    // Axis from [x0,x1,x2] to [x3,x4,x5]
    // Vector6 == Zero for non-motorized body
    muda::DeviceBuffer<Vector6> body_id_to_motor_rotation_axis;
    gipc::Converter             converter3x3;

    size_t triplet_vertex_hessian_reserve_size = 0;
    size_t abd_system_hessian_reserve_size     = 0;

    // ---- Joint Constraint Data ----
    int                                       m_num_joints = 0;
    muda::DeviceBuffer<JointConstraintGPUData> m_joint_data;       // [num_joints]
    muda::DeviceBuffer<Matrix12x12>            m_joint_cross_hessian; // [num_joints] H_pc
    muda::DeviceVar<Float>                     m_joint_energy;
    muda::DeviceBuffer<Float>                  m_joint_energy_per_joint; // [num_joints]

    // ---- Revolute Driving Joint Data ----
    int                                         m_num_revolute_driving = 0;
    muda::DeviceBuffer<RevoluteDrivingGPUData>   m_revolute_driving_data;
    muda::DeviceBuffer<Matrix12x12>              m_revolute_driving_cross_hessian;
    muda::DeviceVar<Float>                       m_revolute_driving_energy;
    muda::DeviceBuffer<Float>                    m_revolute_driving_energy_per;

    // ---- Prismatic Joint Constraint Data ----
    int                                         m_num_prismatic = 0;
    muda::DeviceBuffer<PrismaticJointGPUData>    m_prismatic_data;
    muda::DeviceBuffer<Matrix12x12>              m_prismatic_cross_hessian;
    muda::DeviceVar<Float>                       m_prismatic_energy;
    muda::DeviceBuffer<Float>                    m_prismatic_energy_per;

    // ---- Prismatic Driving Joint Data ----
    int                                         m_num_prismatic_driving = 0;
    muda::DeviceBuffer<PrismaticDrivingGPUData>  m_prismatic_driving_data;
    muda::DeviceBuffer<Matrix12x12>              m_prismatic_driving_cross_hessian;
    muda::DeviceVar<Float>                       m_prismatic_driving_energy;
    muda::DeviceBuffer<Float>                    m_prismatic_driving_energy_per;

    // ---- Surface Mesh Bodies (for native surface integral path) ----
    std::vector<ABDSurfaceMeshBody> m_surface_mesh_bodies;

    // Per-body density overrides (body_id -> density). When a body_id is
    // present, _fix_surface_mesh_vertex_masses uses this density instead of
    // the global parms.mass_density, so a scene can mix densities.
    std::unordered_map<int, double> m_body_density_override;

    // Per-body total-mass overrides for surface-mesh ABD bodies. The
    // mesh-derived COM is preserved and the dyadic inertia is scaled to the
    // requested mass. A mass override takes precedence over density.
    std::unordered_map<int, double> m_body_mass_override;

    // Per-body inertial overrides (body_id -> {mass, com (world, at load time),
    // inertia 3x3 about com}). When present, _apply_surface_mesh_body_overrides
    // builds the ABD dyadic mass from THESE authored values (e.g. URDF inertial
    // tags via Newton body_mass/body_com/body_inertia) instead of from the
    // welded collision-mesh geometry — whose centroid can be far off for a
    // multi-shape link, skewing the revolute driving torque arm.
    struct InertiaOverride { double mass; Eigen::Vector3d com; Eigen::Matrix3d inertia; };
    std::unordered_map<int, InertiaOverride> m_body_inertia_override;

    // ---- Bilateral Stitch Constraint Data ----
    // Set from GIPC before each call to setup_abd_system_gradient_hessian.
    // These are GPU pointers owned by device_TetraData (not managed here).
    int        m_stitch_count              = 0;
    int*       m_d_stitch_paired_vertex    = nullptr;  // ABD unique point id per spring
    double3*   m_d_stitch_rest_offset      = nullptr;  // rest offset per spring
    int*       m_d_stitch_abd_body_id      = nullptr;  // ABD body id per spring
    uint32_t*  m_d_stitch_fem_vertex_id    = nullptr;  // FEM vertex (= targetInd)
    double3*   m_d_all_vertexes            = nullptr;  // all vertex positions
    double     m_stitch_motion_rate        = 0.0;
    double     m_stitch_rate               = 0.0;      // animation_fullRate

  public:
    ABDSystemParms parms;

    /// Override the density of one ABD body (by body_id). Call before finalize.
    void set_body_density_override(int body_id, double density)
    {
        m_body_density_override[body_id] = density;
    }

    /// Override the total mass of one surface-mesh ABD body. Call before
    /// finalize. Keep kilograms separate from density (kg/m^3).
    void set_body_mass_override(int body_id, double mass)
    {
        m_body_mass_override[body_id] = mass;
    }

    /// Override the inertial properties (mass, COM in world/load frame, 3x3
    /// inertia about the COM) of one ABD body. Call before finalize. Used to
    /// take mass/COM/inertia from authored values (URDF) instead of the welded
    /// collision-mesh geometry.
    void set_body_inertia_override(int body_id, double mass,
                                   const Eigen::Vector3d& com,
                                   const Eigen::Matrix3d& inertia)
    {
        m_body_inertia_override[body_id] = InertiaOverride{mass, com, inertia};
    }

    /******************************************************************************
    *                             build function
    *******************************************************************************/

    /// <summary>
    /// Main API: init the abd system at frame 0
    /// </summary>
    /// <param name="sim_data"></param>
    void init_system(ABDSimData& sim_data);

    /// <summary>
    /// Main API: rebuild the abd system if needed (body broken)
    /// </summary>
    /// <param name="sim_data"></param>
    void rebuild_system(ABDSimData& sim_data);
    void rebuild_system(ABDSimData& sim_data, muda::CBufferView<double3> vertices);

    // init == true, means we are at frame 0, just init the abd system
    // init == false, means we are at frame > 0, we need to rebuild the abd system
    void _setup_system(bool init, ABDSimData&);

    void _setup_unique_point_mass(size_t                     unique_point_count,
                                  muda::DeviceBuffer<Float>& unique_point_mass,
                                  muda::CBufferView<TetLocalInfo> tets,
                                  muda::CBufferView<Float>        tet_volumes,
                                  Float                           density,
                                  muda::CBufferView<int> point_id_to_unique_point_id);

    void _calculate_body_mass_center(size_t body_count,
                                     muda::DeviceBuffer<Float>& unique_point_mass,
                                     muda::CBufferView<double3> unique_point_position,
                                     muda::CBufferView<int> unique_point_id_to_body_id);


    // setup at frame 0
    void _setup_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                  muda::CBufferView<double3>     unique_point_position,
                  muda::CBufferView<int>         unique_point_id_to_body_id,
                  muda::CBufferView<Vector12>    q);

    void _setup_abd_state(size_t                        abd_count,
                          muda::DeviceBuffer<Vector12>& q,
                          muda::DeviceBuffer<Vector12>& q_temp,
                          muda::DeviceBuffer<Vector12>& q_tilde,
                          muda::DeviceBuffer<Vector12>& q_prev,
                          muda::DeviceBuffer<Vector12>& q_v,
                          muda::DeviceBuffer<Vector12>& dq);


    // if body breakup happens, we need to spawn state
    void _spawn_abd_state(muda::CBufferView<int>        body_id_to_old_body_id,
                          muda::DeviceBuffer<int>&      body_id_to_is_fixed,
                          muda::DeviceBuffer<Vector12>& q,
                          muda::DeviceBuffer<Vector12>& q_temp,
                          muda::DeviceBuffer<Vector12>& q_tilde,
                          muda::DeviceBuffer<Vector12>& q_prev,
                          muda::DeviceBuffer<Vector12>& q_v,
                          muda::DeviceBuffer<Vector12>& dq);
    // if body breakup happens, we need to spawn J
    void _spawn_J(muda::DeviceBuffer<ABDJacobi>& jacobi,
                  muda::CBufferView<int> unique_point_to_old_unique_point);


    void _setup_tet_abd_mass(muda::CBufferView<TetLocalInfo> tet_local_info,
                             muda::CBufferView<int> point_id_to_unique_point_id,
                             muda::CBufferView<ABDJacobi> jacobi,
                             muda::CBufferView<Float>     tet_volumes,
                             Float                        density,
                             muda::DeviceBuffer<ABDJacobiDyadicMass>& tet_dyadic_mass);

    void _setup_abd_dyadic_mass(size_t affine_body_count,
                                muda::CBufferView<ABDJacobiDyadicMass> tet_dyadic_mass,
                                muda::CBufferView<int> tet_id_to_body_id,
                                muda::DeviceBuffer<ABDJacobiDyadicMass>& abd_dyadic_mass,
                                muda::DeviceBuffer<Matrix12x12>& abd_dyadic_mass_inv);

    void _setup_abd_volume(size_t                     affine_body_count,
                           muda::CBufferView<int>     tet_id_to_body_id,
                           muda::CBufferView<Float>   tet_volumes,
                           muda::DeviceBuffer<Float>& abd_volume);

    void _setup_tet_abd_gravity_force(const Vector3& gravity,
                                      muda::CBufferView<TetLocalInfo> tet_local_info,
                                      muda::CBufferView<int> point_id_to_unique_point_id,
                                      muda::CBufferView<ABDJacobi> jacobi,
                                      muda::CBufferView<Float>     tet_volumes,
                                      Float                        density,
                                      muda::DeviceBuffer<Vector12>& tet_abd_gravity_force);

    void _setup_abd_gravity(muda::CBufferView<Vector12> tet_abd_gravity_force,
                            muda::CBufferView<int>      tet_id_to_body_id,
                            size_t                      affine_body_count,
                            muda::CBufferView<Matrix12x12> abd_dyadic_mass_inv,
                            muda::DeviceBuffer<Vector12>&  abd_gravity);

    // Surface mesh body overrides (host → device uploads)
    void _fix_surface_mesh_vertex_masses(muda::DeviceBuffer<Float>& unique_point_mass);
    void _fix_surface_mesh_mass_centers();
    void _apply_surface_mesh_body_overrides(ABDSimData& data);

    /*******************************************************************************
    *                                 involution
    ********************************************************************************/

    Float suggest_max_tolerance(ABDSimData& sim_data)
    {
        return m_suggest_max_tolerance;
    }

    // update veclocity from q and q_prev
    void update_velocity(ABDSimData& sim_data);
    // calculate predicted position
    void cal_q_tilde(ABDSimData& sim_data);
    // mapping q to x
    void cal_x_from_q(ABDSimData& sim_data, muda::BufferView<double3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<double3> move_dir);
    void cal_x_from_q(ABDSimData& sim_data, muda::BufferView<Vector3> vertices);
    void cal_dx_from_dq(ABDSimData& sim_data, muda::BufferView<Vector3> move_dir);


    /********************************************************************************/

    /// <summary>
    /// Main API: calculate abd system gradient and hessian.
    /// before calling this, you need to fill the `triplet_vertex_hessian` (barrier + ground barrier)
    /// </summary>
    /// <param name="sim_data"></param>
    /// <param name="vertex_barrier_gradient"></param>
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           muda::CBufferView<double3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           GIPCTripletMatrix& global_triplets,
                                           muda::CBufferView<Vector3> vertex_barrier_gradient);
    void setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                           int*        fbtype,
                                           muda::CBufferView<double3> vertex_barrier_gradient,
                                           GIPCTripletMatrix& global_triplets);

    void setup_abd_non_contact_gradient(ABDSimData& sim_data);
    void add_abd_contact_gradient(ABDSimData& sim_data,
                                  muda::CBufferView<double3> vertex_contact_gradient);

    // [multi-env determinism 4.3] open/close the ABD binned accumulators (zero+bind globals /
    // combine back into system_gradient + abd_body_hessian). Bracket the coupling-gradient
    // funcs (joint/driving/stitch/barrier) which scatter via atomic_add → binned.
    void _abd_binned_open(ABDSimData& sim_data);
    void _abd_binned_close(ABDSimData& sim_data);
    // [4.3] bracket for the FEM-pin → ABD coupling (GIPC.cu): reuse m_abd_sysbin to bin that
    // atomic_add too. n_dofs = system_gradient.size(). close combines += system_gradient.
    void couple_bin_open(int n_dofs);
    void couple_bin_close(int n_dofs);
    void _cal_abd_body_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          muda::CBufferView<double3> vertex_barrier_gradient);
    void _cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                          muda::CBufferView<Vector3> vertex_barrier_gradient);
    void _setup_abd_system_hessian(ABDSimData& sim_data,
                                   GIPCTripletMatrix& global_triplets);
    void _cal_abd_system_preconditioner(ABDSimData& sim_data);

    /********************************************************************************/

    // when doing line search, we need to copy q to q_temp
    void copy_q_to_q_temp(ABDSimData& sim_data);

    // move forward to test the energy
    // [multi-env S2] per_body_alpha (optional, device, size abd_body_num): when
    // non-null, body i steps by per_body_alpha[i] instead of the scalar alpha
    // (env g's bodies step by alpha_g so a CCD pair's verts move uniformly).
    // alpha is still used as fallback for bodies whose per_body_alpha[i] < 0.
    void step_forward(ABDSimData&                sim_data,
                      muda::BufferView<double3>  vertices,
                      double                     alpha,
                      const double*              per_body_alpha = nullptr);

    // when doing line search, we need calculate abd energy from q
    // copy_to_host=false leaves the reduced scalar on the device and avoids
    // DeviceVar's implicit blocking D2H conversion.  The default preserves the
    // public API used by diagnostics and non-line-search callers.
    Float cal_abd_kinetic_energy(ABDSimData& sim_data, bool copy_to_host = true);
    Float cal_abd_shape_energy(ABDSimData& sim_data, bool copy_to_host = true);
    Float cal_abd_joint_energy(ABDSimData& sim_data, bool copy_to_host = true);
    Float cal_abd_revolute_driving_energy(ABDSimData& sim_data,
                                          bool copy_to_host = true);
    Float cal_abd_prismatic_energy(ABDSimData& sim_data, bool copy_to_host = true);
    Float cal_abd_prismatic_driving_energy(ABDSimData& sim_data,
                                           bool copy_to_host = true);

    // Launch all six ABD energy reductions and queue their scalar results into
    // out_six[0..5] (kinetic, shape, joint, revolute drive, prismatic,
    // prismatic drive).  No device-to-host transfer is performed.
    void cal_abd_energy_DeviceOut(ABDSimData& sim_data, Float* out_six);

    // [multi-env S3] per-env ABD energy. Calls the 6 energy terms (filling their
    // per-element arrays), then segment-sums each by env (env = body_to_group of
    // the element's body; constraints keyed by parent_body_id) and ADDS into
    // env_out (device, size ng; caller pre-zeros). Returns the global ABD total.
    // body_to_group is indexed by ABD body id (== collision body id for ABD).
    double cal_abd_energy_perenv(ABDSimData& sim_data,
                                 const int*  body_to_group,
                                 int         ng,
                                 double*     env_out,
                                 bool        copy_total_to_host = true);

    // Joint constraint setup: upload from host data, compute material coords
    void init_joint_constraints(ABDSimData& sim_data,
                                const std::vector<JointConstraintHostInfo>& host_joints);

    /// Initialize revolute driving joints (sin-based angle control).
    void init_revolute_driving(ABDSimData& sim_data,
                               const std::vector<JointAngleControlInfo>& controls,
                               const std::vector<JointConstraintHostInfo>& host_joints);

    /// Update target angles for revolute driving joints (called per frame from UI).
    /// Uses incremental-angle approach: reads q_prev to compute the actual angle,
    /// then sets target = θ_prev + clamp(θ_goal - θ_prev, -step, step).
    /// [drive-substep] substep_ratio in (0,1] scales the (rate-limited) step
    /// toward the goal — uipc-style animation substepping: called each Newton
    /// iteration with ratio=(k+1)/S the driving target ramps across the solve
    /// instead of dumping the whole frame's driving energy into iteration 0.
    void update_revolute_driving_targets(ABDSimData& sim_data,
                                         const std::vector<JointAngleControlInfo>& controls,
                                         double substep_ratio = 1.0);

    void _cal_abd_joint_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_revolute_driving_gradient_and_hessian(ABDSimData& sim_data);

    // Prismatic joint constraint: init, energy, gradient/hessian
    void init_prismatic_constraints(ABDSimData& sim_data,
                                    const std::vector<PrismaticJointHostInfo>& host_prismatic);

    void init_prismatic_driving(ABDSimData& sim_data,
                                const std::vector<PrismaticDrivingControlInfo>& controls,
                                const std::vector<PrismaticJointHostInfo>& host_prismatic);

    void update_prismatic_driving_targets(ABDSimData& sim_data,
                                          const std::vector<PrismaticDrivingControlInfo>& controls,
                                          double substep_ratio = 1.0);

    void _cal_abd_prismatic_gradient_and_hessian(ABDSimData& sim_data);
    void _cal_abd_prismatic_driving_gradient_and_hessian(ABDSimData& sim_data);

    void _cal_abd_stitch_gradient_and_hessian(ABDSimData& sim_data);
};
}  // namespace gipc
