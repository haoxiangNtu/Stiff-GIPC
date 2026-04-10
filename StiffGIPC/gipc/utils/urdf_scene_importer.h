#pragma once
#include <load_mesh.h>
#include <gipc/body_type.h>
#include <body_boundary_type.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <functional>

namespace gipc
{

/// Information about a parsed URDF joint
struct UrdfJointInfo
{
    std::string name;
    std::string parent_link_name;
    std::string child_link_name;

    enum class Type
    {
        Fixed,
        Revolute,
        Continuous,
        Prismatic,
        Unknown
    } type = Type::Unknown;

    Eigen::Matrix4d local_trans = Eigen::Matrix4d::Identity();  // parent_to_joint_origin
    Eigen::Vector3d axis        = Eigen::Vector3d::UnitZ();     // joint axis (for revolute)
    double          lower_limit = 0.0;
    double          upper_limit = 0.0;

    // The global-frame axis computed during transform propagation
    Eigen::Vector3d global_axis = Eigen::Vector3d::UnitZ();
};

/// Information about a parsed URDF link 
struct UrdfLinkInfo
{
    std::string     name;
    std::string     collision_mesh_filename;  // resolved path from URDF
    Eigen::Matrix4d collision_origin = Eigen::Matrix4d::Identity();
    Eigen::Vector3d collision_scale  = Eigen::Vector3d::Ones();
    bool            has_collision    = false;

    // Computed global transform (from root to this link)
    Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();

    // The body id assigned after loading into tetrahedra_obj
    int body_id = -1;
};

/// Configuration for how to map a URDF link to a tet mesh
struct UrdfLinkMeshOverride
{
    std::string msh_path;       // path to .msh tetrahedral mesh
    double      young_modulus = 1e7;
};

/// URDF Scene Importer for GIPC
///
/// Parses a URDF file, traverses the joint tree to compute global transforms,
/// and loads each link's collision mesh as an ABD body into the GIPC simulation.
///
/// Mesh loading supports two paths:
///   1) Direct .obj/.stl surface meshes — if the URDF collision geometry points
///      to an .obj or .stl file, the importer loads it directly as an ABD body
///      using native surface integrals.
///   2) Mesh overrides (.msh) — users can provide a mesh_override_map that maps
///      link names to pre-tetrahedralized .msh files. Overrides take priority.
///
/// Links without an override and without a loadable .obj collision mesh are skipped.
///
/// Usage:
///   UrdfSceneImporter importer;
///   importer.set_urdf_path("path/to/robot.urdf");
///   importer.set_global_transform(transform);  // optional
///   // Option A: auto-load .obj from URDF collision geometry (no overrides needed)
///   importer.import_scene(tetMesh, preconditionerType);
///   // Option B: override specific links with .msh files
///   importer.set_mesh_override("link1", {"path/to/link1.msh", 1e7});
///   importer.import_scene(tetMesh, preconditionerType);
///
class UrdfSceneImporter
{
  public:
    UrdfSceneImporter() = default;

    /// Set the URDF file path
    void set_urdf_path(std::string_view urdf_path);

    /// Set a global transform applied to the entire robot
    void set_global_transform(const Eigen::Matrix4d& transform);

    /// Set the default Young's modulus for bodies without specific override
    void set_default_young_modulus(double E);

    /// Set the default body boundary type
    void set_default_boundary_type(BodyBoundaryType type);

    /// Override the mesh for a specific link (link_name -> .msh tet mesh path)
    void set_mesh_override(const std::string&        link_name,
                           const UrdfLinkMeshOverride& override_info);

    /// Set initial joint angles (radians) for FK computation during import.
    /// Links will be loaded at the FK pose instead of the zero pose.
    void set_initial_joint_angles(const std::map<std::string, double>& angles);

    /// Set whether root link should be fixed
    void set_root_fixed(bool fixed);

    /// Set whether revolute joint child links should be Motor type
    void set_revolute_as_motor(bool motor);

    /// Set the default motor speed (rad/s) for revolute joints
    void set_default_motor_speed(double speed);

    /// Set the default motor strength for revolute joints
    void set_default_motor_strength(double strength);

    /// Parse the URDF and import all links as ABD bodies
    /// Returns true on success
    bool import_scene(tetrahedra_obj& tetras, int preconditionerType = 0);

    /// After import, get the parsed link infos
    const std::map<std::string, UrdfLinkInfo>& link_infos() const { return m_link_infos; }

    /// After import, get the parsed joint infos
    const std::map<std::string, UrdfJointInfo>& joint_infos() const
    {
        return m_joint_infos;
    }

    /// Get the root link name
    const std::string& root_link_name() const { return m_root_link_name; }

  private:
    std::string     m_urdf_path;
    Eigen::Matrix4d m_global_transform    = Eigen::Matrix4d::Identity();
    double          m_default_young_modulus = 1e7;
    BodyBoundaryType m_default_boundary_type = BodyBoundaryType::Free;
    bool            m_root_fixed          = true;
    bool            m_revolute_as_motor        = false;
    double          m_default_motor_speed      = 0.0;   // 0 = use ABDSystemParms default
    double          m_default_motor_strength   = 0.0;   // 0 = use ABDSystemParms default

    std::string m_root_link_name;

    std::map<std::string, double>               m_initial_joint_angles;
    std::map<std::string, UrdfLinkMeshOverride> m_mesh_overrides;
    std::map<std::string, UrdfLinkInfo>         m_link_infos;
    std::map<std::string, UrdfJointInfo>        m_joint_infos;

    // URDF joint tree: parent_link_name -> list of joint names
    std::map<std::string, std::vector<std::string>> m_link_children_joints;

    bool parse_urdf();
    void propagate_transforms(const std::string&  link_name,
                              const Eigen::Matrix4d& parent_global);
    
    // Convert URDF rpy + xyz to 4x4 transform
    static Eigen::Matrix4d pose_to_matrix(double x,
                                           double y,
                                           double z,
                                           double roll,
                                           double pitch,
                                           double yaw);
};

}  // namespace gipc
