#include <gipc/utils/urdf_scene_importer.h>
#include <gipc/utils/simple_scene_importer.h>
#include <urdf_parser/urdf_parser.h>
#include <filesystem>
#include <iostream>
#include <Eigen/Geometry>

namespace gipc
{
namespace fs = std::filesystem;

// ============================================================================
// Public API
// ============================================================================

void UrdfSceneImporter::set_urdf_path(std::string_view urdf_path)
{
    m_urdf_path = std::string{urdf_path};
}

void UrdfSceneImporter::set_global_transform(const Eigen::Matrix4d& transform)
{
    m_global_transform = transform;
}

void UrdfSceneImporter::set_default_young_modulus(double E)
{
    m_default_young_modulus = E;
}

void UrdfSceneImporter::set_default_boundary_type(BodyBoundaryType type)
{
    m_default_boundary_type = type;
}

void UrdfSceneImporter::set_mesh_override(const std::string&          link_name,
                                           const UrdfLinkMeshOverride& override_info)
{
    m_mesh_overrides[link_name] = override_info;
}

void UrdfSceneImporter::set_root_fixed(bool fixed)
{
    m_root_fixed = fixed;
}

void UrdfSceneImporter::set_revolute_as_motor(bool motor)
{
    m_revolute_as_motor = motor;
}

void UrdfSceneImporter::set_default_motor_speed(double speed)
{
    m_default_motor_speed = speed;
}

void UrdfSceneImporter::set_default_motor_strength(double strength)
{
    m_default_motor_strength = strength;
}

// ============================================================================
// Main import entry point
// ============================================================================

bool UrdfSceneImporter::import_scene(tetrahedra_obj& tetras, int preconditionerType)
{
    std::cout << "[UrdfSceneImporter] Parsing URDF: " << m_urdf_path << std::endl;

    if(!parse_urdf())
    {
        std::cerr << "[UrdfSceneImporter] Failed to parse URDF file." << std::endl;
        return false;
    }

    // Propagate transforms from root
    propagate_transforms(m_root_link_name, m_global_transform);

    // Print summary
    std::cout << "[UrdfSceneImporter] Root link: " << m_root_link_name << std::endl;
    std::cout << "[UrdfSceneImporter] Total links: " << m_link_infos.size() << std::endl;
    std::cout << "[UrdfSceneImporter] Total joints: " << m_joint_infos.size() << std::endl;

    // Load each link that has a mesh override as an ABD body
    // Ordering: process in BFS order from root to maintain a predictable body order
    std::vector<std::string> ordered_links;
    {
        std::vector<std::string> queue;
        queue.push_back(m_root_link_name);
        while(!queue.empty())
        {
            std::string current = queue.front();
            queue.erase(queue.begin());
            ordered_links.push_back(current);

            auto it = m_link_children_joints.find(current);
            if(it != m_link_children_joints.end())
            {
                for(auto& joint_name : it->second)
                {
                    auto& joint = m_joint_infos[joint_name];
                    queue.push_back(joint.child_link_name);
                }
            }
        }
    }

    int loaded_count = 0;
    SimpleSceneImporter simple_importer;

    for(auto& link_name : ordered_links)
    {
        auto& link_info = m_link_infos[link_name];

        // Determine the mesh source:
        // 1) If a mesh_override exists, use the .msh file
        // 2) Else if the URDF collision mesh is an .obj, use load_surfaceMesh_ABD
        // 3) Otherwise, skip

        auto    override_it = m_mesh_overrides.find(link_name);
        bool    has_override = (override_it != m_mesh_overrides.end());
        bool    has_obj_collision = false;
        std::string obj_collision_path;

        if(!has_override && link_info.has_collision && !link_info.collision_mesh_filename.empty())
        {
            fs::path coll_path{link_info.collision_mesh_filename};
            std::string ext = coll_path.extension().string();
            // lowercase the extension
            for(auto& c : ext) c = static_cast<char>(std::tolower(c));
            if(ext == ".obj" && fs::exists(coll_path))
            {
                has_obj_collision  = true;
                obj_collision_path = link_info.collision_mesh_filename;
            }
        }

        if(!has_override && !has_obj_collision)
        {
            std::cout << "[UrdfSceneImporter] Skipping link '" << link_name
                      << "' (no mesh override and no loadable .obj collision mesh)" << std::endl;
            continue;
        }

        if(has_override && !fs::exists(override_it->second.msh_path))
        {
            std::cerr << "[UrdfSceneImporter] WARNING: .msh file not found: "
                      << override_it->second.msh_path << ", skipping link '" << link_name
                      << "'" << std::endl;
            continue;
        }

        // Determine boundary type based on joint type
        BodyBoundaryType boundary_type = m_default_boundary_type;

        // Find the joint that connects to this link (this link is the child)
        const UrdfJointInfo* parent_joint = nullptr;
        for(auto& [jname, jinfo] : m_joint_infos)
        {
            if(jinfo.child_link_name == link_name)
            {
                parent_joint = &jinfo;
                break;
            }
        }

        if(link_name == m_root_link_name && m_root_fixed)
        {
            boundary_type = BodyBoundaryType::Fixed;
        }
        else if(parent_joint)
        {
            if(parent_joint->type == UrdfJointInfo::Type::Fixed)
            {
                // Fixed joint: child link is rigidly attached, freeze it
                boundary_type = BodyBoundaryType::Fixed;
            }
            else if(m_revolute_as_motor
                    && (parent_joint->type == UrdfJointInfo::Type::Revolute
                        || parent_joint->type == UrdfJointInfo::Type::Continuous))
            {
                // Revolute/Continuous joint: child link is motor-driven
                boundary_type = BodyBoundaryType::Motor;
            }
        }

        // Compute the full transform: link_global_transform * collision_origin * scale
        Eigen::Matrix4d full_transform = link_info.global_transform;

        // Apply collision origin if available
        if(link_info.has_collision)
        {
            full_transform = full_transform * link_info.collision_origin;

            // Apply collision mesh scale
            Eigen::Matrix4d scale_mat = Eigen::Matrix4d::Identity();
            scale_mat(0, 0) = link_info.collision_scale.x();
            scale_mat(1, 1) = link_info.collision_scale.y();
            scale_mat(2, 2) = link_info.collision_scale.z();
            full_transform  = full_transform * scale_mat;
        }

        double young_modulus = has_override ? override_it->second.young_modulus
                                            : m_default_young_modulus;

        if(has_override)
        {
            // ---- Path A: Use the .msh tetrahedral mesh override ----
            std::cout << "[UrdfSceneImporter] Loading link '" << link_name << "' from .msh: "
                      << override_it->second.msh_path
                      << " (boundary=" << static_cast<int>(boundary_type)
                      << ", E=" << young_modulus << ")" << std::endl;

            simple_importer.load_geometry(tetras,
                                          3,  // 3D tet mesh
                                          gipc::BodyType::ABD,
                                          full_transform,
                                          young_modulus,
                                          override_it->second.msh_path,
                                          preconditionerType,
                                          boundary_type);
        }
        else
        {
            // ---- Path B: Use the .obj surface mesh directly (fan-tet) ----
            std::cout << "[UrdfSceneImporter] Loading link '" << link_name << "' from .obj: "
                      << obj_collision_path
                      << " (boundary=" << static_cast<int>(boundary_type)
                      << ", E=" << young_modulus << ")" << std::endl;

            bool ok = tetras.load_surfaceMesh_ABD(obj_collision_path,
                                                   full_transform,
                                                   young_modulus,
                                                   boundary_type);
            if(!ok)
            {
                std::cerr << "[UrdfSceneImporter] Failed to load .obj for link '"
                          << link_name << "', skipping." << std::endl;
                continue;
            }
        }

        link_info.body_id = tetras.abd_fem_count_info.abd_body_num - 1;

        // Set per-body motor info if this is a Motor body
        if(boundary_type == BodyBoundaryType::Motor && parent_joint != nullptr)
        {
            int body_id = link_info.body_id;
            if(body_id >= 0 && body_id < static_cast<int>(tetras.body_motor_infos.size()))
            {
                auto& mi = tetras.body_motor_infos[body_id];
                // Transform the joint axis from local to global frame
                Eigen::Vector3d global_axis = parent_joint->global_axis.normalized();
                mi.axis_x   = global_axis.x();
                mi.axis_y   = global_axis.y();
                mi.axis_z   = global_axis.z();
                mi.speed    = m_default_motor_speed;
                mi.strength = m_default_motor_strength;

                std::cout << "[UrdfSceneImporter]   Motor body_id=" << body_id
                          << " axis=(" << mi.axis_x << "," << mi.axis_y << ","
                          << mi.axis_z << ") speed=" << mi.speed << std::endl;
            }
        }

        loaded_count++;
    }

    // ---- Disable ALL self-collision among robot links ----
    // Like rbs-uipc: robot_elem vs robot_elem = no collision.
    // Collect all body IDs that belong to this robot, then exclude all pairs.
    std::vector<int> robot_body_ids;
    for(auto& [link_name, link_info] : m_link_infos)
    {
        if(link_info.body_id >= 0)
            robot_body_ids.push_back(link_info.body_id);
    }
    for(size_t i = 0; i < robot_body_ids.size(); i++)
    {
        for(size_t j = i + 1; j < robot_body_ids.size(); j++)
        {
            tetras.collision_exclusion_pairs.emplace_back(robot_body_ids[i], robot_body_ids[j]);
        }
    }
    std::cout << "[UrdfSceneImporter] Robot self-collision disabled for "
              << robot_body_ids.size() << " bodies ("
              << tetras.collision_exclusion_pairs.size() << " exclusion pairs)" << std::endl;

    // ---- Generate joint constraints ----
    // For each joint, create constraint points at the joint location in world space.
    // The ABD solver will convert these to material coordinates after init.
    for(auto& [jname, jinfo] : m_joint_infos)
    {
        auto parent_it = m_link_infos.find(jinfo.parent_link_name);
        auto child_it  = m_link_infos.find(jinfo.child_link_name);
        if(parent_it == m_link_infos.end() || child_it == m_link_infos.end())
            continue;
        if(parent_it->second.body_id < 0 || child_it->second.body_id < 0)
            continue;

        // Joint world position = parent_global * joint_local_trans
        Eigen::Matrix4d joint_world = parent_it->second.global_transform * jinfo.local_trans;
        Eigen::Vector3d joint_pos   = joint_world.block<3, 1>(0, 3);

        JointConstraintHostInfo jc;
        jc.parent_body_id = parent_it->second.body_id;
        jc.child_body_id  = child_it->second.body_id;

        if(jinfo.type == UrdfJointInfo::Type::Fixed)
        {
            // Fixed joint: 4 non-coplanar constraint points (tetrahedron around joint)
            jc.type       = JointConstraintHostInfo::Type::Fixed;
            jc.num_points = 4;
            double spread = 0.05;  // 5cm spread for constraint points
            jc.world_anchor[0] = joint_pos;
            jc.world_anchor[1] = joint_pos + Eigen::Vector3d(spread, 0, 0);
            jc.world_anchor[2] = joint_pos + Eigen::Vector3d(0, spread, 0);
            jc.world_anchor[3] = joint_pos + Eigen::Vector3d(0, 0, spread);
        }
        else if(jinfo.type == UrdfJointInfo::Type::Revolute
                || jinfo.type == UrdfJointInfo::Type::Continuous)
        {
            // Revolute joint: 2 constraint points on the rotation axis
            // These constrain the axis position but allow rotation around it
            jc.type       = JointConstraintHostInfo::Type::Revolute;
            jc.num_points = 2;
            double spread = 0.05;  // 5cm offset along axis from joint center
            Eigen::Vector3d world_axis = jinfo.global_axis.normalized();
            jc.world_anchor[0] = joint_pos + spread * world_axis;
            jc.world_anchor[1] = joint_pos - spread * world_axis;
        }
        else
        {
            // Unsupported joint type, skip
            continue;
        }

        tetras.joint_constraints.push_back(jc);

        std::cout << "[UrdfSceneImporter] Joint constraint '" << jname
                  << "' (" << (jc.type == JointConstraintHostInfo::Type::Fixed ? "Fixed" : "Revolute")
                  << "): body " << jc.parent_body_id << " <-> " << jc.child_body_id
                  << " at (" << joint_pos.x() << ", " << joint_pos.y() << ", "
                  << joint_pos.z() << ")" << std::endl;
    }

    std::cout << "[UrdfSceneImporter] Generated " << tetras.joint_constraints.size()
              << " joint constraints." << std::endl;

    std::cout << "[UrdfSceneImporter] Successfully loaded " << loaded_count
              << " ABD bodies from URDF (" << tetras.collision_exclusion_pairs.size()
              << " collision exclusion pairs)." << std::endl;

    return loaded_count > 0;
}

// ============================================================================
// URDF Parsing (using urdfdom library)
// ============================================================================

Eigen::Matrix4d UrdfSceneImporter::pose_to_matrix(double x,
                                                    double y,
                                                    double z,
                                                    double roll,
                                                    double pitch,
                                                    double yaw)
{
    Eigen::Transform<double, 3, Eigen::Affine> t = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
    t.translate(Eigen::Vector3d{x, y, z});
    // URDF convention: Yaw-Pitch-Roll (ZYX intrinsic = XYZ extrinsic)
    t.rotate(Eigen::AngleAxisd{yaw, Eigen::Vector3d::UnitZ()});
    t.rotate(Eigen::AngleAxisd{pitch, Eigen::Vector3d::UnitY()});
    t.rotate(Eigen::AngleAxisd{roll, Eigen::Vector3d::UnitX()});
    return t.matrix();
}

static Eigen::Matrix4d urdf_pose_to_matrix(const urdf::Pose& pose)
{
    Eigen::Transform<double, 3, Eigen::Affine> t =
        Eigen::Transform<double, 3, Eigen::Affine>::Identity();
    t.translate(Eigen::Vector3d{pose.position.x, pose.position.y, pose.position.z});
    Eigen::Quaterniond q{pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z};
    t.rotate(q);
    return t.matrix();
}

bool UrdfSceneImporter::parse_urdf()
{
    if(!fs::exists(m_urdf_path))
    {
        std::cerr << "[UrdfSceneImporter] URDF file does not exist: " << m_urdf_path
                  << std::endl;
        return false;
    }

    auto model = urdf::parseURDFFile(m_urdf_path);
    if(!model)
    {
        std::cerr << "[UrdfSceneImporter] Failed to parse URDF: " << m_urdf_path
                  << std::endl;
        return false;
    }

    fs::path urdf_folder = fs::path{m_urdf_path}.parent_path();

    std::cout << "[UrdfSceneImporter] Model name: " << model->name_ << std::endl;
    std::cout << "[UrdfSceneImporter] Links: " << model->links_.size()
              << ", Joints: " << model->joints_.size() << std::endl;

    // Store root link name
    if(model->root_link_)
    {
        m_root_link_name = model->root_link_->name;
    }
    else
    {
        std::cerr << "[UrdfSceneImporter] No root link found!" << std::endl;
        return false;
    }

    // Parse all links
    m_link_infos.clear();
    for(auto& [name, link] : model->links_)
    {
        UrdfLinkInfo info;
        info.name = name;

        if(link->collision && link->collision->geometry)
        {
            info.has_collision    = true;
            info.collision_origin = urdf_pose_to_matrix(link->collision->origin);

            auto geom_type = link->collision->geometry->type;
            if(geom_type == urdf::Geometry::MESH)
            {
                auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(link->collision->geometry);
                if(mesh && !mesh->filename.empty())
                {
                    // Resolve the mesh file path
                    std::string filename = mesh->filename;
                    // Handle "package://" prefix
                    auto pos = filename.find("://");
                    if(pos != std::string::npos)
                    {
                        filename = filename.substr(pos + 3);
                    }

                    // Try to resolve relative to URDF folder
                    fs::path mesh_path = urdf_folder / filename;
                    if(fs::exists(mesh_path))
                    {
                        info.collision_mesh_filename = fs::canonical(mesh_path).string();
                    }
                    else
                    {
                        // Store raw filename for reference
                        info.collision_mesh_filename = filename;
                    }

                    info.collision_scale = Eigen::Vector3d{
                        mesh->scale.x, mesh->scale.y, mesh->scale.z};
                }
            }
        }

        m_link_infos[name] = info;
    }

    // Parse all joints
    m_joint_infos.clear();
    m_link_children_joints.clear();
    for(auto& [name, joint] : model->joints_)
    {
        UrdfJointInfo info;
        info.name             = name;
        info.parent_link_name = joint->parent_link_name;
        info.child_link_name  = joint->child_link_name;
        info.local_trans      = urdf_pose_to_matrix(joint->parent_to_joint_origin_transform);

        switch(joint->type)
        {
            case urdf::Joint::FIXED:
                info.type = UrdfJointInfo::Type::Fixed;
                break;
            case urdf::Joint::REVOLUTE:
                info.type = UrdfJointInfo::Type::Revolute;
                info.axis = Eigen::Vector3d{joint->axis.x, joint->axis.y, joint->axis.z};
                if(joint->limits)
                {
                    info.lower_limit = joint->limits->lower;
                    info.upper_limit = joint->limits->upper;
                }
                break;
            case urdf::Joint::CONTINUOUS:
                info.type = UrdfJointInfo::Type::Continuous;
                info.axis = Eigen::Vector3d{joint->axis.x, joint->axis.y, joint->axis.z};
                break;
            case urdf::Joint::PRISMATIC:
                info.type = UrdfJointInfo::Type::Prismatic;
                info.axis = Eigen::Vector3d{joint->axis.x, joint->axis.y, joint->axis.z};
                if(joint->limits)
                {
                    info.lower_limit = joint->limits->lower;
                    info.upper_limit = joint->limits->upper;
                }
                break;
            default:
                info.type = UrdfJointInfo::Type::Unknown;
                std::cout << "[UrdfSceneImporter] WARNING: Unknown joint type for '"
                          << name << "'" << std::endl;
                break;
        }

        m_joint_infos[name] = info;

        // Build parent->children mapping
        m_link_children_joints[info.parent_link_name].push_back(name);

        std::cout << "[UrdfSceneImporter]   Joint '" << name << "': "
                  << info.parent_link_name << " -> " << info.child_link_name << " (type="
                  << static_cast<int>(info.type) << ")" << std::endl;
    }

    return true;
}

// ============================================================================
// Transform propagation through the joint tree
// ============================================================================

void UrdfSceneImporter::propagate_transforms(const std::string&     link_name,
                                              const Eigen::Matrix4d& parent_global)
{
    auto link_it = m_link_infos.find(link_name);
    if(link_it == m_link_infos.end())
        return;

    // This link's global transform is the parent's global transform
    // (the joint transform is between parent and child, not on the link itself)
    link_it->second.global_transform = parent_global;

    // Find child joints
    auto children_it = m_link_children_joints.find(link_name);
    if(children_it == m_link_children_joints.end())
        return;

    for(auto& joint_name : children_it->second)
    {
        auto joint_it = m_joint_infos.find(joint_name);
        if(joint_it == m_joint_infos.end())
            continue;

        // Child global = parent_global * joint_local_transform
        Eigen::Matrix4d child_global = parent_global * joint_it->second.local_trans;

        // Compute the global-frame rotation axis for revolute/continuous joints
        // The joint axis is defined in the joint frame, transform the direction
        if(joint_it->second.type == UrdfJointInfo::Type::Revolute
           || joint_it->second.type == UrdfJointInfo::Type::Continuous)
        {
            Eigen::Matrix3d R = child_global.block<3, 3>(0, 0);
            joint_it->second.global_axis = (R * joint_it->second.axis).normalized();
        }

        propagate_transforms(joint_it->second.child_link_name, child_global);
    }
}

}  // namespace gipc
