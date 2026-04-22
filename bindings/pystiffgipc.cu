#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "sim_engine.h"

namespace py = pybind11;
using namespace gipc;

PYBIND11_MODULE(pystiffgipc, m)
{
    m.doc() = "StiffGIPC Python bindings - IPC-based physics simulation engine";

    // ---- SimEngineConfig ----
    py::class_<SimEngineConfig>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dt",                  &SimEngineConfig::dt)
        .def_readwrite("density",             &SimEngineConfig::density)
        .def_readwrite("young_modulus",        &SimEngineConfig::young_modulus)
        .def_readwrite("poisson_rate",         &SimEngineConfig::poisson_rate)
        .def_readwrite("friction_rate",        &SimEngineConfig::friction_rate)
        .def_readwrite("gd_friction_rate",     &SimEngineConfig::gd_friction_rate)
        .def_readwrite("cloth_thickness",      &SimEngineConfig::cloth_thickness)
        .def_readwrite("cloth_young_modulus",  &SimEngineConfig::cloth_young_modulus)
        .def_readwrite("bend_young_modulus",   &SimEngineConfig::bend_young_modulus)
        .def_readwrite("cloth_density",        &SimEngineConfig::cloth_density)
        .def_readwrite("strain_rate",          &SimEngineConfig::strain_rate)
        .def_readwrite("soft_motion_rate",     &SimEngineConfig::soft_motion_rate)
        .def_readwrite("newton_tol",           &SimEngineConfig::newton_tol)
        .def_readwrite("pcg_tol",              &SimEngineConfig::pcg_tol)
        .def_readwrite("relative_dhat",        &SimEngineConfig::relative_dhat)
        .def_readwrite("joint_strength_ratio",             &SimEngineConfig::joint_strength_ratio)
        .def_readwrite("revolute_driving_strength_ratio",  &SimEngineConfig::revolute_driving_strength_ratio)
        .def_readwrite("prismatic_strength_ratio",         &SimEngineConfig::prismatic_strength_ratio)
        .def_readwrite("prismatic_driving_strength_ratio", &SimEngineConfig::prismatic_driving_strength_ratio)
        .def_readwrite("collision_detection_buff_scale",   &SimEngineConfig::collision_detection_buff_scale)
        .def_readwrite("linear_system_buff_scale",         &SimEngineConfig::linear_system_buff_scale)
        .def_readwrite("preconditioner_type",              &SimEngineConfig::preconditioner_type)
        .def_readwrite("cuda_device",                      &SimEngineConfig::cuda_device)
        .def_readwrite("semi_implicit_enabled",            &SimEngineConfig::semi_implicit_enabled)
        .def_readwrite("semi_implicit_beta_tol",           &SimEngineConfig::semi_implicit_beta_tol)
        .def_readwrite("semi_implicit_min_iter",           &SimEngineConfig::semi_implicit_min_iter)
        .def_readwrite("newton_iter_cap",                  &SimEngineConfig::newton_iter_cap)
        .def_readwrite("skip_all_collision",               &SimEngineConfig::skip_all_collision)
        .def_readwrite("velocity_damping",                 &SimEngineConfig::velocity_damping)
        .def_readwrite("gravity",                          &SimEngineConfig::gravity)
        .def_readwrite("ground_normal",                    &SimEngineConfig::ground_normal)
        .def_readwrite("ground_offset",                    &SimEngineConfig::ground_offset)
        .def_readwrite("assets_dir",                       &SimEngineConfig::assets_dir)
        .def("__repr__", [](const SimEngineConfig& c) {
            return "<pystiffgipc.Config dt=" + std::to_string(c.dt)
                   + " density=" + std::to_string(c.density) + ">";
        });

    // ---- JointInfo ----
    py::class_<JointInfo>(m, "JointInfo")
        .def_readonly("name",           &JointInfo::name)
        .def_readonly("lower_limit",    &JointInfo::lower_limit)
        .def_readonly("upper_limit",    &JointInfo::upper_limit)
        .def_readonly("target",         &JointInfo::target)
        .def_readonly("strength_ratio", &JointInfo::strength_ratio)
        .def_readonly("is_prismatic",   &JointInfo::is_prismatic)
        .def("__repr__", [](const JointInfo& j) {
            return "<JointInfo '" + j.name + "' ["
                   + std::to_string(j.lower_limit) + ", "
                   + std::to_string(j.upper_limit) + "]>";
        });

    // ---- BodyLoadRecord ----
    py::class_<BodyLoadRecord>(m, "BodyLoadRecord")
        .def_readonly("body_type",     &BodyLoadRecord::body_type)
        .def_readonly("body_offset",   &BodyLoadRecord::body_offset)
        .def_readonly("vertex_offset", &BodyLoadRecord::vertex_offset)
        .def_readonly("vertex_count",  &BodyLoadRecord::vertex_count)
        .def_readonly("asset_id",      &BodyLoadRecord::asset_id)
        .def_readonly("instance_id",   &BodyLoadRecord::instance_id)
        .def_readonly("label",         &BodyLoadRecord::label)
        .def("__repr__", [](const BodyLoadRecord& r) {
            return "<BodyLoadRecord type=" + std::to_string(r.body_type)
                   + " body_off=" + std::to_string(r.body_offset)
                   + " vert_off=" + std::to_string(r.vertex_offset)
                   + " vert_cnt=" + std::to_string(r.vertex_count)
                   + " asset=" + std::to_string(r.asset_id)
                   + " inst=" + std::to_string(r.instance_id) + ">";
        });

    // ---- MeshAsset ----
    py::class_<MeshAsset>(m, "MeshAsset")
        .def_readonly("asset_id",       &MeshAsset::asset_id)
        .def_readonly("num_verts",      &MeshAsset::num_verts)
        .def_readonly("num_faces",      &MeshAsset::num_faces)
        .def_readonly("verts_per_face", &MeshAsset::verts_per_face)
        .def_readonly("dimensions",     &MeshAsset::dimensions)
        .def_readonly("body_type",      &MeshAsset::body_type)
        .def_readonly("young_modulus",  &MeshAsset::young_modulus)
        .def_readonly("boundary_type",  &MeshAsset::boundary_type)
        .def("get_rest_vertices", [](const MeshAsset& a) {
            return py::array_t<double>({a.num_verts, 3},
                                       a.rest_vertices.data());
        })
        .def("get_faces", [](const MeshAsset& a) {
            return py::array_t<int>({a.num_faces, a.verts_per_face},
                                    a.faces.data());
        })
        .def("__repr__", [](const MeshAsset& a) {
            return "<MeshAsset id=" + std::to_string(a.asset_id)
                   + " verts=" + std::to_string(a.num_verts)
                   + " faces=" + std::to_string(a.num_faces) + ">";
        });

    // ---- InstancedLoadResult ----
    py::class_<InstancedLoadResult>(m, "InstancedLoadResult")
        .def_readonly("body_offsets",   &InstancedLoadResult::body_offsets)
        .def_readonly("vertex_offsets", &InstancedLoadResult::vertex_offsets)
        .def_readonly("vertex_counts",  &InstancedLoadResult::vertex_counts)
        .def_readonly("asset_id",       &InstancedLoadResult::asset_id)
        .def("__repr__", [](const InstancedLoadResult& r) {
            return "<InstancedLoadResult asset=" + std::to_string(r.asset_id)
                   + " N=" + std::to_string(r.body_offsets.size()) + ">";
        });

    // ---- SimEngine ----
    py::class_<SimEngine>(m, "SimEngine")
        .def(py::init<>())

        .def("set_config", &SimEngine::set_config, py::arg("config"))
        .def("config", &SimEngine::config, py::return_value_policy::reference_internal)

        .def("init_cuda", &SimEngine::init_cuda)

        .def("load_urdf", &SimEngine::load_urdf,
             py::arg("urdf_path"),
             py::arg("global_transform"),
             py::arg("root_fixed")              = true,
             py::arg("revolute_as_motor")        = false,
             py::arg("default_young")            = 1e7,
             py::arg("initial_joint_angles")     = std::map<std::string, double>{})

        .def("add_ground", &SimEngine::add_ground, py::arg("height") = 0.0)

        .def("load_mesh", &SimEngine::load_mesh,
             py::arg("mesh_path"),
             py::arg("dimensions"),
             py::arg("body_type"),
             py::arg("transform"),
             py::arg("young_modulus"),
             py::arg("boundary_type") = 0)

        .def("load_mesh_from_data", [](SimEngine& e,
                py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                py::array_t<int, py::array::c_style | py::array::forcecast> faces,
                int verts_per_face, int dimensions, int body_type,
                const Eigen::Matrix4d& transform, double young_modulus, int boundary_type) {
            auto vbuf = vertices.request();
            auto fbuf = faces.request();
            int num_verts = static_cast<int>(vbuf.shape[0]);
            int num_faces = static_cast<int>(fbuf.shape[0]);
            e.load_mesh_from_data(
                static_cast<const double*>(vbuf.ptr), num_verts,
                static_cast<const int*>(fbuf.ptr), num_faces, verts_per_face,
                dimensions, body_type, transform, young_modulus, boundary_type);
        },  py::arg("vertices"), py::arg("faces"),
            py::arg("verts_per_face"), py::arg("dimensions"),
            py::arg("body_type"), py::arg("transform"),
            py::arg("young_modulus"), py::arg("boundary_type") = 0)

        .def("load_mesh_instanced", [](SimEngine& e,
                py::array_t<double, py::array::c_style | py::array::forcecast> vertices,
                py::array_t<int, py::array::c_style | py::array::forcecast> faces,
                int verts_per_face, int dimensions, int body_type,
                py::list py_transforms,
                double young_modulus, int boundary_type) {
            auto vbuf = vertices.request();
            auto fbuf = faces.request();
            int num_verts = static_cast<int>(vbuf.shape[0]);
            int num_faces = static_cast<int>(fbuf.shape[0]);
            std::vector<Eigen::Matrix4d> transforms;
            transforms.reserve(py_transforms.size());
            for(auto& item : py_transforms) {
                auto arr = item.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                auto tbuf = arr.request();
                if(tbuf.ndim != 2 || tbuf.shape[0] != 4 || tbuf.shape[1] != 4)
                    throw std::runtime_error("Each transform must be (4,4) float64");
                Eigen::Matrix4d mat;
                std::memcpy(mat.data(), tbuf.ptr, 16 * sizeof(double));
                transforms.push_back(mat.transpose());
            }
            return e.load_mesh_instanced(
                static_cast<const double*>(vbuf.ptr), num_verts,
                static_cast<const int*>(fbuf.ptr), num_faces,
                verts_per_face, dimensions, body_type,
                transforms, young_modulus, boundary_type);
        },  py::arg("vertices"), py::arg("faces"),
            py::arg("verts_per_face"), py::arg("dimensions"),
            py::arg("body_type"), py::arg("transforms"),
            py::arg("young_modulus"), py::arg("boundary_type") = 0)

        .def("get_mesh_asset_count", &SimEngine::get_mesh_asset_count)
        .def("get_mesh_asset", &SimEngine::get_mesh_asset, py::arg("asset_id"),
             py::return_value_policy::reference_internal)

        .def("add_collision_exclusion", &SimEngine::add_collision_exclusion,
             py::arg("body_a"), py::arg("body_b"))
        .def("add_ground_collision_skip", &SimEngine::add_ground_collision_skip,
             py::arg("body_id"))

        .def("add_fixed_joint", [](SimEngine& self, int parent, int child,
                                   py::array_t<double> anchor,
                                   py::array_t<double> normal,
                                   py::array_t<double> bitangent) {
            auto a = anchor.unchecked<1>();
            auto n = normal.unchecked<1>();
            auto b = bitangent.unchecked<1>();
            return self.add_fixed_joint(parent, child,
                Eigen::Vector3d(a(0), a(1), a(2)),
                Eigen::Vector3d(n(0), n(1), n(2)),
                Eigen::Vector3d(b(0), b(1), b(2)));
        }, py::arg("parent_body"), py::arg("child_body"),
           py::arg("world_anchor"), py::arg("world_normal"), py::arg("world_bitangent"))

        .def("add_revolute_joint", [](SimEngine& self, int parent, int child,
                                      py::array_t<double> axis,
                                      py::array_t<double> pos,
                                      double lower, double upper,
                                      double init_angle,
                                      const std::string& name) {
            auto ax = axis.unchecked<1>();
            auto p  = pos.unchecked<1>();
            return self.add_revolute_joint(parent, child,
                Eigen::Vector3d(ax(0), ax(1), ax(2)),
                Eigen::Vector3d(p(0), p(1), p(2)),
                lower, upper, init_angle, name);
        }, py::arg("parent_body"), py::arg("child_body"),
           py::arg("world_axis"), py::arg("joint_pos"),
           py::arg("lower_limit"), py::arg("upper_limit"),
           py::arg("initial_angle") = 0.0, py::arg("name") = "")

        .def("add_prismatic_joint", [](SimEngine& self, int parent, int child,
                                       py::array_t<double> center,
                                       py::array_t<double> axis,
                                       double lower, double upper,
                                       const std::string& name) {
            auto c  = center.unchecked<1>();
            auto ax = axis.unchecked<1>();
            return self.add_prismatic_joint(parent, child,
                Eigen::Vector3d(c(0), c(1), c(2)),
                Eigen::Vector3d(ax(0), ax(1), ax(2)),
                lower, upper, name);
        }, py::arg("parent_body"), py::arg("child_body"),
           py::arg("world_center"), py::arg("world_axis"),
           py::arg("lower_limit"), py::arg("upper_limit"),
           py::arg("name") = "")

        .def("set_vertex_boundary", &SimEngine::set_vertex_boundary,
             py::arg("vertex_index"), py::arg("boundary_type"))

        .def("get_abd_body_count",    &SimEngine::get_abd_body_count)
        .def("get_fem_body_count",    &SimEngine::get_fem_body_count)
        .def("get_vertex_count_host", &SimEngine::get_vertex_count_host)

        .def("get_vertex_position_host", [](const SimEngine& e, int idx) {
            double xyz[3];
            e.get_vertex_position_host(idx, xyz);
            return py::make_tuple(xyz[0], xyz[1], xyz[2]);
        }, py::arg("idx"))

        .def("finalize", &SimEngine::finalize)

        .def("step", &SimEngine::step,
             py::call_guard<py::gil_scoped_release>())

        .def("get_assets_dir", &SimEngine::get_assets_dir)

        // Vertex positions as numpy array (N, 3) float64
        .def("get_vertices", [](const SimEngine& e) {
            int n = e.get_vertex_count();
            auto arr = py::array_t<double>({n, 3});
            e.get_vertex_positions(arr.mutable_data(), n);
            return arr;
        })

        // Vertex velocities as numpy array (N, 3) float64
        .def("get_vertex_velocities", [](const SimEngine& e) {
            int n = e.get_vertex_count();
            auto arr = py::array_t<double>({n, 3});
            if(n > 0) e.get_vertex_velocities(arr.mutable_data(), n);
            return arr;
        })

        // Set vertex positions on GPU from (N, 3) float64
        .def("set_vertex_positions_gpu", [](SimEngine& e,
                py::array_t<double, py::array::c_style | py::array::forcecast> positions) {
            auto buf = positions.request();
            e.set_vertex_positions_gpu(static_cast<const double*>(buf.ptr),
                                       static_cast<int>(buf.shape[0]));
        }, py::arg("positions"))

        // Set vertex velocities on GPU from (N, 3) float64
        .def("set_vertex_velocities_gpu", [](SimEngine& e,
                py::array_t<double, py::array::c_style | py::array::forcecast> velocities) {
            auto buf = velocities.request();
            e.set_vertex_velocities_gpu(static_cast<const double*>(buf.ptr),
                                         static_cast<int>(buf.shape[0]));
        }, py::arg("velocities"))

        // ABD body transforms: get (count, 4, 4) float64
        .def("get_abd_body_transforms", [](const SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets) {
            auto buf = offsets.request();
            int count = static_cast<int>(buf.shape[0]);
            auto arr = py::array_t<double>({count, 4, 4});
            e.get_abd_body_transforms(static_cast<const int*>(buf.ptr),
                                      arr.mutable_data(), count);
            return arr;
        }, py::arg("body_offsets"))

        // ABD body transforms: set from (count, 4, 4) float64
        .def("set_abd_body_transforms", [](SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets,
                py::array_t<double, py::array::c_style | py::array::forcecast> transforms) {
            auto obuf = offsets.request();
            auto tbuf = transforms.request();
            int count = static_cast<int>(obuf.shape[0]);
            e.set_abd_body_transforms(static_cast<const int*>(obuf.ptr),
                                      static_cast<const double*>(tbuf.ptr), count);
        }, py::arg("body_offsets"), py::arg("transforms"))

        // Teleport ABD bodies: resets q, q_prev, q_tilde, q_temp and zeros q_v, dq
        .def("teleport_abd_bodies", [](SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets,
                py::array_t<double, py::array::c_style | py::array::forcecast> transforms) {
            auto obuf = offsets.request();
            auto tbuf = transforms.request();
            int count = static_cast<int>(obuf.shape[0]);
            e.teleport_abd_bodies(static_cast<const int*>(obuf.ptr),
                                  static_cast<const double*>(tbuf.ptr), count);
        }, py::arg("body_offsets"), py::arg("transforms"))

        // ABD body velocities: get (count, 4, 4) float64
        .def("get_abd_body_velocities", [](const SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets) {
            auto buf = offsets.request();
            int count = static_cast<int>(buf.shape[0]);
            auto arr = py::array_t<double>({count, 4, 4});
            e.get_abd_body_velocities(static_cast<const int*>(buf.ptr),
                                      arr.mutable_data(), count);
            return arr;
        }, py::arg("body_offsets"))

        // ABD body velocities: set from (count, 4, 4) float64
        .def("set_abd_body_velocities", [](SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets,
                py::array_t<double, py::array::c_style | py::array::forcecast> velocities) {
            auto obuf = offsets.request();
            auto vbuf = velocities.request();
            int count = static_cast<int>(obuf.shape[0]);
            e.set_abd_body_velocities(static_cast<const int*>(obuf.ptr),
                                      static_cast<const double*>(vbuf.ptr), count);
        }, py::arg("body_offsets"), py::arg("velocities"))

        // FEM body vertex range
        .def("get_fem_body_vertex_range", [](const SimEngine& e, int fem_body_idx) {
            int start = 0, count = 0;
            e.get_fem_body_vertex_range(fem_body_idx, &start, &count);
            return py::make_tuple(start, count);
        }, py::arg("fem_body_idx"))

        // Surface triangle indices as numpy array (F, 3) uint32
        .def("get_surface_faces", [](const SimEngine& e) {
            int n = e.get_surface_face_count();
            auto arr = py::array_t<uint32_t>({n, 3});
            e.get_surface_faces(arr.mutable_data(), n);
            return arr;
        })

        // Surface vertex indices (into full vertex array)
        .def("get_surface_vertex_indices", [](const SimEngine& e) {
            int n = e.get_surface_vertex_count();
            auto arr = py::array_t<uint32_t>({n});
            e.get_surface_vertex_indices(arr.mutable_data(), n);
            return arr;
        })

        .def("get_vertex_count",          &SimEngine::get_vertex_count)
        .def("get_surface_face_count",    &SimEngine::get_surface_face_count)
        .def("get_surface_vertex_count",  &SimEngine::get_surface_vertex_count)

        // Load record tracking
        .def("get_load_record_count", &SimEngine::get_load_record_count)
        .def("get_load_record", &SimEngine::get_load_record, py::arg("idx"),
             py::return_value_policy::reference_internal)
        .def("get_all_load_records", [](const SimEngine& e) {
            std::vector<BodyLoadRecord> recs;
            for(int i = 0; i < e.get_load_record_count(); i++)
                recs.push_back(e.get_load_record(i));
            return recs;
        })

        // Joint control
        .def("get_num_revolute_joints",   &SimEngine::get_num_revolute_joints)
        .def("get_num_prismatic_joints",  &SimEngine::get_num_prismatic_joints)

        .def("get_revolute_joint_info",   &SimEngine::get_revolute_joint_info,  py::arg("idx"))
        .def("get_prismatic_joint_info",  &SimEngine::get_prismatic_joint_info, py::arg("idx"))

        .def("set_revolute_target",       &SimEngine::set_revolute_target,
             py::arg("idx"), py::arg("angle_rad"))
        .def("set_revolute_initial_offset", &SimEngine::set_revolute_initial_offset,
             py::arg("idx"), py::arg("offset_rad"))
        .def("set_prismatic_target",      &SimEngine::set_prismatic_target,
             py::arg("idx"), py::arg("distance_m"))

        .def("get_revolute_target",       &SimEngine::get_revolute_target,  py::arg("idx"))
        .def("get_prismatic_target",      &SimEngine::get_prismatic_target, py::arg("idx"))

        .def("get_revolute_current_angles", [](const SimEngine& e) {
            int n = e.get_num_revolute_joints();
            auto arr = py::array_t<double>({n});
            if(n > 0)
                e.get_revolute_current_angles(arr.mutable_data(), n);
            return arr;
        })

        // Convenience: list all joints
        .def("get_all_joint_infos", [](const SimEngine& e) {
            std::vector<JointInfo> joints;
            for(int i = 0; i < e.get_num_revolute_joints(); i++)
                joints.push_back(e.get_revolute_joint_info(i));
            for(int i = 0; i < e.get_num_prismatic_joints(); i++)
                joints.push_back(e.get_prismatic_joint_info(i));
            return joints;
        });
}
