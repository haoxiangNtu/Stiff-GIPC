#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <sstream>
#include "sim_engine.h"
#include "GIPC.cuh"
#include "errors.h"

namespace py = pybind11;
using namespace gipc;

PYBIND11_MODULE(pystiffgipc, m)
{
    m.doc() = "StiffGIPC Python bindings - IPC-based physics simulation engine";
    m.def("fem_model", []() { return std::string(GIPC_FEM_MODEL_NAME); },
          "Return the compile-time tetrahedral constitutive model.");
    auto stiff_error =
        py::register_exception<gipc::StiffGIPCError>(m, "StiffGIPCError");
    py::register_exception<gipc::ConfigurationError>(
        m, "ConfigurationError", stiff_error.ptr());
    py::register_exception<gipc::GeometryError>(
        m, "GeometryError", stiff_error.ptr());
    py::register_exception<gipc::CheckpointError>(
        m, "CheckpointError", stiff_error.ptr());
    py::register_exception<gipc::LifecycleError>(
        m, "LifecycleError", stiff_error.ptr());
    m.def("_test_ccd_nan_max_speed_fail_fast",
          &stiff_test_ccd_nan_max_speed_fail_fast,
          "Internal regression hook for device-CCD NaN fail-fast.");

    py::enum_<frame_fsm::FrameResult>(m, "FrameResult")
        .value("OK", frame_fsm::FRAME_OK)
        .value("RETRY_REQUIRED", frame_fsm::FRAME_RETRY_REQUIRED)
        .value("FATAL", frame_fsm::FRAME_FATAL)
        .value("RUNTIME_ERROR", frame_fsm::FRAME_RUNTIME_ERROR);

    py::class_<frame_fsm::FrameStatus>(m, "FrameStatus")
        .def_readonly("result", &frame_fsm::FrameStatus::result)
        .def_readonly("phase", &frame_fsm::FrameStatus::phase)
        .def_readonly("invalid_bits", &frame_fsm::FrameStatus::invalid_bits)
        .def_readonly("launch_status", &frame_fsm::FrameStatus::launch_status)
        .def_readonly("err_env", &frame_fsm::FrameStatus::err_env)
        .def_readonly("err_primitive", &frame_fsm::FrameStatus::err_primitive)
        .def_readonly("err_newton_iter",
                      &frame_fsm::FrameStatus::err_newton_iter)
        .def_readonly("err_ls_iter", &frame_fsm::FrameStatus::err_ls_iter)
        .def_readonly("error_code", &frame_fsm::FrameStatus::error_code)
        .def_readonly("path_flags", &frame_fsm::FrameStatus::path_flags)
        .def_readonly("graph_launches",
                      &frame_fsm::FrameStatus::graph_launches)
        .def_readonly("host_boundaries",
                      &frame_fsm::FrameStatus::host_boundaries)
        .def_readonly("substeps", &frame_fsm::FrameStatus::substeps)
        .def_readonly("newton_iters", &frame_fsm::FrameStatus::newton_iters)
        .def_readonly("pcg_iters", &frame_fsm::FrameStatus::pcg_iters)
        .def_readonly("ls_trials", &frame_fsm::FrameStatus::ls_trials)
        .def_readonly("hw_dcd_pairs", &frame_fsm::FrameStatus::hw_dcd_pairs)
        .def_readonly("hw_ccd_pairs", &frame_fsm::FrameStatus::hw_ccd_pairs)
        .def_readonly("hw_triplets", &frame_fsm::FrameStatus::hw_triplets)
        .def_readonly("hw_unique_blocks",
                      &frame_fsm::FrameStatus::hw_unique_blocks)
        .def_readonly("required_dcd_pairs",
                      &frame_fsm::FrameStatus::required_dcd_pairs)
        .def_readonly("required_ccd_pairs",
                      &frame_fsm::FrameStatus::required_ccd_pairs)
        .def_readonly("required_triplets",
                      &frame_fsm::FrameStatus::required_triplets)
        .def_readonly("required_unique_blocks",
                      &frame_fsm::FrameStatus::required_unique_blocks)
        .def_readonly("hw_mas_clusters",
                      &frame_fsm::FrameStatus::hw_mas_clusters)
        .def_readonly("required_mas_clusters",
                      &frame_fsm::FrameStatus::required_mas_clusters)
        .def_readonly("root_graph_nodes",
                      &frame_fsm::FrameStatus::root_graph_nodes)
        .def_readonly("root_d2h_nodes",
                      &frame_fsm::FrameStatus::root_d2h_nodes)
        .def_readonly("terminal_graph_nodes",
                      &frame_fsm::FrameStatus::terminal_graph_nodes)
        .def_readonly("terminal_d2h_nodes",
                      &frame_fsm::FrameStatus::terminal_d2h_nodes)
        .def_readonly("final_alpha", &frame_fsm::FrameStatus::final_alpha)
        .def_readonly("final_energy", &frame_fsm::FrameStatus::final_energy)
        .def_readonly("max_movement",
                      &frame_fsm::FrameStatus::max_movement)
        .def_readonly("cfl_alpha", &frame_fsm::FrameStatus::cfl_alpha)
        .def_readonly("kappa", &frame_fsm::FrameStatus::kappa)
        .def_readonly("frame_id", &frame_fsm::FrameStatus::frame_id)
        .def_readonly("attempt", &frame_fsm::FrameStatus::attempt)
        .def_readonly("retry_count", &frame_fsm::FrameStatus::retry_count)
        .def_readonly("retry_invalid_bits",
                      &frame_fsm::FrameStatus::retry_invalid_bits);

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
        .def_readwrite("newton_velocity_tol",  &SimEngineConfig::newton_velocity_tol)
        .def_readwrite("pcg_tol",              &SimEngineConfig::pcg_tol)
        .def_readwrite("relative_dhat",        &SimEngineConfig::relative_dhat)
        .def_readwrite("joint_strength_ratio",             &SimEngineConfig::joint_strength_ratio)
        .def_readwrite("revolute_driving_strength_ratio",  &SimEngineConfig::revolute_driving_strength_ratio)
        .def_readwrite("prismatic_strength_ratio",         &SimEngineConfig::prismatic_strength_ratio)
        .def_readwrite("prismatic_driving_strength_ratio", &SimEngineConfig::prismatic_driving_strength_ratio)
        .def_readwrite("max_revolute_step_per_frame",      &SimEngineConfig::max_revolute_step_per_frame)
        .def_readwrite("max_prismatic_step_per_frame",     &SimEngineConfig::max_prismatic_step_per_frame)
        .def_readwrite("collision_detection_buff_scale",   &SimEngineConfig::collision_detection_buff_scale)
        .def_readwrite("linear_system_buff_scale",         &SimEngineConfig::linear_system_buff_scale)
        .def_readwrite("triplet_internal_margin",          &SimEngineConfig::triplet_internal_margin)
        .def_readwrite("absolute_dhat",                     &SimEngineConfig::absolute_dhat)
        .def_readwrite("preconditioner_type",              &SimEngineConfig::preconditioner_type)
        .def_readwrite("cuda_device",                      &SimEngineConfig::cuda_device)
        .def_readwrite("semi_implicit_enabled",            &SimEngineConfig::semi_implicit_enabled)
        .def_readwrite("semi_implicit_beta_tol",           &SimEngineConfig::semi_implicit_beta_tol)
        .def_readwrite("semi_implicit_min_iter",           &SimEngineConfig::semi_implicit_min_iter)
        .def_readwrite("newton_iter_cap",                  &SimEngineConfig::newton_iter_cap)
        .def_readwrite("env_newton_iter_cap",              &SimEngineConfig::env_newton_iter_cap)
        .def_readwrite("line_search_max_iter",             &SimEngineConfig::line_search_max_iter)
        .def_readwrite("energy_abs_tol",                   &SimEngineConfig::energy_abs_tol)
        .def_readwrite("energy_rel_tol",                   &SimEngineConfig::energy_rel_tol)
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
        .def("set_body_groups", &SimEngine::set_body_groups,
             py::arg("groups"),
             "One dense [0,N) group id per collision body (N<=256). "
             "Wildcard -1 is merged-only; isolated/strict require all bodies grouped.")
        .def("set_vertex_env_ids", &SimEngine::set_vertex_env_ids,
             py::arg("env_ids"))
        .def("set_env_offsets", &SimEngine::set_env_offsets,
             py::arg("per_group_xyz"),
             "[multi-env] per-group world offset (flat xyz, 3 per group). Engine expands "
             "per-vertex; BVH separates envs while narrow-phase stays local (deterministic). "
             "Call after finalize(); keep up-axis component 0.")
        .def("add_ground_collision_skip", &SimEngine::add_ground_collision_skip,
             py::arg("body_id"))
        .def("add_stitch_spring",
             [](SimEngine& e, int fem_v, int abd_v, int abd_body,
                std::array<double, 3> rest_off) {
                 Eigen::Vector3d ro(rest_off[0], rest_off[1], rest_off[2]);
                 e.add_stitch_spring(fem_v, abd_v, abd_body, ro);
             },
             py::arg("fem_vertex_id"), py::arg("abd_anchor_vertex_id"),
             py::arg("abd_body_id"),
             py::arg("rest_offset_world") = std::array<double, 3>{0.0, 0.0, 0.0})
        .def("set_per_tet_young_for_body",
             [](SimEngine& e, int body_offset, py::array_t<double> per_tet_young) {
                 auto b = per_tet_young.request();
                 const double* p = static_cast<const double*>(b.ptr);
                 std::vector<double> vec(p, p + b.size);
                 e.set_per_tet_young_for_body(body_offset, vec);
             },
             py::arg("body_offset"), py::arg("per_tet_young"),
             "Override per-tet Young's modulus for one FEM body. Must call "
             "BEFORE finalize(). Array length must equal the body's tet count "
             "(tets all of whose 4 vertices lie in the body's vertex range).")
        .def("add_fem_pin_to_abd",
             [](SimEngine& e, int fem_v, int abd_anchor_v, int abd_body_id,
                std::array<double, 3> rest_off) {
                 Eigen::Vector3d ro(rest_off[0], rest_off[1], rest_off[2]);
                 e.add_fem_pin_to_abd(fem_v, abd_anchor_v, abd_body_id, ro);
             },
             py::arg("fem_vertex_id"), py::arg("abd_anchor_vertex_id"),
             py::arg("abd_body_id"),
             py::arg("rest_offset_world") = std::array<double, 3>{0.0, 0.0, 0.0},
             "[M1 substitution method] Hard kinematic constraint: FEM vertex's "
             "world position is forced to q.t + R(q) * fem_local_pos each step "
             "(after ABD step_forward, before line-search energy eval). "
             "Implementation: vertex's BoundaryType set to Fixed and mass=1e30, "
             "so PCG gives Δx≈0; this kernel then writes the ABD-derived "
             "position. ABD body doesn't feel reaction force from the pin "
             "(in M1 — M2 will add the cross-term Hessian for proper "
             "force feedback).")

        .def("add_fem_pins_with_local_pos",
             [](SimEngine& e,
                py::array_t<int, py::array::c_style | py::array::forcecast> fem_ids,
                py::array_t<int, py::array::c_style | py::array::forcecast> body_ids,
                py::array_t<double, py::array::c_style | py::array::forcecast> local_pos)
             {
                 auto fi = fem_ids.unchecked<1>();
                 auto bi = body_ids.unchecked<1>();
                 auto lp = local_pos.unchecked<2>();
                 const int n = static_cast<int>(fi.shape(0));
                 if(bi.shape(0) != n || lp.shape(0) != n || lp.shape(1) != 3)
                 {
                     throw std::invalid_argument(
                         "add_fem_pins_with_local_pos: array shapes must be "
                         "(n,), (n,), (n,3)");
                 }
                 std::vector<int> fv(n), bv(n);
                 std::vector<Eigen::Vector3d> lv(n);
                 for(int i = 0; i < n; ++i)
                 {
                     fv[i] = fi(i);
                     bv[i] = bi(i);
                     lv[i] = Eigen::Vector3d(lp(i, 0), lp(i, 1), lp(i, 2));
                 }
                 e.add_fem_pins_with_local_pos(fv, bv, lv);
             },
             py::arg("fem_vertex_ids"), py::arg("abd_body_ids"),
             py::arg("abd_local_positions"),
             "[Hybrid mesh] Bulk-add FEM pins with explicit local positions in "
             "the ABD body's rest frame.  Use this with the .npz output from "
             "tools/build_hybrid_mesh.py — feed in vertex_abd_body_id (filtered "
             "to >=0) and vertex_local_pos (matching rows).  fem_vertex_ids "
             "must be GLOBAL vertex indices (= local_idx + body_offset for the "
             "FEM body).")

        .def("set_abd_body_face_orient", &SimEngine::set_abd_body_face_orient,
             py::arg("body_id"), py::arg("orient"))
        .def("get_abd_body_face_orient", &SimEngine::get_abd_body_face_orient,
             py::arg("body_id"))
        .def("get_abd_surface_body_vertices", [](const SimEngine& e, int body_id) {
            auto flat = e.get_abd_surface_body_vertices(body_id);
            int n = static_cast<int>(flat.size() / 3);
            auto arr = py::array_t<double>({n, 3});
            std::memcpy(arr.mutable_data(), flat.data(), flat.size() * sizeof(double));
            return arr;
        }, py::arg("body_id"))
        .def("get_abd_surface_body_triangles", [](const SimEngine& e, int body_id) {
            auto flat = e.get_abd_surface_body_triangles(body_id);
            int n = static_cast<int>(flat.size() / 3);
            auto arr = py::array_t<int>({n, 3});
            std::memcpy(arr.mutable_data(), flat.data(), flat.size() * sizeof(int));
            return arr;
        }, py::arg("body_id"))

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
                                      const std::string& name,
                                      bool passive) {
            auto ax = axis.unchecked<1>();
            auto p  = pos.unchecked<1>();
            return self.add_revolute_joint(parent, child,
                Eigen::Vector3d(ax(0), ax(1), ax(2)),
                Eigen::Vector3d(p(0), p(1), p(2)),
                lower, upper, init_angle, name, passive);
        }, py::arg("parent_body"), py::arg("child_body"),
           py::arg("world_axis"), py::arg("joint_pos"),
           py::arg("lower_limit"), py::arg("upper_limit"),
           py::arg("initial_angle") = 0.0, py::arg("name") = "",
           py::arg("passive") = false)

        .def("add_prismatic_joint", [](SimEngine& self, int parent, int child,
                                       py::array_t<double> center,
                                       py::array_t<double> axis,
                                       double lower, double upper,
                                       const std::string& name,
                                       bool passive) {
            auto c  = center.unchecked<1>();
            auto ax = axis.unchecked<1>();
            return self.add_prismatic_joint(parent, child,
                Eigen::Vector3d(c(0), c(1), c(2)),
                Eigen::Vector3d(ax(0), ax(1), ax(2)),
                lower, upper, name, passive);
        }, py::arg("parent_body"), py::arg("child_body"),
           py::arg("world_center"), py::arg("world_axis"),
           py::arg("lower_limit"), py::arg("upper_limit"),
           py::arg("name") = "", py::arg("passive") = false)

        .def("set_vertex_boundary", &SimEngine::set_vertex_boundary,
             py::arg("vertex_index"), py::arg("boundary_type"))

        .def("get_abd_body_count",    &SimEngine::get_abd_body_count)
        .def("get_fem_body_count",    &SimEngine::get_fem_body_count)
        .def("get_vertex_count_host", &SimEngine::get_vertex_count_host)
        .def("get_vertices_host", [](const SimEngine& e) {
            int n = e.get_vertex_count_host();
            auto arr = py::array_t<double>({n, 3});
            if(n > 0) e.get_vertex_positions_host(arr.mutable_data(), n);
            return arr;
        })

        .def("get_vertex_position_host", [](const SimEngine& e, int idx) {
            double xyz[3];
            e.get_vertex_position_host(idx, xyz);
            return py::make_tuple(xyz[0], xyz[1], xyz[2]);
        }, py::arg("idx"))

        .def("finalize", &SimEngine::finalize)

        .def("set_log_level", &SimEngine::set_log_level, py::arg("level"),
             "Per-frame solver log verbosity: 0 = silent, >=1 = verbose (default).")
        .def("reset", &SimEngine::reset,
             "Tear down the whole world (bodies, constraints, GPU buffers), keep "
             "Config. Re-run load_*()+finalize() after.")

        .def("step", &SimEngine::step,
             py::call_guard<py::gil_scoped_release>())
        .def("get_frame_status", &SimEngine::get_frame_status,
             "Return the latest frame-boundary status packet.")

        .def("launch_episode_async",
             [](SimEngine& e,
                int frames,
                py::object revolute_actions,
                py::object prismatic_actions)
             {
                 using ActionArray = py::array_t<
                     double,
                     py::array::c_style | py::array::forcecast>;
                 const int revolute_count =
                     e.get_num_revolute_joints();
                 const int prismatic_count =
                     e.get_num_prismatic_joints();
                 ActionArray revolute;
                 ActionArray prismatic;
                 const double* revolute_data = nullptr;
                 const double* prismatic_data = nullptr;

                 auto validate = [frames](
                     py::object object,
                     int joints,
                     const char* label,
                     ActionArray& array,
                     const double*& data)
                 {
                     if(joints == 0 && object.is_none())
                         return;
                     if(object.is_none())
                         throw py::value_error(
                             std::string(label)
                             + " actions are required by this scene");
                     array = ActionArray::ensure(object);
                     if(!array)
                         throw py::type_error(
                             std::string(label)
                             + " actions must be float-compatible");
                     const py::buffer_info info = array.request();
                     if(info.ndim != 3
                        || info.shape[0] != frames
                        || info.shape[1] != joints
                        || info.shape[2] != 3)
                     {
                         std::ostringstream message;
                         message << label
                                 << " actions must have shape ("
                                 << frames << ", " << joints
                                 << ", 3)";
                         throw py::value_error(message.str());
                     }
                     data = static_cast<const double*>(info.ptr);
                 };
                 validate(
                     revolute_actions,
                     revolute_count,
                     "revolute",
                     revolute,
                     revolute_data);
                 validate(
                     prismatic_actions,
                     prismatic_count,
                     "prismatic",
                     prismatic,
                     prismatic_data);
                 py::gil_scoped_release release;
                 e.launch_episode_async(
                     frames,
                     revolute_data,
                     revolute_count,
                     prismatic_data,
                     prismatic_count);
             },
             py::arg("frames"),
             py::arg("revolute_actions") = py::none(),
             py::arg("prismatic_actions") = py::none(),
             "Launch one device-resident RL episode asynchronously. A prior "
             "warm-up step() is required. Action arrays have shape "
             "(frames, joints, 3): target, strength, external force/torque.")
        .def("episode_in_flight", &SimEngine::episode_in_flight)
        .def("episode_observation_ready",
             &SimEngine::episode_observation_ready,
             py::arg("slot"),
             "Non-blocking query for asynchronous observation slot 0 or 1.")
        .def("wait_episode_observation",
             &SimEngine::wait_episode_observation,
             py::arg("slot"),
             py::call_guard<py::gil_scoped_release>(),
             "Wait only for the requested observation slot's CUDA event.")
        .def("get_episode_observation",
             [](const SimEngine& e, int slot)
             {
                 const int frames =
                     e.get_episode_slot_frame_count(slot);
                 const int vertices = e.get_vertex_count();
                 auto positions =
                     py::array_t<double>({frames, vertices, 3});
                 auto velocities =
                     py::array_t<double>({frames, vertices, 3});
                 std::vector<frame_fsm::FrameStatus> statuses(frames);
                 e.get_episode_observation(
                     slot,
                     positions.mutable_data(),
                     velocities.mutable_data(),
                     statuses.data(),
                     frames);
                 py::list py_statuses;
                 for(const auto& status : statuses)
                     py_statuses.append(py::cast(status));
                 py::dict result;
                 result["first_frame"] =
                     e.get_episode_slot_first_frame(slot);
                 result["positions"] = std::move(positions);
                 result["velocities"] = std::move(velocities);
                 result["statuses"] = std::move(py_statuses);
                 return result;
             },
             py::arg("slot"),
             "Return a ready slot as {first_frame, positions, velocities, "
             "statuses}; this performs only pinned-host memory copies.")
        .def("get_episode_attempted_frame_count",
             &SimEngine::get_episode_attempted_frame_count)
        .def("finish_episode",
             &SimEngine::finish_episode,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for the terminal slot, commit telemetry, and return the "
             "number of successful frames.")

        .def("prepare_gpu_rl",
             &SimEngine::prepare_gpu_rl,
             py::call_guard<py::gil_scoped_release>(),
             "Capture the reusable one-frame GPU-native RL graph. This is a "
             "setup boundary; call one warm-up step() first.")
        .def("prepare_gpu_rl_episode",
             &SimEngine::prepare_gpu_rl_episode,
             py::arg("frame_count"),
             py::call_guard<py::gil_scoped_release>(),
             "Capture a multi-frame device-native RL episode. The action "
             "slab and observations remain on device and one graph launch "
             "executes the whole conditional loop.")
        .def("launch_gpu_rl_async",
             &SimEngine::launch_gpu_rl_async,
             py::arg("cuda_stream") = uintptr_t{0},
             "Enqueue one RL simulation step without a host wait. The policy "
             "must write actions and consume outputs on this same CUDA stream.")
        .def("launch_gpu_rl_episode_async",
             &SimEngine::launch_gpu_rl_episode_async,
             py::arg("cuda_stream") = uintptr_t{0},
             "Launch the prepared multi-frame device-native episode once; "
             "no per-frame host graph launch is required.")
        .def("gpu_rl_prepared", &SimEngine::gpu_rl_prepared)
        .def("gpu_rl_ready",
             &SimEngine::gpu_rl_ready,
             "Optional host-side completion query; not part of the steady "
             "GPU-native path.")
        .def("synchronize_gpu_rl",
             &SimEngine::synchronize_gpu_rl,
             py::call_guard<py::gil_scoped_release>(),
             "Explicit debug/teardown wait; omit from steady-state RL.")
        .def("end_gpu_rl",
             &SimEngine::end_gpu_rl,
             py::call_guard<py::gil_scoped_release>(),
             "Synchronize and release GPU-native RL episode resources.")
        .def("launch_gpu_rl_reset_async",
             &SimEngine::launch_gpu_rl_reset_async,
             py::arg("cuda_stream") = uintptr_t{0},
             "[D2] Enqueue an in-stream reset to the prepare-time state "
             "snapshot (pure D2D on the bound stream; no host wait).")
        .def("get_point_to_group_device_ptr",
             &SimEngine::get_point_to_group_device_ptr)
        .def("launch_gpu_rl_reset_masked_async",
             &SimEngine::launch_gpu_rl_reset_masked_async,
             pybind11::arg("d_env_mask"),
             pybind11::arg("cuda_stream") = 0)
        .def("get_gpu_rl_device_abi",
             [](const SimEngine& e)
             {
                 py::dict result;
                 result["action_dtype"] = "float64";
                 result["action_components"] = 3;
                 result["revolute_joints"] =
                     e.get_num_revolute_joints();
                 result["prismatic_joints"] =
                     e.get_num_prismatic_joints();
                 result["vertices"] = e.get_vertex_count();
                 result["revolute_actions"] =
                     e.get_gpu_rl_revolute_actions_device_ptr();
                 result["prismatic_actions"] =
                     e.get_gpu_rl_prismatic_actions_device_ptr();
                 result["positions"] =
                     e.get_gpu_rl_positions_device_ptr();
                 result["velocities"] =
                     e.get_gpu_rl_velocities_device_ptr();
                 result["statuses"] =
                     e.get_gpu_rl_statuses_device_ptr();
                 result["frame_counter"] =
                     e.get_gpu_rl_frame_counter_device_ptr();
                 result["status_bytes"] =
                     e.get_gpu_rl_status_size_bytes();
                 result["graph_nodes"] =
                     e.get_gpu_rl_graph_node_count();
                 result["graph_h2d"] =
                     e.get_gpu_rl_graph_h2d_count();
                 result["graph_d2h"] =
                     e.get_gpu_rl_graph_d2h_count();
                 result["episode_frame_count"] =
                     e.get_gpu_rl_episode_frame_count();
                 // Multi-frame device-native episodes use a contiguous
                 // frame-major slab.  Expose byte/element strides so a
                 // policy can populate it without guessing the layout.
                 result["action_frame_stride"] =
                     3 * (e.get_num_revolute_joints() +
                          e.get_num_prismatic_joints());
                 result["revolute_action_frame_stride"] =
                     3 * e.get_num_revolute_joints();
                 result["prismatic_action_frame_stride"] =
                     3 * e.get_num_prismatic_joints();
                 result["position_frame_stride"] = 3 * e.get_vertex_count();
                 result["velocity_frame_stride"] = 3 * e.get_vertex_count();
                 result["status_frame_stride"] =
                     e.get_gpu_rl_status_size_bytes();
                 // [D2] graph-refreshed joint observations:
                 // {angle, rate} per revolute driving joint, then
                 // {displacement, rate} per prismatic driving joint.
                 result["joint_observations"] =
                     e.get_gpu_rl_joint_observations_device_ptr();
                 result["joint_observation_count"] =
                     e.get_gpu_rl_joint_observation_count();
                 // [D3] merged multi-env batching handles: per-vertex env
                 // id (-1 = wildcard) and per-env quarantine flags
                 // (nonzero = poisoned env, treat as done).
                 result["point_to_group"] =
                     e.get_point_to_group_device_ptr();
                 result["env_quarantined"] =
                     e.get_env_quarantined_device_ptr();
                 result["env_count"] = e.get_env_group_count();
                 return result;
             },
             "Return raw device pointers and the audited graph ABI. Positions "
             "and velocities are contiguous (vertices, 3) float64 arrays in "
             "engine-internal vertex order.")

        .def("get_assets_dir", &SimEngine::get_assets_dir)

        .def("get_vertex_contact_forces", [](SimEngine& e, bool include_ground,
                                             int components) {
                int n = e.get_vertex_count();
                auto arr = py::array_t<double>({n, 3});
                int nw = e.get_vertex_contact_forces(arr.mutable_data(), n,
                                                     include_ground, components);
                (void)nw;
                return arr;
            }, py::arg("include_ground") = true, py::arg("components") = 0,
            "[contact-force distribution] Per-vertex physical contact FORCES "
            "(N,3) in NEWTONS (= -gradient/dt^2). components: 0 = normal only "
            "(body-body barrier, + ground barrier when include_ground) — "
            "historic behavior; 1 = friction_lagged only (the friction-"
            "potential gradient the solver ACTUALLY used this step: positions "
            "current, lambda/tangent basis lagged one step); 2 = total. NOTE: "
            "units differ from the legacy batched API (raw gradients). "
            "Rebuilds normal contacts once when requested; the lagged-friction "
            "path is read-only on the solver's frozen friction set.")
        .def("get_fem_von_mises_stress", [](SimEngine& e) {
                int n = e.get_vertex_count();
                auto arr = py::array_t<double>(n);
                e.get_fem_von_mises_stress(arr.mutable_data(), n);
                return arr;
            },
            "[FEM stress] Per-vertex von Mises stress (Pa) from the configured "
            "tetrahedral constitutive law, max over incident tets. Non-tet "
            "vertices (cloth/ABD) are 0.")
        .def("get_total_newton_iters", &SimEngine::get_total_newton_iters)
        .def("get_ls_exhausted_count", &SimEngine::get_ls_exhausted_count)
        .def("get_ls_nonfinite_count", &SimEngine::get_ls_nonfinite_count)
        .def("get_per_env_newton_iters", &SimEngine::get_per_env_newton_iters,
             "[per-env] Newton iter at which each env froze last solve (-1 = ran "
             "to loop end / absent). 256 slots. Host per-env path only.")
        .def("get_per_env_status", &SimEngine::get_per_env_status,
             "[per-env] Status of the last solve per env: 0 active/absent, "
             "1 converged, 2 timeout (env_newton_iter_cap), 3 diverged. 256 slots.")
        .def("get_total_pcg_iters", &SimEngine::get_total_pcg_iters)
        .def("get_total_collision_pairs", &SimEngine::get_total_collision_pairs)
        .def("get_max_collision_pairs", &SimEngine::get_max_collision_pairs)
        .def("get_total_frames_done", &SimEngine::get_total_frames_done)
#ifdef GIPC_ENABLE_DIAGNOSTICS
        .def("debug_fd_gradient_check", &SimEngine::debug_fd_gradient_check,
             py::arg("h") = 1e-6, py::arg("nprobes") = 64, py::arg("seed") = 12345)
        .def("debug_fd_hessian_check", &SimEngine::debug_fd_hessian_check,
             py::arg("h") = 1e-5, py::arg("nprobes") = 16, py::arg("seed") = 54321)
        .def("debug_fd_activity", &SimEngine::debug_fd_activity)
#endif
        .def("get_total_energy_tolerance_accepts",
             &SimEngine::get_total_energy_tolerance_accepts,
             "Number of final line-search decisions accepted only by the optional "
             "energy tolerance since engine creation/reset.")


        // Vertex positions as numpy array (N, 3) float64
        .def("get_vertices", [](const SimEngine& e) {
            int n = e.get_vertex_count();
            auto arr = py::array_t<double>({n, 3});
            e.get_vertex_positions(arr.mutable_data(), n);
            return arr;
        })

        // [decouple] per-vertex env/group id aligned to get_vertices() order. -1 = ungrouped.
        // verts[groups==g] extracts env g's vertices regardless of the engine's type-grouped layout.
        .def("get_point_groups", [](const SimEngine& e) {
            int n = e.get_vertex_count();
            auto arr = py::array_t<int>(n);
            e.get_point_groups(arr.mutable_data(), n);
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

        // Teleport FEM vertices: writes _vertexes, o_vertexes, xTilta and
        // (optionally) velocities. See sim_engine.h for rationale.
        // Versioned frame-boundary integrator checkpoint.
        .def("save_checkpoint", [](SimEngine& e, const std::string& path) { e.save_checkpoint(path); }, py::arg("path"))
        .def("load_checkpoint", [](SimEngine& e, const std::string& path) { e.load_checkpoint(path); }, py::arg("path"))

        .def("teleport_fem_vertices", [](SimEngine& e,
                py::array_t<double, py::array::c_style | py::array::forcecast> positions,
                py::object velocities) {
            auto pbuf = positions.request();
            const double* vptr = nullptr;
            py::buffer_info vbuf;
            if(!velocities.is_none()) {
                py::array_t<double, py::array::c_style | py::array::forcecast>
                    vel_arr(velocities);
                vbuf = vel_arr.request();
                vptr = static_cast<const double*>(vbuf.ptr);
            }
            e.teleport_fem_vertices(static_cast<const double*>(pbuf.ptr),
                                    static_cast<int>(pbuf.shape[0]),
                                    vptr);
        }, py::arg("positions"), py::arg("velocities") = py::none())

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

        // Per-step soft-target driver for ABD bodies loaded with
        // boundary_type=Animated (=3).  Updates body_motor_params on the GPU
        // so the engine's Animated penalty pulls q.t toward (x,y,z) and
        // q.A toward identity.  The body's 12 DOFs stay in PCG → joint
        // attachments and M3.5 chain-rule pins propagate normally.
        .def("set_body_animated_target", &SimEngine::set_body_animated_target,
             py::arg("body_id"),
             py::arg("target_x"), py::arg("target_y"), py::arg("target_z"),
             py::arg("strength") = 0.0,
             "Set per-step Animated target for an ABD body (target world position, "
             "soft penalty stiffness). Strength<=0 → engine default 1e6. "
             "Body must have been loaded with boundary_type='Animated' or =3.")

        .def("set_body_apply_gravity", &SimEngine::set_body_apply_gravity,
             py::arg("body_id"), py::arg("enabled"),
             "Toggle gravity for all verts of body_id (global body id).  Use "
             "for ABD bodies anchored to Fixed parent via joint to avoid drift "
             "from joint penalty wrestling gravity.  Call AFTER finalize().")
        .def("set_abd_body_density", &SimEngine::set_abd_body_density,
             py::arg("body_id"), py::arg("density"),
             "Override one ABD body's density (mass = density * volume). Lets a "
             "scene mix per-body densities. Call AFTER loading the body and "
             "BEFORE finalize().")
        .def("set_abd_body_mass", &SimEngine::set_abd_body_mass,
             py::arg("body_id"), py::arg("mass"),
             "Override one surface-mesh ABD body's total mass in kilograms. "
             "Call AFTER loading the body and BEFORE finalize().")
        .def("set_body_friction", &SimEngine::set_body_friction,
             py::arg("body_offset"), py::arg("mu"), py::arg("ground_mu") = -1.0,
             "[per-body friction] Override one body's friction coefficient "
             "(body_offset = index into get_load_records()). Pairs combine "
             "sqrt(mu_a*mu_b); ground_mu < 0 keeps the global gd_friction_rate. "
             "Call AFTER loading the body, BEFORE finalize().")
        .def("set_soft_body_density", &SimEngine::set_soft_body_density,
             py::arg("body_offset"), py::arg("density"),
             "[per-body density] Override one SOFT body's density (FEM tets or "
             "cloth shell). body_offset = index into get_load_records(). Call "
             "AFTER loading the body, BEFORE finalize(). Unset bodies keep the "
             "global Config.density / cloth_density.")
        .def("set_abd_body_inertia", [](SimEngine& self, int body_id, double mass,
                py::array_t<double, py::array::c_style | py::array::forcecast> com,
                py::array_t<double, py::array::c_style | py::array::forcecast> inertia) {
            auto c = com.request(); auto in = inertia.request();
            self.set_abd_body_inertia(body_id, mass,
                static_cast<const double*>(c.ptr), static_cast<const double*>(in.ptr));
        }, py::arg("body_id"), py::arg("mass"), py::arg("com"), py::arg("inertia"),
           "Override one ABD body's inertial props: mass (scalar), com (3,), "
           "inertia (9, row-major 3x3 about COM), in the load frame. Use authored "
           "URDF inertia instead of welded-mesh geometry. "
           "Call BEFORE finalize().")
        .def("set_body_external_force", &SimEngine::set_body_external_force,
             py::arg("body_id"), py::arg("fx"), py::arg("fy"), py::arg("fz"),
             "[force-control] Set per-body external LINEAR force (N) on an ABD "
             "body (global body id). Persistent until changed; (0,0,0) clears. "
             "Applied as acceleration M^-1 F in q_tilde, like gravity. Call "
             "AFTER finalize(). Mirrors libuipc AffineBodyExternalBodyForce.")
        .def("set_body_external_wrench", [](SimEngine& self, int body_id,
                                            py::array_t<double> w) {
            auto a = w.unchecked<1>();
            if(a.shape(0) != 12)
                throw std::runtime_error("set_body_external_wrench expects a length-12 array");
            double w12[12];
            for(int i = 0; i < 12; ++i) w12[i] = a(i);
            self.set_body_external_wrench(body_id, w12);
        }, py::arg("body_id"), py::arg("wrench12"),
           "[force-control] Set the FULL 12-DOF external generalized force on an "
           "ABD body: wrench12[0:3]=linear force, wrench12[3:12]=affine force "
           "(row-major vec(F_A)). An affine term with w[5]=+omega, w[9]=-omega is "
           "a spin torque about Y. Mirrors libuipc's body-force test (orbiting "
           "linear force + spinning affine term). Call AFTER finalize().")

        .def("get_urdf_link_transform", &SimEngine::get_urdf_link_transform,
             py::arg("link_name"),
             "Returns 4x4 world transform of a URDF link (computed by importer's "
             "FK after load_urdf). Use to attach extra bodies in correct frame.")

        .def("set_urdf_mesh_override", &SimEngine::set_urdf_mesh_override,
             py::arg("link_name"), py::arg("msh_path"), py::arg("young_modulus") = 1e7,
             "Override the mesh used for a URDF link.  Call BEFORE load_urdf(). "
             "Required when URDF references a mesh file that doesn't exist on "
             "disk.  Cleared after each load_urdf() call.")

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
        .def("get_vertices_device_ptr",   &SimEngine::get_vertices_device_ptr,
             "[gpu-direct] Raw device pointer (int) to the double3 vertex buffer "
             "(length get_vertex_count()). For zero-copy GPU readback via Warp.")
        .def("get_vertex_velocities_device_ptr",
             &SimEngine::get_vertex_velocities_device_ptr,
             "[gpu-direct] Raw device pointer (int) to the double3 velocity "
             "buffer (length get_vertex_count()).")
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
        .def("set_revolute_torque",       &SimEngine::set_revolute_torque,
             py::arg("idx"), py::arg("torque"),
             "[force-control] External torque (N*m) on revolute driving joint "
             "idx. Adds -tau*dtheta/dq to the gradient (no Hessian), like "
             "libuipc external torque. For pure torque control also call "
             "set_revolute_strength(idx, 0).")
        .def("set_revolute_initial_offset", &SimEngine::set_revolute_initial_offset,
             py::arg("idx"), py::arg("offset_rad"))
        .def("set_prismatic_target",      &SimEngine::set_prismatic_target,
             py::arg("idx"), py::arg("distance_m"))
        .def("set_prismatic_force",       &SimEngine::set_prismatic_force,
             py::arg("idx"), py::arg("force"),
             "[force-control] External force (N) along prismatic driving joint "
             "idx, applied via the q_tilde path (no Hessian), like libuipc "
             "external prismatic force. +force pushes the child along +axis. For "
             "pure force control also call set_prismatic_strength(idx, 0).")
        .def("set_prismatic_limit_barrier", &SimEngine::set_prismatic_limit_barrier,
             py::arg("idx"), py::arg("cl"), py::arg("dir"), py::arg("dhat"), py::arg("kappa"),
             py::arg("slot") = 0,
             "[force-control] Arm a one-sided IPC log-barrier on prismatic joint "
             "idx at coordinate `cl`: the solver then never lets the opening d cross "
             "cl no matter how large the (force) drive — a HARD no-overshoot "
             "guarantee while the joint stays force-controlled. dir=+1 if the "
             "allowed side is at d>cl else -1; dhat=activation band (m); "
             "kappa=barrier stiffness (<=0 disarms). slot 0 = closed-end barrier, "
             "slot 1 = open-end barrier (arm both for hard limits on BOTH ends). "
             "Call after finalize().")

        .def("set_fixed_joint_strength",  &SimEngine::set_fixed_joint_strength,
             py::arg("idx"), py::arg("kappa"),
             "Override per-fixed-joint stiffness (kappa). Default ~8e-3 is "
             "too weak for hybrid gripper attachment; set 1e6 for tight tracking.")
        .def("set_revolute_strength",     &SimEngine::set_revolute_strength,
             py::arg("idx"), py::arg("strength"),
             "Set per-joint driving strength multiplier. Lower = joint yields "
             "under contact (e.g. gripper vs cloth). Effective K = "
             "Config.revolute_driving_strength_ratio * strength * (m_p+m_c).")
        .def("set_prismatic_strength",    &SimEngine::set_prismatic_strength,
             py::arg("idx"), py::arg("strength"))

        .def("set_max_revolute_step_per_frame",
             &SimEngine::set_max_revolute_step_per_frame, py::arg("rad"),
             "Cap on per-step revolute joint angle change (radians).  Default "
             "0.1 rad (5.7 deg).  For scenes with fine FEM softpads pinned to "
             "ABD bodies, lower this (e.g. 0.01 rad = 0.6 deg) so the "
             "kinematic teleport of pinned FEM vertices does not outpace the "
             "elastic response of free neighbors and trigger softpad "
             "self-intersection.")
        .def("set_max_prismatic_step_per_frame",
             &SimEngine::set_max_prismatic_step_per_frame, py::arg("m"),
             "Cap on per-step prismatic joint displacement change (meters). "
             "Default 0.002 m. Raise (e.g. 0.02) for position-driving demos that "
             "must reach a larger target within few frames.")

        .def("get_revolute_target",       &SimEngine::get_revolute_target,  py::arg("idx"))
        .def("get_prismatic_target",      &SimEngine::get_prismatic_target, py::arg("idx"))
        .def("get_prismatic_drive_force", &SimEngine::get_prismatic_drive_force, py::arg("idx"),
             "[force-control] Current prismatic driving force K*(target-d) (N). "
             "Used by force-limited position control to cap the grip force.")
        .def("get_prismatic_current_distance", &SimEngine::get_prismatic_current_distance,
             py::arg("idx"),
             "[force-control] Current prismatic opening d (m) along the joint axis. "
             "Lets you see the actual gripper opening (e.g. confirm a force-limited "
             "grasp does not fully close).")
        .def("get_body_contact_force", [](const SimEngine& self, int vert_offset,
                                          int vert_count) {
            double f[3] = {0, 0, 0};
            self.get_vertex_contact_force_sum(vert_offset, vert_count, f);
            auto out = py::array_t<double>(3);
            auto b = out.mutable_unchecked<1>();
            b(0) = f[0]; b(1) = f[1]; b(2) = f[2];
            return out;
        }, py::arg("vert_offset"), py::arg("vert_count"),
           "[force-control][LEGACY UNITS] Sums the body-body barrier GRADIENT over "
           "[vert_offset, vert_offset+vert_count): raw dE/dx = -force*dt^2, NOT "
           "Newtons; excludes ground contact and friction. For physical forces in "
           "Newtons use get_vertex_contact_forces and aggregate per body. "
           "Call AFTER step().")
        .def("get_pair_contact_force", [](const SimEngine& self, int a_off, int a_cnt,
                                          int b_off, int b_cnt) {
            double f[3] = {0, 0, 0};
            self.get_pair_contact_force(a_off, a_cnt, b_off, b_cnt, f);
            auto out = py::array_t<double>(3);
            auto b = out.mutable_unchecked<1>();
            b(0) = f[0]; b(1) = f[1]; b(2) = f[2];
            return out;
        }, py::arg("a_off"), py::arg("a_cnt"), py::arg("b_off"), py::arg("b_cnt"),
           "Net IPC contact force (3-vector) on body A FROM body B: barrier "
           "gradient on A's vertex range, restricted to pairs connecting A and B. "
           "For the contact sensor's per-partner force_matrix. Call AFTER step().")
        .def("get_collision_pairs_clean", [](const SimEngine& self) {
            int n = self.get_collision_pairs_clean(nullptr);
            auto out = py::array_t<int>({n, 4});
            if(n > 0)
                self.get_collision_pairs_clean(static_cast<int*>(out.request().ptr));
            return out;
        }, "Clean export of the current body-body collision pairs as an (N,4) "
           "array of plain vertex indices (-1 padded for PP/PE), UIPC-style. The "
           "solver keeps its MMCVID packing; this is a read-only decoded view. "
           "Call AFTER step().")
        .def("_reset_bvh_traversal_audit", [](const SimEngine&) {
            set_bvh_traversal_audit(1);
            reset_bvh_traversal_audit();
        }, "Validation-only: reset the four aggregate BVH traversal counters.")
        .def("_get_bvh_traversal_audit", [](const SimEngine&) {
            BvhTraversalAudit rows[4] = {};
            get_bvh_traversal_audit(rows);
            auto out = py::array_t<unsigned long long>({4, 4});
            auto view = out.mutable_unchecked<2>();
            for(int family = 0; family < 4; ++family)
            {
                view(family, 0) = rows[family].queries;
                view(family, 1) = rows[family].node_pops;
                view(family, 2) = rows[family].overlapping_children;
                view(family, 3) = rows[family].primitive_tests;
            }
            return out;
        }, "Validation-only: return [queries,pops,overlaps,primitive-tests].")
        .def("get_ccd_pairs_clean", [](const SimEngine& self,
                                        py::array_t<double,
                                            py::array::c_style |
                                            py::array::forcecast> move,
                                        double alpha) {
            const py::buffer_info info = move.request();
            if(info.ndim != 2 || info.shape[1] != 3)
                throw std::runtime_error(
                    "motion must have shape (vertex_count, 3)");
            const auto* data = static_cast<const double*>(info.ptr);
            const int count = static_cast<int>(info.shape[0]);
            const int n = self.get_ccd_pairs_clean(data, count, alpha, nullptr);
            auto out = py::array_t<int>({n, 4});
            if(n > 0)
                self.get_ccd_pairs_clean(
                    data, count, alpha, static_cast<int*>(out.request().ptr));
            return out;
        }, py::arg("motion"), py::arg("alpha") = 1.0,
           "Validation-only clean export of the swept full-CCD candidate set "
           "for an explicit (N,3) host motion field. Mutates direction scratch; "
           "call only after the last frame in a disposable process.")
        .def("get_contacts_device", [](SimEngine& self) {
            int n = self.compute_contacts();
            return py::make_tuple(n, self.contacts_pair_ptr(), self.contacts_force_ptr());
        }, "[Step B] GPU-resident per-contact export. Returns (count, pair_ptr, "
           "force_ptr): device pointers (uintptr) to an int2 (bodyA,bodyB; bodyB=-1 "
           "for ground) and a double3 (world contact force on bodyA, N). Wrap with "
           "warp.array(ptr=..., copy=False). Read-only, valid until next call. "
           "Call AFTER step().")
        .def("get_contacts", [](SimEngine& self) {
            int n = self.compute_contacts();
            auto pair  = py::array_t<int>({n > 0 ? n : 0, 2});
            auto force = py::array_t<double>({n > 0 ? n : 0, 3});
            if(n > 0)
            {
                cudaMemcpy(pair.request().ptr, reinterpret_cast<void*>(self.contacts_pair_ptr()),
                           n * sizeof(int2), cudaMemcpyDeviceToHost);
                cudaMemcpy(force.request().ptr, reinterpret_cast<void*>(self.contacts_force_ptr()),
                           n * sizeof(double3), cudaMemcpyDeviceToHost);
            }
            return py::make_tuple(pair, force);
        }, "[Step B] Host readback of the per-contact export: (pair (N,2) int "
           "(bodyA,bodyB), force (N,3) double world N on bodyA). For validation; "
           "the GPU-direct path uses get_contacts_device(). Call AFTER step().")
        .def("get_stitch_max_stretch", &SimEngine::get_stitch_max_stretch,
             py::arg("pair_start"), py::arg("pair_count"),
             "[force-control] On-GPU MAX stitch-spring stretch (m) over springs "
             "[pair_start, pair_start+pair_count), computed by a device reduction "
             "that returns a single scalar — the soft-gripper grip signal WITHOUT a "
             "full vertex-array D2H. Stitches are stored in add_stitch_spring() call "
             "order, so each finger's pairs are a contiguous range.")
        .def("get_stitch_max_stretch_batched",
             [](const SimEngine& self,
                py::array_t<int, py::array::c_style | py::array::forcecast> starts,
                py::array_t<int, py::array::c_style | py::array::forcecast> counts) {
                 int n = static_cast<int>(starts.size());
                 std::vector<double> out(n > 0 ? n : 1, 0.0);
                 self.get_stitch_max_stretch_batched(starts.data(), counts.data(), n, out.data());
                 auto arr = py::array_t<double>(n);
                 auto b = arr.mutable_unchecked<1>();
                 for(int i = 0; i < n; i++) b(i) = out[i];
                 return arr;
             },
             py::arg("starts"), py::arg("counts"),
             "[force-control] BATCHED on-GPU max stitch stretch: ONE kernel launch, "
             "one CUDA block per segment (= one finger, or one finger of one env). "
             "Returns an array of per-segment maxes. Blocks are independent (no "
             "cross-segment atomics) so different fingers/ENVS never interfere. "
             "Designed for multi-env: pass every finger of every env as a segment.")
        .def("get_body_contact_force_batched",
             [](const SimEngine& self,
                py::array_t<int, py::array::c_style | py::array::forcecast> offsets,
                py::array_t<int, py::array::c_style | py::array::forcecast> counts) {
                 int n = static_cast<int>(offsets.size());
                 std::vector<double> out(n > 0 ? n * 3 : 3, 0.0);
                 self.get_body_contact_force_batched(offsets.data(), counts.data(), n, out.data());
                 auto arr = py::array_t<double>({n, 3});
                 auto b = arr.mutable_unchecked<2>();
                 for(int i = 0; i < n; i++) { b(i, 0) = out[i*3+0]; b(i, 1) = out[i*3+1]; b(i, 2) = out[i*3+2]; }
                 return arr;
             },
             py::arg("offsets"), py::arg("counts"),
             "[force-control][LEGACY UNITS] BATCHED body-body barrier GRADIENT sums "
             "(raw dE/dx = -force*dt^2, NOT Newtons; no ground, no friction): one "
             "contact rebuild + one D2H for all segments -> (n_seg, 3). Kept for "
             "pre-0.8.4 callers; for physical Newtons use get_vertex_contact_forces "
             "and aggregate per segment. Call AFTER step().")

        .def("get_revolute_current_angles", [](const SimEngine& e) {
            int n = e.get_num_revolute_joints();
            auto arr = py::array_t<double>({n});
            if(n > 0)
                e.get_revolute_current_angles(arr.mutable_data(), n);
            return arr;
        })

        .def("get_revolute_initial_offsets", [](const SimEngine& e) {
            int n = e.get_num_revolute_joints();
            auto arr = py::array_t<double>({n});
            if(n > 0)
                e.get_revolute_initial_offsets(arr.mutable_data(), n);
            return arr;
        },
        "Per-joint URDF angle at load time (from load_urdf initial_joint_angles). "
        "Add to get_revolute_current_angles() to get absolute URDF angle.")

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
