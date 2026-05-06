#include <gipc/utils/simple_scene_importer.h>
#include <fstream>
#include <iostream>
#include <Eigen/Geometry>
#include <fem_parameters.h>
#include <metis_sort.h>

namespace gipc
{
template <typename T, int dim, int max_dim = dim>
void from_json(const nlohmann::json&                                  json,
               Eigen::Matrix<T, dim, 1, Eigen::ColMajor, max_dim, 1>& vector)
{
    typedef Eigen::Matrix<T, dim, 1, Eigen::ColMajor, max_dim, 1> Vector;
    std::vector<T> list = json.get<std::vector<T>>();
    vector              = Eigen::Map<Vector>(list.data(), long(list.size()));
}

Eigen::Vector3d read_angular_field(const nlohmann::json& field)
{
    int angular_dim = 3;

    Eigen::Vector3d v;
    if(field.is_number())
    {
        v.setZero();
        v[0] = field.get<double>();
    }
    else
    {
        assert(field.is_array());
        from_json(field, v);
    }

    // Convert to radians for easy use later
    v *= FEM::PI / 180.0;

    return v;
}

SimpleSceneImporter::SimpleSceneImporter(std::string_view scene_file_name,
                                         std::string_view mesh_dir,
                                         BodyType         body_type)
    : mesh_dir{mesh_dir}
    , body_type{body_type}
{
    scene_json = json::parse(std::ifstream(std::string{scene_file_name}));
}


void SimpleSceneImporter::load_geometry(tetrahedra_obj&  tetras,
                                        int              Dimensions,
                                        gipc::BodyType   bodyType,
                                        Eigen::Matrix4d  transform,
                                        double           YoungthM,
                                        std::string      meth_path,
                                        int              preconditionerType,
                                        BodyBoundaryType body_boundary_type,
                                        std::string      metis_output_folder)
{
    int vert_count_before = tetras.vertexes.size();
    std::vector<int> sort_index;  // metis-sorted-idx -> input-vertex-idx (only when MAS active)

    if(Dimensions == 3)
    {

        if(bodyType == gipc::BodyType::ABD)
        {
            tetras.load_tetrahedraMesh(meth_path, transform, YoungthM, bodyType, body_boundary_type);
        }
        else if(bodyType == gipc::BodyType::FEM)
        {
            if(preconditionerType)
            {
                auto paths = metis_sort(meth_path, Dimensions, metis_output_folder, &sort_index);
                tetras.load_tetrahedraMesh(paths[0], transform, YoungthM, bodyType, body_boundary_type);
                tetras.load_parts(paths[1]);
            }
            else
            {
                tetras.load_tetrahedraMesh(meth_path, transform, YoungthM, bodyType, body_boundary_type);
            }
        }
    }
    else if(Dimensions == 2)
    {
        if(preconditionerType)
        {
            auto paths = metis_sort(meth_path, Dimensions, metis_output_folder, &sort_index);
            tetras.load_triMesh(paths[0], transform, 0);
            tetras.load_parts(paths[1]);
        }
        else
        {
            tetras.load_triMesh(meth_path, transform, 0);
        }
    }

    // [MAS-perm] Append per-vertex perm for this body. For non-MAS bodies
    // (or ABD), append identity. For MAS-FEM bodies, append the metis
    // sort_index translated to global indexing (engine_global = body_offset
    // + body_local_engine, input_global = body_offset + body_local_input).
    // The user-facing API (get_vertex_positions / get_surface_faces /
    // set_vertex_positions_gpu) reads this map to transparently expose data
    // in the original input-mesh order.
    int vert_count_after = static_cast<int>(tetras.vertexes.size());
    int n_added          = vert_count_after - vert_count_before;
    if(n_added > 0)
    {
        // First close any gap from earlier loaders (URDF importer) that
        // added vertices without going through SimpleSceneImporter. Identity-
        // pad for those, so perm is contiguous and indexable.
        if(static_cast<int>(tetras.vertex_metis_to_input.size()) < vert_count_before)
        {
            int gap_start = static_cast<int>(tetras.vertex_metis_to_input.size());
            for(int i = gap_start; i < vert_count_before; i++)
                tetras.vertex_metis_to_input.push_back(i);
        }

        size_t old_perm_size = tetras.vertex_metis_to_input.size();
        tetras.vertex_metis_to_input.resize(old_perm_size + n_added);
        if(!sort_index.empty() && static_cast<int>(sort_index.size()) == n_added)
        {
            // MAS active: engine vertex (body_offset + i) corresponds to
            // input vertex (body_offset + sort_index[i]).
            for(int i = 0; i < n_added; i++)
                tetras.vertex_metis_to_input[old_perm_size + i] =
                    vert_count_before + sort_index[i];
        }
        else
        {
            // No MAS or empty perm: identity within this body.
            for(int i = 0; i < n_added; i++)
                tetras.vertex_metis_to_input[old_perm_size + i] = vert_count_before + i;
        }
    }
}


void SimpleSceneImporter::import_scene(tetrahedra_obj& tetras)
{
    const auto& rigid_bodies = scene_json["rigid_body_problem"]["rigid_bodies"];
    for(auto& rigid_body : rigid_bodies)
    {
        const auto& mesh = rigid_body["mesh"];

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d R         = Eigen::Matrix3d::Identity();

        auto T = Eigen::Transform<double, 4, Eigen::Affine>::Identity();

        if(rigid_body.find("rotation") != rigid_body.end())
        {
            Eigen::Vector3d rotation = read_angular_field(rigid_body["rotation"]);
            R = (Eigen::AngleAxisd(rotation.z(), Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxisd(rotation.y(), Eigen::Vector3d::UnitY())
                 * Eigen::AngleAxisd(rotation.x(), Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
        }

        Eigen::Vector3d translation;
        from_json(rigid_body["position"], translation);
        translation -= Eigen::Vector3d(11., 0., 30.);
        transform.block<3, 3>(0, 0) = R;
        transform.block<3, 1>(0, 3) = translation;
        //transform *= Eigen::Vector4d{0.1, 0.1, 0.1, 1.0}.asDiagonal();

        auto is_fixed = rigid_body["is_dof_fixed"].get<bool>();

        auto mesh_file = mesh_dir + std::string{mesh};
        // std::cout << "mesh_file: " << mesh_file << std::endl;
        auto boundary_type = is_fixed ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;
        //tetras.load_tetrahedraMesh(mesh_file, transform, 1e8, body_type, boundary_type);

        load_geometry(tetras, 3, body_type, transform, 1e7, mesh_file, 1, boundary_type);
    }
}

}  // namespace gipc
