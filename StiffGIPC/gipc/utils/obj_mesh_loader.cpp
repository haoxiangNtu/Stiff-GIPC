#include <gipc/utils/obj_mesh_loader.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <map>

namespace gipc
{

bool load_obj_mesh(const std::string& filepath, ObjMeshData& out)
{
    if(!std::filesystem::exists(filepath))
    {
        std::cerr << "[ObjLoader] File not found: " << filepath << std::endl;
        return false;
    }

    std::ifstream file(filepath);
    if(!file.is_open())
    {
        std::cerr << "[ObjLoader] Cannot open file: " << filepath << std::endl;
        return false;
    }

    out.vertices.clear();
    out.faces.clear();
    out.convex_components.clear();

    // Track per-component data.
    // OBJ "o" lines define object groups. Vertex indices are GLOBAL in the file.
    // We track which vertices/faces belong to each component.
    int current_component = -1;  // -1 means no component header seen yet

    // Global vertex list (all 'v' lines)
    std::vector<Eigen::Vector3d>& all_verts = out.vertices;

    // Per-component: which global faces belong to it
    struct ComponentFaces
    {
        std::string                  name;
        std::vector<Eigen::Vector3i> faces;  // global indices
    };
    std::vector<ComponentFaces> comp_faces;

    std::string line;
    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string        prefix;
        iss >> prefix;

        if(prefix == "o" || prefix == "g")
        {
            // New object/group
            std::string name;
            iss >> name;
            comp_faces.push_back({name, {}});
            current_component = static_cast<int>(comp_faces.size()) - 1;
        }
        else if(prefix == "v" && line.size() > 1 && line[1] == ' ')
        {
            // Vertex: "v x y z"
            double x, y, z;
            iss >> x >> y >> z;
            all_verts.push_back(Eigen::Vector3d{x, y, z});
        }
        else if(prefix == "f")
        {
            // Face: "f i j k" or "f i/t j/t k/t" or "f i/t/n j/t/n k/t/n"
            // Only handle triangles (3 vertices per face)
            std::vector<int> face_indices;
            std::string      token;
            while(iss >> token)
            {
                // Parse "idx" or "idx/tex" or "idx/tex/norm" — only need idx
                int idx = 0;
                auto slash_pos = token.find('/');
                if(slash_pos != std::string::npos)
                    idx = std::stoi(token.substr(0, slash_pos));
                else
                    idx = std::stoi(token);

                // OBJ is 1-indexed, convert to 0-indexed
                face_indices.push_back(idx - 1);
            }

            if(face_indices.size() >= 3)
            {
                // Triangulate: fan from first vertex for polygons with > 3 verts
                for(size_t i = 1; i + 1 < face_indices.size(); i++)
                {
                    Eigen::Vector3i tri{face_indices[0],
                                        face_indices[i],
                                        face_indices[i + 1]};
                    out.faces.push_back(tri);

                    if(current_component >= 0)
                        comp_faces[current_component].faces.push_back(tri);
                }
            }
        }
        // Skip mtllib, usemtl, vn, vt, s, etc.
    }

    file.close();

    // Build per-component vertex subsets
    // Each convex component may share global vertices, but for fan-tet generation
    // we need to know which vertices belong to each component.
    if(comp_faces.empty())
    {
        // No 'o' groups — treat entire mesh as one component
        ObjConvexComponent comp;
        comp.vertices = all_verts;
        comp.faces    = out.faces;
        out.convex_components.push_back(std::move(comp));
    }
    else
    {
        for(auto& cf : comp_faces)
        {
            if(cf.faces.empty())
                continue;

            // Find unique vertex indices used by this component
            std::map<int, int> global_to_local;
            ObjConvexComponent comp;

            for(auto& f : cf.faces)
            {
                Eigen::Vector3i local_face;
                for(int j = 0; j < 3; j++)
                {
                    int global_idx = f[j];
                    auto it = global_to_local.find(global_idx);
                    if(it == global_to_local.end())
                    {
                        int local_idx = static_cast<int>(comp.vertices.size());
                        global_to_local[global_idx] = local_idx;
                        comp.vertices.push_back(all_verts[global_idx]);
                        local_face[j] = local_idx;
                    }
                    else
                    {
                        local_face[j] = it->second;
                    }
                }
                comp.faces.push_back(local_face);
            }

            out.convex_components.push_back(std::move(comp));
        }
    }

    std::cout << "[ObjLoader] Loaded '" << filepath << "': "
              << all_verts.size() << " vertices, "
              << out.faces.size() << " faces, "
              << out.convex_components.size() << " components" << std::endl;

    return !all_verts.empty() && !out.faces.empty();
}

}  // namespace gipc
