#include <gipc/utils/stl_mesh_loader.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <map>
#include <algorithm>

namespace gipc
{

bool load_stl_mesh(const std::string& filepath, ObjMeshData& out)
{
    if(!std::filesystem::exists(filepath))
    {
        std::cerr << "[StlLoader] File not found: " << filepath << std::endl;
        return false;
    }

    std::ifstream file(filepath, std::ios::binary);
    if(!file.is_open())
    {
        std::cerr << "[StlLoader] Cannot open file: " << filepath << std::endl;
        return false;
    }

    out.vertices.clear();
    out.faces.clear();
    out.convex_components.clear();

    // Read 80-byte header (ignore)
    char header[80];
    file.read(header, 80);
    if(!file.good())
    {
        std::cerr << "[StlLoader] Failed to read STL header: " << filepath << std::endl;
        return false;
    }

    // Read number of triangles (uint32)
    uint32_t num_triangles = 0;
    file.read(reinterpret_cast<char*>(&num_triangles), 4);
    if(!file.good())
    {
        std::cerr << "[StlLoader] Failed to read triangle count: " << filepath << std::endl;
        return false;
    }

    std::cout << "[StlLoader] Loading " << filepath << " (" << num_triangles << " triangles)" << std::endl;

    // Binary STL format per triangle: 12 bytes normal + 3*12 bytes vertices + 2 bytes attribute = 50 bytes
    // We'll merge duplicate vertices using a map for welding.
    // Key: quantized vertex position. Value: merged vertex index.
    struct Vec3Hash
    {
        size_t operator()(const Eigen::Vector3d& v) const
        {
            // Simple hash combining three doubles
            auto h1 = std::hash<double>{}(v.x());
            auto h2 = std::hash<double>{}(v.y());
            auto h3 = std::hash<double>{}(v.z());
            return h1 ^ (h2 * 2654435761ULL) ^ (h3 * 40343ULL);
        }
    };

    // Use a map with approximate comparison for vertex welding
    // For exact binary welding (same float bits), use unordered_map with exact hash
    std::map<std::tuple<float,float,float>, int> vertex_map;

    out.vertices.reserve(num_triangles);  // rough estimate
    out.faces.reserve(num_triangles);

    auto get_or_add_vertex = [&](float x, float y, float z) -> int
    {
        auto key = std::make_tuple(x, y, z);
        auto it  = vertex_map.find(key);
        if(it != vertex_map.end())
            return it->second;
        int idx = static_cast<int>(out.vertices.size());
        out.vertices.push_back(Eigen::Vector3d(static_cast<double>(x),
                                                static_cast<double>(y),
                                                static_cast<double>(z)));
        vertex_map[key] = idx;
        return idx;
    };

    for(uint32_t t = 0; t < num_triangles; t++)
    {
        // Normal (3 floats) — skip
        float normal[3];
        file.read(reinterpret_cast<char*>(normal), 12);

        // 3 vertices (each 3 floats)
        float v[3][3];
        file.read(reinterpret_cast<char*>(v), 36);

        // Attribute byte count (uint16) — skip
        uint16_t attr;
        file.read(reinterpret_cast<char*>(&attr), 2);

        if(!file.good())
        {
            std::cerr << "[StlLoader] Unexpected end of file at triangle " << t << std::endl;
            return false;
        }

        int i0 = get_or_add_vertex(v[0][0], v[0][1], v[0][2]);
        int i1 = get_or_add_vertex(v[1][0], v[1][1], v[1][2]);
        int i2 = get_or_add_vertex(v[2][0], v[2][1], v[2][2]);

        // Skip degenerate triangles
        if(i0 == i1 || i1 == i2 || i0 == i2)
            continue;

        out.faces.push_back(Eigen::Vector3i(i0, i1, i2));
    }

    std::cout << "[StlLoader] Loaded " << out.vertices.size() << " unique vertices, "
              << out.faces.size() << " triangles from " << filepath << std::endl;

    return !out.vertices.empty() && !out.faces.empty();
}

bool load_surface_mesh(const std::string& filepath, ObjMeshData& out)
{
    namespace fs = std::filesystem;
    std::string ext = fs::path(filepath).extension().string();
    // lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if(ext == ".obj")
    {
        return load_obj_mesh(filepath, out);
    }
    else if(ext == ".stl")
    {
        return load_stl_mesh(filepath, out);
    }
    else
    {
        std::cerr << "[load_surface_mesh] Unsupported format: " << ext
                  << " (" << filepath << ")" << std::endl;
        return false;
    }
}

}  // namespace gipc
