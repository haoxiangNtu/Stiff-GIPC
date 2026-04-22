#include <string>
#include <vector>

// metis_sort: partition + sort an .obj/.msh mesh, write the sorted mesh +
// partition to disk, return the output paths.
//
// output_folder: directory to write intermediate sorted_*/part_* files into.
// If empty, falls back (in order) to the OUTPUT_DIR compile-time macro
// (now empty by default — see MeshProcess/metis_partition/CMakeLists.txt),
// then to "./sorted_mesh/" relative to the process's current working
// directory. Callers should always pass an explicit, runtime-derived
// folder; defaulting was kept only for backwards-compat with the GLUT
// viewer (gl_main.cu) which has its own metis_dir global.
std::vector<std::string> metis_sort(std::string obj_path,
                                    int         dimension,
                                    std::string output_folder = "");
