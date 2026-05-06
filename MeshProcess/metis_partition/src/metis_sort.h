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
// out_sort_index: optional. When non-null, populated with the
// metis-sorted-index → input-vertex-index permutation:
// out_sort_index[i] == j means the i-th vertex of the sorted output
// is the j-th vertex of the original input mesh. Empty when MAS would
// have used a cached sorted file (caller can opt-in to recompute by
// passing the flag, or read the sort_index from disk via the
// "<mesh>_sorted.16.idx" sidecar file if present).
std::vector<std::string> metis_sort(std::string obj_path,
                                    int         dimension,
                                    std::string output_folder = "",
                                    std::vector<int>* out_sort_index = nullptr);
