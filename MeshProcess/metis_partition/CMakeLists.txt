cmake_minimum_required(VERSION 3.18)
find_package(Eigen3 REQUIRED)
add_library(metis_partition STATIC)

file(GLOB_RECURSE metis_partition_SOURCE CONFIGURE_DEPENDS 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

target_sources(metis_partition PRIVATE ${metis_partition_SOURCE})
target_include_directories(metis_partition PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(metis_partition PRIVATE metis Eigen3::Eigen)
# ASSETS_DIR is only used by main.cpp (the standalone metis_partition CLI
# tool) and never enters the wheel binary — leave its compile-time bake-in.
target_compile_definitions(metis_partition PRIVATE ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../Assets/")
# OUTPUT_DIR was previously baked to a source-tree absolute path, which
# leaked into shipped wheel binaries and caused "cannot create directory"
# errors on user machines (Issue 1, see docs/internal/RELEASE_LOG.md).
# Now empty: callers MUST pass an explicit output folder via metis_sort()'s
# 3rd parameter; the empty macro is only a last-resort sentinel checked by
# metis_sort() itself.
target_compile_definitions(metis_partition PRIVATE OUTPUT_DIR="")
target_compile_features(metis_partition PRIVATE cxx_std_17)

source_group(TREE "${CMAKE_CURRENT_LIST_DIR}" FILES ${metis_partition_SOURCE})
set_target_properties(metis_partition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(metis_partition PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



