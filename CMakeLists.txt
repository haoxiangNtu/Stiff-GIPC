cmake_minimum_required(VERSION 3.18)    
# support relative path, cuda_std_17, archive manipulation

# project meta
# ============

set(CMAKE_BUILD_TYPE "Release")
if(UNIX)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
    endif()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;120")
endif()

project(StiffGIPC LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(STIFFGIPC_ENABLE_DIAGNOSTICS
       "Build intrusive finite-difference diagnostics and Python debug hooks"
       OFF)
set(STIFFGIPC_FEM_MODEL "SNK1" CACHE STRING
    "Compile-time tetrahedral constitutive model: SNK1, SNK2, or ARAP")
set_property(CACHE STIFFGIPC_FEM_MODEL PROPERTY STRINGS SNK1 SNK2 ARAP)
string(TOUPPER "${STIFFGIPC_FEM_MODEL}" STIFFGIPC_FEM_MODEL_NORMALIZED)
if(NOT STIFFGIPC_FEM_MODEL_NORMALIZED MATCHES "^(SNK1|SNK2|ARAP)$")
    message(FATAL_ERROR
        "STIFFGIPC_FEM_MODEL must be SNK1, SNK2, or ARAP; got "
        "'${STIFFGIPC_FEM_MODEL}'")
endif()

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find packages
enable_language(CUDA)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(urdfdom CONFIG QUIET)
# Debian's liburdfdom-dev (<=3.0.x) ships a config that sets urdfdom_FOUND
# without defining the namespaced targets — FOUND alone is not enough (broke
# the A800 wheel build). Require the target itself, else pkg-config fallback.
if(urdfdom_FOUND AND TARGET urdfdom::urdf_parser)
    set(URDFDOM_LIBS urdfdom::urdf_parser urdfdom::urdfdom_model urdfdom::urdfdom_world)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(URDFDOM REQUIRED urdfdom)
    set(URDFDOM_LIBS ${URDFDOM_LIBRARIES})
    link_directories(${URDFDOM_LIBRARY_DIRS})
endif()

# add external libraries

# Common compile definitions for physics engine
set(GIPC_PHYSICS_DEFS
    ADAPTIVE_KAPPA 
    USE_FRICTION 
    USE_QUADRATIC_BENDING
    GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"
    GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/"
)
if(STIFFGIPC_FEM_MODEL_NORMALIZED STREQUAL "SNK1")
    list(APPEND GIPC_PHYSICS_DEFS USE_SNK1)
elseif(STIFFGIPC_FEM_MODEL_NORMALIZED STREQUAL "SNK2")
    list(APPEND GIPC_PHYSICS_DEFS USE_SNK2)
else()
    list(APPEND GIPC_PHYSICS_DEFS USE_ARAP)
endif()
list(APPEND GIPC_PHYSICS_DEFS
    GIPC_FEM_MODEL_NAME="${STIFFGIPC_FEM_MODEL_NORMALIZED}")
message(STATUS
    "StiffGIPC tetrahedral constitutive model: "
    "${STIFFGIPC_FEM_MODEL_NORMALIZED}")

# Common CUDA compile options
set(GIPC_CUDA_PRIVATE_OPTS
    ${CMAKE_CUDA_FLAGS} 
    --use_fast_math 
    -lineinfo 
    --ptxas-options=-allow-expensive-optimizations=true 
    -Xcudafe 
    "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
)
set(GIPC_CUDA_PUBLIC_OPTS
    --extended-lambda 
    --expt-relaxed-constexpr 
    --default-stream=per-thread
)

# Common CUDA target properties
macro(set_gipc_cuda_props target)
    set_target_properties(${target} PROPERTIES
        CUDA_EXTENSIONS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(${target}
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${GIPC_CUDA_PRIVATE_OPTS}>
        PUBLIC  $<$<COMPILE_LANGUAGE:CUDA>: ${GIPC_CUDA_PUBLIC_OPTS}>
    )
endmacro()

# ===========================================================================
# Library target: stiffgipc_core (no GL / GLUT / ImGui dependency)
# ===========================================================================
file(GLOB_RECURSE GIPC_ALL_SOURCE CONFIGURE_DEPENDS
    "StiffGIPC/*.cu" "StiffGIPC/*.cpp")
file(GLOB_RECURSE GIPC_ALL_HEADERS CONFIGURE_DEPENDS
    "StiffGIPC/*.h" "StiffGIPC/*.cuh" "StiffGIPC/*.hpp" "StiffGIPC/*.inl" "StiffGIPC/*.inc")

# Exclude gl_main.cu from the library (it contains main() and GL code)
list(FILTER GIPC_ALL_SOURCE EXCLUDE REGEX "gl_main\\.cu$")
if(STIFFGIPC_ENABLE_DIAGNOSTICS)
    list(APPEND GIPC_PHYSICS_DEFS GIPC_ENABLE_DIAGNOSTICS)
else()
    list(FILTER GIPC_ALL_SOURCE EXCLUDE REGEX "diagnostics/fd_check\\.cu$")
endif()

add_library(stiffgipc_core SHARED ${GIPC_ALL_SOURCE} ${GIPC_ALL_HEADERS})
target_compile_features(stiffgipc_core PUBLIC cuda_std_17 cxx_std_17)
target_include_directories(stiffgipc_core PUBLIC
    "StiffGIPC/"
    "StiffGIPC/muda/src/"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(stiffgipc_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()
target_compile_definitions(stiffgipc_core PUBLIC ${GIPC_PHYSICS_DEFS})
target_link_libraries(stiffgipc_core PUBLIC
    cusparse cublas cusolver cudart_static CUDA::cudart
    metis_partition Eigen3::Eigen nlohmann_json::nlohmann_json
)
target_include_directories(stiffgipc_core PUBLIC ${URDFDOM_INCLUDE_DIRS})
target_link_directories(stiffgipc_core PUBLIC ${URDFDOM_LIBRARY_DIRS})
target_link_libraries(stiffgipc_core PUBLIC ${URDFDOM_LIBS})
set_gipc_cuda_props(stiffgipc_core)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_ALL_SOURCE} ${GIPC_ALL_HEADERS})

# ===========================================================================
# Executable target: gipc (GLUT viewer, links stiffgipc_core + GL + ImGui)
# ===========================================================================
option(BUILD_GL_VIEWER "Build the OpenGL viewer executable" ON)
if(BUILD_GL_VIEWER)
    find_package(GLEW REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(imgui CONFIG REQUIRED)

    add_executable(gipc "StiffGIPC/gl_main.cu")
    target_link_libraries(gipc PRIVATE
        stiffgipc_core
        GLUT::GLUT GLEW::GLEW OpenGL::GLU
        imgui::imgui
    )
    set_gipc_cuda_props(gipc)
endif()

# ===========================================================================
# Python bindings: pystiffgipc (optional, requires pybind11)
# ===========================================================================
option(BUILD_PYTHON_BINDINGS "Build pystiffgipc Python module" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(pystiffgipc "bindings/pystiffgipc.cu")
        target_link_libraries(pystiffgipc PRIVATE stiffgipc_core)
        set_gipc_cuda_props(pystiffgipc)
        message(STATUS "pystiffgipc Python module will be built")
    else()
        message(STATUS "pybind11 not found; pystiffgipc will NOT be built")
    endif()
endif()

add_subdirectory("MeshProcess")

# ===========================================================================
# Install rules (used by scikit-build-core to populate the wheel)
# ===========================================================================
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    # Set RPATH so pystiffgipc.so finds libstiffgipc_core.so in the same dir
    set_target_properties(stiffgipc_core PROPERTIES
        INSTALL_RPATH "$ORIGIN")
    set_target_properties(pystiffgipc PROPERTIES
        INSTALL_RPATH "$ORIGIN")

    install(TARGETS pystiffgipc
            LIBRARY DESTINATION stiff_physics/_native)
    install(TARGETS stiffgipc_core
            LIBRARY DESTINATION stiff_physics/_native)
    install(FILES Assets/scene/abd_system_config.json
            DESTINATION stiff_physics/data/scene)
endif()
