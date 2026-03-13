cmake_minimum_required(VERSION 3.22...3.31)
project(GLTF_renderer VERSION 2.0 LANGUAGES C CXX)

# Enable export of compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NVPRO2_ENABLE_nvgl off)
option(USE_DEFAULT_SCENE "Using a default scene at startup" ON)

# Path to glTF-Sample-Assets repository (for testing)
# Optional. Set via -DGLTF_SAMPLE_ASSETS_PATH=..., environment variable GLTF_SAMPLE_ASSETS_PATH, or CMake GUI.
if(NOT DEFINED GLTF_SAMPLE_ASSETS_PATH)
  set(GLTF_SAMPLE_ASSETS_PATH "$ENV{GLTF_SAMPLE_ASSETS_PATH}" CACHE PATH "Path to glTF-Sample-Assets (optional)")
endif()

# Disable unused libraries from nvpro_core2 - we use local copy
set(NVPRO2_ENABLE_nvvkgltf OFF CACHE BOOL "Disable nvvkgltf from nvpro_core2" FORCE)
set(NVPRO2_ENABLE_nvvkglsl OFF CACHE BOOL "Disable nvvkglsl from nvpro_core2" FORCE)

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} VERSION 2.0 LANGUAGES C CXX)
message(STATUS "Processing: ${PROJECT_NAME}")

# Generate version header with build date
string(TIMESTAMP BUILD_DATE "%y%m%d")
set(APP_VERSION_STRING "v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${BUILD_DATE}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/version.hpp.in"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/version.hpp"
  @ONLY
)
message(STATUS "Application version: ${APP_VERSION_STRING}")

#####################################################################################
# Testing framework (opt-in for cleaner builds by default)
option(BUILD_TESTING "Build test suite" OFF)

if(BUILD_TESTING)
  enable_testing()
  
  # Google Test
  include(FetchContent)
  
  # Fetch GoogleTest
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  
  # Fetch Google Benchmark
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  
  # Make available
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # Windows
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)  # Don't build benchmark tests
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)  # Don't install benchmark
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)  # Don't build benchmark's own tests
  
  FetchContent_MakeAvailable(googletest googlebenchmark)
  
  # Organize test dependencies into folders in Visual Studio
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER "External/GoogleTest")
  set_target_properties(benchmark benchmark_main PROPERTIES FOLDER "External/Benchmark")
  
  # Add tests subdirectory
  add_subdirectory(tests)
  
  message(STATUS "Testing framework enabled (GoogleTest + Benchmark)")
endif()

# Adding all sources
file(GLOB EXE_SOURCES "src/*.cpp" "src/*.hpp" "src/*.h" "*.md")
list(FILTER EXE_SOURCES EXCLUDE REGEX "dlss*")
list(FILTER EXE_SOURCES EXCLUDE REGEX "vk_cuda*")
list(FILTER EXE_SOURCES EXCLUDE REGEX "optix_denoiser*")

# source_group("Source Files" FILES ${EXE_SOURCES})

# Add the executable
add_executable(${PROJECT_NAME} ${EXE_SOURCES})

# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvslang
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvshaders_host
  nvpro2::nvaftermath
  nvpro2::nvgpu_monitor
  # Additional dependencies for local gltf (previously provided by nvvkgltf)
  fmt
  glm
  nvimageformats
  stb
  tinygltf
  tinyobjloader
  meshoptimizer
)

add_project_definitions(${PROJECT_NAME})

# Add compile definition for USE_DEFAULT_SCENE option
if(USE_DEFAULT_SCENE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DEFAULT_SCENE)
endif()

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})

#------------------------------------------------------------------------------------------------------------------------------
# Compile Slang shaders
#------------------------------------------------------------------------------------------------------------------------------

# Collect all local shader files (exclude header-only .h.slang files)
# Includes e.g. gizmo_grid.slang, gizmo_visuals.slang
file(GLOB_RECURSE _ALL_SLANG_FILES "shaders/*.slang")
list(FILTER _ALL_SLANG_FILES EXCLUDE REGEX ".*\\.h\\.slang$")
set(SHADER_SRC_SLANG_FILES ${_ALL_SLANG_FILES})
file(GLOB SHADER_H_FILES "shaders/*.h" "shaders/*.h.slang")

# Include shared shaders from nvshaders (tonemapper, sky, HDR processing)
set(SHADER_SHARED_SLANG_FILES
  ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
  ${NVSHADERS_DIR}/nvshaders/sky_physical.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_diffuse.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_glossy.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_dome.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_integrate_brdf.slang
)

# Combine all shader files for compilation
set(SHADER_SLANG_FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_SHARED_SLANG_FILES})
list(LENGTH SHADER_SLANG_FILES _SHADER_COUNT)
message(STATUS "Compiling ${_SHADER_COUNT} Slang shader(s) to SPIR-V")

# Define required SPIR-V capabilities for ray tracing and advanced features
set(SHADER_CAPABILITIES 
  spvInt64Atomics                     # 64-bit atomic operations
  spvShaderClockKHR                   # Shader clock for profiling
  spvRayTracingMotionBlurNV           # Motion blur for ray tracing
  spvRayQueryKHR                      # Ray query operations
  spvGroupNonUniformBallot            # Ballot operations for subgroup functionality
  spvGroupNonUniformArithmetic        # Arithmetic operations across subgroups
)

# Shader compile options: -O0 for both (driver optimizes SPIR-V when compiling to native; pre-optimization can remove driver opportunities)
set(SHADER_DEBUG_LEVEL 2)   # None:0, minimal:1, standard:2, maximal:3 (see Slang documentation for details)
set(SHADER_OPT_LEVEL 0)     # None:0 — let the driver optimize

# Compile all shaders to embedded headers (.h) and SPIR-V binaries (.spv) [optionally generate SPIR-V binaries (.spv) for debugging]
compile_slang(
  "${SHADER_SLANG_FILES}"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/_autogen"
  HEADERS_VAR GENERATED_SHADER_HEADERS
  # SPVS_VAR GENERATED_SHADER_SPIRV  # For debugging purpose
  CAPABILITIES ${SHADER_CAPABILITIES}  # List expands and is automatically joined with `+`
  EXTRA_FLAGS "-I${NVSHADERS_DIR}"
  DEBUG_LEVEL ${SHADER_DEBUG_LEVEL}
  OPTIMIZATION_LEVEL ${SHADER_OPT_LEVEL}
)

# Add shader files to the project for IDE organization
source_group("Shaders" FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})
source_group("Shaders/Shared" FILES ${SHADER_SHARED_SLANG_FILES})

# Add shader files to target for Visual Studio integration
target_sources(${PROJECT_NAME} PRIVATE 
  ${GENERATED_SHADER_HEADERS}
  ${SHADER_SRC_SLANG_FILES}
  ${SHADER_H_FILES}
  ${SHADER_SHARED_SLANG_FILES}
)


#####################################################################################
# Add OptiX header
set(OptiX_VERSION 9.0.0)
download_package(
  NAME OptiX
  URLS https://github.com/NVIDIA/optix-dev/archive/refs/tags/v${OptiX_VERSION}.zip
  VERSION ${OptiX_VERSION}
  LOCATION OptiX_SOURCE_DIR
)
message(STATUS "OptiX Version     : ${OptiX_VERSION}")
message(STATUS "      Source Dir  : ${OptiX_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PRIVATE ${OptiX_SOURCE_DIR}/optix-dev-${OptiX_VERSION}/include)

# Find CUDA for OptiX interop
find_package(CUDAToolkit COMPONENTS CUDA::cudart)

# Add OptiX denoiser option
option(USE_OPTIX_DENOISER "Use OptiX denoiser as alternative to DLSS" ON)
if(USE_OPTIX_DENOISER)
  if(CUDAToolkit_FOUND)

    # Add OptiX denoiser files
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPTIX_DENOISER)
    target_sources(${PROJECT_NAME} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_denoiser.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_denoiser.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_cuda.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_cuda.hpp
    )

    # Add CUDA include directories and link libraries
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found - OptiX denoiser enabled")

    # Windows: Delay-load CUDA runtime DLL so the application can start without it
    # The DLL will only be loaded when OptiX denoiser is actually used
    if(WIN32)
      target_link_libraries(${PROJECT_NAME} PRIVATE delayimp.lib)
      target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
      message(STATUS "CUDA delay-load enabled for cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
    endif()

  else()
    message(WARNING "CUDAToolkit not found - OptiX denoiser will not work properly")
  endif()

endif()

#####################################################################################
# Adding DRACO if selected
option(USE_DRACO "Use Draco for compression" ON)
if(USE_DRACO)
  include(cmake/draco.cmake)
  download_draco()
  add_draco()
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DRACO)
endif()
#####################################################################################
# Add RapidJSON package
download_package(
  NAME RapidJSON
  URLS https://github.com/Tencent/rapidjson/archive/refs/heads/master.zip
  VERSION master
  LOCATION RapidJSON_SOURCE_DIR
)
set(RAPIDJSON_INCLUDE_DIR ${RapidJSON_SOURCE_DIR}/rapidjson-master/include/rapidjson)
target_include_directories(${PROJECT_NAME} PRIVATE ${RAPIDJSON_INCLUDE_DIR})
#####################################################################################
# Add MikkTSpace
set(MIKKTSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MikkTSpace)
set(MIKKTSPACE_SRC ${MIKKTSPACE_DIR}/mikktspace.c ${MIKKTSPACE_DIR}/mikktspace.h)
target_include_directories(${PROJECT_NAME} PRIVATE ${MIKKTSPACE_DIR})
target_sources(${PROJECT_NAME} PRIVATE ${MIKKTSPACE_SRC})
source_group("MikkTSpace" FILES ${MIKKTSPACE_SRC})
#####################################################################################
# Add WebP
find_package(WebP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE webp)
# target_link_libraries(${PROJECT_NAME} PRIVATE webp)
#####################################################################################
# Adding DLSS if selected
option(USE_DLSS "Use DLSS-RR for denoising" ON)
if(USE_DLSS)
  set(DLSS_VERSION "310.5.3")
  find_package(NGX REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DLSS)
  target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_denoiser.hpp
  )
endif()

#####################################################################################
# Adding download resources
# download_files(FILENAMES FlightHelmet.zip EXTRACT)
download_files(FILENAMES daytime.hdr std_env.hdr)

#####################################################################################
# Installation, copy files for runtime
copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${NsightAftermath_DLLS} ${Slang_GLSLANG} ${DLSS_DLLS}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders" "${CMAKE_CURRENT_LIST_DIR}/resources"
  AUTO
)

# Add Windows resource file
if(WIN32)
 set(RESOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resource.rc")
  target_sources(${PROJECT_NAME} PRIVATE ${RESOURCE_FILE})
endif()

# Make Visual Studio use this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Custom target to see files in Solution
file(GLOB CMAKE_FILES CMakeLists.txt cmake/*.cmake)
target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_FILES})
source_group("cmake" FILES ${CMAKE_FILES})

#####################################################################################
# CPack packaging
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${APP_VERSION_STRING}")
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/_package")
include(CPack)

