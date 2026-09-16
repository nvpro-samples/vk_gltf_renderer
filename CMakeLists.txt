#####################################################################################
# vk_gltf_renderer — Vulkan glTF ray tracing renderer (path tracer + rasterizer)
#####################################################################################

cmake_minimum_required(VERSION 3.22...3.31)

# In CI only: disable CMake's "re-run at build time" (must precede any project()).
if(DEFINED ENV{CI})
  set(CMAKE_SUPPRESS_REGENERATION ON)
endif()

# Use the source directory name as the project name so `vk_gltf_renderer.exe`
# tracks the folder even if this file is copy-pasted into a fork with a different
# directory. `get_filename_component` must run before `project()`.
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} VERSION 2.0 LANGUAGES C CXX)
message(STATUS "Processing: ${PROJECT_NAME}")

# Emit compile_commands.json for clangd / other tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Local CMake modules (draco.cmake, etc.).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


#####################################################################################
# ██████╗ ███████╗██╗   ██╗    ██╗  ██╗███╗   ██╗ ██████╗ ██████╗ ███████╗
# ██╔══██╗██╔════╝██║   ██║    ██║ ██╔╝████╗  ██║██╔═══██╗██╔══██╗██╔════╝
# ██║  ██║█████╗  ██║   ██║    █████╔╝ ██╔██╗ ██║██║   ██║██████╔╝███████╗
# ██║  ██║██╔══╝  ╚██╗ ██╔╝    ██╔═██╗ ██║╚██╗██║██║   ██║██╔══██╗╚════██║
# ██████╔╝███████╗ ╚████╔╝     ██║  ██╗██║ ╚████║╚██████╔╝██████╔╝███████║
# ╚═════╝ ╚══════╝  ╚═══╝      ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝
#
# Everything you're likely to toggle for local testing lives in this block.
# The rest of the file is plumbing and should rarely need editing.
#
# HOW TO CHANGE A VALUE
# ---------------------
#   1. Command line (one-off, preferred for quick tests):
#        cmake -B build -S . -DUSE_DLSS=OFF -DDLSS_VERSION="310.9.0"
#      -D always wins, regardless of what CMakeCache.txt already contains.
#
#   2. Edit this file (repo-wide default):
#        Change the default in the option()/set() line below. This only takes
#        effect on a FRESH configure (no CMakeCache.txt yet). To override an
#        already-populated cache without deleting the build directory, use
#        the `CACHE ... FORCE` form — several examples are shown commented-out
#        below and can be copy-pasted next to the value you want to pin.
#
#   3. cmake-gui / ccmake: every knob below is a cache variable and shows up
#      as an editable field.
#
# WHY `set(... CACHE ...)` INSTEAD OF `option()` FOR SOME KNOBS
# -------------------------------------------------------------
#   `option()` is only for booleans. Strings (versions), paths (SDK roots),
#   and integers (shader debug/opt levels) must use `set(... CACHE STRING/PATH ...)`.
#   Both forms produce cache variables you can override with -D or in the GUI.
#####################################################################################

# ---- Feature toggles ---------------------------------------------------------------
option(USE_DLSS               "Enable DLSS-RR denoiser (Windows/RTX; downloads NGX SDK)."             ON)
option(USE_DLSSNR             "Enable DLSS Neural Rendering (beta; requires DLSS beta SDK)."          ON)
option(USE_OPTIX_DENOISER     "Enable OptiX denoiser as alternative to DLSS (requires CUDAToolkit)."  ON)
option(USE_DRACO              "Enable Draco mesh compression when importing glTF."                    ON)
option(USE_AGENTIC            "Build the optional Agentic bridge (ComfyUI HDRI + beautify)."          ON)
option(USE_DEFAULT_SCENE      "Load a bundled default scene at startup."                              OFF)
option(BUILD_TESTING          "Build unit tests and benchmarks (GoogleTest + Google Benchmark)."      OFF)
option(REQUIRE_NVML_MONITOR   "Fail configure if NVML support is unavailable in nvpro_core2."         OFF)

# ---- nvpro_core2 module toggles ----------------------------------------------------
# These are read by nvpro_core2 during its own configure. We disable modules we
# either don't need (nvgl) or replace with a local copy (nvvkgltf, nvvkglsl).
# MUST be set before `find_package(NvproCore2)` below.
option(NVPRO2_ENABLE_nvgl     "Build the OpenGL helpers in nvpro_core2 (unused here)."           OFF)
option(NVPRO2_ENABLE_nvvkgltf "Build the nvvkgltf helper in nvpro_core2 (we ship a local copy)." OFF)
option(NVPRO2_ENABLE_nvvkglsl "Build the nvvkglsl helper in nvpro_core2 (Slang replaces GLSL)."  OFF)
# Optional MCP endpoint for agent-driven shader timing (--mcp, see src/mcp_timing.cpp). Enabling
# this makes nvpro_core2 fetch its pinned cpp-mcp dependency during configure.
option(NVPRO2_ENABLE_nvmcp    "Build the nvmcp MCP endpoint in nvpro_core2 (--mcp enables it)."  ON)
# Expose every registered command-line parameter through nvmcp's parameter tools, instead of
# curating an allowlist here. One place to add a parameter -- the registry -- and the agent sees it.
option(NVPRO2_ENABLE_MCP_AUTO_PARAMETER_REGISTRY
       "Expose all registered parameters over MCP (nvpro_core2 option; declared here to default it ON)." ON)

# ---- SDK versions and local overrides ----------------------------------------------
# These are downloaded automatically on first configure. Point *_ROOT_OVERRIDE
# at a local checkout to skip the download and pin a custom build.

# Slang shader compiler — usually the NvproCore2 default is fine. Uncomment to pin.
# set(Slang_VERSION      "2026.17" CACHE STRING "Slang compiler version to download.")

# DLSS SDK.
set(DLSS_VERSION       "310.7.0" CACHE STRING "DLSS SDK version to download (ignored when DLSS_ROOT_OVERRIDE is set).")
set(DLSS_ROOT_OVERRIDE ""        CACHE PATH   "Path to a local DLSS SDK; skips download when non-empty.")
# Example local overrides — uncomment and re-configure to pick up a private drop:
# set(DLSS_VERSION       "310.9.0" CACHE STRING "" FORCE)
# set(DLSS_ROOT_OVERRIDE "${CMAKE_BINARY_DIR}/_deps/NVIDIA-NGX-DLSS-SDK-INTERNAL-v310.9.0-beta3" CACHE PATH "" FORCE)

# DLSS "develop" DLLs include the debug overlay (Ctrl-Alt-F12 toggles it,
# Ctrl-Alt-F11 cycles views). NEVER ship these — local debugging only.
option(DLSS_USE_DEVELOP_LIBRARIES "Use DLSS dev DLLs with debug overlay instead of release DLLs." OFF)

# OptiX headers-only SDK (downloaded from github.com/NVIDIA/optix-dev).
set(OptiX_VERSION      "9.1.0"   CACHE STRING "OptiX headers version to download.")

# ---- Shader compilation knobs ------------------------------------------------------
# Slang debug/opt levels applied to every shader in this project.
#   OPT_LEVEL 0 is intentional: the driver optimizes SPIR-V when it JITs to native,
#   and pre-optimizing at Slang time can remove opportunities the driver would take.
set(SHADER_DEBUG_LEVEL 2 CACHE STRING "Slang debug info: 0=none, 1=minimal, 2=standard, 3=maximal.")
set(SHADER_OPT_LEVEL   0 CACHE STRING "Slang optimization: 0=none (recommended — let the driver optimize) .. 3=max.")

# ---- Optional paths ---------------------------------------------------------------
# glTF-Sample-Assets checkout (only consumed by the test suite). Falls back to
# the GLTF_SAMPLE_ASSETS_PATH environment variable when the CMake var is unset.
if(NOT DEFINED GLTF_SAMPLE_ASSETS_PATH)
  set(GLTF_SAMPLE_ASSETS_PATH "$ENV{GLTF_SAMPLE_ASSETS_PATH}" CACHE PATH "Path to a glTF-Sample-Assets checkout (optional; used by tests).")
endif()

#####################################################################################
# END OF DEVELOPER KNOBS — everything below is plumbing.
#####################################################################################


#####################################################################################
# nvpro_core2: locate (or auto-download) and enforce optional requirements.
#####################################################################################
find_package(NvproCore2 REQUIRED)

if(REQUIRE_NVML_MONITOR)
  get_target_property(_NVGPU_MONITOR_DEFS nvpro2::nvgpu_monitor INTERFACE_COMPILE_DEFINITIONS)
  if(NOT _NVGPU_MONITOR_DEFS)
    set(_NVGPU_MONITOR_DEFS "")
  endif()
  if(NOT "NVML_SUPPORTED" IN_LIST _NVGPU_MONITOR_DEFS)
    message(FATAL_ERROR "REQUIRE_NVML_MONITOR is ON, but nvpro2::nvgpu_monitor was configured without NVML support. Install the CUDA NVML development component.")
  endif()
endif()


#####################################################################################
# Version header — tag-driven, with a git-describe fallback for dev builds.
#   - CI passes -DRELEASE_VERSION=<tag> (e.g. 2026.5); that wins.
#   - Otherwise ask git for the closest tag + offset (e.g. v2026.5-3-g1a2b3c4-dirty).
#   - Otherwise (e.g. tarball checkout without .git) fall back to "dev-unknown".
#####################################################################################
if(DEFINED RELEASE_VERSION AND NOT "${RELEASE_VERSION}" STREQUAL "")
  set(APP_VERSION_STRING "v${RELEASE_VERSION}")
else()
  find_package(Git QUIET)
  set(APP_VERSION_STRING "dev-unknown")
  if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE _GIT_DESCRIBE
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _GIT_RESULT
    )
    if(_GIT_RESULT EQUAL 0 AND _GIT_DESCRIBE)
      set(APP_VERSION_STRING "${_GIT_DESCRIBE}")
    endif()
  endif()
endif()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/version.hpp.in"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/version.hpp"
  @ONLY
)
message(STATUS "Application version: ${APP_VERSION_STRING}")


#####################################################################################
# Testing framework (opt-in — see BUILD_TESTING knob at the top).
#####################################################################################
if(BUILD_TESTING)
  enable_testing()

  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )

  set(gtest_force_shared_crt        ON  CACHE BOOL "" FORCE)  # Windows: match /MD runtime
  set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)  # skip benchmark's own tests
  set(BENCHMARK_ENABLE_INSTALL      OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS  OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest googlebenchmark)

  # Group test dependencies in Visual Studio.
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER "External/GoogleTest")
  set_target_properties(benchmark benchmark_main         PROPERTIES FOLDER "External/Benchmark")

  add_subdirectory(tests)
  message(STATUS "Testing framework enabled (GoogleTest + Benchmark)")
endif()


#####################################################################################
# Collect application sources.
#   CONFIGURE_DEPENDS re-runs configure when files are added/removed under src/.
#####################################################################################
file(GLOB EXE_SOURCES CONFIGURE_DEPENDS
  "src/*.cpp"
  "src/*.hpp"
  "src/*.h"
  "*.md"
)

# Feature-gated source files: wired in explicitly further down when the
# corresponding backend is available, so exclude them from the generic glob.
list(FILTER EXE_SOURCES EXCLUDE REGEX "dlss*")
list(FILTER EXE_SOURCES EXCLUDE REGEX "vk_cuda*")
list(FILTER EXE_SOURCES EXCLUDE REGEX "optix_denoiser*")

# Optional Agentic bridge: exclude its sources (agentic*.cpp/.hpp, ui_agentic.cpp)
# when the feature is off. The renderer guards its integration points on USE_AGENTIC.
if(NOT USE_AGENTIC)
  list(FILTER EXE_SOURCES EXCLUDE REGEX "agentic")
endif()

# main.cpp guards its integration points on USE_NVMCP, so the renderer builds unchanged without it.
if(NOT NVPRO2_ENABLE_nvmcp)
  list(FILTER EXE_SOURCES EXCLUDE REGEX "src/mcp_")
endif()

add_executable(${PROJECT_NAME} ${EXE_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvslang
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvshaders_host
  nvpro2::nvaftermath
  nvpro2::nvgpu_monitor
  # Additional deps for the local gltf loader (previously provided by nvvkgltf).
  fmt
  glm
  nvimageformats
  stb
  tinygltf
  tinyobjloader
  meshoptimizer
)

# nvmcp is an opt-in nvpro_core2 module; fail loudly rather than silently dropping the tools.
if(NVPRO2_ENABLE_nvmcp)
  if(NOT TARGET nvpro2::nvmcp)
    message(FATAL_ERROR "NVPRO2_ENABLE_nvmcp is ON, but the nvpro2::nvmcp target is not available. "
                        "Update nvpro_core2, or configure with -DNVPRO2_ENABLE_nvmcp=OFF.")
  endif()
  target_link_libraries(${PROJECT_NAME} PRIVATE nvpro2::nvmcp)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NVMCP)
endif()

add_project_definitions(${PROJECT_NAME})

if(USE_DEFAULT_SCENE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DEFAULT_SCENE)
endif()

if(USE_AGENTIC)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AGENTIC)
endif()

# Generated headers (version.hpp, compiled Slang shaders) live under the build dir.
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})


#####################################################################################
# Slang shaders
#####################################################################################

# Collect local shaders (exclude header-only .h.slang files).
file(GLOB_RECURSE _ALL_SLANG_FILES "shaders/*.slang")
list(FILTER _ALL_SLANG_FILES EXCLUDE REGEX ".*\\.h\\.slang$")
set(SHADER_SRC_SLANG_FILES ${_ALL_SLANG_FILES})
file(GLOB SHADER_H_FILES "shaders/*.h" "shaders/*.h.slang")

# Shared shaders from nvshaders (tonemapper, sky, HDR processing).
set(SHADER_SHARED_SLANG_FILES
  ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
  ${NVSHADERS_DIR}/nvshaders/sky_physical.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_diffuse.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_glossy.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_dome.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_integrate_brdf.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_charlie_brdf_lut.slang
)

set(SHADER_SLANG_FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_SHARED_SLANG_FILES})
list(LENGTH SHADER_SLANG_FILES _SHADER_COUNT)
message(STATUS "Compiling ${_SHADER_COUNT} Slang shader(s) to SPIR-V")

# Required SPIR-V capabilities for ray tracing / wave ops.
set(SHADER_CAPABILITIES
  spvGroupNonUniformArithmetic   # arithmetic across subgroups
  spvGroupNonUniform             # subgroup ops (required for wave operations)
)

# Probe DLSS / OptiX backend availability here (before compile_slang) so the
# shader defines and the C++ compile flags stay in lockstep. The full link/source
# wiring for these backends lives further down.
if(USE_DLSS)
  find_package(NGX QUIET)
  # FindNGX.cmake caches NGX_INCLUDE_DIR without FORCE and guards
  # target_include_directories with `if(NOT TARGET ngx)`, so switching
  # DLSS_ROOT_OVERRIDE would leave the cached path stale. Force both the cache
  # variable and the target's interface includes to track the active override.
  if(DLSS_ROOT_OVERRIDE AND NGX_FOUND)
    set(NGX_INCLUDE_DIR "${DLSS_ROOT_OVERRIDE}/include" CACHE PATH "NGX SDK include directory." FORCE)
    set_target_properties(ngx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NGX_INCLUDE_DIR}")
  endif()
endif()

set(DLSS_AVAILABLE OFF)
if(USE_DLSS AND NGX_FOUND)
  set(DLSS_AVAILABLE ON)
elseif(USE_DLSS)
  message(WARNING "NGX SDK not found - DLSS disabled")
endif()

# DLSS-NR (Neural Rendering) requires DLSS + the beta SDK header. Probed here
# so the configure summary and the wiring block agree on a single truth, and
# so the "not available" message fires once instead of deep in the link block.
set(DLSSNR_AVAILABLE OFF)
if(USE_DLSSNR)
  if(DLSS_AVAILABLE AND EXISTS "${NGX_INCLUDE_DIR}/nvsdk_ngx_helpers_dlssnr_vk.h")
    set(DLSSNR_AVAILABLE ON)
  else()
    message(STATUS "DLSS-NR: beta header nvsdk_ngx_helpers_dlssnr_vk.h not found in '${NGX_INCLUDE_DIR}' - USE_DLSSNR disabled (requires beta SDK; point DLSS_ROOT_OVERRIDE at it)")
  endif()
endif()

if(USE_OPTIX_DENOISER)
  find_package(CUDAToolkit COMPONENTS CUDA::cudart)
endif()
set(OPTIX_DENOISER_AVAILABLE OFF)
if(USE_OPTIX_DENOISER AND CUDAToolkit_FOUND)
  set(OPTIX_DENOISER_AVAILABLE ON)
endif()

# Nsight Aftermath is optional and picked up by nvpro_core2 via the
# NSIGHT_AFTERMATH_SDK env variable. Its Find module runs in nvpro_core2's
# subdirectory scope, so NsightAftermath_FOUND isn't visible here. Read the
# authoritative signal off the nvpro2::nvaftermath target instead — same
# pattern used for the NVML check at the top.
set(AFTERMATH_AVAILABLE OFF)
if(TARGET nvpro2::nvaftermath)
  get_target_property(_AFTERMATH_DEFS nvpro2::nvaftermath INTERFACE_COMPILE_DEFINITIONS)
  if(_AFTERMATH_DEFS AND "AFTERMATH_AVAILABLE=1" IN_LIST _AFTERMATH_DEFS)
    set(AFTERMATH_AVAILABLE ON)
  endif()
endif()

# Shader gates:
#   HAS_DLSS_MOTION  : makes gltf_raster.slang emit SV_TARGET2 motion vectors for DLAA.
#   USE_GUIDE_SHADER : enables the path tracer's per-bounce guide buffers
#                      (albedo + normalRoughness) consumed by DLSS AND OptiX.
#   USE_DLSS_SHADER  : DLSS-only path (drops per-frame MS loop to relieve register
#                      pressure). Defaults off in the embedded shader; the runtime
#                      rebuild flips it on when DLSS actually engages.
# All three stay independently controllable.
set(SHADER_EXTRA_FLAGS "-I${NVSHADERS_DIR}")
if(DLSS_AVAILABLE)
  list(APPEND SHADER_EXTRA_FLAGS "-DHAS_DLSS_MOTION=1")
endif()
if(DLSS_AVAILABLE OR OPTIX_DENOISER_AVAILABLE)
  list(APPEND SHADER_EXTRA_FLAGS "-DUSE_GUIDE_SHADER=1")
endif()

# Compile all shaders to embedded headers (.h). Uncomment SPVS_VAR for .spv output.
compile_slang(
  "${SHADER_SLANG_FILES}"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/_autogen"
  HEADERS_VAR GENERATED_SHADER_HEADERS
  # SPVS_VAR GENERATED_SHADER_SPIRV  # For debugging purpose
  EXTRA_FLAGS ${SHADER_EXTRA_FLAGS}
  DEBUG_LEVEL ${SHADER_DEBUG_LEVEL}
  OPTIMIZATION_LEVEL ${SHADER_OPT_LEVEL}
  CAPABILITIES ${SHADER_CAPABILITIES}
)

# IDE grouping.
source_group("Shaders"          FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})
source_group("Shaders/Shared"   FILES ${SHADER_SHARED_SLANG_FILES})

target_sources(${PROJECT_NAME} PRIVATE
  ${GENERATED_SHADER_HEADERS}
  ${SHADER_SRC_SLANG_FILES}
  ${SHADER_H_FILES}
  ${SHADER_SHARED_SLANG_FILES}
)


#####################################################################################
# OptiX headers (always downloaded; needed even when the denoiser is disabled
# because some helper types live in the headers). The C++ link/source wiring
# below is gated on OPTIX_DENOISER_AVAILABLE.
#####################################################################################
download_package(
  NAME OptiX
  URLS https://github.com/NVIDIA/optix-dev/archive/refs/tags/v${OptiX_VERSION}.zip
  VERSION ${OptiX_VERSION}
  LOCATION OptiX_SOURCE_DIR
)
message(STATUS "OptiX Version     : ${OptiX_VERSION}")
message(STATUS "      Source Dir  : ${OptiX_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PRIVATE ${OptiX_SOURCE_DIR}/optix-dev-${OptiX_VERSION}/include)

if(USE_OPTIX_DENOISER)
  if(OPTIX_DENOISER_AVAILABLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPTIX_DENOISER)
    target_sources(${PROJECT_NAME} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_denoiser.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_denoiser.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_cuda.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/vk_cuda.hpp
    )
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found - OptiX denoiser enabled")

    # Windows: delay-load the CUDA runtime DLL so the app can start without it
    # installed. The DLL is only pulled in when the OptiX denoiser is used.
    if(WIN32)
      target_link_libraries(${PROJECT_NAME} PRIVATE delayimp.lib)
      target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
      message(STATUS "CUDA delay-load enabled for cudart64_${CUDAToolkit_VERSION_MAJOR}.dll")
    endif()
  else()
    message(WARNING "CUDAToolkit not found - OptiX denoiser disabled")
  endif()
endif()


#####################################################################################
# DLSS link/source wiring. Availability was already probed above so the Slang
# build and C++ build agree on the backend gate.
#####################################################################################
if(DLSS_AVAILABLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DLSS)
  if(DLSSNR_AVAILABLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DLSSNR)
  endif()
  target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss_wrapper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dlss.hpp
  )
endif()


#####################################################################################
# Third-party deps (Draco, RapidJSON, MikkTSpace, WebP).
#####################################################################################
if(USE_DRACO)
  include(cmake/draco.cmake)
  download_draco()
  add_draco()
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DRACO)
endif()

download_package(
  NAME RapidJSON
  URLS https://github.com/Tencent/rapidjson/archive/refs/heads/master.zip
  VERSION master
  LOCATION RapidJSON_SOURCE_DIR
)
set(RAPIDJSON_INCLUDE_DIR ${RapidJSON_SOURCE_DIR}/rapidjson-master/include/rapidjson)
target_include_directories(${PROJECT_NAME} PRIVATE ${RAPIDJSON_INCLUDE_DIR})

set(MIKKTSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MikkTSpace)
set(MIKKTSPACE_SRC ${MIKKTSPACE_DIR}/mikktspace.c ${MIKKTSPACE_DIR}/mikktspace.h)
target_include_directories(${PROJECT_NAME} PRIVATE ${MIKKTSPACE_DIR})
target_sources(${PROJECT_NAME} PRIVATE ${MIKKTSPACE_SRC})
source_group("MikkTSpace" FILES ${MIKKTSPACE_SRC})

find_package(WebP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE webp)


#####################################################################################
# Bundled runtime resources (HDRIs shipped alongside the exe).
#####################################################################################
# download_files(FILENAMES FlightHelmet.zip EXTRACT)
# download_files(FILENAMES Spaceships.zip EXTRACT)
download_files(FILENAMES daytime.hdr std_env.hdr)


#####################################################################################
# Install / runtime file layout.
#####################################################################################
copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${NsightAftermath_DLLS} ${Slang_GLSLANG} ${DLSS_DLLS}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders" "${CMAKE_CURRENT_LIST_DIR}/resources"
  AUTO
)


#####################################################################################
# Ship the optional Agentic bridge (Python adapter + helper + workflows) next to
# the executable. The renderer's Agentic Window resolves the adapter script at
# runtime via <exe_dir>/utils/comfy_bridge/comfy_bridge.py so the "Adapter not
# running" hint can show an exact, copy-pasteable command for the user.
#
# Build-time: POST_BUILD copy_if_different so dev runs from _bin/<config>/ work.
# Install-time: install(FILES ...) into <prefix>/utils/... so packaged builds match.
#####################################################################################
if(USE_AGENTIC)
  set(AGENTIC_UTILS_SRC "${CMAKE_CURRENT_LIST_DIR}/utils")
  file(GLOB AGENTIC_ADAPTER_FILES
    CONFIGURE_DEPENDS
    "${AGENTIC_UTILS_SRC}/comfy_bridge/*.py"
    "${AGENTIC_UTILS_SRC}/comfy_bridge/*.md"
    "${AGENTIC_UTILS_SRC}/comfy_bridge/*.json"  # prompts.json (Agentic prompt presets)
  )
  file(GLOB AGENTIC_WORKFLOW_FILES
    CONFIGURE_DEPENDS
    "${AGENTIC_UTILS_SRC}/comfy_bridge/workflows/*.json"
  )
  file(GLOB AGENTIC_ROOT_UTILS_FILES
    CONFIGURE_DEPENDS
    "${AGENTIC_UTILS_SRC}/png_to_hdr.py"
  )

  # Make the destination directories exist before per-file copy_if_different runs.
  add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>/utils/comfy_bridge/workflows"
    VERBATIM
  )
  foreach(_F IN LISTS AGENTIC_ADAPTER_FILES)
    add_custom_command(
      TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_F}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/utils/comfy_bridge/"
      VERBATIM
    )
  endforeach()
  foreach(_F IN LISTS AGENTIC_WORKFLOW_FILES)
    add_custom_command(
      TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_F}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/utils/comfy_bridge/workflows/"
      VERBATIM
    )
  endforeach()
  foreach(_F IN LISTS AGENTIC_ROOT_UTILS_FILES)
    add_custom_command(
      TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_F}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/utils/"
      VERBATIM
    )
  endforeach()

  install(FILES ${AGENTIC_ADAPTER_FILES}    DESTINATION "utils/comfy_bridge")
  install(FILES ${AGENTIC_WORKFLOW_FILES}   DESTINATION "utils/comfy_bridge/workflows")
  install(FILES ${AGENTIC_ROOT_UTILS_FILES} DESTINATION "utils")

  # Surface the file lists in the IDE so they're easy to find and edit.
  source_group("Agentic Utils\\Adapter"   FILES ${AGENTIC_ADAPTER_FILES})
  source_group("Agentic Utils\\Workflows" FILES ${AGENTIC_WORKFLOW_FILES})
  source_group("Agentic Utils\\Helpers"   FILES ${AGENTIC_ROOT_UTILS_FILES})
  target_sources(${PROJECT_NAME} PRIVATE
    ${AGENTIC_ADAPTER_FILES}
    ${AGENTIC_WORKFLOW_FILES}
    ${AGENTIC_ROOT_UTILS_FILES}
  )
endif() # USE_AGENTIC


#####################################################################################
# Windows resource file (application icon + version info).
#####################################################################################
if(WIN32)
  set(RESOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resource.rc")
  target_sources(${PROJECT_NAME} PRIVATE ${RESOURCE_FILE})
endif()


#####################################################################################
# IDE ergonomics: startup project + surface CMake files in the Solution.
#####################################################################################
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

file(GLOB CMAKE_FILES CMakeLists.txt cmake/*.cmake)
target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_FILES})
source_group("cmake" FILES ${CMAKE_FILES})


#####################################################################################
# CPack packaging (`cmake --build build --target package`).
#####################################################################################
set(CPACK_PACKAGE_NAME      "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION   "${APP_VERSION_STRING}")
set(CPACK_PACKAGE_VENDOR    "NVIDIA")
set(CPACK_GENERATOR         "ZIP")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/_package")
include(CPack)


#####################################################################################
# ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗    ███████╗██╗   ██╗███╗   ███╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝    ██╔════╝██║   ██║████╗ ████║
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗   ███████╗██║   ██║██╔████╔██║
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║   ╚════██║██║   ██║██║╚██╔╝██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝   ███████║╚██████╔╝██║ ╚═╝ ██║
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝    ╚══════╝ ╚═════╝ ╚═╝     ╚═╝
#
# Configuration summary printed at the end of configure. In cmake-gui / ccmake
# this is the last block scrolled onto the log, so you can see at a glance which
# SDK versions and knobs the build will actually use.
#
# HOW TO ADJUST
# -------------
#   - Add a row       : call `_summary_line("Label"  "${SOME_VAR}")`
#                       or `_summary_bool("Label"    ${SOME_BOOL})`
#   - Remove a row    : delete the call
#   - Add a section   : call `_summary_section("Title")`
#   - Change column   : bump _SUMMARY_LABEL_WIDTH below (default 22)
#   - Empty values render as "(not set)" so misconfigurations stand out.
#####################################################################################

# Re-import Vulkan into our scope: nvpro_core2 finds it, but Vulkan_VERSION is
# a regular (non-cache) variable, so it lives in nvpro_core2's directory scope.
# find_package is idempotent — this just makes the value visible here.
find_package(Vulkan QUIET)

set(_SUMMARY_LABEL_WIDTH 22)

# Print one "  <label> <padding> : <value>" line.
function(_summary_line label value)
  string(LENGTH "${label}" _len)
  math(EXPR _pad_len "${_SUMMARY_LABEL_WIDTH} - ${_len}")
  if(_pad_len LESS 1)
    set(_pad_len 1)
  endif()
  string(REPEAT " " ${_pad_len} _spaces)
  if("${value}" STREQUAL "")
    set(value "(not set)")
  endif()
  message(STATUS "  ${label}${_spaces}: ${value}")
endfunction()

# Print a bool as ON / OFF. Pass the value, not the variable name.
function(_summary_bool label value)
  if(value)
    _summary_line("${label}" "ON")
  else()
    _summary_line("${label}" "OFF")
  endif()
endfunction()

# Print a feature toggle plus its availability on one line:
#   OFF                        -> "OFF"
#   ON, backend detected       -> "ON   (found)"
#   ON, backend missing        -> "ON   (not available)"
function(_summary_feature label enabled available)
  if(NOT enabled)
    _summary_line("${label}" "OFF")
  elseif(available)
    _summary_line("${label}" "ON   (found)")
  else()
    _summary_line("${label}" "ON   (not available)")
  endif()
endfunction()

# Section divider.
function(_summary_section title)
  message(STATUS "")
  message(STATUS "  [ ${title} ]")
endfunction()

message(STATUS "")
message(STATUS "================================================================")
message(STATUS "  ${PROJECT_NAME} - configuration summary  (${APP_VERSION_STRING})")
message(STATUS "================================================================")

_summary_section("SDK versions")
_summary_line("Vulkan SDK"           "${Vulkan_VERSION}")
_summary_line("Slang compiler"       "${Slang_VERSION}")
if(DLSS_AVAILABLE AND DLSS_ROOT_OVERRIDE)
  _summary_line("DLSS SDK"           "override: ${DLSS_ROOT_OVERRIDE}")
elseif(DLSS_AVAILABLE)
  _summary_line("DLSS SDK"           "${DLSS_VERSION}")
else()
  _summary_line("DLSS SDK"           "(disabled)")
endif()
_summary_line("OptiX headers"        "${OptiX_VERSION}")
if(OPTIX_DENOISER_AVAILABLE)
  _summary_line("CUDA Toolkit"       "${CUDAToolkit_VERSION}")
else()
  _summary_line("CUDA Toolkit"       "(disabled)")
endif()
_summary_line("WebP"                 "${WebP_VERSION}")
_summary_line("Draco"                "main (tracking branch)")
_summary_line("RapidJSON"            "master (tracking branch)")
if(AFTERMATH_AVAILABLE)
  _summary_line("Nsight Aftermath"   "found (NSIGHT_AFTERMATH_SDK=$ENV{NSIGHT_AFTERMATH_SDK})")
else()
  _summary_line("Nsight Aftermath"   "not found (set NSIGHT_AFTERMATH_SDK env var)")
endif()

_summary_section("Feature toggles")
_summary_feature("USE_DLSS"           ${USE_DLSS}           ${DLSS_AVAILABLE})
_summary_feature("USE_DLSSNR"         ${USE_DLSSNR}         ${DLSSNR_AVAILABLE})
_summary_feature("USE_OPTIX_DENOISER" ${USE_OPTIX_DENOISER} ${OPTIX_DENOISER_AVAILABLE})
_summary_bool("USE_DRACO"            ${USE_DRACO})
_summary_bool("USE_AGENTIC"          ${USE_AGENTIC})
_summary_bool("USE_DEFAULT_SCENE"    ${USE_DEFAULT_SCENE})
_summary_bool("BUILD_TESTING"        ${BUILD_TESTING})

_summary_section("Shader compilation")
_summary_line("Debug level"          "${SHADER_DEBUG_LEVEL}")
_summary_line("Optimization level"   "${SHADER_OPT_LEVEL}")
_summary_line("Shader count"         "${_SHADER_COUNT}")

_summary_section("Local overrides / paths")
_summary_line("DLSS_ROOT_OVERRIDE"   "${DLSS_ROOT_OVERRIDE}")
_summary_line("Sample assets path"   "${GLTF_SAMPLE_ASSETS_PATH}")
_summary_line("Build type"           "${CMAKE_BUILD_TYPE}")
_summary_line("Install prefix"       "${CMAKE_INSTALL_PREFIX}")

message(STATUS "================================================================")
message(STATUS "")
