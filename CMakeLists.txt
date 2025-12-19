cmake_minimum_required(VERSION 3.22...3.31)
project(GLTF_renderer VERSION 2.0 LANGUAGES C CXX)

option(NVPRO2_ENABLE_nvgl off)
option(USE_DEFAULT_SCENE "Using a default scene at startup" ON)

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
  nvpro2::nvvkgltf
  nvpro2::nvgpu_monitor
)

add_project_definitions(${PROJECT_NAME})

# Add compile definition for USE_DEFAULT_SCENE option
if(USE_DEFAULT_SCENE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DEFAULT_SCENE)
endif()

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})

#------------------------------------------------------------------------------------------------------------------------------
# Compile shaders
file(GLOB_RECURSE _ALL_SLANG_FILES "shaders/*.slang")
list(FILTER _ALL_SLANG_FILES EXCLUDE REGEX ".*\\.h\\.slang$")
set(SHADER_SRC_SLANG_FILES ${_ALL_SLANG_FILES})
file(GLOB SHADER_H_FILES "shaders/*.h" "shaders/*.h.slang")

# Adding local pre-compiled version of the tonemapper
set(SHADER_SHARED_SLANG_FILES
  ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
  ${NVSHADERS_DIR}/nvshaders/sky_physical.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_diffuse.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_prefilter_glossy.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_dome.slang
  ${NVSHADERS_DIR}/nvshaders/hdr_integrate_brdf.slang
)
set(SHADER_SLANG_FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_SHARED_SLANG_FILES})
message(STATUS "Compiling shaders: ${SHADER_SLANG_FILES}")

compile_slang(
  "${SHADER_SLANG_FILES}"
  "${CMAKE_BINARY_DIR}/${PROJECT_NAME}/_autogen"
  GENERATED_SHADER_HEADERS
  EXTRA_FLAGS "-I${NVSHADERS_DIR}"
  OPTIMIZATION_LEVEL 1
  DEBUG_LEVEL 1
)

# Add shader files to the project
source_group("Shaders" FILES ${SHADER_SRC_SLANG_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_HEADERS})
source_group("Shaders/Shared" FILES ${SHADER_SHARED_SLANG_FILES})

# Add the output shader headers (target) directly to the executable
# This allow to compile the shaders when the executable is built
target_sources(${PROJECT_NAME} PRIVATE ${GENERATED_SHADER_HEADERS} ${SHADER_H_FILES})


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
option(USE_DRACO "Use Draco for compression" OFF)
if(USE_DRACO)
  include(cmake/draco.cmake)
  download_draco()
  target_include_directories(${PROJECT_NAME} PRIVATE ${draco_SOURCE_DIR}/src ${draco_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE draco::draco)
  target_compile_definitions(tinygltf INTERFACE TINYGLTF_ENABLE_DRACO) # Tinygltf needs to know about Draco
  target_compile_definitions(nvvkgltf PRIVATE USE_DRACO) # Our glTF needs to know about Draco to support the extension
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
# Adding DLSS if selected
option(USE_DLSS "Use DLSS-RR for denoising" ON)
if(USE_DLSS)
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
