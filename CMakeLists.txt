cmake_minimum_required(VERSION 3.17)

# get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJNAME vk_gltf_renderer)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_SOURCE_DIR}/external
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  set(OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin_x64)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

_add_project_definitions(${PROJNAME})

set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

#####################################################################################
# Adding Slang
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(Slang_VERSION "2024.1.30")
find_package(Slang)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/slang.cmake)

#####################################################################################
# Add RapidJSON package
download_package(
  NAME RapidJSON
  URL https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.zip
  VERSION 1.1.0
  LOCATION RapidJSON_SOURCE_DIR
)
set(RAPIDJSON_INCLUDE_DIR ${RapidJSON_SOURCE_DIR}/rapidjson-1.1.0/include/rapidjson)

#####################################################################################
# Adding DRACO if selected
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()
option(USE_DRACO "Use Draco for compression" OFF)
include(cmake/draco.cmake)
if(USE_DRACO)
  download_draco()
endif()

#####################################################################################
# Additions from packages needed for this sample
_add_package_VulkanSDK()    # Vulkan dependencies
_add_package_ShaderC()      # GLSL Compiler
_add_package_IMGUI()        # Dear ImGui interface and utilities
_add_package_NVML()         # GPU monitoring
_add_package_KTX()          # KTX: BasisU images

# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

#####################################################################################
# Source and shader files for this project
file(GLOB SOURCE_FILES src/*.*)
file(GLOB SHADER_SOURCE shaders/*.glsl)
file(GLOB SHADER_HEADER shaders/*.h)
file(GLOB SHADER_SLANG shaders/*.slang)
file(GLOB SHADER_FILES shaders/*.glsl shaders/*.h shaders/*.slang)
file(GLOB SHADER_COMMON_FILES ${NVPRO_CORE_DIR}/nvvkhl/shaders/*.*)
set(MIKKTSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MikkTSpace)
set(MIKKTSPACE_SRC ${MIKKTSPACE_DIR}/mikktspace.c ${MIKKTSPACE_DIR}/mikktspace.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#####################################################################################
# Executable
#
add_executable(${PROJNAME})
# Sources for the project
target_sources(${PROJNAME} PRIVATE ${SOURCE_FILES})
target_sources(${PROJNAME} PRIVATE ${COMMON_SOURCE_FILES})
target_sources(${PROJNAME} PRIVATE ${PACKAGE_SOURCE_FILES})
target_sources(${PROJNAME} PRIVATE ${SHADER_FILES} )
target_sources(${PROJNAME} PRIVATE ${SHADER_COMMON_FILES})
target_sources(${PROJNAME} PRIVATE ${MIKKTSPACE_SRC})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJNAME})

# Compiler options
set_property(TARGET ${PROJNAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJNAME} PROPERTY CXX_STANDARD_REQUIRED ON)
if(MSVC)
    target_compile_options(${PROJNAME}  PRIVATE /W3)
    # We are not forcing the usage of the GLSL and Slang compiler
    target_link_options(${PROJNAME} PRIVATE "/DELAYLOAD:shaderc_shared.dll")
    target_link_options(${PROJNAME} PRIVATE "/DELAYLOAD:slang.dll") 
endif()

# Preprocessor definitions
target_compile_definitions(${PROJNAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS )
target_compile_definitions(${PROJNAME} PRIVATE NVPRO_CORE_DIR="${NVPRO_CORE_DIR}" )

# Extra include directories
target_include_directories(${PROJNAME} PRIVATE ${RAPIDJSON_INCLUDE_DIR})
target_include_directories(${PROJNAME} PRIVATE ${MIKKTSPACE_DIR})

# Folders for Visual Studio
source_group("Common" FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} )
source_group("Shader Files" FILES ${SHADER_FILES} ${SHADER_SLANG})
source_group("Shader Files/common" FILES ${SHADER_COMMON_FILES})
source_group("MikkTSpace" FILES ${MIKKTSPACE_SRC})




# ---- DRACO ----
if(USE_DRACO)
  add_draco()
endif()


# Linking with other libraries
target_link_libraries(${PROJNAME} PRIVATE nvpro_core)
target_link_libraries(${PROJNAME} PRIVATE optimized ${LIBRARIES_OPTIMIZED})
target_link_libraries(${PROJNAME} PRIVATE debug ${LIBRARIES_DEBUG})
target_link_libraries(${PROJNAME} PRIVATE ${PLATFORM_LIBRARIES})
if(UNIX)
    target_link_libraries(${PROJNAME} PRIVATE dl pthread)
endif()

# Resources
download_files(FILENAMES FlightHelmet.zip EXTRACT)
download_files(FILENAMES daytime.hdr std_env.hdr)


#####################################################################################
# GLSL Shader compilation
#
set(_GLSL_FLAGS  
   -Ishaders 
   -I${NVPRO_CORE_DIR} 
   -g
   -D_glsl)
 
# Compiling shaders to Spir-V header
compile_glsl(
     SOURCE_FILES ${SHADER_SOURCE}
     HEADER_FILES ${SHADER_HEADER}
     DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
     VULKAN_TARGET "vulkan1.3"
     HEADER ON
     DEPENDENCY ${Vulkan_BUILD_DEPENDENCIES}
     FLAGS ${_GLSL_FLAGS}
     )

target_sources(${PROJNAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})

####################################################################################################
# Slang Shader compilation
#
target_sources(${PROJNAME} PRIVATE ${SHADER_SLANG})
target_include_directories(${PROJNAME} PRIVATE "${Slang_SDK}/include")
target_link_libraries(${PROJNAME} PRIVATE ${Slang_LIBRARY})
foreach(SLANG_FILE ${SHADER_SLANG})
    compile_slang_file(
        SOURCE_FILE ${SLANG_FILE}
        DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen"
        FLAGS -I${NVPRO_CORE_DIR}
        )
    # target_sources(${PROJNAME} PRIVATE ${SLANG_FILE})
endforeach()


#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#
_finalize_target( ${PROJNAME} )

install(FILES ${SHADER_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}")
install(FILES ${SHADER_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}")
install(FILES ${SHADER_COMMON_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}/nvvkhl/shaders")
install(FILES ${SHADER_COMMON_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}/nvvkhl/shaders")

# Copying Slang DLLs
add_custom_command(TARGET ${PROJNAME} POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Slang_DLL} ${Slang_glslang_DLL} $<TARGET_FILE_DIR:${PROJNAME}> VERBATIM
    )
install(FILES ${Slang_DLL} ${Slang_glslang_DLL} DESTINATION bin_${ARCH}$<IF:$<CONFIG:Debug>,_debug,>)
