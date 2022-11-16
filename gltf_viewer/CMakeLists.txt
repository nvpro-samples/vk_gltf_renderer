#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.h)


add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${ADVANCE_DIR} )
target_link_libraries (${PROJECT_NAME} ${PLATFORM_LIBRARIES} ${LIBRARIES_OPTIMIZED} nvpro_core)
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
source_group("Other"  FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

_add_project_definitions(${PROJECT_NAME})


#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Local shaders
file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
file(GLOB SHD_SRC ${SHD_DIR}/*.vert ${SHD_DIR}/*.frag ${SHD_DIR}/*.rgen ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rahit ${SHD_DIR}/*.rmiss ${SHD_DIR}/*.comp)

# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
    FLAGS "-I${SHD_DIR} -I${NVPRO_CORE_DIR} -g"
    )

target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
source_group("Shaders/src"   FILES ${GLSL_SOURCES} )
source_group("Shaders/hdr"   FILES ${GLSL_HEADERS} )

# Copy binary
_finalize_target( ${PROJECT_NAME} )