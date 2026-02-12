# Test executable
set(TEST_NAME ${PROJECT_NAME}_tests)

# Common test utilities
set(TEST_COMMON_SOURCES
    common/test_utils.hpp
    common/test_utils.cpp
    test_main.cpp
)

# Test sources (will grow in each phase)
set(TEST_SOURCES
    ${TEST_COMMON_SOURCES}
    test_basic.cpp
    # Need tinygltf implementation
    ${CMAKE_SOURCE_DIR}/src/tiny_stb_implementation.cpp
    # Local glTF scene implementation
    ${CMAKE_SOURCE_DIR}/src/gltf_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_converter.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_animation_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/create_tangent.cpp
    ${CMAKE_SOURCE_DIR}/src/compact_model.cpp
    # MikkTSpace for tangent generation
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace/mikktspace.c
    # Phase-specific tests added here as we progress
)

# Create test executable
add_executable(${TEST_NAME} ${TEST_SOURCES})

# Add compile definitions for tinygltf
target_compile_definitions(${TEST_NAME} PRIVATE
    TINYGLTF_USE_RAPIDJSON
    TINYGLTF_USE_RAPIDJSON_CRTALLOCATOR
    TINYGLTF_NO_EXTERNAL_IMAGE
)

# Add Draco support if enabled
if(USE_DRACO)
    target_include_directories(${TEST_NAME} PRIVATE 
        ${draco_SOURCE_DIR}/src 
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE draco::draco)
    target_compile_definitions(${TEST_NAME} PRIVATE TINYGLTF_ENABLE_DRACO)
endif()

# Link with main project libraries + gtest
target_link_libraries(${TEST_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
    # Same libraries as main project
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvslang
    nvpro2::nvshaders_host
    nvpro2::nvaftermath
    nvpro2::nvgpu_monitor
    # Additional dependencies for local gltf implementation
    tinygltf
    stb
    fmt
    glm
    nvimageformats
    tinyobjloader
    meshoptimizer
)

# RapidJSON (needed for tinygltf) - tinygltf expects rapidjson headers directly
# Main project uses: ${RapidJSON_SOURCE_DIR}/rapidjson-master/include/rapidjson
set(TEST_RAPIDJSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/RapidJSON-master/rapidjson-master/include/rapidjson)

# Include directories
target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace
    ${TEST_RAPIDJSON_INCLUDE_DIR}
)

# Copy test resources
file(COPY ${CMAKE_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Register with CTest
include(GoogleTest)
gtest_discover_tests(${TEST_NAME})

# Add to build
set_target_properties(${TEST_NAME} PROPERTIES
    FOLDER "Tests"
)

#####################################################################################
# Benchmark executable (separate from tests)
set(BENCHMARK_NAME ${PROJECT_NAME}_benchmarks)

add_executable(${BENCHMARK_NAME}
    benchmark_main.cpp
    common/test_utils.cpp
    # Need tinygltf implementation
    ${CMAKE_SOURCE_DIR}/src/tiny_stb_implementation.cpp
    # Local glTF scene implementation
    ${CMAKE_SOURCE_DIR}/src/gltf_scene.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_converter.cpp
    ${CMAKE_SOURCE_DIR}/src/tinygltf_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/gltf_animation_pointer.cpp
    ${CMAKE_SOURCE_DIR}/src/create_tangent.cpp
    ${CMAKE_SOURCE_DIR}/src/compact_model.cpp
    # MikkTSpace for tangent generation
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace/mikktspace.c
)

# Add compile definitions for tinygltf
target_compile_definitions(${BENCHMARK_NAME} PRIVATE
    TINYGLTF_USE_RAPIDJSON
    TINYGLTF_USE_RAPIDJSON_CRTALLOCATOR
    TINYGLTF_NO_EXTERNAL_IMAGE
)

# Add Draco support if enabled
if(USE_DRACO)
    target_include_directories(${BENCHMARK_NAME} PRIVATE 
        ${draco_SOURCE_DIR}/src 
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${BENCHMARK_NAME} PRIVATE draco::draco)
    target_compile_definitions(${BENCHMARK_NAME} PRIVATE TINYGLTF_ENABLE_DRACO)
endif()

target_link_libraries(${BENCHMARK_NAME} PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    # Same libraries as test executable
    nvpro2::nvutils
    nvpro2::nvvk
    nvpro2::nvapp
    nvpro2::nvgui
    nvpro2::nvslang
    nvpro2::nvshaders_host
    nvpro2::nvaftermath
    nvpro2::nvgpu_monitor
    # Additional dependencies for local gltf implementation
    tinygltf
    stb
    fmt
    glm
    nvimageformats
    tinyobjloader
    meshoptimizer
)

target_include_directories(${BENCHMARK_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/third_party/MikkTSpace
    ${TEST_RAPIDJSON_INCLUDE_DIR}
)

set_target_properties(${BENCHMARK_NAME} PROPERTIES
    FOLDER "Tests"
)
