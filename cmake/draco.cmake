# DRACO 
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(download_draco)
    download_package(
        NAME draco
        URLS https://github.com/google/draco/archive/refs/heads/main.zip
        VERSION main
        LOCATION draco_SOURCE_DIR
    )
    
    # Configure draco with minimal build options for tinygltf support
    set(DRACO_BUILD_EXECUTABLES OFF CACHE BOOL "Disable building draco command-line tools" FORCE)
    set(DRACO_TESTS OFF CACHE BOOL "Disable draco tests" FORCE)
    set(DRACO_UNITY_PLUGIN OFF CACHE BOOL "Disable Unity plugin" FORCE)
    set(DRACO_MAYA_PLUGIN OFF CACHE BOOL "Disable Maya plugin" FORCE)
    
    # Suppress CMP0148 warning about deprecated FindPythonInterp module in draco subdirectory
    set(CMAKE_POLICY_DEFAULT_CMP0148 OLD)
    
    set(_ORIGINAL_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
    set(CMAKE_MESSAGE_LOG_LEVEL "WARNING")
    message(STATUS "Setting CMAKE_MESSAGE_LOG_LEVEL to WARNING")
    add_subdirectory(${draco_SOURCE_DIR}/draco-main ${draco_BINARY_DIR} EXCLUDE_FROM_ALL)
    set(CMAKE_MESSAGE_LOG_LEVEL ${_ORIGINAL_CMAKE_MESSAGE_LOG_LEVEL})
    set(draco_targets 
        draco 
        draco_animation 
        draco_animation_dec 
        draco_animation_enc
        draco_compression_attributes_dec
        draco_compression_attributes_enc
        draco_compression_attributes_pred_schemes_dec
        draco_compression_attributes_pred_schemes_enc
        draco_compression_bit_coders
        draco_compression_decode
        draco_compression_encode
        draco_compression_entropy
        draco_compression_mesh_dec
        draco_compression_mesh_enc
        draco_compression_mesh_traverser
        draco_compression_options
        draco_compression_point_cloud_dec
        draco_compression_point_cloud_enc
        draco_core
        draco_dec_config
        draco_attributes
        draco_enc_config
        draco_io
        draco_mesh
        draco_metadata
        draco_metadata_dec
        draco_metadata_enc
        draco_point_cloud
        draco_points_dec
        draco_points_enc
    )
    foreach(target IN LISTS draco_targets)
        if(TARGET ${target})
            set_target_properties(${target} PROPERTIES FOLDER "Draco")
        endif()
    endforeach()
endfunction()


# ---- DRACO
function(add_draco)
    target_include_directories(${PROJNAME} PRIVATE ${draco_SOURCE_DIR}/src ${draco_BINARY_DIR})
    target_link_libraries(${PROJNAME} PRIVATE draco::draco)
    target_compile_definitions(${PROJNAME} PRIVATE TINYGLTF_ENABLE_DRACO)
    target_compile_definitions(nvpro_core PRIVATE USE_DRACO)
endfunction()