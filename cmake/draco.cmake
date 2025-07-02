# DRACO 
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(download_draco)
    download_package(
        NAME draco
        URLS https://github.com/google/draco/archive/refs/tags/1.5.7.zip
        VERSION 1.5.7
        LOCATION draco_SOURCE_DIR
    )
    
    set(_ORIGINAL_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
    set(CMAKE_MESSAGE_LOG_LEVEL "WARNING")
    message(STATUS "Setting CMAKE_MESSAGE_LOG_LEVEL to WARNING")
    add_subdirectory(${draco_SOURCE_DIR}/draco-1.5.7 ${draco_BINARY_DIR} EXCLUDE_FROM_ALL)
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
        draco_decoder
        draco_enc_config
        draco_encoder
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
        set_target_properties(${target} PROPERTIES FOLDER "Draco")
    endforeach()
endfunction()


# ---- DRACO
function(add_draco)
    target_include_directories(${PROJNAME} PRIVATE ${draco_SOURCE_DIR}/src ${draco_BINARY_DIR})
    target_link_libraries(${PROJNAME} PRIVATE draco::draco)
    target_compile_definitions(${PROJNAME} PRIVATE TINYGLTF_ENABLE_DRACO)
    target_compile_definitions(nvpro_core PRIVATE USE_DRACO)
endfunction()