set(NGX_SDK_ROOT NGX-SDK-ROOT-NOTFOUND CACHE STRING "NGX SDK Root Directory")

if ("${NGX_SDK_ROOT}" STREQUAL "NGX-SDK-ROOT-NOTFOUND")
  message(FATAL_ERROR "NGX_SDK_ROOT not set - please set it and rerun CMAKE Configure")
endif()

add_library(ngx IMPORTED STATIC GLOBAL)

set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_property(TARGET ngx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

set_target_properties(ngx PROPERTIES
	MAP_IMPORTED_CONFIG_MINSIZEREL Release
	MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
)

if (WIN32)
  set(NGX_USE_STATIC_MSVCRT OFF CACHE BOOL "[Deprecated?]Use NGX libs with static VC runtime (/MT), otherwise dynamic (/MD)")

  if(NGX_USE_STATIC_MSVCRT)
    set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_DEBUG ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s_dbg.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_RELEASE ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s_dbg.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib)

  else()
    set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_DEBUG ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_IMPLIB_RELEASE ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib)
    set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${NGX_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib)
  endif()

  # set the list of DLLs that need copying to target folder of application
  file(GLOB __NGX_DLLS_LIST_DEBUG "${NGX_SDK_ROOT}/lib/Windows_x86_64/dev/nvngx_*.dll")
  file(GLOB __NGX_DLLS_LIST_RELEASE "${NGX_SDK_ROOT}/lib/Windows_x86_64/rel/nvngx_*.dll")
else ()
  set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_DEBUG ${NGX_SDK_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a)
  set_target_properties(ngx PROPERTIES IMPORTED_LOCATION_RELEASE ${NGX_SDK_ROOT}/lib/Linux_x86_64/libnvsdk_ngx.a)

  file(GLOB __NGX_DLLS_LIST_DEBUG "${NGX_SDK_ROOT}/lib/Linux_x86_64/dev/libnvidia-ngx-*.so.*")
  file(GLOB __NGX_DLLS_LIST_RELEASE "${NGX_SDK_ROOT}/lib/Linux_x86_64/rel/libnvidia-ngx-*.so.*")
endif()

set(__NGX_DLLS_LIST "$<IF:$<CONFIG:Debug>,${__NGX_DLLS_LIST_DEBUG},${__NGX_DLLS_LIST_RELEASE}>")

# message(STATUS "NGX snippets: ${__NGX_DLLS_LIST}")

set_property(TARGET ngx APPEND PROPERTY EXTRA_DLLS "${__NGX_DLLS_LIST}")

target_include_directories(ngx INTERFACE "${NGX_SDK_ROOT}include")
