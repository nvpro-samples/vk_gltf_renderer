# FindSlang.cmake
#
# Downloads the Slang SDK.
#
# Sets the following variables:
# Slang_VERSION: The downloaded version of Slang.
# Slang_ROOT: Path to the Slang SDK root directory.
# Slang_INCLUDE_DIR: Directory that includes slang.h.
# Slang_SLANGC_EXECUTABLE: Path to the Slang compiler.
# Slang_LIBRARY: Linker library.
# Slang_DLL: Shared library.

set(Slang_VERSION "2025.6.1")

# Download Slang SDK.
# We provide two URLs here since some users' proxies might break one or the other.
# The "d4i3qtqj3r0z5.cloudfront.net" address is the public Omniverse Packman
# server; it is not private.
if(WIN32)
    set(Slang_URLS
        "https://d4i3qtqj3r0z5.cloudfront.net/slang%40v${Slang_VERSION}-windows-x64-release.zip"
        "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-windows-x86_64.zip"
    )
else()
    set(Slang_URLS
        "https://d4i3qtqj3r0z5.cloudfront.net/slang%40v${Slang_VERSION}-linux-x86_64-release.zip"
        "https://github.com/shader-slang/slang/releases/download/v${Slang_VERSION}/slang-${Slang_VERSION}-linux-x86_64.zip"
    )
endif()

download_package(
  NAME Slang
  URLS ${Slang_URLS}
  VERSION ${Slang_VERSION}
  LOCATION Slang_SOURCE_DIR
)

# On Linux, the Cloudfront download of Slang might not have the executable bit
# set on its executables and DLLs. This causes find_program to fail. To fix this,
# call chmod a+rwx on those directories:
if(UNIX)
  file(CHMOD_RECURSE ${Slang_SOURCE_DIR}/bin ${Slang_SOURCE_DIR}/lib
       FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE
  )
endif()

set(Slang_ROOT ${Slang_SOURCE_DIR} CACHE PATH "Path to the Slang SDK root directory")
mark_as_advanced(Slang_ROOT)

find_path(Slang_INCLUDE_DIR
  slang.h
  HINTS ${Slang_ROOT}/include
  NO_DEFAULT_PATH
  DOC "Directory that includes slang.h."
)
mark_as_advanced(Slang_INCLUDE_DIR)

find_program(Slang_SLANGC_EXECUTABLE
  NAMES slangc 
  HINTS ${Slang_SOURCE_DIR}/bin
  NO_DEFAULT_PATH
  DOC "Slang compiler (slangc)"
)
mark_as_advanced(Slang_SLANGC_EXECUTABLE)

find_library(Slang_LIBRARY
  NAMES slang
  HINTS ${Slang_SOURCE_DIR}/lib
  NO_DEFAULT_PATH
  DOC "Slang linker library"
)
mark_as_advanced(Slang_LIBRARY)

if(WIN32)
  find_file(Slang_DLL
    NAMES slang.dll
    HINTS ${Slang_SOURCE_DIR}/bin
    NO_DEFAULT_PATH
    DOC "Slang shared library (.dll)"
  )
else() # Unix; uses .so
  set(Slang_DLL ${Slang_LIBRARY} CACHE PATH "Slang shared library (.so)")
endif()
mark_as_advanced(Slang_DLL)

message(STATUS "--> using SLANGC under: ${Slang_SLANGC_EXECUTABLE}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Slang
  REQUIRED_VARS
    Slang_ROOT
    Slang_SLANGC_EXECUTABLE
    Slang_LIBRARY
    Slang_DLL
    Slang_INCLUDE_DIR
  VERSION_VAR
    Slang_VERSION
)