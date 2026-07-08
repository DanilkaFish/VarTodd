cmake_minimum_required(VERSION 3.20)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target macOS architecture")
    endif()
endif()

project(vartodd
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Implementation of FastTodd algorithm for finding minimal rank of tensor decomposition"
)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_DEMONSTRATION "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build examples" ON)

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
include(CheckIPOSupported)

if(APPLE)
    set(Python3_FIND_FRAMEWORK LAST)
endif()

option(VARTODD_ENABLE_IPO "Enable IPO/LTO in Release builds when supported" ON)
if(VARTODD_ENABLE_IPO)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(WARNING "IPO/LTO not supported: ${ipo_error}")
    endif()
endif()

# Output directories
set(EXTERNAL_DEPS_DIR "${CMAKE_SOURCE_DIR}/build/_deps/" CACHE PATH "Directory for external dependencies")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_PYTHON_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pyvartodd)

# Include helper modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/external/cmake)

# Add subdirectories
add_subdirectory(src/external)      # Dependencies first
add_subdirectory(src)           # Main library

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(BUILD_DEMONSTRATION)
    add_subdirectory(demos)
endif()
