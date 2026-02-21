cmake_minimum_required(VERSION 3.6...3.18)

project(vartodd
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Implementation of FastTodd algorithm for finding minimal rank of tensor decomposition"
)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_DEMONSTRATION "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckIPOSupported)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  find_program(GCC_AR gcc-ar REQUIRED)
  find_program(GCC_RANLIB gcc-ranlib REQUIRED)

  set(CMAKE_C_COMPILER_AR "${GCC_AR}")
  set(CMAKE_C_COMPILER_RANLIB "${GCC_RANLIB}")
  set(CMAKE_CXX_COMPILER_AR "${GCC_AR}")
  set(CMAKE_CXX_COMPILER_RANLIB "${GCC_RANLIB}")
endif()
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
  message(WARNING "IPO/LTO not supported: ${ipo_error}")
endif()

# Output directories
set(EXTERNAL_DEPS_DIR "${CMAKE_SOURCE_DIR}/build/_deps/" CACHE PATH "Directory for external dependencies")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_PYTHON_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/vartoddlib)

# Include helper modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/cmake)

# Add subdirectories
add_subdirectory(external)      # Dependencies first
add_subdirectory(src)           # Main library


if(BUILD_DEMONSTRATION)
    add_subdirectory(demos)
endif()
