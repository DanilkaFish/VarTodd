# cmake/cnpy_wrap.cmake
if(NOT DEFINED CNPY_SOURCE_PATH)
  message(FATAL_ERROR "CNPY_SOURCE_PATH not set")
endif()

set(_wrap_dir "${CNPY_SOURCE_PATH}/cmake/gf2-wrap")
file(MAKE_DIRECTORY "${_wrap_dir}")

file(WRITE "${_wrap_dir}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.18)

# Build cnpy ourselves (cnpy.cpp + cnpy.h at repo root)
add_library(cnpy STATIC "${CMAKE_CURRENT_LIST_DIR}/../../cnpy.cpp")
add_library(cnpy::cnpy ALIAS cnpy)

target_include_directories(cnpy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../..>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(cnpy PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

target_compile_features(cnpy PUBLIC cxx_std_11)
message("-- CNPY compiling with wrapper")
]=])