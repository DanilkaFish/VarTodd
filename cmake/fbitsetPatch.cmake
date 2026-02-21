# cmake/cnpy_wrap.cmake
if(NOT DEFINED FBITSET_SOURCE_PATH)
  message(FATAL_ERROR "FBITSET_SOURCE_PATH not set")
endif()

set(_wrap_dir "${FBITSET_SOURCE_PATH}/cmake/gf2-wrap")
file(MAKE_DIRECTORY "${_wrap_dir}")

file(WRITE "${_wrap_dir}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.18)

add_library(fbitset INTERFACE "${CMAKE_CURRENT_LIST_DIR}/../../include/")
add_library(fbitset::fbitset ALIAS fbitset)

# target_include_directories(fbitset PUBLIC
#   $<INSTALL_INTERFACE:include>
# )

message("-- FBITSET compiling with wrapper")
]=])