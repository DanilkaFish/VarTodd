include(FetchContent)

# Boost
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(OpenMP REQUIRED)
find_package(Boost 1.70 REQUIRED )

# ZLIB
find_package(ZLIB REQUIRED)

# CNPY

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  # GIT_TAG <pin-a-commit>

  SOURCE_SUBDIR cmake/gf2-wrap

  PATCH_COMMAND
    ${CMAKE_COMMAND}
      -DCNPY_SOURCE_PATH:PATH=<SOURCE_DIR>
      -P ${PROJECT_SOURCE_DIR}/cmake/cnpyPatch.cmake
)
FetchContent_MakeAvailable(cnpy)

FetchContent_Declare(cnpy++
    GIT_REPOSITORY https://github.com/mreininghaus/cnpypp.git
    GIT_SHALLOW True
)
FetchContent_MakeAvailable(cnpy++)


# add_subdirectory(${cnpy_SOURCE_DIR} ${cnpy_BINARY_DIR})
# Ankerl unordered map implementation (
FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG main
)
FetchContent_MakeAvailable(unordered_dense)

# FBitSet
FetchContent_Declare(
    fbitset
    GIT_REPOSITORY https://github.com/DrudgeCAS/fbitset.git
    GIT_TAG master
      SOURCE_SUBDIR cmake/gf2-wrap

    PATCH_COMMAND
        ${CMAKE_COMMAND}
        -DFBITSET_SOURCE_PATH:PATH=<SOURCE_DIR>
        -P ${PROJECT_SOURCE_DIR}/cmake/fbitsetPatch.cmake
)
set(BUILD_TESTS OFF CACHE BOOL "Disable fbitset tests" FORCE)
FetchContent_MakeAvailable(fbitset)

# pybind11 for Python bindings
if(BUILD_PYTHON_BINDINGS)
    set(PY_VERSION_SUFFIX "")
    set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

    # ---- Zlib: used only in python_wrapper ----
    find_package(ZLIB REQUIRED)
    find_package(OpenMP REQUIRED)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# GoogleTest for testing
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
endif()


# if(BUILD_BENCHMARKS)
#   # Make Benchmark NOT try to use/require googletest at all
#   set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)
#   set(BENCHMARK_ENABLE_GTEST_TESTS  OFF CACHE BOOL "" FORCE)

#   # Also prevent it from looking for the bundled googletest folder
#   set(BENCHMARK_USE_BUNDLED_GTEST   OFF CACHE BOOL "" FORCE)

#   # (optional safety) don't let it try to fetch deps by itself
#   set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)

#   FetchContent_Declare(
#     benchmark
#     GIT_REPOSITORY https://github.com/google/benchmark.git
#     GIT_TAG v1.9.4
#   )
#   FetchContent_MakeAvailable(benchmark)
# endif()
message("-- DEPENDECIES ADDED")
# FetchContent_GetProperties(cnpy)
# patch_cnpy_cmakelists()