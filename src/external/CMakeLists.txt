include(FetchContent)

# Boost
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.70 REQUIRED)

# ZLIB
find_package(ZLIB REQUIRED)
FetchContent_Declare(cnpy++
    GIT_REPOSITORY https://github.com/mreininghaus/cnpypp.git
    GIT_TAG c6cd4e2078e4f39e862720b66fb211c45577c510
)
FetchContent_MakeAvailable(cnpy++)


# Ankerl unordered map implementation (
FetchContent_Declare(
    unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG c3b4667c7cb5b059f2df100db4ff7203203ad564
)
FetchContent_MakeAvailable(unordered_dense)

# pybind11 for Python bindings
if(BUILD_PYTHON_BINDINGS)
    set(PY_VERSION_SUFFIX "")
    set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

    # ---- Zlib: used only in python_wrapper ----
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()
