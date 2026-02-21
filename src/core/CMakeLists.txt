set(GF2_CORE_SOURCES
    algorithms.cpp
    matrix.cpp
    nullspace.cpp
    todd_generator.cpp
    todd_index.cpp
)

# Create main library that combines everything
add_library(gf2 SHARED ${GF2_CORE_SOURCES})
# Link dependencies
target_compile_features(gf2 PUBLIC cxx_std_20)
target_link_libraries(gf2
    PUBLIC
        unordered_dense
        cnpy++
)
target_compile_options(gf2 PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found. Linking to target.")
    # Link the imported targets for the relevant languages
    if (OpenMP_CXX_FOUND)
        target_link_libraries(gf2 PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if (OpenMP_C_FOUND)
        target_link_libraries(gf2 PUBLIC OpenMP::OpenMP_C)
    endif()
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

set_property(TARGET gf2 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
target_compile_definitions(gf2 PUBLIC
  $<$<CONFIG:Release>:NDEBUG>
)
# Include directories
target_include_directories(gf2
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)