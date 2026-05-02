set(VARTODD_CORE_SOURCES
    algorithms.cpp
    matrix.cpp
    nullspace.cpp
    todd_generator.cpp
    todd_index.cpp
)

find_package(Boost 1.70 REQUIRED)
# Create an object library with PIC
add_library(vartodd OBJECT ${VARTODD_CORE_SOURCES})
set_target_properties(vartodd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# target_compile_features(vartodd PUBLIC cxx_std_20)
target_link_libraries(vartodd
    PRIVATE
        cnpy++
    PUBLIC
        unordered_dense
        Boost::headers
)

target_compile_options(vartodd PRIVATE
    $<$<CONFIG:Release>:-O3>
)
# target_link_libraries(vartodd PUBLIC tcmalloc)
# set_property(TARGET vartodd PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
target_compile_definitions(vartodd PUBLIC
  $<$<CONFIG:Release>:NDEBUG>
)
# Include directories
target_include_directories(vartodd
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
