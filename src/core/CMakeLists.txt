set(VARTODD_CORE_SOURCES
    algorithms.cpp
    matrix.cpp
    nullspace.cpp
    todd_generator.cpp
    todd_index.cpp
)

find_package(Boost 1.70 REQUIRED )
add_library(vartodd STATIC ${VARTODD_CORE_SOURCES})

# target_compile_features(vartodd PUBLIC cxx_std_20)
target_link_libraries(vartodd
    PUBLIC
        unordered_dense
        cnpy++
        Boost::headers
)
target_compile_options(vartodd PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)

find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    message(STATUS "OpenMP found. Linking to target.")
    # Link the imported targets for the relevant languages
    if (OpenMP_CXX_FOUND)
        target_link_libraries(vartodd PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if (OpenMP_C_FOUND)
        target_link_libraries(vartodd PUBLIC OpenMP::OpenMP_C)
    endif()
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

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