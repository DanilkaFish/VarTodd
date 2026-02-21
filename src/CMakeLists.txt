add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/python)
endif()