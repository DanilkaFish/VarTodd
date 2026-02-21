# ---- Python extension module ----
pybind11_add_module(gf2lib MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/utils_wrapper.cpp"
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(gf2lib PRIVATE OpenMP::OpenMP_CXX pybind11::headers)
else()
    message(WARNING "OpenMP not found; greedy_policy_iteration will run single-threaded.")
endif()

target_link_libraries(gf2lib
    PRIVATE
        gf2
)

if (NOT DEFINED PY_BUILD_CMAKE_IMPORT_NAME)
    set(PY_BUILD_CMAKE_IMPORT_NAME ${CMAKE_PYTHON_OUTPUT_DIRECTORY})
endif()

set_target_properties(gf2lib PROPERTIES
    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}${PYTHON_MODULE_DEBUG_POSTFIX}"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN TRUE
)

# target_link_options(gf2lib PRIVATE "-rdynamic")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(gf2lib PRIVATE "LINKER:--exclude-libs,ALL")
endif()

target_compile_definitions(gf2lib PRIVATE
    MODULE_NAME=$<TARGET_FILE_BASE_NAME:gf2lib>
    VERSION_INFO="${PY_FULL_VERSION}"
)

install(TARGETS gf2lib
        # EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})


# ---- Stub generation ----
set(WITH_PY_STUBS_DEFAULT ON)
set(Python3_FIND_VIRTUALENV ONLY)

option(WITH_PY_STUBS
    "Generate Python stub files (.pyi) for the Python module."
    ${WITH_PY_STUBS_DEFAULT})

if (WITH_PY_STUBS)
    include(${CMAKE_SOURCE_DIR}/cmake/Pybind11Stubgen.cmake)
    message(STATUS "Stubgen for ${PY_BUILD_CMAKE_IMPORT_NAME}")
    # if (USE_LOCAL_BUILDING)
        if (NOT DEFINED Python3_EXECUTABLE)
            find_package(Python3 REQUIRED COMPONENTS Interpreter)
            message(STATUS "PYTHONPATH: ${Python3_EXECUTABLE}")
        endif()
        add_custom_command(
            TARGET gf2lib
            POST_BUILD
            COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen -o . gf2lib
            WORKING_DIRECTORY $<TARGET_FILE_DIR:gf2lib>
            COMMENT "Generating .pyi file next to gf2lib module"
        )
endif()

install(FILES "$<TARGET_FILE_DIR:gf2lib>/gf2lib.pyi"
        # EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME}
        )
