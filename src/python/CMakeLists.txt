# ---- Python extension module ----
pybind11_add_module(pyvartodd MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/todd_wrapper.cpp"
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(pyvartodd PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found; policy_iteration will run single-threaded.")
endif()

target_link_libraries(pyvartodd
    PRIVATE
        vartodd
)


set_target_properties(pyvartodd PROPERTIES
    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}${PYTHON_MODULE_DEBUG_POSTFIX}"
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN TRUE
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_PYTHON_OUTPUT_DIRECTORY}"
)

# target_link_options(pyvartodd PRIVATE "-rdynamic")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(pyvartodd PRIVATE "LINKER:--exclude-libs,ALL")
endif()

target_compile_definitions(pyvartodd PRIVATE
    MODULE_NAME=$<TARGET_FILE_BASE_NAME:pyvartodd>
    VERSION_INFO="${PY_FULL_VERSION}"
)

# ---- Stub generation ----
set(WITH_PY_STUBS_DEFAULT ON)
# set(Python3_FIND_VIRTUALENV ONLY)

option(WITH_PY_STUBS
    "Generate Python stub files (.pyi) for the Python module."
    ${WITH_PY_STUBS_DEFAULT})

if (WITH_PY_STUBS)
    include(${CMAKE_SOURCE_DIR}/external/cmake/Pybind11Stubgen.cmake)
    message(STATUS "Stubgen for ${PY_BUILD_CMAKE_IMPORT_NAME}")
    # if (USE_LOCAL_BUILDING)
        if (NOT DEFINED Python3_EXECUTABLE)
            find_package(Python3 REQUIRED COMPONENTS Interpreter)
            message(STATUS "PYTHONPATH: ${Python3_EXECUTABLE}")
        endif()
        add_custom_command(
            TARGET pyvartodd
            POST_BUILD
            COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen -o . pyvartodd
            WORKING_DIRECTORY $<TARGET_FILE_DIR:pyvartodd>
            COMMENT "Generating .pyi file next to pyvartodd module"
        )
endif()