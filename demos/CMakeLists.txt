add_executable(vartodd_demo vartodd.cpp)
target_compile_options(vartodd_demo PRIVATE
    $<$<CONFIG:Release>:-O3>
)
if(NOT APPLE)
    target_compile_options(vartodd_demo PRIVATE
        $<$<CONFIG:Release>:-march=native>
    )
endif()

find_package(Boost 1.70 COMPONENTS program_options REQUIRED)
target_link_libraries(vartodd_demo PRIVATE vartodd Boost::program_options)
target_include_directories(vartodd_demo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
