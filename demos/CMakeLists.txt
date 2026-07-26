add_executable(vartodd_demo vartodd.cpp)
add_executable(bench bench.cpp)
add_executable(bench_todd_index bench_todd_index.cpp)
target_compile_options(bench PRIVATE
    $<$<CONFIG:Release>:-O3 -fno-omit-frame-pointer>
    $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
    $<$<BOOL:${VARTODD_ENABLE_NATIVE_ARCH}>:-march=native>
)
target_compile_options(bench_todd_index PRIVATE
    $<$<CONFIG:Release>:-O3 -fno-omit-frame-pointer>
    $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
    $<$<BOOL:${VARTODD_ENABLE_NATIVE_ARCH}>:-march=native>
)
target_link_libraries(bench PRIVATE vartodd)
target_link_libraries(bench_todd_index PRIVATE vartodd)
target_compile_options(vartodd_demo PRIVATE
    $<$<CONFIG:Release>:-O3>
)
if(NOT APPLE AND VARTODD_ENABLE_NATIVE_ARCH)
    target_compile_options(vartodd_demo PRIVATE
        $<$<CONFIG:Release>:-march=native>
    )
endif()

find_package(Boost 1.70 COMPONENTS program_options REQUIRED)
target_link_libraries(vartodd_demo PRIVATE vartodd Boost::program_options)
target_include_directories(vartodd_demo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
