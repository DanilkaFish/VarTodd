add_executable(fast_todd fast_todd.cpp)
target_compile_options(fast_todd PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)
target_link_libraries(fast_todd PRIVATE gf2)
target_include_directories(fast_todd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})