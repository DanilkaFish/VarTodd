add_executable(vartodd_demo vartodd.cpp)
target_compile_options(vartodd_demo PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)
        
FIND_PACKAGE( Boost 1.70 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
target_link_libraries(vartodd_demo PRIVATE vartodd ${Boost_LIBRARIES})
target_include_directories(vartodd_demo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})