add_executable(vartodd_core_regression core_regression.cpp)

target_link_libraries(vartodd_core_regression
    PRIVATE
        vartodd
        cnpy++
)

target_compile_definitions(vartodd_core_regression PRIVATE
    VARTODD_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    VARTODD_TEST_TMP_DIR="${PROJECT_BINARY_DIR}/Testing/tmp"
)

add_test(NAME vartodd_core_regression COMMAND vartodd_core_regression)
