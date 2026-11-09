cmake_minimum_required(VERSION 3.16)
project(matmul_v3_mip_selector LANGUAGES CXX)

add_executable(mmopt_selector
    selector_cli.cpp
    model/mip_solver.cpp
    model/plan_finalizer.cpp)
target_compile_features(mmopt_selector PRIVATE cxx_std_17)
target_include_directories(mmopt_selector PRIVATE model)

enable_testing()
add_executable(mmopt_dtype_tests
    dtype_tests.cpp
    model/mip_solver.cpp
    model/plan_finalizer.cpp)
target_compile_features(mmopt_dtype_tests PRIVATE cxx_std_17)
target_include_directories(mmopt_dtype_tests PRIVATE model)
target_include_directories(mmopt_dtype_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME mmopt_dtype_tests COMMAND mmopt_dtype_tests)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME mmopt_fp16_golden
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fp16_golden.py
            $<TARGET_FILE:mmopt_selector> ${CMAKE_CURRENT_SOURCE_DIR}/fp16_golden.csv)
