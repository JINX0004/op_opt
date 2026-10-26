cmake_minimum_required(VERSION 3.16)
project(matmul_v3_standalone_optimizer LANGUAGES CXX)

add_library(mmopt optimizer.cpp event_builder.cpp plan_finalizer.cpp)
target_compile_features(mmopt PUBLIC cxx_std_17)
target_include_directories(mmopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(mmopt PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mmopt_cli main.cpp)
target_link_libraries(mmopt_cli PRIVATE mmopt)

add_executable(mmopt_nested_example nested_example.cpp)
target_link_libraries(mmopt_nested_example PRIVATE mmopt)

enable_testing()
add_executable(mmopt_tests tests.cpp)
target_link_libraries(mmopt_tests PRIVATE mmopt)
add_test(NAME mmopt_tests COMMAND mmopt_tests)
add_test(NAME mmopt_example
         COMMAND mmopt_cli ${CMAKE_CURRENT_SOURCE_DIR}/example.manifest)
add_test(NAME mmopt_nested_example COMMAND mmopt_nested_example)

# HiGHS is optional so the original CANN-independent event scorer still
# builds where no MILP dependency is installed. Point MMOPT_HIGHS_ROOT at a
# package prefix (including an unpacked distribution package) if needed.
set(MMOPT_HIGHS_ROOT "" CACHE PATH "Root of a HiGHS C API installation")
find_path(MMOPT_HIGHS_INCLUDE_DIR highs/interfaces/highs_c_api.h
          HINTS ${MMOPT_HIGHS_ROOT}/include ${MMOPT_HIGHS_ROOT}/usr/include)
find_library(MMOPT_HIGHS_LIBRARY NAMES highs
             HINTS ${MMOPT_HIGHS_ROOT}/lib ${MMOPT_HIGHS_ROOT}/lib64
                   ${MMOPT_HIGHS_ROOT}/usr/lib/x86_64-linux-gnu)
if(MMOPT_HIGHS_INCLUDE_DIR AND MMOPT_HIGHS_LIBRARY)
  add_library(mmopt_mip mip_solver.cpp)
  target_compile_features(mmopt_mip PUBLIC cxx_std_17)
  target_include_directories(mmopt_mip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(mmopt_mip SYSTEM PRIVATE ${MMOPT_HIGHS_INCLUDE_DIR}
                                                     ${MMOPT_HIGHS_INCLUDE_DIR}/highs)
  target_link_libraries(mmopt_mip PUBLIC mmopt ${MMOPT_HIGHS_LIBRARY})
  target_compile_options(mmopt_mip PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(mmopt_mip_example mip_example.cpp)
  target_link_libraries(mmopt_mip_example PRIVATE mmopt_mip)
  add_test(NAME mmopt_mip_example COMMAND mmopt_mip_example)

  add_executable(mmopt_mip_tests mip_tests.cpp)
  target_link_libraries(mmopt_mip_tests PRIVATE mmopt_mip)
  add_test(NAME mmopt_mip_tests COMMAND mmopt_mip_tests)

  add_executable(mmopt_mip_core_domain_tests mip_core_domain_tests.cpp)
  target_link_libraries(mmopt_mip_core_domain_tests PRIVATE mmopt_mip)
  add_test(NAME mmopt_mip_core_domain_tests COMMAND mmopt_mip_core_domain_tests)

  add_executable(mmopt_directional_read_check bench910/directional_read/model_check.cpp)
  target_link_libraries(mmopt_directional_read_check PRIVATE mmopt_mip)

  add_executable(mmopt_l1_step_read_check bench910/l1_step_read/model_check.cpp)
  target_link_libraries(mmopt_l1_step_read_check PRIVATE mmopt_mip)

  add_executable(mmopt_full_solver_audit bench910/full_solver_audit.cpp)
  target_link_libraries(mmopt_full_solver_audit PRIVATE mmopt_mip)
else()
  message(STATUS "HiGHS not found: reduced MILP target is disabled")
endif()
