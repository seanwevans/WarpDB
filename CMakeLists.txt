cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

# Avoid passing system headers with '-isystem' which may interfere
# with '#include_next' in the standard library on some platforms.
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-I")

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

find_package(Arrow QUIET COMPONENTS cuda)

if(Arrow_FOUND)
    message(STATUS "Found Apache Arrow: ${Arrow_VERSION}")
    add_definitions(-DUSE_ARROW)
    include_directories(${Arrow_INCLUDE_DIRS})
endif()

include_directories(include)

set(WARPDB_SRC
    src/main.cu
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cpp
    src/arrow_utils.cpp
    src/optimizer.cpp
)


if(Arrow_FOUND)
    list(APPEND WARPDB_SRC src/arrow_loader.cpp)
endif()

add_executable(warpdb ${WARPDB_SRC})

set_target_properties(warpdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


if(Arrow_FOUND)
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver ${Arrow_LIBRARIES})
else()
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
endif()

add_executable(expression_test
    tests/test_expression.cpp
    src/expression.cpp
)

add_test(NAME expression_test COMMAND expression_test)


add_executable(expression_tests
    tests/expression_tests.cpp
    src/expression.cpp
)

add_test(NAME expression_tests COMMAND expression_tests)


add_executable(query_parser_test
    tests/query_parser_test.cpp
    src/expression.cpp
)

add_test(NAME query_parser_test COMMAND query_parser_test)


add_executable(sql_features_test
    tests/sql_features_test.cpp
)
target_link_libraries(sql_features_test PRIVATE warpdb_lib)

add_test(NAME sql_features_test COMMAND sql_features_test)

add_executable(jit_error_test
    tests/jit_error_test.cpp
    src/jit.cpp
)
target_link_libraries(jit_error_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_error_test COMMAND jit_error_test)

add_executable(jit_arch_test
    tests/jit_arch_test.cpp
    src/jit.cpp
)
target_link_libraries(jit_arch_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_arch_test COMMAND jit_arch_test)

add_library(warpdb_lib STATIC
    src/warpdb.cpp
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cpp
    src/arrow_utils.cpp
)

set_target_properties(warpdb_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::nvrtc CUDA::cuda_driver)

if(Arrow_FOUND)
    target_sources(warpdb_lib PRIVATE src/arrow_loader.cpp)
endif()
set_target_properties(warpdb_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)


find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pywarpdb bindings/python/pywarpdb.cpp)
    target_link_libraries(pywarpdb PRIVATE warpdb_lib)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME python_module_test
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py)
    set_tests_properties(python_module_test PROPERTIES
        DEPENDS pywarpdb
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:pywarpdb>")
endif()

# Test building without Apache Arrow available
add_test(
    NAME no_arrow_build
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/build_no_arrow.sh
)
