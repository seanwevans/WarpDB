cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

# Avoid passing system headers with '-isystem' which may interfere
# with '#include_next' in the standard library on some platforms.
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-I")

enable_testing()

set(CMAKE_CXX_STANDARD 17)

# Make the public headers available to every target. Several test executables
# compile tests/*.cpp together with src/expression.cpp without linking
# warpdb_lib, so without this they fail to find "expression.hpp".
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ctest runs from the build directory, but several loader tests open data files
# by relative path (e.g. "data/numeric_valid.csv"). Copy the data directory into
# the build tree at configure time so those paths resolve.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(CUDAToolkit REQUIRED)

find_package(Arrow QUIET COMPONENTS cuda)

include(FetchContent)
# Do not build nlohmann/json's bundled unit tests. v3.2.0 vendors an old Catch
# (test/thirdparty/catch/catch.hpp) that no longer compiles on modern glibc,
# where SIGSTKSZ is not a compile-time constant ("size of array 'altStackMem'
# is not an integral constant-expression"). We only consume the library
# headers, so its test suite must not be part of our build.
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.2.0 # or a newer compatible version
)
FetchContent_MakeAvailable(nlohmann_json)

if(Arrow_FOUND)
    message(STATUS "Found Apache Arrow: ${Arrow_VERSION}")
    add_definitions(-DUSE_ARROW)
    include_directories(${Arrow_INCLUDE_DIRS})
endif()

# warpdb.cpp uses Thrust with a CUDA execution policy (thrust::device), which
# must be compiled by nvcc rather than the host C++ compiler. Compile it as a
# CUDA source everywhere it is used (the warpdb executable and warpdb_lib).
set_source_files_properties(src/warpdb.cpp PROPERTIES LANGUAGE CUDA)

set(WARPDB_SRC
    src/main.cu
    src/warpdb.cpp
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cu
    src/arrow_utils.cpp
    src/multi_gpu_utils.cpp
    src/optimizer.cpp
)


if(Arrow_FOUND)
    list(APPEND WARPDB_SRC src/arrow_loader.cpp)
endif()

add_executable(warpdb ${WARPDB_SRC})
target_include_directories(warpdb PRIVATE include)

set_target_properties(warpdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


if(Arrow_FOUND)
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver ${Arrow_LIBRARIES} nlohmann_json::nlohmann_json)
else()
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver nlohmann_json::nlohmann_json)
endif()

add_executable(expression_test
    tests/test_expression.cpp
    src/expression.cpp
)

add_test(NAME expression_test COMMAND expression_test)


# expression_tests exercises the CUDA JIT path (jit_compile_and_launch,
# cudaMalloc/cudaMemcpy, and the skip-if-no-CUDA helper), so it must link the
# full library rather than only src/expression.cpp.
add_executable(expression_tests
    tests/expression_tests.cpp
)
target_link_libraries(expression_tests PRIVATE warpdb_lib)

add_test(NAME expression_tests COMMAND expression_tests)


add_executable(query_parser_test
    tests/query_parser_test.cpp
    src/expression.cpp
)

add_test(NAME query_parser_test COMMAND query_parser_test)

add_executable(parsing_error_tests
    tests/parsing_error_tests.cpp
    src/expression.cpp
)

add_test(NAME parsing_error_tests COMMAND parsing_error_tests)

add_executable(tokenizer_tests
    tests/tokenizer_tests.cpp
    src/expression.cpp
)

add_test(NAME tokenizer_tests COMMAND tokenizer_tests)


add_executable(having_aggregate_test
    tests/having_aggregate_test.cpp
    src/expression.cpp
)

add_test(NAME having_aggregate_test COMMAND having_aggregate_test)
add_executable(equals_operator_test
    tests/equals_operator_test.cpp
    src/expression.cpp
)

add_test(NAME equals_operator_test COMMAND equals_operator_test)


add_executable(select_star_test
    tests/select_star_test.cpp
    src/expression.cpp
)

add_test(NAME select_star_test COMMAND select_star_test)
add_executable(eval_logical_ops_test
    tests/eval_logical_ops_test.cpp
    src/expression.cpp
)
# eval_helpers.hpp pulls in cuda_utils.hpp/<cuda_runtime.h> transitively. This
# test exercises only host logic, so use the lightweight CUDA stub headers
# instead of requiring the real toolkit include path for this CXX target.
target_include_directories(eval_logical_ops_test PRIVATE tests/stubs)

add_test(NAME eval_logical_ops_test COMMAND eval_logical_ops_test)
add_executable(not_equal_operator_test
    tests/not_equal_operator_test.cpp
    src/expression.cpp
)

add_test(NAME not_equal_operator_test COMMAND not_equal_operator_test)


add_executable(sql_features_test
    tests/sql_features_test.cpp
)
target_link_libraries(sql_features_test PRIVATE warpdb_lib)

add_test(NAME sql_features_test COMMAND sql_features_test)

add_executable(join_query_test
    tests/join_query_test.cpp
)
target_link_libraries(join_query_test PRIVATE warpdb_lib)
add_test(NAME join_query_test COMMAND join_query_test)

add_executable(extended_types_test
    tests/extended_types_test.cpp
)
target_link_libraries(extended_types_test PRIVATE warpdb_lib)
add_test(NAME extended_types_test COMMAND extended_types_test)

add_executable(string_column_query_test
    tests/string_column_query_test.cpp
)
target_link_libraries(string_column_query_test PRIVATE warpdb_lib)
add_test(NAME string_column_query_test COMMAND string_column_query_test)

add_executable(multi_gpu_sharding_test
    tests/multi_gpu_sharding_test.cpp
)
target_link_libraries(multi_gpu_sharding_test PRIVATE warpdb_lib)
add_test(NAME multi_gpu_sharding_test COMMAND multi_gpu_sharding_test)

add_executable(multi_gpu_arbitrary_columns_test
    tests/multi_gpu_arbitrary_columns_test.cpp
)
target_link_libraries(multi_gpu_arbitrary_columns_test PRIVATE warpdb_lib)
add_test(NAME multi_gpu_arbitrary_columns_test COMMAND multi_gpu_arbitrary_columns_test)

add_executable(cuda_error_handling_test
    tests/cuda_error_handling_test.cpp)
target_link_libraries(cuda_error_handling_test PRIVATE warpdb_lib)
add_test(NAME cuda_error_handling_test COMMAND cuda_error_handling_test)

add_executable(csv_loader_error_test
    tests/csv_loader_error_test.cpp)
target_link_libraries(csv_loader_error_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_error_test COMMAND csv_loader_error_test)

add_executable(csv_loader_header_only_gpu_test
    tests/csv_loader_header_only_gpu_test.cpp)
target_link_libraries(csv_loader_header_only_gpu_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_header_only_gpu_test COMMAND csv_loader_header_only_gpu_test)

add_executable(csv_loader_row_length_test
    tests/csv_loader_row_length_test.cpp)
target_link_libraries(csv_loader_row_length_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_row_length_test COMMAND csv_loader_row_length_test)

add_executable(csv_loader_chunk_row_length_test
    tests/csv_loader_chunk_row_length_test.cpp)
target_link_libraries(csv_loader_chunk_row_length_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_chunk_row_length_test COMMAND csv_loader_chunk_row_length_test)

add_executable(csv_loader_chunk_mixed_types_test
    tests/csv_loader_chunk_mixed_types_test.cpp)
target_link_libraries(csv_loader_chunk_mixed_types_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_chunk_mixed_types_test COMMAND csv_loader_chunk_mixed_types_test)

add_executable(csv_loader_unsupported_type_test
    tests/csv_loader_unsupported_type_test.cpp
    src/csv_loader.cpp)
target_link_libraries(csv_loader_unsupported_type_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME csv_loader_unsupported_type_test COMMAND csv_loader_unsupported_type_test)

add_executable(json_loader_error_test
    tests/json_loader_error_test.cpp)
target_link_libraries(json_loader_error_test PRIVATE warpdb_lib)
add_test(NAME json_loader_error_test COMMAND json_loader_error_test)

add_executable(json_loader_permissive_test
    tests/json_loader_permissive_test.cpp)
target_link_libraries(json_loader_permissive_test PRIVATE warpdb_lib)
add_test(NAME json_loader_permissive_test COMMAND json_loader_permissive_test)

add_executable(json_loader_schema_inference_test
    tests/json_loader_schema_inference_test.cpp)
target_link_libraries(json_loader_schema_inference_test PRIVATE warpdb_lib)
add_test(NAME json_loader_schema_inference_test COMMAND json_loader_schema_inference_test)


add_executable(jit_error_test
    tests/jit_error_test.cpp
    src/jit.cu
)
target_link_libraries(jit_error_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_error_test COMMAND jit_error_test)

add_executable(jit_arch_test
    tests/jit_arch_test.cpp
    src/jit.cu
)
target_link_libraries(jit_arch_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_arch_test COMMAND jit_arch_test)

add_executable(jit_where_zero_test
    tests/jit_where_zero_test.cpp
    src/jit.cu
)
target_link_libraries(jit_where_zero_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_where_zero_test COMMAND jit_where_zero_test)

add_executable(jit_sort_parallel_test
    tests/jit_sort_parallel_test.cpp
    src/jit.cu
)
target_link_libraries(jit_sort_parallel_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_sort_parallel_test COMMAND jit_sort_parallel_test)
add_executable(jit_cache_test
    tests/jit_cache_test.cpp
    src/jit.cu
)
target_link_libraries(jit_cache_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_cache_test COMMAND jit_cache_test)

# optimizer.cpp no longer references jit.cu; the test only needs the CUDA
# runtime for the device allocations it makes itself.
add_executable(optimizer_test
    tests/optimizer_test.cpp
    src/expression.cpp
    src/optimizer.cpp
)
target_link_libraries(optimizer_test PRIVATE CUDA::cudart)
add_test(NAME optimizer_test COMMAND optimizer_test)

add_library(warpdb_lib STATIC
    src/warpdb.cpp
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cu
    src/arrow_utils.cpp
    src/multi_gpu_utils.cpp
)
target_include_directories(warpdb_lib PUBLIC include)

set_target_properties(warpdb_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::nvrtc CUDA::cuda_driver)

if(Arrow_FOUND)
    target_sources(warpdb_lib PRIVATE src/arrow_loader.cpp)
endif()
target_link_libraries(warpdb_lib PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver nlohmann_json::nlohmann_json)

if(Arrow_FOUND)
    target_link_libraries(warpdb_lib PUBLIC ${Arrow_LIBRARIES})
endif()

if(Arrow_FOUND)
    add_executable(arrow_loader_smoke_test tests/arrow_loader_smoke_test.cpp)
    target_link_libraries(arrow_loader_smoke_test PRIVATE warpdb_lib)
    add_test(NAME arrow_loader_smoke_test COMMAND arrow_loader_smoke_test)
    add_executable(arrow_host_multi_gpu_sql_test tests/arrow_host_multi_gpu_sql_test.cpp)
    target_link_libraries(arrow_host_multi_gpu_sql_test PRIVATE warpdb_lib)
    add_test(NAME arrow_host_multi_gpu_sql_test COMMAND arrow_host_multi_gpu_sql_test)
endif()


# Tests that previously existed in tests/ but were never registered with CTest.

# Host-only parser/query tests (only need the expression sources).
add_executable(parse_query_error_test
    tests/parse_query_error_test.cpp
    src/expression.cpp
)
add_test(NAME parse_query_error_test COMMAND parse_query_error_test)

add_executable(tokenize_error_test
    tests/tokenize_error_test.cpp
    src/expression.cpp
)
add_test(NAME tokenize_error_test COMMAND tokenize_error_test)

add_executable(precedence_tests
    tests/precedence_tests.cpp
    src/expression.cpp
)
add_test(NAME precedence_tests COMMAND precedence_tests)

add_executable(identifier_validation_test
    tests/identifier_validation_test.cpp
    src/expression.cpp
)
add_test(NAME identifier_validation_test COMMAND identifier_validation_test)

add_executable(having_offset_query_test
    tests/having_offset_query_test.cpp
    src/expression.cpp
)
add_test(NAME having_offset_query_test COMMAND having_offset_query_test)

# Tests that link the full library. They run on the host but skip GPU work
# gracefully when no CUDA device is present (see tests/test_utils.hpp).
add_executable(eval_helper_test tests/eval_helper_test.cpp)
target_link_libraries(eval_helper_test PRIVATE warpdb_lib)
add_test(NAME eval_helper_test COMMAND eval_helper_test)

add_executable(csv_from_chars_test tests/csv_from_chars_test.cpp)
target_link_libraries(csv_from_chars_test PRIVATE warpdb_lib)
add_test(NAME csv_from_chars_test COMMAND csv_from_chars_test)

add_executable(csv_loader_chunk_test tests/csv_loader_chunk_test.cpp)
target_link_libraries(csv_loader_chunk_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_chunk_test COMMAND csv_loader_chunk_test)

add_executable(having_distinct_test tests/having_distinct_test.cpp)
target_link_libraries(having_distinct_test PRIVATE warpdb_lib)
add_test(NAME having_distinct_test COMMAND having_distinct_test)

add_executable(query_sql_validation_test tests/query_sql_validation_test.cpp)
target_link_libraries(query_sql_validation_test PRIVATE warpdb_lib)
add_test(NAME query_sql_validation_test COMMAND query_sql_validation_test)

add_executable(jit_compile_log_test
    tests/jit_compile_log_test.cpp
    src/jit.cu
)
target_link_libraries(jit_compile_log_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_compile_log_test COMMAND jit_compile_log_test)


option(WARPDB_BUILD_PYTHON "Build Python bindings with pybind11" ON)

if(WARPDB_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        message(STATUS "Found pybind11 - building Python bindings")
        pybind11_add_module(pywarpdb bindings/python/pywarpdb.cpp)
        target_link_libraries(pywarpdb PRIVATE warpdb_lib)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME python_module_test
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py)
        set_tests_properties(python_module_test PROPERTIES
            DEPENDS pywarpdb
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:pywarpdb>")
    else()
        message(STATUS "pybind11 not found - Python bindings will not be built")
    endif()
endif()

# Test building without Apache Arrow available
add_test(
    NAME no_arrow_build
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/build_no_arrow.sh
)
