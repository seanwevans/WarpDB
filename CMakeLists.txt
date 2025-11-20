cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

# Avoid passing system headers with '-isystem' which may interfere
# with '#include_next' in the standard library on some platforms.
set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-I")

enable_testing()

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

find_package(Arrow QUIET COMPONENTS cuda)

include(FetchContent)
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

include_directories(include)

set(WARPDB_SRC
    src/main.cu
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cpp
    src/arrow_utils.cpp
    src/multi_gpu_utils.cpp
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
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver ${Arrow_LIBRARIES} nlohmann_json::nlohmann_json)
else()
    target_link_libraries(warpdb PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver nlohmann_json::nlohmann_json)
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

add_executable(parsing_error_tests
    tests/parsing_error_tests.cpp
    src/expression.cpp
)

add_test(NAME parsing_error_tests COMMAND parsing_error_tests)


add_executable(sql_features_test
    tests/sql_features_test.cpp
)
target_link_libraries(sql_features_test PRIVATE warpdb_lib)

add_test(NAME sql_features_test COMMAND sql_features_test)

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

add_executable(csv_loader_chunk_error_test
    tests/csv_loader_chunk_error_test.cpp)
target_link_libraries(csv_loader_chunk_error_test PRIVATE warpdb_lib)
add_test(NAME csv_loader_chunk_error_test COMMAND csv_loader_chunk_error_test)

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

add_executable(jit_where_zero_test
    tests/jit_where_zero_test.cpp
    src/jit.cpp
)
target_link_libraries(jit_where_zero_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME jit_where_zero_test COMMAND jit_where_zero_test)

add_executable(optimizer_test
    tests/optimizer_test.cpp
    src/expression.cpp
    src/optimizer.cpp
)
target_link_libraries(optimizer_test PRIVATE CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)
add_test(NAME optimizer_test COMMAND optimizer_test)

add_library(warpdb_lib STATIC
    src/warpdb.cpp
    src/csv_loader.cpp
    src/json_loader.cpp
    src/expression.cpp
    src/jit.cpp
    src/arrow_utils.cpp
    src/multi_gpu_utils.cpp
)

set_target_properties(warpdb_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::nvrtc CUDA::cuda_driver)

if(Arrow_FOUND)
    target_sources(warpdb_lib PRIVATE src/arrow_loader.cpp)
endif()
set_target_properties(warpdb_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
