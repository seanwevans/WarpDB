cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

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
    src/arrow_loader.cpp
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
    target_link_libraries(warpdb PRIVATE CUDA::nvrtc CUDA::cuda_driver ${Arrow_LIBRARIES})
else()
    target_link_libraries(warpdb PRIVATE CUDA::nvrtc CUDA::cuda_driver)
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

add_library(warpdb_lib STATIC
    src/warpdb.cpp
    src/csv_loader.cpp
    src/expression.cpp
    src/jit.cpp
    src/arrow_utils.cpp
)
set_target_properties(warpdb_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::nvrtc CUDA::cuda_driver)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pywarpdb bindings/python/pywarpdb.cpp)
    target_link_libraries(pywarpdb PRIVATE warpdb_lib)
endif()
