cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_executable(warpdb
    src/main.cu
    src/csv_loader.cpp
    src/expression.cpp
    src/jit.cpp
)

set_target_properties(warpdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(warpdb PRIVATE CUDA::nvrtc CUDA::cuda_driver)

add_executable(expression_test
    tests/test_expression.cpp
    src/expression.cpp
)

add_test(NAME expression_test COMMAND expression_test)

add_library(warpdb_lib STATIC
    src/warpdb.cpp
    src/csv_loader.cpp
    src/expression.cpp
    src/jit.cpp
)
set_target_properties(warpdb_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(warpdb_lib PUBLIC CUDA::nvrtc CUDA::cuda_driver)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pywarpdb bindings/python/pywarpdb.cpp)
    target_link_libraries(pywarpdb PRIVATE warpdb_lib)
endif()

