cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

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
    src/expression.cpp
    src/jit.cpp
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

