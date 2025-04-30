cmake_minimum_required(VERSION 3.18)

project(WarpDB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

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

target_link_libraries(warpdb PRIVATE /usr/lib/x86_64-linux-gnu/libnvrtc.so cuda)

