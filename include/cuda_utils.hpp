#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(err) \
  do { \
    if ((err) != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
  } while (0)
