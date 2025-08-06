#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Helper macro to check CUDA runtime API calls.
// Throws std::runtime_error on failure so callers can handle or
// propagate exceptions as needed.
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA Error: ") +              \
                                     cudaGetErrorString(err__));                 \
        }                                                                        \
    } while (0)

