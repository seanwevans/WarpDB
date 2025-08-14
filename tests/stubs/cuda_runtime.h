#pragma once
#include <cstddef>

typedef int cudaError_t;
static constexpr cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind { cudaMemcpyHostToDevice = 0, cudaMemcpyDeviceToHost = 1 };

inline const char *cudaGetErrorString(cudaError_t) { return "cuda stub"; }
inline cudaError_t cudaMalloc(void **, std::size_t) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void *, const void *, std::size_t, cudaMemcpyKind) {
  return cudaSuccess;
}
inline cudaError_t cudaFree(void *) { return cudaSuccess; }
