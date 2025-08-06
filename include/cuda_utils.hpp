#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(err)                                                         \
  do {                                                                          \
    cudaError_t err__ = (err);                                                  \
    if (err__ != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA Error: ") +                   \
                               cudaGetErrorString(err__));                      \
    }                                                                           \
  } while (0)
#endif

// Simple RAII wrapper for device memory
template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() : ptr_(nullptr) {}
  explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, sizeof(T) * count));
    }
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  T* get() const { return ptr_; }
private:
  T* ptr_;
};
