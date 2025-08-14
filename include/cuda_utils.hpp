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
    if (ptr_) CUDA_CHECK(cudaFree(ptr_));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) CUDA_CHECK(cudaFree(ptr_));
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  T* get() const { return ptr_; }
private:
  T* ptr_;
};

// RAII wrapper for untyped device pointers used by Table columns.
class ColumnDevicePtr {
public:
  ColumnDevicePtr() : ptr_(nullptr) {}
  explicit ColumnDevicePtr(void* ptr) : ptr_(ptr) {}
  ~ColumnDevicePtr() {
    if (ptr_) CUDA_CHECK(cudaFree(ptr_));
  }
  ColumnDevicePtr(const ColumnDevicePtr&) = delete;
  ColumnDevicePtr& operator=(const ColumnDevicePtr&) = delete;
  ColumnDevicePtr(ColumnDevicePtr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  ColumnDevicePtr& operator=(ColumnDevicePtr&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  void* get() const { return ptr_; }
  void reset(void* p = nullptr) {
    if (ptr_) CUDA_CHECK(cudaFree(ptr_));
    ptr_ = p;
  }

private:
  void* ptr_;
};
