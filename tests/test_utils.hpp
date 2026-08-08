#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <utility>

#include "csv_loader.hpp"

// Shared helpers for the test suite.
//
// Many WarpDB tests exercise the GPU (uploading columns, launching JIT
// kernels, etc.) and therefore require a CUDA-capable device at runtime.
// Continuous integration frequently runs on machines without a GPU, so those
// tests should skip cleanly instead of aborting. Call this helper at the top of
// a GPU-dependent test's main():
//
//     if (warpdb_skip_if_no_cuda("my_test")) return 0;
//
// It returns true (after printing a skip notice) when no CUDA device is
// available, and false otherwise, leaving GPU-present behavior unchanged.
inline bool warpdb_skip_if_no_cuda(const char *test_name) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::printf("[SKIP] %s: no CUDA device available\n", test_name);
    return true;
  }
  return false;
}

// Build a device-resident numeric column for tests. ColumnDesc holds a
// move-only ColumnDevicePtr (with an explicit void* constructor) plus a
// separate string_data buffer, so it can't be aggregate-initialized from a raw
// pointer. This helper takes ownership of `device_ptr` (freed when the column
// is destroyed).
inline ColumnDesc warpdb_make_device_column(std::string name, DataType type,
                                            void *device_ptr, int length) {
  ColumnDesc col;
  col.name = std::move(name);
  col.type = type;
  col.device_ptr = ColumnDevicePtr(device_ptr);
  col.length = length;
  return col;
}
