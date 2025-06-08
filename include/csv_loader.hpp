#pragma once
#include <string>
#include <vector>
#ifdef USE_ARROW
#include <memory>
#include <arrow/api.h>
#include <arrow/cuda/api.h>
#endif

struct Table {
#ifdef USE_ARROW
  std::shared_ptr<arrow::cuda::CudaBuffer> d_price; // Device buffers
  std::shared_ptr<arrow::cuda::CudaBuffer> d_quantity;
#else
  float *d_price; // Device pointers
  int *d_quantity;
#endif
  int num_rows;
};

Table load_csv_to_gpu(const std::string &filepath);
