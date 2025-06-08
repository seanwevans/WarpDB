#pragma once

#include <memory>
#include <string>
#include <stdexcept>
#include "csv_loader.hpp" // for Table structure
#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/cuda/api.h>
#endif

struct ArrowTable {
#ifdef USE_ARROW
    std::shared_ptr<arrow::cuda::CudaBuffer> d_price;
    std::shared_ptr<arrow::cuda::CudaBuffer> d_quantity;
    int64_t num_rows;
#endif
};

#ifdef USE_ARROW
ArrowTable load_csv_arrow(const std::string &filepath);
Table load_parquet_to_gpu(const std::string &filepath);
Table load_arrow_to_gpu(const std::string &filepath);
Table load_orc_to_gpu(const std::string &filepath);
#else
inline ArrowTable load_csv_arrow(const std::string &) {
    throw std::runtime_error("Arrow support not available");
}
inline Table load_parquet_to_gpu(const std::string &) {
    throw std::runtime_error("Arrow support not available");
}
inline Table load_arrow_to_gpu(const std::string &) {
    throw std::runtime_error("Arrow support not available");
}
inline Table load_orc_to_gpu(const std::string &) {
    throw std::runtime_error("Arrow support not available");
}
#endif

