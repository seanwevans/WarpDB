#pragma once

#include <memory>
#include <string>
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
#endif

#include <string>
#include "csv_loader.hpp" // for Table structure

Table load_parquet_to_gpu(const std::string &filepath);
Table load_arrow_to_gpu(const std::string &filepath);
Table load_orc_to_gpu(const std::string &filepath);

