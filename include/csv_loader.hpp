#pragma once
#include <string>
#include <vector>
#ifdef USE_ARROW
#include <memory>
#include <arrow/api.h>
#include <arrow/cuda/api.h>
#endif


enum class DataType { Int32, Float32 };

struct ColumnDesc {
  std::string name;
  DataType type;
  void *device_ptr;
  int length;
};

struct ColumnStatsFloat {
  float min = 0.0f;
  float max = 0.0f;
  int null_count = 0;
};

struct ColumnStatsInt {
  int min = 0;
  int max = 0;
  int null_count = 0;
};

struct TableStats {
  ColumnStatsFloat price;
  ColumnStatsInt quantity;

};

struct Table {

#ifdef USE_ARROW
  std::shared_ptr<arrow::cuda::CudaBuffer> d_price; // Device buffers
  std::shared_ptr<arrow::cuda::CudaBuffer> d_quantity;
#else
  float *d_price; // Device pointers
  int *d_quantity;
#endif

  std::vector<ColumnDesc> columns;

  int num_rows;

  TableStats stats; // basic column statistics

  template <typename T>
  T *get_column_ptr(const std::string &name) const {
    for (const auto &col : columns) {
      if (col.name == name)
        return static_cast<T *>(col.device_ptr);
    }
    return nullptr;
  }
};

Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema = {});

struct HostTable {
  std::vector<float> price;
  std::vector<int> quantity;
  int num_rows() const { return static_cast<int>(price.size()); }
};

HostTable load_csv_to_host(const std::string &filepath);
Table upload_to_gpu(const HostTable &table,
                    const std::vector<DataType> &schema);
Table upload_to_gpu(const HostTable &table);

