#pragma once
#include <string>
#include <vector>
#ifdef USE_ARROW
#include <memory>
#include <arrow/api.h>
#include <arrow/cuda/api.h>
#endif


#include <variant>

enum class DataType { Int32, Int64, Float32, Float64, String };

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
  std::vector<ColumnDesc> columns;
  int num_rows = 0;

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

using ColumnData = std::variant<std::vector<int32_t>, std::vector<int64_t>,
                                std::vector<float>, std::vector<double>,
                                std::vector<std::string>>;

struct HostColumn {
  std::string name;
  DataType type;
  ColumnData data;
};

struct HostTable {
  std::vector<HostColumn> columns;
  int num_rows() const {
    if (columns.empty()) return 0;
    return std::visit([](auto &&v) { return static_cast<int>(v.size()); },
                      columns[0].data);
  }
  const HostColumn *get_column(const std::string &name) const {
    for (const auto &c : columns)
      if (c.name == name) return &c;
    return nullptr;
  }
};

HostTable load_csv_to_host(const std::string &filepath,
                           const std::vector<DataType> &schema = {});
Table upload_to_gpu(const HostTable &table);

// Load at most `max_rows` CSV rows from an open input stream. `finished`
// will be set to true when no more rows are available.
HostTable load_csv_chunk(std::istream &stream, int max_rows, bool &finished);

