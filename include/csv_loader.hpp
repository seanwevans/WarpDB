#pragma once
#include <string>
#include <vector>
#ifdef USE_ARROW
#include <memory>
#include <arrow/api.h>
#include <arrow/cuda/api.h>
#endif


#include <variant>
#include "cuda_utils.hpp"

enum class DataType { Int32, Int64, Float32, Float64, String };

// Policy for handling conversion errors when parsing CSV values.
// Strict mode throws on any malformed numeric field, while Permissive mode
// logs the error and substitutes a default value.
enum class ParsePolicy { Strict, Permissive };

struct ColumnDesc {
  std::string name;
  DataType type;
  // For numeric columns, device_ptr points directly to the values array.
  // For string columns, device_ptr stores the offsets buffer (int32_t[N+1])
  // and string_data holds the concatenated character data.
  ColumnDevicePtr device_ptr;
  ColumnDevicePtr string_data;
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
        return static_cast<T *>(col.device_ptr.get());
    }
    return nullptr;
  }

  // Returns the character buffer for the given string column, or nullptr if
  // the column doesn't exist or isn't a string column.
  const char *get_string_data(const std::string &name) const {
    for (const auto &col : columns) {
      if (col.name == name)
        return static_cast<const char *>(col.string_data.get());
    }
    return nullptr;
  }
};

Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema = {},
                      ParsePolicy policy = ParsePolicy::Strict);

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
                           const std::vector<DataType> &schema = {},
                           ParsePolicy policy = ParsePolicy::Strict);
Table upload_to_gpu(const HostTable &table);

// Load at most `max_rows` CSV rows from an open input stream using the provided
// `column_names`. The header should be consumed by the caller so the loader
// does not re-read or skip the first data row. `finished` will be set to true
// when no more rows are available.
HostTable load_csv_chunk(std::istream &stream, int max_rows, bool &finished,
                         const std::vector<std::string> &column_names,
                         ParsePolicy policy = ParsePolicy::Strict);

