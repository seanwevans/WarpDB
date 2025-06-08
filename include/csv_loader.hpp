#pragma once
#include <string>
#include <vector>

enum class DataType { Int32, Float32 };

struct ColumnDesc {
  std::string name;
  DataType type;
  void *device_ptr;
  int length;
};

struct Table {
  std::vector<ColumnDesc> columns;
  int num_rows;

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
