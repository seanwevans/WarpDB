#pragma once
#include <string>
#include <vector>

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
  float *d_price; // Device pointers
  int *d_quantity;
  int num_rows;
  TableStats stats; // basic column statistics
};

struct HostTable {
  std::vector<float> price;
  std::vector<int> quantity;
  int num_rows() const { return static_cast<int>(price.size()); }
};

HostTable load_csv_to_host(const std::string &filepath);
Table upload_to_gpu(const HostTable &table);
Table load_csv_to_gpu(const std::string &filepath);
