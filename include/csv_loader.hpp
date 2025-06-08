#pragma once
#include <string>
#include <vector>

struct Table {
  float *d_price; // Device pointers
  int *d_quantity;
  int num_rows;
};

struct HostTable {
  std::vector<float> price;
  std::vector<int> quantity;
  int num_rows() const { return static_cast<int>(price.size()); }
};

HostTable load_csv_to_host(const std::string &filepath);
Table upload_to_gpu(const HostTable &table);
Table load_csv_to_gpu(const std::string &filepath);
