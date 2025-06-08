#include "csv_loader.hpp"
#ifdef USE_ARROW
#include "arrow_loader.hpp"
#endif
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <arrow/cuda/api.h>
#endif

#define CUDA_CHECK(err) \
  do { \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
      exit(1); \
    } \
  } while (0)

namespace {
TableStats compute_stats(const HostTable &host) {
  TableStats stats;
  if (!host.price.empty()) {
    auto [min_it, max_it] = std::minmax_element(host.price.begin(), host.price.end());
    stats.price.min = *min_it;
    stats.price.max = *max_it;
  }
  if (!host.quantity.empty()) {
    auto [min_it, max_it] = std::minmax_element(host.quantity.begin(), host.quantity.end());
    stats.quantity.min = *min_it;
    stats.quantity.max = *max_it;
  }
  return stats;
}
} // namespace

HostTable load_csv_to_host(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

  std::string line;
  std::getline(file, line); // skip header

  std::vector<float> h_price;
  std::vector<int> h_quantity;

  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string price_str, qty_str;
    std::getline(ss, price_str, ',');
    std::getline(ss, qty_str, ',');
    h_price.push_back(std::stof(price_str));
    h_quantity.push_back(std::stoi(qty_str));
  }

  HostTable host;
  host.price = std::move(h_price);
  host.quantity = std::move(h_quantity);
  return host;
}

Table upload_to_gpu(const HostTable &host, const std::vector<DataType> &schema) {
  const int N = host.num_rows();

  float *h_price_pinned;
  int *h_quantity_pinned;
  CUDA_CHECK(cudaMallocHost((void **)&h_price_pinned, sizeof(float) * N));
  CUDA_CHECK(cudaMallocHost((void **)&h_quantity_pinned, sizeof(int) * N));
  std::copy(host.price.begin(), host.price.end(), h_price_pinned);
  std::copy(host.quantity.begin(), host.quantity.end(), h_quantity_pinned);

  float *d_price;
  int *d_quantity;
  CUDA_CHECK(cudaMalloc((void **)&d_price, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void **)&d_quantity, sizeof(int) * N));
  CUDA_CHECK(cudaMemcpy(d_price, h_price_pinned, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_quantity, h_quantity_pinned, sizeof(int) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaFreeHost(h_price_pinned));
  CUDA_CHECK(cudaFreeHost(h_quantity_pinned));

  ColumnDesc price_desc{"price", DataType::Float32, d_price, N};
  ColumnDesc qty_desc{"quantity", DataType::Int32, d_quantity, N};

  Table table;
  table.d_price = d_price;
  table.d_quantity = d_quantity;
  table.num_rows = N;
  table.columns = {price_desc, qty_desc};
  table.stats = compute_stats(host);
  (void)schema; // schema currently unused
  return table;
}

Table load_csv_to_gpu(const std::string &filepath, const std::vector<DataType> &schema) {
#ifdef USE_ARROW
  if (schema.empty()) {
    ArrowTable atable = load_csv_arrow(filepath);
    Table table{atable.d_price, atable.d_quantity, static_cast<int>(atable.num_rows)};
    ColumnDesc price_desc{"price", DataType::Float32, table.d_price, table.num_rows};
    ColumnDesc qty_desc{"quantity", DataType::Int32, table.d_quantity, table.num_rows};
    table.columns = {price_desc, qty_desc};
    return table;
  }
#endif
  HostTable host = load_csv_to_host(filepath);
  return upload_to_gpu(host, schema);
}

Table load_csv_to_gpu(const std::string &filepath) {
  return load_csv_to_gpu(filepath, {});
}
