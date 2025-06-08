#include "csv_loader.hpp"
#ifdef USE_ARROW
#include "arrow_loader.hpp"
#endif
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <arrow/cuda/api.h>
#endif

#include <algorithm>


#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


Table load_csv_to_gpu(const std::string &filepath) {
#ifdef USE_ARROW
  ArrowTable atable = load_csv_arrow(filepath);
  Table table{atable.d_price, atable.d_quantity,
              static_cast<int>(atable.num_rows)};
  return table;
#else


Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema) {

HostTable load_csv_to_host(const std::string &filepath) {


  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }


  std::string line;
  std::vector<float> h_price;
  std::vector<int> h_quantity;
  TableStats stats;
  bool first_price = true;
  bool first_quantity = true;

  std::string line;
  // Read header
  std::getline(file, line);
  std::vector<std::string> headers;
  std::stringstream header_ss(line);
  std::string cell;
  while (std::getline(header_ss, cell, ',')) {
    headers.push_back(cell);
  }
  const int num_cols = headers.size();

  std::vector<std::vector<std::string>> raw_cols(num_cols);
  while (std::getline(file, line)) {

    std::stringstream ss(line);
    for (int c = 0; c < num_cols; ++c) {
      std::string val;
      std::getline(ss, val, ',');
      raw_cols[c].push_back(val);
    }
  }

  const int N = raw_cols[0].size();

    std::istringstream ss(line);
    std::string price_str, qty_str;
    std::getline(ss, price_str, ',');
    std::getline(ss, qty_str, ',');

    if (!price_str.empty()) {
      float price_val = std::stof(price_str);
      h_price.push_back(price_val);
      if (first_price) {
        stats.price.min = stats.price.max = price_val;
        first_price = false;
      } else {
        stats.price.min = std::min(stats.price.min, price_val);
        stats.price.max = std::max(stats.price.max, price_val);
      }
    } else {
      h_price.push_back(0.0f);
      stats.price.null_count++;
    }

    if (!qty_str.empty()) {
      int qty_val = std::stoi(qty_str);
      h_quantity.push_back(qty_val);
      if (first_quantity) {
        stats.quantity.min = stats.quantity.max = qty_val;
        first_quantity = false;
      } else {
        stats.quantity.min = std::min(stats.quantity.min, qty_val);
        stats.quantity.max = std::max(stats.quantity.max, qty_val);
      }
    } else {
      h_quantity.push_back(0);
      stats.quantity.null_count++;
    }
  }

  HostTable table;
  table.price = std::move(h_price);
  table.quantity = std::move(h_quantity);
  return table;
}

Table upload_to_gpu(const HostTable &host) {
  const int N = host.num_rows();

  std::vector<DataType> types = schema;
  if (types.empty()) {
    types.resize(num_cols);
    for (int c = 0; c < num_cols; ++c) {
      bool is_float = false;
      for (const auto &v : raw_cols[c]) {
        if (v.find('.') != std::string::npos || v.find('e') != std::string::npos ||
            v.find('E') != std::string::npos) {
          is_float = true;
          break;
        }
      }
      types[c] = is_float ? DataType::Float32 : DataType::Int32;
    }
  }


  Table table;
  table.num_rows = N;

  std::copy(host.price.begin(), host.price.end(), h_price_pinned);
  std::copy(host.quantity.begin(), host.quantity.end(), h_quantity_pinned);


  for (int c = 0; c < num_cols; ++c) {
    ColumnDesc desc;
    desc.name = headers[c];
    desc.type = types[c];
    desc.length = N;

    if (desc.type == DataType::Float32) {
      std::vector<float> host(N);
      for (int i = 0; i < N; ++i)
        host[i] = std::stof(raw_cols[c][i]);
      float *h_pinned;
      CUDA_CHECK(cudaMallocHost((void **)&h_pinned, sizeof(float) * N));
      std::copy(host.begin(), host.end(), h_pinned);
      float *d_ptr;
      CUDA_CHECK(cudaMalloc((void **)&d_ptr, sizeof(float) * N));
      CUDA_CHECK(
          cudaMemcpy(d_ptr, h_pinned, sizeof(float) * N, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaFreeHost(h_pinned));
      desc.device_ptr = d_ptr;
    } else {
      std::vector<int> host(N);
      for (int i = 0; i < N; ++i)
        host[i] = std::stoi(raw_cols[c][i]);
      int *h_pinned;
      CUDA_CHECK(cudaMallocHost((void **)&h_pinned, sizeof(int) * N));
      std::copy(host.begin(), host.end(), h_pinned);
      int *d_ptr;
      CUDA_CHECK(cudaMalloc((void **)&d_ptr, sizeof(int) * N));
      CUDA_CHECK(
          cudaMemcpy(d_ptr, h_pinned, sizeof(int) * N, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaFreeHost(h_pinned));
      desc.device_ptr = d_ptr;
    }

    table.columns.push_back(desc);
  }

  Table table = {d_price, d_quantity, N, stats};
  std::cout << "[Stats] price min=" << stats.price.min
            << " max=" << stats.price.max
            << " nulls=" << stats.price.null_count << "\n";
  std::cout << "[Stats] quantity min=" << stats.quantity.min
            << " max=" << stats.quantity.max
            << " nulls=" << stats.quantity.null_count << "\n";

  return table;
#endif
}

Table load_csv_to_gpu(const std::string &filepath) {
  HostTable host = load_csv_to_host(filepath);
  return upload_to_gpu(host);
}
