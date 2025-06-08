#include "csv_loader.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

Table load_csv_to_gpu(const std::string &filepath) {
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

  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
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

  const int N = h_price.size();

  // Allocate pinned host memory (optional for now)
  float *h_price_pinned;
  int *h_quantity_pinned;
  CUDA_CHECK(cudaMallocHost((void **)&h_price_pinned, sizeof(float) * N));
  CUDA_CHECK(cudaMallocHost((void **)&h_quantity_pinned, sizeof(int) * N));

  std::copy(h_price.begin(), h_price.end(), h_price_pinned);
  std::copy(h_quantity.begin(), h_quantity.end(), h_quantity_pinned);

  // Allocate device memory
  float *d_price;
  int *d_quantity;
  CUDA_CHECK(cudaMalloc((void **)&d_price, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc((void **)&d_quantity, sizeof(int) * N));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_price, h_price_pinned, sizeof(float) * N,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_quantity, h_quantity_pinned, sizeof(int) * N,
                        cudaMemcpyHostToDevice));

  // Free pinned host mem
  CUDA_CHECK(cudaFreeHost(h_price_pinned));
  CUDA_CHECK(cudaFreeHost(h_quantity_pinned));

  Table table = {d_price, d_quantity, N, stats};
  std::cout << "[Stats] price min=" << stats.price.min
            << " max=" << stats.price.max
            << " nulls=" << stats.price.null_count << "\n";
  std::cout << "[Stats] quantity min=" << stats.quantity.min
            << " max=" << stats.quantity.max
            << " nulls=" << stats.quantity.null_count << "\n";
  return table;
}
