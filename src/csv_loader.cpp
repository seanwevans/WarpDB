#include "csv_loader.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

HostTable load_csv_to_host(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }
  std::string line;
  std::vector<float> h_price;
  std::vector<int> h_quantity;

  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string price_str, qty_str;
    std::getline(ss, price_str, ',');
    std::getline(ss, qty_str, ',');

    h_price.push_back(std::stof(price_str));
    h_quantity.push_back(std::stoi(qty_str));
  }

  HostTable table;
  table.price = std::move(h_price);
  table.quantity = std::move(h_quantity);
  return table;
}

Table upload_to_gpu(const HostTable &host) {
  const int N = host.num_rows();

  // Allocate pinned host memory (optional for now)
  float *h_price_pinned;
  int *h_quantity_pinned;
  CUDA_CHECK(cudaMallocHost((void **)&h_price_pinned, sizeof(float) * N));
  CUDA_CHECK(cudaMallocHost((void **)&h_quantity_pinned, sizeof(int) * N));

  std::copy(host.price.begin(), host.price.end(), h_price_pinned);
  std::copy(host.quantity.begin(), host.quantity.end(), h_quantity_pinned);

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

  Table table = {d_price, d_quantity, N};
  return table;
}

Table load_csv_to_gpu(const std::string &filepath) {
  HostTable host = load_csv_to_host(filepath);
  return upload_to_gpu(host);
}
