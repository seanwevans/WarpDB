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

Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

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

  return table;
}
