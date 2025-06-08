#include "json_loader.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(err) \
  do { \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
      exit(1); \
    } \
  } while (0)

HostTable load_json_to_host(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

  HostTable host;
  host.columns = {
      {"price", DataType::Float32, std::vector<float>()},
      {"quantity", DataType::Int32, std::vector<int32_t>()}};

  std::string line;
  while (std::getline(file, line)) {
    float price = 0.0f;
    int quantity = 0;
    size_t p = line.find("\"price\"");
    size_t q = line.find("\"quantity\"");
    if (p == std::string::npos || q == std::string::npos)
      continue;
    p = line.find(':', p);
    q = line.find(':', q);
    if (p == std::string::npos || q == std::string::npos)
      continue;
    std::stringstream ss1(line.substr(p + 1));
    ss1 >> price;
    std::stringstream ss2(line.substr(q + 1));
    ss2 >> quantity;
    std::get<std::vector<float>>(host.columns[0].data).push_back(price);
    std::get<std::vector<int32_t>>(host.columns[1].data).push_back(quantity);
  }
  return host;
}

Table load_json_to_gpu(const std::string &filepath) {
  HostTable host = load_json_to_host(filepath);
  return upload_to_gpu(host);
}
