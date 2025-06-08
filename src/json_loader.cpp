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
  std::string line;
  while (std::getline(file, line)) {
    float price = 0.0f;
    int quantity = 0;
    // very simple JSON line parser assuming format {"price": X, "quantity": Y}
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
    host.price.push_back(price);
    host.quantity.push_back(quantity);
  }
  return host;
}

Table load_json_to_gpu(const std::string &filepath) {
  HostTable host = load_json_to_host(filepath);
  return upload_to_gpu(host);
}
