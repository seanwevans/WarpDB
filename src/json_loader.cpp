#include "json_loader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cuda_utils.hpp"

HostTable load_json_to_host(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

  HostTable host;
  host.columns = {{"price", DataType::Float32, std::vector<float>()},
                  {"quantity", DataType::Int32, std::vector<int32_t>()}};

  std::string line;
  size_t line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    try {
      auto j = nlohmann::json::parse(line);
      if (!j.contains("price") || !j.contains("quantity")) {
        throw std::runtime_error("Missing required fields");
      }
      float price = j.at("price").get<float>();
      int quantity = j.at("quantity").get<int>();
      std::get<std::vector<float>>(host.columns[0].data).push_back(price);
      std::get<std::vector<int32_t>>(host.columns[1].data).push_back(quantity);
    } catch (const std::exception &e) {
      std::cerr << "Error parsing JSON line " << line_num << ": " << e.what()
                << std::endl;
      throw std::runtime_error("Failed to parse JSON line");
    }
  }
  return host;
}

Table load_json_to_gpu(const std::string &filepath) {
  HostTable host = load_json_to_host(filepath);
  return upload_to_gpu(host);
}
