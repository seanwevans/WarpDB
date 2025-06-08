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

size_t dtype_size(DataType t) {
  switch (t) {
  case DataType::Int32:
    return sizeof(int32_t);
  case DataType::Int64:
    return sizeof(int64_t);
  case DataType::Float32:
    return sizeof(float);
  case DataType::Float64:
    return sizeof(double);
  case DataType::String:
    return sizeof(char *); // unused for now
  }
  return 0;
}

} // namespace

HostTable load_csv_to_host(const std::string &filepath,
                           const std::vector<DataType> &schema) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

  std::string header_line;
  if (!std::getline(file, header_line))
    throw std::runtime_error("Empty CSV file");
  std::stringstream header_ss(header_line);
  std::vector<std::string> names;
  std::string col;
  while (std::getline(header_ss, col, ',')) names.push_back(col);

  std::vector<DataType> types = schema;
  if (!types.empty() && types.size() != names.size())
    throw std::runtime_error("Schema size does not match column count");
  if (types.empty()) types.assign(names.size(), DataType::Float32);

  HostTable host;
  host.columns.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    host.columns[i].name = names[i];
    host.columns[i].type = types[i];
    switch (types[i]) {
    case DataType::Int32:
      host.columns[i].data = std::vector<int32_t>();
      break;
    case DataType::Int64:
      host.columns[i].data = std::vector<int64_t>();
      break;
    case DataType::Float32:
      host.columns[i].data = std::vector<float>();
      break;
    case DataType::Float64:
      host.columns[i].data = std::vector<double>();
      break;
    case DataType::String:
      host.columns[i].data = std::vector<std::string>();
      break;
    }
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string value;
    for (size_t i = 0; i < names.size(); ++i) {
      if (!std::getline(ss, value, ',')) value.clear();
      HostColumn &col = host.columns[i];
      switch (col.type) {
      case DataType::Int32:
        std::get<std::vector<int32_t>>(col.data).push_back(std::stoi(value));
        break;
      case DataType::Int64:
        std::get<std::vector<int64_t>>(col.data).push_back(std::stoll(value));
        break;
      case DataType::Float32:
        std::get<std::vector<float>>(col.data).push_back(std::stof(value));
        break;
      case DataType::Float64:
        std::get<std::vector<double>>(col.data).push_back(std::stod(value));
        break;
      case DataType::String:
        std::get<std::vector<std::string>>(col.data).push_back(value);
        break;
      }
    }
  }

  return host;
}

Table upload_to_gpu(const HostTable &host) {
  Table table;
  table.num_rows = host.num_rows();

  for (const auto &hcol : host.columns) {
    int N = host.num_rows();
    void *d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, dtype_size(hcol.type) * N));

    if (hcol.type == DataType::Int32) {
      const auto &vec = std::get<std::vector<int32_t>>(hcol.data);
      CUDA_CHECK(cudaMemcpy(d_ptr, vec.data(), sizeof(int32_t) * N,
                            cudaMemcpyHostToDevice));
    } else if (hcol.type == DataType::Int64) {
      const auto &vec = std::get<std::vector<int64_t>>(hcol.data);
      CUDA_CHECK(cudaMemcpy(d_ptr, vec.data(), sizeof(int64_t) * N,
                            cudaMemcpyHostToDevice));
    } else if (hcol.type == DataType::Float32) {
      const auto &vec = std::get<std::vector<float>>(hcol.data);
      CUDA_CHECK(cudaMemcpy(d_ptr, vec.data(), sizeof(float) * N,
                            cudaMemcpyHostToDevice));
    } else if (hcol.type == DataType::Float64) {
      const auto &vec = std::get<std::vector<double>>(hcol.data);
      CUDA_CHECK(cudaMemcpy(d_ptr, vec.data(), sizeof(double) * N,
                            cudaMemcpyHostToDevice));
    } else if (hcol.type == DataType::String) {
      // string columns not supported on GPU yet
      CUDA_CHECK(cudaFree(d_ptr));
      d_ptr = nullptr;
    }

    table.columns.push_back({hcol.name, hcol.type, d_ptr, N});
  }

  return table;
}

Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema) {
#ifdef USE_ARROW
  if (schema.empty()) {
    ArrowTable atable = load_csv_arrow(filepath);
    Table table;
    table.num_rows = static_cast<int>(atable.num_rows);
    table.columns.push_back({"price", DataType::Float32,
                             (void *)atable.d_price->address(), table.num_rows});
    table.columns.push_back({"quantity", DataType::Int32,
                             (void *)atable.d_quantity->address(),
                             table.num_rows});
    return table;
  }
#endif
  HostTable host = load_csv_to_host(filepath, schema);
  return upload_to_gpu(host);
}

Table load_csv_to_gpu(const std::string &filepath) {
  return load_csv_to_gpu(filepath, {});
}

HostTable load_csv_chunk(std::istream &stream, int max_rows, bool &finished) {
  std::string header;
  std::streampos pos = stream.tellg();
  if (!(stream >> header)) {
    finished = true;
    return {};
  }
  stream.seekg(pos);

  std::getline(stream, header);
  std::stringstream hs(header);
  std::vector<std::string> names;
  std::string tmp;
  while (std::getline(hs, tmp, ',')) names.push_back(tmp);

  HostTable table;
  table.columns.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    table.columns[i].name = names[i];
    table.columns[i].type = DataType::Float32;
    table.columns[i].data = std::vector<float>();
  }

  int count = 0;
  std::string line;
  while (count < max_rows && std::getline(stream, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string val;
    for (size_t i = 0; i < names.size(); ++i) {
      if (!std::getline(ss, val, ',')) val.clear();
      std::get<std::vector<float>>(table.columns[i].data).push_back(std::stof(val));
    }
    ++count;
  }
  finished = !stream.good();
  return table;
}
