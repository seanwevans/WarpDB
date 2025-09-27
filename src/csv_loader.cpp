#include "csv_loader.hpp"
#ifdef USE_ARROW
#include "arrow_loader.hpp"
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <charconv>
#include <system_error>

#include <cstring>

#if defined(__cpp_lib_unreachable)
#include <utility>
#endif
#include <cctype>

#include "cuda_utils.hpp"

#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <arrow/cuda/api.h>
#endif

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
#if defined(__cpp_lib_unreachable)
  std::unreachable();
#else
  throw std::runtime_error("Unknown DataType");
#endif
}

} // namespace

HostTable load_csv_to_host(const std::string &filepath,
                           const std::vector<DataType> &schema,
                           ParsePolicy policy) {
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
  int row = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    ++row;
    std::stringstream ss(line);
    std::string value;
    for (size_t i = 0; i < names.size(); ++i) {
      if (!std::getline(ss, value, ',')) value.clear();
      value.erase(value.begin(),
                  std::find_if(value.begin(), value.end(),
                               [](unsigned char ch) { return !std::isspace(ch); }));
      value.erase(std::find_if(value.rbegin(), value.rend(),
                               [](unsigned char ch) { return !std::isspace(ch); })
                      .base(),
                  value.end());
      HostColumn &col = host.columns[i];
      switch (col.type) {
      case DataType::Int32: {
        int32_t parsed = 0;
        auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (ec != std::errc() || ptr != value.data() + value.size()) {
          std::error_code code = std::make_error_code(ec);
          std::cerr << "Failed to parse int32 value '" << value
                    << "' at row " << row << " column '" << col.name
                    << "': " << code.message() << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<int32_t>>(col.data).push_back(0);
          } else {
            throw std::runtime_error("Invalid int32");
          }
        } else {
          std::get<std::vector<int32_t>>(col.data).push_back(parsed);
        }
        break;
      }
      case DataType::Int64: {
        int64_t parsed = 0;
        auto [ptr, ec] =
            std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (ec != std::errc() || ptr != value.data() + value.size()) {
          std::error_code code = std::make_error_code(ec);
          std::cerr << "Failed to parse int64 value '" << value
                    << "' at row " << row << " column '" << col.name
                    << "': " << code.message() << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<int64_t>>(col.data).push_back(0);
          } else {
            throw std::runtime_error("Invalid int64");
          }
        } else {
          std::get<std::vector<int64_t>>(col.data).push_back(parsed);
        }
        break;
      }
      case DataType::Float32: {
        float parsed = 0.0f;
        auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                         parsed, std::chars_format::general);
        if (ec != std::errc() || ptr != value.data() + value.size()) {
          std::error_code code = std::make_error_code(ec);
          std::cerr << "Failed to parse float value '" << value
                    << "' at row " << row << " column '" << col.name
                    << "': " << code.message() << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<float>>(col.data).push_back(0.0f);
          } else {
            throw std::runtime_error("Invalid float");
          }
        } else {
          std::get<std::vector<float>>(col.data).push_back(parsed);
        }
        break;
      }
      case DataType::Float64: {
        double parsed = 0.0;
        auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                         parsed, std::chars_format::general);
        if (ec != std::errc() || ptr != value.data() + value.size()) {
          std::error_code code = std::make_error_code(ec);
          std::cerr << "Failed to parse double value '" << value
                    << "' at row " << row << " column '" << col.name
                    << "': " << code.message() << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<double>>(col.data).push_back(0.0);
          } else {
            throw std::runtime_error("Invalid double");
          }
        } else {
          std::get<std::vector<double>>(col.data).push_back(parsed);
        }
        break;
      }
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
    ColumnDesc col;
    col.name = hcol.name;
    col.type = hcol.type;
    col.length = N;

    if (hcol.type != DataType::String) {
      void *d_ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&d_ptr, dtype_size(hcol.type) * N));
      col.device_ptr.reset(d_ptr);

      if (hcol.type == DataType::Int32) {
        const auto &vec = std::get<std::vector<int32_t>>(hcol.data);
        CUDA_CHECK(cudaMemcpy(col.device_ptr.get(), vec.data(),
                              sizeof(int32_t) * N, cudaMemcpyHostToDevice));
      } else if (hcol.type == DataType::Int64) {
        const auto &vec = std::get<std::vector<int64_t>>(hcol.data);
        CUDA_CHECK(cudaMemcpy(col.device_ptr.get(), vec.data(),
                              sizeof(int64_t) * N, cudaMemcpyHostToDevice));
      } else if (hcol.type == DataType::Float32) {
        const auto &vec = std::get<std::vector<float>>(hcol.data);
        CUDA_CHECK(cudaMemcpy(col.device_ptr.get(), vec.data(),
                              sizeof(float) * N, cudaMemcpyHostToDevice));
      } else if (hcol.type == DataType::Float64) {
        const auto &vec = std::get<std::vector<double>>(hcol.data);
        CUDA_CHECK(cudaMemcpy(col.device_ptr.get(), vec.data(),
                              sizeof(double) * N, cudaMemcpyHostToDevice));
      }
    } else {
      const auto &vec = std::get<std::vector<std::string>>(hcol.data);
      std::vector<int32_t> offsets(N + 1, 0);
      size_t total_chars = 0;
      for (int i = 0; i < N; ++i) {
        offsets[i] = static_cast<int32_t>(total_chars);
        total_chars += vec[i].size();
      }
      offsets[N] = static_cast<int32_t>(total_chars);

      std::vector<char> chars(total_chars);
      for (int i = 0; i < N; ++i) {
        std::memcpy(chars.data() + offsets[i], vec[i].data(), vec[i].size());
      }

      void *d_offsets = nullptr;
      void *d_chars = nullptr;
      CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int32_t) * (N + 1)));
      CUDA_CHECK(cudaMalloc(&d_chars, total_chars));
      CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(),
                            sizeof(int32_t) * (N + 1), cudaMemcpyHostToDevice));
      if (total_chars > 0) {
        CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), total_chars,
                              cudaMemcpyHostToDevice));
      }
      col.device_ptr.reset(d_offsets);
      col.string_data.reset(d_chars);
    }

    table.columns.push_back(std::move(col));
  }

  return table;
}

Table load_csv_to_gpu(const std::string &filepath,
                      const std::vector<DataType> &schema,
                      ParsePolicy policy) {
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
  HostTable host = load_csv_to_host(filepath, schema, policy);
  return upload_to_gpu(host);
}

Table load_csv_to_gpu(const std::string &filepath) {
  return load_csv_to_gpu(filepath, {}, ParsePolicy::Strict);
}

HostTable load_csv_chunk(std::istream &stream, int max_rows, bool &finished,
                         const std::vector<std::string> &column_names,
                         ParsePolicy policy) {
  if (column_names.empty()) {
    throw std::runtime_error("Column names must be provided when loading CSV chunks");
  }

  HostTable table;
  table.columns.resize(column_names.size());
  for (size_t i = 0; i < column_names.size(); ++i) {
    table.columns[i].name = column_names[i];
    table.columns[i].type = DataType::Float32;
    table.columns[i].data = std::vector<float>();
  }

  finished = false;
  int count = 0;
  std::string line;
  while (count < max_rows) {
    if (!std::getline(stream, line)) {
      finished = true;
      break;
    }
    if (line.empty())
      continue;

    std::stringstream ss(line);
    std::string val;
    for (size_t i = 0; i < column_names.size(); ++i) {
      if (!std::getline(ss, val, ','))
        val.clear();
      val.erase(val.begin(),
                std::find_if(val.begin(), val.end(),
                             [](unsigned char ch) { return !std::isspace(ch); }));
      val.erase(std::find_if(val.rbegin(), val.rend(),
                             [](unsigned char ch) { return !std::isspace(ch); })
                    .base(),
                val.end());
      float parsed = 0.0f;
      auto [ptr, ec] =
          std::from_chars(val.data(), val.data() + val.size(), parsed,
                          std::chars_format::general);
      if (ec != std::errc() || ptr != val.data() + val.size()) {
        std::error_code code = std::make_error_code(ec);
        std::cerr << "Failed to parse float value '" << val
                  << "' at row " << (count + 1) << " column '"
                  << table.columns[i].name << "': " << code.message()
                  << std::endl;
        if (policy == ParsePolicy::Permissive) {
          std::get<std::vector<float>>(table.columns[i].data).push_back(0.0f);
        } else {
          throw std::runtime_error("Invalid float");
        }
      } else {
        std::get<std::vector<float>>(table.columns[i].data).push_back(parsed);
      }
    }
    ++count;
  }

  if (stream.fail() && !stream.eof()) {
    throw std::runtime_error("Error reading CSV: partial line or I/O error");
  }

  if (!finished)
    finished = stream.eof();

  return table;
}
