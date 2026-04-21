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

std::string trim(const std::string &input) {
  auto begin = std::find_if(input.begin(), input.end(),
                            [](unsigned char ch) { return !std::isspace(ch); });
  auto end = std::find_if(input.rbegin(), input.rend(),
                          [](unsigned char ch) { return !std::isspace(ch); })
                 .base();
  if (begin >= end) return "";
  return std::string(begin, end);
}

bool try_parse_int32(const std::string &value, int32_t &out) {
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), out);
  return ec == std::errc() && ptr == value.data() + value.size();
}

bool try_parse_int64(const std::string &value, int64_t &out) {
  auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), out);
  return ec == std::errc() && ptr == value.data() + value.size();
}

template <typename FloatT>
bool try_parse_float(const std::string &value, FloatT &out) {
  auto [ptr, ec] =
      std::from_chars(value.data(), value.data() + value.size(), out,
                      std::chars_format::general);
  return ec == std::errc() && ptr == value.data() + value.size();
}

DataType infer_column_type(const std::vector<std::string> &values) {
  bool has_int64 = false;
  bool has_float = false;
  bool saw_non_empty = false;
  bool numeric_seen = false;

  for (const auto &v : values) {
    if (v.empty()) continue;
    saw_non_empty = true;

    int32_t i32 = 0;
    if (try_parse_int32(v, i32)) {
      numeric_seen = true;
      continue;
    }

    int64_t i64 = 0;
    if (try_parse_int64(v, i64)) {
      numeric_seen = true;
      has_int64 = true;
      continue;
    }

    double d = 0.0;
    if (try_parse_float(v, d)) {
      numeric_seen = true;
      has_float = true;
      continue;
    }

    if (!numeric_seen) return DataType::String;
  }

  if (!saw_non_empty) return DataType::Float32;
  if (has_float) return DataType::Float32;
  if (has_int64) return DataType::Int64;
  return DataType::Int32;
}

DataType widen_type(DataType current, DataType candidate) {
  if (current == candidate) return current;

  if (current == DataType::String || candidate == DataType::String)
    return DataType::String;

  const bool current_float =
      current == DataType::Float32 || current == DataType::Float64;
  const bool candidate_float =
      candidate == DataType::Float32 || candidate == DataType::Float64;

  if (current_float || candidate_float) {
    if (current == DataType::Float64 || candidate == DataType::Float64)
      return DataType::Float64;
    return DataType::Float32;
  }

  if (current == DataType::Int64 || candidate == DataType::Int64)
    return DataType::Int64;

  return DataType::Int32;
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

  bool inferred_schema = types.empty();
  std::vector<std::vector<std::string>> buffered_rows;
  if (inferred_schema) {
    std::string line;
    int row = 0;
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      ++row;
      const size_t delimiter_count =
          static_cast<size_t>(std::count(line.begin(), line.end(), ','));
      const size_t field_count = delimiter_count + 1;
      if (field_count != names.size()) {
        std::cerr << "Row " << row << " has " << field_count
                  << " fields but " << names.size() << " were expected"
                  << std::endl;
        if (policy == ParsePolicy::Strict) {
          throw std::runtime_error("Incorrect number of fields in row");
        }
        continue;
      }
      std::stringstream ss(line);
      std::string value;
      std::vector<std::string> parsed(names.size());
      for (size_t i = 0; i < names.size(); ++i) {
        if (!std::getline(ss, value, ',')) value.clear();
        parsed[i] = trim(value);
      }
      buffered_rows.push_back(std::move(parsed));
    }

    std::vector<DataType> inferred_types(names.size(), DataType::Float32);
    for (size_t i = 0; i < names.size(); ++i) {
      std::vector<std::string> col_values;
      col_values.reserve(buffered_rows.size());
      for (const auto &row : buffered_rows) col_values.push_back(row[i]);
      inferred_types[i] = infer_column_type(col_values);
    }
    types = inferred_types;
  }

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

  auto append_row = [&](const std::vector<std::string> &values, int row) {
    for (size_t i = 0; i < names.size(); ++i) {
      const std::string &value = values[i];
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
  };

  int row = 0;
  if (inferred_schema) {
    for (const auto &values : buffered_rows) {
      ++row;
      append_row(values, row);
    }
  } else {
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty())
        continue;
      ++row;
      const size_t delimiter_count =
          static_cast<size_t>(std::count(line.begin(), line.end(), ','));
      const size_t field_count = delimiter_count + 1;
      if (field_count != names.size()) {
        std::cerr << "Row " << row << " has " << field_count
                  << " fields but " << names.size() << " were expected"
                  << std::endl;
        if (policy == ParsePolicy::Strict) {
          throw std::runtime_error("Incorrect number of fields in row");
        }
        continue;
      }
      std::stringstream ss(line);
      std::string value;
      std::vector<std::string> parsed(names.size());
      for (size_t i = 0; i < names.size(); ++i) {
        if (!std::getline(ss, value, ',')) value.clear();
        parsed[i] = trim(value);
      }
      append_row(parsed, row);
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
      if (total_chars > 0) {
        CUDA_CHECK(cudaMalloc(&d_chars, total_chars));
      }
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
  HostTable host = load_csv_to_host(filepath, schema, policy);
  return upload_to_gpu(host);
}

Table load_csv_to_gpu(const std::string &filepath) {
  return load_csv_to_gpu(filepath, {}, ParsePolicy::Strict);
}

HostTable load_csv_chunk(std::istream &stream, int max_rows, bool &finished,
                         const std::vector<std::string> &column_names,
                         ParsePolicy policy, std::vector<DataType> *schema) {
  if (column_names.empty()) {
    throw std::runtime_error("Column names must be provided when loading CSV chunks");
  }

  std::vector<DataType> local_schema;
  std::vector<DataType> &types = schema ? *schema : local_schema;
  if (!types.empty() && types.size() != column_names.size()) {
    throw std::runtime_error("Schema size does not match column count");
  }

  finished = false;
  int count = 0;
  int input_row = 0;
  std::string line;
  std::vector<std::vector<std::string>> rows;
  while (count < max_rows) {
    if (!std::getline(stream, line)) {
      finished = true;
      break;
    }
    if (line.empty())
      continue;
    ++input_row;

    const size_t delimiter_count =
        static_cast<size_t>(std::count(line.begin(), line.end(), ','));
    const size_t field_count = delimiter_count + 1;
    if (field_count != column_names.size()) {
      std::cerr << "Row " << input_row << " has " << field_count
                << " fields but " << column_names.size() << " were expected"
                << std::endl;
      if (policy == ParsePolicy::Strict) {
        throw std::runtime_error("Incorrect number of fields in row");
      }
      continue;
    }

    std::stringstream ss(line);
    std::string val;
    std::vector<std::string> parsed_row(column_names.size());
    for (size_t i = 0; i < column_names.size(); ++i) {
      if (!std::getline(ss, val, ',')) val.clear();
      parsed_row[i] = trim(val);
    }
    rows.push_back(std::move(parsed_row));
    ++count;
  }

  if (rows.empty()) {
    finished = finished || stream.eof();
    if (stream.fail() && !stream.eof()) {
      throw std::runtime_error("Error reading CSV: partial line or I/O error");
    }
    return {};
  }

  std::vector<DataType> inferred_types(column_names.size(), DataType::Float32);
  for (size_t col = 0; col < column_names.size(); ++col) {
    std::vector<std::string> col_values;
    col_values.reserve(rows.size());
    for (const auto &r : rows) col_values.push_back(r[col]);
    inferred_types[col] = infer_column_type(col_values);
  }

  if (types.empty()) {
    types = inferred_types;
  } else {
    for (size_t col = 0; col < column_names.size(); ++col) {
      types[col] = widen_type(types[col], inferred_types[col]);
    }
  }

  HostTable table;
  table.columns.resize(column_names.size());
  for (size_t i = 0; i < column_names.size(); ++i) {
    table.columns[i].name = column_names[i];
    table.columns[i].type = types[i];
    switch (types[i]) {
    case DataType::Int32:
      table.columns[i].data = std::vector<int32_t>();
      break;
    case DataType::Int64:
      table.columns[i].data = std::vector<int64_t>();
      break;
    case DataType::Float32:
      table.columns[i].data = std::vector<float>();
      break;
    case DataType::Float64:
      table.columns[i].data = std::vector<double>();
      break;
    case DataType::String:
      table.columns[i].data = std::vector<std::string>();
      break;
    }
  }

  int row_idx = 0;
  for (const auto &row : rows) {
    for (size_t col = 0; col < column_names.size(); ++col) {
      const std::string &value = row[col];
      HostColumn &hcol = table.columns[col];

      switch (hcol.type) {
      case DataType::Int32: {
        int32_t parsed = 0;
        if (!try_parse_int32(value, parsed)) {
          std::cerr << "Failed to parse int32 value '" << value << "' at row "
                    << (row_idx + 1) << " column '" << hcol.name << "'"
                    << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<int32_t>>(hcol.data).push_back(0);
          } else {
            throw std::runtime_error("Invalid int32");
          }
        } else {
          std::get<std::vector<int32_t>>(hcol.data).push_back(parsed);
        }
        break;
      }
      case DataType::Int64: {
        int64_t parsed = 0;
        if (!try_parse_int64(value, parsed)) {
          std::cerr << "Failed to parse int64 value '" << value << "' at row "
                    << (row_idx + 1) << " column '" << hcol.name << "'"
                    << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<int64_t>>(hcol.data).push_back(0);
          } else {
            throw std::runtime_error("Invalid int64");
          }
        } else {
          std::get<std::vector<int64_t>>(hcol.data).push_back(parsed);
        }
        break;
      }
      case DataType::Float32: {
        float parsed = 0.0f;
        if (!try_parse_float(value, parsed)) {
          std::cerr << "Failed to parse float value '" << value << "' at row "
                    << (row_idx + 1) << " column '" << hcol.name << "'"
                    << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<float>>(hcol.data).push_back(0.0f);
          } else {
            throw std::runtime_error("Invalid float");
          }
        } else {
          std::get<std::vector<float>>(hcol.data).push_back(parsed);
        }
        break;
      }
      case DataType::Float64: {
        double parsed = 0.0;
        if (!try_parse_float(value, parsed)) {
          std::cerr << "Failed to parse double value '" << value << "' at row "
                    << (row_idx + 1) << " column '" << hcol.name << "'"
                    << std::endl;
          if (policy == ParsePolicy::Permissive) {
            std::get<std::vector<double>>(hcol.data).push_back(0.0);
          } else {
            throw std::runtime_error("Invalid double");
          }
        } else {
          std::get<std::vector<double>>(hcol.data).push_back(parsed);
        }
        break;
      }
      case DataType::String:
        std::get<std::vector<std::string>>(hcol.data).push_back(value);
        break;
      }
    }
    ++row_idx;
  }

  if (stream.fail() && !stream.eof()) {
    throw std::runtime_error("Error reading CSV: partial line or I/O error");
  }

  if (!finished)
    finished = stream.eof();

  return table;
}
