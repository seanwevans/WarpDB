#include "json_loader.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "cuda_utils.hpp"

namespace {

using nlohmann::json;

DataType infer_json_type(const json &value) {
  if (value.is_number_integer() || value.is_number_unsigned()) {
    if (value.is_number_unsigned()) {
      const auto parsed = value.get<uint64_t>();
      if (parsed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return DataType::String;
      }
      return parsed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
                 ? DataType::Int64
                 : DataType::Int32;
    }

    const auto parsed = value.get<int64_t>();
    return (parsed > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
            parsed < static_cast<int64_t>(std::numeric_limits<int32_t>::min()))
               ? DataType::Int64
               : DataType::Int32;
  }

  if (value.is_number_float()) return DataType::Float32;
  if (value.is_boolean()) return DataType::Int32;
  if (value.is_string()) return DataType::String;
  return DataType::String;
}

DataType widen_type(DataType current, DataType candidate) {
  if (current == candidate) return current;

  if (current == DataType::String || candidate == DataType::String) {
    return DataType::String;
  }

  const bool current_float =
      current == DataType::Float32 || current == DataType::Float64;
  const bool candidate_float =
      candidate == DataType::Float32 || candidate == DataType::Float64;

  if (current_float || candidate_float) {
    if (current == DataType::Float64 || candidate == DataType::Float64) {
      return DataType::Float64;
    }
    return DataType::Float32;
  }

  if (current == DataType::Int64 || candidate == DataType::Int64) {
    return DataType::Int64;
  }

  return DataType::Int32;
}

HostTable init_host_table(const std::vector<std::pair<std::string, DataType>> &schema) {
  HostTable host;
  host.columns.reserve(schema.size());
  for (const auto &[name, type] : schema) {
    HostColumn col;
    col.name = name;
    col.type = type;
    switch (type) {
    case DataType::Int32:
      col.data = std::vector<int32_t>();
      break;
    case DataType::Int64:
      col.data = std::vector<int64_t>();
      break;
    case DataType::Float32:
      col.data = std::vector<float>();
      break;
    case DataType::Float64:
      col.data = std::vector<double>();
      break;
    case DataType::String:
      col.data = std::vector<std::string>();
      break;
    }
    host.columns.push_back(std::move(col));
  }
  return host;
}

bool parse_value_for_type(const json &value, DataType type, ColumnData &output) {
  try {
    switch (type) {
    case DataType::Int32: {
      int32_t parsed = 0;
      if (value.is_boolean()) {
        parsed = value.get<bool>() ? 1 : 0;
      } else if (value.is_number_integer()) {
        auto v = value.get<int64_t>();
        if (v > std::numeric_limits<int32_t>::max() ||
            v < std::numeric_limits<int32_t>::min()) {
          return false;
        }
        parsed = static_cast<int32_t>(v);
      } else if (value.is_number_unsigned()) {
        auto v = value.get<uint64_t>();
        if (v > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          return false;
        }
        parsed = static_cast<int32_t>(v);
      } else if (value.is_number_float()) {
        parsed = static_cast<int32_t>(value.get<double>());
      } else if (value.is_string()) {
        parsed = static_cast<int32_t>(std::stoll(value.get<std::string>()));
      } else {
        return false;
      }
      output = std::vector<int32_t>{parsed};
      return true;
    }
    case DataType::Int64: {
      int64_t parsed = 0;
      if (value.is_boolean()) {
        parsed = value.get<bool>() ? 1 : 0;
      } else if (value.is_number_integer()) {
        parsed = value.get<int64_t>();
      } else if (value.is_number_unsigned()) {
        auto v = value.get<uint64_t>();
        if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return false;
        }
        parsed = static_cast<int64_t>(v);
      } else if (value.is_number_float()) {
        parsed = static_cast<int64_t>(value.get<double>());
      } else if (value.is_string()) {
        parsed = std::stoll(value.get<std::string>());
      } else {
        return false;
      }
      output = std::vector<int64_t>{parsed};
      return true;
    }
    case DataType::Float32: {
      float parsed = 0.0f;
      if (value.is_boolean()) {
        parsed = value.get<bool>() ? 1.0f : 0.0f;
      } else if (value.is_number()) {
        parsed = value.get<float>();
      } else if (value.is_string()) {
        parsed = std::stof(value.get<std::string>());
      } else {
        return false;
      }
      output = std::vector<float>{parsed};
      return true;
    }
    case DataType::Float64: {
      double parsed = 0.0;
      if (value.is_boolean()) {
        parsed = value.get<bool>() ? 1.0 : 0.0;
      } else if (value.is_number()) {
        parsed = value.get<double>();
      } else if (value.is_string()) {
        parsed = std::stod(value.get<std::string>());
      } else {
        return false;
      }
      output = std::vector<double>{parsed};
      return true;
    }
    case DataType::String:
      if (value.is_string()) {
        output = std::vector<std::string>{value.get<std::string>()};
      } else {
        output = std::vector<std::string>{value.dump()};
      }
      return true;
    }
  } catch (const std::exception &) {
    return false;
  }

  return false;
}

void append_single_value(HostColumn &column, const ColumnData &single_value) {
  switch (column.type) {
  case DataType::Int32:
    std::get<std::vector<int32_t>>(column.data).push_back(
        std::get<std::vector<int32_t>>(single_value).front());
    break;
  case DataType::Int64:
    std::get<std::vector<int64_t>>(column.data).push_back(
        std::get<std::vector<int64_t>>(single_value).front());
    break;
  case DataType::Float32:
    std::get<std::vector<float>>(column.data).push_back(
        std::get<std::vector<float>>(single_value).front());
    break;
  case DataType::Float64:
    std::get<std::vector<double>>(column.data).push_back(
        std::get<std::vector<double>>(single_value).front());
    break;
  case DataType::String:
    std::get<std::vector<std::string>>(column.data).push_back(
        std::get<std::vector<std::string>>(single_value).front());
    break;
  }
}

} // namespace

HostTable load_json_to_host(const std::string &filepath, ParsePolicy policy) {
  return load_json_to_host(filepath, std::unordered_map<std::string, DataType>{},
                           policy);
}

HostTable load_json_to_host(
    const std::string &filepath,
    const std::unordered_map<std::string, DataType> &schema,
    ParsePolicy policy) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    throw std::runtime_error("Unable to open file");
  }

  std::vector<json> rows;
  std::string line;
  size_t line_num = 0;
  while (std::getline(file, line)) {
    ++line_num;
    if (line.empty()) continue;

    try {
      auto parsed = json::parse(line);
      if (!parsed.is_object()) {
        throw std::runtime_error("Expected JSON object per line");
      }
      rows.push_back(std::move(parsed));
    } catch (const std::exception &e) {
      std::cerr << "Error parsing JSON line " << line_num << ": " << e.what()
                << std::endl;
      if (policy == ParsePolicy::Strict) {
        throw std::runtime_error("Failed to parse JSON line");
      }
    }
  }

  std::vector<std::pair<std::string, DataType>> schema_entries;
  schema_entries.reserve(schema.size());

  if (!schema.empty()) {
    for (const auto &entry : schema) {
      schema_entries.push_back(entry);
    }
  } else if (!rows.empty()) {
    for (auto it = rows.front().begin(); it != rows.front().end(); ++it) {
      if (!it.value().is_null()) {
        schema_entries.push_back({it.key(), infer_json_type(it.value())});
      }
    }

    for (const auto &row : rows) {
      for (const auto &item : row.items()) {
        if (item.value().is_null()) continue;

        auto found = std::find_if(
            schema_entries.begin(), schema_entries.end(),
            [&](const auto &entry) { return entry.first == item.key(); });
        if (found == schema_entries.end()) {
          schema_entries.push_back({item.key(), infer_json_type(item.value())});
          continue;
        }

        found->second = widen_type(found->second, infer_json_type(item.value()));
      }
    }
  }

  HostTable host = init_host_table(schema_entries);

  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    std::vector<ColumnData> row_values(host.columns.size());
    bool row_valid = true;

    for (size_t c = 0; c < host.columns.size(); ++c) {
      const auto &col_name = host.columns[c].name;
      const auto &col_type = host.columns[c].type;

      if (!row.contains(col_name) || row.at(col_name).is_null()) {
        std::cerr << "Error parsing JSON line " << (i + 1)
                  << ": missing field '" << col_name << "'" << std::endl;
        row_valid = false;
        break;
      }

      if (!parse_value_for_type(row.at(col_name), col_type, row_values[c])) {
        std::cerr << "Error parsing JSON line " << (i + 1)
                  << ": invalid value for field '" << col_name << "'"
                  << std::endl;
        row_valid = false;
        break;
      }
    }

    if (!row_valid) {
      if (policy == ParsePolicy::Strict) {
        throw std::runtime_error("Failed to parse JSON line");
      }
      continue;
    }

    for (size_t c = 0; c < host.columns.size(); ++c) {
      append_single_value(host.columns[c], row_values[c]);
    }
  }

  return host;
}

Table load_json_to_gpu(const std::string &filepath, ParsePolicy policy) {
  HostTable host = load_json_to_host(filepath, policy);
  return upload_to_gpu(host);
}

Table load_json_to_gpu(const std::string &filepath,
                       const std::unordered_map<std::string, DataType> &schema,
                       ParsePolicy policy) {
  HostTable host = load_json_to_host(filepath, schema, policy);
  return upload_to_gpu(host);
}
