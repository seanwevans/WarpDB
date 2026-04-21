#pragma once

#include <string>
#include <unordered_map>

#include "csv_loader.hpp" // for HostTable and Table

HostTable load_json_to_host(const std::string &filepath,
                            ParsePolicy policy = ParsePolicy::Strict);
HostTable load_json_to_host(
    const std::string &filepath,
    const std::unordered_map<std::string, DataType> &schema,
    ParsePolicy policy = ParsePolicy::Strict);

Table load_json_to_gpu(const std::string &filepath,
                       ParsePolicy policy = ParsePolicy::Strict);
Table load_json_to_gpu(const std::string &filepath,
                       const std::unordered_map<std::string, DataType> &schema,
                       ParsePolicy policy = ParsePolicy::Strict);
