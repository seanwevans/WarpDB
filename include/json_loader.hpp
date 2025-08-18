#pragma once
#include <string>
#include "csv_loader.hpp" // for HostTable and Table

HostTable load_json_to_host(const std::string &filepath,
                            ParsePolicy policy = ParsePolicy::Strict);
Table load_json_to_gpu(const std::string &filepath,
                       ParsePolicy policy = ParsePolicy::Strict);
