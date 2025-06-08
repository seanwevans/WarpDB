#pragma once
#include <string>
#include "csv_loader.hpp" // for Table structure

Table load_parquet_to_gpu(const std::string &filepath);
Table load_arrow_to_gpu(const std::string &filepath);
Table load_orc_to_gpu(const std::string &filepath);
