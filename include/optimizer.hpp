#pragma once
#include <string>
#include <memory>
#include "csv_loader.hpp"
#include "expression.hpp"

void execute_query_optimized(const std::string &expr_part,
                             const std::string &where_part, Table &table);
