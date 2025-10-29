#pragma once
#include <string>
#include <memory>
#include "csv_loader.hpp"
#include "expression.hpp"

void execute_query_optimized(const std::string &expr_part,
                             const std::string &where_part, Table &table);

// Populate table statistics by copying the required columns from device to
// host memory. Returns true when statistics for the supported columns were
// gathered, or false if they are unavailable (e.g. missing columns).
bool compute_table_stats(const Table &table, TableStats &stats);

// Analyze a filter condition to determine if it is always true or always
// false based on table statistics. The AST is assumed to represent a boolean
// expression. If the expression can be evaluated statically, either
// `always_true` or `always_false` will be set to true.
void analyze_condition(const ASTNode *node, const TableStats &stats,
                       bool &always_true, bool &always_false);
