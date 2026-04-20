#pragma once
#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include "csv_loader.hpp"
#include "expression.hpp"

void execute_query_optimized(const std::string &expr_part,
                             const std::string &where_part, Table &table);

// Populate table statistics by copying numeric columns from device to host
// memory. Returns true when statistics for at least one numeric column were
// gathered, or false if they are unavailable (e.g. missing columns).
bool compute_table_stats(const Table &table, TableStats &stats);

// Analyze a filter condition to determine if it is always true or always
// false based on table statistics. The AST is assumed to represent a boolean
// expression. If the expression can be evaluated statically, either
// `always_true` or `always_false` will be set to true.
void analyze_condition(const ASTNode *node, const TableStats &stats,
                       bool &always_true, bool &always_false);

struct RelationStats {
    std::string relation;
    double row_count = 0.0;
    std::unordered_map<std::string, double> distinct_counts;
};

struct JoinPredicate {
    std::string left_relation;
    std::string left_column;
    std::string right_relation;
    std::string right_column;
};

struct JoinPlanStep {
    std::string relation;
    double estimated_rows = 0.0;
    std::optional<JoinPredicate> predicate;
};

struct JoinPlan {
    std::vector<JoinPlanStep> steps;
    double estimated_total_cost = 0.0;
};

// Extract `a.x = b.y` predicates from JOIN ... ON clauses.
std::vector<JoinPredicate> extract_equi_join_predicates(const QueryAST &ast);

// Estimate cardinality for an equality join between two inputs.
double estimate_equi_join_rows(double left_rows, double right_rows,
                               double left_distinct, double right_distinct);

// Build a greedy left-deep join plan using row count and NDV stats.
JoinPlan build_greedy_join_plan(
    const QueryAST &ast,
    const std::unordered_map<std::string, RelationStats> &stats_by_relation);
