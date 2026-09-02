#include "optimizer.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <unordered_set>

namespace {

double parse_constant(const std::string &val) {
    return std::stod(val);
}

const ColumnDesc *find_column(const Table &table, const std::string &name) {
    for (const auto &col : table.columns) {
        if (col.name == name)
            return &col;
    }
    return nullptr;
}

template <typename T>
bool copy_device_column(const ColumnDesc &col, std::vector<T> &out) {
    if (!col.device_ptr.get() || col.length <= 0)
        return false;
    out.resize(col.length);
    CUDA_CHECK(cudaMemcpy(out.data(), col.device_ptr.get(),
                          sizeof(T) * col.length,
                          cudaMemcpyDeviceToHost));
    return true;
}

} // namespace

namespace {

struct QualifiedColumn {
    std::string relation;
    std::string column;
};

std::optional<QualifiedColumn> parse_qualified_column(const ASTNode *node) {
    auto *var = dynamic_cast<const VariableNode *>(node);
    if (!var) {
        return std::nullopt;
    }

    const std::string &name = var->name;
    const auto dot = name.find('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= name.size()) {
        return std::nullopt;
    }

    return QualifiedColumn{name.substr(0, dot), name.substr(dot + 1)};
}

} // namespace

bool compute_table_stats(const Table &table, TableStats &stats) {
    stats = TableStats{};
    bool gathered_any = false;

    for (const auto &col : table.columns) {
        TableStats::NumericColumnStats col_stats;
        bool found_stats = false;

        switch (col.type) {
        case DataType::Float32: {
            std::vector<float> host;
            if (copy_device_column(col, host) && !host.empty()) {
                auto [min_it, max_it] =
                    std::minmax_element(host.begin(), host.end());
                col_stats.min = *min_it;
                col_stats.max = *max_it;
                found_stats = true;
            }
            break;
        }
        case DataType::Float64: {
            std::vector<double> host;
            if (copy_device_column(col, host) && !host.empty()) {
                auto [min_it, max_it] =
                    std::minmax_element(host.begin(), host.end());
                col_stats.min = *min_it;
                col_stats.max = *max_it;
                found_stats = true;
            }
            break;
        }
        case DataType::Int32: {
            std::vector<int32_t> host;
            if (copy_device_column(col, host) && !host.empty()) {
                auto [min_it, max_it] =
                    std::minmax_element(host.begin(), host.end());
                col_stats.min = *min_it;
                col_stats.max = *max_it;
                col_stats.min_int = static_cast<int64_t>(*min_it);
                col_stats.max_int = static_cast<int64_t>(*max_it);
                found_stats = true;
            }
            break;
        }
        case DataType::Int64: {
            std::vector<int64_t> host;
            if (copy_device_column(col, host) && !host.empty()) {
                auto [min_it, max_it] =
                    std::minmax_element(host.begin(), host.end());
                col_stats.min_int = *min_it;
                col_stats.max_int = *max_it;
                col_stats.min = static_cast<double>(*min_it);
                col_stats.max = static_cast<double>(*max_it);
                found_stats = true;
            }
            break;
        }
        default:
            break;
        }

        if (found_stats) {
            col_stats.null_count = 0;
            col_stats.valid = true;
            col_stats.type = col.type;
            stats.numeric_columns[col.name] = col_stats;
            gathered_any = true;
        }
    }

    return gathered_any;
}

void analyze_condition(const ASTNode *node, const TableStats &stats,
                       bool &always_true, bool &always_false) {
    always_true = false;
    always_false = false;
    if (!node)
        return;

    if (auto bin = dynamic_cast<const BinaryOpNode *>(node)) {
        // Handle logical operators by recursively analyzing children
        if (bin->op == "&&" || bin->op == "||") {
            bool l_true = false, l_false = false;
            bool r_true = false, r_false = false;
            analyze_condition(bin->left.get(), stats, l_true, l_false);
            analyze_condition(bin->right.get(), stats, r_true, r_false);

            if (bin->op == "&&") {
                if (l_false || r_false)
                    always_false = true;
                if (l_true && r_true)
                    always_true = true;
            } else { // ||
                if (l_true || r_true)
                    always_true = true;
                if (l_false && r_false)
                    always_false = true;
            }
            return;
        }

        // Check for comparisons between a column and a constant
        const VariableNode *var = nullptr;
        const ConstantNode *cnst = nullptr;
        bool var_left = true;

        if (bin->left->type() == ASTNodeType::Variable &&
            bin->right->type() == ASTNodeType::Constant) {
            var = static_cast<const VariableNode *>(bin->left.get());
            cnst = static_cast<const ConstantNode *>(bin->right.get());
        } else if (bin->left->type() == ASTNodeType::Constant &&
                   bin->right->type() == ASTNodeType::Variable) {
            var = static_cast<const VariableNode *>(bin->right.get());
            cnst = static_cast<const ConstantNode *>(bin->left.get());
            var_left = false;
        }

        if (var && cnst) {
            const auto it = stats.numeric_columns.find(var->name);
            if (it != stats.numeric_columns.end() && it->second.valid) {
                const auto &col_stats = it->second;
                std::string op = bin->op;
                if (!var_left) {
                    if (op == ">")
                        op = "<";
                    else if (op == "<")
                        op = ">";
                    else if (op == ">=")
                        op = "<=";
                    else if (op == "<=")
                        op = ">=";
                }

                bool known = true;

                if ((col_stats.type == DataType::Int32 ||
                     col_stats.type == DataType::Int64) &&
                    col_stats.min_int.has_value() &&
                    col_stats.max_int.has_value()) {
                    int64_t c_int = 0;
                    try {
                        std::size_t parsed = 0;
                        c_int = std::stoll(cnst->value, &parsed, 10);
                        if (parsed != cnst->value.size())
                            known = false;
                    } catch (const std::exception &) {
                        known = false;
                    }

                    if (known) {
                        const auto min = *col_stats.min_int;
                        const auto max = *col_stats.max_int;

                        if (op == ">") {
                            if (min > c_int)
                                always_true = true;
                            else if (max <= c_int)
                                always_false = true;
                        } else if (op == ">=") {
                            if (min >= c_int)
                                always_true = true;
                            else if (max < c_int)
                                always_false = true;
                        } else if (op == "<") {
                            if (max < c_int)
                                always_true = true;
                            else if (min >= c_int)
                                always_false = true;
                        } else if (op == "<=") {
                            if (max <= c_int)
                                always_true = true;
                            else if (min > c_int)
                                always_false = true;
                        } else if (op == "==" || op == "=") {
                            if (min == max && min == c_int)
                                always_true = true;
                            else if (c_int < min || c_int > max)
                                always_false = true;
                        } else if (op == "!=") {
                            if (min == max && min == c_int)
                                always_false = true;
                            else if (c_int < min || c_int > max)
                                always_true = true;
                        }
                    }
                } else {
                    double c = parse_constant(cnst->value);
                    double min = col_stats.min;
                    double max = col_stats.max;

                    if (op == ">") {
                        if (min > c)
                            always_true = true;
                        else if (max <= c)
                            always_false = true;
                    } else if (op == ">=") {
                        if (min >= c)
                            always_true = true;
                        else if (max < c)
                            always_false = true;
                    } else if (op == "<") {
                        if (max < c)
                            always_true = true;
                        else if (min >= c)
                            always_false = true;
                    } else if (op == "<=") {
                        if (max <= c)
                            always_true = true;
                        else if (min > c)
                            always_false = true;
                    } else if (op == "==" || op == "=") {
                        if (min == max && min == c)
                            always_true = true;
                        else if (c < min || c > max)
                            always_false = true;
                    } else if (op == "!=") {
                        if (min == max && min == c)
                            always_false = true;
                        else if (c < min || c > max)
                            always_true = true;
                    }
                }
            }
        }
    }
}

std::vector<JoinPredicate> extract_equi_join_predicates(const QueryAST &ast) {
    std::vector<JoinPredicate> predicates;
    predicates.reserve(ast.joins.size());

    for (const auto &join : ast.joins) {
        auto *cmp = dynamic_cast<const BinaryOpNode *>(join.condition.get());
        if (!cmp || (cmp->op != "=" && cmp->op != "==")) {
            continue;
        }

        auto left = parse_qualified_column(cmp->left.get());
        auto right = parse_qualified_column(cmp->right.get());
        if (!left || !right || left->relation == right->relation) {
            continue;
        }

        predicates.push_back(
            JoinPredicate{left->relation, left->column,
                          right->relation, right->column});
    }

    return predicates;
}

double estimate_equi_join_rows(double left_rows, double right_rows,
                               double left_distinct, double right_distinct) {
    const double safe_left = std::max(left_rows, 1.0);
    const double safe_right = std::max(right_rows, 1.0);
    const double safe_left_ndv = std::max(left_distinct, 1.0);
    const double safe_right_ndv = std::max(right_distinct, 1.0);
    const double denom = std::max(safe_left_ndv, safe_right_ndv);
    return std::max(1.0, (safe_left * safe_right) / denom);
}

JoinPlan build_greedy_join_plan(
    const QueryAST &ast,
    const std::unordered_map<std::string, RelationStats> &stats_by_relation) {
    JoinPlan plan;

    std::unordered_set<std::string> relations;
    if (!ast.from_table.empty()) {
        relations.insert(ast.from_table);
    }
    for (const auto &join : ast.joins) {
        relations.insert(join.table);
    }
    if (relations.empty()) {
        return plan;
    }

    auto predicates = extract_equi_join_predicates(ast);

    auto relation_rows = [&](const std::string &relation) {
        const auto it = stats_by_relation.find(relation);
        if (it == stats_by_relation.end()) {
            return 1.0;
        }
        return std::max(it->second.row_count, 1.0);
    };

    auto column_ndv = [&](const std::string &relation,
                          const std::string &column) {
        const auto rel_it = stats_by_relation.find(relation);
        if (rel_it == stats_by_relation.end()) {
            return relation_rows(relation);
        }
        const auto ndv_it = rel_it->second.distinct_counts.find(column);
        if (ndv_it == rel_it->second.distinct_counts.end()) {
            return relation_rows(relation);
        }
        return std::max(ndv_it->second, 1.0);
    };

    std::unordered_set<std::string> joined;
    double current_rows = 0.0;

    auto seed_it = std::min_element(
        relations.begin(), relations.end(),
        [&](const std::string &a, const std::string &b) {
            return relation_rows(a) < relation_rows(b);
        });

    const std::string seed = *seed_it;
    joined.insert(seed);
    current_rows = relation_rows(seed);
    plan.steps.push_back(JoinPlanStep{seed, current_rows, std::nullopt});

    while (joined.size() < relations.size()) {
        double best_rows = std::numeric_limits<double>::max();
        std::string best_relation;
        std::optional<JoinPredicate> best_predicate;

        for (const auto &candidate : relations) {
            if (joined.count(candidate)) {
                continue;
            }

            double candidate_rows = relation_rows(candidate);
            bool connected = false;
            std::optional<JoinPredicate> candidate_predicate;

            for (const auto &pred : predicates) {
                const bool left_is_candidate = pred.left_relation == candidate;
                const bool right_is_candidate = pred.right_relation == candidate;

                if (!left_is_candidate && !right_is_candidate) {
                    continue;
                }

                const std::string other = left_is_candidate
                                              ? pred.right_relation
                                              : pred.left_relation;
                if (!joined.count(other)) {
                    continue;
                }

                connected = true;
                const double candidate_ndv = left_is_candidate
                                                 ? column_ndv(candidate,
                                                              pred.left_column)
                                                 : column_ndv(candidate,
                                                              pred.right_column);
                const double other_ndv = left_is_candidate
                                             ? column_ndv(other,
                                                          pred.right_column)
                                             : column_ndv(other,
                                                          pred.left_column);
                const double est_rows = estimate_equi_join_rows(
                    current_rows, relation_rows(candidate), other_ndv,
                    candidate_ndv);
                if (est_rows < candidate_rows) {
                    candidate_rows = est_rows;
                    candidate_predicate = pred;
                }
            }

            if (!connected) {
                candidate_rows = current_rows * relation_rows(candidate);
                candidate_predicate = std::nullopt;
            }

            if (candidate_rows < best_rows) {
                best_rows = candidate_rows;
                best_relation = candidate;
                best_predicate = candidate_predicate;
            }
        }

        if (best_relation.empty()) {
            break;
        }

        joined.insert(best_relation);
        current_rows = std::max(best_rows, 1.0);
        plan.steps.push_back(
            JoinPlanStep{best_relation, current_rows, best_predicate});
        plan.estimated_total_cost += current_rows;
    }

    return plan;
}
