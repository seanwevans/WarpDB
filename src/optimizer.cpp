#include "optimizer.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <exception>

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

void execute_query_optimized(const std::string &expr_part,
                             const std::string &where_part, Table &table) {
    auto expr_tokens = tokenize(expr_part);
    auto expr_ast = parse_expression(expr_tokens);

    std::unique_ptr<ASTNode> cond_ast;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        cond_ast = parse_expression(cond_tokens);
    }

    bool always_true = false;
    bool always_false = false;
    TableStats stats;
    bool have_stats = compute_table_stats(table, stats);

    if (cond_ast && have_stats) {
        analyze_condition(cond_ast.get(), stats, always_true, always_false);
    }

    if (always_false) {
        std::cout << "[Optimizer] Filter eliminates all rows.\n";
        return;
    }

    std::string expr_cuda = expr_ast->to_cuda_expr();
    std::string cond_cuda;
    if (cond_ast && !always_true) {
        cond_cuda = cond_ast->to_cuda_expr();
    }

    DeviceBuffer<float> d_output(table.num_rows);
    // Pass 0 for block_size so jit_compile_and_launch selects an occupancy
    // optimised configuration.
    jit_compile_and_launch(expr_cuda, cond_cuda, table, d_output.get(), 0, 0);

    std::vector<float> h_out(table.num_rows);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_output.get(),
                          sizeof(float) * table.num_rows,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < table.num_rows; ++i) {
        std::cout << "Result[" << i << "] = " << h_out[i] << "\n";
    }

}
