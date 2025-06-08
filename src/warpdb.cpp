#include "warpdb.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <memory>

namespace {
// Recursively validate that all variable references exist in the table.
void validate_ast(const ASTNode *node,
                  const std::unordered_set<std::string> &cols) {
    if (!node) return;
    if (auto var = dynamic_cast<const VariableNode *>(node)) {
        if (cols.find(var->name) == cols.end()) {
            throw std::runtime_error("Unknown column: " + var->name);
        }
    } else if (auto bin = dynamic_cast<const BinaryOpNode *>(node)) {
        validate_ast(bin->left.get(), cols);
        validate_ast(bin->right.get(), cols);
    } else if (auto func = dynamic_cast<const FunctionCallNode *>(node)) {
        for (const auto &a : func->args) {
            validate_ast(a.get(), cols);
        }
    } else if (auto agg = dynamic_cast<const AggregationNode *>(node)) {
        validate_ast(agg->expr.get(), cols);
    } else if (auto win = dynamic_cast<const WindowFunctionNode *>(node)) {
        validate_ast(win->expr.get(), cols);
        for (const auto &p : win->partition_by) {
            validate_ast(p.get(), cols);
        }
        if (win->order_by) {
            validate_ast(win->order_by->expr.get(), cols);
        }
    }
}
} // namespace

WarpDB::WarpDB(const std::string &csv_path) {
    table_ = load_csv_to_gpu(csv_path);
}

WarpDB::~WarpDB() {
#ifdef USE_ARROW
    table_.d_price.reset();
    table_.d_quantity.reset();
#else
    cudaFree(table_.d_price);
    cudaFree(table_.d_quantity);
#endif
}

std::vector<float> WarpDB::query(const std::string &expr) {
    if (expr.empty()) {
        throw std::runtime_error("Empty query expression");
    }

    std::string upper = expr;
    for (auto &c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string expr_part = expr;
    std::string where_part;
    auto where_pos = upper.find("WHERE");
    if (where_pos != std::string::npos) {
        expr_part = expr.substr(0, where_pos);
        where_part = expr.substr(where_pos + 5);
    }

    std::unique_ptr<ASTNode> expr_ast;
    try {
        auto expr_tokens = tokenize(expr_part);
        expr_ast = parse_expression(expr_tokens);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to parse expression: ") + e.what());
    }

    std::unordered_set<std::string> cols;
    for (const auto &c : table_.columns) {
        cols.insert(c.name);
    }
    validate_ast(expr_ast.get(), cols);

    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (!where_part.empty()) {
        try {
            auto cond_tokens = tokenize(where_part);
            auto cond_ast = parse_expression(cond_tokens);
            validate_ast(cond_ast.get(), cols);
            condition_cuda = cond_ast->to_cuda_expr();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Failed to parse WHERE clause: ") + e.what());
        }
    }

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table_.num_rows);

    try {
        jit_compile_and_launch(expr_cuda, condition_cuda, table_.d_price, table_.d_quantity, d_output, table_.num_rows);
    } catch (const std::exception &e) {
        cudaFree(d_output);
        throw;
    }

    std::vector<float> result(table_.num_rows);
    cudaMemcpy(result.data(), d_output, sizeof(float) * table_.num_rows, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    return result;
}

void WarpDB::query_arrow(const std::string &expr, ArrowArray *out_array,
                         ArrowSchema *out_schema, bool use_shared_memory) {
    auto result = query(expr);
    export_to_arrow(result.data(), static_cast<int64_t>(result.size()),
                    use_shared_memory, out_array, out_schema);
}
