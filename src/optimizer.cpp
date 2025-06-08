#include "optimizer.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

namespace {

float parse_constant(const std::string &val) {
    return std::stof(val);
}

void analyze_condition(const ASTNode *node, const TableStats &stats,
                       bool &always_true, bool &always_false) {
    const BinaryOpNode *bin = dynamic_cast<const BinaryOpNode *>(node);
    if (!bin) return; // only handle binary comparisons

    const VariableNode *var = dynamic_cast<const VariableNode *>(bin->left.get());
    const ConstantNode *cnst = dynamic_cast<const ConstantNode *>(bin->right.get());
    if (!var || !cnst) return;

    float val = parse_constant(cnst->value);
    if (var->name == "price") {
        if (bin->op == ">") {
            if (stats.price.max <= val) always_false = true;
            if (stats.price.min > val) always_true = true;
        } else if (bin->op == "<") {
            if (stats.price.min >= val) always_false = true;
            if (stats.price.max < val) always_true = true;
        }
    } else if (var->name == "quantity") {
        int ival = static_cast<int>(val);
        if (bin->op == ">") {
            if (stats.quantity.max <= ival) always_false = true;
            if (stats.quantity.min > ival) always_true = true;
        } else if (bin->op == "<") {
            if (stats.quantity.min >= ival) always_false = true;
            if (stats.quantity.max < ival) always_true = true;
        }
    }
}

} // namespace

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
    if (cond_ast) {
        analyze_condition(cond_ast.get(), table.stats, always_true, always_false);
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

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table.num_rows);
    jit_compile_and_launch(expr_cuda, cond_cuda, table.d_price, table.d_quantity,
                           d_output, table.num_rows);

    float *h_out = new float[table.num_rows];
    cudaMemcpy(h_out, d_output, sizeof(float) * table.num_rows,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < table.num_rows; ++i) {
        std::cout << "Result[" << i << "] = " << h_out[i] << "\n";
    }
    delete[] h_out;
    cudaFree(d_output);
}
