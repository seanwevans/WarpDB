#include "optimizer.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

namespace {

float parse_constant(const std::string &val) {
    return std::stof(val);
}

void analyze_condition(const ASTNode *, const TableStats &,
                       bool &always_true, bool &always_false) {
    always_true = false;
    always_false = false;
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
        analyze_condition(cond_ast.get(), {}, always_true, always_false);
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
    jit_compile_and_launch(expr_cuda, cond_cuda, table, d_output);

    float *h_out = new float[table.num_rows];
    cudaMemcpy(h_out, d_output, sizeof(float) * table.num_rows,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < table.num_rows; ++i) {
        std::cout << "Result[" << i << "] = " << h_out[i] << "\n";
    }
    delete[] h_out;
    cudaFree(d_output);
}
