#include "warpdb.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <iostream>

WarpDB::WarpDB(const std::string &csv_path) {
    table_ = load_csv_to_gpu(csv_path);
}

WarpDB::~WarpDB() {
    cudaFree(table_.d_price);
    cudaFree(table_.d_quantity);
}

std::vector<float> WarpDB::query(const std::string &expr) {
    std::string upper = expr;
    for (auto &c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string expr_part = expr;
    std::string where_part;
    auto where_pos = upper.find("WHERE");
    if (where_pos != std::string::npos) {
        expr_part = expr.substr(0, where_pos);
        where_part = expr.substr(where_pos + 5);
    }

    auto expr_tokens = tokenize(expr_part);
    auto expr_ast = parse_expression(expr_tokens);
    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        auto cond_ast = parse_expression(cond_tokens);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table_.num_rows);

    jit_compile_and_launch(expr_cuda, condition_cuda, table_.d_price, table_.d_quantity, d_output, table_.num_rows);

    std::vector<float> result(table_.num_rows);
    cudaMemcpy(result.data(), d_output, sizeof(float) * table_.num_rows, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    return result;
}
