#include "expression.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

void test_malformed_expression() {
    auto tokens = tokenize("1 2");
    bool threw = false;
    try {
        auto ast = parse_expression(tokens);
        (void)ast;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Unexpected token") != std::string::npos);
    }
    assert(threw && "Expected exception for malformed expression");
}

void test_scientific_literal_compiles() {
    auto tokens = tokenize("1e3");
    auto ast = parse_expression(tokens);
    std::string code = ast->to_cuda_expr();
    assert(code == "1e3f");

    Table table;
    table.num_rows = 1;

    float *d_output = nullptr;
    cudaError_t alloc_status = cudaMalloc(&d_output, sizeof(float));
    assert(alloc_status == cudaSuccess);

    bool threw = false;
    try {
        jit_compile_and_launch(code, "", table, d_output);
    } catch (const std::exception &e) {
        threw = true;
        std::cerr << e.what() << '\n';
    }
    assert(!threw && "CUDA JIT compilation failed for scientific literal");

    float host_output = 0.0f;
    cudaError_t copy_status =
        cudaMemcpy(&host_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    assert(copy_status == cudaSuccess);
    assert(host_output == 1000.0f);

    cudaFree(d_output);
}

int main() {
    test_malformed_expression();
    test_scientific_literal_compiles();
    std::cout << "All tests passed\n";
    return 0;
}
