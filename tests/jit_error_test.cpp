#include "jit.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

int main() {
    float *price; cudaMalloc(&price, sizeof(float));
    int *quantity; cudaMalloc(&quantity, sizeof(int));
    float *output; cudaMalloc(&output, sizeof(float));

    Table table;
    table.num_rows = 1;
    table.columns.push_back({"price", DataType::Float32, price, 1});
    table.columns.push_back({"quantity", DataType::Int32, quantity, 1});

    bool threw = false;
    try {
        // invalid code will fail to compile
        jit_compile_and_launch("invalid@", "", table, output);
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw && "Expected compilation to fail");

    // A second valid invocation should still succeed if resources were cleaned up
    threw = false;
    try {
        jit_compile_and_launch("price + 1", "", table, output);
    } catch (const std::exception& e) {
        threw = true;
        std::cerr << e.what() << "\n";
    }
    assert(!threw && "Second invocation failed");

    cudaFree(price);
    cudaFree(quantity);
    cudaFree(output);
    std::cout << "RAII test passed\n";
    return 0;
}
