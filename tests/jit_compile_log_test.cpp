#include "jit.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

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
        // invalid code will fail to compile and include log in the error
        jit_compile_and_launch("invalid@", "", table, output);
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("error:") != std::string::npos &&
               "Expected compile log in error message");
    }
    assert(threw && "Expected compilation to fail");

    cudaFree(price);
    cudaFree(quantity);
    cudaFree(output);

    std::cout << "JIT compile log test passed\n";
    return 0;
}
