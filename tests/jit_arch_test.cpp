#include "jit.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

int main() {
    float h_price = 2.0f;
    int h_quantity = 0;
    float h_output = 0.0f;
    float *d_price; cudaMalloc(&d_price, sizeof(float));
    int *d_quantity; cudaMalloc(&d_quantity, sizeof(int));
    float *d_output; cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_price, &h_price, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantity, &h_quantity, sizeof(int), cudaMemcpyHostToDevice);

    bool threw = false;
    try {
        jit_compile_and_launch("price", "", d_price, d_quantity, d_output, 1);
    } catch (const std::exception &e) {
        threw = true;
        std::cerr << e.what() << "\n";
    }
    assert(!threw && "JIT compilation failed");

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    assert(h_output == h_price);

    cudaFree(d_price);
    cudaFree(d_quantity);
    cudaFree(d_output);
    std::cout << "Architecture detection test passed\n";
    return 0;
}
