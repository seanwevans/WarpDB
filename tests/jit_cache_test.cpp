#include "test_utils.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    if (warpdb_skip_if_no_cuda("jit_cache_test")) return 0;
    reset_jit_cache();

    const int N = 256;
    std::vector<float> h_price(N, 1.0f);
    std::vector<int> h_quantity(N, 2);

    float *d_price;
    int *d_quantity;
    float *d_output;
    cudaMalloc(&d_price, sizeof(float) * N);
    cudaMalloc(&d_quantity, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(float) * N);

    cudaMemcpy(d_price, h_price.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantity, h_quantity.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    Table table;
    table.num_rows = N;
    table.columns.push_back({"price", DataType::Float32, d_price, N});
    table.columns.push_back({"quantity", DataType::Int32, d_quantity, N});

    auto timed_run = [&](float *out_ptr) {
        auto start = std::chrono::steady_clock::now();
        jit_compile_and_launch("price + quantity", "quantity > 0", table, out_ptr);
        cudaDeviceSynchronize();
        return std::chrono::steady_clock::now() - start;
    };

    auto first_duration = timed_run(d_output);
    auto second_duration = timed_run(d_output);

    assert(get_jit_compile_count() == 1 && "Expected compiled kernel to be cached");
    std::cout << "JIT cache timings (us): first="
              << std::chrono::duration_cast<std::chrono::microseconds>(first_duration).count()
              << " second="
              << std::chrono::duration_cast<std::chrono::microseconds>(second_duration).count()
              << "\n";

    reset_jit_cache();

    float *d_output_a;
    float *d_output_b;
    cudaMalloc(&d_output_a, sizeof(float) * N);
    cudaMalloc(&d_output_b, sizeof(float) * N);

    std::thread t1([&] { jit_compile_and_launch("price + quantity", "quantity > 0", table, d_output_a); });
    std::thread t2([&] { jit_compile_and_launch("price + quantity", "quantity > 0", table, d_output_b); });
    t1.join();
    t2.join();

    assert(get_jit_compile_count() == 1 && "Kernel should be compiled once across threads");

    cudaFree(d_output_a);
    cudaFree(d_output_b);
    cudaFree(d_output);

    std::cout << "JIT cache test passed\n";
    return 0;
}
