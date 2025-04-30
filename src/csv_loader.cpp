#include "csv_loader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
} while (0)

Table load_csv_to_gpu(const std::string& filepath) {
    std::ifstream file(filepath);
    std::string line;
    std::vector<float> h_price;
    std::vector<int> h_quantity;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string price_str, qty_str;
        std::getline(ss, price_str, ',');
        std::getline(ss, qty_str, ',');

        h_price.push_back(std::stof(price_str));
        h_quantity.push_back(std::stoi(qty_str));
    }

    const int N = h_price.size();

    // Allocate pinned host memory (optional for now)
    float* h_price_pinned;
    int* h_quantity_pinned;
    CUDA_CHECK(cudaMallocHost((void**)&h_price_pinned, sizeof(float) * N));
    CUDA_CHECK(cudaMallocHost((void**)&h_quantity_pinned, sizeof(int) * N));

    std::copy(h_price.begin(), h_price.end(), h_price_pinned);
    std::copy(h_quantity.begin(), h_quantity.end(), h_quantity_pinned);

    // Allocate device memory
    float* d_price;
    int* d_quantity;
    CUDA_CHECK(cudaMalloc((void**)&d_price, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_quantity, sizeof(int) * N));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_price, h_price_pinned, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_quantity, h_quantity_pinned, sizeof(int) * N, cudaMemcpyHostToDevice));

    // Free pinned host mem
    CUDA_CHECK(cudaFreeHost(h_price_pinned));
    CUDA_CHECK(cudaFreeHost(h_quantity_pinned));

    Table table = { d_price, d_quantity, N };
    return table;
}

