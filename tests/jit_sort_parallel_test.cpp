#include "test_utils.hpp"
#include "jit.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {
void fill_random(std::vector<float> &values) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto &v : values) {
        v = dist(rng);
    }
}

void validate_sorted(const std::vector<float> &values, bool ascending) {
    for (size_t i = 1; i < values.size(); ++i) {
        if (ascending) {
            assert(values[i - 1] <= values[i] + 1e-5f);
        } else {
            assert(values[i - 1] >= values[i] - 1e-5f);
        }
    }
}
}

int main() {
    if (warpdb_skip_if_no_cuda("jit_sort_parallel_test")) return 0;
    const int device_id = 0;
    const int count = 1 << 18; // 262,144 elements

    // Float-only sort benchmark
    std::vector<float> host_values(count);
    fill_random(host_values);

    float *d_values = nullptr;
    cudaMalloc(&d_values, sizeof(float) * count);
    cudaMemcpy(d_values, host_values.data(), sizeof(float) * count, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    jit_sort_float(d_values, count, true, device_id);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(host_values.data(), d_values, sizeof(float) * count, cudaMemcpyDeviceToHost);
    validate_sorted(host_values, true);
    std::cout << "Ascending float sort took " << elapsed_ms << " ms for " << count
              << " elements\n";

    // Descending float sort
    fill_random(host_values);
    cudaMemcpy(d_values, host_values.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    jit_sort_float(d_values, count, false, device_id);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(host_values.data(), d_values, sizeof(float) * count, cudaMemcpyDeviceToHost);
    validate_sorted(host_values, false);
    std::cout << "Descending float sort took " << elapsed_ms << " ms for " << count
              << " elements\n";

    // Key/value pair sort
    std::vector<int> host_keys(count);
    std::iota(host_keys.begin(), host_keys.end(), 0);
    std::shuffle(host_keys.begin(), host_keys.end(), std::mt19937(42));

    std::vector<float> host_vals(count);
    for (int i = 0; i < count; ++i) host_vals[i] = static_cast<float>(host_keys[i]) * 0.5f;

    int *d_keys = nullptr;
    float *d_vals = nullptr;
    cudaMalloc(&d_keys, sizeof(int) * count);
    cudaMalloc(&d_vals, sizeof(float) * count);
    cudaMemcpy(d_keys, host_keys.data(), sizeof(int) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, host_vals.data(), sizeof(float) * count, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    jit_sort_pairs(d_keys, d_vals, count, false, device_id);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(host_keys.data(), d_keys, sizeof(int) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vals.data(), d_vals, sizeof(float) * count, cudaMemcpyDeviceToHost);

    for (int i = 1; i < count; ++i) {
        assert(host_keys[i - 1] >= host_keys[i]);
        assert(host_vals[i - 1] >= host_vals[i]);
        assert(std::fabs(host_vals[i] - host_keys[i] * 0.5f) < 1e-5f);
    }

    std::cout << "Descending key/value sort took " << elapsed_ms << " ms for " << count
              << " pairs\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_values);
    cudaFree(d_keys);
    cudaFree(d_vals);

    return 0;
}
