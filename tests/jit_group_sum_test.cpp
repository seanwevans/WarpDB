#include "jit.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

int main() {
    const int N = 1 << 15;
    const int num_groups = 4096;

    std::vector<float> price(N);
    std::vector<int> quantity(N);
    for (int i = 0; i < N; ++i) {
        price[i] = static_cast<float>((i % 100)) * 0.25f;
        quantity[i] = i % num_groups;
    }

    std::vector<double> expected(num_groups, 0.0);
    for (int i = 0; i < N; ++i) {
        expected[quantity[i]] += price[i];
    }

    float *d_price = nullptr;
    int *d_quantity = nullptr;
    float *d_out_vals = nullptr;
    int *d_out_keys = nullptr;
    int *d_count = nullptr;
    cudaMalloc(&d_price, sizeof(float) * N);
    cudaMalloc(&d_quantity, sizeof(int) * N);
    cudaMalloc(&d_out_vals, sizeof(float) * N);
    cudaMalloc(&d_out_keys, sizeof(int) * N);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_price, price.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantity, quantity.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    jit_group_sum("price", "quantity", d_price, d_quantity, d_out_vals,
                  d_out_keys, d_count, N);

    int host_count = 0;
    cudaMemcpy(&host_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    assert(host_count == num_groups);

    std::vector<int> out_keys(host_count);
    std::vector<float> out_vals(host_count);
    cudaMemcpy(out_keys.data(), d_out_keys, sizeof(int) * host_count,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_vals.data(), d_out_vals, sizeof(float) * host_count,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < host_count; ++i) {
        assert(out_keys[i] == i);
        assert(std::abs(out_vals[i] - expected[i]) < 1e-4f);
    }

    // Reuse allocations for a second scenario with fewer groups and a
    // different distribution to exercise reduce-by-key on irregular keys.
    const int N2 = 2048;
    const int small_groups = 200;
    price.assign(N2, 0.0f);
    quantity.assign(N2, 0);
    for (int i = 0; i < N2; ++i) {
        price[i] = 1.0f + static_cast<float>(i % 5);
        quantity[i] = (i * 13) % small_groups;
    }

    std::unordered_map<int, double> expected_small;
    for (int i = 0; i < N2; ++i) {
        expected_small[quantity[i]] += price[i];
    }

    cudaMemcpy(d_price, price.data(), sizeof(float) * N2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantity, quantity.data(), sizeof(int) * N2, cudaMemcpyHostToDevice);

    jit_group_sum("price", "quantity", d_price, d_quantity, d_out_vals,
                  d_out_keys, d_count, N2);

    host_count = 0;
    cudaMemcpy(&host_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    assert(host_count == static_cast<int>(expected_small.size()));

    out_keys.resize(host_count);
    out_vals.resize(host_count);
    cudaMemcpy(out_keys.data(), d_out_keys, sizeof(int) * host_count,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_vals.data(), d_out_vals, sizeof(float) * host_count,
               cudaMemcpyDeviceToHost);

    std::vector<std::pair<int, double>> expected_pairs(expected_small.begin(),
                                                       expected_small.end());
    std::sort(expected_pairs.begin(), expected_pairs.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    for (int i = 0; i < host_count; ++i) {
        assert(out_keys[i] == expected_pairs[i].first);
        assert(std::abs(out_vals[i] - expected_pairs[i].second) < 1e-4f);
    }

    cudaFree(d_price);
    cudaFree(d_quantity);
    cudaFree(d_out_vals);
    cudaFree(d_out_keys);
    cudaFree(d_count);
    return 0;
}
