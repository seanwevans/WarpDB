#include "multi_gpu_utils.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "No CUDA devices available; skipping multi-GPU sharding test.\n";
        return 0;
    }

    const int rows = 16;

    HostTable host;

    HostColumn f32_col;
    f32_col.name = "f32_col";
    f32_col.type = DataType::Float32;
    f32_col.data = std::vector<float>(rows);
    auto &f32_vec = std::get<std::vector<float>>(f32_col.data);
    for (int i = 0; i < rows; ++i) {
        f32_vec[i] = 0.5f * static_cast<float>(i);
    }
    host.columns.push_back(std::move(f32_col));

    HostColumn i32_col;
    i32_col.name = "i32_col";
    i32_col.type = DataType::Int32;
    i32_col.data = std::vector<int32_t>(rows);
    auto &i32_vec = std::get<std::vector<int32_t>>(i32_col.data);
    for (int i = 0; i < rows; ++i) {
        i32_vec[i] = i;
    }
    host.columns.push_back(std::move(i32_col));

    HostColumn i64_col;
    i64_col.name = "i64_col";
    i64_col.type = DataType::Int64;
    i64_col.data = std::vector<int64_t>(rows);
    auto &i64_vec = std::get<std::vector<int64_t>>(i64_col.data);
    for (int i = 0; i < rows; ++i) {
        i64_vec[i] = static_cast<int64_t>(1000 + i);
    }
    host.columns.push_back(std::move(i64_col));

    HostColumn f64_col;
    f64_col.name = "f64_col";
    f64_col.type = DataType::Float64;
    f64_col.data = std::vector<double>(rows);
    auto &f64_vec = std::get<std::vector<double>>(f64_col.data);
    for (int i = 0; i < rows; ++i) {
        f64_vec[i] = 1.25 * static_cast<double>(i);
    }
    host.columns.push_back(std::move(f64_col));

    HostColumn str_col;
    str_col.name = "str_col";
    str_col.type = DataType::String;
    std::vector<std::string> strings(rows);
    for (int i = 0; i < rows; ++i) {
        strings[i] = "row_" + std::to_string(i);
    }
    str_col.data = std::move(strings);
    host.columns.push_back(std::move(str_col));

    auto results = run_multi_gpu_jit_host(host, "f32_col + 1.0f", "");
    assert(static_cast<int>(results.size()) == rows);

    for (int i = 0; i < rows; ++i) {
        float expected = f32_vec[i] + 1.0f;
        assert(std::fabs(results[i] - expected) < 1e-6f);
    }

    if (device_count < 2) {
        std::cout << "Only " << device_count
                  << " CUDA device available; multi-GPU sharding path not exercised.\n";
    }

    std::cout << "Multi-GPU sharding upload test passed.\n";
    return 0;
}
