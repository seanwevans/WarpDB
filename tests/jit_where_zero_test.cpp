#include "test_utils.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

int main() {
    if (warpdb_skip_if_no_cuda("jit_where_zero_test")) return 0;
    const int N = 3;
    std::vector<float> h_price = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_output_init(N, 5.0f);

    float *d_price;
    float *d_output;
    cudaMalloc(&d_price, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    cudaMemcpy(d_price, h_price.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output_init.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    Table table;
    table.num_rows = N;
    table.columns.push_back(warpdb_make_device_column("price", DataType::Float32, d_price, N));

    jit_compile_and_launch("price * 2.0f", "price > 2.0f", table, d_output);

    std::vector<float> h_output(N);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    assert(h_output[0] == 0.0f);
    assert(h_output[1] == 0.0f);
    assert(h_output[2] == 6.0f);

    cudaFree(d_output);

    return 0;
}
