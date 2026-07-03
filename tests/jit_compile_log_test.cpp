#include "test_utils.hpp"
#include "jit.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    if (warpdb_skip_if_no_cuda("jit_compile_log_test")) return 0;
    float *price = nullptr;
    int *quantity = nullptr;
    float *output = nullptr;
    CUDA_CHECK(cudaMalloc(&price, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&quantity, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&output, sizeof(float)));

    Table table;
    table.num_rows = 1;

    // Device pointers are handed to the columns, which own and free them
    // via their ColumnDevicePtr RAII members when `table` is destroyed.
    ColumnDesc price_col;
    price_col.name = "price";
    price_col.type = DataType::Float32;
    price_col.length = 1;
    price_col.device_ptr.reset(price);
    table.columns.push_back(std::move(price_col));

    ColumnDesc quantity_col;
    quantity_col.name = "quantity";
    quantity_col.type = DataType::Int32;
    quantity_col.length = 1;
    quantity_col.device_ptr.reset(quantity);
    table.columns.push_back(std::move(quantity_col));

    bool threw = false;
    try {
        // Invalid code fails to compile and should include the log in the error.
        jit_compile_and_launch("invalid@", "", table, output);
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("error:") != std::string::npos &&
               "Expected compile log in error message");
    }
    assert(threw && "Expected compilation to fail");

    cudaFree(output);

    std::cout << "JIT compile log test passed\n";
    return 0;
}
