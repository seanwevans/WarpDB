#pragma once
#include <vector>
#include <string>

struct Table {
    float* d_price;     // Device pointers
    int* d_quantity;
    int num_rows;
};

Table load_csv_to_gpu(const std::string& filepath);

