#pragma once
#include <string>

#include "csv_loader.hpp"

void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            const Table &table, float *d_output);
                            float *d_price, int *d_quantity, float *d_output,
                            int N, int device_id = 0);
