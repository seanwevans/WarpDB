#pragma once
#include <string>

void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            float *d_price, int *d_quantity, float *d_output,
                            int N, int device_id = 0);
