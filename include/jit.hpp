#pragma once
#include <string>

// Compile the given expression and optional condition using NVRTC and launch
// the generated kernel. The kernel operates on the provided price and quantity
// device arrays and writes the results to d_output.
void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            float *d_price, int *d_quantity,
                            float *d_output, int N, int device_id = 0);
