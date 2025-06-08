#pragma once
#include <string>

// Compile the given expression and optional condition using NVRTC and launch
// the generated kernel. Columns are provided via the Table descriptor.
void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            const Table &table, float *d_output,
                            int device_id = 0);
