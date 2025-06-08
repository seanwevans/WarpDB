#pragma once
#include <string>

// Compile the given expression and optional condition using NVRTC and launch
// the generated kernel. Columns are provided via the Table descriptor.
void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            float *d_price, int *d_quantity,
                            float *d_output, int N, int device_id = 0);

// JIT compile a kernel that groups rows by `key_expr_code` and sums
// `val_expr_code`. The number of unique groups is written to d_count and
// results are stored in d_out_vals and d_out_keys.
void jit_group_sum(const std::string &val_expr_code,
                   const std::string &key_expr_code, float *d_price,
                   int *d_quantity, float *d_out_vals, int *d_out_keys,
                   int *d_count, int N, int device_id = 0);

// Sort integer keys and corresponding values in place using a simple GPU
// kernel. When ascending is false results are sorted in descending order.
void jit_sort_pairs(int *d_keys, float *d_vals, int count, bool ascending,
                    int device_id = 0);

// Sort a single float array in-place.
void jit_sort_float(float *d_vals, int count, bool ascending,
                    int device_id = 0);

