#pragma once
#include <cstdint>
#include <string>
#include "csv_loader.hpp"

// Compile the given expression and optional condition using NVRTC and launch
// the generated kernel. Columns are provided via the Table descriptor.
// Launches a JIT compiled kernel for the given expression and optional
// condition. `block_size` specifies the CUDA block dimension; when set to 0 the
// function selects a block size using `cuOccupancyMaxPotentialBlockSize`.
void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            const Table &table,
                            float *d_output, int device_id = 0,
                            int block_size = 0);

// Inspect or reset the JIT compilation cache. Primarily used for tests and
// performance measurements.
size_t get_jit_compile_count();
void reset_jit_cache();

enum class GroupReduceOp { Sum, Min, Max };

// Reduce numeric values by integer keys on GPU. Keys must be Int32 or Int64
// and values must be numeric. Returns number of unique groups and writes
// grouped keys and aggregated values to output buffers.
int jit_group_reduce_by_key(const void *d_keys, DataType key_type,
                            const void *d_values, DataType value_type,
                            GroupReduceOp op, int64_t *d_out_keys,
                            float *d_out_vals, int N, int device_id = 0);

// Count rows per key on GPU. Keys must be Int32 or Int64. Returns number of
// unique groups and writes grouped keys and counts (as float) to outputs.
int jit_group_count_by_key(const void *d_keys, DataType key_type,
                           int64_t *d_out_keys, float *d_out_counts, int N,
                           int device_id = 0);

// Sort integer keys and corresponding values in place using a parallel GPU
// implementation. When ascending is false results are sorted in descending
// order.
void jit_sort_pairs(int *d_keys, float *d_vals, int count, bool ascending,
                    int device_id = 0);

// Sort a single float array in-place using a parallel GPU implementation.
void jit_sort_float(float *d_vals, int count, bool ascending,
                    int device_id = 0);
