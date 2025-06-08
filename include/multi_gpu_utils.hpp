#pragma once
#include <string>
#include <vector>
#include "csv_loader.hpp"
#include "jit.hpp"

// Execute a JIT compiled expression across all available GPUs for the provided
// host table chunk. Falls back to single GPU execution when only one device is
// present.
std::vector<float> run_multi_gpu_jit_host(const HostTable &host,
                                          const std::string &expr_cuda,
                                          const std::string &cond_cuda);
