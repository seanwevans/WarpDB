#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "arrow_c_abi.h"

// Simple utility to export a float array to Arrow buffers.
// If use_shared_memory is true, the data buffer will be mapped using
// POSIX shared memory so external processes can read it.  When
// shared memory is used, `shm_name` specifies the name of the
// POSIX shared memory region.  It defaults to "/warpdb_result".
void export_to_arrow(const float* data, int64_t length, bool use_shared_memory,
                     ArrowArray* out_array, ArrowSchema* out_schema,
                     const char* shm_name = "/warpdb_result");
