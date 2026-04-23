// src/jit.cu
#include "jit.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nvrtc.h>
#include <sstream>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <atomic>
#include <memory>

#define NVRTC_CHECK(stmt)                                                      \
  do {                                                                         \
    nvrtcResult result = stmt;                                                 \
    if (result != NVRTC_SUCCESS) {                                             \
      throw std::runtime_error("NVRTC error: " +                               \
                               std::string(nvrtcGetErrorString(result)));      \
    }                                                                          \
  } while (0)

#define CU_CHECK(stmt)                                                         \
  do {                                                                         \
    CUresult result = stmt;                                                    \
    if (result != CUDA_SUCCESS) {                                              \
      const char *errStr;                                                      \
      cuGetErrorString(result, &errStr);                                       \
      throw std::runtime_error("CUDA error: " + std::string(errStr));          \
    }                                                                          \
  } while (0)

#define CUDA_RUNTIME_CHECK(stmt)                                               \
  do {                                                                         \
    cudaError_t result = stmt;                                                 \
    if (result != cudaSuccess) {                                               \
      throw std::runtime_error("CUDA runtime error: " +                         \
                               std::string(cudaGetErrorString(result)));       \
    }                                                                          \
  } while (0)

namespace {

std::string cuda_type(DataType t) {
  switch (t) {
  case DataType::Int32:
    return "int";
  case DataType::Int64:
    return "long long";
  case DataType::Float32:
    return "float";
  case DataType::Float64:
    return "double";
  case DataType::String:
    return "void*"; // unsupported
  }
  return "void*";
}

struct CudaDeviceInfo {
  CUdevice device;
  std::string arch_flag;
};

const CudaDeviceInfo &initialize_cuda_device(int device_id) {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() { CU_CHECK(cuInit(0)); });

  static std::unordered_map<int, CudaDeviceInfo> cache;
  auto it = cache.find(device_id);
  if (it == cache.end()) {
    CudaDeviceInfo info;
    CU_CHECK(cuDeviceGet(&info.device, device_id));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, info.device));
    CU_CHECK(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, info.device));
    info.arch_flag = "--gpu-architecture=compute_" + std::to_string(major) +
                     std::to_string(minor);
    it = cache.emplace(device_id, info).first;
  }
  return it->second;
}

struct ContextRef {
  CUdevice device{0};
  CUcontext ctx{nullptr};

  explicit ContextRef(CUdevice device_in) : device(device_in) {
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
  }

  ~ContextRef() { cuDevicePrimaryCtxRelease(device); }
};

std::shared_ptr<ContextRef> get_primary_context(const CudaDeviceInfo &info,
                                                int device_id) {
  static std::mutex ctx_mutex;
  static std::unordered_map<int, std::weak_ptr<ContextRef>> ctx_cache;

  std::lock_guard<std::mutex> lock(ctx_mutex);
  auto it = ctx_cache.find(device_id);
  if (it != ctx_cache.end()) {
    if (auto existing = it->second.lock()) {
      return existing;
    }
  }

  auto ctx = std::make_shared<ContextRef>(info.device);
  ctx_cache[device_id] = ctx;
  return ctx;
}

std::string compile_cuda_source(const std::string &src,
                                const std::string &name,
                                const std::string &arch_flag) {
  struct NvrtcProgramGuard {
    nvrtcProgram prog{nullptr};
    ~NvrtcProgramGuard() {
      if (prog)
        nvrtcDestroyProgram(&prog);
    }
  } prog;

  NVRTC_CHECK(nvrtcCreateProgram(&prog.prog, src.c_str(), name.c_str(), 0,
                                 nullptr, nullptr));
  const char *opts[] = {arch_flag.c_str()};
  nvrtcResult compileResult = nvrtcCompileProgram(prog.prog, 1, opts);
  size_t logSize;
  nvrtcGetProgramLogSize(prog.prog, &logSize);
  std::string log(logSize, '\0');
  nvrtcGetProgramLog(prog.prog, &log[0]);
  if (compileResult != NVRTC_SUCCESS) {
    std::cerr << "NVRTC Compile Log:\n" << log << "\n";
    throw std::runtime_error("Kernel compilation failed.\n" + log);
  }
  size_t ptxSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog.prog, &ptxSize));
  std::string ptx(ptxSize, '\0');
  NVRTC_CHECK(nvrtcGetPTX(prog.prog, &ptx[0]));
  return ptx;
}

struct ContextSetGuard {
  CUcontext prev_ctx{nullptr};
  int prev_runtime_device{-1};
  bool has_prev_runtime_device{false};

  ContextSetGuard(CUcontext target, int device_id) {
    CU_CHECK(cuCtxGetCurrent(&prev_ctx));
    int runtime_device = 0;
    cudaError_t get_device_result = cudaGetDevice(&runtime_device);
    if (get_device_result == cudaSuccess) {
      prev_runtime_device = runtime_device;
      has_prev_runtime_device = true;
    } else if (get_device_result != cudaErrorNoDevice) {
      throw std::runtime_error("CUDA runtime error: " +
                               std::string(cudaGetErrorString(get_device_result)));
    }
    CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cuCtxSetCurrent(target));
  }

  ~ContextSetGuard() {
    cuCtxSetCurrent(prev_ctx);
    if (has_prev_runtime_device) {
      cudaSetDevice(prev_runtime_device);
    }
  }
};

struct KernelCacheKey {
  int device_id;
  std::string code;

  bool operator==(const KernelCacheKey &other) const {
    return device_id == other.device_id && code == other.code;
  }
};

struct KernelCacheKeyHash {
  size_t operator()(const KernelCacheKey &k) const {
    size_t h1 = std::hash<int>{}(k.device_id);
    size_t h2 = std::hash<std::string>{}(k.code);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

struct KernelCacheEntry {
  std::shared_ptr<ContextRef> context;
  std::shared_ptr<CUmodule> module_holder;
  CUfunction function{nullptr};
};

std::unordered_map<KernelCacheKey, std::shared_ptr<KernelCacheEntry>,
                   KernelCacheKeyHash> &kernel_cache_instance() {
  static auto *cache = new std::unordered_map<
      KernelCacheKey, std::shared_ptr<KernelCacheEntry>, KernelCacheKeyHash>();
  return *cache;
}

std::mutex &kernel_cache_mutex() {
  static auto *m = new std::mutex();
  return *m;
}

std::shared_ptr<KernelCacheEntry> load_or_get_cached_kernel(
    const KernelCacheKey &key, const CudaDeviceInfo &device,
    const std::string &kernel_src, const char *kernel_name,
    std::atomic<size_t> &compile_counter) {
  std::unique_lock<std::mutex> lock(kernel_cache_mutex());
  auto it = kernel_cache_instance().find(key);
  if (it != kernel_cache_instance().end()) {
    return it->second;
  }

  std::string ptx =
      compile_cuda_source(kernel_src, std::string(kernel_name) + ".cu",
                          device.arch_flag);
  compile_counter.fetch_add(1, std::memory_order_relaxed);

  auto context = get_primary_context(device, key.device_id);
  ContextSetGuard guard(context->ctx, key.device_id);

  CUmodule raw_module{nullptr};
  CU_CHECK(cuModuleLoadDataEx(&raw_module, ptx.c_str(), 0, nullptr, nullptr));

  auto module_holder = std::shared_ptr<CUmodule>(
      new CUmodule(raw_module),
      [context](CUmodule *mod) {
        if (!mod)
          return;
        CUcontext prev{nullptr};
        cuCtxGetCurrent(&prev);
        cuCtxSetCurrent(context->ctx);
        cuModuleUnload(*mod);
        cuCtxSetCurrent(prev);
        delete mod;
      });

  CUfunction kernel_func{nullptr};
  CU_CHECK(cuModuleGetFunction(&kernel_func, raw_module, kernel_name));

  auto entry = std::make_shared<KernelCacheEntry>(
      KernelCacheEntry{context, module_holder, kernel_func});

  auto [insert_it, inserted] = kernel_cache_instance().emplace(key, entry);
  if (!inserted) {
    return insert_it->second;
  }
  return entry;
}

std::atomic<size_t> &compile_counter_instance() {
  static std::atomic<size_t> counter{0};
  return counter;
}

void reset_kernel_cache() {
  std::lock_guard<std::mutex> lock(kernel_cache_mutex());
  kernel_cache_instance().clear();
  compile_counter_instance().store(0, std::memory_order_relaxed);
}

} // namespace

size_t get_jit_compile_count() {
  return compile_counter_instance().load(std::memory_order_relaxed);
}

void reset_jit_cache() { reset_kernel_cache(); }

void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            const Table &table, float *d_output,
                            int device_id, int block_size) {

  int N = table.num_rows;

  std::string body;
  if (!condition_code.empty()) {
    body = "if (" + condition_code + ") {\n    output[idx] = " + expr_code +
           ";\n} else {\n    output[idx] = 0.0f;\n}";
  } else {
    body = "output[idx] = " + expr_code + ";";
  }



  std::string custom_code;
  {
    std::ifstream in("custom.cu");
    if (in) {
      std::stringstream ss;
      ss << in.rdbuf();
      custom_code = ss.str();
    }
  }

  std::string params;
  for (const auto &c : table.columns) {
    params += cuda_type(c.type) + "* " + c.name + ", ";
  }
  params += "float* output, int N";

  std::string kernel = custom_code + "\nextern \"C\" __global__\n    void user_kernel(" +
                     params + ") {\n        int idx = blockIdx.x * blockDim.x + threadIdx.x;\n        if (idx >= N) return;\n" +
                     body + "\n    }\n";


  const auto &device = initialize_cuda_device(device_id);
  KernelCacheKey key{device_id, kernel};
  auto &counter = compile_counter_instance();
  auto entry =
      load_or_get_cached_kernel(key, device, kernel, "user_kernel", counter);
  ContextSetGuard context_guard(entry->context->ctx, device_id);

  // Launch
  std::vector<void *> column_ptrs;
  column_ptrs.reserve(table.columns.size());
  for (const auto &c : table.columns) {
    column_ptrs.push_back(c.device_ptr.get());
  }
  std::vector<void *> args;
  args.reserve(column_ptrs.size() + 2);
  for (auto &p : column_ptrs) args.push_back(&p);
  args.push_back(&d_output);
  args.push_back(&N);

  // Determine launch configuration. If block_size is zero, choose an
  // occupancy-optimised size using cuOccupancyMaxPotentialBlockSize.
  int threads = block_size;
  int minGridSize = 0;
  if (threads <= 0) {
    CU_CHECK(cuOccupancyMaxPotentialBlockSize(&minGridSize, &threads,
                                              entry->function, nullptr, 0, 0));
  }
  int blocks = std::max(minGridSize, (N + threads - 1) / threads);
  CU_CHECK(cuLaunchKernel(entry->function, blocks, 1, 1, threads, 1, 1, 0, 0,
                          args.data(), nullptr));
  CU_CHECK(cuCtxSynchronize());
}

// Compile and launch a parallel GROUP BY SUM kernel. Rows are first transformed
// into key/value pairs using a JIT kernel, then grouped with Thrust
// sort/reduce-by-key to support thousands of groups efficiently.
void jit_group_sum(const std::string &val_expr_code,
                   const std::string &key_expr_code, float *d_price,
                   int *d_quantity, float *d_out_vals, int *d_out_keys,
                   int *d_count, int N, int device_id) {

  std::string custom_code;
  {
    std::ifstream in("custom.cu");
    if (in) {
      std::stringstream ss;
      ss << in.rdbuf();
      custom_code = ss.str();
    }
  }

  std::string kernel = custom_code + R"(
    extern "C" __global__
    void group_kernel(float* price, int* quantity, float* tmp_vals, int* tmp_keys,
                      int N){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int idx = tid; idx < N; idx += stride){
            float val = )" + val_expr_code + R"(;
            int key = )" + key_expr_code + R"(;
            tmp_keys[idx] = key;
            tmp_vals[idx] = val;
        }
    }
  )";

  const auto &device = initialize_cuda_device(device_id);
  std::string ptx = compile_cuda_source(kernel, "group.cu", device.arch_flag);

  CUdevice cuDevice = device.device;
  struct CuPrimaryContextGuard {
    CUdevice device{0};
    CUcontext ctx{nullptr};
    CUcontext prev_ctx{nullptr};
    bool retained{false};
    int prev_runtime_device{-1};
    bool has_prev_runtime_device{false};
    ~CuPrimaryContextGuard() {
      if (ctx) {
        cuCtxSetCurrent(prev_ctx);
      }
      if (has_prev_runtime_device) {
        cudaSetDevice(prev_runtime_device);
      }
      if (retained) {
        cuDevicePrimaryCtxRelease(device);
      }
    }
  } context;
  struct CuModuleGuard {
    CUmodule mod{nullptr};
    ~CuModuleGuard() {
      if (mod)
        cuModuleUnload(mod);
    }
  } module;
  CUfunction kernel_func;

  CU_CHECK(cuCtxGetCurrent(&context.prev_ctx));
  int runtime_device = 0;
  cudaError_t get_device_result = cudaGetDevice(&runtime_device);
  if (get_device_result == cudaSuccess) {
    context.prev_runtime_device = runtime_device;
    context.has_prev_runtime_device = true;
  } else if (get_device_result != cudaErrorNoDevice) {
    throw std::runtime_error("CUDA runtime error: " +
                             std::string(cudaGetErrorString(get_device_result)));
  }

  CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));
  context.device = cuDevice;
  CU_CHECK(cuDevicePrimaryCtxRetain(&context.ctx, cuDevice));
  context.retained = true;
  CU_CHECK(cuCtxSetCurrent(context.ctx));
  CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(&kernel_func, module.mod, "group_kernel"));

  CUDA_RUNTIME_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  DeviceBuffer<float> tmp_vals(static_cast<size_t>(N));
  DeviceBuffer<int> tmp_keys(static_cast<size_t>(N));

  int threads = 0;
  int minGridSize = 0;
  CU_CHECK(cuOccupancyMaxPotentialBlockSize(&minGridSize, &threads, kernel_func,
                                            nullptr, 0, 0));
  if (threads <= 0)
    threads = 256;
  int blocks = std::max(1, std::max(minGridSize, (N + threads - 1) / threads));
  float *tmp_vals_ptr = tmp_vals.get();
  int *tmp_keys_ptr = tmp_keys.get();
  std::vector<void *> args{&d_price, &d_quantity, &tmp_vals_ptr,
                           &tmp_keys_ptr, &N};
  CU_CHECK(cuLaunchKernel(kernel_func, blocks, 1, 1, threads, 1, 1, 0,
                          reinterpret_cast<CUstream>(cudaStreamPerThread),
                          args.data(), nullptr));
  CU_CHECK(cuCtxSynchronize());

  auto exec_policy = thrust::cuda::par.on(cudaStreamPerThread);
  auto tmp_keys_begin = thrust::device_pointer_cast(tmp_keys.get());
  auto tmp_vals_begin = thrust::device_pointer_cast(tmp_vals.get());
  thrust::sort_by_key(exec_policy, tmp_keys_begin, tmp_keys_begin + N,
                      tmp_vals_begin);

  auto out_keys_begin = thrust::device_pointer_cast(d_out_keys);
  auto out_vals_begin = thrust::device_pointer_cast(d_out_vals);
  auto reduce_end = thrust::reduce_by_key(exec_policy, tmp_keys_begin,
                                          tmp_keys_begin + N, tmp_vals_begin,
                                          out_keys_begin, out_vals_begin);

  int host_count = static_cast<int>(reduce_end.first - out_keys_begin);
  CUDA_RUNTIME_CHECK(
      cudaMemcpy(d_count, &host_count, sizeof(int), cudaMemcpyHostToDevice));
}

template <typename T> struct DeviceToFloat {
  __host__ __device__ float operator()(T v) const {
    return static_cast<float>(v);
  }
};

template <typename T> struct DeviceToInt64 {
  __host__ __device__ int64_t operator()(T v) const {
    return static_cast<int64_t>(v);
  }
};

template <typename KeyT, typename ValueT, typename ReduceOp>
int jit_group_reduce_impl(const void *d_keys_raw, const void *d_values_raw,
                          int64_t *d_out_keys, float *d_out_vals, int N,
                          ReduceOp reduce_op) {
  auto exec_policy = thrust::cuda::par.on(cudaStreamPerThread);
  const auto *d_keys = static_cast<const KeyT *>(d_keys_raw);
  const auto *d_values = static_cast<const ValueT *>(d_values_raw);

  DeviceBuffer<KeyT> tmp_keys(static_cast<size_t>(N));
  DeviceBuffer<float> tmp_vals(static_cast<size_t>(N));
  CUDA_RUNTIME_CHECK(cudaMemcpy(tmp_keys.get(), d_keys, sizeof(KeyT) * static_cast<size_t>(N),
                                cudaMemcpyDeviceToDevice));
  thrust::transform(exec_policy, d_values, d_values + N,
                    thrust::device_pointer_cast(tmp_vals.get()),
                    DeviceToFloat<ValueT>());

  auto tmp_keys_begin = thrust::device_pointer_cast(tmp_keys.get());
  auto tmp_vals_begin = thrust::device_pointer_cast(tmp_vals.get());
  thrust::sort_by_key(exec_policy, tmp_keys_begin, tmp_keys_begin + N, tmp_vals_begin);

  DeviceBuffer<KeyT> reduced_keys(static_cast<size_t>(N));
  auto reduced_keys_begin = thrust::device_pointer_cast(reduced_keys.get());
  auto out_vals_begin = thrust::device_pointer_cast(d_out_vals);
  auto reduce_end = thrust::reduce_by_key(exec_policy, tmp_keys_begin, tmp_keys_begin + N,
                                          tmp_vals_begin, reduced_keys_begin,
                                          out_vals_begin, thrust::equal_to<KeyT>(),
                                          reduce_op);
  int host_count = static_cast<int>(reduce_end.first - reduced_keys_begin);
  thrust::transform(exec_policy, reduced_keys_begin, reduced_keys_begin + host_count,
                    thrust::device_pointer_cast(d_out_keys), DeviceToInt64<KeyT>());
  return host_count;
}

template <typename KeyT>
int jit_group_count_impl(const void *d_keys_raw, int64_t *d_out_keys,
                         float *d_out_counts, int N) {
  auto exec_policy = thrust::cuda::par.on(cudaStreamPerThread);
  const auto *d_keys = static_cast<const KeyT *>(d_keys_raw);

  DeviceBuffer<KeyT> tmp_keys(static_cast<size_t>(N));
  CUDA_RUNTIME_CHECK(cudaMemcpy(tmp_keys.get(), d_keys, sizeof(KeyT) * static_cast<size_t>(N),
                                cudaMemcpyDeviceToDevice));
  auto tmp_keys_begin = thrust::device_pointer_cast(tmp_keys.get());
  thrust::sort(exec_policy, tmp_keys_begin, tmp_keys_begin + N);

  DeviceBuffer<KeyT> reduced_keys(static_cast<size_t>(N));
  DeviceBuffer<int> reduced_counts(static_cast<size_t>(N));
  auto reduced_keys_begin = thrust::device_pointer_cast(reduced_keys.get());
  auto reduced_counts_begin = thrust::device_pointer_cast(reduced_counts.get());
  auto reduce_end = thrust::reduce_by_key(
      exec_policy, tmp_keys_begin, tmp_keys_begin + N,
      thrust::make_constant_iterator(1), reduced_keys_begin, reduced_counts_begin);

  int host_count = static_cast<int>(reduce_end.first - reduced_keys_begin);
  thrust::transform(exec_policy, reduced_keys_begin, reduced_keys_begin + host_count,
                    thrust::device_pointer_cast(d_out_keys), DeviceToInt64<KeyT>());
  thrust::transform(exec_policy, reduced_counts_begin, reduced_counts_begin + host_count,
                    thrust::device_pointer_cast(d_out_counts), DeviceToFloat<int>());
  return host_count;
}

int jit_group_reduce_by_key(const void *d_keys, DataType key_type,
                            const void *d_values, DataType value_type,
                            GroupReduceOp op, int64_t *d_out_keys,
                            float *d_out_vals, int N, int device_id) {
  CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));
  if (N <= 0) return 0;

  if (key_type == DataType::Int32) {
    switch (value_type) {
    case DataType::Int32:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int32_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int32_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int32_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Int64:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int32_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int32_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int32_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Float32:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int32_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int32_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int32_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Float64:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int32_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int32_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int32_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::String:
      break;
    }
  } else if (key_type == DataType::Int64) {
    switch (value_type) {
    case DataType::Int32:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int64_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int64_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int64_t, int32_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Int64:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int64_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int64_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int64_t, int64_t>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Float32:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int64_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int64_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int64_t, float>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::Float64:
      if (op == GroupReduceOp::Sum) return jit_group_reduce_impl<int64_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::plus<float>());
      if (op == GroupReduceOp::Min) return jit_group_reduce_impl<int64_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::minimum<float>());
      return jit_group_reduce_impl<int64_t, double>(d_keys, d_values, d_out_keys, d_out_vals, N, thrust::maximum<float>());
    case DataType::String:
      break;
    }
  }

  throw std::runtime_error("jit_group_reduce_by_key requires integer keys and numeric values");
}

int jit_group_count_by_key(const void *d_keys, DataType key_type,
                           int64_t *d_out_keys, float *d_out_counts, int N,
                           int device_id) {
  CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));
  if (N <= 0) return 0;
  if (key_type == DataType::Int32) {
    return jit_group_count_impl<int32_t>(d_keys, d_out_keys, d_out_counts, N);
  }
  if (key_type == DataType::Int64) {
    return jit_group_count_impl<int64_t>(d_keys, d_out_keys, d_out_counts, N);
  }
  throw std::runtime_error("jit_group_count_by_key requires integer keys");
}

void jit_sort_pairs(int *d_keys, float *d_vals, int count, bool ascending,
                    int device_id) {
  CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));

  thrust::device_ptr<int> key_begin(d_keys);
  thrust::device_ptr<float> val_begin(d_vals);

  if (ascending) {
    thrust::sort_by_key(key_begin, key_begin + count, val_begin,
                        thrust::less<int>());
  } else {
    thrust::sort_by_key(key_begin, key_begin + count, val_begin,
                        thrust::greater<int>());
  }

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

void jit_sort_float(float *d_vals, int count, bool ascending, int device_id) {
  CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));

  thrust::device_ptr<float> begin(d_vals);
  thrust::device_ptr<float> end = begin + count;
  if (ascending) {
    thrust::sort(begin, end, thrust::less<float>());
  } else {
    thrust::sort(begin, end, thrust::greater<float>());
  }

  CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}
