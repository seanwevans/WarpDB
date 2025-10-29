// src/jit.cpp
#include "jit.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <algorithm>

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

} // namespace

void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            const Table &table, float *d_output,
                            int device_id, int block_size) {

  int N = table.num_rows;

  std::string body;
  if (!condition_code.empty()) {
    body = "if (" + condition_code + ") {\n    output[idx] = " + expr_code +
           ";\n}";
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


  // Compile and initialize device
  const auto &device = initialize_cuda_device(device_id);
  std::string ptx =
      compile_cuda_source(kernel, "user_kernel.cu", device.arch_flag);

  // Load to CUDA
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
        // Intentionally ignore the result to avoid throwing from a destructor.
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
      if (mod) cuModuleUnload(mod);
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
  CU_CHECK(cuModuleGetFunction(&kernel_func, module.mod, "user_kernel"));

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
                                              kernel_func, nullptr, 0, 0));
  }
  int blocks = std::max(minGridSize, (N + threads - 1) / threads);
  CU_CHECK(cuLaunchKernel(kernel_func, blocks, 1, 1, threads, 1, 1, 0, 0,
                          args.data(), nullptr));
  CU_CHECK(cuCtxSynchronize());

  // Cleanup handled by RAII guards
}

// Compile and launch a naive GROUP BY SUM kernel. This implementation is not
// optimised and processes the table using a single CUDA thread but keeps the
// logic on the GPU for testing purposes.
void jit_group_sum(const std::string &val_expr_code,
                   const std::string &key_expr_code, float *d_price,
                   int *d_quantity, float *d_out_vals, int *d_out_keys,
                   int *d_count, int N, int device_id) {

  std::string custom_code;
  {
    std::ifstream in("custom.cu");
    if (in) {
      std::stringstream ss; ss << in.rdbuf(); custom_code = ss.str();
    }
  }

  std::string kernel = custom_code + R"(
    extern "C" __global__
    void group_kernel(float* price, int* quantity, float* out_vals, int* out_keys,
                      int* out_count, int N){
        if(threadIdx.x==0 && blockIdx.x==0){
            int count = 0;
            for(int idx=0; idx<N; ++idx){
                float val = )" + val_expr_code + R"(;
                int key = )" + key_expr_code + R"(;
                int pos=-1;
                for(int i=0;i<count;i++){
                    if(out_keys[i]==key){ pos=i; break; }
                }
                if(pos==-1){
                    pos=count++;
                    out_keys[pos]=key;
                    out_vals[pos]=0.0f;
                }
                out_vals[pos]+=val;
            }
            *out_count = count;
        }
    }
  )";

  const auto &device = initialize_cuda_device(device_id);
  std::string ptx = compile_cuda_source(kernel, "group.cu", device.arch_flag);

  CUdevice cuDevice = device.device; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx); }} context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod); }} module; CUfunction kernel_func;
  CU_CHECK(cuCtxCreate(&context.ctx, 0, cuDevice));
  CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(&kernel_func, module.mod, "group_kernel"));

  std::vector<void*> args{&d_price,&d_quantity,&d_out_vals,&d_out_keys,&d_count,&N};
  CU_CHECK(cuLaunchKernel(kernel_func, 1,1,1, 1,1,1, 0,0, args.data(), nullptr));
  CU_CHECK(cuCtxSynchronize());
}

void jit_sort_pairs(int *d_keys, float *d_vals, int count, bool ascending,
                    int device_id) {
  std::string kernel = R"(
    extern "C" __global__
    void sort_kernel(int* keys, float* vals, int count, int asc){
        if(threadIdx.x==0 && blockIdx.x==0){
            for(int i=0;i<count-1;i++){
                for(int j=0;j<count-i-1;j++){
                    bool cond = asc ? keys[j] > keys[j+1] : keys[j] < keys[j+1];
                    if(cond){
                        int kt = keys[j]; keys[j]=keys[j+1]; keys[j+1]=kt;
                        float vt = vals[j]; vals[j]=vals[j+1]; vals[j+1]=vt;
                    }
                }
            }
        }
    }
  )";

  const auto &device = initialize_cuda_device(device_id);
  std::string ptx =
      compile_cuda_source(kernel, "sort.cu", device.arch_flag);
  CUdevice cuDevice = device.device; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx);} } context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod);} } module; CUfunction func;
  CU_CHECK(cuCtxCreate(&context.ctx,0,cuDevice)); CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(),0,nullptr,nullptr)); CU_CHECK(cuModuleGetFunction(&func, module.mod, "sort_kernel"));
  int asc = ascending ? 1 : 0; std::vector<void*> args{&d_keys,&d_vals,&count,&asc};
  CU_CHECK(cuLaunchKernel(func,1,1,1,1,1,1,0,0,args.data(),nullptr)); CU_CHECK(cuCtxSynchronize());
}

void jit_sort_float(float *d_vals, int count, bool ascending, int device_id) {
  std::string kernel = R"(
    extern "C" __global__
    void sortf(float* vals, int count, int asc){
        if(threadIdx.x==0 && blockIdx.x==0){
            for(int i=0;i<count-1;i++){
                for(int j=0;j<count-i-1;j++){
                    bool cond = asc ? vals[j] > vals[j+1] : vals[j] < vals[j+1];
                    if(cond){ float t=vals[j]; vals[j]=vals[j+1]; vals[j+1]=t; }
                }
            }
        }
    }
  )";

  const auto &device = initialize_cuda_device(device_id);
  std::string ptx =
      compile_cuda_source(kernel, "sortf.cu", device.arch_flag);
  CUdevice cuDevice = device.device; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx);} } context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod);} } module; CUfunction func;
  CU_CHECK(cuCtxCreate(&context.ctx,0,cuDevice)); CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(),0,nullptr,nullptr)); CU_CHECK(cuModuleGetFunction(&func, module.mod, "sortf"));
  int asc=ascending?1:0; std::vector<void*> args{&d_vals,&count,&asc}; CU_CHECK(cuLaunchKernel(func,1,1,1,1,1,1,0,0,args.data(),nullptr)); CU_CHECK(cuCtxSynchronize());
}
