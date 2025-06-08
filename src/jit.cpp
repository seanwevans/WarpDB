// src/jit.cpp
#include "jit.hpp"
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <sstream>
#include <stdexcept>
#include <vector>

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

void jit_compile_and_launch(const std::string &expr_code,
                            const std::string &condition_code,
                            float *d_price, int *d_quantity,
                            float *d_output, int N, int device_id) {

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

  std::string kernel = custom_code + R"(
    extern "C" __global__
    void user_kernel(float* price, int* quantity, float* output, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
    )" + body + R"(
    }
    )";


  // Compile
  // RAII wrapper to destroy the NVRTC program in all code paths.
  // Fix: capture compile result and clean up before throwing on failure.
  struct NvrtcProgramGuard {
    nvrtcProgram prog{nullptr};
    ~NvrtcProgramGuard() {
      if (prog) {
        nvrtcDestroyProgram(&prog); // ensure destruction on all paths
      }
    }
  } prog;

  NVRTC_CHECK(
      nvrtcCreateProgram(&prog.prog, kernel.c_str(), "user_kernel.cu", 0,
                         nullptr, nullptr));

  // Determine the compute capability of the target device and compile
  // for that architecture rather than a hard coded value.
  CU_CHECK(cuInit(0));
  CUdevice query_device;
  CU_CHECK(cuDeviceGet(&query_device, device_id));
  int major = 0, minor = 0;
  CU_CHECK(cuDeviceGetAttribute(&major,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                query_device));
  CU_CHECK(cuDeviceGetAttribute(&minor,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                query_device));
  std::string arch_flag = "--gpu-architecture=compute_" +
                          std::to_string(major) + std::to_string(minor);
  const char *opts[] = {arch_flag.c_str()};
  nvrtcResult compileResult = nvrtcCompileProgram(prog.prog, 1, opts);

  size_t logSize;
  nvrtcGetProgramLogSize(prog.prog, &logSize);
  std::string log(logSize, '\0');
  nvrtcGetProgramLog(prog.prog, &log[0]);
  if (compileResult != NVRTC_SUCCESS) {
    std::cerr << "NVRTC Compile Log:\n" << log << "\n";
    // Explicitly destroy before throwing to avoid leaks
    nvrtcDestroyProgram(&prog.prog);
    prog.prog = nullptr;
    throw std::runtime_error("Kernel compilation failed.");
  }

  size_t ptxSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog.prog, &ptxSize));
  std::string ptx(ptxSize, '\0');
  NVRTC_CHECK(nvrtcGetPTX(prog.prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog.prog));
  prog.prog = nullptr;

  // Load to CUDA
  CUdevice cuDevice;
  struct CuContextGuard {
    CUcontext ctx{nullptr};
    ~CuContextGuard() {
      if (ctx) cuCtxDestroy(ctx);
    }
  } context;
  struct CuModuleGuard {
    CUmodule mod{nullptr};
    ~CuModuleGuard() {
      if (mod) cuModuleUnload(mod);
    }
  } module;
  CUfunction kernel_func;
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cuDevice, device_id));
  CU_CHECK(cuCtxCreate(&context.ctx, 0, cuDevice));
  CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(&kernel_func, module.mod, "user_kernel"));

  // Launch
  std::vector<void *> args;
  args.push_back(&d_price);
  args.push_back(&d_quantity);
  args.push_back(&d_output);
  args.push_back(&N);

  int threads = 128;
  int blocks = (N + threads - 1) / threads;
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

  struct NvrtcProgramGuard { nvrtcProgram prog{nullptr}; ~NvrtcProgramGuard(){ if(prog) nvrtcDestroyProgram(&prog);} } prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog.prog, kernel.c_str(), "group.cu", 0, nullptr, nullptr));

  CU_CHECK(cuInit(0));
  CUdevice query_device; CU_CHECK(cuDeviceGet(&query_device, device_id));
  int major=0, minor=0; CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, query_device));
  CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, query_device));
  std::string arch_flag = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char* opts[] = { arch_flag.c_str() };
  nvrtcResult compileResult = nvrtcCompileProgram(prog.prog, 1, opts);

  size_t logSize; nvrtcGetProgramLogSize(prog.prog, &logSize);
  std::string log(logSize, '\0'); nvrtcGetProgramLog(prog.prog, &log[0]);
  if(compileResult != NVRTC_SUCCESS){ std::cerr << "NVRTC Compile Log:\n" << log << "\n"; nvrtcDestroyProgram(&prog.prog); prog.prog=nullptr; throw std::runtime_error("Kernel compilation failed."); }

  size_t ptxSize; NVRTC_CHECK(nvrtcGetPTXSize(prog.prog, &ptxSize));
  std::string ptx(ptxSize, '\0'); NVRTC_CHECK(nvrtcGetPTX(prog.prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog.prog)); prog.prog=nullptr;

  CUdevice cuDevice; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx); }} context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod); }} module; CUfunction kernel_func;
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cuDevice, device_id));
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

  struct NvrtcProgramGuard{ nvrtcProgram prog{nullptr}; ~NvrtcProgramGuard(){ if(prog) nvrtcDestroyProgram(&prog);} } prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog.prog, kernel.c_str(), "sort.cu", 0, nullptr, nullptr));
  CU_CHECK(cuInit(0)); CUdevice query_device; CU_CHECK(cuDeviceGet(&query_device, device_id));
  int major=0, minor=0; CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, query_device));
  CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, query_device));
  std::string arch_flag = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  const char* opts[]={arch_flag.c_str()}; nvrtcResult compileResult = nvrtcCompileProgram(prog.prog,1,opts);
  size_t logSize; nvrtcGetProgramLogSize(prog.prog,&logSize); std::string log(logSize,'\0'); nvrtcGetProgramLog(prog.prog,&log[0]);
  if(compileResult!=NVRTC_SUCCESS){ std::cerr<<"NVRTC Compile Log:\n"<<log<<"\n"; nvrtcDestroyProgram(&prog.prog); prog.prog=nullptr; throw std::runtime_error("Kernel compilation failed."); }
  size_t ptxSize; NVRTC_CHECK(nvrtcGetPTXSize(prog.prog,&ptxSize)); std::string ptx(ptxSize,'\0'); NVRTC_CHECK(nvrtcGetPTX(prog.prog,&ptx[0])); NVRTC_CHECK(nvrtcDestroyProgram(&prog.prog)); prog.prog=nullptr;
  CUdevice cuDevice; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx);} } context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod);} } module; CUfunction func;
  CU_CHECK(cuInit(0)); CU_CHECK(cuDeviceGet(&cuDevice, device_id)); CU_CHECK(cuCtxCreate(&context.ctx,0,cuDevice)); CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(),0,nullptr,nullptr)); CU_CHECK(cuModuleGetFunction(&func, module.mod, "sort_kernel"));
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

  struct NvrtcProgramGuard{ nvrtcProgram prog{nullptr}; ~NvrtcProgramGuard(){ if(prog) nvrtcDestroyProgram(&prog);} } prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog.prog, kernel.c_str(), "sortf.cu",0,nullptr,nullptr));
  CU_CHECK(cuInit(0)); CUdevice query_device; CU_CHECK(cuDeviceGet(&query_device, device_id)); int major=0, minor=0; CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, query_device)); CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, query_device));
  std::string arch_flag="--gpu-architecture=compute_"+std::to_string(major)+std::to_string(minor); const char* opts[]={arch_flag.c_str()}; nvrtcResult compileResult=nvrtcCompileProgram(prog.prog,1,opts);
  size_t logSize; nvrtcGetProgramLogSize(prog.prog,&logSize); std::string log(logSize,'\0'); nvrtcGetProgramLog(prog.prog,&log[0]); if(compileResult!=NVRTC_SUCCESS){ std::cerr<<"NVRTC Compile Log:\n"<<log<<"\n"; nvrtcDestroyProgram(&prog.prog); prog.prog=nullptr; throw std::runtime_error("Kernel compilation failed."); }
  size_t ptxSize; NVRTC_CHECK(nvrtcGetPTXSize(prog.prog,&ptxSize)); std::string ptx(ptxSize,'\0'); NVRTC_CHECK(nvrtcGetPTX(prog.prog,&ptx[0])); NVRTC_CHECK(nvrtcDestroyProgram(&prog.prog)); prog.prog=nullptr;
  CUdevice cuDevice; struct CuContextGuard{ CUcontext ctx{nullptr}; ~CuContextGuard(){ if(ctx) cuCtxDestroy(ctx);} } context; struct CuModuleGuard{ CUmodule mod{nullptr}; ~CuModuleGuard(){ if(mod) cuModuleUnload(mod);} } module; CUfunction func;
  CU_CHECK(cuInit(0)); CU_CHECK(cuDeviceGet(&cuDevice, device_id)); CU_CHECK(cuCtxCreate(&context.ctx,0,cuDevice)); CU_CHECK(cuModuleLoadDataEx(&module.mod, ptx.c_str(),0,nullptr,nullptr)); CU_CHECK(cuModuleGetFunction(&func, module.mod, "sortf"));
  int asc=ascending?1:0; std::vector<void*> args{&d_vals,&count,&asc}; CU_CHECK(cuLaunchKernel(func,1,1,1,1,1,1,0,0,args.data(),nullptr)); CU_CHECK(cuCtxSynchronize());
}
