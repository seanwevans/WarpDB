// src/jit.cpp
#include "jit.hpp"
#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <stdexcept>

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
                            const std::string &condition_code, float *d_price,
                            int *d_quantity, float *d_output, int N) {
  std::string body;
  if (!condition_code.empty()) {
    body = "if (" + condition_code + ") {\n    output[idx] = " + expr_code +
           ";\n}";
  } else {
    body = "output[idx] = " + expr_code + ";";
  }

  std::string kernel = R"(
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
  const char *opts[] = {"--gpu-architecture=compute_70"};
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
  CUcontext context;
  CUmodule module;
  CUfunction kernel_func;
  CU_CHECK(cuInit(0));
  CU_CHECK(cuDeviceGet(&cuDevice, 0));
  CU_CHECK(cuCtxCreate(&context, 0, cuDevice));
  CU_CHECK(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr));
  CU_CHECK(cuModuleGetFunction(&kernel_func, module, "user_kernel"));

  // Launch
  void *args[] = {&d_price, &d_quantity, &d_output, &N};
  int threads = 128;
  int blocks = (N + threads - 1) / threads;
  CU_CHECK(cuLaunchKernel(kernel_func, blocks, 1, 1, threads, 1, 1, 0, 0, args,
                          nullptr));
  CU_CHECK(cuCtxSynchronize());

  // Cleanup
  cuModuleUnload(module);
  cuCtxDestroy(context);
}
