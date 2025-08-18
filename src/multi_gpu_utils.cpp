#include "multi_gpu_utils.hpp"
#include "cuda_utils.hpp"
#include <algorithm>

// All CUDA runtime calls are wrapped in CUDA_CHECK, which throws
// std::runtime_error on failure. This allows errors to propagate to callers
// for handling while keeping this function's control flow simple.
std::vector<float> run_multi_gpu_jit_host(const HostTable &host,
                                          const std::string &expr_cuda,
                                          const std::string &cond_cuda) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        Table dtab = upload_to_gpu(host);
        DeviceBuffer<float> d_out(host.num_rows());
        // Use block_size=0 to auto-select via occupancy.
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out.get(), 0, 0);
        std::vector<float> result(host.num_rows());
        CUDA_CHECK(cudaMemcpy(result.data(), d_out.get(),
                              sizeof(float) * host.num_rows(),
                              cudaMemcpyDeviceToHost));
        return result;
    }

    int N = host.num_rows();
    int chunk = (N + device_count - 1) / device_count;
    std::vector<float> results(N);

    for (int dev = 0; dev < device_count; ++dev) {
        int start = dev * chunk;
        int end = std::min(start + chunk, N);
        if (start >= end)
            break;
        int local_N = end - start;

        HostTable sub;
        sub.columns.resize(host.columns.size());
        for (size_t i=0;i<host.columns.size();++i) {
            sub.columns[i].name = host.columns[i].name;
            sub.columns[i].type = host.columns[i].type;
            if (host.columns[i].type == DataType::Float32) {
                auto &vec = std::get<std::vector<float>>(host.columns[i].data);
                sub.columns[i].data = std::vector<float>(vec.begin()+start, vec.begin()+end);
            } else if (host.columns[i].type == DataType::Int32) {
                auto &vec = std::get<std::vector<int32_t>>(host.columns[i].data);
                sub.columns[i].data = std::vector<int32_t>(vec.begin()+start, vec.begin()+end);
            }
        }
        CUDA_CHECK(cudaSetDevice(dev));
        Table dtab = upload_to_gpu(sub);

        DeviceBuffer<float> d_out(local_N);

        // Each device launches with block_size=0 for automatic selection.
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out.get(), dev, 0);

        CUDA_CHECK(cudaMemcpy(results.data() + start, d_out.get(),
                              sizeof(float) * local_N,
                              cudaMemcpyDeviceToHost));
    }

    return results;
}
