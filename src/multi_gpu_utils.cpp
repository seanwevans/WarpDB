#include "multi_gpu_utils.hpp"
#include <cuda_runtime.h>
#include <algorithm>

std::vector<float> run_multi_gpu_jit_host(const HostTable &host,
                                          const std::string &expr_cuda,
                                          const std::string &cond_cuda) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        Table dtab = upload_to_gpu(host);
        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * host.num_rows());
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out, 0);
        std::vector<float> result(host.num_rows());
        cudaMemcpy(result.data(), d_out, sizeof(float) * host.num_rows(),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        for (auto &c : dtab.columns) cudaFree(c.device_ptr);
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
        cudaSetDevice(dev);
        Table dtab = upload_to_gpu(sub);

        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * local_N);

        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out, dev);

        cudaMemcpy(results.data() + start, d_out, sizeof(float) * local_N,
                   cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        for (auto &c : dtab.columns) cudaFree(c.device_ptr);
    }

    return results;
}
