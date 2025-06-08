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
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab.d_price,
                               dtab.d_quantity, d_out, host.num_rows(), 0);
        std::vector<float> result(host.num_rows());
        cudaMemcpy(result.data(), d_out, sizeof(float) * host.num_rows(),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        cudaFree(dtab.d_price);
        cudaFree(dtab.d_quantity);
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
        sub.price.assign(host.price.begin() + start, host.price.begin() + end);
        sub.quantity.assign(host.quantity.begin() + start,
                            host.quantity.begin() + end);
        cudaSetDevice(dev);
        Table dtab = upload_to_gpu(sub);

        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * local_N);

        jit_compile_and_launch(expr_cuda, cond_cuda, dtab.d_price,
                               dtab.d_quantity, d_out, local_N, dev);

        cudaMemcpy(results.data() + start, d_out, sizeof(float) * local_N,
                   cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(dtab.d_price);
        cudaFree(dtab.d_quantity);
    }

    return results;
}
