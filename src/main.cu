// src/main.cu

#include <iostream>

#include "csv_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"
#include "arrow_utils.hpp"
#include "optimizer.hpp"

// Simple multi-GPU execution of a JIT compiled expression.
// Splits the input table across all available devices and aggregates the output.
void run_multi_gpu_jit(const std::string &expr_cuda,
                       const std::string &cond_cuda) {
  HostTable host = load_csv_to_host("data/test.csv");

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    std::cout << "Only " << device_count
              << " GPU detected. Skipping multi-device example.\n";
    return;
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
    Table dtab = upload_to_gpu(sub);

    float *d_out;
    cudaMalloc(&d_out, sizeof(float) * local_N);

    jit_compile_and_launch(expr_cuda, cond_cuda, dtab.d_price, dtab.d_quantity,
                           d_out, local_N, dev);

    cudaMemcpy(results.data() + start, d_out, sizeof(float) * local_N,
               cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(dtab.d_price);
    cudaFree(dtab.d_quantity);
  }

  for (int i = 0; i < N; ++i) {
    std::cout << "MultiGPU Result[" << i << "] = " << results[i] << "\n";
  }
}

__global__ void print_first_few(float *price, int *quantity, int N) {
  int idx = threadIdx.x;
  if (idx < N && idx < 4) {
    printf("Row %d: price = %.2f, quantity = %d\n", idx, price[idx],
           quantity[idx]);
  }
}

__global__ void filter_price_gt(float *price, int *quantity, float *out_price,
                                int *out_quantity, int *out_count, int N,
                                float threshold) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;

  if (price[idx] > threshold) {
    int write_idx = atomicAdd(out_count, 1);
    out_price[write_idx] = price[idx];
    out_quantity[write_idx] = quantity[idx];
  }
}

__global__ void project_columns(float *price, int *quantity, float *out_price,
                                int *out_quantity, int *out_count, int N,
                                bool select_price, bool select_quantity) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;

  int write_idx = atomicAdd(out_count, 1);
  if (select_price)
    out_price[write_idx] = price[idx];
  if (select_quantity)
    out_quantity[write_idx] = quantity[idx];
}

__global__ void project_revenue(float *price, int *quantity, float *revenue_out,
                                int *out_count, int N, float threshold) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;

  if (price[idx] > threshold) {
    int write_idx = atomicAdd(out_count, 1);
    revenue_out[write_idx] = price[idx] * quantity[idx];
  }
}

__global__ void project_revenue_and_adjusted(float *price, int *quantity,
                                             float *revenue_out,
                                             float *adjusted_price_out,
                                             int *out_count, int N,
                                             float threshold) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;

  if (price[idx] > threshold) {
    int write_idx = atomicAdd(out_count, 1);
    revenue_out[write_idx] = price[idx] * quantity[idx];
    adjusted_price_out[write_idx] = price[idx] - 10.0f;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./warpdb \"<expression>\"\n";
    return 1;
  }
  std::string user_query = argv[1];
  std::string upper_query = user_query;
  for (auto &c : upper_query)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

  std::string expr_part = user_query;
  std::string where_part;
  auto where_pos = upper_query.find("WHERE");
  if (where_pos != std::string::npos) {
    expr_part = user_query.substr(0, where_pos);
    where_part = user_query.substr(where_pos + 5); // skip "WHERE"
  }

  std::cout << "Expr: " << expr_part << "\n";
  if (!where_part.empty())
    std::cout << "Where: " << where_part << "\n";

  Table table = load_csv_to_gpu("data/test.csv");
  std::cout << "Loaded " << table.num_rows << " rows.\n";
  float *d_price = table.get_column_ptr<float>("price");
  int *d_quantity = table.get_column_ptr<int>("quantity");

  int *d_quantity_filtered;
  int *d_count;
  int *d_selected_quantity;
  int *d_select_count;
  int *d_revenue_count;
  int *d_multi_count;

  float *d_price_filtered;
  float *d_selected_price;
  float *d_revenue;
  float *d_revenue_multi;
  float *d_adjusted_price;
  float *d_jit_output;

  int h_count;
  int h_select_count;
  int h_revenue_count;
  int h_multi_count;

  bool select_price = true;
  bool select_quantity = true;

  int threads = 128;
  int blocks = (table.num_rows + threads - 1) / threads;

  float threshold = 25.0f;

  cudaMalloc(&d_price_filtered, sizeof(float) * table.num_rows);
  cudaMalloc(&d_quantity_filtered, sizeof(int) * table.num_rows);
  cudaMalloc(&d_count, sizeof(int));
  cudaMalloc(&d_selected_price, sizeof(float) * table.num_rows);
  cudaMalloc(&d_selected_quantity, sizeof(int) * table.num_rows);
  cudaMalloc(&d_select_count, sizeof(int));
  cudaMalloc(&d_revenue, sizeof(float) * table.num_rows);
  cudaMalloc(&d_revenue_count, sizeof(int));
  cudaMemset(d_revenue_count, 0, sizeof(int));
  cudaMalloc(&d_revenue_multi, sizeof(float) * table.num_rows);
  cudaMalloc(&d_adjusted_price, sizeof(float) * table.num_rows);
  cudaMalloc(&d_multi_count, sizeof(int));
  cudaMalloc(&d_jit_output, sizeof(float) * table.num_rows);

  cudaMemset(d_count, 0, sizeof(int));
  cudaMemset(d_select_count, 0, sizeof(int));
  cudaMemset(d_multi_count, 0, sizeof(int));
  std::cout << "Allocated space\n";


  print_first_few<<<1, 4>>>(
#ifdef USE_ARROW
      reinterpret_cast<float *>(table.d_price->mutable_data()),
      reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
      table.d_price, table.d_quantity,
#endif
      table.num_rows);
  cudaDeviceSynchronize();

  filter_price_gt<<<blocks, threads>>>(
#ifdef USE_ARROW
                                       reinterpret_cast<float *>(table.d_price->mutable_data()),
                                       reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
                                       table.d_price, table.d_quantity,
#endif
                                       d_price_filtered, d_quantity_filtered,
                                       d_count, table.num_rows, threshold);

  print_first_few<<<1, 4>>>(d_price, d_quantity, table.num_rows);
  cudaDeviceSynchronize();

  filter_price_gt<<<blocks, threads>>>(d_price, d_quantity, d_price_filtered,
                                       d_quantity_filtered, d_count,
                                       table.num_rows, threshold);

  cudaDeviceSynchronize();

  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Filtered rows: " << h_count << "\n";

  float *h_price_filtered = new float[h_count];
  int *h_quantity_filtered = new int[h_count];
  cudaMemcpy(h_price_filtered, d_price_filtered, sizeof(float) * h_count,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_quantity_filtered, d_quantity_filtered, sizeof(int) * h_count,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < h_count; ++i) {
    std::cout << "Filtered Row " << i << ": price = " << h_price_filtered[i]
              << ", quantity = " << h_quantity_filtered[i] << "\n";
  }

  std::cout << "\nRunning SELECT projection:\n";

  project_columns<<<blocks, threads>>>(
#ifdef USE_ARROW
      reinterpret_cast<float *>(table.d_price->mutable_data()),
      reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
      table.d_price, table.d_quantity,
#endif
      d_selected_price, d_selected_quantity, d_select_count, table.num_rows,
      select_price, select_quantity);


  cudaDeviceSynchronize();

  cudaMemcpy(&h_select_count, d_select_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  std::cout << "Selected rows: " << h_select_count << "\n";

  float *h_selected_price = new float[h_select_count];
  int *h_selected_quantity = new int[h_select_count];
  cudaMemcpy(h_selected_price, d_selected_price, sizeof(float) * h_select_count,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_selected_quantity, d_selected_quantity,
             sizeof(int) * h_select_count, cudaMemcpyDeviceToHost);
  for (int i = 0; i < h_select_count; ++i) {
    std::cout << "Selected Row " << i;
    if (select_price)
      std::cout << ", price = " << h_selected_price[i];
    if (select_quantity)
      std::cout << ", quantity = " << h_selected_quantity[i];
    std::cout << "\n";
  }

  std::cout << "\nRunning SELECT revenue (price * quantity) with WHERE price > "
               "threshold:\n";

  project_revenue<<<blocks, threads>>>(
#ifdef USE_ARROW
      reinterpret_cast<float *>(table.d_price->mutable_data()),
      reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
      table.d_price, table.d_quantity,
#endif
      d_revenue, d_revenue_count, table.num_rows, threshold);

  cudaDeviceSynchronize();

  cudaMemcpy(&h_revenue_count, d_revenue_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  std::cout << "Computed revenue rows: " << h_revenue_count << "\n";

  float *h_revenue = new float[h_revenue_count];
  cudaMemcpy(h_revenue, d_revenue, sizeof(float) * h_revenue_count,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_revenue_count; ++i) {
    std::cout << "Revenue Row " << i << ": revenue = " << h_revenue[i] << "\n";
  }

  std::cout << "\nRunning SELECT revenue and adjusted_price:\n";
  project_revenue_and_adjusted<<<blocks, threads>>>(

#ifdef USE_ARROW
      reinterpret_cast<float *>(table.d_price->mutable_data()),
      reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
      table.d_price, table.d_quantity,
#endif
      d_revenue_multi, d_adjusted_price, d_multi_count, table.num_rows,
      threshold);

  cudaDeviceSynchronize();

  cudaMemcpy(&h_multi_count, d_multi_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  float *h_revenue_multi = new float[h_multi_count];
  float *h_adjusted_price = new float[h_multi_count];
  cudaMemcpy(h_revenue_multi, d_revenue_multi, sizeof(float) * h_multi_count,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_adjusted_price, d_adjusted_price, sizeof(float) * h_multi_count,
             cudaMemcpyDeviceToHost);

  std::cout << "Computed multi-expression rows: " << h_multi_count << "\n";
  for (int i = 0; i < h_multi_count; ++i) {
    std::cout << "Row " << i << ": revenue = " << h_revenue_multi[i]
              << ", adjusted price = " << h_adjusted_price[i] << "\n";
  }

  std::cout << "\n[ Optimizer Demo ]\n";
  execute_query_optimized(expr_part, where_part, table);


  std::string condition_cuda;
  if (!where_part.empty()) {
    auto cond_tokens = tokenize(where_part);
    auto cond_ast = parse_expression(cond_tokens);
    condition_cuda = cond_ast->to_cuda_expr();
  }

  auto tokens = tokenize(user_query);
  std::cout << "\nTokenized Expression:\n";
  for (auto &tok : tokens) {
    std::cout << "  ["
              << (tok.type == TokenType::Identifier ? "ID"
                  : tok.type == TokenType::Number   ? "NUM"
                  : tok.type == TokenType::Operator ? "OP"
                                                    : "END")
              << "] " << tok.value << "\n";
  }

  // parse
  auto ast = parse_expression(tokens);
  std::cout << "\nParsed Expression (CUDA):\n";

  std::string cuda_expr = ast->to_cuda_expr();
  std::cout << cuda_expr << "\n";

  // compile
  std::cout << "\n[ JIT Kernel Execution for Expression ]\n";

  jit_compile_and_launch(expr_cuda, condition_cuda,
#ifdef USE_ARROW
                         reinterpret_cast<float *>(table.d_price->mutable_data()),
                         reinterpret_cast<int *>(table.d_quantity->mutable_data()),
#else
                         table.d_price, table.d_quantity,
#endif
                         d_jit_output, table.num_rows);

  float *h_jit_output = new float[table.num_rows];
  cudaMemcpy(h_jit_output, d_jit_output, sizeof(float) * table.num_rows,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < table.num_rows; ++i) {
    std::cout << "JIT Result[" << i << "] = " << h_jit_output[i] << "\n";
  }


  // Export results to Arrow for external visualization
  ArrowArray arr;
  ArrowSchema schema;
  export_to_arrow(h_jit_output, table.num_rows, false, &arr, &schema);
  std::cout << "Arrow result length: " << arr.length << "\n";
  arr.release(&arr);
  schema.release(&schema);

  std::cout << "\n[ Multi-GPU JIT Example ]\n";
  run_multi_gpu_jit(expr_cuda, condition_cuda);


  delete[] h_jit_output;

  delete[] h_revenue_multi;
  delete[] h_adjusted_price;
  delete[] h_revenue;
  delete[] h_selected_price;
  delete[] h_selected_quantity;
  delete[] h_price_filtered;
  delete[] h_quantity_filtered;

  cudaFree(d_revenue_multi);
  cudaFree(d_adjusted_price);
  cudaFree(d_multi_count);
  cudaFree(d_jit_output);
  cudaFree(d_revenue);
  cudaFree(d_revenue_count);
  cudaFree(d_selected_price);
  cudaFree(d_selected_quantity);
  cudaFree(d_select_count);
  cudaFree(d_price_filtered);
  cudaFree(d_quantity_filtered);
  cudaFree(d_count);

#ifndef USE_ARROW
  cudaFree(table.d_price);
  cudaFree(table.d_quantity);
  for (auto &col : table.columns) {
    cudaFree(col.device_ptr);
  }
#endif

  return 0;
}
