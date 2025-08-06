// src/main.cu

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include "csv_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"
#include "arrow_utils.hpp"
#include "optimizer.hpp"
#include "multi_gpu_utils.hpp"
#include "cuda_utils.hpp"

// Convenience wrapper that loads a CSV and prints the results.
void run_multi_gpu_jit(const std::string &csv_path,
                       const std::string &expr_cuda,
                       const std::string &cond_cuda) {
  HostTable host = load_csv_to_host(csv_path);
  auto results = run_multi_gpu_jit_host(host, expr_cuda, cond_cuda);
  for (size_t i = 0; i < results.size(); ++i) {
    std::cout << "MultiGPU Result[" << i << "] = " << results[i] << "\n";
  }
}

// Process a CSV file in chunks using all GPUs and aggregate results.
void run_multi_gpu_jit_large(const std::string &csv_path,
                             const std::string &expr_cuda,
                             const std::string &cond_cuda,
                             int rows_per_chunk = 1000000) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << csv_path << "\n";
    return;
  }

  std::string header;
  std::getline(file, header);

  bool finished = false;
  std::vector<float> all_results;
  while (!finished) {
    HostTable chunk = load_csv_chunk(file, rows_per_chunk, finished);
    if (chunk.num_rows() == 0)
      break;
    auto part = run_multi_gpu_jit_host(chunk, expr_cuda, cond_cuda);
    all_results.insert(all_results.end(), part.begin(), part.end());
  }

  for (size_t i = 0; i < all_results.size(); ++i) {
    std::cout << "Large MultiGPU Result[" << i << "] = " << all_results[i]
              << "\n";
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
  try {
    if (argc < 2) {
      std::cerr << "Usage: ./warpdb \"<expression>\" [data_file]\n";
      return 1;
    }
    std::string user_query = argv[1];
    std::string csv_path = "data/test.csv";
    if (argc >= 3)
      csv_path = argv[2];
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

  Table table = load_csv_to_gpu(csv_path);
  std::cout << "Loaded " << table.num_rows << " rows.\n";
  float *d_price = table.get_column_ptr<float>("price");
  int *d_quantity = table.get_column_ptr<int>("quantity");

  struct TableDeleter {
    void operator()(Table *t) const {
      if (!t) return;
      for (auto &col : t->columns) cudaFree(col.device_ptr);
    }
  };
  std::unique_ptr<Table, TableDeleter> table_guard(&table);

  DeviceBuffer<int> d_quantity_filtered(table.num_rows);
  DeviceBuffer<int> d_count(1);
  DeviceBuffer<int> d_selected_quantity(table.num_rows);
  DeviceBuffer<int> d_select_count(1);
  DeviceBuffer<int> d_revenue_count(1);
  DeviceBuffer<int> d_multi_count(1);

  DeviceBuffer<float> d_price_filtered(table.num_rows);
  DeviceBuffer<float> d_selected_price(table.num_rows);
  DeviceBuffer<float> d_revenue(table.num_rows);
  DeviceBuffer<float> d_revenue_multi(table.num_rows);
  DeviceBuffer<float> d_adjusted_price(table.num_rows);
  DeviceBuffer<float> d_jit_output(table.num_rows);

  int h_count;
  int h_select_count;
  int h_revenue_count;
  int h_multi_count;

  bool select_price = true;
  bool select_quantity = true;

  int threads = 128;
  int blocks = (table.num_rows + threads - 1) / threads;

  float threshold = 25.0f;

  CUDA_CHECK(cudaMemset(d_revenue_count.get(), 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count.get(), 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_select_count.get(), 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_multi_count.get(), 0, sizeof(int)));
  std::cout << "Allocated space\n";


  print_first_few<<<1, 4>>>(d_price, d_quantity, table.num_rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  filter_price_gt<<<blocks, threads>>>(d_price, d_quantity,
                                       d_price_filtered.get(),
                                       d_quantity_filtered.get(), d_count.get(),
                                       table.num_rows, threshold);

  print_first_few<<<1, 4>>>(d_price, d_quantity, table.num_rows);
  CUDA_CHECK(cudaDeviceSynchronize());

  filter_price_gt<<<blocks, threads>>>(d_price, d_quantity,
                                       d_price_filtered.get(),
                                       d_quantity_filtered.get(), d_count.get(),
                                       table.num_rows, threshold);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_count, d_count.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::cout << "Filtered rows: " << h_count << "\n";

  std::vector<float> h_price_filtered(h_count);
  std::vector<int> h_quantity_filtered(h_count);
  CUDA_CHECK(cudaMemcpy(h_price_filtered.data(), d_price_filtered.get(),
                        sizeof(float) * h_count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_quantity_filtered.data(),
                        d_quantity_filtered.get(), sizeof(int) * h_count,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < h_count; ++i) {
    std::cout << "Filtered Row " << i << ": price = "
              << h_price_filtered[i] << ", quantity = "
              << h_quantity_filtered[i] << "\n";
  }

  std::cout << "\nRunning SELECT projection:\n";

  project_columns<<<blocks, threads>>>(d_price, d_quantity,
      d_selected_price.get(), d_selected_quantity.get(), d_select_count.get(),
      table.num_rows, select_price, select_quantity);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_select_count, d_select_count.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::cout << "Selected rows: " << h_select_count << "\n";

  std::vector<float> h_selected_price(h_select_count);
  std::vector<int> h_selected_quantity(h_select_count);
  CUDA_CHECK(cudaMemcpy(h_selected_price.data(), d_selected_price.get(),
                        sizeof(float) * h_select_count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_selected_quantity.data(), d_selected_quantity.get(),
                        sizeof(int) * h_select_count, cudaMemcpyDeviceToHost));
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

  project_revenue<<<blocks, threads>>>(d_price, d_quantity,
      d_revenue.get(), d_revenue_count.get(), table.num_rows, threshold);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_revenue_count, d_revenue_count.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::cout << "Computed revenue rows: " << h_revenue_count << "\n";

  std::vector<float> h_revenue(h_revenue_count);
  CUDA_CHECK(cudaMemcpy(h_revenue.data(), d_revenue.get(),
                        sizeof(float) * h_revenue_count,
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < h_revenue_count; ++i) {
    std::cout << "Revenue Row " << i << ": revenue = " << h_revenue[i] << "\n";
  }

  std::cout << "\nRunning SELECT revenue and adjusted_price:\n";
  project_revenue_and_adjusted<<<blocks, threads>>>(
      d_price, d_quantity,
      d_revenue_multi.get(), d_adjusted_price.get(), d_multi_count.get(),
      table.num_rows, threshold);

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_multi_count, d_multi_count.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::vector<float> h_revenue_multi(h_multi_count);
  std::vector<float> h_adjusted_price(h_multi_count);
  CUDA_CHECK(cudaMemcpy(h_revenue_multi.data(), d_revenue_multi.get(),
                        sizeof(float) * h_multi_count,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_adjusted_price.data(), d_adjusted_price.get(),
                        sizeof(float) * h_multi_count,
                        cudaMemcpyDeviceToHost));

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

  std::string expr_cuda = ast->to_cuda_expr();
  std::cout << expr_cuda << "\n";

  // compile
  std::cout << "\n[ JIT Kernel Execution for Expression ]\n";


  jit_compile_and_launch(expr_cuda, condition_cuda, table, d_jit_output.get());

  std::vector<float> h_jit_output(table.num_rows);
  CUDA_CHECK(cudaMemcpy(h_jit_output.data(), d_jit_output.get(),
                        sizeof(float) * table.num_rows,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < table.num_rows; ++i) {
    std::cout << "JIT Result[" << i << "] = " << h_jit_output[i] << "\n";
  }


  // Export results to Arrow for external visualization
  ArrowArray arr;
  ArrowSchema schema;
  export_to_arrow(h_jit_output.data(), table.num_rows, false, &arr, &schema);
  std::cout << "Arrow result length: " << arr.length << "\n";
  arr.release(&arr);
  schema.release(&schema);

  std::cout << "\n[ Multi-GPU JIT Example ]\n";

  run_multi_gpu_jit(csv_path, expr_cuda, condition_cuda);

  std::cout << "\n[ Large Multi-GPU Example ]\n";
  run_multi_gpu_jit_large(csv_path, expr_cuda, condition_cuda, 1024);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
