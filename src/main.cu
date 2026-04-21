// src/main.cu

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <sstream>

#include "csv_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"
#include "arrow_utils.hpp"
#include "optimizer.hpp"
#include "multi_gpu_utils.hpp"
#include "cuda_utils.hpp"
#include "warpdb.hpp"

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
  if (!std::getline(file, header)) {
    std::cerr << "Failed to read header from: " << csv_path << "\n";
    return;
  }
  std::stringstream header_ss(header);
  std::vector<std::string> column_names;
  std::string col_name;
  while (std::getline(header_ss, col_name, ',')) {
    column_names.push_back(col_name);
  }
  if (column_names.empty()) {
    std::cerr << "CSV file has no columns: " << csv_path << "\n";
    return;
  }

  bool finished = false;
  std::vector<DataType> schema;
  std::vector<float> all_results;
  while (!finished) {
    HostTable chunk = load_csv_chunk(file, rows_per_chunk, finished, column_names,
                                     ParsePolicy::Strict, &schema);
    if (chunk.num_rows() == 0 && finished)
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


int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: ./warpdb \"<sql_query>\" [data_file]\n";
      return 1;
    }
    std::string sql_query = argv[1];
    std::string csv_path = "data/test.csv";
    if (argc >= 3)
      csv_path = argv[2];
    WarpDB db(csv_path);
    auto results = db.query_sql(sql_query);

    std::cout << "Query returned " << results.size() << " rows.\n";
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "Result[" << i << "] = " << results[i] << "\n";
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
