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
  const auto &out = std::get<std::vector<float>>(results);
  for (size_t i = 0; i < out.size(); ++i) {
    std::cout << "MultiGPU Result[" << i << "] = " << out[i] << "\n";
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
  ColumnData all_results = std::vector<float>{};
  while (!finished) {
    HostTable chunk = load_csv_chunk(file, rows_per_chunk, finished, column_names,
                                     ParsePolicy::Strict, &schema);
    if (chunk.num_rows() == 0 && finished)
      break;
    auto part = run_multi_gpu_jit_host(chunk, expr_cuda, cond_cuda);
    auto &dst = std::get<std::vector<float>>(all_results);
    const auto &src = std::get<std::vector<float>>(part);
    dst.insert(dst.end(), src.begin(), src.end());
  }

  const auto &out = std::get<std::vector<float>>(all_results);
  for (size_t i = 0; i < out.size(); ++i) {
    std::cout << "Large MultiGPU Result[" << i << "] = " << out[i]
              << "\n";
  }
}



// Print query results, supporting multiple SELECT columns of arbitrary type.
static void print_query_results(const QueryResult &results) {
  const size_t rows = results.size();
  const size_t cols = results.column_count();
  std::cout << "Query returned " << rows << " rows.\n";
  for (size_t r = 0; r < rows; ++r) {
    std::cout << "Result[" << r << "] = ";
    for (size_t c = 0; c < cols; ++c) {
      if (c > 0)
        std::cout << ", ";
      std::visit([r](const auto &vec) { std::cout << vec[r]; },
                 results.column_data(c));
    }
    std::cout << "\n";
  }
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

    print_query_results(results);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
