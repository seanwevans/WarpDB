#include "arrow_loader.hpp"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/adapters/orc/adapter.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) \
  do { \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
      exit(1); \
    } \
  } while (0)

namespace {
Table table_from_arrow(std::shared_ptr<arrow::Table> table) {
  auto price_array = std::static_pointer_cast<arrow::DoubleArray>(table->GetColumnByName("price")->chunk(0));
  auto quantity_array = std::static_pointer_cast<arrow::Int32Array>(table->GetColumnByName("quantity")->chunk(0));
  int64_t N = table->num_rows();

  float *d_price;
  int *d_quantity;
  CUDA_CHECK(cudaMalloc(&d_price, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&d_quantity, sizeof(int) * N));

  std::vector<float> h_price(N);
  std::vector<int> h_quantity(N);
  for (int64_t i = 0; i < N; ++i) {
    h_price[i] = static_cast<float>(price_array->Value(i));
    h_quantity[i] = quantity_array->Value(i);
  }

  CUDA_CHECK(cudaMemcpy(d_price, h_price.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_quantity, h_quantity.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  return {d_price, d_quantity, static_cast<int>(N)};
}
} // namespace

Table load_parquet_to_gpu(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(filepath));
  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
  return table_from_arrow(table);
}

Table load_arrow_to_gpu(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filepath));
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, arrow::ipc::RecordBatchFileReader::Open(infile));
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, reader->ReadAll());
  return table_from_arrow(table);
}

Table load_orc_to_gpu(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filepath));
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow::adapters::orc::ORCFileReader::Read(*infile));
  return table_from_arrow(table);
}
