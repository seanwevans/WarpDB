#ifdef USE_ARROW
#include "arrow_loader.hpp"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/adapters/orc/adapter.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
  do { \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
      exit(1); \
    } \
  } while (0)

ArrowTable load_csv_arrow(const std::string &filepath) {
    auto input_res = arrow::io::ReadableFile::Open(filepath);
    if (!input_res.ok()) {
        throw std::runtime_error(input_res.status().ToString());
    }
    std::shared_ptr<arrow::io::InputStream> input = *input_res;

    auto read_opts = arrow::csv::ReadOptions::Defaults();
    auto parse_opts = arrow::csv::ParseOptions::Defaults();
    auto convert_opts = arrow::csv::ConvertOptions::Defaults();

    ARROW_ASSIGN_OR_RAISE(
        auto reader,
        arrow::csv::TableReader::Make(arrow::io::IOContext(), input, read_opts,
                                      parse_opts, convert_opts));

    ARROW_ASSIGN_OR_RAISE(auto table, reader->Read());

    int64_t num_rows = table->num_rows();

    auto price_array = std::static_pointer_cast<arrow::FloatArray>(
        table->GetColumnByName("price")->chunk(0));
    auto qty_array = std::static_pointer_cast<arrow::Int32Array>(
        table->GetColumnByName("quantity")->chunk(0));

    std::shared_ptr<arrow::cuda::CudaDeviceManager> manager;
    ARROW_THROW_NOT_OK(arrow::cuda::CudaDeviceManager::GetInstance(&manager));
    std::shared_ptr<arrow::cuda::CudaContext> context;
    ARROW_THROW_NOT_OK(manager->GetContext(0, &context));

    std::shared_ptr<arrow::cuda::CudaBuffer> d_price;
    std::shared_ptr<arrow::cuda::CudaBuffer> d_quantity;
    ARROW_THROW_NOT_OK(context->AllocateBuffer(sizeof(float) * num_rows, &d_price));
    ARROW_THROW_NOT_OK(context->AllocateBuffer(sizeof(int32_t) * num_rows, &d_quantity));

    ARROW_THROW_NOT_OK(
        context->CopyHostToDevice(d_price->address(), price_array->raw_values(),
                                  sizeof(float) * num_rows));
    ARROW_THROW_NOT_OK(
        context->CopyHostToDevice(d_quantity->address(), qty_array->raw_values(),
                                  sizeof(int32_t) * num_rows));

    return {d_price, d_quantity, num_rows};
}


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

  Table table;
  table.d_price = d_price;
  table.d_quantity = d_quantity;
  table.num_rows = static_cast<int>(N);
  ColumnDesc price_desc{"price", DataType::Float32, d_price, table.num_rows};
  ColumnDesc qty_desc{"quantity", DataType::Int32, d_quantity, table.num_rows};
  table.columns = {price_desc, qty_desc};
  TableStats stats;
  if (!h_price.empty()) {
    auto [min_it, max_it] = std::minmax_element(h_price.begin(), h_price.end());
    stats.price.min = *min_it;
    stats.price.max = *max_it;
  }
  if (!h_quantity.empty()) {
    auto [min_it, max_it] = std::minmax_element(h_quantity.begin(), h_quantity.end());
    stats.quantity.min = *min_it;
    stats.quantity.max = *max_it;
  }
  table.stats = stats;
  return table;
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

#endif // USE_ARROW
