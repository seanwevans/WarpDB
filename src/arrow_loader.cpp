#if defined(USE_ARROW)
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

#elif defined(USE_SIMPLE_READERS)
#include "arrow_loader.hpp"
#include "csv_loader.hpp"
#include <parquet/api/reader.h>
#include <orc/OrcFile.hh>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
  do { \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; \
      exit(1); \
    } \
  } while (0)

namespace {
Table finalize_table(std::vector<float> &price, std::vector<int> &qty) {
  float *d_price;
  int *d_quantity;
  size_t N = price.size();
  CUDA_CHECK(cudaMalloc(&d_price, sizeof(float) * N));
  CUDA_CHECK(cudaMalloc(&d_quantity, sizeof(int) * N));
  CUDA_CHECK(cudaMemcpy(d_price, price.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_quantity, qty.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
  Table table;
  table.d_price = d_price;
  table.d_quantity = d_quantity;
  table.num_rows = static_cast<int>(N);
  table.columns = { {"price", DataType::Float32, d_price, table.num_rows},
                    {"quantity", DataType::Int32, d_quantity, table.num_rows} };
  TableStats stats;
  if(!price.empty()) {
    auto [min_p, max_p] = std::minmax_element(price.begin(), price.end());
    stats.price.min = *min_p; stats.price.max = *max_p;
  }
  if(!qty.empty()) {
    auto [min_q, max_q] = std::minmax_element(qty.begin(), qty.end());
    stats.quantity.min = *min_q; stats.quantity.max = *max_q;
  }
  table.stats = stats;
  return table;
}
} // namespace

Table load_parquet_to_gpu(const std::string &filepath) {
  auto reader = parquet::ParquetFileReader::OpenFile(filepath, false);
  auto metadata = reader->metadata();
  int64_t rows = metadata->num_rows();
  std::vector<float> price(rows);
  std::vector<int> quantity(rows);
  int64_t offset = 0;
  for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
    auto group = reader->RowGroup(rg);
    auto price_col = static_cast<parquet::DoubleReader*>(group->Column(0).get());
    auto qty_col = static_cast<parquet::Int32Reader*>(group->Column(1).get());
    while (price_col->HasNext()) {
      int64_t values_read = 0;
      double p; int32_t q; int16_t d, r;
      price_col->ReadBatch(1, &d, &r, &p, &values_read);
      qty_col->ReadBatch(1, &d, &r, &q, &values_read);
      price[offset] = static_cast<float>(p);
      quantity[offset] = q;
      ++offset;
    }
  }
  reader->Close();
  return finalize_table(price, quantity);
}

Table load_arrow_to_gpu(const std::string &filepath) {
  throw std::runtime_error("Arrow IPC files require USE_ARROW");
}

Table load_orc_to_gpu(const std::string &filepath) {
  std::unique_ptr<orc::InputStream> in = orc::readLocalFile(filepath);
  orc::ReaderOptions opts;
  auto reader = orc::createReader(std::move(in), opts);
  uint64_t rows = reader->getNumberOfRows();
  std::vector<float> price(rows);
  std::vector<int> quantity(rows);
  orc::RowReaderOptions row_opts;
  std::unique_ptr<orc::RowReader> row_reader = reader->createRowReader(row_opts);
  uint64_t offset = 0;
  std::unique_ptr<orc::ColumnVectorBatch> batch = row_reader->createRowBatch(1024);
  while (row_reader->next(*batch)) {
    auto *struct_batch = dynamic_cast<orc::StructVectorBatch*>(batch.get());
    auto *price_batch = dynamic_cast<orc::DoubleVectorBatch*>(struct_batch->fields[0]);
    auto *qty_batch = dynamic_cast<orc::LongVectorBatch*>(struct_batch->fields[1]);
    for (uint64_t i = 0; i < batch->numElements; ++i) {
      price[offset] = static_cast<float>(price_batch->data[i]);
      quantity[offset] = static_cast<int>(qty_batch->data[i]);
      ++offset;
    }
  }
  return finalize_table(price, quantity);
}

#endif // USE_SIMPLE_READERS
