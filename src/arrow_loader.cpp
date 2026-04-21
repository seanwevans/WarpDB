#ifdef USE_ARROW
#include "arrow_loader.hpp"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/adapters/orc/adapter.h>

#include "cuda_utils.hpp"

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

DataType data_type_from_arrow(const std::shared_ptr<arrow::DataType> &type) {
  switch (type->id()) {
  case arrow::Type::INT32:
    return DataType::Int32;
  case arrow::Type::INT64:
    return DataType::Int64;
  case arrow::Type::FLOAT:
    return DataType::Float32;
  case arrow::Type::DOUBLE:
    return DataType::Float64;
  case arrow::Type::STRING:
  case arrow::Type::LARGE_STRING:
    return DataType::String;
  default:
    throw std::runtime_error("Unsupported Arrow column type: " + type->ToString());
  }
}

HostColumn host_column_from_arrow(const std::string &name,
                                  const std::shared_ptr<arrow::Array> &array,
                                  DataType type) {
  HostColumn col;
  col.name = name;
  col.type = type;

  const int64_t N = array->length();
  switch (type) {
  case DataType::Int32: {
    auto typed = std::static_pointer_cast<arrow::Int32Array>(array);
    std::vector<int32_t> values(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
      values[static_cast<size_t>(i)] = typed->IsNull(i) ? 0 : typed->Value(i);
    }
    col.data = std::move(values);
    break;
  }
  case DataType::Int64: {
    auto typed = std::static_pointer_cast<arrow::Int64Array>(array);
    std::vector<int64_t> values(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
      values[static_cast<size_t>(i)] = typed->IsNull(i) ? 0 : typed->Value(i);
    }
    col.data = std::move(values);
    break;
  }
  case DataType::Float32: {
    auto typed = std::static_pointer_cast<arrow::FloatArray>(array);
    std::vector<float> values(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
      values[static_cast<size_t>(i)] = typed->IsNull(i) ? 0.0f : typed->Value(i);
    }
    col.data = std::move(values);
    break;
  }
  case DataType::Float64: {
    auto typed = std::static_pointer_cast<arrow::DoubleArray>(array);
    std::vector<double> values(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
      values[static_cast<size_t>(i)] = typed->IsNull(i) ? 0.0 : typed->Value(i);
    }
    col.data = std::move(values);
    break;
  }
  case DataType::String: {
    std::vector<std::string> values(static_cast<size_t>(N));
    if (array->type_id() == arrow::Type::STRING) {
      auto typed = std::static_pointer_cast<arrow::StringArray>(array);
      for (int64_t i = 0; i < N; ++i) {
        values[static_cast<size_t>(i)] = typed->IsNull(i) ? "" : typed->GetString(i);
      }
    } else {
      auto typed = std::static_pointer_cast<arrow::LargeStringArray>(array);
      for (int64_t i = 0; i < N; ++i) {
        values[static_cast<size_t>(i)] = typed->IsNull(i) ? "" : typed->GetString(i);
      }
    }
    col.data = std::move(values);
    break;
  }
  }

  return col;
}

HostTable host_table_from_arrow(const std::shared_ptr<arrow::Table> &table) {
  ARROW_ASSIGN_OR_RAISE(auto combined, table->CombineChunks());

  HostTable host;
  host.columns.reserve(static_cast<size_t>(combined->num_columns()));

  for (int i = 0; i < combined->num_columns(); ++i) {
    const auto &field = combined->schema()->field(i);
    const auto &chunked = combined->column(i);
    if (chunked->num_chunks() != 1) {
      throw std::runtime_error("Failed to combine Arrow chunks for column: " + field->name());
    }

    const auto type = data_type_from_arrow(field->type());
    host.columns.push_back(
        host_column_from_arrow(field->name(), chunked->chunk(0), type));
  }

  return host;
}

Table table_from_arrow(const std::shared_ptr<arrow::Table> &table) {
  return upload_to_gpu(host_table_from_arrow(table));
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

HostTable load_parquet_to_host(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(filepath));
  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
  return host_table_from_arrow(table);
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

HostTable load_arrow_to_host(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filepath));
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
  ARROW_ASSIGN_OR_RAISE(reader, arrow::ipc::RecordBatchFileReader::Open(infile));
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, reader->ReadAll());
  return host_table_from_arrow(table);
}

Table load_orc_to_gpu(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filepath));
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow::adapters::orc::ORCFileReader::Read(*infile));
  return table_from_arrow(table);
}

HostTable load_orc_to_host(const std::string &filepath) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile, arrow::io::ReadableFile::Open(filepath));
  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow::adapters::orc::ORCFileReader::Read(*infile));
  return host_table_from_arrow(table);
}

#endif // USE_ARROW
