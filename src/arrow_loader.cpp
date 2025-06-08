#ifdef USE_ARROW
#include "arrow_loader.hpp"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/util/logging.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

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
#endif
