#include "warpdb.hpp"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/feather.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/adapters/orc/adapter.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

#ifdef USE_ARROW

namespace {
std::shared_ptr<arrow::Table> make_table() {
    auto prices = arrow::ArrayFromJSON(arrow::float64(), "[1.0, 2.5, 3.5, 5.0]").ValueOrDie();
    auto qty = arrow::ArrayFromJSON(arrow::int32(), "[1, 2, 3, 4]").ValueOrDie();
    auto schema = arrow::schema({arrow::field("price", arrow::float64()),
                                 arrow::field("quantity", arrow::int32())});
    return arrow::Table::Make(schema, {prices, qty});
}

std::string write_parquet(const std::shared_ptr<arrow::Table> &table) {
    auto path = (std::filesystem::temp_directory_path() / "warpdb_arrow.parquet").string();
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(path));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));
    return path;
}

std::string write_arrow_file(const std::shared_ptr<arrow::Table> &table) {
    auto path = (std::filesystem::temp_directory_path() / "warpdb_arrow.arrow").string();
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeFileWriter(outfile.get(), table->schema()));
    ARROW_THROW_NOT_OK(writer->WriteTable(*table));
    ARROW_THROW_NOT_OK(writer->Close());
    return path;
}

std::string write_feather(const std::shared_ptr<arrow::Table> &table) {
    auto path = (std::filesystem::temp_directory_path() / "warpdb_arrow.feather").string();
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(path));
    ARROW_THROW_NOT_OK(arrow::ipc::feather::WriteTable(*table, outfile.get()));
    return path;
}

std::string write_orc(const std::shared_ptr<arrow::Table> &table) {
    auto path = (std::filesystem::temp_directory_path() / "warpdb_arrow.orc").string();
    ARROW_ASSIGN_OR_RAISE(auto outfile, arrow::io::FileOutputStream::Open(path));
    ARROW_THROW_NOT_OK(arrow::adapters::orc::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));
    return path;
}
} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "No CUDA devices; skipping Arrow host table test.\n";
        return 0;
    }

    auto table = make_table();
    struct TestCase {
        std::string name;
        std::string (*writer)(const std::shared_ptr<arrow::Table> &);
    };

    TestCase cases[] = {
        {"parquet", write_parquet},
        {"arrow", write_arrow_file},
        {"feather", write_feather},
        {"orc", write_orc},
    };

    for (const auto &tc : cases) {
        auto path = tc.writer(table);
        WarpDB db(path);

        auto sql_result = db.query_sql("SELECT price WHERE quantity >= 2 ORDER BY price");
        std::vector<float> expected_sql = {2.5f, 3.5f, 5.0f};
        assert(sql_result.size() == expected_sql.size());
        for (size_t i = 0; i < expected_sql.size(); ++i) {
            assert(std::fabs(sql_result[i] - expected_sql[i]) < 1e-6f);
        }

        auto mg_result = db.query_multi_gpu("price * 2 WHERE quantity >= 2");
        std::vector<float> expected_mg = {5.0f, 7.0f, 10.0f};
        assert(mg_result.size() == expected_mg.size());
        for (size_t i = 0; i < expected_mg.size(); ++i) {
            assert(std::fabs(mg_result[i] - expected_mg[i]) < 1e-5f);
        }

        std::filesystem::remove(path);
    }

    std::cout << "Arrow host table SQL and multi-GPU test passed.\n";
    return 0;
}

#else
int main() {
    std::cout << "Arrow not enabled; skipping test.\n";
    return 0;
}
#endif
