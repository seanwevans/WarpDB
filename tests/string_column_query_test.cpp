#include "warpdb.hpp"
#include "csv_loader.hpp"
#include "cuda_utils.hpp"
#include <cassert>
#include <vector>
#include <string>
#include <cmath>

int main() {
    std::vector<DataType> schema{DataType::String, DataType::Float32, DataType::Int32};

    HostTable host = load_csv_to_host("data/string_test.csv", schema);
    Table device = upload_to_gpu(host);

    // Validate device-side string storage
    const int32_t* d_offsets = device.get_column_ptr<int32_t>("name");
    const char* d_chars = device.get_string_data("name");

    std::vector<int32_t> offsets(host.num_rows() + 1);
    CUDA_CHECK(cudaMemcpy(offsets.data(), d_offsets,
                          sizeof(int32_t) * (host.num_rows() + 1),
                          cudaMemcpyDeviceToHost));
    std::vector<char> chars(offsets.back());
    if (offsets.back() > 0) {
        CUDA_CHECK(cudaMemcpy(chars.data(), d_chars, offsets.back(),
                              cudaMemcpyDeviceToHost));
    }

    const auto& expected = std::get<std::vector<std::string>>(host.columns[0].data);
    for (int i = 0; i < host.num_rows(); ++i) {
        std::string s(chars.data() + offsets[i], offsets[i + 1] - offsets[i]);
        assert(s == expected[i]);
    }

    // Validate upload behavior when total string bytes == 0.
    HostTable empty_strings_host;
    HostColumn empty_name_col;
    empty_name_col.name = "name";
    empty_name_col.type = DataType::String;
    empty_name_col.data = std::vector<std::string>{"", "", ""};
    empty_strings_host.columns.push_back(std::move(empty_name_col));

    HostColumn value_col;
    value_col.name = "value";
    value_col.type = DataType::Int32;
    value_col.data = std::vector<int32_t>{1, 2, 3};
    empty_strings_host.columns.push_back(std::move(value_col));

    Table empty_strings_device = upload_to_gpu(empty_strings_host);
    const int32_t* empty_offsets_dev =
        empty_strings_device.get_column_ptr<int32_t>("name");
    const char* empty_chars_dev = empty_strings_device.get_string_data("name");
    assert(empty_offsets_dev != nullptr);
    assert(empty_chars_dev == nullptr);

    std::vector<int32_t> empty_offsets(empty_strings_host.num_rows() + 1);
    CUDA_CHECK(cudaMemcpy(empty_offsets.data(), empty_offsets_dev,
                          sizeof(int32_t) * (empty_strings_host.num_rows() + 1),
                          cudaMemcpyDeviceToHost));
    for (int v : empty_offsets) {
        assert(v == 0);
    }

    // Ensure queries still execute when string columns are present
    WarpDB db("data/string_test.csv", schema);
    auto res = db.query("price + quantity");
    assert(res.size() == 3);
    assert(std::abs(res[0] - (10.0f + 5)) < 1e-5);
    assert(std::abs(res[1] - (20.5f + 3)) < 1e-5);
    assert(std::abs(res[2] - (30.25f + 7)) < 1e-5);

    bool threw_on_string_expr = false;
    try {
        (void)db.query("name");
    } catch (const std::runtime_error& e) {
        threw_on_string_expr =
            std::string(e.what()).find("String columns are not supported in GPU JIT expressions") !=
            std::string::npos;
    }
    assert(threw_on_string_expr && "Expected query using string column to throw");

    return 0;
}
