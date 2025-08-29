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

    // Ensure queries still execute when string columns are present
    WarpDB db("data/string_test.csv", schema);
    auto res = db.query("price + quantity");
    assert(res.size() == 3);
    assert(std::abs(res[0] - (10.0f + 5)) < 1e-5);
    assert(std::abs(res[1] - (20.5f + 3)) < 1e-5);
    assert(std::abs(res[2] - (30.25f + 7)) < 1e-5);

    return 0;
}
