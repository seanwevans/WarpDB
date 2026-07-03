#include "test_utils.hpp"
#include "warpdb.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    if (warpdb_skip_if_no_cuda("multi_gpu_arbitrary_columns_test")) return 0;
    WarpDB db("data/arbitrary_columns.csv");
    auto direct = db.query_multi_gpu("alpha * beta");
    assert(direct.size() == 3);
    assert(std::fabs(direct[0] - 3.0f) < 1e-5f);
    assert(std::fabs(direct[1] - 12.0f) < 1e-5f);
    assert(std::fabs(direct[2] - 30.0f) < 1e-5f);

    auto streamed = WarpDB::query_multi_gpu_csv("data/arbitrary_columns.csv", "alpha * beta", 2);
    assert(streamed.size() == direct.size());

    bool threw_invalid_chunk = false;
    try {
        (void)WarpDB::query_multi_gpu_csv("data/arbitrary_columns.csv", "alpha * beta", 0);
    } catch (const std::runtime_error &e) {
        threw_invalid_chunk = std::string(e.what()) == "rows_per_chunk must be > 0";
    }
    assert(threw_invalid_chunk);
    for (size_t i = 0; i < streamed.size(); ++i) {
        assert(std::fabs(streamed[i] - direct[i]) < 1e-5f);
    }

    std::cout << "Multi-GPU arbitrary column test passed\n";
    return 0;
}
