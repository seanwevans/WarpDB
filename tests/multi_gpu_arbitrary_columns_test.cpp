#include "warpdb.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    WarpDB db("data/arbitrary_columns.csv");
    auto direct = db.query_multi_gpu("alpha * beta");
    assert(direct.size() == 3);
    assert(std::fabs(direct[0] - 3.0f) < 1e-5f);
    assert(std::fabs(direct[1] - 12.0f) < 1e-5f);
    assert(std::fabs(direct[2] - 30.0f) < 1e-5f);

    auto streamed = WarpDB::query_multi_gpu_csv("data/arbitrary_columns.csv", "alpha * beta", 2);
    assert(streamed.size() == direct.size());
    for (size_t i = 0; i < streamed.size(); ++i) {
        assert(std::fabs(streamed[i] - direct[i]) < 1e-5f);
    }

    std::cout << "Multi-GPU arbitrary column test passed\n";
    return 0;
}
