#include "test_utils.hpp"
#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main(){
    if (warpdb_skip_if_no_cuda("extended_types_test")) return 0;
    std::vector<DataType> schema = {DataType::Float32, DataType::Int32, DataType::Float32};
    WarpDB db("data/extended.csv", schema);
    auto res = db.query("price * discount");
    assert(res.size() == 4);
    assert(static_cast<int>(res[0]) == 1); // 10.5*0.1 approx 1.05
    std::cout << "extended types test passed\n";
    return 0;
}
