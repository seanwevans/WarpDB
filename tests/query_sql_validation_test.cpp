#include "test_utils.hpp"
#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main() {
    if (warpdb_skip_if_no_cuda("query_sql_validation_test")) return 0;
    WarpDB db("data/test.csv");
    bool threw = false;
    try {
        db.query_sql("SELECT foo FROM test");
    } catch (const std::runtime_error &e) {
        threw = std::string(e.what()).find("Unknown column") != std::string::npos;
    }
    assert(threw);
    std::cout << "query_sql_validation_test passed\n";
    return 0;
}
