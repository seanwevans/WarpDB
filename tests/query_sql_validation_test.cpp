#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main() {
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
