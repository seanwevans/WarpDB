#include "warpdb.hpp"
#include <cassert>
#include <vector>

int main() {
    WarpDB db("data/test.csv");

    bool rejected_join = false;
    try {
        (void)db.query_sql(
            "SELECT right.price FROM left JOIN right ON left.quantity = right.quantity "
            "WHERE left.price > 10");
    } catch (const std::runtime_error &e) {
        rejected_join = std::string(e.what()).find(
                            "JOIN is parsed but not supported for execution yet") !=
                        std::string::npos;
    }
    assert(rejected_join);

    return 0;
}
