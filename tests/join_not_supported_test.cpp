#include "warpdb.hpp"
#include <cassert>
#include <vector>

int main() {
    WarpDB db("data/test.csv");

    auto joined = db.query_sql(
        "SELECT right.price FROM left JOIN right ON left.quantity = right.quantity "
        "WHERE left.price > 10");
    const auto &vals = joined.as<float>();
    assert(vals.size() == 4);
    assert(vals[0] == 20.0f);
    assert(vals[1] == 15.25f);
    assert(vals[2] == 30.0f);
    assert(vals[3] == 10.5f);

    return 0;
}
