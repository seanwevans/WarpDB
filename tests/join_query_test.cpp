#include "warpdb.hpp"
#include <cassert>
#include <vector>

int main() {
    WarpDB db("data/test.csv");

    auto joined = db.query_sql(
        "SELECT right.price FROM left JOIN right ON left.quantity = right.quantity "
        "WHERE left.price > 10 ORDER BY left.price ASC");
    const auto &vals = joined.as<float>();
    assert(vals.size() == 3);
    assert(vals[0] == 15.25f);
    assert(vals[1] == 20.0f);
    assert(vals[2] == 30.0f);

    return 0;
}
