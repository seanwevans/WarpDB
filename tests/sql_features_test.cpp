#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main(){
    WarpDB db("data/test.csv");
    auto res = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity ORDER BY quantity ASC");
    std::cout << "rows " << res.size() << "\n";

    auto limited = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2");
    assert(limited.size() == 2);
    return 0;
}
