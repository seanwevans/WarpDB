#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main(){
    WarpDB db("data/test.csv");
    auto res = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING COUNT(price) > 1");
    assert(res.empty());

    auto res2 = db.query_sql("SELECT DISTINCT quantity FROM test ORDER BY quantity DESC");
    assert(res2.size() == 4);
    assert(res2.front() > res2.back());
    std::cout << "HAVING/DISTINCT tests passed\n";
    return 0;
}
