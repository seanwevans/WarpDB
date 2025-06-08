#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main(){
    WarpDB db("data/test.csv");
    auto res = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity ORDER BY quantity ASC");
    std::cout << "rows " << res.size() << "\n";

    auto limited = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2");
    assert(limited.size() == 2);

    auto offset = db.query_sql("SELECT price FROM test ORDER BY price DESC OFFSET 1 LIMIT 2");
    assert(offset.size() == 2);

    auto having = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 15 ORDER BY quantity ASC");
    assert(having.size() == 3);
    return 0;
}
