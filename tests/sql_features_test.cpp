#include "warpdb.hpp"
#include <cassert>
#include <iostream>

int main(){
    WarpDB db("data/test.csv");
    auto res = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity ORDER BY quantity ASC");
    std::cout << "rows " << res.size() << "\n";
    return 0;
}
