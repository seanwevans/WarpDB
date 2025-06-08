#include "warpdb.hpp"
#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

int main(){
    WarpDB db("data/test.csv");
    auto res = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity ORDER BY quantity ASC");

    HostTable h = load_csv_to_host("data/test.csv");
    std::map<int,double> groups;
    for(size_t i=0;i<h.price.size();++i){
        groups[h.quantity[i]] += h.price[i];
    }
    std::vector<float> expected;
    for(auto &kv : groups) expected.push_back(static_cast<float>(kv.second));

    assert(res.size() == expected.size());
    for(size_t i=0;i<res.size();++i) assert(std::abs(res[i]-expected[i])<1e-5);

    auto limited = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2");
    std::vector<float> prices = h.price;
    std::sort(prices.begin(), prices.end(), std::greater<float>());
    assert(limited.size() == 2);

    assert(std::abs(limited[0]-prices[0])<1e-5);
    assert(std::abs(limited[1]-prices[1])<1e-5);


    auto offset = db.query_sql("SELECT price FROM test ORDER BY price DESC OFFSET 1 LIMIT 2");
    assert(offset.size() == 2);

    auto having = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 15 ORDER BY quantity ASC");
    assert(having.size() == 3);

    return 0;
}
