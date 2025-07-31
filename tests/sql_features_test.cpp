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
    const HostColumn *price_col = h.get_column("price");
    const HostColumn *qty_col = h.get_column("quantity");
    const auto &prices_vec = std::get<std::vector<float>>(price_col->data);
    const auto &qty_vec = std::get<std::vector<int32_t>>(qty_col->data);
    std::map<int,double> groups;
    for(size_t i=0;i<prices_vec.size();++i){
        groups[qty_vec[i]] += prices_vec[i];
    }
    std::vector<float> expected;
    for(auto &kv : groups) expected.push_back(static_cast<float>(kv.second));

    assert(res.size() == expected.size());
    for(size_t i=0;i<res.size();++i) assert(std::abs(res[i]-expected[i])<1e-5);

    auto limited = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2");
    std::vector<float> prices = prices_vec;
    std::sort(prices.begin(), prices.end(), std::greater<float>());
    assert(limited.size() == 2);

    assert(std::abs(limited[0]-prices[0])<1e-5);
    assert(std::abs(limited[1]-prices[1])<1e-5);


    auto offset = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2 OFFSET 1");
    assert(offset.size() == 1);
    assert(std::abs(offset[0] - prices[1]) < 1e-5);

    auto having = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 15 ORDER BY quantity ASC");
    assert(having.size() == 3);

    return 0;
}
