#include "warpdb.hpp"
#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

int main(){
    WarpDB db("data/test.csv");
    auto single_expr = db.query_sql("SELECT price FROM test");
    assert(single_expr.size() == 4);

    bool multiple_expr_threw = false;
    try {
        (void)db.query_sql("SELECT price, quantity FROM test");
    } catch (const std::runtime_error &e) {
        multiple_expr_threw =
            std::string(e.what()).find(
                "Multiple SELECT expressions are not yet supported in query_sql") !=
            std::string::npos;
    }
    assert(multiple_expr_threw);

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
    assert(offset.size() == 2);
    assert(std::abs(offset[0] - prices[1]) < 1e-5);
    assert(std::abs(offset[1] - prices[2]) < 1e-5);

    auto offset_only = db.query_sql("SELECT price FROM test ORDER BY price DESC OFFSET 2");
    assert(offset_only.size() == 2);
    assert(std::abs(offset_only[0] - prices[2]) < 1e-5);
    assert(std::abs(offset_only[1] - prices[3]) < 1e-5);

    auto offset_then_limit = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 1 OFFSET 2");
    assert(offset_then_limit.size() == 1);
    assert(std::abs(offset_then_limit[0] - prices[2]) < 1e-5);

    auto large_limit = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 5 OFFSET 3");
    assert(large_limit.size() == 1);
    assert(std::abs(large_limit[0] - prices[3]) < 1e-5);

    auto offset_beyond = db.query_sql("SELECT price FROM test ORDER BY price DESC LIMIT 2 OFFSET 10");
    assert(offset_beyond.empty());

    auto having = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 15 ORDER BY quantity ASC");
    assert(having.size() == 3);

    auto float_group = db.query_sql(
        "SELECT SUM(price) FROM test GROUP BY price / 10.0 ORDER BY price / 10.0 ASC");
    std::map<double, double> float_groups;
    for (size_t i = 0; i < prices_vec.size(); ++i) {
        double key = static_cast<double>(prices_vec[i]) / 10.0;
        float_groups[key] += prices_vec[i];
    }
    std::vector<float> expected_float;
    for (const auto &kv : float_groups) {
        expected_float.push_back(static_cast<float>(kv.second));
    }

    assert(float_group.size() == expected_float.size());
    for (size_t i = 0; i < expected_float.size(); ++i) {
        assert(std::abs(float_group[i] - expected_float[i]) < 1e-5);
    }

    auto float_group_desc = db.query_sql(
        "SELECT SUM(price) FROM test GROUP BY price / 10.0 ORDER BY price / 10.0 DESC");
    std::vector<float> expected_desc(expected_float.rbegin(), expected_float.rend());
    assert(float_group_desc.size() == expected_desc.size());
    for (size_t i = 0; i < expected_desc.size(); ++i) {
        assert(std::abs(float_group_desc[i] - expected_desc[i]) < 1e-5);
    }

    return 0;
}
