#include "warpdb.hpp"
#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>
#include <fstream>

int main(){
    WarpDB db("data/test.csv");
    auto single_expr = db.query_sql("SELECT price FROM test");
    assert(single_expr.size() == 4);

    auto multiple_expr = db.query_sql("SELECT price, quantity FROM test");
    assert(multiple_expr.column_count() == 2);
    const auto &prices_sel = multiple_expr.column_as<float>(0);
    const auto &qty_sel = multiple_expr.column_as<int32_t>(1);
    assert(prices_sel.size() == 4);
    assert(qty_sel.size() == 4);
    assert(std::abs(prices_sel[0] - 10.5f) < 1e-5);
    assert(qty_sel[0] == 3);

    bool grouped_multiple_expr_threw = false;
    try {
        (void)db.query_sql(
            "SELECT SUM(price), quantity FROM test GROUP BY quantity");
    } catch (const std::runtime_error &e) {
        grouped_multiple_expr_threw =
            std::string(e.what()).find(
                "GROUP BY queries currently support exactly one SELECT expression") !=
            std::string::npos;
    }
    assert(grouped_multiple_expr_threw);

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

    auto where_equal = db.query_sql("SELECT price FROM test WHERE price = 20");
    auto where_double_equal = db.query_sql("SELECT price FROM test WHERE price == 20");
    assert(where_equal.size() == where_double_equal.size());
    for (size_t i = 0; i < where_equal.size(); ++i) {
        assert(std::abs(where_equal[i] - where_double_equal[i]) < 1e-5);
    }

    auto having = db.query_sql("SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 15 ORDER BY quantity ASC");
    assert(having.size() == 3);

    auto having_equal = db.query_sql(
        "SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) = 20 ORDER BY quantity ASC");
    auto having_double_equal = db.query_sql(
        "SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) == 20 ORDER BY quantity ASC");
    assert(having_equal.size() == having_double_equal.size());
    for (size_t i = 0; i < having_equal.size(); ++i) {
        assert(std::abs(having_equal[i] - having_double_equal[i]) < 1e-5);
    }

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

    const std::string composite_csv = "data/test_having_composite.csv";
    {
        std::ofstream out(composite_csv);
        out << "price,quantity\n";
        out << "6,1\n";
        out << "7,1\n";
        out << "50,2\n";
    }

    WarpDB having_db(composite_csv);

    auto having_and = having_db.query_sql(
        "SELECT SUM(price) FROM test_having_composite "
        "GROUP BY quantity "
        "HAVING SUM(price) > 10 AND COUNT(price) >= 2 "
        "ORDER BY quantity ASC");
    assert(having_and.size() == 1);
    assert(std::abs(having_and[0] - 13.0f) < 1e-5);

    auto having_or = having_db.query_sql(
        "SELECT SUM(price) FROM test_having_composite "
        "GROUP BY quantity "
        "HAVING SUM(price) > 100 OR COUNT(price) = 1 "
        "ORDER BY quantity ASC");
    assert(having_or.size() == 1);
    assert(std::abs(having_or[0] - 50.0f) < 1e-5);
    bool multi_key_group_by_threw = false;
    try {
        (void)db.query_sql(
            "SELECT SUM(price) FROM test GROUP BY quantity, price ORDER BY quantity ASC");
    } catch (const std::runtime_error &e) {
        multi_key_group_by_threw =
            std::string(e.what()).find(
                "GROUP BY in query_sql currently supports exactly one key expression") !=
            std::string::npos;
    }
    assert(multi_key_group_by_threw);

    const std::string generic_gpu_csv = "data/test_group_fastpath_generic.csv";
    {
        std::ofstream out(generic_gpu_csv);
        out << "gid,score\n";
        out << "10,3\n";
        out << "10,5\n";
        out << "20,7\n";
        out << "20,9\n";
    }

    WarpDB generic_gpu_db(generic_gpu_csv);
    auto sum_generic = generic_gpu_db.query_sql(
        "SELECT SUM(score) FROM test_group_fastpath_generic GROUP BY gid ORDER BY gid ASC");
    assert(sum_generic.size() == 2);
    assert(std::abs(sum_generic[0] - 8.0f) < 1e-5);
    assert(std::abs(sum_generic[1] - 16.0f) < 1e-5);

    auto avg_generic = generic_gpu_db.query_sql(
        "SELECT AVG(score) FROM test_group_fastpath_generic GROUP BY gid ORDER BY gid ASC");
    assert(avg_generic.size() == 2);
    assert(std::abs(avg_generic[0] - 4.0f) < 1e-5);
    assert(std::abs(avg_generic[1] - 8.0f) < 1e-5);

    auto min_generic = generic_gpu_db.query_sql(
        "SELECT MIN(score) FROM test_group_fastpath_generic GROUP BY gid ORDER BY gid ASC");
    assert(min_generic.size() == 2);
    assert(std::abs(min_generic[0] - 3.0f) < 1e-5);
    assert(std::abs(min_generic[1] - 7.0f) < 1e-5);

    auto max_generic = generic_gpu_db.query_sql(
        "SELECT MAX(score) FROM test_group_fastpath_generic GROUP BY gid ORDER BY gid ASC");
    assert(max_generic.size() == 2);
    assert(std::abs(max_generic[0] - 5.0f) < 1e-5);
    assert(std::abs(max_generic[1] - 9.0f) < 1e-5);

    return 0;
}
