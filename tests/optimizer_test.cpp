#include "optimizer.hpp"
#include <cassert>
#include <sstream>
#include <iostream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"

static Table make_price_table(const std::vector<float> &values) {
    Table table;
    table.num_rows = static_cast<int>(values.size());

    ColumnDesc price;
    price.name = "price";
    price.type = DataType::Float32;
    price.length = static_cast<int>(values.size());

    float *d_values = nullptr;
    if (!values.empty()) {
        CUDA_CHECK(cudaMalloc(&d_values, sizeof(float) * values.size()));
        CUDA_CHECK(cudaMemcpy(d_values, values.data(),
                              sizeof(float) * values.size(),
                              cudaMemcpyHostToDevice));
    }
    price.device_ptr.reset(d_values);
    table.columns.push_back(std::move(price));
    return table;
}

static void test_missing_stats_skip_elimination() {
    Table table;
    table.num_rows = 5;
    TableStats stats;
    bool has_stats = compute_table_stats(table, stats);
    assert(!has_stats);
    assert(stats.numeric_columns.empty());
}

static void test_analyze_condition_always_false() {
    auto tokens = tokenize("price > 100");
    auto ast = parse_expression(tokens);
    TableStats stats;
    stats.numeric_columns["price"] = {10.0, 50.0, 0, true};
    bool always_true = false;
    bool always_false = false;
    analyze_condition(ast.get(), stats, always_true, always_false);
    assert(!always_true);
    assert(always_false);
}

static void test_real_stats_prevent_elimination() {
    auto tokens = tokenize("price > 100");
    auto ast = parse_expression(tokens);

    Table table = make_price_table({10.0f, 200.0f, 50.0f});
    TableStats stats;
    bool has_stats = compute_table_stats(table, stats);
    assert(has_stats);
    assert(stats.numeric_columns.count("price") == 1);

    bool always_true = false;
    bool always_false = false;
    analyze_condition(ast.get(), stats, always_true, always_false);
    assert(!always_true);
    assert(!always_false);
}

static void test_join_is_parsed_for_optimizer_input() {
    auto tokens = tokenize(
        "SELECT a.v FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id");
    QueryAST ast = parse_query(tokens);
    assert(ast.joins.size() == 2);
    assert(ast.joins[0].table == "b");
    assert(ast.joins[1].table == "c");
}

static void test_estimate_equi_join_rows_uses_max_ndv() {
    const double rows = estimate_equi_join_rows(1000.0, 500.0, 1000.0, 50.0);
    assert(rows == 500.0);
}

int main() {
    test_missing_stats_skip_elimination();
    test_analyze_condition_always_false();
    test_real_stats_prevent_elimination();
    test_join_is_parsed_for_optimizer_input();
    test_estimate_equi_join_rows_uses_max_ndv();
    std::cout << "Optimizer tests passed\n";
    return 0;
}
