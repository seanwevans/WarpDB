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

static void test_extract_equi_join_predicates() {
    auto tokens = tokenize(
        "SELECT a.v FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id");
    QueryAST ast = parse_query(tokens);
    auto predicates = extract_equi_join_predicates(ast);
    assert(predicates.size() == 2);
    assert(predicates[0].left_relation == "a");
    assert(predicates[0].left_column == "id");
    assert(predicates[0].right_relation == "b");
    assert(predicates[0].right_column == "a_id");
}

static void test_build_greedy_join_plan_prefers_low_ndv_join() {
    auto tokens = tokenize(
        "SELECT a.v FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id");
    QueryAST ast = parse_query(tokens);

    std::unordered_map<std::string, RelationStats> stats;
    stats["a"] = RelationStats{"a", 100000.0, {{"id", 100000.0}}};
    stats["b"] = RelationStats{"b", 1000.0, {{"a_id", 1000.0}, {"id", 1000.0}}};
    stats["c"] = RelationStats{"c", 500000.0, {{"b_id", 1000.0}}};

    JoinPlan plan = build_greedy_join_plan(ast, stats);
    assert(plan.steps.size() == 3);
    assert(plan.steps[0].relation == "b");
    assert(plan.steps[1].relation == "a");
    assert(plan.steps[2].relation == "c");
}

static void test_estimate_equi_join_rows_uses_max_ndv() {
    const double rows = estimate_equi_join_rows(1000.0, 500.0, 1000.0, 50.0);
    assert(rows == 500.0);
}

int main() {
    test_missing_stats_skip_elimination();
    test_analyze_condition_always_false();
    test_real_stats_prevent_elimination();
    test_extract_equi_join_predicates();
    test_build_greedy_join_plan_prefers_low_ndv_join();
    test_estimate_equi_join_rows_uses_max_ndv();
    std::cout << "Optimizer tests passed\n";
    return 0;
}
