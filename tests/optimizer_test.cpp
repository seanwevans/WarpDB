#include "optimizer.hpp"
#include <cassert>
#include <sstream>
#include <iostream>

static void test_always_false_via_execute() {
    Table table;
    table.num_rows = 5;
    // Capture output to verify the optimizer triggers early return
    std::ostringstream oss;
    std::streambuf *old_buf = std::cout.rdbuf(oss.rdbuf());
    execute_query_optimized("price", "price > 0", table);
    std::cout.rdbuf(old_buf);
    std::string out = oss.str();
    assert(out.find("[Optimizer] Filter eliminates all rows.") != std::string::npos);
}

static void test_analyze_condition_always_false() {
    auto tokens = tokenize("price > 100");
    auto ast = parse_expression(tokens);
    TableStats stats;
    stats.price.min = 10.0f;
    stats.price.max = 50.0f;
    bool always_true = false;
    bool always_false = false;
    analyze_condition(ast.get(), stats, always_true, always_false);
    assert(!always_true);
    assert(always_false);
}

int main() {
    test_always_false_via_execute();
    test_analyze_condition_always_false();
    std::cout << "Optimizer tests passed\n";
    return 0;
}
