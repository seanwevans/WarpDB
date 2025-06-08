#include "expression.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::string q = "SELECT SUM(price), quantity FROM sales JOIN items ON sales.id = items.id WHERE price > 10 GROUP BY quantity ORDER BY price DESC LIMIT 5";
    auto tokens = tokenize(q);
    QueryAST ast = parse_query(tokens);
    assert(ast.select_list.size() == 2);
    assert(!ast.joins.empty());
    assert(ast.where.has_value());
    assert(ast.group_by.has_value());
    assert(ast.order_by.has_value());
    assert(ast.limit.has_value());
    std::cout << "Query parse test passed\n";
    return 0;
}
