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

    std::string order_limit_q = "SELECT price FROM sales ORDER BY price LIMIT 5";
    auto order_limit_tokens = tokenize(order_limit_q);
    QueryAST order_limit_ast = parse_query(order_limit_tokens);
    assert(order_limit_ast.order_by.has_value());
    assert(order_limit_ast.order_by->ascending);
    assert(order_limit_ast.limit.has_value());
    assert(order_limit_ast.limit->count == 5);

    std::string order_offset_q = "SELECT price FROM sales ORDER BY price OFFSET 10";
    auto order_offset_tokens = tokenize(order_offset_q);
    QueryAST order_offset_ast = parse_query(order_offset_tokens);
    assert(order_offset_ast.order_by.has_value());
    assert(order_offset_ast.order_by->ascending);
    assert(order_offset_ast.offset.has_value());
    assert(order_offset_ast.offset->count == 10);

    std::string order_only_q = "SELECT price FROM sales ORDER BY price";
    auto order_only_tokens = tokenize(order_only_q);
    QueryAST order_only_ast = parse_query(order_only_tokens);
    assert(order_only_ast.order_by.has_value());
    assert(order_only_ast.order_by->ascending);
    assert(!order_only_ast.limit.has_value());
    assert(!order_only_ast.offset.has_value());

    std::cout << "Query parse test passed\n";
    return 0;
}
