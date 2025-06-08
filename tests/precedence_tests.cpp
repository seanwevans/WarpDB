#include "expression.hpp"
#include <cassert>
#include <iostream>

void test_precedence() {
    auto tokens = tokenize("price + quantity * 2");
    auto ast = parse_expression(tokens);
    std::string code = ast->to_cuda_expr();
    assert(code == "(price[idx] + (quantity[idx] * 2.0f))");
}

void test_parentheses() {
    auto tokens = tokenize("(price + quantity) * 2");
    auto ast = parse_expression(tokens);
    std::string code = ast->to_cuda_expr();
    assert(code == "((price[idx] + quantity[idx]) * 2.0f)");
}

int main() {
    test_precedence();
    test_parentheses();
    std::cout << "All precedence tests passed\n";
    return 0;
}
