#include "expression.hpp"
#include <cassert>
#include <iostream>

int main() {
    // price > 10
    auto tokens1 = tokenize("price > 10");
    auto ast1 = parse_expression(tokens1);
    std::string code1 = ast1->to_cuda_expr();
    assert(code1 == "(price[idx] > 10.0f)");

    // quantity <= 5
    auto tokens2 = tokenize("quantity <= 5");
    auto ast2 = parse_expression(tokens2);
    std::string code2 = ast2->to_cuda_expr();
    assert(code2 == "(quantity[idx] <= 5.0f)");

    // custom function call
    auto tokens3 = tokenize("discount(price, 0.9)");
    auto ast3 = parse_expression(tokens3);
    std::string code3 = ast3->to_cuda_expr();
    assert(code3 == "discount(price[idx], 0.9f)");

    // logical AND/OR
    auto tokens4 = tokenize("price > 10 AND quantity < 5");
    auto ast4 = parse_expression(tokens4);
    std::string code4 = ast4->to_cuda_expr();
    assert(code4 == "((price[idx] > 10.0f) && (quantity[idx] < 5.0f))");

    auto tokens5 = tokenize("price > 10 OR quantity < 5");
    auto ast5 = parse_expression(tokens5);
    std::string code5 = ast5->to_cuda_expr();
    assert(code5 == "((price[idx] > 10.0f) || (quantity[idx] < 5.0f))");

    std::cout << "All parser tests passed\n";
    return 0;
}
