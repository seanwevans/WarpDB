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

    std::cout << "All parser tests passed\n";
    return 0;
}
