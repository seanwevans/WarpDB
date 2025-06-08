#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

void test_malformed_expression() {
    auto tokens = tokenize("1 2");
    bool threw = false;
    try {
        auto ast = parse_expression(tokens);
        (void)ast;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Unexpected token") != std::string::npos);
    }
    assert(threw && "Expected exception for malformed expression");
}

int main() {
    test_malformed_expression();
    std::cout << "All tests passed\n";
    return 0;
}
