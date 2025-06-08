#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    bool threw = false;
    try {
        auto tokens = tokenize("SELECT price");
        parse_query(tokens);
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("line") != std::string::npos);
        assert(msg.find("column") != std::string::npos);
    }
    assert(threw && "Expected parse_query to throw");
    std::cout << "parse_query_error_test passed\n";
    return 0;
}
