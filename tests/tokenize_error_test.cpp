#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    bool threw = false;
    try {
        auto t = tokenize("price # 1\n");
        (void)t;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("line 1") != std::string::npos);
        assert(msg.find("column") != std::string::npos);
    }
    assert(threw && "Expected tokenizer to throw");
    std::cout << "tokenize_error_test passed\n";
    return 0;
}
