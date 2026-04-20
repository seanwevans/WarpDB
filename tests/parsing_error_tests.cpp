#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>

void test_invalid_character() {
    bool threw = false;
    try {
        auto toks = tokenize("price & 5");
        (void)toks;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Unknown character") != std::string::npos);
    }
    assert(threw && "Expected tokenizer failure");
}

void test_unexpected_token_query() {
    bool threw = false;
    try {
        auto tokens = tokenize("SELECT price FROM test EXTRA");
        QueryAST q = parse_query(tokens);
        (void)q;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Unexpected token") != std::string::npos);
    }
    assert(threw && "Expected parse_query to fail");
}

void test_unbalanced_parentheses() {
    bool threw = false;
    try {
        auto tokens = tokenize("(price + 5");
        auto ast = parse_expression(tokens);
        (void)ast;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Expected ')'" ) != std::string::npos);
    }
    assert(threw && "Expected expression parse failure");
}

void test_malformed_exponent_missing_digits() {
    bool threw = false;
    try {
        auto tokens = tokenize("1e");
        (void)tokens;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Malformed exponent in numeric literal") != std::string::npos);
        assert(msg.find("line 1 column 2") != std::string::npos);
    }
    assert(threw && "Expected malformed exponent error");
}

void test_malformed_exponent_missing_digits_after_sign() {
    bool threw = false;
    try {
        auto tokens = tokenize("1e+");
        (void)tokens;
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Malformed exponent in numeric literal") != std::string::npos);
        assert(msg.find("line 1 column 2") != std::string::npos);
    }
    assert(threw && "Expected malformed exponent error with explicit sign");
}

int main() {
    test_invalid_character();
    test_unexpected_token_query();
    test_unbalanced_parentheses();
    test_malformed_exponent_missing_digits();
    test_malformed_exponent_missing_digits_after_sign();
    std::cout << "All regression tests passed\n";
    return 0;
}
