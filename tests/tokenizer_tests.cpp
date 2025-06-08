#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_basic_tokenize() {
    auto tokens = tokenize("price > 10");
    assert(tokens.size() == 4);
    assert(tokens[0].type == TokenType::Identifier && tokens[0].value == "price");
    assert(tokens[1].type == TokenType::Operator && tokens[1].value == ">");
    assert(tokens[2].type == TokenType::Number && tokens[2].value == "10");
    assert(tokens[3].type == TokenType::End);
}

void test_parentheses_tokenize() {
    auto tokens = tokenize("(price + 5) * quantity");
    std::vector<TokenType> expected = {
        TokenType::Operator,    // (
        TokenType::Identifier,  // price
        TokenType::Operator,    // +
        TokenType::Number,      // 5
        TokenType::Operator,    // )
        TokenType::Operator,    // *
        TokenType::Identifier,  // quantity
        TokenType::End
    };
    assert(tokens.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(tokens[i].type == expected[i]);
    }
}

void test_logical_keywords() {
    auto tokens = tokenize("price > 10 AND quantity < 5");
    bool found_and = false;
    bool found_or = false;
    for (const auto &t : tokens) {
        if (t.type == TokenType::Keyword && t.value == "AND") found_and = true;
        if (t.type == TokenType::Keyword && t.value == "OR") found_or = true;
    }
    assert(found_and && !found_or);
}

int main() {
    test_basic_tokenize();
    test_parentheses_tokenize();
    test_logical_keywords();
    std::cout << "All tokenizer tests passed\n";
    return 0;
}
