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

void test_exponent_number_tokenize() {
    auto tokens = tokenize("1e3 .5e+2 2.0E-4");
    assert(tokens.size() == 4);
    assert(tokens[0].type == TokenType::Number && tokens[0].value == "1e3");
    assert(tokens[1].type == TokenType::Number && tokens[1].value == ".5e+2");
    assert(tokens[2].type == TokenType::Number && tokens[2].value == "2.0E-4");
    assert(tokens[3].type == TokenType::End);
}

void test_exponent_to_cuda_suffix() {
    auto tokens = tokenize("1e3 + .5e+2 + 2E-1");
    auto ast = parse_expression(tokens);
    std::string code = ast->to_cuda_expr();
    assert(code == "((1e3f + .5e+2f) + 2E-1f)");
}

int main() {
    test_basic_tokenize();
    test_parentheses_tokenize();
    test_logical_keywords();
    test_exponent_number_tokenize();
    test_exponent_to_cuda_suffix();
    std::cout << "All tokenizer tests passed\n";
    return 0;
}
