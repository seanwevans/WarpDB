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

    // unary minus on literal
    auto tokens6 = tokenize("-5");
    auto ast6 = parse_expression(tokens6);
    std::string code6 = ast6->to_cuda_expr();
    assert(code6 == "(-1.0f * 5.0f)");

    // unary minus on parenthesized expression
    auto tokens7 = tokenize("-(3 + 4)");
    auto ast7 = parse_expression(tokens7);
    std::string code7 = ast7->to_cuda_expr();
    assert(code7 == "(-1.0f * (3.0f + 4.0f))");

    // split WHERE on keyword token only
    auto tokens8 = tokenize("price * 2 WHERE quantity >= 2");
    auto split8 = split_where_clause_tokens(tokens8);
    assert(split8.has_where);
    auto expr8 = parse_expression(split8.expression_tokens);
    auto where8 = parse_expression(split8.where_tokens);
    assert(expr8->to_cuda_expr() == "(price[idx] * 2.0f)");
    assert(where8->to_cuda_expr() == "(quantity[idx] >= 2.0f)");

    // identifiers containing "where" must not split the clause
    auto tokens9 = tokenize("somewhere_value + 1");
    auto split9 = split_where_clause_tokens(tokens9);
    assert(!split9.has_where);
    assert(split9.where_tokens.empty());
    auto expr9 = parse_expression(split9.expression_tokens);
    assert(expr9->to_cuda_expr() == "(somewhere_value[idx] + 1.0f)");

    auto tokens10 = tokenize("anyWHERE_metric + 3");
    auto split10 = split_where_clause_tokens(tokens10);
    assert(!split10.has_where);
    assert(split10.where_tokens.empty());
    auto expr10 = parse_expression(split10.expression_tokens);
    assert(expr10->to_cuda_expr() == "(anyWHERE_metric[idx] + 3.0f)");

    std::cout << "All parser tests passed\n"; 
    return 0; 
}
