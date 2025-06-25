#include "expression.hpp"
#include <cassert>
#include <iostream>

int main() {
    {
        std::string q = "SELECT price FROM test GROUP BY price HAVING price > 5";
        auto tokens = tokenize(q);
        QueryAST ast = parse_query(tokens);
        assert(ast.having.has_value());
        assert(!ast.offset.has_value());
    }
    {
        std::string q = "SELECT price FROM test GROUP BY price HAVING price > 5 OFFSET 2";
        auto tokens = tokenize(q);
        QueryAST ast = parse_query(tokens);
        assert(ast.having.has_value());
        assert(ast.offset.has_value());
        assert(ast.offset->count == 2);
    }
    std::cout << "HAVING OFFSET parse tests passed\n";
    return 0;
}
