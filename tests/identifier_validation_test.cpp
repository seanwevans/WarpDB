#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace {
void validate(const ASTNode* node, const std::unordered_set<std::string>& cols) {
    if (!node) return;
    if (auto v = dynamic_cast<const VariableNode*>(node)) {
        if (cols.find(v->name) == cols.end())
            throw std::runtime_error(std::string("Unknown column: ") + v->name);
    } else if (auto b = dynamic_cast<const BinaryOpNode*>(node)) {
        validate(b->left.get(), cols);
        validate(b->right.get(), cols);
    } else if (auto f = dynamic_cast<const FunctionCallNode*>(node)) {
        for (const auto& a : f->args) validate(a.get(), cols);
    } else if (auto a = dynamic_cast<const AggregationNode*>(node)) {
        validate(a->expr.get(), cols);
    }
}
}

int main() {
    auto tokens = tokenize("SELECT foo FROM test");
    QueryAST ast;
    bool threw = false;
    try {
        ast = parse_query(tokens);
        std::unordered_set<std::string> cols{"price","quantity"};
        validate(ast.select_list[0].get(), cols);
    } catch (const std::runtime_error &e) {
        threw = true;
        std::string msg = e.what();
        assert(msg.find("Unknown column") != std::string::npos);
    }
    assert(threw && "Expected validation to fail");
    std::cout << "identifier_validation_test passed\n";
    return 0;
}
