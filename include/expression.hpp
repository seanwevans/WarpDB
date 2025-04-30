#pragma once
#include <string>
#include <vector>
#include <memory>

enum class TokenType {
    Identifier, Number, Operator, End
};

struct Token {
    TokenType type;
    std::string value;
};

std::vector<Token> tokenize(const std::string& input);

enum class ASTNodeType {
    Constant, Variable, BinaryOp
};

struct ASTNode {
    virtual ~ASTNode() {}
    virtual std::string to_cuda_expr() const = 0;
    virtual ASTNodeType type() const = 0;
};

using ASTNodePtr = std::unique_ptr<ASTNode>;

struct ConstantNode : public ASTNode {
    std::string value;
    ConstantNode(const std::string& val) : value(val) {}
    std::string to_cuda_expr() const override { 
      if (value.find('.') == std::string::npos) {
        return value + ".0f";  // if no '.', append '.0f'
      } else {
        return value + "f";    // if already decimal, just add 'f'
      }
    }
    ASTNodeType type() const override { return ASTNodeType::Constant; }
};

struct VariableNode : public ASTNode {
    std::string name;
    VariableNode(const std::string& n) : name(n) {}
    std::string to_cuda_expr() const override { return name + "[idx]"; }
    ASTNodeType type() const override { return ASTNodeType::Variable; }
};

struct BinaryOpNode : public ASTNode {
    std::string op;
    ASTNodePtr left;
    ASTNodePtr right;
    BinaryOpNode(std::string o, ASTNodePtr l, ASTNodePtr r)
        : op(std::move(o)), left(std::move(l)), right(std::move(r)) {}

    std::string to_cuda_expr() const override {
        return "(" + left->to_cuda_expr() + " " + op + " " + right->to_cuda_expr() + ")";
    }

    ASTNodeType type() const override { return ASTNodeType::BinaryOp; }
};

// Entry point
ASTNodePtr parse_expression(const std::vector<Token>& tokens);

