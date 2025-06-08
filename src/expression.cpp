#include "expression.hpp"
#include <cctype>
#include <stdexcept>

std::vector<Token> tokenize(const std::string &input) {
  std::vector<Token> tokens;
  size_t i = 0;

  while (i < input.size()) {
    if (std::isspace(static_cast<unsigned char>(input[i]))) {
      i++;
      continue;
    }

    if (std::isalpha(static_cast<unsigned char>(input[i])) || input[i] == '_') {
      std::string ident;
      while (i < input.size() &&
             (std::isalnum(static_cast<unsigned char>(input[i])) ||
              input[i] == '_')) {
        ident += input[i++];
      }
      tokens.push_back({TokenType::Identifier, ident});
    } else if (std::isdigit(static_cast<unsigned char>(input[i])) ||
               input[i] == '.') {
      std::string num;
      while (i < input.size() &&
             (std::isdigit(static_cast<unsigned char>(input[i])) ||
              input[i] == '.')) {
        num += input[i++];
      }
      tokens.push_back({TokenType::Number, num});
    } else if (input[i] == '>' || input[i] == '<' || input[i] == '=' ||
               input[i] == '!') {
      // Handle two-character comparison operators (>=, <=, ==, !=) first
      std::string op(1, input[i]);
      if (i + 1 < input.size() && input[i + 1] == '=') {
        op += '=';
        i++;
      }
      i++;
      tokens.push_back({TokenType::Operator, op});
    } else if (std::string("+-*/()<>,").find(input[i]) != std::string::npos) {
      // Remaining single-character operators
      tokens.push_back({TokenType::Operator, std::string(1, input[i++])});
    } else {
      throw std::runtime_error("Unknown character: " +
                               std::string(1, input[i]));
    }
  }

  tokens.push_back({TokenType::End, ""});
  return tokens;
}

namespace {
size_t current;
std::vector<Token> toks;

const Token &peek() { return toks[current]; }
const Token &advance() { return toks[current++]; }

bool match(const std::string &op) {
  if (peek().type == TokenType::Operator && peek().value == op) {
    advance();
    return true;
  }
  return false;
}

// Forward declarations
ASTNodePtr parse_term();
ASTNodePtr parse_factor();

// Parses: expr = term ( ("+"|"-") term )*
ASTNodePtr parse_expression_internal() {
  ASTNodePtr node = parse_term();
  while (match("+") || match("-")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_term();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

// Parses: comparison = add (comp_op add)* where comp_op is >, <, >=, <=, ==, !=
ASTNodePtr parse_comparison() {
  ASTNodePtr node = parse_expression_internal();
  while (match(">") || match("<") || match(">=") || match("<=") ||
         match("==") || match("!=")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_expression_internal();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

// Parses: term = factor ( ("*"|"/") factor )*
ASTNodePtr parse_term() {
  ASTNodePtr node = parse_factor();
  while (match("*") || match("/")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_factor();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

// Parses: factor = number | identifier | "(" expr ")"
ASTNodePtr parse_factor() {
  const Token &tok = peek();
  if (tok.type == TokenType::Number) {
    advance();
    return std::make_unique<ConstantNode>(tok.value);
  } else if (tok.type == TokenType::Identifier) {
    std::string ident = tok.value;
    advance();
    if (match("(")) {
      std::vector<ASTNodePtr> args;
      if (!match(")")) {
        do {
          args.push_back(parse_expression_internal());
        } while (match(","));
        if (!match(")"))
          throw std::runtime_error("Expected ')' after arguments");
      }
      return std::make_unique<FunctionCallNode>(ident, std::move(args));
    }
    return std::make_unique<VariableNode>(ident);
  } else if (match("(")) {
    ASTNodePtr node = parse_expression_internal();
    if (!match(")"))
      throw std::runtime_error("Expected ')'");
    return node;
  } else {
    throw std::runtime_error("Unexpected token: " + tok.value);
  }
}
} // end anonymous namespace

ASTNodePtr parse_expression(const std::vector<Token> &tokens) {
  current = 0;
  toks = tokens;

  ASTNodePtr node = parse_comparison();
  if (peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + peek().value);
  }
  return node;

}
