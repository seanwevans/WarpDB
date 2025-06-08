#include "expression.hpp"
#include <cctype>
#include <unordered_set>
#include <stdexcept>

static const char *token_type_name(TokenType t) {
  switch (t) {
  case TokenType::Identifier:
    return "Identifier";
  case TokenType::Number:
    return "Number";
  case TokenType::Operator:
    return "Operator";
  case TokenType::Keyword:
    return "Keyword";
  case TokenType::End:
    return "End";
  }
  return "Unknown";
}

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
              input[i] == '_' || input[i] == '.')) {
        ident += input[i++];
      }
      std::string upper = ident;
      for (auto &c : upper)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      static const std::unordered_set<std::string> keywords = {
          "SELECT",   "FROM",  "WHERE", "JOIN", "ON",  "GROUP",
          "BY",       "ORDER", "ASC",  "DESC", "LIMIT", "SUM", "AVG",
          "AVG",      "COUNT", "MIN",  "MAX",  "OVER", "PARTITION",          
          "COUNT",    "MIN",   "MAX",  "OVER", "PARTITION",
          "AND",      "OR"};
      if (keywords.count(upper)) {
        tokens.push_back({TokenType::Keyword, upper});
      } else {
        tokens.push_back({TokenType::Identifier, ident});
      }
    } else if (std::isdigit(static_cast<unsigned char>(input[i])) ||
               (input[i] == '.' && i + 1 < input.size() && std::isdigit(static_cast<unsigned char>(input[i + 1])))) {
      std::string num;
      bool has_dot = false;
      while (i < input.size() &&
             (std::isdigit(static_cast<unsigned char>(input[i])) ||
              (!has_dot && input[i] == '.'))) {
        if (input[i] == '.') has_dot = true;
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
    } else if (std::string("+-*/()<>,.").find(input[i]) != std::string::npos) {
      // Remaining single-character operators
      tokens.push_back({TokenType::Operator, std::string(1, input[i++])});
    } else {
      throw std::runtime_error("Unknown character '" +
                               std::string(1, input[i]) +
                               "' at position " + std::to_string(i));
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
ASTNodePtr parse_logical_or_internal();
ASTNodePtr parse_logical_and_internal();

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
         match("==") || match("!=") || match("=")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_expression_internal();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

// Parses: logical_and = comparison (AND comparison)*
ASTNodePtr parse_logical_and_internal() {
  ASTNodePtr node = parse_comparison();
  while (peek().type == TokenType::Keyword && peek().value == "AND") {
    advance();
    ASTNodePtr right = parse_comparison();
    node = std::make_unique<BinaryOpNode>("&&", std::move(node),
                                          std::move(right));
  }
  return node;
}

// Parses: logical_or = logical_and (OR logical_and)*
ASTNodePtr parse_logical_or_internal() {
  ASTNodePtr node = parse_logical_and_internal();
  while (peek().type == TokenType::Keyword && peek().value == "OR") {
    advance();
    ASTNodePtr right = parse_logical_and_internal();
    node = std::make_unique<BinaryOpNode>("||", std::move(node),
                                          std::move(right));
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
    throw std::runtime_error(std::string("Unexpected token (") +
                             token_type_name(tok.type) + ": " + tok.value +
                             ")");
  }
}
} // end anonymous namespace

ASTNodePtr parse_expression(const std::vector<Token> &tokens) {
  current = 0;
  toks = tokens;

  ASTNodePtr node = parse_logical_or_internal();
  if (peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + peek().value);
  }
  return node;

}

ASTNodePtr parse_logical_and(const std::vector<Token> &tokens) {
  current = 0;
  toks = tokens;
  ASTNodePtr node = parse_logical_and_internal();
  if (peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + peek().value);
  }
  return node;
}

ASTNodePtr parse_logical_or(const std::vector<Token> &tokens) {
  current = 0;
  toks = tokens;
  ASTNodePtr node = parse_logical_or_internal();
  if (peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + peek().value);
  }
  return node;
}

QueryAST parse_query(const std::vector<Token> &tokens) {
  size_t end = tokens.size();
  if (end > 0 && tokens[end - 1].type == TokenType::End)
    --end;
  size_t pos = 0;
  auto expect_kw = [&](const std::string &kw) {
    if (pos >= end || tokens[pos].type != TokenType::Keyword ||
        tokens[pos].value != kw)
      throw std::runtime_error("Expected keyword: " + kw);
    pos++;
  };

  QueryAST query;
  expect_kw("SELECT");

  auto parse_select_item = [&](const std::vector<Token> &it) -> ASTNodePtr {
    if (!it.empty() && it[0].type == TokenType::Keyword) {
      std::string kw = it[0].value;
      if (kw == "SUM" || kw == "AVG" || kw == "COUNT" || kw == "MIN" ||
          kw == "MAX") {
        size_t over_idx = it.size();
        for (size_t i = 0; i < it.size(); ++i) {
          if (it[i].type == TokenType::Keyword && it[i].value == "OVER") {
            over_idx = i;
            break;
          }
        }
        bool has_paren = over_idx > 1 && it[1].type == TokenType::Operator &&
                         it[1].value == "(" &&
                         it[over_idx - 1].type == TokenType::Operator &&
                         it[over_idx - 1].value == ")";
        if (has_paren) {
          std::vector<Token> inner(it.begin() + 2, it.begin() + over_idx - 1);
          inner.push_back({TokenType::End, ""});
          AggregationType at = AggregationType::Sum;
          if (kw == "AVG")
            at = AggregationType::Avg;
          else if (kw == "COUNT")
            at = AggregationType::Count;
          else if (kw == "MIN")
            at = AggregationType::Min;
          else if (kw == "MAX")
            at = AggregationType::Max;
          if (over_idx < it.size()) {
            return std::make_unique<WindowFunctionNode>(at,
                                                       parse_expression(inner));
          }
          return std::make_unique<AggregationNode>(at, parse_expression(inner));
        } else {
          throw std::runtime_error("Invalid syntax for " + kw + " aggregation");
        }
      }
    }
    std::vector<Token> tmp = it;
    tmp.push_back({TokenType::End, ""});
    return parse_expression(tmp);
  };

  while (pos < end) {
    if (tokens[pos].type == TokenType::Keyword && tokens[pos].value == "FROM")
      break;
    std::vector<Token> item;
    int depth = 0;
    while (pos < end) {
      if (tokens[pos].type == TokenType::Operator && tokens[pos].value == "(")
        depth++;
      if (tokens[pos].type == TokenType::Operator && tokens[pos].value == ")")
        depth--;
      if (depth == 0 &&
          ((tokens[pos].type == TokenType::Operator &&
            tokens[pos].value == ",") ||
           (tokens[pos].type == TokenType::Keyword &&
            tokens[pos].value == "FROM")))
        break;
      item.push_back(tokens[pos++]);
    }
    query.select_list.push_back(parse_select_item(item));
    if (pos < end && tokens[pos].type == TokenType::Operator &&
        tokens[pos].value == ",")
      pos++; // skip comma
  }

  expect_kw("FROM");
  if (pos >= end || tokens[pos].type != TokenType::Identifier)
    throw std::runtime_error("Expected table name after FROM");
  query.from_table = tokens[pos++].value;

  if (pos < end && tokens[pos].type == TokenType::Keyword &&
      tokens[pos].value == "JOIN") {
    pos++;
    if (pos >= end || tokens[pos].type != TokenType::Identifier)
      throw std::runtime_error("Expected table name after JOIN");
    JoinClause jc;
    jc.table = tokens[pos++].value;
    expect_kw("ON");
    size_t start = pos;
    while (pos < end &&
           !(tokens[pos].type == TokenType::Keyword &&
             (tokens[pos].value == "WHERE" || tokens[pos].value == "GROUP" ||
              tokens[pos].value == "ORDER")))
      pos++;
    std::vector<Token> cond(tokens.begin() + start, tokens.begin() + pos);
    cond.push_back({TokenType::End, ""});
    jc.condition = parse_expression(cond);
    query.join = std::move(jc);
  }

  if (pos < end && tokens[pos].type == TokenType::Keyword &&
      tokens[pos].value == "WHERE") {
    pos++;
    size_t start = pos;
    while (pos < end &&
           !(tokens[pos].type == TokenType::Keyword &&
             (tokens[pos].value == "GROUP" || tokens[pos].value == "ORDER")))
      pos++;
    std::vector<Token> w(tokens.begin() + start, tokens.begin() + pos);
    w.push_back({TokenType::End, ""});
    query.where = parse_expression(w);
  }

  if (pos < end && tokens[pos].type == TokenType::Keyword &&
      tokens[pos].value == "GROUP") {
    pos++;
    expect_kw("BY");
    GroupByClause gb;
    while (pos < end) {
      size_t start = pos;
      while (pos < end &&
             !(tokens[pos].type == TokenType::Operator &&
               tokens[pos].value == ",") &&
             !(tokens[pos].type == TokenType::Keyword &&
               tokens[pos].value == "ORDER"))
        pos++;
      std::vector<Token> key(tokens.begin() + start, tokens.begin() + pos);
      key.push_back({TokenType::End, ""});
      gb.keys.push_back(parse_expression(key));
      if (pos < end && tokens[pos].type == TokenType::Operator &&
          tokens[pos].value == ",")
        pos++;
      if (pos < end && tokens[pos].type == TokenType::Keyword &&
          tokens[pos].value == "ORDER")
        break;
    }
    query.group_by = std::move(gb);
  }

  if (pos < end && tokens[pos].type == TokenType::Keyword &&
      tokens[pos].value == "ORDER") {
    pos++;
    expect_kw("BY");
    size_t start = pos;
    while (pos < end &&
           !(tokens[pos].type == TokenType::Keyword &&
             (tokens[pos].value == "ASC" || tokens[pos].value == "DESC")))
      pos++;
    std::vector<Token> ord(tokens.begin() + start, tokens.begin() + pos);
    ord.push_back({TokenType::End, ""});
    OrderByClause ob;
    ob.expr = parse_expression(ord);
    ob.ascending = true;
    if (pos < end && tokens[pos].type == TokenType::Keyword &&
        (tokens[pos].value == "ASC" || tokens[pos].value == "DESC")) {
      ob.ascending = tokens[pos].value == "ASC";
      pos++;
    }
    query.order_by = std::move(ob);
  }

  if (pos < end && tokens[pos].type == TokenType::Keyword &&
      tokens[pos].value == "LIMIT") {
    pos++;
    if (pos >= end || tokens[pos].type != TokenType::Number)
      throw std::runtime_error("Expected numeric value after LIMIT");
    LimitClause lc{std::stoi(tokens[pos].value)};
    pos++;
    query.limit = lc;
  }

  if (pos != end) {
    throw std::runtime_error("Unexpected token in query near: " +
                             tokens[pos].value);
  }

  return query;
}
