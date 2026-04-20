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
  int line = 1;
  int column = 1;

  auto advance_char = [&](char c) {
    if (c == '\n') {
      line++; column = 1;
    } else {
      column++;
    }
  };

  while (i < input.size()) {
    if (input[i] == '\n') { advance_char('\n'); i++; continue; }
    if (std::isspace(static_cast<unsigned char>(input[i]))) {
      advance_char(input[i]);
      i++;
      continue;
    }

    if (std::isalpha(static_cast<unsigned char>(input[i])) || input[i] == '_') {
      int start_col = column;
      int start_line = line;
      std::string ident;
      while (i < input.size() &&
             (std::isalnum(static_cast<unsigned char>(input[i])) ||
              input[i] == '_' || input[i] == '.')) {
        ident += input[i];
        advance_char(input[i]);
        i++;
      }
      std::string upper = ident;
      for (auto &c : upper)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      static const std::unordered_set<std::string> keywords = {
          "SELECT",    "FROM",    "WHERE",  "JOIN",   "ON",    "GROUP",
          "BY",        "ORDER",   "ASC",    "DESC",   "LIMIT", "OFFSET",
          "SUM",       "AVG",     "COUNT",  "MIN",    "MAX",   "OVER", 
          "PARTITION", "AND",     "OR",     "HAVING", "DISTINCT"};

      if (keywords.count(upper)) {
        tokens.push_back({TokenType::Keyword, upper, start_line, start_col});
      } else {
        tokens.push_back({TokenType::Identifier, ident, start_line, start_col});
      }
    } else if (std::isdigit(static_cast<unsigned char>(input[i])) ||
               (input[i] == '.' && i + 1 < input.size() &&
                std::isdigit(static_cast<unsigned char>(input[i + 1])))) {
      int start_col = column;
      int start_line = line;
      std::string num;

      while (i < input.size() &&
             std::isdigit(static_cast<unsigned char>(input[i]))) {
        num += input[i];
        advance_char(input[i]);
        i++;
      }

      if (i < input.size() && input[i] == '.') {
        num += input[i];
        advance_char(input[i]);
        i++;

        while (i < input.size() &&
               std::isdigit(static_cast<unsigned char>(input[i]))) {
          num += input[i];
          advance_char(input[i]);
          i++;
        }
      }

      if (i < input.size() && (input[i] == 'e' || input[i] == 'E')) {
        int exp_line = line;
        int exp_col = column;
        num += input[i];
        advance_char(input[i]);
        i++;

        if (i < input.size() && (input[i] == '+' || input[i] == '-')) {
          num += input[i];
          advance_char(input[i]);
          i++;
        }

        if (i >= input.size() ||
            !std::isdigit(static_cast<unsigned char>(input[i]))) {
          std::string msg =
              "Malformed exponent in numeric literal at line " +
              std::to_string(exp_line) + " column " + std::to_string(exp_col);
          throw std::runtime_error(msg);
        }

        while (i < input.size() &&
               std::isdigit(static_cast<unsigned char>(input[i]))) {
          num += input[i];
          advance_char(input[i]);
          i++;
        }
      }
      tokens.push_back({TokenType::Number, num, start_line, start_col});
    } else if (input[i] == '>' || input[i] == '<' || input[i] == '=' ||
               input[i] == '!') {
      // Handle two-character comparison operators (>=, <=, ==, !=) first
      int start_col = column;
      int start_line = line;
      std::string op(1, input[i]);
      if (i + 1 < input.size() && input[i + 1] == '=') {
        op += '=';
        advance_char(input[i]);
        i++;
      }
      advance_char(input[i]);
      i++;
      tokens.push_back({TokenType::Operator, op, start_line, start_col});
    } else if (std::string("+-*/()<>,.").find(input[i]) != std::string::npos) {
      // Remaining single-character operators
      int start_col = column;
      int start_line = line;
      char ch = input[i];
      advance_char(ch);
      i++;
      tokens.push_back({TokenType::Operator, std::string(1, ch), start_line,
                        start_col});
    } else {

      std::string msg = "Unknown character '" + std::string(1, input[i]) +
                        "' at line " + std::to_string(line) + " column " +
                        std::to_string(column);
      throw std::runtime_error(msg);

    }
  }

  tokens.push_back({TokenType::End, "", line, column});
  return tokens;
}

namespace {

class Parser {
public:
  explicit Parser(const std::vector<Token> &tokens) : current(0), toks(tokens) {}

  ASTNodePtr parse_expression() { return parse_logical_or_internal(); }
  ASTNodePtr parse_logical_and() { return parse_logical_and_internal(); }
  ASTNodePtr parse_logical_or() { return parse_logical_or_internal(); }
  QueryAST parse_query();

  const Token &peek() const { return toks[current]; }
  const Token &advance() { return toks[current++]; }

  bool match(const std::string &op) {
    if (peek().type == TokenType::Operator && peek().value == op) {
      advance();
      return true;
    }
    return false;
  }

private:
  ASTNodePtr parse_expression_internal();
  ASTNodePtr parse_comparison();
  ASTNodePtr parse_logical_and_internal();
  ASTNodePtr parse_logical_or_internal();
  ASTNodePtr parse_term();
  ASTNodePtr parse_factor();

  size_t current;
  const std::vector<Token> &toks;
};

ASTNodePtr Parser::parse_expression_internal() {
  ASTNodePtr node = parse_term();
  while (match("+") || match("-")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_term();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

ASTNodePtr Parser::parse_comparison() {
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

ASTNodePtr Parser::parse_logical_and_internal() {
  ASTNodePtr node = parse_comparison();
  while (peek().type == TokenType::Keyword && peek().value == "AND") {
    advance();
    ASTNodePtr right = parse_comparison();
    node = std::make_unique<BinaryOpNode>("&&", std::move(node),
                                          std::move(right));
  }
  return node;
}

ASTNodePtr Parser::parse_logical_or_internal() {
  ASTNodePtr node = parse_logical_and_internal();
  while (peek().type == TokenType::Keyword && peek().value == "OR") {
    advance();
    ASTNodePtr right = parse_logical_and_internal();
    node = std::make_unique<BinaryOpNode>("||", std::move(node),
                                          std::move(right));
  }
  return node;
}

ASTNodePtr Parser::parse_term() {
  ASTNodePtr node = parse_factor();
  while (match("*") || match("/")) {
    std::string op = toks[current - 1].value;
    ASTNodePtr right = parse_factor();
    node =
        std::make_unique<BinaryOpNode>(op, std::move(node), std::move(right));
  }
  return node;
}

ASTNodePtr Parser::parse_factor() {
  if (match("+")) {
    return parse_factor();
  }
  if (match("-")) {
    auto operand = parse_factor();
    auto minus_one = std::make_unique<ConstantNode>("-1");
    return std::make_unique<BinaryOpNode>("*", std::move(minus_one),
                                          std::move(operand));
  }

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

QueryAST Parser::parse_query() {
  size_t end = toks.size();
  if (end > 0 && toks[end - 1].type == TokenType::End)
    --end;
  current = 0;

  auto expect_kw = [&](const std::string &kw) {
    if (current >= toks.size() || toks[current].type != TokenType::Keyword ||
        toks[current].value != kw) {
      int l = current < toks.size() ? toks[current].line : toks.back().line;
      int c = current < toks.size() ? toks[current].column : toks.back().column;
      throw std::runtime_error("Expected keyword '" + kw + "' at line " +
                               std::to_string(l) + " column " +
                               std::to_string(c));
    }

    current++;
  };

  QueryAST query;
  expect_kw("SELECT");
  if (current < toks.size() && toks[current].type == TokenType::Keyword &&
      toks[current].value == "DISTINCT") {
    query.distinct = true;
    current++;
  }

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
          inner.push_back({TokenType::End, "", 0, 0});
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
            return std::make_unique<WindowFunctionNode>(
                at, ::parse_expression(inner));
          }
          return std::make_unique<AggregationNode>(at, ::parse_expression(inner));
        } else {
          throw std::runtime_error("Invalid syntax for " + kw + " aggregation");
        }
      }
    }
    std::vector<Token> tmp = it;
    tmp.push_back({TokenType::End, "", 0, 0});
    return ::parse_expression(tmp);
  };

  while (current < end) {
    if (toks[current].type == TokenType::Keyword &&
        toks[current].value == "FROM")
      break;
    std::vector<Token> item;
    int depth = 0;
    while (current < end) {
      if (toks[current].type == TokenType::Operator &&
          toks[current].value == "(")
        depth++;
      if (toks[current].type == TokenType::Operator &&
          toks[current].value == ")")
        depth--;
      if (depth == 0 &&
          ((toks[current].type == TokenType::Operator &&
            toks[current].value == ",") ||
           (toks[current].type == TokenType::Keyword &&
            toks[current].value == "FROM")))
        break;
      item.push_back(toks[current++]);
    }
    query.select_list.push_back(parse_select_item(item));
    if (current < end && toks[current].type == TokenType::Operator &&
        toks[current].value == ",")
      current++; // skip comma
  }

  expect_kw("FROM");

  if (current >= toks.size() || toks[current].type != TokenType::Identifier) {
    int l = current < toks.size() ? toks[current].line : toks.back().line;
    int c = current < toks.size() ? toks[current].column : toks.back().column;
    throw std::runtime_error("Expected table name after FROM at line " +
                             std::to_string(l) + " column " + std::to_string(c));
  }

  query.from_table = toks[current++].value;

  while (current < toks.size() && toks[current].type == TokenType::Keyword &&
         toks[current].value == "JOIN") {
    current++;

    if (current >= toks.size() || toks[current].type != TokenType::Identifier) {
      int l = current < toks.size() ? toks[current].line : toks.back().line;
      int c = current < toks.size() ? toks[current].column : toks.back().column;
      throw std::runtime_error("Expected table name after JOIN at line " +
                               std::to_string(l) + " column " +
                               std::to_string(c));
    }
    JoinClause jc;
    jc.table = toks[current++].value;
    expect_kw("ON");
    size_t start = current;
    while (current < end &&
           !(toks[current].type == TokenType::Keyword &&
             (toks[current].value == "WHERE" ||
              toks[current].value == "GROUP" || toks[current].value == "ORDER" ||
              toks[current].value == "HAVING" ||
              toks[current].value == "JOIN" || toks[current].value == "LIMIT")))
      current++;
    std::vector<Token> cond(toks.begin() + start, toks.begin() + current);
    cond.push_back({TokenType::End, "", 0, 0});
    jc.condition = ::parse_expression(cond);
    query.joins.push_back(std::move(jc));
  }

  if (current < end && toks[current].type == TokenType::Keyword &&
      toks[current].value == "WHERE") {
    current++;
    size_t start = current;
    while (current < end &&
           !(toks[current].type == TokenType::Keyword &&
             (toks[current].value == "GROUP" ||
              toks[current].value == "ORDER" ||
              toks[current].value == "HAVING" ||
              toks[current].value == "LIMIT")))
      current++;
    std::vector<Token> w(toks.begin() + start, toks.begin() + current);
    w.push_back({TokenType::End, "", 0, 0});
    query.where = ::parse_expression(w);
  }

  if (current < end && toks[current].type == TokenType::Keyword &&
      toks[current].value == "GROUP") {
    current++;
    expect_kw("BY");
    GroupByClause gb;
    while (current < end) {
      size_t start = current;
      while (current < end &&
             !(toks[current].type == TokenType::Operator &&
               toks[current].value == ",") &&
             !(toks[current].type == TokenType::Keyword &&
               (toks[current].value == "ORDER" ||
                toks[current].value == "HAVING")))
        current++;
      std::vector<Token> key(toks.begin() + start, toks.begin() + current);
      key.push_back({TokenType::End, "", 0, 0});
      gb.keys.push_back(::parse_expression(key));
      if (current < end && toks[current].type == TokenType::Operator &&
          toks[current].value == ",")
        current++;

      if (current < toks.size() && toks[current].type == TokenType::Keyword &&
          (toks[current].value == "ORDER" ||
           toks[current].value == "HAVING"))
        break;
    }
    query.group_by = std::move(gb);
  }

  if (current < toks.size() && toks[current].type == TokenType::Keyword &&
      toks[current].value == "HAVING") {
    current++;
    size_t start = current;
    while (current < toks.size() &&
           !(toks[current].type == TokenType::Keyword &&
             (toks[current].value == "ORDER" ||
              toks[current].value == "LIMIT" ||
              toks[current].value == "OFFSET")))
      current++;
    std::vector<Token> hv(toks.begin() + start, toks.begin() + current);
    hv.push_back({TokenType::End, ""});
    query.having = ::parse_expression(hv);
    if (current == toks.size())
      end = current;
  }

  if (current < toks.size() && toks[current].type == TokenType::Keyword &&
      toks[current].value == "ORDER") {
    current++;
    expect_kw("BY");
    size_t start = current;
    while (current < end &&
           !(toks[current].type == TokenType::Keyword &&
             (toks[current].value == "ASC" ||
              toks[current].value == "DESC")))
      current++;
    std::vector<Token> ord(toks.begin() + start, toks.begin() + current);
    ord.push_back({TokenType::End, "", 0, 0});
    OrderByClause ob;
    ob.expr = ::parse_expression(ord);
    ob.ascending = true;
    if (current < end && toks[current].type == TokenType::Keyword &&
        (toks[current].value == "ASC" || toks[current].value == "DESC")) {
      ob.ascending = toks[current].value == "ASC";
      current++;
    }
    query.order_by = std::move(ob);
  }

  if (current < end && toks[current].type == TokenType::Keyword &&
      toks[current].value == "LIMIT") {
    current++;

    if (current >= toks.size() || toks[current].type != TokenType::Number) {
      int l = current < toks.size() ? toks[current].line : toks.back().line;
      int c = current < toks.size() ? toks[current].column : toks.back().column;
      throw std::runtime_error("Expected numeric value after LIMIT at line " +
                               std::to_string(l) + " column " +
                               std::to_string(c));
    }

    LimitClause lc{std::stoi(toks[current].value)};
    current++;
    query.limit = lc;
  }

  if (current < toks.size() && toks[current].type == TokenType::Keyword &&
      toks[current].value == "OFFSET") {
    current++;
    if (current >= toks.size() || toks[current].type != TokenType::Number)
      throw std::runtime_error("Expected numeric value after OFFSET");
    OffsetClause oc{std::stoi(toks[current].value)};
    current++;
    query.offset = oc;
  }

  if (current != end) {
    throw std::runtime_error("Unexpected token in query near: " +
                             toks[current].value);
  }

  return query;
}

} // end anonymous namespace

ASTNodePtr parse_expression(const std::vector<Token> &tokens) {
  Parser p(tokens);
  ASTNodePtr node = p.parse_expression();
  if (p.peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + p.peek().value);
  }
  return node;
}

ASTNodePtr parse_logical_and(const std::vector<Token> &tokens) {
  Parser p(tokens);
  ASTNodePtr node = p.parse_logical_and();
  if (p.peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + p.peek().value);
  }
  return node;
}

ASTNodePtr parse_logical_or(const std::vector<Token> &tokens) {
  Parser p(tokens);
  ASTNodePtr node = p.parse_logical_or();
  if (p.peek().type != TokenType::End) {
    throw std::runtime_error("Unexpected tokens remaining: " + p.peek().value);
  }
  return node;
}

QueryAST parse_query(const std::vector<Token> &tokens) {
  Parser p(tokens);
  return p.parse_query();
}

std::string AggregationNode::agg_kernel() const {
  switch (agg) {
  case AggregationType::Sum:
    return "sum";
  case AggregationType::Avg:
    return "avg";
  case AggregationType::Count:
    return "count";
  case AggregationType::Min:
    return "min";
  case AggregationType::Max:
    return "max";
  }
  return "";
}
