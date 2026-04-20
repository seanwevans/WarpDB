#pragma once
#include <memory>
#include <string>
#include <vector>
#include <optional>

enum class TokenType { Identifier, Number, Operator, Keyword, End };

struct Token {
  TokenType type;
  std::string value;
  int line = 1;
  int column = 1;
};

std::vector<Token> tokenize(const std::string &input);

struct WhereClauseSplit {
  std::vector<Token> expression_tokens;
  std::vector<Token> where_tokens;
  bool has_where = false;
};

WhereClauseSplit split_where_clause_tokens(const std::vector<Token> &tokens);

enum class ASTNodeType { Constant, Variable, BinaryOp, FunctionCall, Aggregation };


struct ASTNode {
  virtual ~ASTNode() {}
  virtual std::string to_cuda_expr() const = 0;
  virtual ASTNodeType type() const = 0;
};

using ASTNodePtr = std::unique_ptr<ASTNode>;

struct ConstantNode : public ASTNode {
  std::string value;
  ConstantNode(const std::string &val) : value(val) {}
  std::string to_cuda_expr() const override {
    auto has_suffix_f = !value.empty() &&
                        (value.back() == 'f' || value.back() == 'F');
    if (value.find('e') != std::string::npos ||
        value.find('E') != std::string::npos) {
      if (has_suffix_f)
        return value;
      return value + "f"; // scientific notation needs only 'f'
    }
    if (value.find('.') == std::string::npos) {
      return value + ".0f"; // pure integers promoted to float literal
    }
    if (has_suffix_f)
      return value;
    return value + "f"; // decimal literals get a trailing 'f'
  }
  ASTNodeType type() const override { return ASTNodeType::Constant; }
};

struct VariableNode : public ASTNode {
  std::string name;
  VariableNode(const std::string &n) : name(n) {}
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
    return "(" + left->to_cuda_expr() + " " + op + " " + right->to_cuda_expr() +
           ")";
  }

  ASTNodeType type() const override { return ASTNodeType::BinaryOp; }
};

struct FunctionCallNode : public ASTNode {
  std::string name;
  std::vector<ASTNodePtr> args;
  FunctionCallNode(std::string n, std::vector<ASTNodePtr> a)
      : name(std::move(n)), args(std::move(a)) {}
  std::string to_cuda_expr() const override {
    std::string result = name + "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i > 0)
        result += ", ";
      result += args[i]->to_cuda_expr();
    }
    result += ")";
    return result;
  }
  ASTNodeType type() const override { return ASTNodeType::FunctionCall; }
};

// Entry point
ASTNodePtr parse_expression(const std::vector<Token> &tokens);
ASTNodePtr parse_logical_and(const std::vector<Token> &tokens);
ASTNodePtr parse_logical_or(const std::vector<Token> &tokens);
enum class AggregationType { Sum, Avg, Count, Min, Max };

struct AggregationNode : public ASTNode {
  AggregationType agg;
  ASTNodePtr expr;
  AggregationNode(AggregationType a, ASTNodePtr e)
      : agg(a), expr(std::move(e)) {}
  std::string to_cuda_expr() const override { return expr->to_cuda_expr(); }
  ASTNodeType type() const override { return ASTNodeType::Aggregation; }
  // Helper used by JIT code generation to name the reduction kernel
  std::string agg_kernel() const;
};

struct OrderByClause {
  ASTNodePtr expr;
  bool ascending;
};

struct LimitClause {
  int count;
};

struct OffsetClause {
  int count;
};

struct WindowFunctionNode : public ASTNode {
  AggregationType agg;
  ASTNodePtr expr;
  std::vector<ASTNodePtr> partition_by;
  std::optional<OrderByClause> order_by;
  WindowFunctionNode(AggregationType a, ASTNodePtr e)
      : agg(a), expr(std::move(e)) {}
  std::string to_cuda_expr() const override { return "<window>"; }
  ASTNodeType type() const override { return ASTNodeType::Aggregation; }
};

struct JoinClause {
  std::string table;
  ASTNodePtr condition;
};

struct GroupByClause {
  std::vector<ASTNodePtr> keys;
};

struct QueryAST {
  std::vector<ASTNodePtr> select_list;
  std::string from_table;
  std::vector<JoinClause> joins;
  std::optional<ASTNodePtr> where;
  std::optional<GroupByClause> group_by;
  std::optional<ASTNodePtr> having;
  std::optional<OrderByClause> order_by;
  std::optional<LimitClause> limit;
  std::optional<OffsetClause> offset;
  bool distinct = false;
};

QueryAST parse_query(const std::vector<Token> &tokens);
