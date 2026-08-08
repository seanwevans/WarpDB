#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <string>

// HAVING clauses are documented as supported, and filtering groups by an
// aggregate is their primary purpose. The host-side evaluator already handles
// AggregationNode inside HAVING; this verifies the parser produces one so
// queries like "HAVING SUM(price) > 10" parse instead of throwing.
static QueryAST parse(const std::string &sql) {
  return parse_query(tokenize(sql));
}

int main() {
  QueryAST q = parse(
      "SELECT SUM(price) FROM test GROUP BY quantity HAVING SUM(price) > 10");
  assert(q.having.has_value() && "expected a HAVING clause");

  auto *cmp = dynamic_cast<const BinaryOpNode *>(q.having->get());
  assert(cmp && cmp->op == ">" && "HAVING should be a comparison");
  auto *agg = dynamic_cast<const AggregationNode *>(cmp->left.get());
  assert(agg && agg->agg == AggregationType::Sum &&
         "left side of HAVING should be a SUM aggregation");

  // Other aggregates and compound predicates should parse as well.
  QueryAST q2 = parse(
      "SELECT AVG(price) FROM test GROUP BY quantity "
      "HAVING AVG(price) > 5 AND COUNT(price) > 1");
  assert(q2.having.has_value());
  auto *root = dynamic_cast<const BinaryOpNode *>(q2.having->get());
  assert(root && root->op == "&&" && "expected AND-combined HAVING predicate");

  std::cout << "having aggregate test passed\n";
  return 0;
}
