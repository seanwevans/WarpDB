#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <string>

// "SELECT *" is parsed by setting QueryAST::select_all and leaving the
// select_list empty; query_sql expands it into one VariableNode per table
// column at execution time. The parser only accepts '*' as the sole item.
static QueryAST parse(const std::string &sql) {
  return parse_query(tokenize(sql));
}

static bool throws(const std::string &sql) {
  try {
    parse(sql);
    return false;
  } catch (const std::exception &) {
    return true;
  }
}

int main() {
  QueryAST q = parse("SELECT * FROM test");
  assert(q.select_all && "expected select_all to be set");
  assert(q.select_list.empty() && "select_list stays empty for '*'");
  assert(q.from_table == "test");

  // '*' combines with the rest of the clauses.
  QueryAST q2 = parse("SELECT * FROM test WHERE price > 10 ORDER BY price LIMIT 2");
  assert(q2.select_all && q2.where.has_value() && q2.order_by.has_value() &&
         q2.limit.has_value());

  // A normal projection is unaffected (no select_all).
  QueryAST q3 = parse("SELECT price, quantity FROM test");
  assert(!q3.select_all && q3.select_list.size() == 2);

  // '*' must be alone: mixed forms are rejected.
  assert(throws("SELECT *, price FROM test"));
  assert(throws("SELECT price, * FROM test"));

  // Multiplication is still multiplication.
  QueryAST q4 = parse("SELECT price * quantity FROM test");
  assert(!q4.select_all && q4.select_list.size() == 1);

  std::cout << "select star test passed\n";
  return 0;
}
