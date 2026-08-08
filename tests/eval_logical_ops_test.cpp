#include "expression.hpp"
#include "eval_helpers.hpp"
#include <cassert>
#include <iostream>
#include <vector>

// The host-side interpreter (eval_helpers.hpp) drives WHERE filtering, ORDER BY
// keys, and non-column SELECT expressions in query_sql. It must evaluate the
// logical operators produced by AND/OR; otherwise a predicate like
// "price > 10 AND qty < 5" collapses to 0 and filters out every row.
static int count_passing(const std::string &predicate, const HostTable &t) {
  auto ast = parse_expression(tokenize(predicate));
  int n = 0;
  for (int i = 0; i < t.num_rows(); ++i)
    if (eval_condition(ast.get(), t, i))
      ++n;
  return n;
}

int main() {
  HostTable t;
  t.columns.push_back({"price", DataType::Float32,
                       std::vector<float>{5.0f, 20.0f, 30.0f}});
  t.columns.push_back({"qty", DataType::Int32, std::vector<int32_t>{1, 4, 2}});

  // rows: (5,1) (20,4) (30,2)
  assert(count_passing("price > 10 AND qty < 5", t) == 2); // (20,4),(30,2)
  assert(count_passing("price > 25 OR qty < 2", t) == 2);  // (5,1),(30,2)
  assert(count_passing("price > 10 AND qty < 3", t) == 1); // (30,2)
  assert(count_passing("price > 100 OR qty > 100", t) == 0);

  // Sanity: a bare comparison still works.
  assert(count_passing("price > 10", t) == 2);

  std::cout << "eval logical ops test passed\n";
  return 0;
}
