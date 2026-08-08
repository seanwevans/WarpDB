#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <string>

// SQL uses a single '=' for equality. Verify that it is normalized so the
// generated CUDA expression performs a comparison ('==') rather than an
// assignment ('='), which would both always evaluate truthy and corrupt the
// column data on the device.
static std::string cuda_for(const std::string &expr) {
  auto tokens = tokenize(expr);
  auto ast = parse_expression(tokens);
  return ast->to_cuda_expr();
}

int main() {
  std::string single = cuda_for("price = 10");
  std::string doubled = cuda_for("price == 10");
  assert(single == doubled &&
         "single '=' should generate the same CUDA as '=='");
  assert(single.find("==") != std::string::npos &&
         "expected an equality comparison in generated CUDA");

  // A parenthesized/compound predicate must not contain a lone '=' either.
  std::string compound = cuda_for("price = 10 AND quantity = 2");
  assert(compound.find(" = ") == std::string::npos &&
         "generated CUDA must not contain an assignment operator");

  std::cout << "equals operator test passed\n";
  return 0;
}
