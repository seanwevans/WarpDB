#include "expression.hpp"
#include <cassert>
#include <iostream>
#include <string>

// SQL's '<>' is the not-equal operator. Verify the tokenizer normalizes it to
// '!=' and that it generates the same CUDA as '!=' (and never a stray '<' '>'
// pair, which would parse as two separate comparisons).
static std::string cuda_for(const std::string &expr) {
  return parse_expression(tokenize(expr))->to_cuda_expr();
}

int main() {
  auto toks = tokenize("price <> 10");
  // Expect: Identifier, Operator '!=', Number, End
  assert(toks.size() == 4);
  assert(toks[1].type == TokenType::Operator && toks[1].value == "!=" &&
         "'<>' should tokenize as a single '!=' operator");

  std::string diamond = cuda_for("price <> 10");
  std::string bang = cuda_for("price != 10");
  assert(diamond == bang && "'<>' should generate the same CUDA as '!='");
  assert(diamond.find("!=") != std::string::npos);

  // Works inside a compound predicate too.
  auto q = parse_query(tokenize("SELECT price FROM t WHERE price <> 10"));
  assert(q.where.has_value());

  std::cout << "not-equal operator test passed\n";
  return 0;
}
