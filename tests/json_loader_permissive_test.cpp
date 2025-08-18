#include "json_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
  bool threw = false;
  HostTable table;
  try {
    table = load_json_to_host("data/malformed.json", ParsePolicy::Permissive);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  assert(!threw && "Permissive policy should not throw");
  assert(table.num_rows() == 2 && "Expected 2 valid rows");
  std::cout << "json loader permissive test passed\n";
  return 0;
}
