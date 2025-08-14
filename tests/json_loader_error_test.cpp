#include "json_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
  bool threw = false;
  try {
    load_json_to_host("data/malformed.json");
  } catch (const std::runtime_error &) {
    threw = true;
    std::cout << "Caught JSON parse error\n";
  }
  assert(threw && "Expected load_json_to_host to throw on malformed data");
  std::cout << "json loader error test passed\n";
  return 0;
}
