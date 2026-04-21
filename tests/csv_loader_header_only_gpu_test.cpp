#include "csv_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
    bool threw = false;
    Table table;
    try {
        table = load_csv_to_gpu("data/header_only.csv");
    } catch (const std::exception &) {
        threw = true;
    }

    assert(!threw && "Expected load_csv_to_gpu to succeed for header-only CSV");
    assert(table.num_rows == 0);

    std::cout << "CSV header-only GPU load test passed\n";
    return 0;
}
