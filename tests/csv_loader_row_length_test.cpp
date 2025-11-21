#include "csv_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::vector<DataType> schema = {DataType::Float32, DataType::Int32};

    bool threw = false;
    try {
        load_csv_to_host("data/malformed_row_counts.csv", schema);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "Expected load_csv_to_host to throw on row field count mismatch");

    HostTable table = load_csv_to_host("data/malformed_row_counts.csv", schema, ParsePolicy::Permissive);
    const auto &price = std::get<std::vector<float>>(table.columns[0].data);
    const auto &quantity = std::get<std::vector<int32_t>>(table.columns[1].data);
    assert(price.size() == 2 && quantity.size() == 2);
    assert(price[0] == 1.5f && quantity[0] == 10);
    assert(price[1] == 4.0f && quantity[1] == 40);

    std::cout << "csv loader row length test passed\n";
    return 0;
}
