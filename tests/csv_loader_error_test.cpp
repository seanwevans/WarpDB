#include "csv_loader.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::vector<DataType> schema = {DataType::Float32, DataType::Int32};

    bool threw = false;
    try {
        load_csv_to_host("data/malformed.csv", schema);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "Expected load_csv_to_host to throw on malformed data");

    HostTable table = load_csv_to_host("data/malformed.csv", schema, ParsePolicy::Permissive);
    const auto &price = std::get<std::vector<float>>(table.columns[0].data);
    const auto &quantity = std::get<std::vector<int32_t>>(table.columns[1].data);
    assert(price.size() == 3 && quantity.size() == 3);
    assert(price[0] == 1.5f);
    assert(price[1] == 0.0f);
    assert(price[2] == 2.5f);
    assert(quantity[0] == 10);
    assert(quantity[1] == 20);
    assert(quantity[2] == 0);

    std::cout << "csv loader error test passed\n";
    return 0;
}
