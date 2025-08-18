#include "csv_loader.hpp"
#include <cassert>
#include <sstream>
#include <iostream>

int main() {
    std::string data =
        "price,quantity\n"
        "1.0,2.0\n"
        "foo,3.0\n"
        "1e39,4.0\n";
    std::istringstream ss(data);
    bool finished = false;
    bool threw = false;
    try {
        load_csv_chunk(ss, 10, finished);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "Expected strict parsing to throw");

    std::istringstream ss_perm(data);
    finished = false;
    HostTable table = load_csv_chunk(ss_perm, 10, finished, ParsePolicy::Permissive);
    const auto &price = std::get<std::vector<float>>(table.columns[0].data);
    const auto &quantity = std::get<std::vector<float>>(table.columns[1].data);
    assert(price.size() == 3 && quantity.size() == 3);
    assert(price[0] == 1.0f);
    assert(price[1] == 0.0f);
    assert(price[2] == 0.0f);
    assert(quantity[0] == 2.0f);
    assert(quantity[1] == 3.0f);
    assert(quantity[2] == 4.0f);

    std::cout << "csv loader chunk error test passed\n";
    return 0;
}
