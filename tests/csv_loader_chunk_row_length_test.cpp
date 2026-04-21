#include "csv_loader.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
    const std::string data =
        "price,quantity\n"
        "1.5,10\n"
        "2.0,20,999\n"
        "4.0,40\n";

    std::istringstream ss_strict(data);
    std::string header;
    std::getline(ss_strict, header);
    std::stringstream header_ss(header);
    std::vector<std::string> columns;
    std::string name;
    while (std::getline(header_ss, name, ',')) columns.push_back(name);

    bool finished = false;
    bool threw = false;
    try {
        load_csv_chunk(ss_strict, 10, finished, columns, ParsePolicy::Strict);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "Expected strict chunk parsing to throw on row width mismatch");

    std::istringstream ss_perm(data);
    std::getline(ss_perm, header);
    std::stringstream header_ss_perm(header);
    std::vector<std::string> columns_perm;
    while (std::getline(header_ss_perm, name, ',')) columns_perm.push_back(name);

    finished = false;
    HostTable table =
        load_csv_chunk(ss_perm, 10, finished, columns_perm, ParsePolicy::Permissive);
    const auto &price = std::get<std::vector<float>>(table.columns[0].data);
    const auto &quantity = std::get<std::vector<int32_t>>(table.columns[1].data);
    assert(price.size() == 2 && quantity.size() == 2);
    assert(price[0] == 1.5f && quantity[0] == 10);
    assert(price[1] == 4.0f && quantity[1] == 40);
    assert(finished);

    std::cout << "csv loader chunk row length test passed\n";
    return 0;
}
