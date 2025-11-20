#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
    std::string data =
        "item,price,quantity\n"
        "apple,1.5,10\n"
        "banana,2.75,20\n"
        "carrot,3,7\n"
        "date,4.1,12\n";

    std::istringstream ss(data);
    std::string header;
    std::getline(ss, header);
    std::stringstream header_ss(header);
    std::vector<std::string> columns;
    std::string name;
    while (std::getline(header_ss, name, ',')) columns.push_back(name);

    bool finished = false;
    std::vector<DataType> schema;
    HostTable first = load_csv_chunk(ss, 2, finished, columns, ParsePolicy::Strict, &schema);

    assert(schema.size() == 3);
    assert(schema[0] == DataType::String);
    assert(schema[1] == DataType::Float32);
    assert(schema[2] == DataType::Int32);

    const auto &items = std::get<std::vector<std::string>>(first.columns[0].data);
    const auto &prices = std::get<std::vector<float>>(first.columns[1].data);
    const auto &quantities = std::get<std::vector<int32_t>>(first.columns[2].data);
    assert(items.size() == 2 && items[0] == "apple" && items[1] == "banana");
    assert(prices[0] == 1.5f && prices[1] == 2.75f);
    assert(quantities[0] == 10 && quantities[1] == 20);
    assert(!finished);

    HostTable second = load_csv_chunk(ss, 2, finished, columns, ParsePolicy::Strict, &schema);
    assert(second.columns[0].type == DataType::String);
    assert(second.columns[1].type == DataType::Float32);
    assert(second.columns[2].type == DataType::Int32);

    const auto &items2 = std::get<std::vector<std::string>>(second.columns[0].data);
    const auto &prices2 = std::get<std::vector<float>>(second.columns[1].data);
    const auto &quantities2 = std::get<std::vector<int32_t>>(second.columns[2].data);
    assert(items2.size() == 2 && items2[0] == "carrot" && items2[1] == "date");
    assert(prices2[0] == 3.0f && prices2[1] == 4.1f);
    assert(quantities2[0] == 7 && quantities2[1] == 12);
    assert(finished);

    std::cout << "CSV chunk mixed types test passed\n";
    return 0;
}
