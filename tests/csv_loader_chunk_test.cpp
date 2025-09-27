#include "csv_loader.hpp"
#include <cassert>
#include <sstream>
#include <iostream>
#include <vector>

int main() {
    std::string data =
        "price,quantity\n"
        " 10.5 , 3 \n"
        " 20.0, 4\n"
        " 15.25 , 2 \n"
        " 30.0 , 5 \n";
    std::istringstream ss(data);

    std::string header;
    std::getline(ss, header);
    std::stringstream header_ss(header);
    std::vector<std::string> columns;
    std::string name;
    while (std::getline(header_ss, name, ',')) columns.push_back(name);

    bool finished = false;
    HostTable first = load_csv_chunk(ss, 2, finished, columns);
    assert(first.num_rows() == 2);
    const auto &p1 = std::get<std::vector<float>>(first.columns[0].data);
    const auto &q1 = std::get<std::vector<float>>(first.columns[1].data);
    assert(p1[0] == 10.5f && p1[1] == 20.0f);
    assert(q1[0] == 3.0f && q1[1] == 4.0f);
    assert(!finished);

    HostTable second = load_csv_chunk(ss, 2, finished, columns);
    assert(second.num_rows() == 2);
    const auto &p2 = std::get<std::vector<float>>(second.columns[0].data);
    const auto &q2 = std::get<std::vector<float>>(second.columns[1].data);
    assert(p2[0] == 15.25f && p2[1] == 30.0f);
    assert(q2[0] == 2.0f && q2[1] == 5.0f);
    assert(finished);

    std::cout << "CSV chunk test passed\n";
    return 0;
}
