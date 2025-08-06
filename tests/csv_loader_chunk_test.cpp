#include "csv_loader.hpp"
#include <cassert>
#include <sstream>
#include <iostream>

int main() {
    std::string data =
        "price,quantity\n"
        "10.5,3\n"
        "20.0,4\n"
        "price,quantity\n"
        "15.25,2\n"
        "30.0,5\n";
    std::istringstream ss(data);

    bool finished = false;
    HostTable first = load_csv_chunk(ss, 2, finished);
    assert(first.num_rows() == 2);
    assert(!finished);

    HostTable second = load_csv_chunk(ss, 2, finished);
    assert(second.num_rows() == 2);
    assert(finished);

    std::cout << "CSV chunk test passed\n";
    return 0;
}
