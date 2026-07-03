#include "test_utils.hpp"
#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    if (warpdb_skip_if_no_cuda("csv_loader_unsupported_type_test")) return 0;
    HostTable table;
    HostColumn col;
    col.name = "bad";
    col.type = static_cast<DataType>(999);
    col.data = std::vector<int32_t>{1};
    table.columns.push_back(col);

    bool threw = false;
    try {
        upload_to_gpu(table);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "Expected upload_to_gpu to throw on unsupported DataType");
    std::cout << "csv loader unsupported type test passed\n";
    return 0;
}
