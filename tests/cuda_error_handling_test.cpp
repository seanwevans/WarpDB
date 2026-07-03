#include "test_utils.hpp"
#include "csv_loader.hpp"
#include "json_loader.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    if (warpdb_skip_if_no_cuda("cuda_error_handling_test")) return 0;
    // Hide any available GPUs so CUDA calls fail deterministically.
    setenv("CUDA_VISIBLE_DEVICES", "", 1);

    bool csv_threw = false;
    try {
        load_csv_to_gpu("data/test.csv");
    } catch (const std::runtime_error &e) {
        csv_threw = true;
        std::cout << "Caught CSV error: " << e.what() << "\n";
    }
    assert(csv_threw && "Expected load_csv_to_gpu to throw");

    bool json_threw = false;
    try {
        load_json_to_gpu("data/test.json");
    } catch (const std::runtime_error &e) {
        json_threw = true;
        std::cout << "Caught JSON error: " << e.what() << "\n";
    }
    assert(json_threw && "Expected load_json_to_gpu to throw");

    std::cout << "CUDA error handling test passed\n";
    return 0;
}
