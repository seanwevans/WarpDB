#include "csv_loader.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    std::vector<DataType> schema = {DataType::Int32, DataType::Int64,
                                    DataType::Float32, DataType::Float64};

    // Valid numbers including boundaries
    HostTable valid = load_csv_to_host("data/numeric_valid.csv", schema);
    const auto &i32 = std::get<std::vector<int32_t>>(valid.columns[0].data);
    const auto &i64 = std::get<std::vector<int64_t>>(valid.columns[1].data);
    const auto &f32 = std::get<std::vector<float>>(valid.columns[2].data);
    const auto &f64 = std::get<std::vector<double>>(valid.columns[3].data);
    assert(i32[0] == 42);
    assert(i32[1] == std::numeric_limits<int32_t>::max());
    assert(i64[0] == 43);
    assert(i64[1] == std::numeric_limits<int64_t>::max());
    assert(f32[0] == 1.0f);
    assert(f32[1] == std::numeric_limits<float>::max());
    assert(f64[0] == 2.0);
    assert(f64[1] == std::numeric_limits<double>::max());

    // Invalid strings should throw in strict mode
    bool threw_invalid = false;
    try {
        load_csv_to_host("data/numeric_invalid.csv", schema);
    } catch (const std::exception &) {
        threw_invalid = true;
    }
    assert(threw_invalid);

    // Invalid strings should default to zero in permissive mode
    HostTable invalid = load_csv_to_host("data/numeric_invalid.csv", schema,
                                        ParsePolicy::Permissive);
    const auto &i32_inv = std::get<std::vector<int32_t>>(invalid.columns[0].data);
    const auto &i64_inv = std::get<std::vector<int64_t>>(invalid.columns[1].data);
    const auto &f32_inv = std::get<std::vector<float>>(invalid.columns[2].data);
    const auto &f64_inv = std::get<std::vector<double>>(invalid.columns[3].data);
    assert(i32_inv[0] == 0);
    assert(i64_inv[0] == 0);
    assert(f32_inv[0] == 0.0f);
    assert(f64_inv[0] == 0.0);

    // Out-of-range numbers should throw in strict mode
    bool threw_oor = false;
    try {
        load_csv_to_host("data/numeric_out_of_range.csv", schema);
    } catch (const std::exception &) {
        threw_oor = true;
    }
    assert(threw_oor);

    // Out-of-range numbers should default to zero in permissive mode
    HostTable oor = load_csv_to_host("data/numeric_out_of_range.csv", schema,
                                     ParsePolicy::Permissive);
    const auto &i32_oor = std::get<std::vector<int32_t>>(oor.columns[0].data);
    const auto &i64_oor = std::get<std::vector<int64_t>>(oor.columns[1].data);
    const auto &f32_oor = std::get<std::vector<float>>(oor.columns[2].data);
    const auto &f64_oor = std::get<std::vector<double>>(oor.columns[3].data);
    assert(i32_oor[0] == 0);
    assert(i64_oor[0] == 0);
    assert(f32_oor[0] == 0.0f);
    assert(f64_oor[0] == 0.0);

    std::cout << "from_chars numeric tests passed\n";
    return 0;
}
