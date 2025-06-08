#pragma once
#include <string>
#include <vector>

#include "csv_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"

class WarpDB {
public:
    explicit WarpDB(const std::string &csv_path);
    ~WarpDB();

    // Execute an expression with optional WHERE clause.
    // Example: "price * quantity WHERE price > 10"
    std::vector<float> query(const std::string &expr);

private:
    Table table_;
};
