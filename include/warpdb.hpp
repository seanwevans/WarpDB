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

    // Execute a full SQL query supporting GROUP BY and ORDER BY.
    // Currently JOIN loads the same table for demonstration purposes.
    std::vector<float> query_sql(const std::string &sql);

private:
    Table table_;
    HostTable host_table_;
};
