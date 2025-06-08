#pragma once
#include <string>
#include <vector>

#include "csv_loader.hpp"
#include "json_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"
#include "arrow_utils.hpp"

class WarpDB {
public:
    explicit WarpDB(const std::string &filepath);
    ~WarpDB();

    // Execute an expression with optional WHERE clause.
    // Example: "price * quantity WHERE price > 10"
    std::vector<float> query(const std::string &expr);

    // Execute a query and export the results as Arrow buffers.
    // The ArrowArray and ArrowSchema must be provided by the caller.
    // When use_shared_memory is true, the result buffer is created in
    // POSIX shared memory so other processes can access it.
    void query_arrow(const std::string &expr, ArrowArray *out_array,
                     ArrowSchema *out_schema, bool use_shared_memory = false);

private:
    Table table_;
};
