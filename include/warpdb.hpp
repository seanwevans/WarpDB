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

    // Execute a full SQL query supporting GROUP BY and ORDER BY.
    // Currently JOIN loads the same table for demonstration purposes.
    std::vector<float> query_sql(const std::string &sql);

    // Execute a query using all available GPUs on the data loaded in this
    // WarpDB instance. Falls back to single-GPU execution when only one device
    // is present.
    std::vector<float> query_multi_gpu(const std::string &expr);

    // Stream a CSV file in chunks and execute the same expression across all
    // GPUs. Useful when the dataset is larger than GPU memory. This static
    // helper does not require constructing a WarpDB instance.
    static std::vector<float> query_multi_gpu_csv(const std::string &csv_path,
                                                 const std::string &expr,
                                                 int rows_per_chunk = 1000000);

    // Execute a query and export the results as Arrow buffers.
    // The ArrowArray and ArrowSchema must be provided by the caller.
    // When use_shared_memory is true, the result buffer is created in
    // POSIX shared memory so other processes can access it.
    void query_arrow(const std::string &expr, ArrowArray *out_array,
                     ArrowSchema *out_schema, bool use_shared_memory = false);


private:
    Table table_;
    HostTable host_table_;
};
