#pragma once
#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "csv_loader.hpp"
#include "json_loader.hpp"
#include "expression.hpp"
#include "jit.hpp"
#include "arrow_utils.hpp"

class QueryResult {
public:
    QueryResult() : data_(std::vector<float>{}) {}
    explicit QueryResult(ColumnData data) : data_(std::move(data)) {}

    size_t size() const {
        return std::visit([](const auto &v) { return v.size(); }, data_);
    }

    bool empty() const { return size() == 0; }

    const ColumnData &data() const { return data_; }

    template <typename T>
    const std::vector<T> &as() const {
        return std::get<std::vector<T>>(data_);
    }

    float operator[](size_t idx) const {
        return std::visit(
            [idx](const auto &v) -> float {
                using VecT = std::decay_t<decltype(v)>;
                using ValueT = typename VecT::value_type;
                if constexpr (std::is_same_v<ValueT, std::string>) {
                    throw std::runtime_error(
                        "String results cannot be accessed via float operator[]");
                } else {
                    return static_cast<float>(v[idx]);
                }
            },
            data_);
    }

private:
    ColumnData data_;
};

class WarpDB {
public:
    explicit WarpDB(const std::string &filepath,
                    const std::vector<DataType> &schema = {},
                    ParsePolicy policy = ParsePolicy::Strict);
    ~WarpDB();

    // Execute an expression with optional WHERE clause.
    // Example: "price * quantity WHERE price > 10"
    QueryResult query(const std::string &expr);

    // Execute a full SQL query supporting WHERE/GROUP BY/HAVING/ORDER BY,
    // DISTINCT/LIMIT/OFFSET on a single loaded table.
    // JOIN clauses are parsed by the SQL frontend but are not executed yet.
    QueryResult query_sql(const std::string &sql);

    // Execute a query using all available GPUs on the data loaded in this
    // WarpDB instance. Falls back to single-GPU execution when only one device
    // is present.
    QueryResult query_multi_gpu(const std::string &expr);

    // Stream a CSV file in chunks and execute the same expression across all
    // GPUs. Useful when the dataset is larger than GPU memory. This static
    // helper does not require constructing a WarpDB instance.
    static QueryResult query_multi_gpu_csv(const std::string &csv_path,
                                           const std::string &expr,
                                           int rows_per_chunk = 1000000);

    // Execute a query and export the results as Arrow buffers.
    // The ArrowArray and ArrowSchema must be provided by the caller.
    // When use_shared_memory is true, the result buffer is created in
    // POSIX shared memory so other processes can access it.  The name of
    // the shared memory region can be customized with `shm_name`.
    void query_arrow(const std::string &expr, ArrowArray *out_array,
                     ArrowSchema *out_schema, bool use_shared_memory = false,
                     const char* shm_name = "/warpdb_result");


private:
    Table table_;
    HostTable host_table_;
};
