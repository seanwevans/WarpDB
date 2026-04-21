#include "warpdb.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <sstream>

#include <map>
#include <unordered_map>
#include <utility>
#include "arrow_loader.hpp"
#include "multi_gpu_utils.hpp"
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include "eval_helpers.hpp"


namespace {
// Recursively validate that all variable references exist in the table.
void validate_ast(const ASTNode *node,
                  const std::unordered_set<std::string> &cols) {
    if (!node) return;
    if (auto var = dynamic_cast<const VariableNode *>(node)) {
        if (cols.find(var->name) == cols.end()) {
            throw std::runtime_error("Unknown column: " + var->name);
        }
    } else if (auto bin = dynamic_cast<const BinaryOpNode *>(node)) {
        validate_ast(bin->left.get(), cols);
        validate_ast(bin->right.get(), cols);
    } else if (auto func = dynamic_cast<const FunctionCallNode *>(node)) {
        for (const auto &a : func->args) {
            validate_ast(a.get(), cols);
        }
    } else if (auto agg = dynamic_cast<const AggregationNode *>(node)) {
        validate_ast(agg->expr.get(), cols);
    } else if (auto win = dynamic_cast<const WindowFunctionNode *>(node)) {
        validate_ast(win->expr.get(), cols);
        for (const auto &p : win->partition_by) {
            validate_ast(p.get(), cols);
        }
        if (win->order_by) {
            validate_ast(win->order_by->expr.get(), cols);
        }
    }
}

} // namespace

WarpDB::WarpDB(const std::string &filepath, const std::vector<DataType> &schema,
               ParsePolicy policy) {
    auto dot = filepath.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : filepath.substr(dot + 1);
    for (auto &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "csv") {
        table_ = load_csv_to_gpu(filepath, schema, policy);
        host_table_ = load_csv_to_host(filepath, schema, policy);
    } else if (ext == "json") {
        table_ = load_json_to_gpu(filepath, policy);

        host_table_ = load_json_to_host(filepath, policy);

#ifdef USE_ARROW

    } else if (ext == "parquet") {
        table_ = load_parquet_to_gpu(filepath);
        host_table_ = load_parquet_to_host(filepath);
    } else if (ext == "arrow" || ext == "feather") {
        table_ = load_arrow_to_gpu(filepath);
        host_table_ = load_arrow_to_host(filepath);
    } else if (ext == "orc") {
        table_ = load_orc_to_gpu(filepath);
        host_table_ = load_orc_to_host(filepath);
#else
    } else if (ext == "parquet" || ext == "arrow" || ext == "feather" ||
               ext == "orc") {
        throw std::runtime_error(
            "Arrow support is not compiled into WarpDB");
#endif
    } else {
        throw std::runtime_error("Unsupported file format: " + filepath);
    }

}

WarpDB::~WarpDB() {
    for (auto &c : table_.columns) {
        c.device_ptr.reset();
    }
}

QueryResult WarpDB::query(const std::string &expr) {
    if (expr.empty()) {
        throw std::runtime_error("Empty query expression");
    }

    auto tokens = tokenize(expr);
    auto split = split_where_clause_tokens(tokens);

    std::unique_ptr<ASTNode> expr_ast;
    try {
        expr_ast = parse_expression(split.expression_tokens);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to parse expression: ") + e.what());
    }

    std::unordered_set<std::string> cols;
    for (const auto &c : table_.columns) {
        cols.insert(c.name);
    }
    validate_ast(expr_ast.get(), cols);

    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (split.has_where) {
        try {
            auto cond_ast = parse_expression(split.where_tokens);
            validate_ast(cond_ast.get(), cols);
            condition_cuda = cond_ast->to_cuda_expr();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Failed to parse WHERE clause: ") + e.what());
        }
    }

    DeviceBuffer<float> d_output(table_.num_rows);

    // block_size=0 lets jit_compile_and_launch pick an occupancy-optimised
    // value.
    jit_compile_and_launch(expr_cuda, condition_cuda, table_, d_output.get(), 0,
                           0);

    std::vector<float> result(table_.num_rows);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output.get(),
                         sizeof(float) * table_.num_rows,
                         cudaMemcpyDeviceToHost));
    return QueryResult(std::move(result));
}



// Helper utilities for query_sql
namespace {

void validate_query_ast(const QueryAST &ast, const std::unordered_set<std::string> &cols) {
    auto validate_ctx = [&](const ASTNode *node, const std::string &ctx) {
        if (!node) return;
        try {
            validate_ast(node, cols);
        } catch (const std::exception &e) {
            throw std::runtime_error(ctx + ": " + e.what());
        }
    };

    for (const auto &expr : ast.select_list) {
        validate_ctx(expr.get(), "SELECT clause");
    }
    for (const auto &j : ast.joins) {
        validate_ctx(j.condition.get(), "JOIN condition");
    }
    if (ast.where) {
        validate_ctx(ast.where.value().get(), "WHERE clause");
    }
    if (ast.group_by) {
        for (const auto &k : ast.group_by->keys) {
            validate_ctx(k.get(), "GROUP BY");
        }
    }
    if (ast.order_by) {
        validate_ctx(ast.order_by->expr.get(), "ORDER BY");
    }
}

std::vector<int> filter_rows(const QueryAST &ast, const HostTable &table) {
    std::vector<int> rows;
    int N = table.num_rows();
    rows.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (ast.where) {
            if (!eval_condition(ast.where.value().get(), table, i)) continue;
        }
        rows.push_back(i);
    }
    return rows;
}

struct AggData {
    double sum = 0.0;
    double count = 0.0;
    double min = 0.0;
    double max = 0.0;
    bool init = false;
};

float eval_having_node(const ASTNode *node, const AggData &gd) {
    if (auto c = dynamic_cast<const ConstantNode *>(node)) {
        return std::stof(c->value);
    }
    if (auto b = dynamic_cast<const BinaryOpNode *>(node)) {
        float l = eval_having_node(b->left.get(), gd);
        float r = eval_having_node(b->right.get(), gd);
        const std::string &op = b->op;
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return l / r;
        if (op == ">") return l > r;
        if (op == "<") return l < r;
        if (op == ">=") return l >= r;
        if (op == "<=") return l <= r;
        if (op == "==") return l == r;
        if (op == "=") return l == r;
        if (op == "!=") return l != r;
    }
    if (auto ag = dynamic_cast<const AggregationNode *>(node)) {
        switch (ag->agg) {
        case AggregationType::Sum: return gd.sum;
        case AggregationType::Avg: return gd.sum / gd.count;
        case AggregationType::Count: return gd.count;
        case AggregationType::Min: return gd.min;
        case AggregationType::Max: return gd.max;
        }
    }
    return 0.0f;
}

bool eval_having(const QueryAST &ast, const AggData &gd) {
    if (!ast.having) return true;
    return eval_having_node(ast.having.value().get(), gd) != 0.0f;
}

void collect_variable_names(const ASTNode *node,
                            std::unordered_set<std::string> &names) {
    if (!node) return;
    if (auto var = dynamic_cast<const VariableNode *>(node)) {
        names.insert(var->name);
        return;
    }
    if (auto bin = dynamic_cast<const BinaryOpNode *>(node)) {
        collect_variable_names(bin->left.get(), names);
        collect_variable_names(bin->right.get(), names);
        return;
    }
    if (auto fn = dynamic_cast<const FunctionCallNode *>(node)) {
        for (const auto &arg : fn->args) {
            collect_variable_names(arg.get(), names);
        }
        return;
    }
    if (auto agg = dynamic_cast<const AggregationNode *>(node)) {
        collect_variable_names(agg->expr.get(), names);
    }
}

bool can_use_gpu_group_sum_fast_path(const QueryAST &ast) {
    if (!ast.group_by || ast.group_by->keys.size() != 1) return false;
    if (ast.select_list.size() != 1) return false;
    if (ast.where || ast.having || !ast.joins.empty()) return false;

    auto *agg = dynamic_cast<const AggregationNode *>(ast.select_list[0].get());
    if (!agg || agg->agg != AggregationType::Sum) return false;

    auto *key_var = dynamic_cast<const VariableNode *>(ast.group_by->keys[0].get());
    if (!key_var || key_var->name != "quantity") return false;

    std::unordered_set<std::string> vars;
    collect_variable_names(agg->expr.get(), vars);
    for (const auto &name : vars) {
        if (name != "price" && name != "quantity") {
            return false;
        }
    }
    return true;
}

std::vector<float> execute_group_by_gpu_sum(const QueryAST &ast, const Table &table) {
    const int N = table.num_rows;
    float *d_price = table.get_column_ptr<float>("price");
    int *d_quantity = table.get_column_ptr<int>("quantity");
    if (!d_price || !d_quantity) {
        throw std::runtime_error(
            "GPU GROUP BY SUM fast path requires float 'price' and int 'quantity' columns");
    }

    DeviceBuffer<float> d_out_vals(static_cast<size_t>(N));
    DeviceBuffer<int> d_out_keys(static_cast<size_t>(N));
    DeviceBuffer<int> d_count(1);

    const auto *agg = dynamic_cast<const AggregationNode *>(ast.select_list[0].get());
    jit_group_sum(agg->expr->to_cuda_expr(), ast.group_by->keys[0]->to_cuda_expr(),
                  d_price, d_quantity, d_out_vals.get(), d_out_keys.get(),
                  d_count.get(), N);

    int host_count = 0;
    CUDA_CHECK(cudaMemcpy(&host_count, d_count.get(), sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (host_count <= 0) return {};

    std::vector<int> keys(static_cast<size_t>(host_count));
    std::vector<float> vals(static_cast<size_t>(host_count));
    CUDA_CHECK(cudaMemcpy(keys.data(), d_out_keys.get(),
                          sizeof(int) * static_cast<size_t>(host_count),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vals.data(), d_out_vals.get(),
                          sizeof(float) * static_cast<size_t>(host_count),
                          cudaMemcpyDeviceToHost));

    std::vector<std::pair<int, float>> keyed;
    keyed.reserve(static_cast<size_t>(host_count));
    for (int i = 0; i < host_count; ++i) {
        keyed.push_back({keys[static_cast<size_t>(i)], vals[static_cast<size_t>(i)]});
    }

    if (ast.order_by && !ast.order_by->ascending) {
        std::reverse(keyed.begin(), keyed.end());
    }

    std::vector<float> result;
    result.reserve(keyed.size());
    for (const auto &kv : keyed) {
        result.push_back(kv.second);
    }
    return result;
}

std::vector<float> execute_group_by(const QueryAST &ast,
                                    const HostTable &table,
                                    const std::vector<int> &rows) {
    auto *agg = dynamic_cast<AggregationNode *>(ast.select_list[0].get());
    if (!agg) throw std::runtime_error("Only aggregation queries supported with GROUP BY");

    std::unordered_map<double, AggData> groups;
    for (int idx : rows) {
        double key = static_cast<double>(
            eval_node(ast.group_by->keys[0].get(), table, idx));
        float val = 1.0f;
        if (agg->agg != AggregationType::Count) {
            val = eval_node(agg->expr.get(), table, idx);
        }
        auto &g = groups[key];
        if (!g.init) { g.min = g.max = val; g.init = true; }
        g.sum += val;
        g.count += 1.0;
        g.min = std::min(g.min, static_cast<double>(val));
        g.max = std::max(g.max, static_cast<double>(val));
    }

    std::vector<std::pair<double,float>> keyed;
    for (const auto &kv : groups) {
        const AggData &g = kv.second;
        if (!eval_having(ast, g)) continue;

        float out = 0.0f;
        switch (agg->agg) {
        case AggregationType::Sum: out = g.sum; break;
        case AggregationType::Avg: out = g.sum / g.count; break;
        case AggregationType::Count: out = g.count; break;
        case AggregationType::Min: out = g.min; break;
        case AggregationType::Max: out = g.max; break;
        }
        keyed.push_back({kv.first, out});
    }

    if (ast.order_by) {
        std::sort(keyed.begin(), keyed.end(), [&](auto &a, auto &b) {
            if (ast.order_by->ascending) return a.first < b.first;
            return a.first > b.first;
        });
    }

    std::vector<float> result;
    result.reserve(keyed.size());
    for (auto &p : keyed) result.push_back(p.second);
    return result;
}

template <typename T>
void apply_order_by_typed(const QueryAST &ast, const HostTable &table,
                          const std::vector<int> &rows, std::vector<T> &result) {
    std::vector<std::pair<float, T>> keyed;
    for (size_t i = 0; i < rows.size(); ++i) {
        float key = eval_node(ast.order_by->expr.get(), table, rows[i]);
        keyed.push_back({key, result[i]});
    }
    std::sort(keyed.begin(), keyed.end(), [&](auto &a, auto &b) {
        if (ast.order_by->ascending) return a.first < b.first;
        return a.first > b.first;
    });
    for (size_t i = 0; i < keyed.size(); ++i) {
        result[i] = keyed[i].second;
    }
}

template <typename T>
void apply_limit_offset_typed(const QueryAST &ast, std::vector<T> &result) {
    if (ast.offset) {
        size_t off = static_cast<size_t>(ast.offset->count);
        if (off >= result.size()) {
            result.clear();
            return;
        }
        result.erase(result.begin(), result.begin() + off);
    }
    if (ast.limit && static_cast<size_t>(ast.limit->count) < result.size()) {
        result.resize(ast.limit->count);
    }
}

template <typename T>
void apply_distinct_typed(std::vector<T> &result) {
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
}

ColumnData select_typed_column(const HostColumn &col, const std::vector<int> &rows) {
    switch (col.type) {
    case DataType::Int32: {
        const auto &src = std::get<std::vector<int32_t>>(col.data);
        std::vector<int32_t> out;
        out.reserve(rows.size());
        for (int idx : rows) out.push_back(src[idx]);
        return out;
    }
    case DataType::Int64: {
        const auto &src = std::get<std::vector<int64_t>>(col.data);
        std::vector<int64_t> out;
        out.reserve(rows.size());
        for (int idx : rows) out.push_back(src[idx]);
        return out;
    }
    case DataType::Float32: {
        const auto &src = std::get<std::vector<float>>(col.data);
        std::vector<float> out;
        out.reserve(rows.size());
        for (int idx : rows) out.push_back(src[idx]);
        return out;
    }
    case DataType::Float64: {
        const auto &src = std::get<std::vector<double>>(col.data);
        std::vector<double> out;
        out.reserve(rows.size());
        for (int idx : rows) out.push_back(src[idx]);
        return out;
    }
    case DataType::String: {
        const auto &src = std::get<std::vector<std::string>>(col.data);
        std::vector<std::string> out;
        out.reserve(rows.size());
        for (int idx : rows) out.push_back(src[idx]);
        return out;
    }
    }
    return std::vector<float>{};
}

void apply_order_by(const QueryAST &ast, const HostTable &table,
                    const std::vector<int> &rows, ColumnData &result) {
    std::visit(
        [&](auto &vec) {
            apply_order_by_typed(ast, table, rows, vec);
        },
        result);
}

void apply_limit_offset(const QueryAST &ast, ColumnData &result) {
    std::visit(
        [&](auto &vec) {
            apply_limit_offset_typed(ast, vec);
        },
        result);
}

void apply_distinct(ColumnData &result) {
    std::visit(
        [&](auto &vec) {
            apply_distinct_typed(vec);
        },
        result);
}

} // namespace

QueryResult WarpDB::query_sql(const std::string &sql) {
    auto tokens = tokenize(sql);
    QueryAST ast;
    try {
        ast = parse_query(tokens);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to parse SQL: ") + e.what());
    }

    std::unordered_set<std::string> cols;
    for (const auto &c : table_.columns) cols.insert(c.name);
    validate_query_ast(ast, cols);

    if (!ast.group_by && ast.select_list.size() > 1) {
        throw std::runtime_error(
            "Multiple SELECT expressions are not yet supported in query_sql");
    }

    if (!ast.joins.empty()) {
        throw std::runtime_error(
            "JOIN is parsed but not executed yet. SQL execution currently "
            "supports single-table queries only.");
    }

    std::vector<int> rows = filter_rows(ast, host_table_);

    ColumnData result = std::vector<float>{};
    if (ast.group_by) {
        if (can_use_gpu_group_sum_fast_path(ast)) {
            result = execute_group_by_gpu_sum(ast, table_);
        } else {
            result = execute_group_by(ast, host_table_, rows);
        }
    } else {
        if (auto *var =
                dynamic_cast<VariableNode *>(ast.select_list[0].get())) {
            const HostColumn *col = host_table_.get_column(var->name);
            if (!col) {
                throw std::runtime_error("Unknown column in SELECT: " + var->name);
            }
            result = select_typed_column(*col, rows);
        } else {
            auto &out = std::get<std::vector<float>>(result);
            for (int idx : rows) {
                out.push_back(eval_node(ast.select_list[0].get(), host_table_, idx));
            }
        }
        if (ast.order_by) {
            apply_order_by(ast, host_table_, rows, result);
        }
    }

    if (ast.distinct) {
        apply_distinct(result);
    }

    apply_limit_offset(ast, result);
    return QueryResult(std::move(result));
}



void WarpDB::query_arrow(const std::string &expr, ArrowArray *out_array,
                         ArrowSchema *out_schema, bool use_shared_memory,
                         const char* shm_name) {
    QueryResult qr = query(expr);
    const auto &result = qr.as<float>();
    export_to_arrow(result.data(), static_cast<int64_t>(result.size()),
                    use_shared_memory, out_array, out_schema, shm_name);

}

QueryResult WarpDB::query_multi_gpu(const std::string &expr) {
    if (host_table_.num_rows() == 0) {
        throw std::runtime_error("Host table not available for multi-GPU query");
    }

    auto tokens = tokenize(expr);
    auto split = split_where_clause_tokens(tokens);

    std::unique_ptr<ASTNode> expr_ast;
    expr_ast = parse_expression(split.expression_tokens);

    std::unordered_set<std::string> cols;
    for (const auto &c : host_table_.columns) {
        cols.insert(c.name);
    }
    validate_ast(expr_ast.get(), cols);

    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (split.has_where) {
        auto cond_ast = parse_expression(split.where_tokens);
        validate_ast(cond_ast.get(), cols);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    return QueryResult(run_multi_gpu_jit_host(host_table_, expr_cuda, condition_cuda));
}

QueryResult WarpDB::query_multi_gpu_csv(const std::string &csv_path,
                                        const std::string &expr,
                                        int rows_per_chunk) {
    auto tokens = tokenize(expr);
    auto split = split_where_clause_tokens(tokens);
    auto expr_ast = parse_expression(split.expression_tokens);
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + csv_path);
    }

    std::string header;
    if (!std::getline(file, header)) {
        throw std::runtime_error("Failed to read CSV header from: " + csv_path);
    }
    std::stringstream header_ss(header);
    std::vector<std::string> column_names;
    std::string col_name;
    while (std::getline(header_ss, col_name, ',')) {
        column_names.push_back(col_name);
    }

    if (column_names.empty()) {
        throw std::runtime_error("CSV file has no columns: " + csv_path);
    }

    std::unordered_set<std::string> cols(column_names.begin(), column_names.end());
    validate_ast(expr_ast.get(), cols);
    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (split.has_where) {
        auto cond_ast = parse_expression(split.where_tokens);
        validate_ast(cond_ast.get(), cols);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    bool finished = false;
    std::vector<DataType> schema;
    std::vector<float> all_results;
    while (!finished) {
        HostTable chunk =
            load_csv_chunk(file, rows_per_chunk, finished, column_names,
                           ParsePolicy::Strict, &schema);
        if (chunk.num_rows() == 0 && finished) {
            break;
        }
        auto part = run_multi_gpu_jit_host(chunk, expr_cuda, condition_cuda);
        all_results.insert(all_results.end(), part.begin(), part.end());
    }

    return QueryResult(std::move(all_results));
}
