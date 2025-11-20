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

std::vector<float> WarpDB::query(const std::string &expr) {
    if (expr.empty()) {
        throw std::runtime_error("Empty query expression");
    }

    std::string upper = expr;
    for (auto &c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string expr_part = expr;
    std::string where_part;
    auto where_pos = upper.find("WHERE");
    if (where_pos != std::string::npos) {
        expr_part = expr.substr(0, where_pos);
        where_part = expr.substr(where_pos + 5);
    }

    std::unique_ptr<ASTNode> expr_ast;
    try {
        auto expr_tokens = tokenize(expr_part);
        expr_ast = parse_expression(expr_tokens);
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
    if (!where_part.empty()) {
        try {
            auto cond_tokens = tokenize(where_part);
            auto cond_ast = parse_expression(cond_tokens);
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
    return result;
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

void apply_order_by(const QueryAST &ast, const HostTable &table,
                    const std::vector<int> &rows, std::vector<float> &result) {
    std::vector<std::pair<float,float>> keyed;
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

void apply_limit_offset(const QueryAST &ast, std::vector<float> &result) {
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

} // namespace

std::vector<float> WarpDB::query_sql(const std::string &sql) {
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

    std::vector<int> rows = filter_rows(ast, host_table_);

    std::vector<float> result;
    if (ast.group_by) {
        result = execute_group_by(ast, host_table_, rows);
    } else {
        for (int idx : rows) {
            result.push_back(eval_node(ast.select_list[0].get(), host_table_, idx));
        }
        if (ast.order_by) {
            apply_order_by(ast, host_table_, rows, result);
        }
    }

    if (ast.distinct) {
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
    }

    apply_limit_offset(ast, result);
    return result;
}



void WarpDB::query_arrow(const std::string &expr, ArrowArray *out_array,
                         ArrowSchema *out_schema, bool use_shared_memory,
                         const char* shm_name) {
    auto result = query(expr);
    export_to_arrow(result.data(), static_cast<int64_t>(result.size()),
                    use_shared_memory, out_array, out_schema, shm_name);

}

std::vector<float> WarpDB::query_multi_gpu(const std::string &expr) {
    if (host_table_.num_rows() == 0) {
        throw std::runtime_error("Host table not available for multi-GPU query");
    }

    std::string upper = expr;
    for (auto &c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string expr_part = expr;
    std::string where_part;
    auto where_pos = upper.find("WHERE");
    if (where_pos != std::string::npos) {
        expr_part = expr.substr(0, where_pos);
        where_part = expr.substr(where_pos + 5);
    }

    std::unique_ptr<ASTNode> expr_ast;
    auto expr_tokens = tokenize(expr_part);
    expr_ast = parse_expression(expr_tokens);

    std::unordered_set<std::string> cols;
    for (const auto &c : host_table_.columns) {
        cols.insert(c.name);
    }
    validate_ast(expr_ast.get(), cols);

    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        auto cond_ast = parse_expression(cond_tokens);
        validate_ast(cond_ast.get(), cols);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    return run_multi_gpu_jit_host(host_table_, expr_cuda, condition_cuda);
}

std::vector<float> WarpDB::query_multi_gpu_csv(const std::string &csv_path,
                                               const std::string &expr,
                                               int rows_per_chunk) {
    std::string upper = expr;
    for (auto &c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    std::string expr_part = expr;
    std::string where_part;
    auto where_pos = upper.find("WHERE");
    if (where_pos != std::string::npos) {
        expr_part = expr.substr(0, where_pos);
        where_part = expr.substr(where_pos + 5);
    }

    auto expr_tokens = tokenize(expr_part);
    auto expr_ast = parse_expression(expr_tokens);
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
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        auto cond_ast = parse_expression(cond_tokens);
        validate_ast(cond_ast.get(), cols);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    bool finished = false;
    std::vector<float> all_results;
    while (!finished) {
        HostTable chunk = load_csv_chunk(file, rows_per_chunk, finished, column_names);
        if (chunk.num_rows() == 0 && finished) {
            break;
        }
        auto part = run_multi_gpu_jit_host(chunk, expr_cuda, condition_cuda);
        all_results.insert(all_results.end(), part.begin(), part.end());
    }

    return all_results;
}
