#include "warpdb.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <sstream>
#include <variant>

#include <map>
#include <unordered_map>
#include <utility>
#include "arrow_loader.hpp"
#include "multi_gpu_utils.hpp"
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include "eval_helpers.hpp"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


namespace {
// Recursively validate that all variable references exist in the table.
void validate_ast(const ASTNode *node,
                  const std::unordered_set<std::string> &cols) {
    if (!node) return;
    if (auto var = dynamic_cast<const VariableNode *>(node)) {
        std::string lookup = var->name;
        const auto dot = lookup.find('.');
        if (dot != std::string::npos && dot + 1 < lookup.size()) {
            lookup = lookup.substr(dot + 1);
        }
        if (cols.find(lookup) == cols.end()) {
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

enum class PlanExecutionKind {
    GroupByGpuSum,
    GroupByHost,
    SelectColumn,
    SelectExpression
};

struct QueryPlan {
    QueryAST ast;
    PlanExecutionKind execution_kind = PlanExecutionKind::SelectExpression;
};

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
        if (op == "&&") return (l != 0.0f) && (r != 0.0f);
        if (op == "||") return (l != 0.0f) || (r != 0.0f);
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

const ColumnDesc *get_device_column(const Table &table, const std::string &name) {
    for (const auto &col : table.columns) {
        if (col.name == name) return &col;
    }
    return nullptr;
}

bool is_numeric_type(DataType type) {
    return type == DataType::Int32 || type == DataType::Int64 ||
           type == DataType::Float32 || type == DataType::Float64;
}

bool can_use_gpu_group_fast_path(const QueryAST &ast, const Table &table) {
    if (!ast.group_by || ast.group_by->keys.size() != 1) return false;
    if (ast.select_list.size() != 1) return false;
    if (ast.where || ast.having || !ast.joins.empty()) return false;

    auto *agg = dynamic_cast<const AggregationNode *>(ast.select_list[0].get());
    if (!agg) return false;
    if (agg->agg == AggregationType::Count) return false;

    auto *key_var = dynamic_cast<const VariableNode *>(ast.group_by->keys[0].get());
    auto *val_var = dynamic_cast<const VariableNode *>(agg->expr.get());
    if (!key_var || !val_var) return false;

    const ColumnDesc *key_col = get_device_column(table, key_var->name);
    const ColumnDesc *val_col = get_device_column(table, val_var->name);
    if (!key_col || !val_col) return false;
    if (!(key_col->type == DataType::Int32 || key_col->type == DataType::Int64)) return false;
    return is_numeric_type(val_col->type);
}

std::vector<float> execute_group_by_gpu_fast(const QueryAST &ast, const Table &table) {
    const int N = table.num_rows;
    const auto *agg = dynamic_cast<const AggregationNode *>(ast.select_list[0].get());
    const auto *key_var = dynamic_cast<const VariableNode *>(ast.group_by->keys[0].get());
    const auto *val_var = dynamic_cast<const VariableNode *>(agg->expr.get());
    const ColumnDesc *key_col = get_device_column(table, key_var->name);
    const ColumnDesc *val_col = get_device_column(table, val_var->name);
    if (!key_col || !val_col) {
        throw std::runtime_error("GPU GROUP BY fast path failed to resolve key/value columns");
    }

    DeviceBuffer<float> d_out_vals(static_cast<size_t>(N));
    DeviceBuffer<int64_t> d_out_keys(static_cast<size_t>(N));

    GroupReduceOp op = GroupReduceOp::Sum;
    if (agg->agg == AggregationType::Min) op = GroupReduceOp::Min;
    if (agg->agg == AggregationType::Max) op = GroupReduceOp::Max;

    int host_count = 0;
    if (agg->agg == AggregationType::Avg) {
        host_count = jit_group_reduce_by_key(
            key_col->device_ptr.get(), key_col->type, val_col->device_ptr.get(),
            val_col->type, GroupReduceOp::Sum, d_out_keys.get(), d_out_vals.get(), N);
        if (host_count > 0) {
            DeviceBuffer<float> d_counts(static_cast<size_t>(N));
            DeviceBuffer<int64_t> d_count_keys(static_cast<size_t>(N));
            int count_group_count = jit_group_count_by_key(
                key_col->device_ptr.get(), key_col->type, d_count_keys.get(),
                d_counts.get(), N);
            if (count_group_count != host_count) {
                throw std::runtime_error("GPU GROUP BY AVG fast path produced mismatched group counts");
            }
            thrust::transform(thrust::device, d_out_vals.get(), d_out_vals.get() + host_count,
                              d_counts.get(), d_out_vals.get(), thrust::divides<float>());
        }
    } else {
        host_count = jit_group_reduce_by_key(
            key_col->device_ptr.get(), key_col->type, val_col->device_ptr.get(),
            val_col->type, op, d_out_keys.get(), d_out_vals.get(), N);
    }

    if (host_count <= 0) return {};

    std::vector<int64_t> keys(static_cast<size_t>(host_count));
    std::vector<float> vals(static_cast<size_t>(host_count));
    CUDA_CHECK(cudaMemcpy(keys.data(), d_out_keys.get(),
                          sizeof(int64_t) * static_cast<size_t>(host_count),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vals.data(), d_out_vals.get(),
                          sizeof(float) * static_cast<size_t>(host_count),
                          cudaMemcpyDeviceToHost));

    std::vector<std::pair<int64_t, float>> keyed;
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
    if (!ast.group_by || ast.group_by->keys.size() != 1) {
        throw std::runtime_error(
            "GROUP BY in query_sql currently supports exactly one key expression");
    }

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

struct JoinVariableRef {
    std::string relation;
    std::string column;
};

std::optional<JoinVariableRef> parse_join_variable_name(const std::string &name) {
    const auto dot = name.find('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= name.size()) {
        return std::nullopt;
    }
    return JoinVariableRef{name.substr(0, dot), name.substr(dot + 1)};
}

using JoinKey = std::variant<int32_t, int64_t, float, double, std::string>;

struct JoinKeyHash {
    size_t operator()(const JoinKey &k) const {
        return std::visit(
            [](const auto &v) -> size_t {
                using T = std::decay_t<decltype(v)>;
                return std::hash<T>{}(v);
            },
            k);
    }
};

bool make_join_key(const HostColumn &col, int idx, JoinKey &out_key) {
    switch (col.type) {
    case DataType::Int32:
        out_key = std::get<std::vector<int32_t>>(col.data)[idx];
        return true;
    case DataType::Int64:
        out_key = std::get<std::vector<int64_t>>(col.data)[idx];
        return true;
    case DataType::Float32:
        out_key = std::get<std::vector<float>>(col.data)[idx];
        return true;
    case DataType::Float64:
        out_key = std::get<std::vector<double>>(col.data)[idx];
        return true;
    case DataType::String:
        out_key = std::get<std::vector<std::string>>(col.data)[idx];
        return true;
    }
    return false;
}

const HostColumn *resolve_join_column(const HostTable &table, const std::string &relation,
                                      const std::string &from_table,
                                      const std::string &join_table,
                                      const std::string &column_name) {
    if (relation != from_table && relation != join_table) {
        throw std::runtime_error("Unknown relation in JOIN expression: " + relation);
    }
    const HostColumn *col = table.get_column(column_name);
    if (!col) {
        throw std::runtime_error("Unknown column in JOIN expression: " + column_name);
    }
    return col;
}

std::pair<std::vector<int>, std::vector<int>>
build_inner_join_rows(const QueryAST &ast, const HostTable &table) {
    if (ast.joins.size() != 1) {
        throw std::runtime_error(
            "JOIN execution currently supports exactly one JOIN clause");
    }

    const JoinClause &join = ast.joins[0];
    auto *cmp = dynamic_cast<const BinaryOpNode *>(join.condition.get());
    if (!cmp || (cmp->op != "=" && cmp->op != "==")) {
        throw std::runtime_error(
            "JOIN execution currently supports equi-join predicates only");
    }

    auto *left_var = dynamic_cast<const VariableNode *>(cmp->left.get());
    auto *right_var = dynamic_cast<const VariableNode *>(cmp->right.get());
    if (!left_var || !right_var) {
        throw std::runtime_error(
            "JOIN ON clause must compare two columns (e.g. a.id = b.id)");
    }

    auto left_ref = parse_join_variable_name(left_var->name);
    auto right_ref = parse_join_variable_name(right_var->name);
    if (!left_ref || !right_ref) {
        throw std::runtime_error(
            "JOIN ON clause must use qualified columns (e.g. a.id = b.id)");
    }

    const HostColumn *left_col =
        resolve_join_column(table, left_ref->relation, ast.from_table, join.table,
                            left_ref->column);
    const HostColumn *right_col =
        resolve_join_column(table, right_ref->relation, ast.from_table, join.table,
                            right_ref->column);

    if (left_col->type != right_col->type) {
        throw std::runtime_error("JOIN key type mismatch between " + left_var->name +
                                 " and " + right_var->name);
    }

    std::unordered_map<JoinKey, std::vector<int>, JoinKeyHash> right_index;
    right_index.reserve(static_cast<size_t>(table.num_rows()));
    for (int r = 0; r < table.num_rows(); ++r) {
        JoinKey key;
        if (!make_join_key(*right_col, r, key)) continue;
        right_index[key].push_back(r);
    }

    std::vector<int> left_rows;
    std::vector<int> right_rows;
    for (int l = 0; l < table.num_rows(); ++l) {
        JoinKey key;
        if (!make_join_key(*left_col, l, key)) continue;
        auto it = right_index.find(key);
        if (it == right_index.end()) continue;
        for (int r : it->second) {
            left_rows.push_back(l);
            right_rows.push_back(r);
        }
    }
    return {std::move(left_rows), std::move(right_rows)};
}

float eval_join_node(const ASTNode *node, const HostTable &table, int left_idx,
                     int right_idx, const std::string &from_table,
                     const std::string &join_table) {
    if (auto c = dynamic_cast<const ConstantNode *>(node)) {
        return std::stof(c->value);
    }
    if (auto v = dynamic_cast<const VariableNode *>(node)) {
        auto qualified = parse_join_variable_name(v->name);
        if (qualified) {
            const HostColumn *col = resolve_join_column(
                table, qualified->relation, from_table, join_table, qualified->column);
            int idx = qualified->relation == from_table ? left_idx : right_idx;
            return get_value(table, col->name, idx);
        }
        const HostColumn *col = table.get_column(v->name);
        if (!col) {
            throw std::runtime_error("Unknown column: " + v->name);
        }
        return get_value(table, col->name, left_idx);
    }
    if (auto b = dynamic_cast<const BinaryOpNode *>(node)) {
        float l = eval_join_node(b->left.get(), table, left_idx, right_idx, from_table,
                                 join_table);
        float r = eval_join_node(b->right.get(), table, left_idx, right_idx, from_table,
                                 join_table);
        const std::string &op = b->op;
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") return l / r;
        if (op == "&&") return (l != 0.0f) && (r != 0.0f);
        if (op == "||") return (l != 0.0f) || (r != 0.0f);
        if (op == ">") return l > r;
        if (op == "<") return l < r;
        if (op == ">=") return l >= r;
        if (op == "<=") return l <= r;
        if (op == "==") return l == r;
        if (op == "=") return l == r;
        if (op == "!=") return l != r;
    }
    return 0.0f;
}

} // namespace

QueryPlan plan_query(QueryAST &&ast, const std::unordered_set<std::string> &cols, const Table &table) {
    validate_query_ast(ast, cols);

    if (ast.select_list.size() != 1) {
        throw std::runtime_error(
            "query_sql currently supports a single SELECT expression only");
    }
    if (ast.group_by && ast.group_by->keys.size() != 1) {
        throw std::runtime_error(
            "GROUP BY in query_sql currently supports exactly one key expression");
    }
    if (!ast.joins.empty()) {
        if (ast.joins.size() != 1) {
            throw std::runtime_error(
                "JOIN execution currently supports exactly one JOIN clause");
        }
        if (ast.group_by || ast.having || ast.order_by) {
            throw std::runtime_error(
                "JOIN execution currently supports SELECT/WHERE/DISTINCT/LIMIT/OFFSET "
                "without GROUP BY, HAVING, or ORDER BY");
        }
    }

    QueryPlan plan;
    plan.ast = std::move(ast);

    if (plan.ast.group_by) {
        if (can_use_gpu_group_fast_path(plan.ast, table)) {
            plan.execution_kind = PlanExecutionKind::GroupByGpuSum;
        } else {
            plan.execution_kind = PlanExecutionKind::GroupByHost;
        }
        return plan;
    }

    if (dynamic_cast<VariableNode *>(plan.ast.select_list[0].get())) {
        plan.execution_kind = PlanExecutionKind::SelectColumn;
    } else {
        plan.execution_kind = PlanExecutionKind::SelectExpression;
    }
    return plan;
}

QueryResult execute_plan(const QueryPlan &plan, const Table &table,
                         const HostTable &host_table) {
    std::vector<int> rows = filter_rows(plan.ast, host_table);

    ColumnData result = std::vector<float>{};
    if (plan.ast.group_by) {
        if (plan.execution_kind == PlanExecutionKind::GroupByGpuSum) {
            result = execute_group_by_gpu_fast(plan.ast, table);
        } else {
            result = execute_group_by(plan.ast, host_table, rows);
        }
    } else {
        if (plan.execution_kind == PlanExecutionKind::SelectColumn) {
            auto *var =
                dynamic_cast<VariableNode *>(plan.ast.select_list[0].get());
            const HostColumn *col = host_table.get_column(var->name);
            if (!col) {
                throw std::runtime_error("Unknown column in SELECT: " + var->name);
            }
            result = select_typed_column(*col, rows);
        } else {
            auto &out = std::get<std::vector<float>>(result);
            for (int idx : rows) {
                out.push_back(eval_node(plan.ast.select_list[0].get(), host_table, idx));
            }
        }
        if (plan.ast.order_by) {
            apply_order_by(plan.ast, host_table, rows, result);
        }
    }

    if (plan.ast.distinct) {
        apply_distinct(result);
    }

    apply_limit_offset(plan.ast, result);
    return QueryResult(std::move(result));
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

    if (!ast.joins.empty()) {
        validate_query_ast(ast, cols);
        if (ast.select_list.size() != 1) {
            throw std::runtime_error(
                "query_sql currently supports a single SELECT expression only");
        }
        if (ast.joins.size() != 1) {
            throw std::runtime_error(
                "JOIN execution currently supports exactly one JOIN clause");
        }
        if (ast.group_by || ast.having || ast.order_by) {
            throw std::runtime_error(
                "JOIN execution currently supports SELECT/WHERE/DISTINCT/LIMIT/OFFSET "
                "without GROUP BY, HAVING, or ORDER BY");
        }

        auto [left_rows, right_rows] = build_inner_join_rows(ast, host_table_);
        ColumnData result = std::vector<float>{};

        if (auto *var = dynamic_cast<VariableNode *>(ast.select_list[0].get())) {
            auto qualified = parse_join_variable_name(var->name);
            if (!qualified) {
                throw std::runtime_error(
                    "JOIN SELECT column must be qualified (e.g. a.col or b.col)");
            }
            const HostColumn *col = resolve_join_column(host_table_, qualified->relation,
                                                        ast.from_table, ast.joins[0].table,
                                                        qualified->column);
            switch (col->type) {
            case DataType::Int32: {
                const auto &src = std::get<std::vector<int32_t>>(col->data);
                std::vector<int32_t> out;
                out.reserve(left_rows.size());
                for (size_t i = 0; i < left_rows.size(); ++i) {
                    if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                    left_rows[i], right_rows[i],
                                                    ast.from_table, ast.joins[0].table) == 0.0f)
                        continue;
                    int idx = qualified->relation == ast.from_table ? left_rows[i] : right_rows[i];
                    out.push_back(src[idx]);
                }
                result = std::move(out);
                break;
            }
            case DataType::Int64: {
                const auto &src = std::get<std::vector<int64_t>>(col->data);
                std::vector<int64_t> out;
                out.reserve(left_rows.size());
                for (size_t i = 0; i < left_rows.size(); ++i) {
                    if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                    left_rows[i], right_rows[i],
                                                    ast.from_table, ast.joins[0].table) == 0.0f)
                        continue;
                    int idx = qualified->relation == ast.from_table ? left_rows[i] : right_rows[i];
                    out.push_back(src[idx]);
                }
                result = std::move(out);
                break;
            }
            case DataType::Float32: {
                const auto &src = std::get<std::vector<float>>(col->data);
                std::vector<float> out;
                out.reserve(left_rows.size());
                for (size_t i = 0; i < left_rows.size(); ++i) {
                    if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                    left_rows[i], right_rows[i],
                                                    ast.from_table, ast.joins[0].table) == 0.0f)
                        continue;
                    int idx = qualified->relation == ast.from_table ? left_rows[i] : right_rows[i];
                    out.push_back(src[idx]);
                }
                result = std::move(out);
                break;
            }
            case DataType::Float64: {
                const auto &src = std::get<std::vector<double>>(col->data);
                std::vector<double> out;
                out.reserve(left_rows.size());
                for (size_t i = 0; i < left_rows.size(); ++i) {
                    if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                    left_rows[i], right_rows[i],
                                                    ast.from_table, ast.joins[0].table) == 0.0f)
                        continue;
                    int idx = qualified->relation == ast.from_table ? left_rows[i] : right_rows[i];
                    out.push_back(src[idx]);
                }
                result = std::move(out);
                break;
            }
            case DataType::String: {
                const auto &src = std::get<std::vector<std::string>>(col->data);
                std::vector<std::string> out;
                out.reserve(left_rows.size());
                for (size_t i = 0; i < left_rows.size(); ++i) {
                    if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                    left_rows[i], right_rows[i],
                                                    ast.from_table, ast.joins[0].table) == 0.0f)
                        continue;
                    int idx = qualified->relation == ast.from_table ? left_rows[i] : right_rows[i];
                    out.push_back(src[idx]);
                }
                result = std::move(out);
                break;
            }
            }
        } else {
            auto &out = std::get<std::vector<float>>(result);
            out.reserve(left_rows.size());
            for (size_t i = 0; i < left_rows.size(); ++i) {
                if (ast.where && eval_join_node(ast.where.value().get(), host_table_,
                                                left_rows[i], right_rows[i], ast.from_table,
                                                ast.joins[0].table) == 0.0f)
                    continue;
                out.push_back(eval_join_node(ast.select_list[0].get(), host_table_,
                                             left_rows[i], right_rows[i], ast.from_table,
                                             ast.joins[0].table));
            }
        }

        if (ast.distinct) {
            apply_distinct(result);
        }
        apply_limit_offset(ast, result);
        return QueryResult(std::move(result));
    }

    QueryPlan plan = plan_query(std::move(ast), cols, table_);
    return execute_plan(plan, table_, host_table_);
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
    if (rows_per_chunk <= 0) {
        throw std::runtime_error("rows_per_chunk must be > 0");
    }

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
    ColumnData all_results = std::vector<float>{};
    while (!finished) {
        HostTable chunk =
            load_csv_chunk(file, rows_per_chunk, finished, column_names,
                           ParsePolicy::Strict, &schema);
        if (chunk.num_rows() == 0 && finished) {
            break;
        }
        auto part = run_multi_gpu_jit_host(chunk, expr_cuda, condition_cuda);
        auto &dst = std::get<std::vector<float>>(all_results);
        const auto &src = std::get<std::vector<float>>(part);
        dst.insert(dst.end(), src.begin(), src.end());
    }

    return QueryResult(std::move(all_results));
}
