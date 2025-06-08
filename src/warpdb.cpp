#include "warpdb.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <iostream>

#include <map>
#include <utility>

WarpDB::WarpDB(const std::string &csv_path) {
    host_table_ = load_csv_to_host(csv_path);
    table_ = upload_to_gpu(host_table_);

#include <stdexcept>
#include <unordered_set>
#include <memory>

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

WarpDB::WarpDB(const std::string &filepath) {
    auto dot = filepath.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : filepath.substr(dot + 1);
    for (auto &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "csv") {
        table_ = load_csv_to_gpu(filepath);
    } else if (ext == "json") {
        table_ = load_json_to_gpu(filepath);
    } else if (ext == "parquet") {
        table_ = load_parquet_to_gpu(filepath);
    } else if (ext == "arrow" || ext == "feather") {
        table_ = load_arrow_to_gpu(filepath);
    } else if (ext == "orc") {
        table_ = load_orc_to_gpu(filepath);
    } else {
        throw std::runtime_error("Unsupported file format: " + filepath);
    }

}

WarpDB::~WarpDB() {
#ifdef USE_ARROW
    table_.d_price.reset();
    table_.d_quantity.reset();
#else
    cudaFree(table_.d_price);
    cudaFree(table_.d_quantity);
#endif
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

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table_.num_rows);

    try {
        jit_compile_and_launch(expr_cuda, condition_cuda, table_.d_price, table_.d_quantity, d_output, table_.num_rows);
    } catch (const std::exception &e) {
        cudaFree(d_output);
        throw;
    }

    std::vector<float> result(table_.num_rows);
    cudaMemcpy(result.data(), d_output, sizeof(float) * table_.num_rows, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    return result;
}


namespace {
struct Row { float price; int quantity; };

float eval_node(const ASTNode *node, const Row &r) {
    if (auto c = dynamic_cast<const ConstantNode *>(node)) {
        return std::stof(c->value);
    }
    if (auto v = dynamic_cast<const VariableNode *>(node)) {
        if (v->name == "price") return r.price;
        if (v->name == "quantity") return static_cast<float>(r.quantity);
        return 0.0f;
    }
    if (auto b = dynamic_cast<const BinaryOpNode *>(node)) {
        float l = eval_node(b->left.get(), r);
        float rr = eval_node(b->right.get(), r);
        const std::string &op = b->op;
        if (op == "+") return l + rr;
        if (op == "-") return l - rr;
        if (op == "*") return l * rr;
        if (op == "/") return l / rr;
        if (op == ">") return l > rr;
        if (op == "<") return l < rr;
        if (op == ">=") return l >= rr;
        if (op == "<=") return l <= rr;
        if (op == "==") return l == rr;
        if (op == "!=") return l != rr;
    }
    return 0.0f;
}

bool eval_condition(const ASTNode *node, const Row &r) {
    return eval_node(node, r) != 0.0f;
}
}

std::vector<float> WarpDB::query_sql(const std::string &sql) {
    auto tokens = tokenize(sql);
    QueryAST ast = parse_query(tokens);

    std::vector<Row> rows;
    rows.reserve(host_table_.num_rows());
    for (int i = 0; i < host_table_.num_rows(); ++i) {
        rows.push_back({host_table_.price[i], host_table_.quantity[i]});
    }

    if (ast.where) {
        std::vector<Row> filtered;
        for (const auto &r : rows) {
            if (eval_condition(ast.where.value().get(), r)) filtered.push_back(r);
        }
        rows.swap(filtered);
    }

    std::vector<float> result;

    if (ast.group_by) {
        struct AggData { double sum=0.0; double count=0.0; double min=0.0; double max=0.0; bool init=false; };
        std::map<int, AggData> groups;
        auto *agg = dynamic_cast<AggregationNode *>(ast.select_list[0].get());
        for (const auto &r : rows) {
            int key = static_cast<int>(eval_node(ast.group_by->keys[0].get(), r));
            float val = 0.0f;
            if (agg && agg->agg != AggregationType::Count) {
                val = eval_node(agg->expr.get(), r);
            }
            auto &g = groups[key];
            if (!g.init) { g.min = g.max = val; g.init = true; }
            g.sum += val;
            g.count += 1.0;
            g.min = std::min(g.min, (double)val);
            g.max = std::max(g.max, (double)val);
        }
        for (const auto &kv : groups) {
            const AggData &g = kv.second;
            switch (agg->agg) {
            case AggregationType::Sum: result.push_back(g.sum); break;
            case AggregationType::Avg: result.push_back(g.sum / g.count); break;
            case AggregationType::Count: result.push_back(g.count); break;
            case AggregationType::Min: result.push_back(g.min); break;
            case AggregationType::Max: result.push_back(g.max); break;
            }
        }
    } else {
        for (const auto &r : rows) {
            result.push_back(eval_node(ast.select_list[0].get(), r));
        }
    }

    if (ast.order_by) {
        std::vector<std::pair<float,float>> keyed;
        for (size_t i=0;i<result.size();++i) {
            Row tmp{rows[i].price, rows[i].quantity};
            float key = eval_node(ast.order_by->expr.get(), tmp);
            keyed.push_back({key, result[i]});
        }
        std::sort(keyed.begin(), keyed.end(), [&](const auto &a, const auto &b){
            if (ast.order_by->ascending) return a.first < b.first; else return a.first > b.first;
        });
        result.clear();
        for (const auto &kv : keyed) result.push_back(kv.second);
    }

    return result;

void WarpDB::query_arrow(const std::string &expr, ArrowArray *out_array,
                         ArrowSchema *out_schema, bool use_shared_memory) {
    auto result = query(expr);
    export_to_arrow(result.data(), static_cast<int64_t>(result.size()),
                    use_shared_memory, out_array, out_schema);

}
