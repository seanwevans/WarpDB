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
}

WarpDB::~WarpDB() {
    cudaFree(table_.d_price);
    cudaFree(table_.d_quantity);
}

std::vector<float> WarpDB::query(const std::string &expr) {
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
    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        auto cond_ast = parse_expression(cond_tokens);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table_.num_rows);

    jit_compile_and_launch(expr_cuda, condition_cuda, table_.d_price, table_.d_quantity, d_output, table_.num_rows);

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
}
