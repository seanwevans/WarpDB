#include "warpdb.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <iostream>

#include <map>
#include <utility>
#include "arrow_loader.hpp"
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

std::vector<float> run_multi_gpu_jit_host(const HostTable &host,
                                          const std::string &expr_cuda,
                                          const std::string &cond_cuda) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < 2) {
        Table dtab = upload_to_gpu(host);
        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * host.num_rows());
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab.d_price,
                               dtab.d_quantity, d_out, host.num_rows(), 0);
        std::vector<float> result(host.num_rows());
        cudaMemcpy(result.data(), d_out, sizeof(float) * host.num_rows(),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        cudaFree(dtab.d_price);
        cudaFree(dtab.d_quantity);
        return result;
    }

    int N = host.num_rows();
    int chunk = (N + device_count - 1) / device_count;
    std::vector<float> results(N);

    for (int dev = 0; dev < device_count; ++dev) {
        int start = dev * chunk;
        int end = std::min(start + chunk, N);
        if (start >= end)
            break;
        int local_N = end - start;

        HostTable sub;
        sub.price.assign(host.price.begin() + start, host.price.begin() + end);
        sub.quantity.assign(host.quantity.begin() + start,
                            host.quantity.begin() + end);
        cudaSetDevice(dev);
        Table dtab = upload_to_gpu(sub);

        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * local_N);

        jit_compile_and_launch(expr_cuda, cond_cuda, dtab.d_price,
                               dtab.d_quantity, d_out, local_N, dev);

        cudaMemcpy(results.data() + start, d_out, sizeof(float) * local_N,
                   cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(dtab.d_price);
        cudaFree(dtab.d_quantity);
    }

    return results;
}
} // namespace

WarpDB::WarpDB(const std::string &filepath) {
    auto dot = filepath.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : filepath.substr(dot + 1);
    for (auto &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "csv") {
        table_ = load_csv_to_gpu(filepath);
        host_table_ = load_csv_to_host(filepath);
    } else if (ext == "json") {
        table_ = load_json_to_gpu(filepath);

        host_table_ = load_json_to_host(filepath);

#ifdef USE_ARROW

    } else if (ext == "parquet") {
        table_ = load_parquet_to_gpu(filepath);
    } else if (ext == "arrow" || ext == "feather") {
        table_ = load_arrow_to_gpu(filepath);
    } else if (ext == "orc") {
        table_ = load_orc_to_gpu(filepath);
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

    std::vector<float> result;

    if (ast.group_by) {
        auto *agg = dynamic_cast<AggregationNode *>(ast.select_list[0].get());
        if (!agg) throw std::runtime_error("Only aggregation queries supported with GROUP BY");

        float *d_vals; int *d_keys; int *d_count;
        cudaMalloc(&d_vals, sizeof(float)*table_.num_rows);
        cudaMalloc(&d_keys, sizeof(int)*table_.num_rows);
        cudaMalloc(&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));

        std::string val_expr = agg->expr->to_cuda_expr();
        std::string key_expr = ast.group_by->keys[0]->to_cuda_expr();
        jit_group_sum(val_expr, key_expr, table_.d_price, table_.d_quantity,
                      d_vals, d_keys, d_count, table_.num_rows);

        int h_count = 0;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (ast.order_by) {
            jit_sort_pairs(d_keys, d_vals, h_count, ast.order_by->ascending);
        }

        std::vector<float> h_vals(h_count);
        cudaMemcpy(h_vals.data(), d_vals, sizeof(float)*h_count, cudaMemcpyDeviceToHost);
        cudaFree(d_vals); cudaFree(d_keys); cudaFree(d_count);

        int limit = ast.limit ? std::min(ast.limit->count, h_count) : h_count;
        for (int i=0;i<limit;i++) result.push_back(h_vals[i]);
    } else {
        float *d_out;
        cudaMalloc(&d_out, sizeof(float)*table_.num_rows);
        std::string expr_code = ast.select_list[0]->to_cuda_expr();
        jit_compile_and_launch(expr_code, "", table_.d_price, table_.d_quantity,
                               d_out, table_.num_rows);

        if (ast.order_by && expr_code == ast.order_by->expr->to_cuda_expr()) {
            jit_sort_float(d_out, table_.num_rows, ast.order_by->ascending);
        }

        result.resize(table_.num_rows);
        cudaMemcpy(result.data(), d_out, sizeof(float)*table_.num_rows, cudaMemcpyDeviceToHost);
        cudaFree(d_out);

        if (ast.limit && static_cast<size_t>(ast.limit->count) < result.size())
            result.resize(ast.limit->count);
    }

    return result;
}

void WarpDB::query_arrow(const std::string &expr, ArrowArray *out_array,
                         ArrowSchema *out_schema, bool use_shared_memory) {
    auto result = query(expr);
    export_to_arrow(result.data(), static_cast<int64_t>(result.size()),
                    use_shared_memory, out_array, out_schema);

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

    std::unordered_set<std::string> cols{"price", "quantity"};
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
    std::unordered_set<std::string> cols{"price", "quantity"};
    validate_ast(expr_ast.get(), cols);
    std::string expr_cuda = expr_ast->to_cuda_expr();

    std::string condition_cuda;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        auto cond_ast = parse_expression(cond_tokens);
        validate_ast(cond_ast.get(), cols);
        condition_cuda = cond_ast->to_cuda_expr();
    }

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + csv_path);
    }

    std::string header;
    std::getline(file, header);

    bool finished = false;
    std::vector<float> all_results;
    while (!finished) {
        HostTable chunk = load_csv_chunk(file, rows_per_chunk, finished);
        if (chunk.num_rows() == 0) break;
        auto part = run_multi_gpu_jit_host(chunk, expr_cuda, condition_cuda);
        all_results.insert(all_results.end(), part.begin(), part.end());
    }

    return all_results;
}
