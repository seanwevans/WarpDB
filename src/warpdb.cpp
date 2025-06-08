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
        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out, 0);
        std::vector<float> result(host.num_rows());
        cudaMemcpy(result.data(), d_out, sizeof(float) * host.num_rows(),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        for (auto &c : dtab.columns) cudaFree(c.device_ptr);
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
        sub.columns.resize(host.columns.size());
        for (size_t i=0;i<host.columns.size();++i) {
            sub.columns[i].name = host.columns[i].name;
            sub.columns[i].type = host.columns[i].type;
            if (host.columns[i].type == DataType::Float32) {
                auto &vec = std::get<std::vector<float>>(host.columns[i].data);
                sub.columns[i].data = std::vector<float>(vec.begin()+start, vec.begin()+end);
            } else if (host.columns[i].type == DataType::Int32) {
                auto &vec = std::get<std::vector<int32_t>>(host.columns[i].data);
                sub.columns[i].data = std::vector<int32_t>(vec.begin()+start, vec.begin()+end);
            }
        }
        cudaSetDevice(dev);
        Table dtab = upload_to_gpu(sub);

        float *d_out;
        cudaMalloc(&d_out, sizeof(float) * local_N);

        jit_compile_and_launch(expr_cuda, cond_cuda, dtab, d_out, dev);

        cudaMemcpy(results.data() + start, d_out, sizeof(float) * local_N,
                   cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        for (auto &c : dtab.columns) cudaFree(c.device_ptr);
    }

    return results;
}
} // namespace

namespace {

float get_value(const HostTable &table, const std::string &name, int idx) {
    const HostColumn *col = table.get_column(name);
    if (!col) return 0.0f;
    switch (col->type) {
    case DataType::Int32:
        return static_cast<float>(std::get<std::vector<int32_t>>(col->data)[idx]);
    case DataType::Int64:
        return static_cast<float>(std::get<std::vector<int64_t>>(col->data)[idx]);
    case DataType::Float32:
        return std::get<std::vector<float>>(col->data)[idx];
    case DataType::Float64:
        return static_cast<float>(std::get<std::vector<double>>(col->data)[idx]);
    default:
        return 0.0f;
    }
}

float eval_node(const ASTNode *node, const HostTable &table, int idx) {
    if (auto c = dynamic_cast<const ConstantNode *>(node)) {
        return std::stof(c->value);
    }
    if (auto v = dynamic_cast<const VariableNode *>(node)) {
        return get_value(table, v->name, idx);
    }
    if (auto b = dynamic_cast<const BinaryOpNode *>(node)) {
        float l = eval_node(b->left.get(), table, idx);
        float r = eval_node(b->right.get(), table, idx);
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
    return 0.0f;
}

bool eval_condition(const ASTNode *node, const HostTable &table, int idx) {
    return eval_node(node, table, idx) != 0.0f;
}

} // anonymous namespace

WarpDB::WarpDB(const std::string &filepath, const std::vector<DataType> &schema) {
    auto dot = filepath.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : filepath.substr(dot + 1);
    for (auto &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "csv") {
        table_ = load_csv_to_gpu(filepath, schema);
        host_table_ = load_csv_to_host(filepath, schema);
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
    for (auto &c : table_.columns) {
        if (c.device_ptr)
            cudaFree(c.device_ptr);
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

    float *d_output;
    cudaMalloc(&d_output, sizeof(float) * table_.num_rows);

    try {
        jit_compile_and_launch(expr_cuda, condition_cuda, table_, d_output);
    } catch (const std::exception &e) {
        cudaFree(d_output);
        throw;
    }

    std::vector<float> result(table_.num_rows);
    cudaMemcpy(result.data(), d_output, sizeof(float) * table_.num_rows, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    return result;
}

std::vector<float> WarpDB::query_sql(const std::string &sql) {
    auto tokens = tokenize(sql);
    QueryAST ast = parse_query(tokens);

    std::vector<int> rows;
    int N = host_table_.num_rows();
    rows.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (ast.where) {
            if (!eval_condition(ast.where.value().get(), host_table_, i)) continue;
        }
        rows.push_back(i);
    }

    std::vector<float> result;

    if (ast.group_by) {
        struct AggData { double sum=0.0; double count=0.0; double min=0.0; double max=0.0; bool init=false; };
        std::map<int, AggData> groups;
        auto *agg = dynamic_cast<AggregationNode *>(ast.select_list[0].get());
        for (int idx : rows) {
            int key = static_cast<int>(eval_node(ast.group_by->keys[0].get(), host_table_, idx));
            float val = 0.0f;
            if (agg && agg->agg != AggregationType::Count) {
                val = eval_node(agg->expr.get(), host_table_, idx);
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
        for (int idx : rows) {
            result.push_back(eval_node(ast.select_list[0].get(), host_table_, idx));
        }
    }

    if (ast.order_by) {
        std::vector<std::pair<float,float>> keyed;
        for (size_t i=0;i<rows.size();++i) {
            float key = eval_node(ast.order_by->expr.get(), host_table_, rows[i]);
            keyed.push_back({key, result[i]});
        }
        std::sort(keyed.begin(), keyed.end(), [&](const auto &a, const auto &b){
            if (ast.order_by->ascending) return a.first < b.first; else return a.first > b.first;
        });
        result.clear();
        for (const auto &kv : keyed) result.push_back(kv.second);
    }

    if (ast.limit) {
        if (static_cast<size_t>(ast.limit->count) < result.size())
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
