#include "optimizer.hpp"
#include "jit.hpp"
#include <cuda_runtime.h>
#include "cuda_utils.hpp"
#include <iostream>
#include <memory>
#include <vector>

namespace {

float parse_constant(const std::string &val) {
    return std::stof(val);
}

} // namespace

void analyze_condition(const ASTNode *node, const TableStats &stats,
                       bool &always_true, bool &always_false) {
    always_true = false;
    always_false = false;
    if (!node)
        return;

    if (auto bin = dynamic_cast<const BinaryOpNode *>(node)) {
        // Handle logical operators by recursively analyzing children
        if (bin->op == "&&" || bin->op == "||") {
            bool l_true = false, l_false = false;
            bool r_true = false, r_false = false;
            analyze_condition(bin->left.get(), stats, l_true, l_false);
            analyze_condition(bin->right.get(), stats, r_true, r_false);

            if (bin->op == "&&") {
                if (l_false || r_false)
                    always_false = true;
                if (l_true && r_true)
                    always_true = true;
            } else { // ||
                if (l_true || r_true)
                    always_true = true;
                if (l_false && r_false)
                    always_false = true;
            }
            return;
        }

        // Check for comparisons between a column and a constant
        const VariableNode *var = nullptr;
        const ConstantNode *cnst = nullptr;
        bool var_left = true;

        if (bin->left->type() == ASTNodeType::Variable &&
            bin->right->type() == ASTNodeType::Constant) {
            var = static_cast<const VariableNode *>(bin->left.get());
            cnst = static_cast<const ConstantNode *>(bin->right.get());
        } else if (bin->left->type() == ASTNodeType::Constant &&
                   bin->right->type() == ASTNodeType::Variable) {
            var = static_cast<const VariableNode *>(bin->right.get());
            cnst = static_cast<const ConstantNode *>(bin->left.get());
            var_left = false;
        }

        if (var && cnst) {
            float c = parse_constant(cnst->value);
            float min = 0.0f, max = 0.0f;
            bool known = true;

            if (var->name == "price") {
                min = stats.price.min;
                max = stats.price.max;
            } else if (var->name == "quantity") {
                min = static_cast<float>(stats.quantity.min);
                max = static_cast<float>(stats.quantity.max);
            } else {
                known = false;
            }

            if (known) {
                std::string op = bin->op;
                if (!var_left) {
                    if (op == ">")
                        op = "<";
                    else if (op == "<")
                        op = ">";
                    else if (op == ">=")
                        op = "<=";
                    else if (op == "<=")
                        op = ">=";
                }

                if (op == ">") {
                    if (min > c)
                        always_true = true;
                    else if (max <= c)
                        always_false = true;
                } else if (op == ">=") {
                    if (min >= c)
                        always_true = true;
                    else if (max < c)
                        always_false = true;
                } else if (op == "<") {
                    if (max < c)
                        always_true = true;
                    else if (min >= c)
                        always_false = true;
                } else if (op == "<=") {
                    if (max <= c)
                        always_true = true;
                    else if (min > c)
                        always_false = true;
                } else if (op == "==" || op == "=") {
                    if (min == max && min == c)
                        always_true = true;
                    else if (c < min || c > max)
                        always_false = true;
                } else if (op == "!=") {
                    if (min == max && min == c)
                        always_false = true;
                    else if (c < min || c > max)
                        always_true = true;
                }
            }
        }
    }
}

void execute_query_optimized(const std::string &expr_part,
                             const std::string &where_part, Table &table) {
    auto expr_tokens = tokenize(expr_part);
    auto expr_ast = parse_expression(expr_tokens);

    std::unique_ptr<ASTNode> cond_ast;
    if (!where_part.empty()) {
        auto cond_tokens = tokenize(where_part);
        cond_ast = parse_expression(cond_tokens);
    }

    bool always_true = false;
    bool always_false = false;
    if (cond_ast) {
        analyze_condition(cond_ast.get(), {}, always_true, always_false);
    }

    if (always_false) {
        std::cout << "[Optimizer] Filter eliminates all rows.\n";
        return;
    }

    std::string expr_cuda = expr_ast->to_cuda_expr();
    std::string cond_cuda;
    if (cond_ast && !always_true) {
        cond_cuda = cond_ast->to_cuda_expr();
    }

    DeviceBuffer<float> d_output(table.num_rows);
    // Pass 0 for block_size so jit_compile_and_launch selects an occupancy
    // optimised configuration.
    jit_compile_and_launch(expr_cuda, cond_cuda, table, d_output.get(), 0, 0);

    std::vector<float> h_out(table.num_rows);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_output.get(),
                          sizeof(float) * table.num_rows,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < table.num_rows; ++i) {
        std::cout << "Result[" << i << "] = " << h_out[i] << "\n";
    }

    delete[] h_out;
}
