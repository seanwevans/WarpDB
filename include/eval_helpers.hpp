#pragma once
#include "csv_loader.hpp"
#include "expression.hpp"
#include <string>

inline float get_value(const HostTable &table, const std::string &name, int idx) {
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

template <typename Context>
float eval_node(const ASTNode *node, const Context &ctx, int idx) {
    if (auto c = dynamic_cast<const ConstantNode *>(node)) {
        return std::stof(c->value);
    }
    if (auto v = dynamic_cast<const VariableNode *>(node)) {
        return get_value(ctx, v->name, idx);
    }
    if (auto b = dynamic_cast<const BinaryOpNode *>(node)) {
        float l = eval_node(b->left.get(), ctx, idx);
        float r = eval_node(b->right.get(), ctx, idx);
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

template <typename Context>
bool eval_condition(const ASTNode *node, const Context &ctx, int idx) {
    return eval_node(node, ctx, idx) != 0.0f;
}

