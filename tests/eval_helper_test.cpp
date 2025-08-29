#include "expression.hpp"
#include "csv_loader.hpp"
#include "eval_helpers.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

struct TestRow {
    float price;
    int quantity;
};

float get_value(const TestRow &r, const std::string &name, int) {
    if (name == "price") return r.price;
    if (name == "quantity") return static_cast<float>(r.quantity);
    return 0.0f;
}

int main() {
    HostColumn price;
    price.name = "price";
    price.type = DataType::Float32;
    price.data = std::vector<float>{10.0f, 5.0f};

    HostColumn quantity;
    quantity.name = "quantity";
    quantity.type = DataType::Int32;
    quantity.data = std::vector<int32_t>{2, 3};

    HostTable table;
    table.columns = {price, quantity};

    auto expr_tokens = tokenize("price * quantity + 1");
    auto expr_ast = parse_expression(expr_tokens);

    auto cond_tokens = tokenize("price > 7");
    auto cond_ast = parse_expression(cond_tokens);

    const auto &prices = std::get<std::vector<float>>(table.columns[0].data);
    const auto &qtys = std::get<std::vector<int32_t>>(table.columns[1].data);

    for (int i = 0; i < table.num_rows(); ++i) {
        TestRow r{prices[i], qtys[i]};
        float table_val = eval_node(expr_ast.get(), table, i);
        float row_val = eval_node(expr_ast.get(), r, 0);
        assert(std::fabs(table_val - row_val) < 1e-5);

        bool table_cond = eval_condition(cond_ast.get(), table, i);
        bool row_cond = eval_condition(cond_ast.get(), r, 0);
        assert(table_cond == row_cond);
    }

    std::cout << "All tests passed\n";
    return 0;
}

