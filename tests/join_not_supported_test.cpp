#include "warpdb.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

int main() {
    WarpDB db("data/test.csv");

    bool saw_error = false;
    try {
        (void)db.query_sql(
            "SELECT price FROM test JOIN test ON id == id WHERE price > 0");
    } catch (const std::runtime_error &e) {
        saw_error = std::string(e.what()).find("JOIN is parsed but not executed yet") !=
                    std::string::npos;
    }

    assert(saw_error);
    return 0;
}
