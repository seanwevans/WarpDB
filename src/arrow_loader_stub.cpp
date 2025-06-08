#include "arrow_loader.hpp"
#include <stdexcept>

#if !defined(USE_ARROW) && !defined(USE_SIMPLE_READERS)
Table load_parquet_to_gpu(const std::string &filepath) {
    throw std::runtime_error("Arrow support not enabled");
}

Table load_arrow_to_gpu(const std::string &filepath) {
    throw std::runtime_error("Arrow support not enabled");
}

Table load_orc_to_gpu(const std::string &filepath) {
    throw std::runtime_error("Arrow support not enabled");
}
#endif
