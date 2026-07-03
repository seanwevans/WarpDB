#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "warpdb.hpp"

namespace py = pybind11;

static py::object column_data_to_python(const ColumnData &data) {
    return std::visit([](const auto &vec) -> py::object { return py::cast(vec); }, data);
}

PYBIND11_MODULE(pywarpdb, m) {
    py::enum_<DataType>(m, "DataType")
        .value("Int32", DataType::Int32)
        .value("Int64", DataType::Int64)
        .value("Float32", DataType::Float32)
        .value("Float64", DataType::Float64)
        .value("String", DataType::String);

    py::enum_<ParsePolicy>(m, "ParsePolicy")
        .value("Strict", ParsePolicy::Strict)
        .value("Permissive", ParsePolicy::Permissive);

    m.def("cuda_device_count", []() {
                int count = 0;
                if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
                return count;
             },
             R"pbdoc(Return the number of available CUDA devices (0 if none).)pbdoc");

    py::class_<WarpDB>(m, "WarpDB")
        .def(py::init<const std::string &, const std::vector<DataType>&, ParsePolicy>(),
             py::arg("filepath"),
             py::arg("schema") = std::vector<DataType>(),
             py::arg("policy") = ParsePolicy::Strict)
        .def("query", [](WarpDB &db, const std::string &expr) {
                return column_data_to_python(db.query(expr).data());
             }, py::arg("expr"))
        .def("query_sql", [](WarpDB &db, const std::string &sql) {
                return column_data_to_python(db.query_sql(sql).data());
             }, py::arg("sql"),
             R"pbdoc(Execute a full SQL query supporting GROUP BY and ORDER BY.)pbdoc")
        .def("query_multi_gpu", [](WarpDB &db, const std::string &expr) {
                return column_data_to_python(db.query_multi_gpu(expr).data());
             },
             py::arg("expr"),
             R"pbdoc(Execute expression using all available GPUs on the current table.)pbdoc")
        .def_static("query_multi_gpu_csv", [](const std::string &csv_path,
                                              const std::string &expr,
                                              int rows_per_chunk) {
                return column_data_to_python(
                    WarpDB::query_multi_gpu_csv(csv_path, expr, rows_per_chunk).data());
             },
                    py::arg("csv_path"), py::arg("expr"),
                    py::arg("rows_per_chunk") = 1000000,
                    R"pbdoc(Stream a CSV file in chunks across all GPUs and return results.)pbdoc")
        .def("query_arrow",
             [](WarpDB &db, const std::string &expr, bool shared_memory,
                const std::string &shm_name) {
                auto arr = new ArrowArray();
                auto schema = new ArrowSchema();
                 db.query_arrow(expr, arr, schema, shared_memory,
                                shm_name.c_str());
                py::capsule array_capsule(arr, [](void *ptr) {
                    ArrowArray *arr = reinterpret_cast<ArrowArray *>(ptr);
                    if (arr->release) arr->release(arr);
                    delete arr;
                });
                py::capsule schema_capsule(schema, [](void *ptr) {
                    ArrowSchema *schema = reinterpret_cast<ArrowSchema *>(ptr);
                    if (schema->release) schema->release(schema);
                    delete schema;
                });
                return py::make_tuple(array_capsule, schema_capsule);
            },
             py::arg("expr"), py::arg("shared_memory") = false,
             py::arg("shm_name") = "/warpdb_result",
             R"pbdoc(Return result as Arrow C Data Interface capsules.

The returned tuple contains (ArrowArray capsule, ArrowSchema capsule).
Use pyarrow.Array._import_from_c(schema, array) to construct a PyArrow Array.)pbdoc");
}
