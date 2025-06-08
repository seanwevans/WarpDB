#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "warpdb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pywarpdb, m) {
    py::class_<WarpDB>(m, "WarpDB")
        .def(py::init<const std::string &>())
        .def("query", &WarpDB::query)
        .def("query_arrow",
             [](WarpDB &db, const std::string &expr, bool shared_memory) {
                 auto arr = new ArrowArray();
                 auto schema = new ArrowSchema();
                 db.query_arrow(expr, arr, schema, shared_memory);
                 py::capsule array_capsule(arr, [](void *ptr) {
                     ArrowArray *arr = reinterpret_cast<ArrowArray *>(ptr);
                     if (arr->release) arr->release(arr);
                 });
                 py::capsule schema_capsule(schema, [](void *ptr) {
                     ArrowSchema *schema = reinterpret_cast<ArrowSchema *>(ptr);
                     if (schema->release) schema->release(schema);
                 });
                 return py::make_tuple(array_capsule, schema_capsule);
             },
             py::arg("expr"), py::arg("shared_memory") = false,
             R"pbdoc(Return result as Arrow C Data Interface capsules.

The returned tuple contains (ArrowArray capsule, ArrowSchema capsule).
Use pyarrow.Array._import_from_c(schema, array) to construct a PyArrow Array.)pbdoc");
}
