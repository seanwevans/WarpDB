#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "warpdb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pywarpdb, m) {
    py::class_<WarpDB>(m, "WarpDB")
        .def(py::init<const std::string &>())
        .def("query", &WarpDB::query);
}
