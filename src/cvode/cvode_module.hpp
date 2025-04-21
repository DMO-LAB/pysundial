#pragma once

#include "../sundials_wrapper.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_cvode_module(py::module_ &m);