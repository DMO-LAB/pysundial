#include "sundials_wrapper.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "cvode/cvode_module.hpp"
#include "arkode/arkode_module.hpp"
#include "common/common.hpp"

namespace py = pybind11;
using namespace sundials_py;

PYBIND11_MODULE(_SundialsPy, m) {
    m.doc() = "Python bindings for SUNDIALS solvers";
    
    // Initialize SUNDIALS context
    initialize_sundials_context();
    
    // Add version info
    m.attr("__version__") = "0.1.0";
    
    // Create submodules
    py::module_ cvode_module = m.def_submodule("cvode", "CVODE solver for ODE systems");
    py::module_ arkode_module = m.def_submodule("arkode", "ARKODE solver for ODE systems");
    
    // Initialize the modules
    init_cvode_module(cvode_module);
    sundials_py::init_arkode_module(arkode_module);
    
    // Register context cleanup
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            // Optional: log exceptions if desired
        }
    });
    
    // Add cleanup function
    m.def("_cleanup", []() {
        finalize_sundials_context();
    });
    
    // Inject atexit handler to ensure context is freed
    m.add_object("_cleanup_handler", py::module::import("atexit").attr("register")(m.attr("_cleanup")));
}
