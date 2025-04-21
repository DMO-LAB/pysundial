#pragma once

// Include our SUNDIALS wrapper to ensure types are properly defined
#include "../sundials_wrapper.hpp"

// Then include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

namespace sundials_py {

// Enum for linear solver types
enum class LinearSolverType {
    DENSE,
    BAND,
    SPARSE,
    SPGMR,
    SPBCG,
    SPTFQMR,
    PCG
};

// Enum for iteration type
enum class IterationType {
    FUNCTIONAL,
    NEWTON
};

// Global SUNContext
extern SUNContext sunctx;

// Initialize SUNDIALS context
void initialize_sundials_context();

// Finalize SUNDIALS context
void finalize_sundials_context();

// Convert Python numpy array to SUNDIALS N_Vector
N_Vector numpy_to_nvector(py::array_t<realtype>& np_array);

// Convert SUNDIALS N_Vector to numpy array
py::array_t<realtype> nvector_to_numpy(N_Vector vec);

// RHS function type for ODEs: dy/dt = f(t, y)
using PyRhsFn = std::function<py::array_t<realtype>(double, py::array_t<realtype>)>;

// Jacobian function type: J = df/dy
using PyJacFn = std::function<py::array_t<realtype>(double, py::array_t<realtype>)>;

// Root finding function type: g(t, y) = 0
using PyRootFn = std::function<py::array_t<realtype>(double, py::array_t<realtype>)>;

// Error handling
void check_flag(void* flagvalue, const char* funcname, int opt);

// Class to manage the lifetime of an N_Vector
class NVectorOwner {
private:
    N_Vector vec_;

public:
    NVectorOwner(N_Vector vec) : vec_(vec) {}
    ~NVectorOwner() { 
        if (vec_) {
            N_VDestroy_Serial(vec_);
        }
    }
    
    // Disable copying
    NVectorOwner(const NVectorOwner&) = delete;
    NVectorOwner& operator=(const NVectorOwner&) = delete;
    
    // Enable moving
    NVectorOwner(NVectorOwner&& other) noexcept : vec_(other.vec_) {
        other.vec_ = nullptr;
    }
    NVectorOwner& operator=(NVectorOwner&& other) noexcept {
        if (this != &other) {
            if (vec_) {
                N_VDestroy_Serial(vec_);
            }
            vec_ = other.vec_;
            other.vec_ = nullptr;
        }
        return *this;
    }
    
    N_Vector get() const { return vec_; }
};

}  // namespace sundials_py