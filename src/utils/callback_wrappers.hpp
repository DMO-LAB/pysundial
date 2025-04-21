#pragma once
#include <iostream>
#include <optional>

// Include our SUNDIALS wrapper to ensure types are properly defined
#include "../sundials_wrapper.hpp"

// Include our common utilities
#include "../common/common.hpp"

// Then include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;
namespace sundials_py {

// Forward declaration of the ARKodeSolver class
class ARKodeSolver;

// Declare the external pointer for use in callbacks
extern ARKodeSolver* current_solver_for_jacobian;

// Structures and function wrappers for RHS, Jacobian, Root finding
// Structure to hold Python RHS function and additional data
struct PyRhsFnData {
    PyRhsFn py_rhs_fn;
};

// CVODE right-hand side function wrapper with improved error handling
inline int rhs_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    // Extra validation to catch null pointers
    if (user_data == nullptr) {
        std::cerr << "ERROR: user_data is NULL in rhs_wrapper" << std::endl;
        return -1;
    }
    
    if (y == nullptr || ydot == nullptr) {
        std::cerr << "ERROR: Null vector passed to rhs_wrapper" << std::endl;
        return -1;
    }
    
    // Extract Python function from user_data
    PyRhsFnData* data = static_cast<PyRhsFnData*>(user_data);
    
    try {
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Get the dimensions of the input vector
        long int N = N_VGetLength_Serial(y);
        
        // Create a flat 1D numpy array from N_Vector
        py::array_t<realtype, py::array::c_style> y_array({N});
        py::buffer_info y_buf = y_array.request();
        realtype* y_ptr = static_cast<realtype*>(y_buf.ptr);
        realtype* y_data = N_VGetArrayPointer_Serial(y);
        
        // Copy data from N_Vector to numpy array
        for (long int i = 0; i < N; ++i) {
            y_ptr[i] = y_data[i];
        }
        
        // Call Python function with proper array shape
        py::array_t<realtype> result;
        try {
            result = data->py_rhs_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python RHS function: " << e.what() << std::endl;
            return -1;
        }
        
        // Verify the result shape
        py::buffer_info result_buf = result.request();
        long int result_size = result_buf.size;
        
        // Check for size mismatch
        if (result_size != N) {
            std::cerr << "[ERROR] Size mismatch in rhs_wrapper. Expected: " 
                      << N << ", Got: " << result_size << std::endl;
            return -1;
        }
        
        // Copy result back to N_Vector
        realtype* result_ptr = static_cast<realtype*>(result_buf.ptr);
        realtype* ydot_data = N_VGetArrayPointer_Serial(ydot);
        
        for (long int i = 0; i < N; ++i) {
            ydot_data[i] = result_ptr[i];
        }
        
        return 0;  // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in rhs_wrapper: " << e.what() << std::endl;
        return -1;  // Failure
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in rhs_wrapper" << std::endl;
        return -1;  // Failure
    }
}

// Structure to hold Python Jacobian function and additional data
struct PyJacFnData {
    PyJacFn py_jac_fn;
    int N;  // System size
};

// Improved dense Jacobian function wrapper - forward declaration
inline int jac_dense_wrapper(realtype t, N_Vector y, N_Vector fy, 
                      SUNMatrix J, void* user_data, 
                      N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

// Structure to hold Python root function data
struct PyRootFnData {
    PyRootFn py_root_fn;
    int nrtfn;  // Number of root functions
};

// Root function wrapper
inline int root_wrapper(realtype t, N_Vector y, realtype* gout, void* user_data) {
    // Validate input
    if (user_data == nullptr) {
        std::cerr << "ERROR: user_data is NULL in root_wrapper" << std::endl;
        return -1;
    }
    
    if (y == nullptr || gout == nullptr) {
        std::cerr << "ERROR: Null vector or output array in root_wrapper" << std::endl;
        return -1;
    }
    
    // Extract Python function from user_data
    PyRootFnData* data = static_cast<PyRootFnData*>(user_data);
    
    try {
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> g_array;
        try {
            g_array = data->py_root_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python root function: " << e.what() << std::endl;
            return -1;
        }
        
        // Copy result back to gout
        py::buffer_info g_buffer = g_array.request();
        realtype* g_data = static_cast<realtype*>(g_buffer.ptr);
        
        // Check for size mismatch
        if (g_buffer.size != data->nrtfn) {
            std::cerr << "[ERROR] Root function size mismatch. Expected: " 
                      << data->nrtfn << ", Got: " << g_buffer.size << "\n";
            return -1;
        }
        
        for (int i = 0; i < data->nrtfn; ++i) {
            gout[i] = g_data[i];
        }
        
        return 0;  // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in root_wrapper: " << e.what() << "\n";
        return -1;  // Failure
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in root_wrapper\n";
        return -1;
    }
}

// Structure to hold Python functions for ARKODE
struct PyArkFnData {
    PyRhsFn py_explicit_fn;  // Explicit (non-stiff) part
    std::optional<PyRhsFn> py_implicit_fn;  // Implicit (stiff) part
};

// ARKODE explicit (non-stiff) function wrapper with improved error handling
inline int ark_explicit_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    // Validate input
    if (user_data == nullptr) {
        std::cerr << "ERROR: user_data is NULL in ark_explicit_wrapper" << std::endl;
        return -1;
    }
    
    if (y == nullptr || ydot == nullptr) {
        std::cerr << "ERROR: Null vector in ark_explicit_wrapper" << std::endl;
        return -1;
    }
    
    PyArkFnData* data = static_cast<PyArkFnData*>(user_data);
    
    try {
        if (!data->py_explicit_fn) {
            std::cerr << "[ERROR] ARKODE: explicit_fn is NULL\n";
            return -1;
        }
        
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> ydot_array;
        try {
            ydot_array = data->py_explicit_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python explicit function: " << e.what() << std::endl;
            // Try to continue with zero derivative as fallback
            N_VConst(0.0, ydot);
            return 0;  // Return success to prevent solver termination
        }
        
        // Copy result back to N_Vector
        py::buffer_info ydot_buf = ydot_array.request();
        realtype* ydot_data = N_VGetArrayPointer_Serial(ydot);
        realtype* py_ydot_data = static_cast<realtype*>(ydot_buf.ptr);
        
        // Check for size mismatch
        if (ydot_buf.size != N_VGetLength_Serial(y)) {
            std::cerr << "[ERROR] ARKODE: Output size mismatch in explicit_fn. Expected: " 
                      << N_VGetLength_Serial(y) << ", Got: " << ydot_buf.size << "\n";
            // Try to continue with zero derivative as fallback
            N_VConst(0.0, ydot);
            return 0;  // Return success to prevent solver termination
        }
        
        for (long int i = 0; i < ydot_buf.size; ++i) {
            ydot_data[i] = py_ydot_data[i];
        }
        
        return 0;  // Success
    } 
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in ark_explicit_wrapper: " << e.what() << "\n";
        // Try to continue with zero derivative as fallback
        N_VConst(0.0, ydot);
        return 0;  // Return success to prevent solver termination
    } 
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in ark_explicit_wrapper\n";
        // Try to continue with zero derivative as fallback
        N_VConst(0.0, ydot);
        return 0;  // Return success to prevent solver termination
    }
}

// ARKODE implicit (stiff) function wrapper with improved error handling
inline int ark_implicit_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    // Validate input
    if (user_data == nullptr) {
        std::cerr << "ERROR: user_data is NULL in ark_implicit_wrapper" << std::endl;
        N_VConst(0.0, ydot);  // Set to zero
        return 0;  // Return success to prevent solver termination
    }
    
    if (y == nullptr || ydot == nullptr) {
        std::cerr << "ERROR: Null vector in ark_implicit_wrapper" << std::endl;
        return -1;
    }
    
    // Extract Python function from user_data
    PyArkFnData* data = static_cast<PyArkFnData*>(user_data);
    
    try {
        // Skip if function is not provided
        if (!data->py_implicit_fn.has_value()) {
            N_VConst(0.0, ydot);  // Set to zero
            return 0;
        }
        
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> ydot_array;
        try {
            ydot_array = data->py_implicit_fn.value()(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python implicit function: " << e.what() << std::endl;
            // Try to continue with zero derivative as fallback
            N_VConst(0.0, ydot);
            return 0;  // Return success to prevent solver termination
        }
        
        // Copy result back to N_Vector
        py::buffer_info ydot_buffer = ydot_array.request();
        realtype* ydot_data = N_VGetArrayPointer_Serial(ydot);
        realtype* py_ydot_data = static_cast<realtype*>(ydot_buffer.ptr);
        
        // Check for size mismatch
        if (ydot_buffer.size != N_VGetLength_Serial(y)) {
            std::cerr << "[ERROR] ARKODE: Output size mismatch in implicit_fn. Expected: " 
                      << N_VGetLength_Serial(y) << ", Got: " << ydot_buffer.size << "\n";
            // Try to continue with zero derivative as fallback
            N_VConst(0.0, ydot);
            return 0;  // Return success to prevent solver termination
        }
        
        for (long int i = 0; i < N_VGetLength_Serial(y); ++i) {
            ydot_data[i] = py_ydot_data[i];
        }
        
        return 0;  // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in ark_implicit_wrapper: " << e.what() << "\n";
        // Try to continue with zero derivative as fallback
        N_VConst(0.0, ydot);
        return 0;  // Return success to prevent solver termination
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in ark_implicit_wrapper\n";
        // Try to continue with zero derivative as fallback
        N_VConst(0.0, ydot);
        return 0;  // Return success to prevent solver termination
    }
}

// Implementation of the Jacobian wrapper function after full class declaration
inline int jac_dense_wrapper(realtype t, N_Vector y, N_Vector fy, 
                      SUNMatrix J, void* user_data, 
                      N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    try {
        // Get the jacobian data - either from user_data or current_solver_for_jacobian
        PyJacFnData* data = nullptr;
        
        if (user_data != nullptr) {
            // Try to use user_data if provided
            data = static_cast<PyJacFnData*>(user_data);
        } else if (current_solver_for_jacobian != nullptr) {
            // We'll need to get jac_data_ from the current solver
            // This requires the full ARKodeSolver class definition
            // Just set it to nullptr and print an error message for now
            std::cerr << "WARNING: Unable to access jac_data_ yet, will be fixed in arkode_module.cpp" << std::endl;
            return -1;
        } else {
            std::cerr << "ERROR: No valid Jacobian data found" << std::endl;
            return -1;
        }
        
        if (data == nullptr) {
            std::cerr << "ERROR: Jacobian data is null" << std::endl;
            return -1;
        }
        
        // Get the actual system size for validation
        int N = N_VGetLength_Serial(y);
        data->N = N;  // Update the N value in the data structure
        
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> jac_array;
        try {
            jac_array = data->py_jac_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python Jacobian function: " << e.what() << std::endl;
            return -1;
        }
        
        // Copy result back to SUNMatrix (assuming dense matrix)
        py::buffer_info jac_buffer = jac_array.request();
        
        // Handle different dimensionality cases
        if (jac_buffer.ndim == 2) {
            // 2D array case - check dimensions
            int rows = jac_buffer.shape[0];
            int cols = jac_buffer.shape[1];
            
            if (rows != N || cols != N) {
                std::cerr << "ERROR: Jacobian dimensions mismatch. Expected: " << N << "x" << N 
                          << ", Got: " << rows << "x" << cols << std::endl;
                return -1;
            }
            
            realtype* jac_data = static_cast<realtype*>(jac_buffer.ptr);
            
            // Fill the SUNMatrix based on stride information (handle row/column major)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    // Calculate index based on storage order
                    int index;
                    if (jac_buffer.strides[0] <= jac_buffer.strides[1]) {
                        // Row-major (C-style)
                        index = i * cols + j;
                    } else {
                        // Column-major (Fortran-style)
                        index = i + j * rows;
                    }
                    
                    // Set matrix element
                    SM_ELEMENT_D(J, i, j) = jac_data[index];
                }
            }
        } else if (jac_buffer.ndim == 1 && jac_buffer.size == N * N) {
            // 1D array case - reshape as needed
            realtype* jac_data = static_cast<realtype*>(jac_buffer.ptr);
            
            // Fill the SUNMatrix assuming row-major order for 1D array
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    SM_ELEMENT_D(J, i, j) = jac_data[i * N + j];
                }
            }
        } else {
            std::cerr << "ERROR: Invalid Jacobian dimensions. Expected 2D array or 1D array of size N*N" << std::endl;
            return -1;
        }
        
        return 0;  // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in jac_dense_wrapper: " << e.what() << std::endl;
        return -1;  // Failure
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in jac_dense_wrapper" << std::endl;
        return -1;
    }
}

}  // namespace sundials_py