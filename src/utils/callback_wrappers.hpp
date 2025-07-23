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

// Structure to hold Python functions for ARKODE - this will be the MAIN user_data
struct PyArkFnData {
    PyRhsFn py_explicit_fn;  // Explicit (non-stiff) part
    std::optional<PyRhsFn> py_implicit_fn;  // Implicit (stiff) part
    PyJacFn py_jacobian_fn;  // Jacobian function (optional)
    bool has_jacobian;       // Flag to indicate if Jacobian is set
    int N;                   // System size for validation
};

// Structure to hold Python RHS function and additional data (for backwards compatibility)
struct PyRhsFnData {
    PyRhsFn py_rhs_fn;
};

// Structure to hold Python Jacobian function and additional data
struct PyJacFnData {
    PyJacFn py_jac_fn;
    int N;  // System size
};

// Structure to hold Python root function data
struct PyRootFnData {
    PyRootFn py_root_fn;
    int nrtfn;  // Number of root functions
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
        
        // Get system size for validation
        long int N = N_VGetLength_Serial(y);
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> ydot_array;
        try {
            ydot_array = data->py_explicit_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python explicit function: " << e.what() << std::endl;
            return -1;
        }
        
        // Validate result before copying
        py::buffer_info ydot_buf = ydot_array.request();
        
        // Check for size mismatch
        if (ydot_buf.size != N) {
            std::cerr << "[ERROR] ARKODE: Output size mismatch in explicit_fn. Expected: " 
                      << N << ", Got: " << ydot_buf.size << std::endl;
            return -1;  // Return error to stop solver
        }
        
        // Copy result back to N_Vector
        realtype* ydot_data = N_VGetArrayPointer_Serial(ydot);
        realtype* py_ydot_data = static_cast<realtype*>(ydot_buf.ptr);
        
        for (long int i = 0; i < N; ++i) {
            ydot_data[i] = py_ydot_data[i];
        }
        
        return 0;  // Success
    } 
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in ark_explicit_wrapper: " << e.what() << std::endl;
        return -1;  // Return error to stop solver
    } 
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in ark_explicit_wrapper" << std::endl;
        return -1;  // Return error to stop solver
    }
}

// ARKODE implicit (stiff) function wrapper with improved error handling
inline int ark_implicit_wrapper(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    // Validate input
    if (user_data == nullptr) {
        std::cerr << "ERROR: user_data is NULL in ark_implicit_wrapper" << std::endl;
        N_VConst(0.0, ydot);  // Set to zero as fallback
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
        
        // Get system size for validation
        long int N = N_VGetLength_Serial(y);
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> ydot_array;
        try {
            ydot_array = data->py_implicit_fn.value()(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python implicit function: " << e.what() << std::endl;
            return -1;
        }
        
        // Validate result before copying
        py::buffer_info ydot_buffer = ydot_array.request();
        
        // Check for size mismatch
        if (ydot_buffer.size != N) {
            std::cerr << "[ERROR] ARKODE: Output size mismatch in implicit_fn. Expected: " 
                      << N << ", Got: " << ydot_buffer.size << std::endl;
            return -1;
        }
        
        // Copy result back to N_Vector
        realtype* ydot_data = N_VGetArrayPointer_Serial(ydot);
        realtype* py_ydot_data = static_cast<realtype*>(ydot_buffer.ptr);
        
        for (long int i = 0; i < N; ++i) {
            ydot_data[i] = py_ydot_data[i];
        }
        
        return 0;  // Success
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] in ark_implicit_wrapper: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in ark_implicit_wrapper" << std::endl;
        return -1;
    }
}

// Improved dense Jacobian function wrapper 
inline int jac_dense_wrapper(realtype t, N_Vector y, N_Vector fy, 
                      SUNMatrix J, void* user_data, 
                      N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    try {
        // Get the jacobian data from user_data (which should be PyArkFnData*)
        if (user_data == nullptr) {
            std::cerr << "ERROR: user_data is NULL in jac_dense_wrapper" << std::endl;
            return -1;
        }
        
        PyArkFnData* data = static_cast<PyArkFnData*>(user_data);
        
        if (!data->has_jacobian) {
            std::cerr << "ERROR: Jacobian function not set" << std::endl;
            return -1;
        }
        
        // Get the actual system size for validation
        int N = N_VGetLength_Serial(y);
        
        // Acquire the GIL before calling Python
        py::gil_scoped_acquire gil;
        
        // Convert N_Vector to numpy array
        py::array_t<realtype> y_array = nvector_to_numpy(y);
        
        // Call Python function
        py::array_t<realtype> jac_array;
        try {
            jac_array = data->py_jacobian_fn(t, y_array);
        } catch (const std::exception& e) {
            std::cerr << "Exception in Python Jacobian function: " << e.what() << std::endl;
            return -1;
        }
        
        // Debug output
        py::buffer_info jac_buffer = jac_array.request();
        std::cout << "[C++] Jacobian buffer: ndim=" << jac_buffer.ndim 
                  << ", size=" << jac_buffer.size 
                  << ", shape=[" << (jac_buffer.ndim > 0 ? jac_buffer.shape[0] : 0);
        if (jac_buffer.ndim > 1) {
            std::cout << "," << jac_buffer.shape[1];
        }
        std::cout << "]" << std::endl;
        
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
            
            // Fill the SUNMatrix - always assume row-major (C-style) storage for numpy arrays
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    // Row-major indexing
                    int index = i * cols + j;
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
            // print the jacobian buffer
            std::cout << "jac_buffer.ndim: " << jac_buffer.ndim << std::endl;
            std::cout << "jac_buffer.size: " << jac_buffer.size << std::endl;
            std::cout << "jac_buffer.shape: " << jac_buffer.shape[0] << "x" << jac_buffer.shape[1] << std::endl;
            std::cout << "[C++] ERROR: Invalid Jacobian dimensions. Expected 2D array (" << N << "x" << N 
                      << ") or 1D array of size " << (N*N) 
                      << ", got " << jac_buffer.ndim << "D array of size " << jac_buffer.size << std::endl;
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
        py::array_t<realtype, py::array::c_style> y_array(N);
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
        std::cerr << "[EXCEPTION] in root_wrapper: " << e.what() << std::endl;
        return -1;  // Failure
    }
    catch (...) {
        std::cerr << "[UNKNOWN EXCEPTION] in root_wrapper" << std::endl;
        return -1;
    }
}

}  // namespace sundials_py