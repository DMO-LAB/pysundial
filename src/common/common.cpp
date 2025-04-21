#include "common.hpp"
#include <stdexcept>
#include <string>

namespace sundials_py {

// Define global SUNContext
SUNContext sunctx = nullptr;

// Initialize SUNDIALS context
void initialize_sundials_context() {
    if (sunctx == nullptr) {
        // Pass 0 as the first argument (SUN_COMM_NULL equivalent)
        int flag = SUNContext_Create(0, &sunctx);
        if (flag != 0) {
            throw std::runtime_error("Failed to create SUNDIALS context");
        }
    }
}

// Finalize SUNDIALS context
void finalize_sundials_context() {
    if (sunctx != nullptr) {
        SUNContext_Free(&sunctx);
        sunctx = nullptr;
    }
}

N_Vector numpy_to_nvector(py::array_t<realtype>& np_array) {
    // Ensure SUNDIALS context is initialized
    if (sunctx == nullptr) {
        initialize_sundials_context();
    }
    
    py::buffer_info buffer = np_array.request();
    
    // Check if the array is 1D
    if (buffer.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 1");
    }
    
    // Create a new N_Vector
    long int length = buffer.shape[0];
    N_Vector vec = N_VNew_Serial(length, sunctx);
    if (vec == nullptr) {
        throw std::runtime_error("Failed to create N_Vector");
    }
    
    // Copy data from numpy array to N_Vector
    realtype* vec_data = N_VGetArrayPointer_Serial(vec);
    realtype* np_data = static_cast<realtype*>(buffer.ptr);
    
    for (long int i = 0; i < length; ++i) {
        vec_data[i] = np_data[i];
    }
    
    return vec;
}

py::array_t<realtype> nvector_to_numpy(N_Vector vec) {
    // Get vector length and data
    sunindextype length = N_VGetLength_Serial(vec);
    realtype* vec_data = N_VGetArrayPointer_Serial(vec);
    
    // Create numpy array and copy data
    py::array_t<realtype> np_array(length);
    py::buffer_info buffer = np_array.request();
    realtype* np_data = static_cast<realtype*>(buffer.ptr);
    
    for (sunindextype i = 0; i < length; ++i) {
        np_data[i] = vec_data[i];
    }
    
    return np_array;
}

void check_flag(void* flagvalue, const char* funcname, int opt) {
    // Check if flagvalue is NULL
    if (opt == 0 && flagvalue == nullptr) {
        std::string msg = "SUNDIALS ERROR: " + std::string(funcname) + " returned NULL pointer";
        throw std::runtime_error(msg);
    }
    
    // Check if flagvalue is 0
    else if (opt == 1) {
        int* errflag = static_cast<int*>(flagvalue);
        if (*errflag < 0) {
            std::string msg = "SUNDIALS ERROR: " + std::string(funcname) + 
                              " failed with flag = " + std::to_string(*errflag);
            throw std::runtime_error(msg);
        }
    }
    
    // Check if flagvalue < 0
    else if (opt == 2 && *static_cast<int*>(flagvalue) < 0) {
        int* errflag = static_cast<int*>(flagvalue);
        std::string msg = "SUNDIALS ERROR: " + std::string(funcname) + 
                          " failed with flag = " + std::to_string(*errflag);
        throw std::runtime_error(msg);
    }
}

}  // namespace sundials_py