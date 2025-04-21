#include "cvode_module.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <cvode/cvode.h>
// #include <cvode/cvode_direct.h>
#include <cvode/cvode_ls.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_serial.h>

#include "../common/common.hpp"
#include "../utils/callback_wrappers.hpp"

namespace py = pybind11;
using namespace sundials_py;

// Class definition for CVodeSolver
class CVodeSolver {
private:
    // SUNDIALS memory and problem size
    void* cvode_mem_;
    int N_;  // System size
    
    // Integration parameters
    double t0_;
    
    // Solver type
    bool using_newton_iteration_;
    
    // Callback data containers
    PyRhsFnData* rhs_data_;
    PyJacFnData* jac_data_;
    PyRootFnData* root_data_;
    
    // Linear solver and matrix
    SUNMatrix A_;
    SUNLinearSolver LS_;
    
    // Vectors
    N_Vector y_;
    bool y_owner_;

public:
    // Constructor
    // Constructor
CVodeSolver(int system_size, 
            PyRhsFn rhs_fn,
            IterationType iter_type = IterationType::NEWTON,
            LinearSolverType linsol_type = LinearSolverType::DENSE) 
    : N_(system_size), 
      t0_(0.0), 
      using_newton_iteration_(iter_type == IterationType::NEWTON),
      rhs_data_(new PyRhsFnData{rhs_fn}),
      jac_data_(nullptr),
      root_data_(nullptr),
      A_(nullptr),
      LS_(nullptr),
      y_(nullptr),
      y_owner_(false) {

    // Ensure context is initialized
    if (sunctx == nullptr) {
        initialize_sundials_context();
    }

    // Create dummy initial vector
    N_Vector y0_dummy = N_VNew_Serial(N_, sunctx);
    if (!y0_dummy) {
        throw std::runtime_error("Failed to allocate dummy y0 vector");
    }

    // Initialize CVODE memory with correct method
    cvode_mem_ = CVodeCreate(using_newton_iteration_ ? CV_BDF : CV_ADAMS, sunctx);
    if (!cvode_mem_) {
        N_VDestroy_Serial(y0_dummy);
        throw std::runtime_error("Failed to create CVODE memory");
    }

    // Initialize CVODE with dummy vector
    int flag = CVodeInit(cvode_mem_, rhs_wrapper, 0.0, y0_dummy);
    if (flag != CV_SUCCESS) {
        N_VDestroy_Serial(y0_dummy);
        CVodeFree(&cvode_mem_);
        throw std::runtime_error("Failed to initialize CVODE");
    }

    // Set user data for RHS
    flag = CVodeSetUserData(cvode_mem_, rhs_data_);
    check_flag(&flag, "CVodeSetUserData", 1);

    if (using_newton_iteration_) {
        // Only Newton iteration needs a linear solver
        if (linsol_type == LinearSolverType::DENSE) {
            A_ = SUNDenseMatrix(N_, N_, sunctx);
            if (!A_) {
                throw std::runtime_error("Failed to create dense matrix");
            }

            LS_ = SUNLinSol_Dense(y0_dummy, A_, sunctx);
            if (!LS_) {
                SUNMatDestroy(A_);
                throw std::runtime_error("Failed to create dense linear solver");
            }

            flag = CVodeSetLinearSolver(cvode_mem_, LS_, A_);
            check_flag(&flag, "CVodeSetLinearSolver", 1);
        }

        // You can add more solver types here in the future
    } else {
        // Functional iteration: no linear solver, just set max nonlin iters
        flag = CVodeSetMaxNonlinIters(cvode_mem_, 25);
        check_flag(&flag, "CVodeSetMaxNonlinIters", 1);
    }

    N_VDestroy_Serial(y0_dummy);  // cleanup dummy vector
}

    
    // Destructor
    ~CVodeSolver() {
        // Free memory in reverse order of allocation
        if (y_owner_ && y_ != nullptr) {
            N_VDestroy_Serial(y_);
        }
        
        if (LS_ != nullptr) {
            SUNLinSolFree(LS_);
        }
        
        if (A_ != nullptr) {
            SUNMatDestroy(A_);
        }
        
        if (cvode_mem_ != nullptr) {
            CVodeFree(&cvode_mem_);
        }
        
        delete rhs_data_;
        delete jac_data_;
        delete root_data_;
    }
    
    // Initialize the solver with initial conditions
    void initialize(py::array_t<realtype> y0, double t0 = 0.0, 
               double rel_tol = 1.0e-6, py::array_t<realtype> abs_tol = py::array_t<realtype>()) {
        // Set initial time
        t0_ = t0;
        
        try {
            // Create and fill N_Vector for initial conditions
            N_Vector y_tmp = numpy_to_nvector(y0);
            
            // Check vector length
            if (N_VGetLength_Serial(y_tmp) != N_) {
                N_VDestroy_Serial(y_tmp);
                throw std::runtime_error("Initial condition vector length doesn't match system size");
            }
            
            // Clone the vector to keep a copy
            if (y_ != nullptr && y_owner_) {
                N_VDestroy_Serial(y_);
            }
            
            y_ = N_VClone(y_tmp);
            if (y_ == nullptr) {
                N_VDestroy_Serial(y_tmp);
                throw std::runtime_error("Failed to clone y vector");
            }
            
            // Copy data from temporary vector to y_
            realtype* dest = N_VGetArrayPointer_Serial(y_);
            realtype* src = N_VGetArrayPointer_Serial(y_tmp);
            
            for (int i = 0; i < N_; ++i) {
                dest[i] = src[i];
            }
            
            N_VDestroy_Serial(y_tmp);  // Clean up temporary
            y_owner_ = true;
            
            int flag;
            
            // Reinitialize the solver
            flag = CVodeReInit(cvode_mem_, t0_, y_);
            check_flag(&flag, "CVodeReInit", 1);
            
            // Set tolerances
            if (abs_tol.size() > 0) {
                // Vector absolute tolerance
                N_Vector atol_vec = numpy_to_nvector(abs_tol);
                flag = CVodeSVtolerances(cvode_mem_, rel_tol, atol_vec);
                check_flag(&flag, "CVodeSVtolerances", 1);
                N_VDestroy_Serial(atol_vec);
            } else {
                // Scalar absolute tolerance
                flag = CVodeSStolerances(cvode_mem_, rel_tol, rel_tol * 1.0e-3);
                check_flag(&flag, "CVodeSStolerances", 1);
            }
            
            // Make sure user data is properly set
            flag = CVodeSetUserData(cvode_mem_, rhs_data_);
            check_flag(&flag, "CVodeSetUserData in initialize", 1);
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error initializing solver: ") + e.what());
        }
    }
    
    // Set the Jacobian function
    void set_jacobian(PyJacFn jac_fn) {
        if (A_ == nullptr || LS_ == nullptr) {
            throw std::runtime_error("Cannot set Jacobian: No linear solver attached");
        }
        
        if (!using_newton_iteration_) {
            throw std::runtime_error("Cannot set Jacobian: Not using Newton iteration");
        }
        
        // Create or update Jacobian data
        if (jac_data_ == nullptr) {
            jac_data_ = new PyJacFnData{jac_fn, N_};
        } else {
            jac_data_->py_jac_fn = jac_fn;
        }
        
        // Set Jacobian function in CVODE
        int flag = CVodeSetJacFn(cvode_mem_, jac_dense_wrapper);
        check_flag(&flag, "CVodeSetJacFn", 1);
        
        // Set user data (note: this will override previous user data)
        flag = CVodeSetUserData(cvode_mem_, jac_data_);
        check_flag(&flag, "CVodeSetUserData", 1);
    }
    
    // Set root finding function
    void set_root_function(PyRootFn root_fn, int nrtfn) {
        // Create or update root data
        if (root_data_ == nullptr) {
            root_data_ = new PyRootFnData{root_fn, nrtfn};
        } else {
            root_data_->py_root_fn = root_fn;
            root_data_->nrtfn = nrtfn;
        }
        
        // Set root function in CVODE
        int flag = CVodeRootInit(cvode_mem_, nrtfn, root_wrapper);
        check_flag(&flag, "CVodeRootInit", 1);
        
        // Set user data
        flag = CVodeSetUserData(cvode_mem_, root_data_);
        check_flag(&flag, "CVodeSetUserData", 1);
    }
    
    // Solve to a specific time point
    py::array_t<realtype> solve_to(double tout) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        // std::cout << "Starting solve_to with t0=" << t0_ << ", tout=" << tout 
        //         << ", method=" << (using_newton_iteration_ ? "BDF/Newton" : "ADAMS/Functional") << std::endl;
        
        realtype t = t0_;
        int flag = CVode(cvode_mem_, tout, y_, &t, CV_NORMAL);
        
        if (flag < 0) {
            std::cerr << "CVODE solver error code: " << flag << std::endl;
            throw std::runtime_error("CVODE solver error: " + std::to_string(flag));
        }
        
        // std::cout << "Integration completed successfully to t=" << t << std::endl;
        
        // Convert result to numpy array
        return nvector_to_numpy(y_);
    }
    
    // Solve for multiple time points
    py::array_t<realtype> solve(py::array_t<realtype> time_points) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        // Get time points
        py::buffer_info time_buffer = time_points.request();
        int num_times = time_buffer.shape[0];
        double* times = static_cast<double*>(time_buffer.ptr);
        
        // Create output array [num_times x N_]
        std::vector<ssize_t> shape = {num_times, N_};
        py::array_t<realtype> results(shape);
        py::buffer_info results_buffer = results.request();
        realtype* results_data = static_cast<realtype*>(results_buffer.ptr);
        
        // Integrate and store results
        realtype t = t0_;
        
        for (int i = 0; i < num_times; ++i) {
            double tout = times[i];
            
            int flag = CVode(cvode_mem_, tout, y_, &t, CV_NORMAL);
            
            if (flag < 0) {
                throw std::runtime_error("CVODE solver error at step " + std::to_string(i) + 
                                        ": " + std::to_string(flag));
            }
            
            // Copy current solution to results array
            realtype* y_data = N_VGetArrayPointer_Serial(y_);
            
            for (int j = 0; j < N_; ++j) {
                results_data[i * N_ + j] = y_data[j];
            }
        }
        
        return results;
    }
    
    // Get solver statistics
    py::dict get_stats() {
        py::dict stats;
        
        long int nsteps, nfevals, nlinsetups, netfails;
        int flag;
        
        flag = CVodeGetNumSteps(cvode_mem_, &nsteps);
        if (flag == 0) {
            stats["num_steps"] = nsteps;
        }
        
        flag = CVodeGetNumRhsEvals(cvode_mem_, &nfevals);
        if (flag == 0) {
            stats["num_rhs_evals"] = nfevals;
        }
        
        if (using_newton_iteration_) {
            flag = CVodeGetNumLinSolvSetups(cvode_mem_, &nlinsetups);
            if (flag == 0) {
                stats["num_lin_setups"] = nlinsetups;
            }
        }
        
        flag = CVodeGetNumErrTestFails(cvode_mem_, &netfails);
        if (flag == 0) {
            stats["num_error_test_fails"] = netfails;
        }
        
        return stats;
    }
    
    // Get the last step size used by the solver
    double get_last_step() {
        realtype hlast;
        int flag = CVodeGetLastStep(cvode_mem_, &hlast);
        check_flag(&flag, "CVodeGetLastStep", 1);
        return static_cast<double>(hlast);
    }
    
    // Get the current integration time
    double get_current_time() {
        realtype tcur;
        int flag = CVodeGetCurrentTime(cvode_mem_, &tcur);
        check_flag(&flag, "CVodeGetCurrentTime", 1);
        return static_cast<double>(tcur);
    }
};

// Module initialization function
void init_cvode_module(py::module_ &m) {
    // Register IterationType enum
    py::enum_<IterationType>(m, "IterationType")
        .value("FUNCTIONAL", IterationType::FUNCTIONAL)
        .value("NEWTON", IterationType::NEWTON)
        .export_values();
    
    // Register LinearSolverType enum
    py::enum_<LinearSolverType>(m, "LinearSolverType")
        .value("DENSE", LinearSolverType::DENSE)
        .value("BAND", LinearSolverType::BAND)
        .value("SPARSE", LinearSolverType::SPARSE)
        .value("SPGMR", LinearSolverType::SPGMR)
        .value("SPBCG", LinearSolverType::SPBCG)
        .value("SPTFQMR", LinearSolverType::SPTFQMR)
        .value("PCG", LinearSolverType::PCG)
        .export_values();
    
    // Register CVodeSolver class
    py::class_<CVodeSolver>(m, "CVodeSolver")
        .def(py::init<int, PyRhsFn, IterationType, LinearSolverType>(),
             py::arg("system_size"),
             py::arg("rhs_fn"),
             py::arg("iter_type") = IterationType::NEWTON,
             py::arg("linsol_type") = LinearSolverType::DENSE,
             "Create a CVODE solver for an ODE system")
        .def("initialize", &CVodeSolver::initialize,
             py::arg("y0"),
             py::arg("t0") = 0.0,
             py::arg("rel_tol") = 1.0e-6,
             py::arg("abs_tol") = py::array_t<realtype>(),
             "Initialize the solver with initial conditions")
        .def("set_jacobian", &CVodeSolver::set_jacobian,
             py::arg("jac_fn"),
             "Set the Jacobian function for implicit solves")
        .def("set_root_function", &CVodeSolver::set_root_function,
             py::arg("root_fn"),
             py::arg("nrtfn"),
             "Set the root finding function")
        .def("solve_to", &CVodeSolver::solve_to,
             py::arg("tout"),
             "Solve the ODE system to a specific time point")
        .def("solve_single", &CVodeSolver::solve_to)
        .def("solve", &CVodeSolver::solve,
             py::arg("time_points"),
             "Solve the ODE system for multiple time points")
        .def("get_stats", &CVodeSolver::get_stats,
             "Get solver statistics")
        .def("get_last_step", &CVodeSolver::get_last_step,
             "Get the last step size used by the solver")
        .def("get_current_time", &CVodeSolver::get_current_time,
             "Get the current integration time")
        .def("__repr__",
             [](const CVodeSolver &solver) {
                 return "<CVodeSolver>";
             });
    
    // Add module information about the API version
    m.attr("__api_version__") = "CVODE API (SUNDIALS 7.3.0+)";
        // For backward compatibility, keep original function names but map to new ones
        
}