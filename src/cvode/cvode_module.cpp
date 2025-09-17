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
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

#include "../common/common.hpp"
#include "../utils/callback_wrappers.hpp"
#include <iostream>

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
    
    // Unified callback/user data container (single pointer passed to CVODE)
    CvodeUserData* user_data_;
    
    // Linear solver and matrix
    SUNMatrix A_;
    SUNLinearSolver LS_;
    
    // Vectors
    N_Vector y_;
    bool y_owner_;

    SUNNonlinearSolver NLS_;

    int mxsteps_;

public:
    // Constructor
CVodeSolver(int system_size, 
        PyRhsFn rhs_fn,
        IterationType iter_type = IterationType::NEWTON,
        LinearSolverType linsol_type = LinearSolverType::DENSE,
        bool use_bdf = true,
        int mxsteps = 1000)  // Add parameter to choose BDF vs Adams
    : N_(system_size), 
    t0_(0.0), 
    using_newton_iteration_(iter_type == IterationType::NEWTON),
    user_data_(nullptr),
    A_(nullptr),
    LS_(nullptr),
    y_(nullptr),
    y_owner_(false),
    NLS_(nullptr),
    mxsteps_(mxsteps) {
    // std::cout << "[DEBUG] CVodeSolver constructor start. system_size=" << system_size << ", iter_type=" << (using_newton_iteration_ ? "NEWTON" : "FUNCTIONAL") << ", use_bdf=" << use_bdf << std::endl;

    // Ensure context is initialized
    if (sunctx == nullptr) {
    initialize_sundials_context();
    }

    // Create dummy initial vector
    N_Vector y0_dummy = N_VNew_Serial(N_, sunctx);
    if (!y0_dummy) {
    throw std::runtime_error("Failed to allocate dummy y0 vector");
    }

    // Initialize CVODE memory with LINEAR MULTISTEP METHOD (not iteration type)
    // BDF is typically used with Newton iteration
    // Adams is typically used with functional iteration, but can use Newton too
    int lmm = use_bdf ? CV_BDF : CV_ADAMS;
    cvode_mem_ = CVodeCreate(lmm, sunctx);

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

    // Create unified user data and set once
    user_data_ = new CvodeUserData();
    user_data_->py_rhs_fn = rhs_fn;
    user_data_->has_jacobian = false;
    user_data_->has_root = false;
    user_data_->nrtfn = 0;
    user_data_->N = N_;
    flag = CVodeSetUserData(cvode_mem_, user_data_);
    check_flag(&flag, "CVodeSetUserData", 1);

    // Set up linear solver based on iteration type
    if (using_newton_iteration_) {
    // Newton iteration requires a linear solver
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
    } else {
    // Functional iteration: use fixed-point nonlinear solver
    NLS_ = SUNNonlinSol_FixedPoint(y0_dummy, 1, sunctx);
    if (!NLS_) {
        N_VDestroy_Serial(y0_dummy);
        throw std::runtime_error("Failed to create fixed-point nonlinear solver");
    }
    flag = CVodeSetNonlinearSolver(cvode_mem_, NLS_);
    check_flag(&flag, "CVodeSetNonlinearSolver", 1);

    flag = CVodeSetMaxNumSteps(cvode_mem_, mxsteps_);
    check_flag(&flag, "CVodeSetMaxNumSteps", 1);

    // Set maximum number of nonlinear iterations
    flag = CVodeSetMaxNonlinIters(cvode_mem_, 25);
    check_flag(&flag, "CVodeSetMaxNonlinIters", 1);

    // For Adams with functional iteration, you might want to adjust other parameters
    if (!use_bdf) {
        // Set maximum order for Adams method (default is 12, you might want lower)
        flag = CVodeSetMaxOrd(cvode_mem_, 5);
        check_flag(&flag, "CVodeSetMaxOrd", 1);
    }
    }

    N_VDestroy_Serial(y0_dummy);  // cleanup dummy vector
    //std::cout << "[DEBUG] CVodeSolver constructor end." << std::endl;
    }

    
    // Destructor
    ~CVodeSolver() {
        // std::cout << "[DEBUG] CVodeSolver destructor start." << std::endl;
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
        
        if (NLS_ != nullptr) {
            SUNNonlinSolFree(NLS_);
        }

        if (cvode_mem_ != nullptr) {
            CVodeFree(&cvode_mem_);
        }
        
    delete user_data_;
        // std::cout << "[DEBUG] CVodeSolver destructor end." << std::endl;
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
        // Scalar absolute tolerance - use a more reasonable default
        realtype scalar_atol = rel_tol * 1.0e-6;  // Better default than 1e-3
        flag = CVodeSStolerances(cvode_mem_, rel_tol, scalar_atol);
        check_flag(&flag, "CVodeSStolerances", 1);
    }
    
    // IMPORTANT: Set maximum number of steps (this was missing!)
    flag = CVodeSetMaxNumSteps(cvode_mem_, mxsteps_);
    check_flag(&flag, "CVodeSetMaxNumSteps", 1);
    
    // Set maximum step size if needed for stiff problems
    // flag = CVodeSetMaxStep(cvode_mem_, 1.0e-3);  // Uncomment if needed
    // check_flag(&flag, "CVodeSetMaxStep", 1);
    
    // Set minimum step size if needed
    flag = CVodeSetMinStep(cvode_mem_, 1.0e-16);  // Uncomment if needed
    check_flag(&flag, "CVodeSetMinStep", 1);
    
    // For stiff problems, you might want to set initial step size
    flag = CVodeSetInitStep(cvode_mem_, 1.0e-12);  // Very small initial step
    check_flag(&flag, "CVodeSetInitStep", 1);
    
    // Set stability limit detection (can help with stiff problems)
    flag = CVodeSetStabLimDet(cvode_mem_, SUNTRUE);
    check_flag(&flag, "CVodeSetStabLimDet", 1);
    
    // For Newton iteration, set additional parameters
    if (using_newton_iteration_) {
        // Set maximum number of nonlinear iterations
        flag = CVodeSetMaxNonlinIters(cvode_mem_, 10);  // Increased from default 3
        check_flag(&flag, "CVodeSetMaxNonlinIters", 1);
        
        // Set maximum number of convergence failures
        flag = CVodeSetMaxConvFails(cvode_mem_, 20);  // Increased from default 10
        check_flag(&flag, "CVodeSetMaxConvFails", 1);
        
        // Set nonlinear convergence coefficient
        flag = CVodeSetNonlinConvCoef(cvode_mem_, 0.1);  // Default is 0.1
        check_flag(&flag, "CVodeSetNonlinConvCoef", 1);
    } else {
        // For functional iteration, set maximum number of iterations
        flag = CVodeSetMaxNonlinIters(cvode_mem_, 25);  // Higher for functional
        check_flag(&flag, "CVodeSetMaxNonlinIters", 1);
    }
    
    // Make sure user data is properly set
    flag = CVodeSetUserData(cvode_mem_, user_data_);
    check_flag(&flag, "CVodeSetUserData in initialize", 1);
    
    // // Optional: Set error handler for better debugging
    // flag = CVodeSetErrHandlerFn(cvode_mem_, NULL, NULL);  // Use default error handler
    // check_flag(&flag, "CVodeSetErrHandlerFn", 1);
    
    } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Error initializing solver: ") + e.what());
    }
    }

    void setState(const std::vector<realtype>& y_new, double t_new = 0.0) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized. Call initialize() first.");
        }
        
        if (y_new.size() != static_cast<size_t>(N_)) {
            throw std::runtime_error("State vector length doesn't match system size");
        }
        
        // Update internal time
        t0_ = t_new;
        
        // Update state vector
        realtype* y_data = N_VGetArrayPointer_Serial(y_);
        for (int i = 0; i < N_; ++i) {
            y_data[i] = y_new[i];
        }
        
        // Reset CVODE's internal state
        int flag = CVodeReInit(cvode_mem_, t0_, y_);
        check_flag(&flag, "CVodeReInit in setState", 1);
    }
    

    // Set the Jacobian function
    void set_jacobian(PyJacFn jac_fn) {
        if (A_ == nullptr || LS_ == nullptr) {
            throw std::runtime_error("Cannot set Jacobian: No linear solver attached");
        }
        
        if (!using_newton_iteration_) {
            throw std::runtime_error("Cannot set Jacobian: Not using Newton iteration");
        }
        
        // Update unified user data and attach CVODE-specific Jacobian wrapper
        user_data_->py_jac_fn = jac_fn;
        user_data_->has_jacobian = true;
        int flag = CVodeSetJacFn(cvode_mem_, cvode_jac_dense_wrapper);
        check_flag(&flag, "CVodeSetJacFn", 1);
    }
    
    // Set root finding function
    void set_root_function(PyRootFn root_fn, int nrtfn) {
        // Update unified user data and set CVODE root wrapper
        user_data_->py_root_fn = root_fn;
        user_data_->has_root = true;
        user_data_->nrtfn = nrtfn;
        int flag = CVodeRootInit(cvode_mem_, nrtfn, cvode_root_wrapper);
        check_flag(&flag, "CVodeRootInit", 1);
    }
    
    // Solve to a specific time point
    py::array_t<realtype> solve_to(double tout) {
        // std::cout << "[DEBUG] solve_to() start. tout=" << tout << std::endl;
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        // std::cout << "Starting solve_to with t0=" << t0_ << ", tout=" << tout 
        //         << ", method=" << (using_newton_iteration_ ? "BDF/Newton" : "ADAMS/Functional") << std::endl;
        
        realtype t = t0_;
        int flag = CVode(cvode_mem_, tout, y_, &t, CV_NORMAL);
        // std::cout << "[DEBUG] solve_to() after CVode call. flag=" << flag << std::endl;
        
        if (flag < 0) {
            std::cerr << "CVODE solver error code: " << flag << std::endl;
            throw std::runtime_error("CVODE solver error: " + std::to_string(flag));
        }
        
        // std::cout << "Integration completed successfully to t=" << t << std::endl;
        
        // Convert result to numpy array
        // std::cout << "[DEBUG] solve_to() end." << std::endl;
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
    
    // Register CVodeSolver class with updated constructor
    py::class_<CVodeSolver>(m, "CVodeSolver")
        .def(py::init<int, PyRhsFn, IterationType, LinearSolverType, bool, int>(),
             py::arg("system_size"),
             py::arg("rhs_fn"),
             py::arg("iter_type") = IterationType::NEWTON,
             py::arg("linsol_type") = LinearSolverType::DENSE,
             py::arg("use_bdf") = true,  // Add this parameter
             py::arg("mxsteps") = 1000,
             "Create a CVODE solver for an ODE system")
        .def("initialize", &CVodeSolver::initialize,
             py::arg("y0"),
             py::arg("t0") = 0.0,
             py::arg("rel_tol") = 1.0e-6,
             py::arg("abs_tol") = py::array_t<realtype>(),
             "Initialize the solver with initial conditions")
        .def("set_state", &CVodeSolver::setState,
             py::arg("y_new"),
             py::arg("t_new") = 0.0,
             "Set the state of the solver")
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
    
    m.attr("__api_version__") = "CVODE API (SUNDIALS 7.3.0+)";
}