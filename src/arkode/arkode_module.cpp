#include "arkode_module.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <arkode/arkode.h>
#include <arkode/arkode_butcher.h>
#include <arkode/arkode_butcher_dirk.h>
#include <arkode/arkode_butcher_erk.h>
#include <arkode/arkode_ls.h>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_arkstep.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_serial.h>
#include <optional>
#include <iostream>

#include "../common/common.hpp"
#include "../utils/callback_wrappers.hpp"

namespace py = pybind11;

namespace sundials_py {

// ARKODE Solver Class
class ARKodeSolver {
private:
    // SUNDIALS memory and problem size
    void* arkode_mem_;
    int N_;  // System size
    
    // Integration parameters
    double t0_;
    double tcur_;  // Current integration time
    
    // Single unified callback data container
    PyArkFnData* ark_data_;
    PyRootFnData* root_data_;  // Keep separate for root finding
    
    // Linear solver and matrix
    SUNMatrix A_;
    SUNLinearSolver LS_;
    
    // Vectors
    N_Vector y_;
    bool y_owner_;
    
    // Butcher table information
    ButcherTable butcher_table_;
    
    // Flag to determine which solver we're using
    bool using_arkstep_;

public:
    // Constructor
    ARKodeSolver(int system_size, 
            PyRhsFn explicit_fn, 
            std::optional<PyRhsFn> implicit_fn = std::nullopt,
            ButcherTable butcher_table = ButcherTable::ARK436L2SA_ERK_6_3_4,
            LinearSolverType linsol_type = LinearSolverType::DENSE) 
        : N_(system_size), 
        t0_(0.0),
        tcur_(0.0),
        ark_data_(new PyArkFnData{explicit_fn, implicit_fn, nullptr, false, system_size}),
        root_data_(nullptr),
        A_(nullptr),
        LS_(nullptr),
        y_(nullptr),
        y_owner_(false),
        butcher_table_(butcher_table) {
        
        // Ensure context is initialized
        if (sunctx == nullptr) {
            initialize_sundials_context();
        }
        
        // Determine which solver to use (ARKStep or ERKStep)
        using_arkstep_ = (implicit_fn.has_value() || is_implicit_method(butcher_table) || 
                         is_imex_pair(butcher_table));
        
        // Create a temporary vector for initialization
        N_Vector y0_dummy = N_VNew_Serial(system_size, sunctx);
        if (!y0_dummy) {
            throw std::runtime_error("Failed to allocate dummy y0 vector");
        }
        
        if (using_arkstep_) {
            // Use ARKStep for problems with implicit components
            arkode_mem_ = ARKStepCreate(
                ark_explicit_wrapper, ark_implicit_wrapper, 
                t0_, y0_dummy, sunctx);
        } else {
            // Use ERKStep for purely explicit problems
            arkode_mem_ = ERKStepCreate(
                ark_explicit_wrapper, t0_, y0_dummy, sunctx);
        }
        
        N_VDestroy_Serial(y0_dummy);  // Clean up temporary vector
        
        if (arkode_mem_ == nullptr) {
            throw std::runtime_error("Failed to create ARKODE memory");
        }
        
        // Set user data right away to ensure it's not null when functions are called
        int flag = ARKodeSetUserData(arkode_mem_, ark_data_);
        check_flag(&flag, "ARKodeSetUserData", 1);
        
        // Create matrix and linear solver if needed for implicit methods
        if (using_arkstep_ && (implicit_fn.has_value() || is_implicit_method(butcher_table))) {
            if (linsol_type == LinearSolverType::DENSE) {
                A_ = SUNDenseMatrix(N_, N_, sunctx);
                if (A_ == nullptr) {
                    throw std::runtime_error("Failed to create dense matrix");
                }

                LS_ = SUNLinSol_Dense(nullptr, A_, sunctx);
                if (LS_ == nullptr) {
                    SUNMatDestroy(A_);
                    throw std::runtime_error("Failed to create dense linear solver");
                }
                
                // Use the new unified ARKODE functions instead of deprecated ARKStep functions
                int flag = ARKodeSetLinearSolver(arkode_mem_, LS_, A_);
                check_flag(&flag, "ARKodeSetLinearSolver", 1);

                // Set ARKODE to use its internal difference quotient Jacobian by default
                flag = ARKodeSetJacFn(arkode_mem_, NULL);
                check_flag(&flag, "ARKodeSetJacFn to NULL", 1);
            }
        }
    }
    
    // Destructor
    ~ARKodeSolver() {
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
        
        if (arkode_mem_ != nullptr) {
            ARKodeFree(&arkode_mem_);
        }
        
        delete ark_data_;
        delete root_data_;
    }
    
    // Initialize the solver with initial conditions
    void initialize(py::array_t<realtype> y0, double t0 = 0.0, 
                double rel_tol = 1.0e-6, py::array_t<realtype> abs_tol = py::array_t<realtype>()) {
        // Set initial time
        t0_ = t0;
        tcur_ = t0;
        
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
            
            // Reset the solver with initial conditions
            if (using_arkstep_) {
                flag = ARKStepReInit(arkode_mem_, ark_explicit_wrapper, ark_implicit_wrapper, t0_, y_);
                check_flag(&flag, "ARKStepReInit", 1);
            } else {
                flag = ERKStepReInit(arkode_mem_, ark_explicit_wrapper, t0_, y_);
                check_flag(&flag, "ERKStepReInit", 1);
            }
            
            // Set tolerances
            if (abs_tol.size() > 0) {
                N_Vector abs_tol_vec = numpy_to_nvector(abs_tol);
                flag = ARKodeSVtolerances(arkode_mem_, rel_tol, abs_tol_vec);
                N_VDestroy_Serial(abs_tol_vec);
            } else {
                flag = ARKodeSStolerances(arkode_mem_, rel_tol, 1e-8);
            }
            check_flag(&flag, "ARKode tolerances", 1);
            
            // Make sure user data is properly set after reinitialization
            flag = ARKodeSetUserData(arkode_mem_, ark_data_);
            check_flag(&flag, "ARKodeSetUserData in initialize", 1);
            
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error initializing solver: ") + e.what());
        }
    }

    void set_jacobian(PyJacFn jac_fn) {
        if (!using_arkstep_) {
            throw std::runtime_error("Cannot set Jacobian: Not using ARKStep");
        }
        
        if (A_ == nullptr || LS_ == nullptr) {
            throw std::runtime_error("Cannot set Jacobian: No linear solver attached");
        }

        // Update the ark_data_ structure with the Jacobian function
        ark_data_->py_jacobian_fn = jac_fn;
        ark_data_->has_jacobian = true;
        ark_data_->N = N_;  // Make sure N is properly set
        
        // Set Jacobian function in ARKODE - use the unified function
        int flag = ARKodeSetJacFn(arkode_mem_, jac_dense_wrapper);
        check_flag(&flag, "ARKodeSetJacFn", 1);
        
        // std::cout << "Jacobian function set successfully with user_data=" << ark_data_ << std::endl;
    }
        
    // Set root finding function
    void set_root_function(PyRootFn root_fn, int nrtfn) {
        // Attach root function to the main user_data and initialize ARKODE roots
        ark_data_->py_root_fn = root_fn;
        ark_data_->has_root = true;
        ark_data_->nrtfn = nrtfn;

        int flag = ARKodeRootInit(arkode_mem_, nrtfn, root_wrapper);
        check_flag(&flag, "ARKodeRootInit", 1);
    }
    
    // Set adaptive step size parameters (using modern SUNDIALS approach)
    void set_adaptive_params(double adapt_params[3]) {
        // Note: The old ARKStepSetAdaptivityMethod is deprecated
        // For now, we'll skip this or implement with the new SUNAdaptController
        std::cout << "Warning: Adaptive parameters setting skipped - use SUNAdaptController for modern SUNDIALS" << std::endl;
    }
    
    // Set fixed step size
    void set_fixed_step_size(double step_size) {
        int flag = ARKodeSetFixedStep(arkode_mem_, step_size);
        check_flag(&flag, "ARKodeSetFixedStep", 1);
    }

    // Set maximum number of steps
    void set_max_num_steps(long int mxsteps) {
        int flag = ARKodeSetMaxNumSteps(arkode_mem_, mxsteps);
        check_flag(&flag, "ARKodeSetMaxNumSteps", 1);
    }
    
    // Get the current solution
    py::array_t<realtype> get_current_solution() {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        return nvector_to_numpy(y_);
    }
    
    // Get the current time
    double get_current_time() {
        return tcur_;
    }
    
    // Integrate to a specified time
    int integrate_to_time(double tout) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        int flag = ARKodeEvolve(arkode_mem_, tout, y_, &tcur_, ARK_NORMAL);
        
        if (flag < 0) {
            throw std::runtime_error("ARKODE solver error: " + std::to_string(flag));
        }
        
        return flag;
    }
    
    // Take a single step toward the target time
    int advance_one_step(double tout) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        int flag = ARKodeEvolve(arkode_mem_, tout, y_, &tcur_, ARK_ONE_STEP);
        
        if (flag < 0) {
            throw std::runtime_error("ARKODE solver error: " + std::to_string(flag));
        }
        
        return flag;
    }
    
    // Solve to a specific time and return the solution
    py::array_t<realtype> solve_to(double tout) {
        if (y_ == nullptr) {
            throw std::runtime_error("Solver not initialized with initial conditions");
        }
        
        integrate_to_time(tout);
        
        // Convert result to numpy array
        return get_current_solution();
    }
    
    // Solve for multiple time points
    py::array_t<realtype> solve_sequence(py::array_t<realtype> time_points) {
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
        for (int i = 0; i < num_times; ++i) {
            double tout = times[i];
            
            try {
                integrate_to_time(tout);
                
                // Copy current solution to results array
                realtype* y_data = N_VGetArrayPointer_Serial(y_);
                if (y_data == nullptr) {
                    throw std::runtime_error("y_ data pointer became null during evolution");
                }
                
                // Additional bounds checking
                if (i * N_ + N_ > results_buffer.size) {
                    throw std::runtime_error("Results array bounds error");
                }
                
                for (int j = 0; j < N_; ++j) {
                    results_data[i * N_ + j] = y_data[j];
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error in step " << i << ": " << e.what() << std::endl;
                throw;
            }
        }
    
        return results;
    }
    
    // Get solver statistics
    py::dict get_stats() {
        py::dict stats;
        
        stats["solver_type"] = using_arkstep_ ? "ARKStep" : "ERKStep";
        
        try {
            long int nsteps = 0;
            
            int flag = ARKodeGetNumSteps(arkode_mem_, &nsteps);
            if (flag == 0) {
                stats["num_steps"] = nsteps;
            } else {
                stats["num_steps"] = "Error retrieving";
            }
            
            // Get RHS evaluations
            try {
                long int nfevals_explicit = 0, nfevals_implicit = 0;
                if (using_arkstep_) {
                    int flag_exp = ARKodeGetNumRhsEvals(arkode_mem_, 0, &nfevals_explicit); // explicit
                    int flag_imp = ARKodeGetNumRhsEvals(arkode_mem_, 1, &nfevals_implicit); // implicit
                    if (flag_exp == 0) {
                        stats["num_rhs_evals_explicit"] = nfevals_explicit;
                    } else {
                        stats["num_rhs_evals_explicit"] = "Error retrieving";
                    }
                    if (flag_imp == 0) {
                        stats["num_rhs_evals_implicit"] = nfevals_implicit;
                    } else {
                        stats["num_rhs_evals_implicit"] = "Error retrieving";
                    }
                } else {
                    int flag_exp = ARKodeGetNumRhsEvals(arkode_mem_, 0, &nfevals_explicit);
                    if (flag_exp == 0) {
                        stats["num_rhs_evals_explicit"] = nfevals_explicit;
                    } else {
                        stats["num_rhs_evals_explicit"] = "Error retrieving";
                    }
                    stats["num_rhs_evals_implicit"] = 0;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception getting RHS evals: " << e.what() << std::endl;
                stats["num_rhs_evals_note"] = "Error retrieving RHS evaluations";
            }
            
            // Get linear solver setups if we're using ARKStep
            try {
                if (using_arkstep_) {
                    long int nlinsetups = 0;
                    flag = ARKodeGetNumLinSolvSetups(arkode_mem_, &nlinsetups);
                    if (flag == 0) {
                        stats["num_lin_setups"] = nlinsetups;
                    } else {
                        stats["num_lin_setups"] = "Error retrieving";
                    }
                } else {
                    stats["num_lin_setups"] = 0;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception getting lin setups: " << e.what() << std::endl;
                stats["num_lin_setups"] = "Exception";
            }
            
            // Get error test failures
            try {
                long int netfails = 0;
                flag = ARKodeGetNumErrTestFails(arkode_mem_, &netfails);
                
                if (flag == 0) {
                    stats["num_error_test_fails"] = netfails;
                } else {
                    stats["num_error_test_fails"] = "Error retrieving";
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception getting error test fails: " << e.what() << std::endl;
                stats["num_error_test_fails"] = "Exception";
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in get_stats: " << e.what() << std::endl;
            stats["error"] = "Exception retrieving statistics";
        }
        
        return stats;
    }
    
    // Get information about the Butcher table being used
    py::dict get_butcher_info() {
        py::dict info;
        
        info["name"] = py::cast(static_cast<int>(butcher_table_));
        info["description"] = get_butcher_table_description(butcher_table_);
        info["is_explicit"] = is_explicit_method(butcher_table_);
        info["is_implicit"] = is_implicit_method(butcher_table_);
        info["is_imex_pair"] = is_imex_pair(butcher_table_);
        
        return info;
    }
    
    // Get the last step size used by the solver
    double get_last_step() {
        realtype hlast;
        
        int flag = ARKodeGetLastStep(arkode_mem_, &hlast);
        check_flag(&flag, "ARKodeGetLastStep", 1);
        return static_cast<double>(hlast);
    }
};

// Module initialization function
void init_arkode_module(py::module_ &m) {
    // Register Butcher table enum
    py::enum_<ButcherTable>(m, "ButcherTable")
        // Explicit methods
        .value("HEUN_EULER_2_1_2", ButcherTable::HEUN_EULER_2_1_2)
        .value("BOGACKI_SHAMPINE_4_2_3", ButcherTable::BOGACKI_SHAMPINE_4_2_3)
        .value("ARK324L2SA_ERK_4_2_3", ButcherTable::ARK324L2SA_ERK_4_2_3)
        .value("ZONNEVELD_5_3_4", ButcherTable::ZONNEVELD_5_3_4)
        .value("ARK436L2SA_ERK_6_3_4", ButcherTable::ARK436L2SA_ERK_6_3_4)
        .value("ARK437L2SA_ERK_7_3_4", ButcherTable::ARK437L2SA_ERK_7_3_4)
        .value("ARK548L2SA_ERK_8_4_5", ButcherTable::ARK548L2SA_ERK_8_4_5)
        .value("VERNER_8_5_6", ButcherTable::VERNER_8_5_6)
        .value("FEHLBERG_13_7_8", ButcherTable::FEHLBERG_13_7_8)
        
        // Implicit methods
        .value("SDIRK_2_1_2", ButcherTable::SDIRK_2_1_2)
        .value("BILLINGTON_3_3_2", ButcherTable::BILLINGTON_3_3_2)
        .value("TRBDF2_3_3_2", ButcherTable::TRBDF2_3_3_2)
        .value("KVAERNO_4_2_3", ButcherTable::KVAERNO_4_2_3)
        .value("ARK324L2SA_DIRK_4_2_3", ButcherTable::ARK324L2SA_DIRK_4_2_3)
        .value("CASH_5_2_4", ButcherTable::CASH_5_2_4)
        .value("CASH_5_3_4", ButcherTable::CASH_5_3_4)
        .value("SDIRK_5_3_4", ButcherTable::SDIRK_5_3_4)
        .value("ARK436L2SA_DIRK_6_3_4", ButcherTable::ARK436L2SA_DIRK_6_3_4)
        .value("ARK437L2SA_DIRK_7_3_4", ButcherTable::ARK437L2SA_DIRK_7_3_4)
        .value("KVAERNO_7_4_5", ButcherTable::KVAERNO_7_4_5)
        .value("ARK548L2SA_DIRK_8_4_5", ButcherTable::ARK548L2SA_DIRK_8_4_5)
        
        // ImEx pairs
        .value("ARK324L2SA_ERK_4_2_3_DIRK_4_2_3", ButcherTable::ARK324L2SA_ERK_4_2_3_DIRK_4_2_3)
        .value("ARK436L2SA_ERK_6_3_4_DIRK_6_3_4", ButcherTable::ARK436L2SA_ERK_6_3_4_DIRK_6_3_4)
        .value("ARK437L2SA_ERK_7_3_4_DIRK_7_3_4", ButcherTable::ARK437L2SA_ERK_7_3_4_DIRK_7_3_4)
        .value("ARK548L2SA_ERK_8_4_5_DIRK_8_4_5", ButcherTable::ARK548L2SA_ERK_8_4_5_DIRK_8_4_5)
        .export_values();

    
    // Register ARKodeSolver class
    py::class_<ARKodeSolver>(m, "ARKodeSolver", py::dynamic_attr())
        .def(py::init<int, PyRhsFn, std::optional<PyRhsFn>, ButcherTable, LinearSolverType>(),
             py::arg("system_size"),
             py::arg("explicit_fn"),
             py::arg("implicit_fn") = std::nullopt,
             py::arg("butcher_table") = ButcherTable::ARK436L2SA_ERK_6_3_4,
             py::arg("linsol_type") = LinearSolverType::DENSE,
             "Create an ARKODE solver for an ODE system with explicit and optional implicit parts")

        .def("initialize", &ARKodeSolver::initialize,
             py::arg("y0"),
             py::arg("t0") = 0.0,
             py::arg("rel_tol") = 1.0e-6,
             py::arg("abs_tol") = py::array_t<realtype>(),
             "Initialize the solver with initial conditions")
        .def("set_jacobian", &ARKodeSolver::set_jacobian,
             py::arg("jac_fn"),
             "Set the Jacobian function for implicit solves")
        .def("set_root_function", &ARKodeSolver::set_root_function,
             py::arg("root_fn"),
             py::arg("nrtfn"),
             "Set the root finding function")
        .def("set_adaptive_params", &ARKodeSolver::set_adaptive_params,
             py::arg("adapt_params"),
             "Set parameters for adaptive step size control")
        .def("set_fixed_step_size", &ARKodeSolver::set_fixed_step_size,
             py::arg("step_size"),
             "Set a fixed step size and disable adaptive stepping")
        .def("get_current_solution", &ARKodeSolver::get_current_solution,
             "Get the current solution vector")
        .def("get_current_time", &ARKodeSolver::get_current_time,
             "Get the current integration time")
        .def("integrate_to_time", &ARKodeSolver::integrate_to_time,
             py::arg("tout"),
             "Integrate to a specified time")
        .def("advance_one_step", &ARKodeSolver::advance_one_step,
             py::arg("tout"),
             "Take a single internal step toward the target time")
        .def("solve_to", &ARKodeSolver::solve_to,
             py::arg("tout"),
             "Solve the ODE system to a specific time point")
        .def("solve_sequence", &ARKodeSolver::solve_sequence,
             py::arg("time_points"),
             "Solve the ODE system for multiple time points")
        .def("get_stats", &ARKodeSolver::get_stats,
             "Get solver statistics")
        .def("get_butcher_info", &ARKodeSolver::get_butcher_info,
             "Get information about the Butcher table being used")
        .def("get_last_step", &ARKodeSolver::get_last_step,
             "Get the last step size used by the solver")
        .def("__repr__",
             [](const ARKodeSolver &solver) {
                 return "<ARKodeSolver>";
             })
        // For backward compatibility with older code
        .def("solve_single", &ARKodeSolver::solve_to)
        .def("solve", &ARKodeSolver::solve_sequence)
        .def("set_max_num_steps", &ARKodeSolver::set_max_num_steps,
             py::arg("mxsteps"),
             "Set the maximum number of steps the solver will take");
}

}