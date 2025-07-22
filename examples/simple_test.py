import numpy as np
import matplotlib.pyplot as plt
import time
import sundials_py
import sys

def test_all_solvers():
    """Test all SUNDIALS solvers on a simple test problem."""
    
    print(f"Testing all SUNDIALS solvers...")
    
    # Define a simple 3-variable stiff ODE system (Van der Pol)
    # dy1/dt = y2
    # dy2/dt = mu * (1 - y1^2) * y2 - y1
    # where mu is the stiffness parameter
    
    mu = 1000.0  # High mu makes this a stiff problem
    
    def rhs_fn(t, y):
        """Right-hand side function for Van der Pol oscillator."""
        # Print debug info occasionally to diagnose segfaults
        if np.random.random() < 0.01:
            print(f"RHS called with t={t:.6f}, y1={y[0]:.6f}, y2={y[1]:.6f}")
        
        dydt = np.zeros(2)
        dydt[0] = y[1]
        dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
        return dydt
    
    # Initial conditions
    y0 = np.array([2.0, 0.0])
    t0 = 0.0
    t_end = 3000.0
    
    # Set tolerances
    rtol = 1.0e-6
    atol = np.array([1.0e-8, 1.0e-8])
    
    # Time points for sampling the solution
    time_points = np.linspace(t0, t_end, 100)
    time_points = time_points[1:]
    
    # Dictionary to store results
    results = {}
    
    # Test CVODE BDF 
    print("\n--- Testing CVODE BDF ---")
    try:
        start_time = time.time()
        
        print("Creating solver...")
        cvode_bdf = sundials_py.cvode.CVodeSolver(
            system_size=len(y0),
            rhs_fn=rhs_fn,
            iter_type=sundials_py.cvode.IterationType.NEWTON
        )
        
        print("Initializing solver...")
        cvode_bdf.initialize(y0, t0, rtol, atol)
        
        print("Solving...")
        solution_bdf = cvode_bdf.solve_sequence(time_points)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"CVODE BDF completed in {cpu_time:.4f} seconds")
        stats = cvode_bdf.get_stats()
        print(f"Steps: {stats.get('num_steps', 'N/A')}")
        print(f"RHS evaluations: {stats.get('num_rhs_evals', 'N/A')}")
        
        results['CVODE BDF'] = {
            'solution': solution_bdf,
            'cpu_time': cpu_time,
            'stats': stats
        }
    except Exception as e:
        print(f"CVODE BDF failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test CVODE Adams
    print("\n--- Testing CVODE Adams ---")
    try:
        start_time = time.time()
        
        print("Creating solver...")
        cvode_adams = sundials_py.cvode.CVodeSolver(
            system_size=len(y0),
            rhs_fn=rhs_fn,
            iter_type=sundials_py.cvode.IterationType.FUNCTIONAL
        )

        
        print("Initializing solver...")
        cvode_adams.initialize(y0, t0, rtol, atol)
        
        print("Solving...")
        solution_adams = cvode_adams.solve_sequence(time_points)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"CVODE Adams completed in {cpu_time:.4f} seconds")
        stats = cvode_adams.get_stats()
        print(f"Steps: {stats.get('num_steps', 'N/A')}")
        print(f"RHS evaluations: {stats.get('num_rhs_evals', 'N/A')}")
        
        results['CVODE Adams'] = {
            'solution': solution_adams,
            'cpu_time': cpu_time,
            'stats': stats
        }
    except Exception as e:
        print(f"CVODE Adams failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ARKODE Explicit (ERK)
    print("\n--- Testing ARKODE Explicit (ERK) ---")
    try:
        start_time = time.time()
        
        print("Creating solver...")
        arkode_erk = sundials_py.arkode.ARKodeSolver(
            system_size=len(y0),
            explicit_fn=rhs_fn,
            butcher_table=sundials_py.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3
        )
        
        
        print("Initializing solver...")
        arkode_erk.initialize(y0, t0, rtol, atol)
        
        print("Solving...")
        solution_erk = arkode_erk.solve_sequence(time_points)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"ARKODE ERK completed in {cpu_time:.4f} seconds")
        stats = arkode_erk.get_stats()
        print(f"Steps: {stats.get('num_steps', 'N/A')}")
        print(f"RHS evaluations: {stats.get('num_rhs_evals_explicit', 'N/A')}")
        
        results['ARKODE ERK'] = {
            'solution': solution_erk,
            'cpu_time': cpu_time,
            'stats': stats
        }
    except Exception as e:
        print(f"ARKODE ERK failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ARKODE Implicit (DIRK) - need to provide dummy implicit function
    print("\n--- Testing ARKODE Implicit (DIRK) ---")
    try:
        start_time = time.time()
        
        # For IMEX methods, both explicit and implicit functions are needed
        def implicit_rhs_fn(t, y):
            """Implicit part of the split ODE (zero for this test)."""
            # For testing, just return zeros - all dynamics in explicit part
            if np.random.random() < 0.01:
                print(f"Implicit RHS called with t={t:.6f}, y={y}")
            return np.zeros_like(y)
        
        print("Creating solver...")
        # Use a DIRK method (implicit only) to test the implicit solver
        arkode_dirk = sundials_py.arkode.ARKodeSolver(
            system_size=len(y0),
            explicit_fn=None,  # No explicit part
            implicit_fn=rhs_fn,  # All dynamics in implicit part
            butcher_table=sundials_py.arkode.ButcherTable.SDIRK_5_3_4
        )
        
        print("Initializing solver...")
        arkode_dirk.initialize(y0, t0, rtol, atol)
        
        print("Solving...")
        solution_dirk = arkode_dirk.solve_sequence(time_points)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"ARKODE DIRK completed in {cpu_time:.4f} seconds")
        stats = arkode_dirk.get_stats()
        print(f"Steps: {stats.get('num_steps', 'N/A')}")
        print(f"RHS evaluations (implicit): {stats.get('num_rhs_evals_implicit', 'N/A')}")
        
        results['ARKODE DIRK'] = {
            'solution': solution_dirk,
            'cpu_time': cpu_time,
            'stats': stats
        }
    except Exception as e:
        print(f"ARKODE DIRK failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ARKODE IMEX (both explicit and implicit parts)
    print("\n--- Testing ARKODE IMEX ---")
    try:
        start_time = time.time()
        
        # For IMEX methods, split the ODE into stiff and non-stiff parts
        def explicit_part(t, y):
            """Non-stiff part of the RHS."""
            # For this example, put the non-stiff part here (linear terms)
            dydt = np.zeros(2)
            dydt[0] = y[1]
            dydt[1] = -y[0]  # Linear part
            if np.random.random() < 0.01:
                print(f"Explicit part called: t={t:.6f}, y1={y[0]:.6f}, y2={y[1]:.6f}")
            return dydt
            
        def implicit_part(t, y):
            """Stiff part of the RHS."""
            # Put the stiff part here (nonlinear terms)
            dydt = np.zeros(2)
            dydt[0] = 0
            dydt[1] = mu * (1 - y[0]**2) * y[1]  # Stiff nonlinear part
            if np.random.random() < 0.01:
                print(f"Implicit part called: t={t:.6f}, y1={y[0]:.6f}, y2={y[1]:.6f}")
            return dydt
        
        print("Creating solver...")
        # Use an IMEX pair
        arkode_imex = sundials_py.arkode.ARKodeSolver(
            system_size=len(y0),
            explicit_fn=explicit_part,
            implicit_fn=implicit_part,
            butcher_table=sundials_py.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4_DIRK_6_3_4
        )
        
        print("Initializing solver...")
        arkode_imex.initialize(y0, t0, rtol, atol)
        
        print("Solving...")
        solution_imex = arkode_imex.solve_sequence(time_points)
        
        end_time = time.time()
        cpu_time = end_time - start_time
        
        print(f"ARKODE IMEX completed in {cpu_time:.4f} seconds")
        stats = arkode_imex.get_stats()
        print(f"Steps: {stats.get('num_steps', 'N/A')}")
        print(f"RHS evaluations (explicit): {stats.get('num_rhs_evals_explicit', 'N/A')}")
        print(f"RHS evaluations (implicit): {stats.get('num_rhs_evals_implicit', 'N/A')}")
        
        results['ARKODE IMEX'] = {
            'solution': solution_imex,
            'cpu_time': cpu_time,
            'stats': stats
        }
    except Exception as e:
        print(f"ARKODE IMEX failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot results
    if results:
        plt.figure(figsize=(12, 8))
        
        # Plot solutions
        for name, data in results.items():
            solution = data['solution']
            plt.plot(time_points, solution[:, 0], label=f"{name} (CPU: {data['cpu_time']:.4f}s)")
        
        plt.xlabel('Time')
        plt.ylabel('y[0]')
        plt.title('Van der Pol Oscillator (mu = 1000)')
        plt.grid(True)
        plt.legend()
        plt.savefig('sundials_comparison.png')
        plt.show()
        
        # Print performance summary
        print("\nPerformance Summary:")
        print("-" * 80)
        print(f"{'Method':<15} {'CPU Time (s)':<15} {'Steps':<10} {'RHS Evals':<15}")
        print("-" * 80)
        
        for name, data in results.items():
            stats = data['stats']
            
            # Get the right RHS evals key based on method
            if 'ARKODE' in name:
                if 'IMEX' in name:
                    rhs_evals = f"{stats.get('num_rhs_evals_explicit', 'N/A')}+{stats.get('num_rhs_evals_implicit', 'N/A')}"
                elif 'DIRK' in name:
                    rhs_evals = stats.get('num_rhs_evals_implicit', 'N/A')
                else:
                    rhs_evals = stats.get('num_rhs_evals_explicit', 'N/A')
            else:
                rhs_evals = stats.get('num_rhs_evals', 'N/A')
            
            print(f"{name:<15} {data['cpu_time']:<15.4f} {stats.get('num_steps', 'N/A'):<10} {rhs_evals:<15}")

if __name__ == "__main__":
    # Catch any unexpected errors at the top level
    try:
        test_all_solvers()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()