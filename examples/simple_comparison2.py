import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import SundialsPy

def compare_combustion_solvers_step_by_step():
    """
    Compare CVODE and ARKODE solvers on a 0D combustion problem using step-by-step integration.
    """
    print("Comparing CVODE and ARKODE for 0D combustion (step-by-step integration)...")
    
    # 1. Set up the combustion problem
    mechanism = "gri30.yaml"  # Hydrogen combustion (simple mechanism)
    T_init = 900.0  # Initial temperature (K)
    P_init = 101325.0  # Initial pressure (Pa)
    
    # Create Cantera gas object
    gas = ct.Solution(mechanism)
    
    # Set the initial state (stoichiometric H2-air mixture)
    gas.set_equivalence_ratio(1.0, 'H2', 'O2:1.0, N2:3.76')
    gas.TP = T_init, P_init
    
    # Initial state vector [Temperature, mass_fractions]
    y0 = np.hstack([gas.T, gas.Y])
    system_size = len(y0)
    
    # 2. Define the ODE function (constant pressure combustion)
    def dydt(t, y):
        # Extract state
        temperature = y[0]
        mass_fractions = y[1:]
        
        # Update gas state
        gas.TPY = temperature, P_init, mass_fractions
        
        # Calculate time derivatives
        rho = gas.density_mass
        wdot = gas.net_production_rates
        cp = gas.cp_mass
        h = gas.partial_molar_enthalpies
        
        # Energy equation
        dTdt = -(np.dot(h, wdot) / (rho * cp))
        
        # Species equations
        dYdt = wdot * gas.molecular_weights / rho
        
        # Return combined derivative vector
        return np.hstack([dTdt, dYdt])
    
    # 3. Set up simulation parameters
    t0 = 0.0
    t_final = 1.0e-2  # 1 ms (typical for hydrogen combustion)
    n_points = 1000
    times = np.linspace(t0 + 1e-10, t_final, n_points)
    
    rel_tol = 1.0e-6
    abs_tol = np.ones(system_size) * 1.0e-10
    
    results = {}
    cvode_cpu_times = []
    arkode_cpu_times = []
    # 4. Run CVODE BDF solver step by step
    print("\nRunning CVODE BDF solver (step by step)...")
    try:
        # Create solver
        cvode_solver = SundialsPy.cvode.CVodeSolver(
            system_size=system_size,
            rhs_fn=dydt,
            iter_type=SundialsPy.cvode.IterationType.NEWTON
        )
        
        # Initialize solver
        cvode_solver.initialize(y0, t0, rel_tol, abs_tol)
        
        # Create solution array
        cvode_solution = np.zeros((len(times), system_size))
        
        # Save initial condition
        cvode_solution[0] = y0
        
        # Solve step by step
        start_time = time.time()
        
        for i in range(1, len(times)):
            # Solve for this time point
            start_time_step = time.time()
            y_i = cvode_solver.solve_single(times[i])
            cpu_time_step = time.time() - start_time_step
            cvode_cpu_times.append(cpu_time_step)
            # Store the result
            cvode_solution[i] = y_i
            
            # Optional: add any between-steps processing here
            # For example: print progress every 100 steps
            if i % 100 == 0:
                print(f"  CVODE: Completed step {i}/{len(times)}, t = {times[i]:.2e}")
        
        cvode_time = time.time() - start_time
        
        # Store results
        results['CVODE BDF'] = {
            'times': times,
            'solution': cvode_solution,
            'cpu_time': cvode_time,
            'cpu_times': cvode_cpu_times
        }
        
        # Get statistics
        try:
            stats = cvode_solver.get_stats()
            print(f"  Steps: {stats.get('num_steps', 'N/A')}")
            print(f"  RHS evaluations: {stats.get('num_rhs_evals', 'N/A')}")
        except Exception as e:
            print(f"  Error getting statistics: {e}")
        
        print(f"  CPU time: {cvode_time:.4f} seconds")
    except Exception as e:
        print(f"  CVODE BDF failed: {e}")
    
    # 5. Run ARKODE ERK solver step by step
    print("\nRunning ARKODE ERK solver (step by step)...")
    try:
        # Create solver
        arkode_solver = SundialsPy.arkode.ARKodeSolver(
            system_size=system_size,
            explicit_fn=dydt,
            implicit_fn=None,
            butcher_table=SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2
        )
        
        # Initialize solver
        arkode_solver.initialize(y0, t0, rel_tol, abs_tol)
        
        # Create solution array
        arkode_solution = np.zeros((len(times), system_size))
        
        # Save initial condition
        arkode_solution[0] = y0
        
        # Solve step by step
        start_time = time.time()
        
        for i in range(1, len(times)):
            # Solve for this time point
            arkode_start_time = time.time()
            y_i = arkode_solver.solve_single(times[i])
            arkode_cpu_time = time.time() - arkode_start_time
            arkode_cpu_times.append(arkode_cpu_time)
            
            # Store the result
            arkode_solution[i] = y_i
            
            # Optional: add any between-steps processing here
            if i % 100 == 0:
                print(f"  ARKODE: Completed step {i}/{len(times)}, t = {times[i]:.2e}")
        
        arkode_time = time.time() - start_time
        
        # Store results
        results['ARKODE ERK'] = {
            'times': times,
            'solution': arkode_solution,
            'cpu_time': arkode_time,
            'cpu_times': arkode_cpu_times
        }
        
        # Get statistics
        try:
            stats = arkode_solver.get_stats()
            print(f"  Steps: {stats.get('num_steps', 'N/A')}")
            print(f"  RHS evaluations: {stats.get('num_rhs_evals_explicit', 'N/A')}")
        except Exception as e:
            print(f"  Error getting statistics: {e}")
        
        print(f"  CPU time: {arkode_time:.4f} seconds")
    except Exception as e:
        print(f"  ARKODE ERK failed: {e}")
    
    # 6. Plot results
    if results:
        # Plot temperature
        plt.figure(figsize=(10, 6))
        
        for solver_name, data in results.items():
            T = data['solution'][:, 0]
            plt.plot(data['times'], T, label=f"{solver_name} (CPU: {data['cpu_time']:.4f}s)")
        
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('H2-air Combustion: Temperature Evolution (Step-by-Step)')
        plt.grid(True)
        plt.legend()
        plt.savefig('combustion_temperature_step_by_step.png')
        
        # Calculate ignition delay times (using max dT/dt)
        print("\nIgnition delay times:")
        for solver_name, data in results.items():
            T = data['solution'][:, 0]
            t = data['times']
            dTdt = np.gradient(T, t)
            ign_idx = np.argmax(dTdt)
            t_ign = t[ign_idx]
            print(f"  {solver_name}: {t_ign:.2e} seconds")
        
        # Extract and plot major species for CVODE
        if 'CVODE BDF' in results:
            plt.figure(figsize=(10, 6))
            
            # Get mass fractions
            Y = results['CVODE BDF']['solution'][:, 1:]
            t = results['CVODE BDF']['times']
            
            # Update gas to get species names
            gas.TPY = T_init, P_init, y0[1:]
            species_names = gas.species_names
            
            # Plot major species
            major_species = ['H2', 'O2', 'H2O', 'OH', 'H', 'O']
            for species in major_species:
                if species in species_names:
                    idx = species_names.index(species)
                    plt.semilogy(t, Y[:, idx], label=species)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Mass Fraction')
            plt.title('H2-air Combustion: Species Evolution (Step-by-Step)')
            plt.grid(True)
            plt.legend()
            plt.savefig('combustion_species_step_by_step.png')

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=300)
        
        # Plot 1: Raw CPU times
        ax1.plot(cvode_cpu_times, label='CVODE BDF')
        ax1.plot(arkode_cpu_times, label='ARKODE ERK')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('CPU Time (s)')
        ax1.set_title('Raw CPU Time Comparison')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Log10 CPU times
        ax2.plot(np.log10(cvode_cpu_times), label='CVODE BDF')
        ax2.plot(np.log10(arkode_cpu_times), label='ARKODE ERK')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('log10(CPU Time) (s)')
        ax2.set_title('Log10 CPU Time Comparison')
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Bar graph of total CPU times
        total_times = {
            'CVODE BDF': np.sum(cvode_cpu_times),
            'ARKODE ERK': np.sum(arkode_cpu_times)
        }
        ax3.bar(total_times.keys(), total_times.values())
        ax3.set_ylabel('Total CPU Time (s)')
        ax3.set_title('Total CPU Time Comparison')
        ax3.grid(True)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('combustion_cpu_times_step_by_step.png')

if __name__ == "__main__":
    compare_combustion_solvers_step_by_step()