import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import SundialsPy
from typing import List, Dict, Any, Callable, Tuple, Optional
import os


class SolverConfig:
    """Configuration class for a solver."""
    
    def __init__(self, 
                 name: str,
                 solver_type: str,
                 system_size: int,
                 rhs_fn: Callable,
                 args: Dict[str, Any] = None,
                 color: str = None,
                 dt: float = None):
        """
        Initialize a solver configuration.
        
        Args:
            name: Display name for the solver
            solver_type: Type of solver ('cvode' or 'arkode')
            system_size: Size of the ODE system
            rhs_fn: Right-hand side function
            args: Additional arguments for solver creation
            color: Color to use in plots
            dt: Fixed step size for ARKODE
        """
        self.name = name
        self.solver_type = solver_type.lower()
        self.system_size = system_size
        self.rhs_fn = rhs_fn
        self.args = args or {}
        self.color = color
        self.solver = None
        self.dt = dt
        
        # Validation
        if self.solver_type not in ['cvode', 'arkode']:
            raise ValueError(f"Unknown solver type: {solver_type}. Must be 'cvode' or 'arkode'")


def setup_cantera_simulation(mechanism: str, 
                             T_init: float, 
                             P_init: float,
                             fuel: str = 'H2',
                             equivalence_ratio: float = 1.0,
                             oxidizer: str = 'O2:1.0, N2:3.76') -> Tuple[ct.Solution, np.ndarray, int, Callable]:
    """
    Set up a Cantera simulation for constant pressure combustion.
    
    Args:
        mechanism: Cantera mechanism file
        T_init: Initial temperature (K)
        P_init: Initial pressure (Pa)
        fuel: Fuel species name
        equivalence_ratio: Fuel-oxidizer equivalence ratio
        oxidizer: Oxidizer composition string
        
    Returns:
        Tuple of:
        - gas: Cantera Solution object
        - y0: Initial condition vector
        - system_size: Size of the ODE system
        - rhs_fn: Right-hand side function for the ODE
    """
    # Create Cantera gas object
    gas = ct.Solution(mechanism)
    
    # Set the initial state
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)
    gas.TP = T_init, P_init
    
    # Initial state vector [Temperature, mass_fractions]
    y0 = np.hstack([gas.T, gas.Y])
    system_size = len(y0)
    
    # Define the ODE function (constant pressure combustion)
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
    
    return gas, y0, system_size, dydt


def create_solver(config: SolverConfig, 
                  y0: np.ndarray, 
                  t0: float, 
                  rel_tol: float, 
                  abs_tol: np.ndarray) -> Any:
    """
    Create and initialize a solver based on configuration.
    
    Args:
        config: Solver configuration
        y0: Initial condition vector
        t0: Initial time
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance vector
        
    Returns:
        Initialized solver
    """
    if config.solver_type == 'cvode':
        solver = SundialsPy.cvode.CVodeSolver(
            system_size=config.system_size,
            rhs_fn=config.rhs_fn,
            iter_type=config.args.get('iter_type', SundialsPy.cvode.IterationType.NEWTON),
            linsol_type=config.args.get('linsol_type', SundialsPy.cvode.LinearSolverType.DENSE)
        )
    elif config.solver_type == 'arkode':
        solver = SundialsPy.arkode.ARKodeSolver(
            system_size=config.system_size,
            explicit_fn=config.rhs_fn,
            implicit_fn=config.args.get('implicit_fn', None),
            butcher_table=config.args.get('butcher_table', SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2),
            # linsol_type=config.args.get('linsol_type', SundialsPy.arkode.LinearSolverType.DENSE)
        )
        #solver.set_max_num_steps(10000) 
        # solver.set_fixed_step_size(config.dt) 
    else:
        raise ValueError(f"Unsupported solver type: {config.solver_type}")
    
    # Initialize the solver
    solver.initialize(y0, t0, rel_tol, abs_tol)
    
    return solver


def solve_step_by_step(solver: Any, 
                       times: np.ndarray, 
                       y0: np.ndarray, 
                       integration_method: str = 'solve_to') -> Dict[str, Any]:
    """
    Solve an ODE system step by step.
    
    Args:
        solver: Initialized solver
        times: Array of time points
        y0: Initial condition vector
        integration_method: Method to use for integration ('solve_to', 'integrate_to_time', or 'advance_one_step')
        
    Returns:
        Dictionary with solution results and statistics
    """
    # Create solution array
    solution = np.zeros((len(times), len(y0)))
    
    # Save initial condition
    solution[0] = y0
    
    # Track per-step CPU times
    cpu_times = []
    
    # Start timing
    start_time = time.time()
    
    if integration_method == 'advance_one_step':
        # Special handling for advance_one_step method
        current_time = times[0]
        next_time_idx = 1
        
        while next_time_idx < len(times):
            # Target next time point
            target_time = times[next_time_idx]
            
            # Take a step
            step_start_time = time.time()
            solver.advance_one_step(target_time)
            cpu_times.append(time.time() - step_start_time)
            
            # Get current time and solution
            current_time = solver.get_current_time()
            solution[next_time_idx] = solver.get_current_solution()
            
            # Move to next time point if we've reached this one
            if current_time >= target_time:
                next_time_idx += 1
                
                # Print progress
                if next_time_idx % 100 == 0:
                    print(f"  Completed step {next_time_idx}/{len(times)}, t = {current_time:.2e}")
    else:
        # Normal step-by-step integration
        for i in range(1, len(times)):
            # Solve for this time point
            step_start_time = time.time()
            
            if integration_method == 'solve_to':
                y_i = solver.solve_to(times[i])
                solution[i] = y_i
            elif integration_method == 'integrate_to_time':
                solver.integrate_to_time(times[i])
                solution[i] = solver.get_current_solution()
            else:
                raise ValueError(f"Unknown integration method: {integration_method}")
                
            cpu_times.append(time.time() - step_start_time)
            
            # Print progress
            if i % 100 == 0:
                print(f"  Completed step {i}/{len(times)}, t = {times[i]:.2e}")
    
    # Calculate total solve time
    total_time = time.time() - start_time
    
    # Get solver statistics
    try:
        stats = solver.get_stats()
    except Exception as e:
        print(f"  Error getting statistics: {e}")
        stats = {}
    
    # Return results
    return {
        'solution': solution,
        'cpu_time': total_time,
        'cpu_times': cpu_times,
        'stats': stats
    }


def compare_solvers(solvers_config: List[SolverConfig], 
                    gas: ct.Solution,
                    y0: np.ndarray,
                    times: np.ndarray,
                    rel_tol: float,
                    abs_tol: np.ndarray,
                    t0: float = 0.0,
                    integration_method: str = 'solve_to',
                    output_dir: str = '.'):
    """
    Compare multiple ODE solvers on the same problem.
    
    Args:
        solvers_config: List of solver configurations
        gas: Cantera Solution object
        y0: Initial condition vector
        times: Array of time points
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance vector
        t0: Initial time
        integration_method: Method to use for integration
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Comparing {len(solvers_config)} solvers using {integration_method} method...")
    
    results = {}
    
    # Run each solver
    for config in solvers_config:
        print(f"\nRunning {config.name}...")
        
        try:
            # Create and initialize solver
            solver = create_solver(config, y0, t0, rel_tol, abs_tol)
            config.solver = solver  # Store for potential later use
            
            # Solve the problem
            solver_results = solve_step_by_step(solver, times, y0, integration_method)
            
            # Store results with metadata
            results[config.name] = {
                'times': times,
                'solution': solver_results['solution'],
                'cpu_time': solver_results['cpu_time'],
                'cpu_times': solver_results['cpu_times'],
                'stats': solver_results['stats'],
                'config': config
            }
            
            # Print summary statistics
            print(f"  Total CPU time: {solver_results['cpu_time']:.4f} seconds")
            if 'num_steps' in solver_results['stats']:
                print(f"  Steps: {solver_results['stats']['num_steps']}")
            if config.solver_type == 'cvode' and 'num_rhs_evals' in solver_results['stats']:
                print(f"  RHS evaluations: {solver_results['stats']['num_rhs_evals']}")
            elif config.solver_type == 'arkode' and 'num_rhs_evals_explicit' in solver_results['stats']:
                print(f"  RHS evaluations: {solver_results['stats']['num_rhs_evals_explicit']}")
            
        except Exception as e:
            print(f"  {config.name} failed: {e}")
    
    # Create plots if we have results
    if results:
        # Plot temperature
        plt.figure(figsize=(10, 6))
        
        for name, data in results.items():
            config = data['config']
            T = data['solution'][:, 0]
            plt.plot(data['times'], T, 
                    label=f"{name} (CPU: {data['cpu_time']:.4f}s)",
                    color=config.color)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Combustion: Temperature Evolution')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/combustion_temperature.png")
        
        # Calculate ignition delay times (using max dT/dt)
        print("\nIgnition delay times:")
        for name, data in results.items():
            T = data['solution'][:, 0]
            t = data['times']
            dTdt = np.gradient(T, t)
            ign_idx = np.argmax(dTdt)
            t_ign = t[ign_idx]
            print(f"  {name}: {t_ign:.2e} seconds")
        
        # Extract and plot major species (using first solver result)
        first_result = next(iter(results.values()))
        plt.figure(figsize=(10, 6))
        
        # Get mass fractions
        Y = first_result['solution'][:, 1:]
        t = first_result['times']
        
        # Update gas to get species names
        T_init = y0[0]
        gas.TPY = T_init, gas.P, y0[1:]
        species_names = gas.species_names
        
        # Plot major species
        major_species = ['H2', 'O2', 'H2O', 'OH', 'H', 'O']
        for species in major_species:
            if species in species_names:
                idx = species_names.index(species)
                plt.semilogy(t, Y[:, idx], label=species)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Mass Fraction')
        plt.title('Combustion: Species Evolution')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/combustion_species.png")

        # Create CPU time comparison plots
        if len(results) > 1:
            # Create a figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=300)
            
            # Plot 1: Raw CPU times
            for name, data in results.items():
                config = data['config']
                ax1.plot(data['cpu_times'], label=name, color=config.color)
            
            ax1.set_xlabel('Step')
            ax1.set_ylabel('CPU Time (s)')
            ax1.set_title('Raw CPU Time Comparison')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Log10 CPU times
            for name, data in results.items():
                config = data['config']
                cpu_times = np.array(data['cpu_times'])
                # Avoid log of zero or negative values
                cpu_times = np.maximum(cpu_times, 1e-10)
                ax2.plot(np.log10(cpu_times), label=name, color=config.color)
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel('log10(CPU Time) (s)')
            ax2.set_title('Log10 CPU Time Comparison')
            ax2.legend()
            ax2.grid(True)

            # Plot 3: Bar graph of total CPU times
            total_times = {name: np.sum(data['cpu_times']) for name, data in results.items()}
            colors = [data['config'].color for name, data in results.items()]
            
            ax3.bar(total_times.keys(), total_times.values(), color=colors)
            ax3.set_ylabel('Total CPU Time (s)')
            ax3.set_title('Total CPU Time Comparison')
            ax3.grid(True)

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{output_dir}/combustion_cpu_times.png")
        
        print(f"\nResults plotted and saved to {output_dir}/")


def run_combustion_simulation(integration_method='solve_to'):
    """
    Run the combustion simulation comparing multiple solvers.
    
    Args:
        integration_method: Method to use for integration ('solve_to', 'integrate_to_time', or 'advance_one_step')
    """
    # 1. Set up the combustion problem
    mechanism = "gri30.yaml"
    T_init = 600.0  # Initial temperature (K)
    P_init = 101325.0  # Initial pressure (Pa)
    
    # Set up the combustion model
    gas, y0, system_size, dydt = setup_cantera_simulation(
        mechanism=mechanism,
        T_init=T_init,
        P_init=P_init,
        fuel='H2',
        equivalence_ratio=1.0
    )
    
    # 2. Set up simulation parameters
    t0 = 0.0
    t_final = 4.0e-4  # 10 ms
    dt = 1.0e-6
    times = np.arange(t0, t_final, dt)
    
    rel_tol = 1.0e-6
    abs_tol = np.ones(system_size) * 1.0e-10
    
    # 3. Configure solvers to compare
    solvers_config = [
        SolverConfig(
            name="CVODE BDF",
            solver_type="cvode",
            system_size=system_size,
            rhs_fn=dydt,
            dt=dt,
            args={
                'iter_type': SundialsPy.cvode.IterationType.NEWTON
            },
            color='blue'
        ),
        SolverConfig(
            name="ARKODE ERK (Heun-Euler)",
            solver_type="arkode",
            system_size=system_size,
            rhs_fn=dydt,
            dt=dt,
            args={
                'butcher_table': SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2
            },
            color='red'
        ),
        SolverConfig(
            name="ARKODE ERK (Bogacki-Shampine)",
            solver_type="arkode",
            system_size=system_size,
            rhs_fn=dydt,
            dt=dt,
            args={
                'butcher_table': SundialsPy.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3
            },
            color='green'
        ),
        SolverConfig(
            name="ARKODE ERK (Zonneveld)",
            solver_type="arkode",
            system_size=system_size,
            rhs_fn=dydt,
            dt=dt,
            args={
                'butcher_table': SundialsPy.arkode.ButcherTable.ZONNEVELD_5_3_4
            },
            color='purple'
        )
    ]
    
    # 4. Create output directory based on integration method
    output_dir = f"results_{integration_method}"
    
    # 5. Run the comparison
    compare_solvers(
        solvers_config=solvers_config,
        gas=gas,
        y0=y0,
        times=times,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        t0=t0,
        integration_method=integration_method,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Run using different integration methods
    # Change this to the method you want to use
    integration_method = 'solve_to'  # 'solve_to', 'integrate_to_time', or 'advance_one_step'
    
    run_combustion_simulation(integration_method)