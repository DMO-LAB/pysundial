import os 
import rk_solver_cpp
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import SundialsPy as SP
import cantera as ct
import time
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# CHEMISTRY SETUP FUNCTIONS
# ============================================================================

def setup_combustion_chemistry(mechanism: str, fuel: str, oxidizer: str, 
                              phi: float, temperature: float, pressure: float) -> ct.Solution:
    """Set up the combustion chemistry with Cantera.
    
    Args:
        mechanism: Path to mechanism file
        fuel: Fuel species name
        oxidizer: Oxidizer mixture string
        phi: Equivalence ratio
        temperature: Initial temperature (K)
        pressure: Initial pressure (Pa)
    
    Returns:
        gas: Initialized Cantera gas object
    """
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TPX = temperature, pressure, gas.X
    return gas

def get_initial_state(gas: ct.Solution) -> np.ndarray:
    """Get initial state vector [T, Y1, Y2, ...].
    
    Args:
        gas: Cantera gas object
    
    Returns:
        y: Initial state vector
    """
    return np.hstack([gas.T, gas.Y])

def reset_chemistry_state(gas: ct.Solution, phi: float, fuel: str, oxidizer: str, 
                         temperature: float, pressure: float) -> Tuple[ct.Solution, np.ndarray]:
    """Reset chemistry to initial conditions.
    
    Args:
        gas: Cantera gas object
        phi: Equivalence ratio
        fuel: Fuel species name
        oxidizer: Oxidizer mixture string
        temperature: Initial temperature (K)
        pressure: Initial pressure (Pa)
    
    Returns:
        gas: Reset gas object
        y: Reset state vector
    """
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TPX = temperature, pressure, gas.X
    y = np.hstack([gas.T, gas.Y])
    return gas, y

# ============================================================================
# ODE SYSTEM DEFINITION
# ============================================================================

def combustion_rhs(t: float, y: np.ndarray, gas: ct.Solution, pressure: float) -> np.ndarray:
    """Right-hand side of the combustion ODE system.
    
    Args:
        t: Current time
        y: Current state vector [T, Y1, Y2, ...]
        gas: Cantera gas object
        pressure: Constant pressure
    
    Returns:
        dydt: Time derivatives [dT/dt, dY1/dt, dY2/dt, ...]
    """
    # Extract temperature and mass fractions
    T = y[0]
    Y = y[1:]
    
    # Update the gas state
    gas.TPY = T, pressure, Y
    
    # Get thermodynamic properties
    rho = gas.density_mass
    wdot = gas.net_production_rates
    cp = gas.cp_mass
    h = gas.partial_molar_enthalpies
    
    # Calculate temperature derivative (energy equation)
    dTdt = -(np.dot(h, wdot) / (rho * cp))
    
    # Calculate species derivatives (mass conservation)
    dYdt = wdot * gas.molecular_weights / rho
    
    # Combine into full derivative vector
    return np.hstack([dTdt, dYdt])

# ============================================================================
# SOLVER CREATION FUNCTIONS
# ============================================================================

def create_sundials_solver(method: str, system_size: int, rtol: float, atol: np.ndarray, 
                          gas: ct.Solution, pressure: float) -> Any:
    """Create a SUNDIALS solver.
    
    Args:
        method: Solver method ('cvode_bdf', 'cvode_adams', 'arkode_erk')
        system_size: Size of the ODE system
        rtol: Relative tolerance
        atol: Absolute tolerance array
        gas: Cantera gas object
        pressure: Constant pressure
    
    Returns:
        solver: Initialized SUNDIALS solver
    """
    if method == 'cvode_bdf':
        solver = SP.cvode.CVodeSolver(
            system_size=system_size,
            rhs_fn=lambda t, y: combustion_rhs(t, y, gas, pressure),
            iter_type=SP.cvode.IterationType.NEWTON
        )
    elif method == 'cvode_adams':
        solver = SP.cvode.CVodeSolver(
            system_size=system_size,
            rhs_fn=lambda t, y: combustion_rhs(t, y, gas, pressure),
            iter_type=SP.cvode.IterationType.FUNCTIONAL
        )
    elif method == 'arkode_erk':
        solver = SP.arkode.ARKodeSolver(
            system_size=system_size,
            explicit_fn=lambda t, y: combustion_rhs(t, y, gas, pressure),
            implicit_fn=None,
            butcher_table=SP.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5
        )
    else:
        raise ValueError(f"Unknown SUNDIALS method: {method}")
    
    solver.initialize(np.zeros(system_size), 0.0, rtol, atol)
    return solver

def create_cpp_solver(method: str, t: float, y: np.ndarray, t_end: float, 
                     rtol: float, atol: float, gas: ct.Solution, pressure: float) -> Any:
    """Create a C++ RK solver.
    
    Args:
        method: Solver method ('cpp_rk23', 'cpp_rk45', etc.)
        t: Current time
        y: Current state
        t_end: End time
        rtol: Relative tolerance
        atol: Absolute tolerance
        gas: Cantera gas object
        pressure: Constant pressure
    
    Returns:
        solver: Initialized C++ solver
    """
    if method == 'cpp_rk23':
        return rk_solver_cpp.RK23(
            lambda t, y: combustion_rhs(t, y, gas, pressure), 
            float(t), np.array(y), float(t_end), rtol=rtol, atol=atol
        )
    else:
        raise ValueError(f"Unknown C++ method: {method}")

def create_scipy_solver(method: str, t: float, y: np.ndarray, rtol: float, atol: float,
                       gas: ct.Solution, pressure: float) -> Any:
    """Create a SciPy solver.
    
    Args:
        method: Solver method ('scipy_rk23', 'scipy_bdf', etc.)
        t: Current time
        y: Current state
        rtol: Relative tolerance
        atol: Absolute tolerance
        gas: Cantera gas object
        pressure: Constant pressure
    
    Returns:
        solver: Initialized SciPy solver
    """
    method_parts = method.split('_')
    if len(method_parts) != 3:
        raise ValueError(f"Invalid SciPy method format: {method}")
    
    solver_type, method_name = method_parts[1], method_parts[2]
    solver = ode(lambda t, y: combustion_rhs(t, y, gas, pressure)).set_integrator(
        solver_type, method=method_name, rtol=rtol, atol=atol, nsteps=10000
    )
    solver.set_initial_value(y, t)
    return solver

def create_solver(method: str, gas: ct.Solution, y: np.ndarray, t: float, 
                 rtol: float, atol: float, t_end: Optional[float] = None) -> Any:
    """Create the appropriate solver based on method.
    
    Args:
        method: Solver method string
        gas: Cantera gas object
        y: Current state
        t: Current time
        rtol: Relative tolerance
        atol: Absolute tolerance
        t_end: End time (for some solvers)
    
    Returns:
        solver: Initialized solver
    """
    system_size = 1 + gas.n_species
    
    # Create absolute tolerance array
    if np.isscalar(atol):
        abs_tol = np.ones(system_size) * atol
    else:
        abs_tol = np.asarray(atol)
        if len(abs_tol) == 1:
            abs_tol = np.ones(system_size) * abs_tol[0]
    
    if method.startswith('cvode_') or method.startswith('arkode_'):
        return create_sundials_solver(method, system_size, rtol, abs_tol, gas, pressure)
    elif method.startswith('cpp_'):
        return create_cpp_solver(method, t, y, t_end, rtol, atol, gas, pressure)
    elif method.startswith('scipy_'):
        return create_scipy_solver(method, t, y, rtol, atol, gas, pressure)
    else:
        raise ValueError(f"Unknown solver method: {method}")

# ============================================================================
# INTEGRATION FUNCTIONS
# ============================================================================

def integrate_single_step(method: str, gas: ct.Solution, y: np.ndarray, t: float, 
                         timestep: float, rtol: float, atol: float) -> Dict[str, Any]:
    """Integrate one step with the specified method.
    
    Args:
        method: Solver method
        gas: Cantera gas object
        y: Current state
        t: Current time
        timestep: Time step size
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        result: Dictionary with integration results
    """
    t_end = t + timestep
    previous_state = y.copy()
    
    try:
        # Create solver
        solver = create_solver(method, gas, y, t, rtol, atol, t_end)
        
        # Integrate
        start_time = time.time()
        
        if method.startswith('cpp_'):
            result = rk_solver_cpp.solve_ivp(solver, np.array(t_end))
            new_y = result['y'][-1]
        elif method.startswith('scipy_'):
            solver.integrate(t_end)
            new_y = solver.y
        else:  # SUNDIALS
            new_y = solver.solve_single(t_end)
        
        cpu_time = time.time() - start_time
        
        return {
            'success': True,
            't': t_end,
            'y': new_y,
            'cpu_time': cpu_time,
            'error': 0.0,
            'message': 'Success',
            'timed_out': False,
            'previous_state': previous_state
        }
        
    except Exception as e:
        return {
            'success': False,
            'y': previous_state,
            'cpu_time': 0.0,
            'error': float('inf'),
            'message': str(e),
            'timed_out': False,
            'previous_state': previous_state
        }

def run_integration_experiment(method: str, gas: ct.Solution, y0: np.ndarray, 
                             t0: float, end_time: float, timestep: float,
                             rtol: float, atol: float, species_to_track: List[str]) -> Dict[str, Any]:
    """Run a complete integration experiment with the specified method.
    
    Args:
        method: Solver method to test
        gas: Cantera gas object
        y0: Initial state
        t0: Start time
        end_time: End time
        timestep: Time step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        species_to_track: List of species to monitor
    
    Returns:
        results: Dictionary with complete integration results
    """
    # Initialize tracking arrays
    times = [t0]
    temperatures = [y0[0]]
    species_profiles = {spec: [y0[gas.species_index(spec) + 1]] for spec in species_to_track}
    cpu_times = []
    
    # Integration loop
    t = t0
    y = y0.copy()
    step_count = 0
    
    while t < end_time:
        result = integrate_single_step(method, gas, y, t, timestep, rtol, atol)
        
        if not result['success']:
            print(f"Step {step_count} failed: {result['message']}")
            break
        
        # Update state
        y = result['y']
        t = result['t']
        cpu_times.append(result['cpu_time'])
        step_count += 1
        
        # Record data
        times.append(t)
        temperatures.append(y[0])
        for spec in species_to_track:
            species_profiles[spec].append(y[gas.species_index(spec) + 1])
        
        print(f"Step {step_count} at time {t:.2e} - temperature {y[0]:.1f}K - CPU time {cpu_times[-1]:.2e}s")
    
    return {
        'method': method,
        'times': np.array(times),
        'temperatures': np.array(temperatures),
        'species_profiles': species_profiles,
        'cpu_times': np.array(cpu_times),
        'total_cpu_time': np.sum(cpu_times),
        'steps': step_count,
        'success': step_count > 0
    }

# ============================================================================
# ANALYSIS AND PLOTTING FUNCTIONS
# ============================================================================

def plot_single_experiment(results: Dict[str, Any], species_to_track: List[str]) -> None:
    """Plot results from a single integration experiment.
    
    Args:
        results: Results dictionary from run_integration_experiment
        species_to_track: List of species to plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Temperature plot
    ax1.plot(results['times'], results['temperatures'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title(f'Integration Results: {results["method"]}')
    ax1.grid(True, alpha=0.3)
    
    # Species profiles
    for spec in species_to_track:
        if spec in results['species_profiles']:
            ax2.plot(results['times'], results['species_profiles'][spec], label=spec)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass Fraction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def compare_methods(methods: List[str], gas: ct.Solution, y0: np.ndarray, 
                   t0: float, end_time: float, timestep: float,
                   rtol: float, atol: float, species_to_track: List[str]) -> Dict[str, Any]:
    """Compare multiple integration methods.
    
    Args:
        methods: List of methods to compare
        gas: Cantera gas object
        y0: Initial state
        t0: Start time
        end_time: End time
        timestep: Time step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        species_to_track: List of species to monitor
    
    Returns:
        comparison: Dictionary with comparison results
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}")
        
        # Reset chemistry state for each method
        gas, y_reset = reset_chemistry_state(gas, params['phi'], params['fuel'], params['oxidizer'], 
                                           params['temperature'], params['pressure'])
        
        # Run experiment
        method_results = run_integration_experiment(
            method, gas, y_reset, t0, end_time, timestep, rtol, atol, species_to_track
        )
        
        results[method] = method_results
        
        # Plot individual results
        plot_single_experiment(method_results, species_to_track)
    
    return results

def print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
    """Print a summary comparison of all methods.
    
    Args:
        comparison_results: Results from compare_methods
    """
    print("\n" + "="*80)
    print("INTEGRATION METHOD COMPARISON SUMMARY")
    print("="*80)
    
    for method, results in comparison_results.items():
        print(f"\n{method}:")
        print(f"  Success: {results['success']}")
        print(f"  Steps completed: {results['steps']}")
        print(f"  Total CPU time: {results['total_cpu_time']:.6f}s")
        print(f"  Average CPU time per step: {np.mean(results['cpu_times']):.6f}s")
        if results['success']:
            print(f"  Final temperature: {results['temperatures'][-1]:.1f}K")
            print(f"  Temperature change: {results['temperatures'][-1] - results['temperatures'][0]:.1f}K")

# ============================================================================
# MAIN EXPERIMENTAL SETUP
# ============================================================================

def setup_experiment_parameters() -> Dict[str, Any]:
    """Set up the experimental parameters.
    
    Returns:
        params: Dictionary with experiment parameters
    """
    return {
        'mechanism': "large_mechanism/ch4_53species.yaml",
        'fuel': "CH4",
        'oxidizer': 'O2:1, N2:3.76',
        'phi': 1.0,
        'temperature': 1200.0,
        'pressure': ct.one_atm,
        'rtol': 1e-6,
        'atol': 1e-8,
        'end_time': 1e-1,
        'timestep': 1e-5,
        'species_to_track': ['CH4', 'O2', 'N2', 'CO2', 'H2O', 'CO', 'H2', 'O', 'OH', 'H', 'NO', 'NO2', 'N']
    }

def run_single_method_test(method: str) -> None:
    """Run a test with a single integration method.
    
    Args:
        method: Method to test
    """
    # Setup parameters
    params = setup_experiment_parameters()
    
    # Setup chemistry
    gas = setup_combustion_chemistry(
        params['mechanism'], params['fuel'], params['oxidizer'],
        params['phi'], params['temperature'], params['pressure']
    )
    
    # Get initial state
    y0 = get_initial_state(gas)
    t0 = 0.0
    
    # Run experiment
    results = run_integration_experiment(
        method, gas, y0, t0, params['end_time'], params['timestep'],
        params['rtol'], params['atol'], params['species_to_track']
    )
    
    # Plot results
    plot_single_experiment(results, params['species_to_track'])
    
    # Print summary
    print(f"\nMethod {method} completed successfully!")
    print(f"Total CPU time: {results['total_cpu_time']:.6f}s")
    print(f"Steps completed: {results['steps']}")

def run_method_comparison(methods: List[str]) -> None:
    """Run a comparison of multiple integration methods.
    
    Args:
        methods: List of methods to compare
    """
    # Setup parameters
    params = setup_experiment_parameters()
    
    # Setup chemistry
    gas = setup_combustion_chemistry(
        params['mechanism'], params['fuel'], params['oxidizer'],
        params['phi'], params['temperature'], params['pressure']
    )
    
    # Get initial state
    y0 = get_initial_state(gas)
    t0 = 0.0
    
    # Run comparison
    comparison_results = compare_methods(
        methods, gas, y0, t0, params['end_time'], params['timestep'],
        params['rtol'], params['atol'], params['species_to_track']
    )
    
    # Print comparison summary
    print_comparison_summary(comparison_results)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Test a single method
    print("Testing single method: cpp_rk23")
    run_single_method_test('cpp_rk23')
    
    # Example 2: Compare multiple methods
    print("\n" + "="*80)
    print("COMPARING MULTIPLE METHODS")
    print("="*80)
    
    methods_to_test = ['cpp_rk23', 'cvode_bdf', 'cvode_adams', 'arkode_erk']
    run_method_comparison(methods_to_test)
