import os 
import rk_solver_cpp
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import SundialsPy as SP
import cantera as ct
import time
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm 
import pickle
import pandas as pd
from datetime import datetime

# ============================================================================
# CHEMISTRY SETUP FUNCTIONS
# ============================================================================

# Define fuel to mechanism mapping at the top
fuel_to_mechanism = {
    'nc12h26': 'large_mechanism/n-dodecane.yaml',
    'CH4': 'large_mechanism/ch4_53species.yaml',
    'NC12H26': 'large_mechanism/JetSurF1.0-l.yaml'
}

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

def create_sundials_solver(method: str, y: np.ndarray, t: float, system_size: int, rtol: float, atol: np.ndarray, 
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
    
    solver.initialize(y, t, rtol, atol)
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
                 rtol: float, atol: float, pressure: float, t_end: Optional[float] = None) -> Any:
    """Create the appropriate solver based on method.
    
    Args:
        method: Solver method string
        gas: Cantera gas object
        y: Current state
        t: Current time
        rtol: Relative tolerance
        atol: Absolute tolerance
        pressure: System pressure
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
        return create_sundials_solver(method, y, t, system_size, rtol, abs_tol, gas, pressure)
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
                         timestep: float, rtol: float, atol: float, pressure: float, fuel: str) -> Dict[str, Any]:
    """Integrate one step with the specified method.
    
    Args:
        method: Solver method
        gas: Cantera gas object
        y: Current state
        t: Current time
        timestep: Time step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        pressure: System pressure
        fuel: Fuel name
    
    Returns:
        result: Dictionary with integration results
    """
    t_end = t + timestep
    previous_state = y.copy()
    try:
        # Create solver
        solver = create_solver(method, gas, y, t, rtol, atol, pressure, t_end)
        
        # Integrate
        start_time = time.time()
        
        if method.startswith('cpp_'):
            result = rk_solver_cpp.solve_ivp(solver, np.array([t_end]))
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
            'fuel_mass_fraction': gas.mass_fraction_dict()[fuel] if fuel in gas.mass_fraction_dict().keys() else 0.0,
            'error': 0.0,
            'message': 'Success',
            'timed_out': False,
            'previous_state': previous_state
        }
        
    except Exception as e:
        print(f"Step {t} failed: {e}")
        return {
            'success': False,
            't': t,
            'y': previous_state,
            'fuel_mass_fraction': gas.mass_fraction_dict()[fuel] if fuel in gas.mass_fraction_dict().keys() else 0.0,
            'cpu_time': 0.0,
            'error': float('inf'),
            'message': str(e),
            'timed_out': False,
            'previous_state': previous_state
        }

def run_integration_experiment(method: str, gas: ct.Solution, y0: np.ndarray, 
                             t0: float, end_time: float, timestep: float,
                             rtol: float, atol: float, species_to_track: List[str],
                             fuel: str, pressure: float,
                             time_limit: float = 300.0) -> Dict[str, Any]:
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
        fuel: Fuel name
        pressure: System pressure
        time_limit: Maximum allowed wall clock time in seconds (default 300s)
    
    Returns:
        results: Dictionary with complete integration results
    """
    # Initialize tracking arrays
    times = [t0]
    temperatures = [y0[0]]
    species_profiles = {spec: [y0[gas.species_index(spec) + 1]] for spec in species_to_track}
    cpu_times = []
    fuel_mass_fractions = []
    
    # Integration loop
    t = t0
    y = y0.copy()
    step_count = 0
    start_time = time.time()
    
    bar = tqdm(total=end_time, desc=f"Running {method} with rtol={rtol} and atol={atol}")
    while t < end_time:
        bar.update(timestep)
        # Check if time limit exceeded
        if time.time() - start_time > time_limit:
            print(f"Time limit of {time_limit}s exceeded after {step_count} steps")
            break
            
        result = integrate_single_step(method, gas, y, t, timestep, rtol, atol, pressure, fuel)
        
        if not result['success']:
            print(f"Step {step_count} failed: {result['message']}")
            break
        
        # Update state
        y = result['y']
        t = result['t']
        cpu_times.append(result['cpu_time'])
        step_count += 1
        fuel_mass_fractions.append(result['fuel_mass_fraction'])

        # ensure that y is not empty
        if len(y) == 0:
            print(f"Step {step_count} failed: y is empty")
            print(result)
            print(y)
            break
        
        # Record data
        times.append(t)
        temperatures.append(y[0])
        for spec in species_to_track:
            species_profiles[spec].append(y[gas.species_index(spec) + 1])
        
        #print(f"Step {step_count} at time {t:.2e} - temperature {y[0]:.1f}K - CPU time {cpu_times[-1]:.2e}s - time taken {time.time() - start_time:.2f}s | {np.sum(cpu_times):.2e}s")
        bar.set_postfix({
            'step': f"{step_count}",
            'temperature': f"{y[0]:.1f}K",
            'cpu_time': f"{cpu_times[-1]:.2e}s",
            'total_cpu_time': f"{np.sum(cpu_times):.2e}s"
        })
    bar.close() 
    total_wall_time = time.time() - start_time
    return {
        'method': method,
        'phi': gas.equivalence_ratio,
        'rtol': rtol,
        'atol': atol,   
        'times': np.array(times),
        'fuel_mass_fractions': np.array(fuel_mass_fractions),
        'temperatures': np.array(temperatures),
        'species_profiles': species_profiles,
        'cpu_times': np.array(cpu_times),
        'total_cpu_time': np.sum(cpu_times),
        'total_wall_time': total_wall_time,
        'steps': step_count,
        'success': step_count > 0,
        'timed_out': total_wall_time > time_limit
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_solver_performance(simulation_results, methods, tolerances, species_to_track, window_size=100):
    """
    Analyzes performance of different ODE solvers and returns solution data with performance metrics.
    
    Args:
        simulation_results (dict): Dictionary containing simulation results
        methods (list): List of integration methods to analyze
        tolerances (list): List of (rtol, atol) tolerance pairs
        species_to_track (list): List of chemical species to track
        window_size (int): Size of window for averaging CPU times
        
    Returns:
        pd.DataFrame: DataFrame containing solver performance data and solution values
    """
    # Load results once
   

    method_to_number = {}
    counter = 0
    for method in methods:
        for tolerance in tolerances:
            rtol = tolerance[0]
            atol = tolerance[1]
            name = f"{method}_{rtol}_{atol}"
            method_to_number[name] = counter
            counter += 1

    # Pre-allocate solution data dictionary
    solution_data = {}
    
    # Process each method/tolerance combination
    for method in methods:
        for rtol, atol in tolerances:
            method_name = f"{method}_{rtol}_{atol}"
            
            # Skip if method not in results
            if method_name not in simulation_results:
                continue
                
            # Get results for this method/tolerance
            results = simulation_results[method_name]
            
            # Create DataFrame for this method
            data_length = len(results['cpu_times'])
            # print(f"Method - {method} - atol - {atol} - data length - {data_length}")
            
            # Prepare all data columns first
            data_dict = {
                'times': results['times'][:data_length],
                'method': int(method_to_number[method_name]),
                'temperatures': results['temperatures'][:data_length],
                'cpu_times': results['cpu_times'][:data_length],
                'fuel_mass_fractions': results['fuel_mass_fractions']
            }
            
            # Calculate windowed average of CPU times
            data_dict['windowed_cpu_times'] = pd.Series(data_dict['cpu_times']).rolling(
                window=window_size, min_periods=1, center=True).mean()
            
            # Add species concentrations
            if 'species_profiles' in results:
                profiles = results['species_profiles']
                for species in species_to_track:
                    # Check for both lowercase and uppercase versions of the species name
                    species_lower = species.lower()
                    species_upper = species.upper()
                    
                    # Try exact match first
                    if species in profiles.keys():
                        data_dict[species] = profiles[species][:data_length]
                    # Try lowercase version
                    elif species_lower in profiles.keys():
                        data_dict[species] = profiles[species_lower][:data_length]
                    # Try uppercase version
                    elif species_upper in profiles.keys():
                        data_dict[species] = profiles[species_upper][:data_length]
                    else:
                        # If no match found, you might want to handle this case
                        print(f"Warning: Species '{species}' not found in species_profiles")
            
            # Create DataFrame with all columns at once
            df = pd.DataFrame(data_dict)
            solution_data[method_name] = df

    # Use highest precision solution as reference
    reference_method = 'cvode_bdf_1e-12_1e-14'
    if reference_method not in solution_data:
        # Use the first available method as reference
        reference_method = next(iter(solution_data.keys()))
    
    reference_data = solution_data[reference_method]
    
    # Initialize performance DataFrame
    data_length = len(reference_data['times'])
    times = reference_data['times']
    
    # Calculate best method at each timestep based on CPU time
    best_methods = []
    for i, _ in enumerate(times):
        cpu_times = []
        for method_name, results in solution_data.items():
            if i < len(results['times']):
                cpu_times.append((int(results['method'].iloc[i]), results['windowed_cpu_times'].iloc[i]))
        if cpu_times:
            best_methods.append(min(cpu_times, key=lambda x: x[1])[0])
        else:
            best_methods.append('')
    
    # Prepare all data columns first
    solver_data_dict = {
        'time': reference_data['times'],
        'best_method': best_methods,
        'temperature': reference_data['temperatures'],
        'fuel_mass_fraction': reference_data['fuel_mass_fractions']
    }
    
    # Add species data
    for species in species_to_track:
        if species in reference_data.columns:
            solver_data_dict[species] = reference_data[species]
    
    # Create DataFrame with all columns at once
    solver_data = pd.DataFrame(solver_data_dict)

    return solver_data, solution_data

# ============================================================================
# MAIN EXPERIMENTAL SETUP
# ============================================================================

def setup_experiment_parameters(mechanism, fuel, oxidizer, phi, temperature, pressure, 
                               rtol=1e-6, atol=1e-8, end_time=1e-1, timestep=1e-5, species_to_track=None) -> Dict[str, Any]:
    """Set up the experimental parameters.
    
    Returns:
        params: Dictionary with experiment parameters
    """
    return {
        'mechanism': mechanism,
        'fuel': fuel,
        'oxidizer': oxidizer,
        'phi': phi,
        'temperature': temperature,
        'pressure': pressure,  # Fixed: was ct.one_atm
        'rtol': rtol,
        'atol': atol,
        'end_time': end_time,
        'timestep': timestep,
        'species_to_track': species_to_track
    }

def run_experiments_and_rank(methods, tolerances, params, fuel, temperature, pressure, time_limit=120.0, base_dir='results'):
    all_results = {}
    for tolerance in tolerances:
        for method in methods:
            print(f"Running {method} with tolerance {tolerance}")
            rtol = tolerance[0]
            atol = tolerance[1]
            name = f"{method}_{rtol}_{atol}"
            
            # Setup chemistry
            gas = setup_combustion_chemistry(
                params['mechanism'], params['fuel'], params['oxidizer'],
                params['phi'], params['temperature'], params['pressure']
            )
            
            # If species_to_track not specified, use all species
            if params['species_to_track'] is None:
                params['species_to_track'] = gas.species_names
            
            # Get initial state
            y0 = get_initial_state(gas)
            t0 = 0.0
            
            # Run experiment
            results = run_integration_experiment(
                method, gas, y0, t0, params['end_time'], params['timestep'],
                rtol, atol, params['species_to_track'],
                fuel, params['pressure'],
                time_limit=time_limit
            )
            all_results[name] = results
    
    # Save the results to a pickle file
    os.makedirs(base_dir, exist_ok=True)

    temp = temperature
    pressure = int(pressure / ct.one_atm)
    phi = params['phi']
    end_time = params['end_time']


    file_base_name = f'{fuel}_{temp}_{pressure}__{phi}__{end_time}'

    file_name = f'{base_dir}/{file_base_name}_results.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(all_results, f)

    solver_data, solution_data = analyze_solver_performance(all_results, methods, tolerances, params['species_to_track'], window_size=100)
    solver_data.to_csv(f'{base_dir}/{file_base_name}_solver_data.csv')
    
    # Create DataFrame to store rankings for each timestep
    columns = ['time'] + [f'rank{i+1}' for i in range(len(all_results))]
    integrator_ranks = pd.DataFrame(columns=columns)
    
    # Get list of all timesteps from first method's results
    first_result = next(iter(all_results.values()))
    times = first_result['times']
    
    # For each timestep, rank methods by CPU time
    for i, t in enumerate(times[:-1]):  # Skip last timestep since we compare with next
        # Get CPU time for each method at this timestep
        method_times = []
        for method_name, results in all_results.items():
            # Only include methods that have data for this timestep
            if i < len(results['cpu_times']):
                method_times.append((method_name, results['cpu_times'][i]))
        
        # Sort methods by CPU time
        ranked_methods = sorted(method_times, key=lambda x: x[1])
        
        # Create row with rankings
        row = {'time': t}
        for rank, (method, _) in enumerate(ranked_methods, 1):
            row[f'rank{rank}'] = method
            
        # Add row to DataFrame
        integrator_ranks.loc[i] = row
    
    # Save the integrator_ranks dataframe to a CSV file
    integrator_ranks.to_csv(f'{base_dir}/{file_base_name}_integrator_ranks.csv')
    return all_results, integrator_ranks

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define experiment parameters
    methods = ['cvode_bdf', 'cpp_rk23']
    tolerances = [(1e-6, 1e-8), (1e-8, 1e-10), (1e-10, 1e-12)]
    
    # Create different params
    fuels = ['nc12h26']
    temperatures = [700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    pressures = [ct.one_atm, 2*ct.one_atm, 5*ct.one_atm, 10*ct.one_atm, 20*ct.one_atm]
    phis = [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    # num_conditions = 100
    end_times = [5e-3]
    timestep = 1e-6
    base_dir = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_dir, exist_ok=True)

    all_conditions = []
    for temperature in temperatures:
        for pressure in pressures:
            for phi in phis:
                for fuel in fuels:
                    for end_time in end_times:
                        all_conditions.append((temperature, pressure, phi, fuel, end_time))

    print(f"Running {len(all_conditions)} experiments")
    for condition in tqdm(all_conditions, desc="Running experiments", total=len(all_conditions), position=0, leave=True, colour='green'):
        temperature, pressure, phi, fuel, end_time = condition
        
        print(f"{'*' * 100}")
        print(f"Running experiment for {fuel} at {temperature} K and {pressure/ct.one_atm} atm with phi = {phi} and end_time = {end_time}")
        print(f"{'*' * 100}")
        
        mechanism = fuel_to_mechanism[fuel]
        oxidizer = 'O2:1, N2:3.76'
        params = setup_experiment_parameters(
            mechanism=mechanism, fuel=fuel, oxidizer=oxidizer, phi=phi, 
            temperature=temperature, pressure=pressure, end_time=end_time, timestep=timestep
        )
        
        all_results, integrator_ranks = run_experiments_and_rank(
            methods, tolerances, params, fuel, temperature, pressure, 
            time_limit=120.0, base_dir=base_dir
        )
        print("-" * 100)

    # Example of how to analyze results after running experiments
    # Uncomment and modify as needed:
    # pickle_file_name = 'results/your_results_file.pkl'
    # species_to_track = ['CO2', 'CO', 'H2O', 'O2']  # Define species of interest
    # solver_data, solution_data = analyze_solver_performance(
    #     pickle_file_name, methods, tolerances, species_to_track, window_size=100
    # )