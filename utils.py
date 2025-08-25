import rk_solver_cpp
from scipy.integrate import ode
import SundialsPy as SP
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import os
from datetime import datetime
import time
from tqdm import tqdm


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
                          gas: ct.Solution, pressure: float, table_id: Optional[SP.arkode.ButcherTable] = None) -> Any:
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
            butcher_table=SP.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5 if table_id is None else table_id
        )
    elif method == 'arkode_dirk':
        solver = SP.arkode.ARKodeSolver(
            system_size=system_size,
            explicit_fn=lambda t, y: combustion_rhs(t, y, gas, pressure),
            implicit_fn=lambda t, y: combustion_rhs(t, y, gas, pressure),
            butcher_table=SP.arkode.ButcherTable.SDIRK_2_1_2 if table_id is None else table_id,
            linsol_type=SP.cvode.LinearSolverType.DENSE
        )
        solver._py_explicit = lambda t, y: combustion_rhs(t, y, gas, pressure)
        solver._py_implicit = lambda t, y: combustion_rhs(t, y, gas, pressure)
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
                 rtol: float, atol: float, t_end: Optional[float] = None, pressure: float = ct.one_atm, table_id: Optional[SP.arkode.ButcherTable] = None) -> Any:
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
        return create_sundials_solver(method, y, t, system_size, rtol, abs_tol, gas, pressure, table_id)
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
                         timestep: float, rtol: float, atol: float, fuel: str, pressure: float=ct.one_atm, table_id: Optional[SP.arkode.ButcherTable] = None) -> Dict[str, Any]:
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
        solver = create_solver(method, gas, y, t, rtol, atol, t_end, pressure=pressure, table_id=table_id)
        
        # Integrate
        start_time = time.time()
        
        if method.startswith('cpp_'):
            result = rk_solver_cpp.solve_ivp(solver, np.array(t_end))
            new_y = result['y'][-1]
            # ensure that new_y is not empty
            if len(new_y) == 0:
                print("new_y is empty")
                print(result)
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
                             fuel: str, pressure: float=ct.one_atm,
                             time_limit: float = 300.0, table_id: Optional[SP.arkode.ButcherTable] = None) -> Dict[str, Any]:
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
    
    bar = tqdm(total=end_time, desc=f"Running {method}-{str(table_id)} with rtol={rtol} and atol={atol}")
    while t < end_time:
        bar.update(timestep)
        # Check if time limit exceeded
        if time.time() - start_time > time_limit:
            print(f"Time limit of {time_limit}s exceeded after {step_count} steps")
            break
            
        result = integrate_single_step(method, gas, y, t, timestep, rtol, atol, fuel, pressure=pressure, table_id=table_id)
        
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

def setup_combustion_chemistry_with_data(mechanism: str,temperature: float, pressure: float, data: np.ndarray) -> ct.Solution:
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
    gas.TPX = temperature, pressure, data
    return gas

def get_initial_state(gas: ct.Solution) -> np.ndarray:
    """Get initial state vector [T, Y1, Y2, ...].
    
    Args:
        gas: Cantera gas object
    
    Returns:
        y: Initial state vector
    """
    return np.hstack([gas.T, gas.Y])

import pickle
from scipy.ndimage import gaussian_filter1d

def detect_ignition_regions(temperature_profile, time_array=None, 
                           gradient_threshold=None, smooth_sigma=1.0,
                           min_ignition_length=5):
    """
    Detect pre-ignition, ignition, and post-ignition regions in a temperature profile.
    
    Parameters:
    -----------
    temperature_profile : array-like
        Temperature values over time
    time_array : array-like, optional
        Time values corresponding to temperature measurements.
        If None, assumes uniform spacing with indices.
    gradient_threshold : float, optional
        Threshold for detecting ignition based on temperature gradient.
        If None, automatically determined from data.
    smooth_sigma : float, default=1.0
        Gaussian smoothing parameter for gradient calculation
    min_ignition_length : int, default=5
        Minimum number of points for ignition region
    
    Returns:
    --------
    tuple : (pre_ignition_end_idx, ignition_start_idx, ignition_end_idx)
        - pre_ignition_end_idx: Last index of pre-ignition region
        - ignition_start_idx: First index of ignition region  
        - ignition_end_idx: Last index of ignition region
        Post-ignition starts at ignition_end_idx + 1
    """
    
    temp = np.array(temperature_profile)
    n_points = len(temp)
    
    if time_array is None:
        time_array = np.arange(n_points)
    else:
        time_array = np.array(time_array)
    
    # Calculate smoothed gradient
    temp_smooth = gaussian_filter1d(temp, sigma=smooth_sigma)
    dt = np.diff(time_array)
    dt = np.append(dt, dt[-1])  # Extend to same length
    gradient = np.gradient(temp_smooth) / dt
    
    # Auto-determine threshold if not provided
    if gradient_threshold is None:
        # Use a multiple of the standard deviation of the gradient
        gradient_std = np.std(gradient)
        gradient_mean = np.mean(gradient)
        gradient_threshold = gradient_mean + 3 * gradient_std
    
    # Find regions where gradient exceeds threshold
    high_gradient_mask = gradient > gradient_threshold
    
    # Find the start and end of the main ignition event
    # Look for the longest continuous region above threshold
    high_gradient_indices = np.where(high_gradient_mask)[0]
    
    if len(high_gradient_indices) == 0:
        # No ignition detected, return boundaries assuming late ignition
        return n_points//3, 2*n_points//3, n_points-1
    
    # Find continuous regions
    diff_indices = np.diff(high_gradient_indices)
    breaks = np.where(diff_indices > 1)[0]
    
    if len(breaks) == 0:
        # Single continuous region
        ignition_start_idx = high_gradient_indices[0]
        ignition_end_idx = high_gradient_indices[-1]
    else:
        # Multiple regions - find the longest one
        region_starts = [high_gradient_indices[0]] + [high_gradient_indices[b+1] for b in breaks]
        region_ends = [high_gradient_indices[b] for b in breaks] + [high_gradient_indices[-1]]
        region_lengths = [end - start for start, end in zip(region_starts, region_ends)]
        
        longest_region_idx = np.argmax(region_lengths)
        ignition_start_idx = region_starts[longest_region_idx]
        ignition_end_idx = region_ends[longest_region_idx]
    
    # Extend ignition region if too short
    if ignition_end_idx - ignition_start_idx < min_ignition_length:
        center = (ignition_start_idx + ignition_end_idx) // 2
        half_length = min_ignition_length // 2
        ignition_start_idx = max(0, center - half_length)
        ignition_end_idx = min(n_points - 1, center + half_length)
    
    # Pre-ignition ends just before ignition starts
    pre_ignition_end_idx = max(0, ignition_start_idx - 1)
    
    # Ensure ignition_end_idx doesn't exceed array bounds
    ignition_end_idx = min(ignition_end_idx, n_points - 1)
    
    return pre_ignition_end_idx, ignition_start_idx, ignition_end_idx


def plot_regions(temperature_profile, time_array=None, region_indices=None):
    """
    Plot temperature profile with detected regions highlighted.
    
    Parameters:
    -----------
    temperature_profile : array-like
        Temperature values
    time_array : array-like, optional
        Time values
    region_indices : tuple, optional
        (pre_ignition_end_idx, ignition_start_idx, ignition_end_idx)
        If None, will detect automatically
    """
    
    if time_array is None:
        time_array = np.arange(len(temperature_profile))
    
    if region_indices is None:
        region_indices = detect_ignition_regions(temperature_profile, time_array)
    
    pre_end, ign_start, ign_end = region_indices
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(time_array, temperature_profile, 'b-', linewidth=2, label='Temperature')
    
    # Highlight regions
    plt.axvspan(time_array[0], time_array[pre_end], alpha=0.3, color='green', 
                label='Pre-ignition')
    plt.axvspan(time_array[ign_start], time_array[ign_end], alpha=0.3, color='red', 
                label='Ignition')
    plt.axvspan(time_array[ign_end], time_array[-1], alpha=0.3, color='blue', 
                label='Post-ignition')
    
    # Add vertical lines at boundaries
    plt.axvline(time_array[pre_end], color='green', linestyle='--', alpha=0.7)
    plt.axvline(time_array[ign_start], color='red', linestyle='--', alpha=0.7)
    plt.axvline(time_array[ign_end], color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Profile with Detected Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def running_average_forward(array, window_size):
    """
    Calculate running average where each point uses the window starting from that point.
    Returns array of same length as input.
    
    Parameters:
    -----------
    array : array-like
        Input array
    window_size : int
        Size of the averaging window
        
    Returns:
    --------
    numpy.ndarray
        Running average array of same length as input
    """
    array = np.array(array)
    n = len(array)
    result = np.zeros(n)
    
    # Calculate number of complete windows
    num_windows = (n + window_size - 1) // window_size
    
    # Process each window
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = min((w + 1) * window_size, n)
        window_mean = np.mean(array[start_idx:end_idx])
        
        # Fill result array for all points that use this window
        result[start_idx:end_idx] = window_mean
        
    return result

def calculate_rmse(ref_data, test_data, species_name, use_log=False):
    rmse_dict = {}
    for specie_name in species_name:
        if specie_name == 'temperature':
            ref_profile = ref_data['temperatures']
            test_profile = test_data['temperatures']
        else:
            ref_profile = ref_data['species_profiles'][specie_name]
            test_profile = test_data['species_profiles'][specie_name]
        if use_log:
            ref_profile = np.log10(np.maximum(ref_profile, 1e-20))
            test_profile = np.log10(np.maximum(test_profile, 1e-20))
        else:
            ref_profile = np.array(ref_profile)
            test_profile = np.array(test_profile)
            size = min(ref_profile.shape[0], test_profile.shape[0])
            ref_profile = ref_profile[:size]
            test_profile = test_profile[:size]
        rmse = np.sqrt((ref_profile- test_profile) ** 2)
        rmse_dict[specie_name] = rmse
    return rmse_dict

