import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import SundialsPy  # Your SUNDIALS wrapper

class CombustionStage(Enum):
    """Different stages of combustion process."""
    PREIGNITION = 0
    IGNITION = 1
    POSTIGNITION = 2

class SundialsIntegratorConfig:
    """Configuration for SUNDIALS-based integrator with Butcher table support."""
    
    def __init__(self, 
                 integrator_list=None, 
                 tolerance_list=None,
                 butcher_tables=None):
        """
        Initialize integrator configuration.
        
        Args:
            integrator_list: List of integrator methods
            tolerance_list: List of (rtol, atol) tuples
            butcher_tables: Dict mapping integrator names to Butcher table options
                           or list of Butcher tables for arkode_erk methods
        """
        # Default integrators: CVODE BDF, CVODE Adams, ARKODE ERK
        if integrator_list is None:
            self.integrator_list = ['cvode_bdf', 'cvode_adams', 'arkode_erk']
        else:
            self.integrator_list = integrator_list
            
        # Default tolerances
        if tolerance_list is None:
            self.tolerance_list = [(1e-6, 1e-8), (1e-12, 1e-14)]
        else:
            self.tolerance_list = tolerance_list
        
        # Handle Butcher table configuration
        if butcher_tables is None:
            # Default Butcher tables for ARKode ERK methods
            self.butcher_tables = {
                'arkode_erk': [
                    SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5,  # Default
                    SundialsPy.arkode.ButcherTable.ZONNEVELD_5_3_4,
                    SundialsPy.arkode.ButcherTable.ARK324L2SA_ERK_4_2_3,
                ]
            }
        else:
            self.butcher_tables = butcher_tables
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        # Check that butcher_tables keys are valid integrator names
        for integrator_name in self.butcher_tables.keys():
            if integrator_name not in ['arkode_erk', 'arkode_dirk', 'arkode_imex']:
                print(f"Warning: Butcher tables specified for non-ARKode integrator: {integrator_name}")
        
        # Ensure ARKode methods have Butcher tables if they're in the integrator list
        arkode_methods = [method for method in self.integrator_list if method.startswith('arkode')]
        for method in arkode_methods:
            if method not in self.butcher_tables:
                print(f"Warning: No Butcher tables specified for {method}, using default")
    
    def get_action_list(self):
        """Get list of integrator/tolerance/butcher_table combinations."""
        action_list = []
        
        for integ in self.integrator_list:
            for rtol, atol in self.tolerance_list:
                if integ.startswith('arkode') and integ in self.butcher_tables:
                    # Add each Butcher table as a separate action for ARKode methods
                    for butcher_table in self.butcher_tables[integ]:
                        action_list.append((integ, rtol, atol, butcher_table))
                else:
                    # For non-ARKode methods, no Butcher table needed
                    action_list.append((integ, rtol, atol, None))
        
        return action_list
    
    @classmethod
    def create_arkode_config(cls, butcher_table_names=None):
        """
        Create a configuration focused on ARKode ERK methods with specific Butcher tables.
        
        Args:
            butcher_table_names: List of Butcher table names to use
            
        Returns:
            SundialsIntegratorConfig: Configuration with specified ARKode options
        """
        # Available Butcher tables (extend this list as needed)
        available_tables = {
            'HEUN_EULER_2_1_2': SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2,
            'BOGACKI_SHAMPINE_4_2_3': SundialsPy.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3,
            'ARK324L2SA_ERK_4_2_3': SundialsPy.arkode.ButcherTable.ARK324L2SA_ERK_4_2_3,
            'ZONNEVELD_5_3_4': SundialsPy.arkode.ButcherTable.ZONNEVELD_5_3_4,
            'ARK548L2SA_ERK_8_4_5': SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5,
            'ARK436L2SA_ERK_6_3_4': SundialsPy.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4,
            'ARK437L2SA_ERK_7_3_4': SundialsPy.arkode.ButcherTable.ARK437L2SA_ERK_7_3_4,
            'ARK548L2SA_ERK_8_4_5': SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5,
            'VERNER_8_5_6': SundialsPy.arkode.ButcherTable.VERNER_8_5_6,
            'FEHLBERG_13_7_8': SundialsPy.arkode.ButcherTable.FEHLBERG_13_7_8,
            'SDIRK_2_1_2': SundialsPy.arkode.ButcherTable.SDIRK_2_1_2,
            'BILLINGTON_3_3_2': SundialsPy.arkode.ButcherTable.BILLINGTON_3_3_2,
            'TRBDF2_3_3_2': SundialsPy.arkode.ButcherTable.TRBDF2_3_3_2,
            'KVAERNO_4_2_3': SundialsPy.arkode.ButcherTable.KVAERNO_4_2_3,
            'ARK324L2SA_DIRK_4_2_3': SundialsPy.arkode.ButcherTable.ARK324L2SA_DIRK_4_2_3,
            'CASH_5_2_4': SundialsPy.arkode.ButcherTable.CASH_5_2_4,
            'CASH_5_3_4': SundialsPy.arkode.ButcherTable.CASH_5_3_4,
            'SDIRK_5_3_4': SundialsPy.arkode.ButcherTable.SDIRK_5_3_4,
            'ARK436L2SA_DIRK_6_3_4': SundialsPy.arkode.ButcherTable.ARK436L2SA_DIRK_6_3_4,
            'ARK437L2SA_DIRK_7_3_4': SundialsPy.arkode.ButcherTable.ARK437L2SA_DIRK_7_3_4,
            'KVAERNO_7_4_5': SundialsPy.arkode.ButcherTable.KVAERNO_7_4_5,
            'ARK548L2SA_DIRK_8_4_5': SundialsPy.arkode.ButcherTable.ARK548L2SA_DIRK_8_4_5,
            'ARK324L2SA_ERK_4_2_3_DIRK_4_2_3': SundialsPy.arkode.ButcherTable.ARK324L2SA_ERK_4_2_3_DIRK_4_2_3,
            'ARK436L2SA_ERK_6_3_4_DIRK_6_3_4': SundialsPy.arkode.ButcherTable.ARK436L2SA_ERK_6_3_4_DIRK_6_3_4,
            'ARK437L2SA_ERK_7_3_4_DIRK_7_3_4': SundialsPy.arkode.ButcherTable.ARK437L2SA_ERK_7_3_4_DIRK_7_3_4,
            'ARK548L2SA_ERK_8_4_5_DIRK_8_4_5': SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5_DIRK_8_4_5
        }
        
        if butcher_table_names is None:
            # Use a good selection of tables
            butcher_table_names = ['ARK548L2SA_ERK_8_4_5', 'ZONNEVELD_5_3_4', 'DORMAND_PRINCE_7_4_5']
        
        # Get the actual Butcher table objects
        butcher_tables_list = []
        for name in butcher_table_names:
            if name in available_tables:
                butcher_tables_list.append(available_tables[name])
            else:
                print(f"Warning: Unknown Butcher table '{name}', skipping")
        
        return cls(
            integrator_list=['arkode_erk'],
            tolerance_list=[(1e-6, 1e-8), (1e-9, 1e-11), (1e-12, 1e-14)],
            butcher_tables={'arkode_erk': butcher_tables_list}
        )
    
    def add_butcher_table(self, integrator_name, butcher_table):
        """Add a Butcher table to an integrator configuration."""
        if integrator_name not in self.butcher_tables:
            self.butcher_tables[integrator_name] = []
        self.butcher_tables[integrator_name].append(butcher_table)
    
    def get_method_info(self, action_idx):
        """Get detailed information about a specific action."""
        actions = self.get_action_list()
        if action_idx >= len(actions):
            raise IndexError(f"Action index {action_idx} out of range")
        
        action = actions[action_idx]
        if len(action) == 4:
            method, rtol, atol, butcher_table = action
            return {
                'method': method,
                'rtol': rtol,
                'atol': atol,
                'butcher_table': butcher_table,
                'description': f"{method} with rtol={rtol}, atol={atol}, Butcher table={butcher_table}"
            }
        else:
            method, rtol, atol = action
            return {
                'method': method,
                'rtol': rtol,
                'atol': atol,
                'butcher_table': None,
                'description': f"{method} with rtol={rtol}, atol={atol}"
            }


class SundialsChemicalIntegrator:
    """Handles integration of chemical kinetics using SUNDIALS solvers."""
    
    def __init__(self, 
                 mechanism_file, 
                 temperature,
                 pressure,
                 fuel,
                 oxidizer='O2:1.0, N2:3.76',
                 phi=1.0,
                 timestep=1e-5,
                 state_change_threshold=1e-3,
                 species_to_track=None,
                 config=None):
        """Initialize SUNDIALS-based integrator.
        
        Args:
            mechanism_file: Path to Cantera mechanism file
            temperature: Initial temperature in K
            pressure: Initial pressure in Pa
            fuel: Fuel composition
            oxidizer: Oxidizer composition
            phi: Equivalence ratio
            timestep: Integration timestep
            state_change_threshold: Threshold for detecting stage changes
            species_to_track: List of species to track
            config: Integrator configuration
        """
        # Initialize Cantera
        self.gas = ct.Solution(mechanism_file)
        
        # Store simulation parameters
        self.mechanism_file = mechanism_file
        self.temperature = temperature
        self.pressure = pressure
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.phi = phi
        self.timestep = timestep
        self.state_change_threshold = state_change_threshold
        
        # Species to track
        if species_to_track is None:
            # Default to major species
            self.species_to_track = ['H2', 'O2', 'H2O', 'OH', 'CO', 'CO2', 'CH4']
            # Filter to those present in the mechanism
            self.species_to_track = [s for s in self.species_to_track 
                                     if s in self.gas.species_names]
        else:
            self.species_to_track = species_to_track
        
        # Set up integrator configuration
        self.config = config or SundialsIntegratorConfig()
        self.action_list = self.config.get_action_list()
        
        # Initialize system state
        self.reset()
        
        # Set up reference solution (can be overridden later)
        self.reference_solution = None
        self.completed_steps = 1000  # Default max steps
        
    def reset(self):
        """Reset the integrator state."""
        # Reset history
        self.history = {
            'times': [],
            'states': [],
            'temperatures': [],
            'pressures': [],
            'species_profiles': {spec: [] for spec in self.species_to_track},
            'cpu_times': [],
            'actions_taken': [],
            'success_flags': [],
            'errors': [],
            'stages': [],
            'stage_values': []
        }
        
        # Reset state variables
        self.step_count = 0
        self.current_stage = CombustionStage.PREIGNITION
        self.t = 0.0
        
        # Reset the gas state using equivalence ratio
        self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
        self.initial_mixture = self.gas.X
        self.gas.TPX = self.temperature, self.pressure, self.initial_mixture
        
        # Initial state vector = [T, Y]
        self.y = np.hstack([self.gas.T, self.gas.Y])
        
        # Store initial state
        self._store_state(self.y, 0.0, None, True, 0.0, 0.0, self.current_stage, 0.0)
        
        # Track stage changes
        self.stage_changes = [False]
        self.stage_steps = {stage.value: 0 for stage in CombustionStage}
        self.stage_cpu_times = {stage.value: 0.0 for stage in CombustionStage}
        
        # End flag
        self.end_simulation = False
        
        # Current active solver
        self.current_solver = None
    
    def dydt(self, t, y):
        """Compute ODE right-hand side - dy/dt for the combustion system.
        
        Args:
            t: Current time
            y: Current state vector [T, Y1, Y2, ...]
            
        Returns:
            dydt: Time derivatives [dT/dt, dY1/dt, dY2/dt, ...]
        """
        # Extract temperature and mass fractions
        T = y[0]
        Y = y[1:]
        
        # Update the gas state
        self.gas.TPY = T, self.pressure, Y
        
        # Get thermodynamic properties
        rho = self.gas.density_mass
        wdot = self.gas.net_production_rates
        cp = self.gas.cp_mass
        h = self.gas.partial_molar_enthalpies
        
        # Calculate temperature derivative (energy equation)
        dTdt = -(np.dot(h, wdot) / (rho * cp))
        
        # Calculate species derivatives (mass conservation)
        dYdt = wdot * self.gas.molecular_weights / rho
        
        # Combine into full derivative vector
        return np.hstack([dTdt, dYdt])
    
    def create_solver(self, method, rtol, atol, butcher_table=None):
        """Create a SUNDIALS solver for the combustion problem.
        
        Args:
            method: Solver method ('cvode_bdf', 'cvode_adams', 'arkode_erk', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            butcher_table: Butcher table for ARKode
        Returns:
            solver: Initialized SUNDIALS solver
        """
        # System size = 1 (temperature) + number of species
        system_size = 1 + self.gas.n_species
        
        # Create array for absolute tolerance
        if np.isscalar(atol):
            abs_tol = np.ones(system_size) * atol
        else:
            abs_tol = np.asarray(atol)
            if len(abs_tol) == 1:
                abs_tol = np.ones(system_size) * abs_tol[0]
        
        # Create the appropriate solver
        if method == 'cvode_bdf':
            solver = SundialsPy.cvode.CVodeSolver(
                system_size=system_size,
                rhs_fn=self.dydt,
                iter_type=SundialsPy.cvode.IterationType.NEWTON
            )
        elif method == 'cvode_adams':
            solver = SundialsPy.cvode.CVodeSolver(
                system_size=system_size,
                rhs_fn=self.dydt,
                iter_type=SundialsPy.cvode.IterationType.FUNCTIONAL
            )
        elif method == 'arkode_erk':
            # Use the provided Butcher table or default
            table = butcher_table if butcher_table else SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5
            solver = SundialsPy.arkode.ARKodeSolver(
                system_size=system_size,
                explicit_fn=self.dydt,
                implicit_fn=None,
                butcher_table=table
            )
        else:
            raise ValueError(f"Unknown solver method: {method}")
        
        # Initialize the solver
        solver.initialize(self.y, self.t, rtol, abs_tol)
        
        return solver
    
    def integrate_step(self, action_idx, time_limit=3.0):
        """Perform one integration step with the selected solver.
        
        Args:
            action_idx: Index of action (solver/tolerance) from action_list
            time_limit: Maximum time allowed for integration
            
        Returns:
            result: Dictionary with integration results
        """
        method, rtol, atol, butcher_table = self.action_list[action_idx]
        t_end = self.t + self.timestep
        previous_state = self.y.copy()
        
        try:
            # Create and initialize the solver
            solver = self.create_solver(method, rtol, atol, butcher_table)
            self.current_solver = solver
            
            # Perform the integration
            start_time = time.time()
            new_y = solver.solve_single(t_end)
            cpu_time = time.time() - start_time
            
            # Calculate error if reference solution exists
            if self.reference_solution and self.step_count < len(self.reference_solution['temperatures']):
                ref_T = self.reference_solution['temperatures'][self.step_count]
                T_current = new_y[0]
                error = abs(T_current/self.temperature - ref_T/self.temperature)
            else:
                error = 0.0
            
            # Check for significant state changes
            stage_change, stage_value = self._state_changed_significantly(previous_state, new_y)
            self._store_state(new_y, t_end, action_idx, True, cpu_time, error, 
                             self.current_stage, stage_value)
            self.stage_changes.append(stage_change)
            
            # Handle stage transitions
            if len(self.stage_changes) >= 2 and self.stage_changes[-1] != self.stage_changes[-2]:
                if self.current_stage == CombustionStage.PREIGNITION:
                    self.stage_steps[self.current_stage.value] = self.step_count
                    self.stage_cpu_times[self.current_stage.value] += np.sum(self.history['cpu_times'])
                    print(f"State changed to IGNITION at step {self.step_count}")
                    self.current_stage = CombustionStage.IGNITION
                elif self.current_stage == CombustionStage.IGNITION:
                    self.stage_steps[self.current_stage.value] = self.step_count
                    self.stage_cpu_times[self.current_stage.value] += (
                        np.sum(self.history['cpu_times']) - 
                        self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                    )
                    print(f"State changed to POSTIGNITION at step {self.step_count}")
                    self.current_stage = CombustionStage.POSTIGNITION 
            
            # Handle end conditions        
            if self.current_stage == CombustionStage.POSTIGNITION:
                if (self.stage_steps[CombustionStage.IGNITION.value] > 0 and 
                    self.step_count > 2 * self.stage_steps[CombustionStage.IGNITION.value]):
                    print(f"Stopping simulation at step {self.step_count}")
                    self.end_simulation = True
                    self.stage_cpu_times[self.current_stage.value] += (
                        np.sum(self.history['cpu_times']) - 
                        self.stage_cpu_times[CombustionStage.IGNITION.value] - 
                        self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                    )
            
            # Update step count
            self.step_count += 1
            
            # Update current state
            self.t = t_end
            self.y = new_y
            
            # Check if we've reached the maximum number of steps
            if self.step_count >= self.completed_steps:
                self.end_simulation = True
                print(f"Ending simulation at step {self.step_count}")
                self.stage_cpu_times[self.current_stage.value] += (
                    np.sum(self.history['cpu_times']) - 
                    self.stage_cpu_times[CombustionStage.IGNITION.value] - 
                    self.stage_cpu_times[CombustionStage.PREIGNITION.value]
                )
            
            # Return results
            return {
                'success': True,
                'y': new_y,
                'cpu_time': cpu_time,
                'error': error,
                'message': 'Success',
                'current_stage': self.current_stage,
                'end_simulation': self.end_simulation,
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': False,
                'previous_state': previous_state
            }
            
        except Exception as e:
            print(f"Error in integration step: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'y': previous_state,
                'cpu_time': 0.0,
                'error': float('inf'),
                'message': str(e),
                'current_stage': self.current_stage,
                'end_simulation': False,
                'stage_cpu_times': self.stage_cpu_times,
                'timed_out': False,
                'previous_state': previous_state
            }
    
    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant."""
        if isinstance(previous_state, list):
            previous_state = np.array(previous_state)
        if isinstance(current_state, list):
            current_state = np.array(current_state)
            
        state_change = np.linalg.norm(current_state - previous_state)
        return state_change > self.state_change_threshold, state_change
    
    def _store_state(self, y, t, action_idx, success, cpu_time, error, stage, stage_value):
        """Store the current state and integration results."""
        self.history['times'].append(t)
        self.history['states'].append(y.copy())
        self.history['temperatures'].append(y[0])
        
        # Update gas state to get pressure
        self.gas.TPY = y[0], self.pressure, y[1:]
        self.history['pressures'].append(self.gas.P)
        
        # Store species information
        for spec in self.species_to_track:
            idx = self.gas.species_index(spec)
            self.history['species_profiles'][spec].append(y[idx + 1])
        
        # Store integration info (if this was an actual integration step)
        if action_idx is not None:
            self.history['actions_taken'].append(action_idx)
            self.history['success_flags'].append(success)
            self.history['cpu_times'].append(cpu_time)
            self.history['errors'].append(error)
            self.history['stages'].append(stage)
            self.history['stage_values'].append(stage_value)
    
    def solve(self, end_time, action_idx=0, n_points=None):
        """Solve the combustion problem up to a specified end time.
        
        Args:
            end_time: Final simulation time
            action_idx: Index of the solver/tolerance to use
            n_points: Number of output points (if None, use adaptive stepping)
            
        Returns:
            history: Dictionary with simulation results
        """
        if n_points:
            # Fixed output points
            times = np.linspace(self.t + 1e-10, end_time, n_points)
            
            method, rtol, atol, butcher_table = self.action_list[action_idx]
            print(f"Solving with {method}, rtol={rtol}, atol={atol}")
            
            # Create and initialize solver
            solver = self.create_solver(method, rtol, atol, butcher_table)
            
            # Solve and time it
            start_time = time.time()
            states = solver.solve(times)
            cpu_time = time.time() - start_time
            
            # Include initial state
            full_times = np.concatenate(([self.t], times))
            full_states = np.vstack([self.y.reshape(1, -1), states])
            
            # Process results
            for i, (t, state) in enumerate(zip(times, states)):
                step_time = cpu_time / n_points  # Approximate
                self._store_state(state, t, action_idx, True, step_time, 0.0, 
                                 self.current_stage, 0.0)
                
            print(f"Solution completed in {cpu_time:.4f} seconds")
            
            # Update current state
            self.t = end_time
            self.y = states[-1]
            
            # Get solver statistics (if available)
            try:
                stats = solver.get_stats()
                print("Solver statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Error getting solver statistics: {e}")
            
            return self.history
        else:
            # Adaptive stepping
            print(f"Solving with adaptive stepping to t={end_time}")
            
            while self.t < end_time and not self.end_simulation:
                # Calculate next timestep (don't overshoot end_time)
                next_dt = min(self.timestep, end_time - self.t)
                if next_dt <= 0:
                    print(f"Reached end_time at t={self.t}")
                    break
                
                # Adjust timestep
                self.timestep = next_dt
                
                # Perform integration step
                result = self.integrate_step(action_idx)
                
                if not result['success']:
                    print(f"Integration failed at t={self.t}: {result['message']}")
                    break
            
            print(f"Solution completed in {np.sum(self.history['cpu_times']):.4f} seconds")
            return self.history
            
    def plot_results(self, save_path=None, show_plot=True):
        """Plot the simulation results."""
        times = np.array(self.history['times'])
        
        # Create a multi-panel figure
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot temperature
        axs[0].plot(times, self.history['temperatures'], 'r-', linewidth=2)
        axs[0].set_ylabel('Temperature (K)')
        axs[0].set_title('Temperature Evolution')
        axs[0].grid(True)
        
        # Plot pressure
        axs[1].plot(times, np.array(self.history['pressures'])/1e5, 'b-', linewidth=2)
        axs[1].set_ylabel('Pressure (bar)')
        axs[1].set_title('Pressure Evolution')
        axs[1].grid(True)
        
        # Plot species
        for species, profile in self.history['species_profiles'].items():
            axs[2].semilogy(times, profile, label=species)
        
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Mass Fraction')
        axs[2].set_title('Species Evolution')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_ignition_delay(self, criterion='temperature', threshold=None):
        """Calculate ignition delay time.
        
        Args:
            criterion: 'temperature', 'species', or 'pressure'
            threshold: Threshold value (if None, use gradient method)
            
        Returns:
            t_ign: Ignition delay time
        """
        times = np.array(self.history['times'])
        
        if criterion == 'temperature':
            profile = np.array(self.history['temperatures'])
            
            if threshold:
                # Find first time when temperature exceeds threshold
                for i, T in enumerate(profile):
                    if T > threshold:
                        return times[i]
                return None  # Threshold not reached
            else:
                # Use maximum gradient method
                dTdt = np.gradient(profile, times)
                idx = np.argmax(dTdt)
                return times[idx]
        
        elif criterion == 'pressure':
            profile = np.array(self.history['pressures'])
            
            if threshold:
                # Find first time when pressure exceeds threshold
                for i, P in enumerate(profile):
                    if P > threshold:
                        return times[i]
                return None  # Threshold not reached
            else:
                # Use maximum gradient method
                dPdt = np.gradient(profile, times)
                idx = np.argmax(dPdt)
                return times[idx]
        
        elif criterion == 'species':
            if not threshold or not isinstance(threshold, tuple):
                raise ValueError("For species criterion, threshold must be a tuple (species_name, value)")
            
            species_name, value = threshold
            if species_name not in self.species_to_track:
                raise ValueError(f"Species {species_name} not tracked")
            
            profile = np.array(self.history['species_profiles'][species_name])
            
            # Find first time when species exceeds threshold
            for i, Y in enumerate(profile):
                if Y > value:
                    return times[i]
            return None  # Threshold not reached
        
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def get_statistics(self):
        """Get integration statistics."""
        return {
            'total_cpu_time': sum(self.history['cpu_times']),
            'average_cpu_time': np.mean(self.history['cpu_times']),
            'max_cpu_time': np.max(self.history['cpu_times']),
            'average_error': np.mean(self.history['errors']),
            'max_error': np.max(self.history['errors']),
            'success_rate': np.mean(self.history['success_flags']),
            'num_steps': len(self.history['times']) - 1,
            'final_temperature': self.history['temperatures'][-1],
            'ignition_delay': self.get_ignition_delay()
        }


# Example usage
if __name__ == "__main__":
    # Create a simple hydrogen-air ignition problem
    integrator = SundialsChemicalIntegrator(
        mechanism_file="h2o2.yaml",  # Use hydrogen mechanism
        temperature=1200.0,         # Initial temperature (K)
        pressure=101325.0,          # Initial pressure (Pa)
        fuel="H2",                  # Hydrogen fuel
        phi=1.0,                    # Stoichiometric mixture
        timestep=1e-6               # Initial timestep
    )
    
    # Solve with CVODE BDF
    integrator.solve(end_time=1e-3, action_idx=0, n_points=100)
    
    # Calculate ignition delay time
    t_ign = integrator.get_ignition_delay()
    print(f"Ignition delay time: {t_ign:.2e} seconds")
    
    # Plot results
    integrator.plot_results(save_path="hydrogen_ignition.png")
    
    # Try a more complex problem
    # This will use adaptive stepping
    methane_integrator = SundialsChemicalIntegrator(
        mechanism_file="gri30.yaml",  # Use GRI-Mech 3.0
        temperature=1200.0,         # Initial temperature (K)
        pressure=101325.0,          # Initial pressure (Pa)
        fuel="CH4",                 # Methane fuel
        phi=0.6,                    # Lean mixture
        timestep=1e-7               # Smaller timestep for stiff problem
    )
    
    # Solve with CVODE BDF using adaptive stepping
    methane_integrator.solve(end_time=1e-2, action_idx=0)
    
    # Plot results
    methane_integrator.plot_results(save_path="methane_ignition.png")