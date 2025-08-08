from combustion import CombustionStage, SundialsChemicalIntegrator, SundialsIntegratorConfig
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import SundialsPy  # Your SUNDIALS wrapper
from tqdm import tqdm

# Enums and Constants
class CombustionStage(Enum):
    """Different stages of combustion process."""
    PREIGNITION = 0
    IGNITION = 1
    POSTIGNITION = 2

# Core Problem Class
class CombustionProblem:
    """Defines and manages a combustion problem including reference solution computation."""

    def __init__(self, 
                 mech_file: str,
                 temperature: float,
                 pressure: float,
                 phi: float,
                 fuel: str = 'CH4',
                 oxidizer: str = 'O2:1, N2:3.76',
                 end_time: float = 1e-3,
                 timestep: float = 1e-6,
                 species_to_track: Optional[List[str]] = None,
                 initial_mixture: str = None,
                 reference_rtol: float = 1e-10,
                 reference_atol: float = 1e-20,
                 state_change_threshold: float = 1,
                 normalizing_temperature: float = 1000,
                 verbose: bool = True
                 ):
        """Initialize combustion problem."""
        self._initialize_parameters(locals())
        self._setup_cantera()
        self._initialize_tracking()
        if self.verbose:
            print(f"Combustion problem initialized with T={self.temperature}, P={self.pressure}, phi={self.phi} and timestep={self.timestep}")
        self._compute_reference_solution()

    def _initialize_parameters(self, params):
        """Initialize all parameters from constructor."""
        for key, value in params.items():
            if key != 'self':
                setattr(self, key, value)
        self.num_steps = int(self.end_time / self.timestep)
        self.ignition_delay = 0.0
        self.reference_solution = None
        self.current_state = CombustionStage.PREIGNITION

    def _setup_cantera(self):
        """Initialize Cantera solution and mixture."""
        try:
            self.gas = ct.Solution(self.mech_file)
            if self.initial_mixture is None:
                self.initial_mixture = f"{self.fuel}:1, {self.oxidizer}"
            self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
            self.gas.TPX = self.temperature, self.pressure, self.initial_mixture
        except Exception as e:
            print(f"Error in Cantera setup: {e}")
            raise e

    def _initialize_tracking(self):
        """Initialize species tracking."""
        if self.species_to_track is None:
            self.species_to_track = ['H2', 'O2', 'H', 'OH', 'H2O', 'HO2', 'H2O2']

    def _state_changed_significantly(self, previous_state, current_state):
        """Check if state change is significant."""
        previous_state = np.array(previous_state) if isinstance(previous_state, list) else previous_state
        current_state = np.array(current_state) if isinstance(current_state, list) else current_state
        state_change = np.linalg.norm(current_state - previous_state)
        #return state_change > self.state_change_threshold, state_change
        return False, state_change

    def _compute_reference_solution(self) -> None:
        """Compute reference solution using Cantera."""
        reactor = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([reactor])
        sim.rtol, sim.atol = self.reference_rtol, self.reference_atol

        results = self._run_simulation(reactor, sim)
        self._store_reference_solution(results)

    def _run_simulation(self, reactor, sim):
        """Run the actual simulation."""
        times = np.zeros(self.num_steps)
        temperatures = np.zeros(self.num_steps)
        pressures = np.zeros(self.num_steps)
        species_profiles = {spec: np.zeros(self.num_steps) for spec in self.species_to_track}
        
        start_time = time.time()
        stage_steps = {stage: 0 for stage in CombustionStage}
        stage_changes = [False]
        
        for i in tqdm(range(self.num_steps), desc="Computing reference solution", disable=not self.verbose):
            if not self._process_simulation_step(i, reactor, sim, times, temperatures, pressures, species_profiles, stage_steps, stage_changes):
                break
                
        return {
            'times': times,
            'temperatures': temperatures,
            'pressures': pressures,
            'species_profiles': species_profiles,
            'computation_time': time.time() - start_time,
            'completed_steps': i
        }

    def _process_simulation_step(self, i, reactor, sim, times, temperatures, pressures, species_profiles, stage_steps, stage_changes):
        """Process a single simulation step."""
        previous_state = reactor.thermo.state
        t = i * self.timestep
        sim.advance(t)
        
        # Record data
        times[i] = t
        temperatures[i] = reactor.T
        pressures[i] = reactor.thermo.P
        for spec in self.species_to_track:
            species_profiles[spec][i] = float(reactor.thermo[spec].Y[0])
            
        # Check state changes
        state_changed, state_change = self._state_changed_significantly(previous_state, reactor.thermo.state)
        stage_changes.append(state_changed)
        
        return self._handle_stage_transitions(i, stage_changes, stage_steps)

    def _handle_stage_transitions(self, i, stage_changes, stage_steps):
        """Handle transitions between combustion stages."""
        if stage_changes[-1] != stage_changes[-2]:
            if self.current_state == CombustionStage.PREIGNITION:
                stage_steps[self.current_state] = i
                self.current_state = CombustionStage.IGNITION
                self.ignition_delay = i * self.timestep
            elif self.current_state == CombustionStage.IGNITION:
                stage_steps[self.current_state] = i
                self.current_state = CombustionStage.POSTIGNITION
        
        if self.current_state == CombustionStage.POSTIGNITION and i > 4 * stage_steps.get(CombustionStage.IGNITION, 0):
            self.completed_steps = i
            if self.verbose:
                print(f"Postignition stage completed at step {i}")
            return False
        return True

    def _store_reference_solution(self, results):
        """Store the computed reference solution."""
        self.completed_steps = results['completed_steps']
        self.reference_solution = {
            'times': results['times'],
            'temperatures': results['temperatures'],
            'pressures': results['pressures'],
            'species_profiles': results['species_profiles'],
            'computation_time': results['computation_time']
        }
        if self.verbose:
            print(f"Reference solution computed in {results['computation_time']:.2f} seconds")

    # Public Interface Methods
    def get_reference_solution(self) -> Dict[str, Any]:
        """Get the pre-computed reference solution."""
        if self.reference_solution is None:
            self._compute_reference_solution()
        return self.reference_solution

    def get_initial_state(self) -> Dict[str, Any]:
        """Get initial state dictionary."""
        return {'T': self.temperature, 'P': self.pressure, 'X': self.initial_mixture}

    def get_problem_params(self) -> Dict[str, Any]:
        """Get problem parameters dictionary."""
        return {
            'end_time': self.end_time,
            'timestep': self.timestep,
            'num_steps': self.num_steps,
            'species_to_track': self.species_to_track,
            'mech_file': self.mech_file,
            'phi': self.phi,
            'fuel': self.fuel,
            'oxidizer': self.oxidizer
        }

    def plot_reference_solution(self, save_path: Optional[str] = None) -> None:
        """Plot the reference solution."""
        if self.reference_solution is None:
            print("No reference solution available")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        times = self.reference_solution['times'][:self.completed_steps]
        
        # Temperature plot
        ax1.plot(times, self.reference_solution['temperatures'][:self.completed_steps])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Temperature Evolution')
        
        # Species plot
        for species, profile in self.reference_solution['species_profiles'].items():
            ax2.plot(times, profile[:self.completed_steps], label=species)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mass Fraction')
        ax2.set_title('Species Evolution')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

# Problem Setup Helper Function
def setup_problem(temperature_range: np.ndarray, 
                 pressure_range: np.ndarray, 
                 phi_range: np.ndarray, 
                 mech_file: str,
                 fuel: str = 'CH4',
                 oxidizer: str = 'O2:1, N2:3.76',
                 species_to_track: Optional[List[str]] = None,
                 end_time: float = 1e-3,
                 reference_rtol: float = 1e-10,
                 reference_atol: float = 1e-20,
                 state_change_threshold: float = 1,
                 min_time_steps_range: Tuple[int, int] = (1e-6, 1e-5),
                 max_time_steps_range: Tuple[int, int] = (1e-4, 1e-3),
                 randomize: bool = True,
                 verbose: bool = True,
                 fixed_temperature = None,
                 fixed_pressure = None,
                 fixed_phi = None,
                 fixed_dt = None,
                 initial_mixture = None
                 ):
    """Create a new combustion problem with either random or fixed parameters."""
    params = _determine_problem_parameters(
        temperature_range, pressure_range, phi_range,
        randomize, fixed_temperature, fixed_pressure, fixed_phi,
        fixed_dt, min_time_steps_range, max_time_steps_range
    )
    
    return CombustionProblem(
        mech_file=mech_file,
        temperature=params['temperature'],
        pressure=params['pressure'] * ct.one_atm,
        phi=params['phi'],
        fuel=fuel,
        oxidizer=oxidizer,
        end_time=params['end_time'],
        timestep=params['timestep'],
        species_to_track=species_to_track,
        initial_mixture=initial_mixture,
        reference_rtol=reference_rtol,
        reference_atol=reference_atol,
        state_change_threshold=state_change_threshold,
        verbose=verbose
    )

def _determine_problem_parameters(temperature_range, pressure_range, phi_range,
                                randomize, fixed_temperature, fixed_pressure, fixed_phi,
                                fixed_dt, min_time_steps_range, max_time_steps_range):
    """Helper function to determine problem parameters based on inputs."""
    if randomize:
        temperature = float(np.random.choice(temperature_range))
        pressure = float(np.random.choice(pressure_range))
        phi = float(np.random.choice(phi_range))
    else:
        temperature = fixed_temperature if fixed_temperature is not None else temperature_range[0]
        pressure = fixed_pressure if fixed_pressure is not None else pressure_range[0]
        phi = fixed_phi if fixed_phi is not None else phi_range[0]
    
    timestep = fixed_dt if fixed_dt is not None else (
        np.random.choice(max_time_steps_range) if temperature <= 1000 else np.random.choice(min_time_steps_range)
    )
    
    end_time = 0.05 if temperature <= 1000 else 1e-3
    
    return {
        'temperature': temperature,
        'pressure': pressure,
        'phi': phi,
        'timestep': timestep,
        'end_time': end_time
    }


if __name__ == "__main__":
    species_to_track = ['H2', 'O2', 'H', 'OH', 'H2O', 'HO2', 'H2O2']
    initial_mixture = None #'nc12h26:1, O2:18.5, N2:69.56'  # Increased O2 and N2 for stoichiometric combustion
    reference_rtol = 1e-10
    reference_atol = 1e-20
    state_change_threshold = 1
    verbose = True

    problem = CombustionProblem(
        mech_file="large_mechanism/ch4_53species.yaml",
        temperature=1100, 
        pressure=10 * ct.one_atm,
        phi=2,
        fuel="CH4",
        oxidizer="O2:1, N2:3.76",
        end_time=1e-2,  
        timestep=1e-6,
        species_to_track=species_to_track,
        initial_mixture=initial_mixture,
        reference_rtol=reference_rtol,
        reference_atol=reference_atol,
        state_change_threshold=state_change_threshold,
        verbose=verbose
    )
    problem.plot_reference_solution()