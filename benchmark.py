import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
try:
    from qss_integrator import QssIntegrator, PyQssOde
except ImportError:
    QssIntegrator = None
    PyQssOde = None
import time
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm
import seaborn as sns

# Import your existing integration modules
from utils import *

class CanteraQSSODE:
    """QSS ODE wrapper for Cantera gas objects."""
    
    def __init__(self, gas, pressure):
        self.gas = gas
        self.pressure = pressure
        self.n_species = self.gas.n_species
        # caches for corrector
        self._T_cache = None
        self._rho_cache = None
        self._cp_cache = None
        self._hform_cache = None

    def __call__(self, t, y, corrector=False):
        # unpack state
        T_in = max(y[0], 300.0)
        Y = np.maximum(np.array(y[1:], dtype=float), 0.0)
        s = Y.sum()
        if s > 1e-12: Y /= s

        if not corrector:
            # predictor: set state and compute thermo
            self.gas.TPY = T_in, self.pressure, Y
            rho = self.gas.density
            cp  = self.gas.cp_mass
            h_form = self.gas.standard_enthalpies_RT * ct.gas_constant * self.gas.T
            # cache for the corrector
            self._T_cache = self.gas.T
            self._rho_cache = rho
            self._cp_cache  = cp
            self._hform_cache = h_form
        else:
            # corrector: freeze T/thermo like C++
            T_frozen = self._T_cache if self._T_cache is not None else T_in
            self.gas.TPY = T_frozen, self.pressure, Y
            rho = self._rho_cache if self._rho_cache is not None else self.gas.density
            cp  = self._cp_cache  if self._cp_cache  is not None else self.gas.cp_mass
            h_form = self._hform_cache if self._hform_cache is not None \
                    else self.gas.standard_enthalpies_RT * ct.gas_constant * T_frozen

        # rates (ensure nonnegative split)
        wQ = np.maximum(self.gas.creation_rates,   0.0)  # kmol/m^3/s
        wD = np.maximum(self.gas.destruction_rates, 0.0) # kmol/m^3/s
        net = wQ - wD
        qdot = -np.dot(net, h_form)  # J/m^3/s

        # temperature parts
        dTdt_q = qdot/(rho*cp)
        dTdt_d = 0.0

        # species parts (mass-fraction rates)
        W = self.gas.molecular_weights         # kg/kmol
        dYdt_q = wQ * W / rho
        dYdt_d = wD * W / rho

        q = np.concatenate(([dTdt_q], dYdt_q))
        d = np.concatenate(([dTdt_d], dYdt_d))
        return q.tolist(), d.tolist()


def setup_qss_integrator(gas: ct.Solution, pressure: float, initial_state: List[float],
                        epsmin: float = 2e-2, epsmax: float = 10.0,
                        dtmin: float = 1e-16, dtmax: float = 1e-6,
                        itermax: int = 2, abstol: float = 1e-11) -> Tuple[QssIntegrator, CanteraQSSODE]:
    """Setup QSS integrator with given parameters."""
    
    chem = CanteraQSSODE(gas, pressure)
    integrator = QssIntegrator()
    ode = PyQssOde(chem)

    # Setup integrator
    integrator.setOde(ode)
    integrator.initialize(chem.n_species + 1)  # T + all species

    # QSS integrator settings
    integrator.epsmin = epsmin
    integrator.epsmax = epsmax
    integrator.dtmin = dtmin
    integrator.dtmax = dtmax
    integrator.itermax = itermax
    integrator.abstol = abstol
    integrator.stabilityCheck = False

    integrator.setState(initial_state, 0.0)
    
    return integrator, chem


def run_qss_integration(gas: ct.Solution, temp: float, pressure: float, 
                       t_final: float, dt_output: float, fuel: str,
                       species_to_track: List[str], time_limit: float = 300.0,
                       **qss_params) -> Dict[str, Any]:
    """Run QSS integration experiment."""
    print(f"Running QSS integration with t_final {t_final} and dt_output {dt_output}")
    # Setup initial state
    initial_state = [temp] + gas.Y.tolist()
    
    # Setup integrator
    integrator, chem = setup_qss_integrator(gas, pressure, initial_state, **qss_params)
    print(f"Setup integrator with {integrator.epsmin} {integrator.epsmax} {integrator.dtmin} {integrator.dtmax} {integrator.itermax} {integrator.abstol} {integrator.stabilityCheck}")
    # Integration with output control
    times = [0.0]
    states = [initial_state.copy()]
    n_outputs = int(t_final / dt_output)
    cpu_times = []
    print(f"Setup initial state with {initial_state}")
    start_time = time.time()
    with tqdm(total=n_outputs, desc="Running QSS") as bar:
        for i in range(1, n_outputs + 1):
            # Check time limit
            if time.time() - start_time > time_limit:
                print(f"QSS integration timed out after {time.time() - start_time:.1f}s")
                break
                
            t_target = i * dt_output
            step_start = time.time()
            result = integrator.integrateToTime(t_target)
            cpu_times.append(time.time() - step_start)
            
            if result != 0:
                print(f"QSS integration failed at t = {t_target*1000:.2f} ms")
                break
            
            times.append(integrator.tn)
            states.append(integrator.y.copy())
            
            bar.update(1)
            bar.set_postfix({
                'T': f"{integrator.y[0]:.1f}K",
                'cpu_time': f"{cpu_times[-1]:.2e}s",
                'total_cpu': f"{np.sum(cpu_times):.2e}s"
            })
    
    total_wall_time = time.time() - start_time
    
    # Extract data
    temperatures = [s[0] for s in states]
    species_profiles = {}
    fuel_mass_fractions = []
    
    for species in species_to_track:
        try:
            species_idx = gas.species_index(species) + 1  # +1 for temperature
            species_profiles[species] = [s[species_idx] for s in states]
        except:
            species_profiles[species] = [0.0] * len(states)
    
    # Get fuel mass fractions
    fuel_idx = gas.species_index(fuel) + 1 if fuel in gas.species_names else None
    if fuel_idx:
        fuel_mass_fractions = [s[fuel_idx] for s in states]
    else:
        fuel_mass_fractions = [0.0] * len(states)
    
    return {
        'method': 'qss',
        'phi': gas.equivalence_ratio(),
        'times': np.array(times),
        'temperatures': np.array(temperatures),
        'fuel_mass_fractions': np.array(fuel_mass_fractions),
        'species_profiles': species_profiles,
        'cpu_times': np.array(cpu_times),
        'total_cpu_time': np.sum(cpu_times),
        'total_wall_time': total_wall_time,
        'steps': len(times) - 1,
        'success': len(times) > 1,
        'timed_out': total_wall_time > time_limit,
        'qss_stats': {
            'ode_evals': integrator.gcount,
            'failed_steps': integrator.rcount,
            'final_time': integrator.tn
        }
    }


class IntegratorBenchmark:
    """Comprehensive benchmarking suite for different integrators."""
    
    def __init__(self, mechanism_file: str, temp: float = 900, pressure: float = 6*ct.one_atm,
                 fuel: str = 'nc12h26', phi: float = 1.0):
        """Initialize benchmark with combustion conditions."""
        
        self.mechanism_file = mechanism_file
        self.temp = temp
        self.pressure = pressure
        self.fuel = fuel
        self.phi = phi
        
        # Setup gas
        self.gas = ct.Solution(mechanism_file)
        fuel_spec = f'{fuel}:1.0'
        oxidizer = 'n2:3.76, o2:1.0'
        self.gas.set_equivalence_ratio(phi, fuel_spec, oxidizer)
        self.gas.TP = temp, pressure
        
        # Species to track
        self.major_species = ['o', 'h', 'oh', 'h2o', 'o2', 'h2', 'co', 'co2']
        
        # Initial state
        self.y0 = np.hstack([temp, self.gas.Y])
        
        print(f"Initialized benchmark:")
        print(f"  Mechanism: {mechanism_file}")
        print(f"  Species count: {self.gas.n_species}")
        print(f"  Temperature: {temp} K")
        print(f"  Pressure: {pressure/ct.one_atm:.1f} atm")
        print(f"  Equivalence ratio: {phi}")
        print(f"  Fuel: {fuel}")
    
    def get_sundials_methods(self):
        """Get list of SUNDIALS methods to test."""
        return ['cvode_bdf', 'cvode_adams', 'arkode_erk']
    
    def get_cpp_methods(self):
        """Get list of C++ methods to test."""
        return ['cpp_rk23']
    
    def get_scipy_methods(self):
        """Get list of SciPy methods to test."""
        return ['scipy_dopri_dopri5', 'scipy_dop853_dop853']
    
    def run_single_benchmark(self, method: str, t_final: float = 0.02, 
                           timestep: float = 1e-6, rtol: float = 1e-6, 
                           atol: float = 1e-10, time_limit: float = 300.0) -> Dict[str, Any]:
        """Run benchmark for a single method."""
        
        print(f"\nRunning {method} benchmark...")
        
        if method == 'qss':
            # Use dt_output = timestep for QSS
            return run_qss_integration(
                self.gas, self.temp, self.pressure, t_final, timestep,
                self.fuel, self.major_species, time_limit
            )
        else:
            # Use existing integration function (you'll need to import this)
            # Assuming you have the run_integration_experiment function available
            return run_integration_experiment(
                method, self.gas, self.y0, 0.0, t_final, timestep,
                rtol, atol, self.major_species, self.fuel, self.pressure, time_limit
            )
    
    def run_comprehensive_benchmark(self, methods: List[str], 
                                  tolerances: List[Tuple[float, float]] = [(1e-6, 1e-10)],
                                  t_final: float = 0.02, timestep: float = 1e-6,
                                  time_limit: float = 300.0) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple methods and tolerances."""
        
        results = []
        
        for method in methods:
            if method == 'qss':
                # QSS doesn't use rtol/atol in the same way
                print(f"Running QSS benchmark with method {method}")
                result = self.run_single_benchmark(
                    method, t_final, timestep, time_limit=time_limit
                )
                result['rtol'] = None
                result['atol'] = None
                results.append(result)
            else:
                for rtol, atol in tolerances:
                    print(f"Running {method} benchmark with rtol {rtol} and atol {atol}")
                    result = self.run_single_benchmark(
                        method, t_final, timestep, rtol, atol, time_limit
                    )
                    results.append(result)
        
        return self._process_results(results)
    
    def _process_results(self, results: List[Dict]) -> pd.DataFrame:
        """Process results into a pandas DataFrame for analysis."""
        
        processed = []
        
        for result in results:
            if not result['success']:
                continue
                
            row = {
                'method': result['method'],
                'rtol': result.get('rtol', None),
                'atol': result.get('atol', None),
                'final_temp': result['temperatures'][-1],
                'total_cpu_time': result['total_cpu_time'],
                'total_wall_time': result['total_wall_time'],
                'steps': result['steps'],
                'success': result['success'],
                'timed_out': result['timed_out'],
                'final_fuel_fraction': result['fuel_mass_fractions'][-1],
                'avg_step_time': result['total_cpu_time'] / result['steps'] if result['steps'] > 0 else 0
            }
            
            # Add QSS-specific stats if available
            if 'qss_stats' in result:
                row.update({
                    'ode_evals': result['qss_stats']['ode_evals'],
                    'failed_steps': result['qss_stats']['failed_steps']
                })
            
            processed.append(row)
        
        return pd.DataFrame(processed)
    
    def plot_benchmark_results(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive plots of benchmark results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. CPU Time Comparison
        ax = axes[0, 0]
        methods = df['method'].unique()
        cpu_times = [df[df['method'] == method]['total_cpu_time'].iloc[0] 
                    if len(df[df['method'] == method]) > 0 else 0 
                    for method in methods]
        
        bars = ax.bar(methods, cpu_times)
        ax.set_ylabel('Total CPU Time (s)')
        ax.set_title('CPU Time Comparison')
        ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars, cpu_times):
            if time > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{time:.2e}', ha='center', va='bottom', fontsize=8)
        
        # 2. Wall Time vs CPU Time
        ax = axes[0, 1]
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                ax.scatter(method_data['total_cpu_time'], method_data['total_wall_time'], 
                          label=method, s=60, alpha=0.7)
        
        ax.set_xlabel('CPU Time (s)')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title('CPU vs Wall Time')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], 
               [ax.get_xlim()[0], ax.get_xlim()[1]], 'k--', alpha=0.5)
        
        # 3. Steps vs Performance
        ax = axes[0, 2]
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                ax.scatter(method_data['steps'], method_data['total_cpu_time'], 
                          label=method, s=60, alpha=0.7)
        
        ax.set_xlabel('Number of Steps')
        ax.set_ylabel('Total CPU Time (s)')
        ax.set_title('Steps vs CPU Time')
        ax.set_yscale('log')
        ax.legend()
        
        # 4. Final Temperature
        ax = axes[1, 0]
        final_temps = [df[df['method'] == method]['final_temp'].iloc[0] 
                      if len(df[df['method'] == method]) > 0 else 0 
                      for method in methods]
        
        bars = ax.bar(methods, final_temps)
        ax.set_ylabel('Final Temperature (K)')
        ax.set_title('Final Temperature Comparison')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, temp in zip(bars, final_temps):
            if temp > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{temp:.0f}K', ha='center', va='bottom', fontsize=8)
        
        # 5. Average Step Time
        ax = axes[1, 1]
        avg_step_times = [df[df['method'] == method]['avg_step_time'].iloc[0] 
                         if len(df[df['method'] == method]) > 0 else 0 
                         for method in methods]
        
        bars = ax.bar(methods, avg_step_times)
        ax.set_ylabel('Average Step Time (s)')
        ax.set_title('Average Step Time Comparison')
        ax.set_yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 6. Success/Failure Summary
        ax = axes[1, 2]
        success_counts = []
        failed_counts = []
        
        for method in methods:
            method_data = df[df['method'] == method]
            success_counts.append(len(method_data[method_data['success'] == True]))
            failed_counts.append(len(method_data[method_data['success'] == False]))
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, success_counts, width, label='Success', color='green', alpha=0.7)
        ax.bar(x + width/2, failed_counts, width, label='Failed', color='red', alpha=0.7)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Count')
        ax.set_title('Success/Failure Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_full_benchmark_suite(self):
        """Run the complete benchmark suite."""
        
        print("="*60)
        print("COMPREHENSIVE INTEGRATOR BENCHMARK SUITE")
        print("="*60)
        
        # Define methods to test
        methods = ['cvode_bdf']  # Start with QSS
        
        # Add other methods if available
        try:
            methods.extend(self.get_sundials_methods())
            print("Added SUNDIALS methods")
        except:
            print("SUNDIALS methods not available")
        
        try:
            methods.extend(self.get_cpp_methods())
            print("Added C++ methods")
        except:
            print("C++ methods not available")
        
        # try:
        #     methods.extend(self.get_scipy_methods())
        #     print("Added SciPy methods")
        # except:
        #     print("SciPy methods not available")
        
        print(f"Testing methods: {methods}")
        
        # Define tolerances to test (for non-QSS methods)
        tolerances = [
            (1e-4, 1e-8),   # Loose
            (1e-6, 1e-10),  # Medium
            (1e-8, 1e-12),  # Tight
        ]
        
        # Run benchmark
        df = self.run_comprehensive_benchmark(
            methods=methods,
            tolerances=tolerances,
            t_final=0.02,
            timestep=1e-6,
            time_limit=300.0
        )
        
        # Display results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        
        # Create plots
        self.plot_benchmark_results(df, 'integrator_benchmark.png')
        
        # Save detailed results
        df.to_csv('benchmark_results.csv', index=False)
        print(f"\nDetailed results saved to 'benchmark_results.csv'")
        
        return df


# Example usage
if __name__ == "__main__":
    # Setup benchmark
    mechanism_file = "large_mechanism/n-dodecane.yaml"  # Update this path
    
    benchmark = IntegratorBenchmark(
        mechanism_file=mechanism_file,
        temp=900,
        pressure=6*ct.one_atm,
        fuel='nc12h26',
        phi=1.0
    )
    
    # Run full benchmark suite
    results_df = benchmark.run_full_benchmark_suite()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if len(results_df) > 0:
        fastest_method = results_df.loc[results_df['total_cpu_time'].idxmin()]
        print(f"Fastest method: {fastest_method['method']} ({fastest_method['total_cpu_time']:.2e}s)")
        
        most_efficient = results_df.loc[results_df['avg_step_time'].idxmin()]
        print(f"Most efficient per step: {most_efficient['method']} ({most_efficient['avg_step_time']:.2e}s/step)")
        
        if 'qss' in results_df['method'].values:
            qss_result = results_df[results_df['method'] == 'qss'].iloc[0]
            print(f"\nQSS Performance:")
            print(f"  CPU Time: {qss_result['total_cpu_time']:.2e}s")
            print(f"  Steps: {qss_result['steps']}")
            print(f"  Final Temperature: {qss_result['final_temp']:.1f}K")
            if 'ode_evals' in qss_result:
                print(f"  ODE Evaluations: {qss_result['ode_evals']}")
                print(f"  Failed Steps: {qss_result['failed_steps']}")
    
    print("\nBenchmark complete!")