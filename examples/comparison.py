import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import time
import os
from combustion import SundialsChemicalIntegrator, SundialsIntegratorConfig
import SundialsPy


def compare_sundials_solvers():
    """
    Compare CVODE and ARKODE solvers on 0D combustion problems.
    """
    # Define test cases
    test_cases = [
        {
            'name': 'methane',
            'mechanism': 'gri30.yaml',
            'fuel': 'CH4',
            'temperature': 1200.0,
            'pressure': 101325.0 * 1,  # 5 atm
            'phi': 1
        }
    ]
    
    # Define integrator methods
    methods = [
        ('cvode_bdf', 'CVODE BDF'),
        # ('arkode_erk', 'ARKODE ERK'),
        ('cpp_rk23', 'C++ RK23')
    ]

    available_tables = {
            'HEUN_EULER_2_1_2': SundialsPy.arkode.ButcherTable.HEUN_EULER_2_1_2,
            'BOGACKI_SHAMPINE_4_2_3': SundialsPy.arkode.ButcherTable.BOGACKI_SHAMPINE_4_2_3,
            'ARK548L2SA_ERK_8_4_5': SundialsPy.arkode.ButcherTable.ARK548L2SA_ERK_8_4_5,
        }
    
    # Define tolerances
    tolerances = [
        (1e-6, 1e-8, 'Loose'),
        (1e-10, 1e-12, 'Tight')
    ]
    
    # Create results directory
    results_dir = "combustion_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Iterate over all combinations
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"CASE: {case['name']}")
        print(f"{'='*50}")
        
        # # Make sure mechanism file exists
        # if not os.path.exists(case['mechanism']):
        #     print(f"Skipping case {case['name']} - mechanism file {case['mechanism']} not found")
        #     continue
        
        case_results = {}
        
        for method_key, method_name in methods:
            for rtol, atol, tol_name in tolerances:
                print(f"\nRunning {method_name} with {tol_name} tolerances")
                
                # Configure integrator
                config = SundialsIntegratorConfig(
                    integrator_list=[method_key],
                    tolerance_list=[(rtol, atol)],
                    butcher_tables={
                        'arkode_erk': available_tables
                    }
                )
                
                # Create integrator
                integrator = SundialsChemicalIntegrator(
                    mechanism_file=case['mechanism'],
                    temperature=case['temperature'],
                    pressure=case['pressure'],
                    fuel=case['fuel'],
                    phi=case['phi'],
                    timestep=1e-5,  # Start with a small timestep
                    config=config
                )
                
                # Solve
                test_key = f"{method_key}_{tol_name}"
                try:
                    start_time = time.time()
                    
                    # # For hydrogen, use fixed timesteps
                    # if case['fuel'] == 'H2':
                    #     end_time = 1e-3  # 1 ms for hydrogen
                    #     n_points = 200
                    #     integrator.solve(end_time=end_time, action_idx=0, n_points=n_points)
                    # else:
                    #     # For other fuels, use adaptive stepping
                    end_time = 0.05  # 5 ms for methane (may need adjustment)
                    integrator.solve(end_time=end_time, action_idx=0)
                    
                    wall_time = time.time() - start_time
                    
                    # Calculate ignition delay
                    t_ign = integrator.get_ignition_delay()
                    
                    # Get statistics
                    stats = integrator.get_statistics()
                    stats['wall_time'] = wall_time
                    
                    # Save results
                    case_results[test_key] = {
                        'times': np.array(integrator.history['times']),
                        'temperatures': np.array(integrator.history['temperatures']),
                        'pressures': np.array(integrator.history['pressures']),
                        'species': {k: np.array(v) for k, v in integrator.history['species_profiles'].items()},
                        'cpu_times': np.array(integrator.history['cpu_times']),
                        'ignition_delay': t_ign,
                        'stats': stats
                    }
                    
                    # Plot individual result
                    plot_path = os.path.join(results_dir, f"{case['name']}_{test_key}.png")
                    integrator.plot_results(save_path=plot_path, show_plot=False)
                    
                    print(f"  Completed in {wall_time:.4f} seconds")
                    print(f"  Ignition delay: {t_ign:.2e} seconds")
                    print(f"  Total CPU time: {stats['total_cpu_time']:.4f} seconds")
                    print(f"  Number of steps: {stats['num_steps']}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Create comparison plots for this case
        if case_results:
            # Temperature comparison
            plt.figure(figsize=(10, 6))
            
            for test_key, results in case_results.items():
                plt.plot(results['times'], results['temperatures'], label=test_key)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (K)')
            plt.title(f'{case["name"]} - Temperature Evolution')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"{case['name']}_temperature_comparison.png"))
            plt.close()
            
            # Ignition delay comparison
            plt.figure(figsize=(8, 5))
            
            test_keys = list(case_results.keys())
            ignition_delays = [case_results[key]['ignition_delay'] for key in test_keys]
            cpu_times = [case_results[key]['stats']['total_cpu_time'] for key in test_keys]
            
            # Create bar chart
            x = np.arange(len(test_keys))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Ignition delay on left y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Solver Configuration')
            ax1.set_ylabel('Ignition Delay (s)', color=color)
            ax1.bar(x - width/2, ignition_delays, width, label='Ignition Delay', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # CPU time on right y-axis
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('CPU Time (s)', color=color)
            ax2.bar(x + width/2, cpu_times, width, label='CPU Time', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add labels
            ax1.set_title(f'{case["name"]} - Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(test_keys, rotation=45, ha='right')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{case['name']}_performance_comparison.png"))
            plt.close()
            
            # Create a summary table
            print("\nSummary for case: " + case['name'])
            print("-" * 80)
            print(f"{'Solver':<20} {'Ignition Delay (s)':<20} {'CPU Time (s)':<15} {'Steps':<10} {'Efficiency':<15}")
            print("-" * 80)
            for test_key in test_keys:
                ign_delay = case_results[test_key]['ignition_delay']
                cpu_time = case_results[test_key]['stats']['total_cpu_time']
                steps = case_results[test_key]['stats']['num_steps']
                efficiency = ign_delay / cpu_time  # seconds of simulation per second of CPU time
                
                print(f"{test_key:<20} {ign_delay:<20.6e} {cpu_time:<15.6f} {steps:<10} {efficiency:<15.6f}")
            print("-" * 80)
    
    print("\nAll simulations complete. Results saved to " + results_dir)

if __name__ == "__main__":
    compare_sundials_solvers()