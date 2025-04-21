#!/usr/bin/env python3
"""
Test script for implicit ARKODE methods (DIRK).

This script tests various DIRK (Diagonally Implicit Runge-Kutta) methods from SUNDIALS
on a stiff ODE problem. It also explores what happens when a Jacobian is not provided.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sundials_py
from enum import Enum

class TestCase(Enum):
    WITH_JACOBIAN = 1
    WITHOUT_JACOBIAN = 2

def run_test():
    """Run tests on implicit ARKODE methods."""
    print("Testing implicit ARKODE methods (DIRK)...")
    
    # Define a stiff ODE problem: Van der Pol oscillator
    # y'₁ = y₂
    # y'₂ = μ(1 - y₁²)y₂ - y₁
    # where μ controls stiffness (larger μ = more stiff)
    mu = 1000.0  # Use a large value to ensure the problem is stiff
    system_size = 2  # The Van der Pol system has 2 equations
    
    def rhs_explicit(t, y):
        """Explicit (non-stiff) part of the ODE."""
        # For this test, we'll put everything in the implicit part
        return np.zeros_like(y)
    
    def rhs_implicit(t, y):
        """Implicit (stiff) part of the ODE."""
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = mu * (1.0 - y[0]**2) * y[1] - y[0]
        return dy
    
    def jacobian(t, y):
        """Analytical Jacobian of the system."""
        # The Jacobian must be a 2D array with shape (system_size, system_size)
        J = np.zeros((system_size, system_size), dtype=float)
        J[0, 1] = 1.0
        J[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
        J[1, 1] = mu * (1.0 - y[0]**2)
        return J
    
    # Simulation parameters
    t0 = 0.0
    t_end = 0.1  # Short simulation for testing
    y0 = np.array([2.0, 0.0])  # Initial condition
    
    # Define DIRK methods to test
    dirk_methods = [
        # Method name, Butcher table enum
        ("SDIRK_2_1_2", sundials_py.arkode.ButcherTable.SDIRK_2_1_2),
        ("BILLINGTON_3_3_2", sundials_py.arkode.ButcherTable.BILLINGTON_3_3_2),
        ("TRBDF2_3_3_2", sundials_py.arkode.ButcherTable.TRBDF2_3_3_2),
        ("KVAERNO_4_2_3", sundials_py.arkode.ButcherTable.KVAERNO_4_2_3),
        ("ARK324L2SA_DIRK_4_2_3", sundials_py.arkode.ButcherTable.ARK324L2SA_DIRK_4_2_3),
        ("CASH_5_2_4", sundials_py.arkode.ButcherTable.CASH_5_2_4),
        ("CASH_5_3_4", sundials_py.arkode.ButcherTable.CASH_5_3_4),
        ("SDIRK_5_3_4", sundials_py.arkode.ButcherTable.SDIRK_5_3_4),
        ("ARK436L2SA_DIRK_6_3_4", sundials_py.arkode.ButcherTable.ARK436L2SA_DIRK_6_3_4),
        ("ARK437L2SA_DIRK_7_3_4", sundials_py.arkode.ButcherTable.ARK437L2SA_DIRK_7_3_4),
        ("KVAERNO_7_4_5", sundials_py.arkode.ButcherTable.KVAERNO_7_4_5),
        ("ARK548L2SA_DIRK_8_4_5", sundials_py.arkode.ButcherTable.ARK548L2SA_DIRK_8_4_5)
    ]
    
    # Test cases to run
    test_cases = [

        TestCase.WITHOUT_JACOBIAN
    ]
    
    # Tolerance settings
    rtol = 1e-6
    atol = 1e-8 * np.ones_like(y0)
    
    # Output time points
    num_points = 100
    t_points = np.linspace(t0, t_end, num_points)
    t_points = t_points[1:]
    # Store results for each test
    results = {}
    
    for test_case in test_cases:
        print(f"\n===== Testing {test_case.name} =====")
        results[test_case.name] = {}
        
        for method_name, butcher_table in dirk_methods:
            print(f"\nTesting {method_name}...")
            
            try:
                # Create ARKode solver with implicit function
                solver = sundials_py.arkode.ARKodeSolver(
                    system_size=system_size,  # Explicitly provide system size
                    explicit_fn=rhs_explicit,
                    implicit_fn=rhs_implicit,  # This is where we provide the implicit part
                    butcher_table=butcher_table,
                    linsol_type=sundials_py.cvode.LinearSolverType.DENSE
                )
                
                # Initialize solver
                solver.initialize(y0, t0, rtol, atol)
                
                # Only attach the linear solver for ARKStep 
                # (This is handled internally in the C++ code)
                
                # Set Jacobian if this test case requires it
                if test_case == TestCase.WITH_JACOBIAN:
                    solver.set_jacobian(jacobian)
                
                # Time the solution
                start_time = time.time()
                
                # Solve the ODE
                solution = solver.solve(t_points)
                
                end_time = time.time()
                
                # Get solver statistics
                stats = solver.get_stats()
                
                # Store results
                results[test_case.name][method_name] = {
                    'solution': solution,
                    'time': end_time - start_time,
                    'stats': stats,
                    'success': True
                }
                
                # Print some information
                print(f"  Completed in {end_time - start_time:.4f} seconds")
                print(f"  Steps taken: {stats.get('num_steps', 'N/A')}")
                print(f"  RHS evaluations: Explicit={stats.get('num_rhs_evals_explicit', 'N/A')}, " + 
                      f"Implicit={stats.get('num_rhs_evals_implicit', 'N/A')}")
                print(f"  Linear solver setups: {stats.get('num_lin_setups', 'N/A')}")
                print(f"  Error test failures: {stats.get('num_error_test_fails', 'N/A')}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  Error: {str(e)}")
                results[test_case.name][method_name] = {
                    'solution': None,
                    'time': float('inf'),
                    'stats': {},
                    'success': False,
                    'error': str(e)
                }
    
    # Plot results
    plot_results(results, t_points)
    
    # Print performance comparison
    print_comparison(results)

def plot_results(results, t_points):
    """Plot the results of the tests."""
    # Create figure for solutions
    plt.figure(figsize=(12, 10))
    
    legend_entries = []
    
    # Plot WITH_JACOBIAN results
    if 'WITH_JACOBIAN' in results:
        for method_name, result in results['WITH_JACOBIAN'].items():
            if result['success']:
                plt.plot(t_points, result['solution'][:, 0], linestyle='-')
                legend_entries.append(f"{method_name} (with Jacobian)")
    
    # Plot WITHOUT_JACOBIAN results
    if 'WITHOUT_JACOBIAN' in results:
        for method_name, result in results['WITHOUT_JACOBIAN'].items():
            if result['success']:
                plt.plot(t_points, result['solution'][:, 0], linestyle='--')
                legend_entries.append(f"{method_name} (without Jacobian)")
    
    plt.xlabel('Time')
    plt.ylabel('y₁')
    plt.title('Van der Pol Oscillator - Comparison of DIRK Methods')
    plt.legend(legend_entries)
    plt.grid(True)
    plt.savefig('dirk_methods_comparison.png')
    plt.close()
    
    # Create performance bar charts
    plt.figure(figsize=(14, 8))
    
    # Organize data for bar chart
    methods = []
    times_with_jac = []
    times_without_jac = []
    
    for method_name in results.get('WITH_JACOBIAN', {}):
        methods.append(method_name)
        
        # Time with Jacobian
        with_jac_result = results.get('WITH_JACOBIAN', {}).get(method_name, {})
        times_with_jac.append(with_jac_result.get('time', float('nan')) if with_jac_result.get('success', False) else float('nan'))
        
        # Time without Jacobian
        without_jac_result = results.get('WITHOUT_JACOBIAN', {}).get(method_name, {})
        times_without_jac.append(without_jac_result.get('time', float('nan')) if without_jac_result.get('success', False) else float('nan'))
    
    # Set up bar positions
    x = np.arange(len(methods))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, times_with_jac, width, label='With Jacobian')
    plt.bar(x + width/2, times_without_jac, width, label='Without Jacobian')
    
    # Add labels and title
    plt.xlabel('DIRK Method')
    plt.ylabel('Solution Time (s)')
    plt.title('Performance Comparison of DIRK Methods')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dirk_methods_performance.png')
    plt.close()

def print_comparison(results):
    """Print a comparative summary of all methods tested."""
    print("\n===== Performance Summary =====")
    
    # Create a table for the data
    print("\nWith Jacobian:")
    print("-" * 100)
    print(f"{'Method':<20} | {'Time (s)':<10} | {'Steps':<10} | {'RHS Evals':<20} | {'Lin Setups':<10} | {'Success':<10}")
    print("-" * 100)
    
    for method_name, result in results.get('WITH_JACOBIAN', {}).items():
        if result['success']:
            stats = result['stats']
            rhs_evals = f"{stats.get('num_rhs_evals_implicit', 'N/A')}"
            print(f"{method_name:<20} | {result['time']:<10.4f} | {stats.get('num_steps', 'N/A'):<10} | {rhs_evals:<20} | {stats.get('num_lin_setups', 'N/A'):<10} | {'✓':<10}")
        else:
            print(f"{method_name:<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<20} | {'N/A':<10} | {'✗':<10}")
    
    print("\nWithout Jacobian:")
    print("-" * 100)
    print(f"{'Method':<20} | {'Time (s)':<10} | {'Steps':<10} | {'RHS Evals':<20} | {'Lin Setups':<10} | {'Success':<10}")
    print("-" * 100)
    
    for method_name, result in results.get('WITHOUT_JACOBIAN', {}).items():
        if result['success']:
            stats = result['stats']
            rhs_evals = f"{stats.get('num_rhs_evals_implicit', 'N/A')}"
            print(f"{method_name:<20} | {result['time']:<10.4f} | {stats.get('num_steps', 'N/A'):<10} | {rhs_evals:<20} | {stats.get('num_lin_setups', 'N/A'):<10} | {'✓':<10}")
        else:
            print(f"{method_name:<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<20} | {'N/A':<10} | {'✗':<10}")
    
    # Print recommendations
    print("\n===== Recommendations =====")
    print("Based on the test results, here are some observations:")
    
    # Compare success rates
    success_with_jac = sum(1 for result in results.get('WITH_JACOBIAN', {}).values() if result['success'])
    success_without_jac = sum(1 for result in results.get('WITHOUT_JACOBIAN', {}).values() if result['success'])
    
    if success_with_jac > success_without_jac:
        print("- Providing a Jacobian generally improves the success rate of implicit methods")
    elif success_with_jac < success_without_jac:
        print("- Surprisingly, methods without Jacobian were more successful in this test")
    else:
        print("- Providing a Jacobian didn't affect the overall success rate in this test")
    
    # Find best methods in terms of speed
    best_with_jac = None
    best_with_jac_time = float('inf')
    for method_name, result in results.get('WITH_JACOBIAN', {}).items():
        if result['success'] and result['time'] < best_with_jac_time:
            best_with_jac = method_name
            best_with_jac_time = result['time']
    
    best_without_jac = None
    best_without_jac_time = float('inf')
    for method_name, result in results.get('WITHOUT_JACOBIAN', {}).items():
        if result['success'] and result['time'] < best_without_jac_time:
            best_without_jac = method_name
            best_without_jac_time = result['time']
    
    if best_with_jac:
        print(f"- Best method with Jacobian: {best_with_jac} ({best_with_jac_time:.4f}s)")
    if best_without_jac:
        print(f"- Best method without Jacobian: {best_without_jac} ({best_without_jac_time:.4f}s)")
    
    print("\nTo integrate these methods into your IntegratorFactory, you should:")
    print("1. Add a mapping from method name to Butcher table for all DIRK methods")
    print("2. Determine whether to automatically provide a Jacobian or allow the user to choose")
    print("3. For the combustion problems, decide whether a pure implicit method or an ImEx approach is better")

if __name__ == "__main__":
    run_test()