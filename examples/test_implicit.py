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
    
    def rhs_explicit(t, y):
        """Explicit (non-stiff) part of the ODE."""
        # For this test, we'll put everything in the implicit part
        return np.zeros_like(y)
    
    def rhs_implicit(t, y):
        """Implicit (stiff) part of the ODE."""
        # Make sure y is 1D array
        y = np.ravel(y)
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = mu * (1.0 - y[0]**2) * y[1] - y[0]
        return dy
    
    def jacobian(t, y):
        y = np.ravel(y)
        J = np.array([
            [0.0, 1.0],
            [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0]**2)]
        ], dtype=np.float64)
        
        # Ensure it's a contiguous 2D array with the right shape
        J = np.ascontiguousarray(J)
        
        # Debug print to verify shape and type
        print(f"[python] Jacobian shape: {J.shape}, ndim: {J.ndim}, dtype: {J.dtype}")
        return J



    
    # Simulation parameters
    t0 = 0.0
    t_end = 0.1  # Short simulation for testing
    y0 = np.array([2.0, 0.0])  # Initial condition
    
    # Define DIRK methods to test
    # We'll only test a few methods to keep the test short
    dirk_methods = [
        # Method name, Butcher table enum
        ("SDIRK_2_1_2", sundials_py.arkode.ButcherTable.SDIRK_2_1_2),
        ("BILLINGTON_3_3_2", sundials_py.arkode.ButcherTable.BILLINGTON_3_3_2),
        ("TRBDF2_3_3_2", sundials_py.arkode.ButcherTable.TRBDF2_3_3_2)
    ]
    
    # Test cases to run
    test_cases = [
        TestCase.WITHOUT_JACOBIAN,
        TestCase.WITH_JACOBIAN
        
    ]
    
    # Tolerance settings
    rtol = 1e-6
    atol = np.ones_like(y0) * 1e-8
    
    # Output time points (just a few points for testing)
    num_points = 10
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
                    system_size=len(y0),
                    explicit_fn=rhs_explicit,
                    implicit_fn=rhs_implicit,  # This is where we provide the implicit part
                    butcher_table=butcher_table,
                    linsol_type=sundials_py.cvode.LinearSolverType.DENSE
                )
                
                # Initialize solver
                solver.initialize(y0, t0, rtol, atol)
                
                # Set Jacobian if this test case requires it
                if test_case == TestCase.WITH_JACOBIAN:
                    print("Setting Jacobian function...")
                    solver.set_jacobian(jacobian)
                
                # Time the solution
                start_time = time.time()
                
                # Solve the ODE
                print("Starting solve...")
                solution = solver.solve(t_points)
                print("Solve completed!")
                
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
    
    # Print performance comparison
    print_comparison(results)

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
    
if __name__ == "__main__":
    run_test()