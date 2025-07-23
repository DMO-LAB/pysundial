#!/usr/bin/env python3
"""
Fixed test script for implicit ARKODE methods (DIRK).

This script tests various DIRK methods with proper error handling and debugging.
"""

import numpy as np
import time
import SundialsPy
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
    mu = 100.0  # Reduced stiffness for testing
    
    def rhs_explicit(t, y):
        """Explicit (non-stiff) part of the ODE."""
        # Ensure y is properly shaped and return same shape
        y = np.asarray(y).ravel()
        result = np.zeros_like(y)
        print(f"[DEBUG] explicit: t={t:.6f}, y.shape={y.shape}, result.shape={result.shape}")
        return result
    
    def rhs_implicit(t, y):
        """Implicit (stiff) part of the ODE."""
        # Ensure y is properly shaped
        y = np.asarray(y).ravel()
        if len(y) != 2:
            raise ValueError(f"Expected y to have length 2, got {len(y)}")
        
        result = np.zeros(2, dtype=np.float64)
        result[0] = y[1]
        result[1] = mu * (1.0 - y[0]**2) * y[1] - y[0]
        
        print(f"[DEBUG] implicit: t={t:.6f}, y={y}, result={result}")
        return result
    
    def jacobian(t, y):
        """Jacobian matrix for the implicit part."""
        print("CALLED JACOBIAN")
        y = np.asarray(y).ravel()
        print(f"[DEBUG] jacobian: t={t:.6f}, y={y}")
        if len(y) != 2:
            raise ValueError(f"Expected y to have length 2, got {len(y)}")
        # Create Jacobian matrix
        J = np.zeros((2, 2), dtype=np.float64)
        J[0, 0] = 0.0
        J[0, 1] = 1.0
        J[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
        J[1, 1] = mu * (1.0 - y[0]**2)
        # Ensure C-contiguous 2D array
        J = np.ascontiguousarray(J)
        print(f"[python] Jacobian shape: {J.shape}, ndim: {J.ndim}, dtype: {J.dtype}, flags: {J.flags}")
        print(f"[python] Jacobian matrix:\n{J}")
        return J

    # Simulation parameters
    t0 = 0.0
    t_end = 0.01  # Very short simulation for testing
    y0 = np.array([2.0, 0.0], dtype=np.float64)
    
    # Define DIRK methods to test
    dirk_methods = [
        ("SDIRK_2_1_2", SundialsPy.arkode.ButcherTable.SDIRK_2_1_2),
        ("BILLINGTON_3_3_2", SundialsPy.arkode.ButcherTable.BILLINGTON_3_3_2),
        ("TRBDF2_3_3_2", SundialsPy.arkode.ButcherTable.TRBDF2_3_3_2)
    ]
    
    # Test cases to run
    test_cases = [
        TestCase.WITHOUT_JACOBIAN,
        TestCase.WITH_JACOBIAN
    ]
    
    # Tolerance settings
    rtol = 1e-6
    atol = np.ones_like(y0) * 1e-8
    
    # Store results for each test
    results = {}
    
    for test_case in test_cases:
        print(f"\n===== Testing {test_case.name} =====")
        results[test_case.name] = {}
        
        for method_name, butcher_table in dirk_methods:
            print(f"\nTesting {method_name}...")
            
            try:
                # Create ARKode solver with implicit function
                solver = SundialsPy.arkode.ARKodeSolver(
                    system_size=len(y0),
                    explicit_fn=rhs_explicit,
                    implicit_fn=rhs_implicit,
                    butcher_table=butcher_table,
                    linsol_type=SundialsPy.cvode.LinearSolverType.DENSE
                )
                
                # Keep references to prevent garbage collection
                solver._py_explicit = rhs_explicit
                solver._py_implicit = rhs_implicit
                
                # Set Jacobian if this test case requires it
                if test_case == TestCase.WITH_JACOBIAN:
                    print("Setting Jacobian function...")
                    solver._py_jacobian = jacobian  # Keep reference
                    solver.set_jacobian(jacobian)
                
                # Initialize solver
                print("Initializing solver...")
                solver.initialize(y0, t0, rtol, atol)
                
                # Time the solution
                start_time = time.time()
                
                # Solve the ODE
                print("Starting solve...")
                solution = solver.solve_to(t_end)
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
                print(f"  Final solution: {solution}")
                
                print("_________________________************************________________________")
                
            except Exception as e:
                import traceback
                print(f"  Error: {str(e)}")
                print("  Full traceback:")
                traceback.print_exc()
                results[test_case.name][method_name] = {
                    'solution': None,
                    'time': float('inf'),
                    'stats': {},
                    'success': False,
                    'error': str(e)
                }
                print("_________________________************************________________________")

    # Print performance comparison
    print_comparison(results)

def print_comparison(results):
    """Print a comparative summary of all methods tested."""
    print("\n===== Performance Summary =====")
    
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

if __name__ == "__main__":
    run_test()