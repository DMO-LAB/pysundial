#!/usr/bin/env python3
"""
Test script to demonstrate clean failure handling of CVodeSolver.
"""

import numpy as np
import sys
import os

# Add the SundialsPy module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SundialsPy'))

try:
    import SundialsPy
except ImportError as e:
    print(f"Error importing SundialsPy: {e}")
    print("Make sure the module is compiled and available.")
    sys.exit(1)

def simple_rhs(t, y, ydot):
    """Simple ODE: dy/dt = -y"""
    ydot[0] = -y[0]
    return 0

def stiff_rhs(t, y, ydot):
    """Stiff ODE that might cause solver to fail: dy/dt = -1000*y"""
    ydot[0] = -1000.0 * y[0]
    return 0

def test_clean_failure_handling():
    """Test clean failure handling without noisy error messages"""
    
    print("=" * 60)
    print("Testing Clean Failure Handling")
    print("=" * 60)
    
    # Test parameters
    y0 = np.array([1.0])
    t0 = 0.0
    t_final = 1.0
    
    # Test 1: Normal case with verbose=-1 (no error messages)
    print("\n1. Testing normal case with verbose=-1:")
    print("-" * 40)
    solver1 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=simple_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=1000,
        verbose=-1  # No output at all
    )
    solver1.initialize(y0, t0)
    
    # Use new solve_to method that returns (result, success, error_code)
    result, success, error_code = solver1.solve_to(t_final)
    
    if success:
        print(f"✓ Success: Result = {result[0]:.6f}")
    else:
        print(f"✗ Failed: Error code = {error_code}")
    
    # Test 2: Stiff problem that might fail with low mxsteps
    print("\n2. Testing stiff problem with low mxsteps (might fail):")
    print("-" * 40)
    solver2 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=stiff_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=10,  # Very low - likely to fail
        verbose=-1   # No error messages
    )
    solver2.initialize(y0, t0)
    
    result, success, error_code = solver2.solve_to(t_final)
    
    if success:
        print(f"✓ Success: Result = {result[0]:.6f}")
    else:
        print(f"✗ Failed: Error code = {error_code}")
        print(f"  Partial result: {result[0]:.6f}")
    
    # Test 3: Training loop simulation
    print("\n3. Simulating training loop with clean failure handling:")
    print("-" * 40)
    
    failure_count = 0
    success_count = 0
    
    for i in range(10):
        # Create solver with random parameters that might cause failure
        mxsteps = np.random.randint(5, 50)  # Random low mxsteps
        
        solver = SundialsPy.cvode.CVodeSolver(
            system_size=1,
            rhs_fn=stiff_rhs,
            iter_type=SundialsPy.cvode.IterationType.NEWTON,
            linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
            use_bdf=True,
            mxsteps=mxsteps,
            verbose=-1  # No noisy output
        )
        solver.initialize(y0, t0)
        
        result, success, error_code = solver.solve_to(t_final)
        
        if success:
            success_count += 1
            # print(f"Step {i+1}: ✓ Success")
        else:
            failure_count += 1
            # print(f"Step {i+1}: ✗ Failed (error {error_code})")
    
    print(f"Training simulation results:")
    print(f"  Successes: {success_count}/10")
    print(f"  Failures: {failure_count}/10")
    print(f"  Success rate: {success_count/10*100:.1f}%")
    
    # Test 4: Using legacy method for backward compatibility
    print("\n4. Testing legacy method (backward compatibility):")
    print("-" * 40)
    solver4 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=simple_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=1000,
        verbose=-1
    )
    solver4.initialize(y0, t0)
    
    # Legacy method returns only the result
    result_legacy = solver4.solve_to_legacy(t_final)
    print(f"Legacy result: {result_legacy[0]:.6f}")
    
    print("\n" + "=" * 60)
    print("Clean failure handling test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_clean_failure_handling()
