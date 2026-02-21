#!/usr/bin/env python3
"""
Test script to demonstrate the verbose functionality of CVodeSolver.
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

def test_verbose_levels():
    """Test different verbose levels"""
    
    print("=" * 60)
    print("Testing CVodeSolver verbose functionality")
    print("=" * 60)
    
    # Test parameters
    y0 = np.array([1.0])
    t0 = 0.0
    t_final = 1.0
    
    # Test verbose level -1 (no output)
    print("\n1. Testing verbose level -1 (no output):")
    print("-" * 40)
    solver1 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=simple_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=1000,
        verbose=-1
    )
    solver1.initialize(y0, t0)
    result1 = solver1.solve_to(t_final)
    print(f"Result: {result1[0]:.6f}")
    
    # Test verbose level 0 (errors only)
    print("\n2. Testing verbose level 0 (errors only):")
    print("-" * 40)
    solver2 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=simple_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=1000,
        verbose=0
    )
    solver2.initialize(y0, t0)
    result2 = solver2.solve_to(t_final)
    print(f"Result: {result2[0]:.6f}")
    
    # Test verbose level 2 (debug info)
    print("\n3. Testing verbose level 2 (debug info):")
    print("-" * 40)
    solver3 = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=simple_rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
        mxsteps=1000,
        verbose=2
    )
    solver3.initialize(y0, t0)
    result3 = solver3.solve_to(t_final)
    print(f"Result: {result3[0]:.6f}")
    
    print("\n" + "=" * 60)
    print("Verbose testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_verbose_levels()
