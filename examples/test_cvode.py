# Fixed test_cvode.py

import numpy as np
import SundialsPy

# Simple linear ODE
def linear_ode(t, y):
    return np.array([-0.5 * y[0]])

# Initial condition
y0 = np.array([1.0])
t0 = 0.0
t_end = 2.0

# Test 1: BDF with Newton iteration (typical combination)
print("Testing BDF with Newton iteration...")
solver_bdf_newton = SundialsPy.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=SundialsPy.cvode.IterationType.NEWTON,
    linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
    use_bdf=True  # Use BDF method
)

# Initialize and solve
solver_bdf_newton.initialize(y0, t0, 1e-6, np.array([1e-8]))
t1 = 0.0001
print(f"Solving to t={t1} with BDF/Newton")
result = solver_bdf_newton.solve_to(t1)
print(f"Result: {result}")

# Test 2: Adams with Newton iteration (less common but valid)
print("\nTesting Adams with Newton iteration...")
solver_adams_newton = SundialsPy.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=SundialsPy.cvode.IterationType.NEWTON,
    linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
    use_bdf=False  # Use Adams method
)

# Initialize and solve
solver_adams_newton.initialize(y0, t0, 1e-6, np.array([1e-8]))
print(f"Solving to t={t1} with Adams/Newton")
result = solver_adams_newton.solve_to(t1)
print(f"Result: {result}")

# Test 3: Adams with Functional iteration (typical combination)
print("\nTesting Adams with Functional iteration...")
solver_adams_func = SundialsPy.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=SundialsPy.cvode.IterationType.FUNCTIONAL,
    use_bdf=False  # Use Adams method
)

# Initialize and solve
solver_adams_func.initialize(y0, t0, 1e-6, np.array([1e-8]))
print(f"Solving to t={t1} with Adams/Functional")
result = solver_adams_func.solve_to(t1)
print(f"Result: {result}")

# Test 4: BDF with Functional iteration (less common but valid)
print("\nTesting BDF with Functional iteration...")
solver_bdf_func = SundialsPy.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=SundialsPy.cvode.IterationType.FUNCTIONAL,
    use_bdf=True  # Use BDF method
)

# Initialize and solve
solver_bdf_func.initialize(y0, t0, 1e-6, np.array([1e-8]))
print(f"Solving to t={t1} with BDF/Functional")
result = solver_bdf_func.solve_to(t1)
print(f"Result: {result}")

print("\nAll tests completed successfully!")