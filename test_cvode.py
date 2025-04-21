# test_cvode.py

import numpy as np
import sundials_py

# Simple linear ODE
def linear_ode(t, y):
    return np.array([-0.5 * y[0]])

# Initial condition
y0 = np.array([1.0])
t0 = 0.0
t_end = 2.0

# Create solver with ADAMS method
solver = sundials_py.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=sundials_py.cvode.IterationType.NEWTON,
    linsol_type=sundials_py.cvode.LinearSolverType.DENSE
)

# Initialize
solver.initialize(y0, t0, 1e-6, np.array([1e-8]))

# Solve single step first
t1 = 0.0001
print(f"Solving to t={t1} with CVODE method")
result = solver.solve_to(t1)
print(f"Result: {result}")


# Create solver with ADAMS method
solver = sundials_py.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=linear_ode,
    iter_type=sundials_py.cvode.IterationType.FUNCTIONAL,
)

# Initialize
solver.initialize(y0, t0, 1e-6, np.array([1e-8]))

# Solve single step first
t1 = 0.0001
print(f"Solving to t={t1} with ADAMS method")
result = solver.solve_to(t1)
print(f"Result: {result}")