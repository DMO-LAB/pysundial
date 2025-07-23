import numpy as np
import SundialsPy
import pytest

def rhs_explicit(t, y):
    arr = np.zeros_like(y)
    assert arr.shape == y.shape, f"Explicit RHS shape {arr.shape} does not match y shape {y.shape}"
    return arr

def rhs_implicit(t, y):
    mu = 1000.0
    arr = np.zeros_like(y)
    arr[0] = y[1]
    arr[1] = mu * (1.0 - y[0]**2) * y[1] - y[0]
    assert arr.shape == y.shape, f"Implicit RHS shape {arr.shape} does not match y shape {y.shape}"
    return arr

def jacobian(t, y):
    mu = 1000.0
    J = np.zeros((2, 2), dtype=np.float64, order='C')
    J[0, 1] = 1.0
    J[1, 0] = -2.0 * mu * y[0] * y[1] - 1.0
    J[1, 1] = mu * (1.0 - y[0]**2)
    J = np.ascontiguousarray(J)
    print(f"[DEBUG] Jacobian shape: {J.shape}, dtype: {J.dtype}, contiguous: {J.flags['C_CONTIGUOUS']}")
    assert J.shape == (2, 2), f"Jacobian shape is {J.shape}, expected (2, 2)"
    assert J.dtype == np.float64, f"Jacobian dtype is {J.dtype}, expected float64"
    return J

@pytest.mark.parametrize("butcher_table", [
    SundialsPy.arkode.ButcherTable.SDIRK_2_1_2,
    SundialsPy.arkode.ButcherTable.BILLINGTON_3_3_2,
    SundialsPy.arkode.ButcherTable.TRBDF2_3_3_2,
])
@pytest.mark.parametrize("with_jacobian", [True, False])
def test_arkode_dirk_methods(butcher_table, with_jacobian):
    y0 = np.array([2.0, 0.0])
    t0 = 0.0
    t1 = 0.01
    rtol = 1e-6
    atol = 1e-8 * np.ones_like(y0)
    solver = SundialsPy.arkode.ARKodeSolver(
        system_size=2,
        explicit_fn=rhs_explicit,
        implicit_fn=rhs_implicit,
        butcher_table=butcher_table,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE
    )
    # Keep references to callbacks to prevent GC
    solver._py_explicit = rhs_explicit
    solver._py_implicit = rhs_implicit
    if with_jacobian:
        solver._py_jacobian = jacobian
        solver.set_jacobian(jacobian)
    solver.initialize(y0, t0, rtol, atol)
    result = solver.solve_to(t1)
    assert np.all(np.isfinite(result)), f"Non-finite result for {butcher_table} with_jacobian={with_jacobian}" 