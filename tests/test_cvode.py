import numpy as np
import SundialsPy
import pytest

def linear_ode(t, y):
    return np.array([-0.5 * y[0]])

y0 = np.array([1.0])
t0 = 0.0
t1 = 0.1
expected = y0 * np.exp(-0.5 * t1)

@pytest.mark.parametrize("use_bdf, iter_type, linsol_type", [
    (True, SundialsPy.cvode.IterationType.NEWTON, SundialsPy.cvode.LinearSolverType.DENSE),
    (True, SundialsPy.cvode.IterationType.FUNCTIONAL, None),
    (False, SundialsPy.cvode.IterationType.NEWTON, SundialsPy.cvode.LinearSolverType.DENSE),
    (False, SundialsPy.cvode.IterationType.FUNCTIONAL, None),
])
def test_cvode_integrators(use_bdf, iter_type, linsol_type):
    kwargs = dict(
        system_size=1,
        rhs_fn=linear_ode,
        iter_type=iter_type,
        use_bdf=use_bdf
    )
    if linsol_type is not None:
        kwargs['linsol_type'] = linsol_type
    solver = SundialsPy.cvode.CVodeSolver(**kwargs)
    solver.initialize(y0, t0, 1e-6, np.array([1e-8]))
    result = solver.solve_to(t1)
    assert np.allclose(result, expected, atol=1e-6), f"Failed for use_bdf={use_bdf}, iter_type={iter_type}" 