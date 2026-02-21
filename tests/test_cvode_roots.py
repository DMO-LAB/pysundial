import numpy as np
import SundialsPy


def test_cvode_with_root_function_does_not_break_rhs():
    # Simple scalar ODE y' = -y with root at y - 0.5 = 0
    def rhs(t, y):
        return np.array([-y[0]], dtype=np.float64)

    def root_fn(t, y):
        return np.array([y[0] - 0.5], dtype=np.float64)

    y0 = np.array([1.0], dtype=np.float64)
    t0 = 0.0
    t1 = 0.2

    solver = SundialsPy.cvode.CVodeSolver(
        system_size=1,
        rhs_fn=rhs,
        iter_type=SundialsPy.cvode.IterationType.FUNCTIONAL,
        use_bdf=False,  # Adams + functional
    )

    # Setting a root function should not break RHS or integration
    solver.set_root_function(root_fn, nrtfn=1)
    solver.initialize(y0, t0, 1e-6, np.array([1e-8], dtype=np.float64))

    y1 = solver.solve_to(t1)
    expected = y0 * np.exp(-t1)
    assert np.allclose(y1, expected, rtol=1e-6, atol=1e-8)
