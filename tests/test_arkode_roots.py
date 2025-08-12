import numpy as np
import SundialsPy
import pytest


def test_arkode_set_root_function_then_integrate():
    # Simple scalar explicit ODE y' = -y with root at y - 0.5 = 0
    def rhs_explicit(t, y):
        return np.array([-y[0]], dtype=np.float64)

    def root_fn(t, y):
        return np.array([y[0] - 0.5], dtype=np.float64)

    y0 = np.array([1.0], dtype=np.float64)
    t0 = 0.0
    t1 = 0.2

    solver = SundialsPy.arkode.ARKodeSolver(
        system_size=1,
        explicit_fn=rhs_explicit,
        implicit_fn=None,
    )

    # Even though root finding is marked as not fully implemented, this should not crash
    solver.set_root_function(root_fn, nrtfn=1)
    solver.initialize(y0, t0, 1e-6, np.array([1e-8], dtype=np.float64))

    y1 = solver.solve_to(t1)
    expected = y0 * np.exp(-t1)
    assert np.all(np.isfinite(y1))
    assert np.allclose(y1, expected, rtol=1e-5, atol=1e-7)
