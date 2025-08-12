import numpy as np
import SundialsPy


def test_cvode_with_user_jacobian_linear_system():
    # Linear system y' = A y with known Jacobian A
    A = np.array([[-1.0, 0.0],
                  [ 0.0,-2.0]], dtype=np.float64)

    def rhs(t, y):
        return A @ y

    def jac(t, y):
        return A

    y0 = np.array([1.0, 1.0], dtype=np.float64)
    t0 = 0.0
    t1 = 0.1

    # Exact solution: exp(A t) y0 = [e^{-t}, e^{-2t}]
    expected = np.array([np.exp(-t1), np.exp(-2.0 * t1)], dtype=np.float64)

    solver = SundialsPy.cvode.CVodeSolver(
        system_size=2,
        rhs_fn=rhs,
        iter_type=SundialsPy.cvode.IterationType.NEWTON,
        linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
        use_bdf=True,
    )

    # Setting a user Jacobian should not break RHS or integration
    solver.set_jacobian(jac)
    solver.initialize(y0, t0, 1e-8, np.array([1e-10, 1e-10], dtype=np.float64))

    y1 = solver.solve_to(t1)
    assert np.allclose(y1, expected, rtol=1e-6, atol=1e-8)
